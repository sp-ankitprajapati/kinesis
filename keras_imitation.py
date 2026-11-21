"""Imitation learning with TensorFlow/Keras: train a small network to copy Pure Pursuit.

Two variants, decided before testing:
  nominal_data       demonstrations collected with no command delay
  random_delay_data  demonstrations collected with a random 0, 100 or 200 ms delay per episode

Both networks have the SAC actor's hidden size (64 x 64). They are tested on the 24 test
routes, nominal and with a 200 ms delay, next to Pure Pursuit itself.

Run from the repo root in its own environment (see requirements-tf.txt):
    python keras_imitation.py
Writes artifacts/keras_imitation/ and leaves the original study's artifacts untouched.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

from kinesis.controllers import PurePursuit
from kinesis.env import TrackingEnv, stresses
from kinesis.evaluation import evaluate, summarize
from kinesis.routes import scenarios

OUT = Path("artifacts/keras_imitation")
EXPERT = PurePursuit(lookahead=0.8, cruise_speed=1.0)   # the frozen Pure Pursuit from the main study
NOISY_ROLLOUTS = 3      # per training route, on top of one clean rollout
ACTION_NOISE = 0.2      # noise on the executed action; the label is always the clean expert action
VARIANTS = {"nominal_data": False, "random_delay_data": True}
CONDITIONS = [s for s in stresses() if s.name in ("nominal", "delay_200ms")]


def collect(split, random_delay, noisy_rollouts, seed):
    """Drive with Pure Pursuit and record (observation, expert action) pairs.

    Noisy rollouts execute a perturbed action, so the data also shows how the expert
    recovers from small mistakes, but every label is the expert's clean action."""
    rng = np.random.default_rng(seed)
    observations, actions = [], []
    for case in scenarios(split):
        for rollout in range(1 + noisy_rollouts):
            env = TrackingEnv(split, random_delay=random_delay)
            obs, _ = env.reset(seed=seed + 100 * case.index + rollout, options={"scenario": case})
            while True:
                label = np.asarray(EXPERT.predict(obs), dtype=np.float32)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(label)
                executed = label if rollout == 0 else np.clip(label + rng.normal(0, ACTION_NOISE, 2), -1, 1)
                obs, _, terminated, truncated, _ = env.step(executed.astype(np.float32))
                if terminated or truncated:
                    break
    return np.stack(observations), np.stack(actions)


def build_model():
    """25 observation values -> 64 -> 64 -> 2 actions, the SAC actor's hidden size."""
    inputs = keras.Input(shape=(25,), name="observation")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="tanh", name="action")(x)   # actions live in [-1, 1]
    return keras.Model(inputs, outputs, name="imitation_policy")


class KerasPolicy:
    """Lets a Keras model drive the same evaluation code as the other controllers."""

    def __init__(self, model):
        self.forward = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    def predict(self, observation):
        x = tf.convert_to_tensor(np.asarray(observation, dtype=np.float32)[None])
        return self.forward(x).numpy()[0]


def export_tflite(model, path):
    """Convert to TensorFlow Lite (now LiteRT) for small devices; returns the size in KB or None."""
    try:
        with tempfile.TemporaryDirectory() as saved_model_dir:
            model.export(saved_model_dir)                     # a TensorFlow SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            path.write_bytes(converter.convert())
        return round(path.stat().st_size / 1024, 1)
    except Exception as exc:                                  # the main results don't depend on this
        print(f"TensorFlow Lite export skipped ({type(exc).__name__}: {exc}); the main results are unaffected.")
        return None


def plot_losses(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(histories), figsize=(9, 3.5), sharey=True)
    for ax, (variant, history) in zip(axes, histories.items()):
        ax.plot(history["loss"], label="training")
        ax.plot(history["val_loss"], label="validation")
        ax.set_yscale("log")
        ax.set_title(variant.replace("_", " "))
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("MSE against Pure Pursuit")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    keras.utils.set_random_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"tensorflow": tf.__version__, "keras": keras.__version__,
               "noisy_rollouts_per_route": NOISY_ROLLOUTS, "action_noise": ACTION_NOISE,
               "variants": {}, "test": []}
    histories, models = {}, {}

    for variant, random_delay in VARIANTS.items():
        x_train, y_train = collect("train", random_delay, NOISY_ROLLOUTS, seed=10_000)
        x_val, y_val = collect("validation", random_delay, 0, seed=20_000)
        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                    .shuffle(len(x_train), seed=0).batch(256).prefetch(tf.data.AUTOTUNE))
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1024)

        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        start = time.perf_counter()
        history = model.fit(train_ds, validation_data=val_ds, epochs=80, callbacks=[stop], verbose=2)
        seconds = time.perf_counter() - start
        model.save(str(OUT / f"{variant}.keras"))

        histories[variant], models[variant] = history.history, model
        summary["variants"][variant] = {
            "random_delay_in_data": random_delay,
            "train_samples": int(len(x_train)), "validation_samples": int(len(x_val)),
            "epochs_run": len(history.history["loss"]),
            "best_validation_mse": float(min(history.history["val_loss"])),
            "parameters": int(model.count_params()), "training_seconds": round(seconds, 1),
        }
        policy = KerasPolicy(model)
        for stress in CONDITIONS:
            summary["test"].append({"controller": variant, "stress": stress.name,
                                    **summarize(evaluate(policy, "test", stress))})

    for stress in CONDITIONS:
        summary["test"].append({"controller": "pure_pursuit", "stress": stress.name,
                                **summarize(evaluate(EXPERT, "test", stress))})

    summary["tflite_kb"] = export_tflite(models["random_delay_data"], OUT / "random_delay_data.tflite")
    plot_losses(histories, OUT / "loss_curves.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nTensorFlow {tf.__version__}, Keras {keras.__version__}")
    print(f"{'controller':18s} {'condition':12s} {'success':>8s} {'RMSE (m)':>9s} {'slew (rad/s^2)':>15s} {'latency (ms)':>13s}")
    for row in summary["test"]:
        wins = round(row["success_rate"] * row["episodes"])
        print(f"{row['controller']:18s} {row['stress']:12s} {wins:>5d}/{row['episodes']:<2d} {row['rmse_m']:9.3f} "
              f"{row['w_command_slew_rms_radps2']:15.2f} {row['latency_median_ms']:13.3f}")
    print(f"saved {OUT}/summary.json and loss_curves.png")


if __name__ == "__main__":
    main()
