"""Reproducible experiments. Test scenarios never select hyperparameters."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
import stable_baselines3
import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .controllers import PurePursuit, SACController
from .env import TrackingEnv, stresses
from .evaluation import evaluate, summarize, selection_score, write_json, write_csv, manifest, plot_trace
from .routes import scenarios

ROOT = Path("artifacts")
PP_CANDIDATES = [
    {"lookahead": lookahead, "cruise_speed": speed}
    for lookahead in (0.8, 1.4) for speed in (0.7, 1.0)
]
SAC_CANDIDATES = [
    {"learning_rate": rate, "ent_coef": entropy}
    for rate in (0.0003, 0.001) for entropy in ("auto_0.1", "auto_0.01")
]
FINAL_SEEDS = (7, 17, 27)


def code_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class Progress(BaseCallback):
    def __init__(self, name, budget):
        super().__init__()
        self.name, self.budget, self.started = name, budget, time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % 5000 == 0:
            record = {"run": self.name, "timesteps": self.num_timesteps,
                      "budget": self.budget, "elapsed_seconds": time.perf_counter() - self.started}
            write_json(ROOT / "progress.json", record)
            print(json.dumps(record), flush=True)
        return True


def train_run(name, config, seed, steps):
    torch.set_num_threads(1)
    folder = ROOT / "training" / name
    folder.mkdir(parents=True, exist_ok=True)
    env = Monitor(TrackingEnv("train"), str(folder / "monitor.csv"))
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                learning_rate=config["learning_rate"], ent_coef=config["ent_coef"],
                learning_starts=1000, buffer_size=100000, batch_size=128,
                train_freq=4, gradient_steps=1,
                policy_kwargs={"net_arch": [64, 64]})
    started = time.perf_counter()
    model.learn(total_timesteps=steps, callback=Progress(name, steps))
    seconds = time.perf_counter() - started
    model_path = folder / "model.zip"
    model.save(model_path)
    validation = evaluate(SACController(model), "validation")
    result = {"run": name, "seed": seed, "config": config, "requested_steps": steps,
              "actual_steps": model.num_timesteps, "training_seconds": seconds,
              "steps_per_second": model.num_timesteps / seconds, "code_revision": code_revision(),
              "model_path": str(model_path),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "validation": summarize(validation), "selection_score": selection_score(validation),
              "validation_episodes": validation}
    write_json(folder / "run.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "validation_episodes"}), flush=True)
    env.close()
    return model, result


def baseline():
    results = []
    for index, config in enumerate(PP_CANDIDATES):
        rows = evaluate(PurePursuit(**config), "validation")
        result = {"candidate": index, "config": config, "validation": summarize(rows),
                  "selection_score": selection_score(rows), "validation_episodes": rows}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "validation_episodes"}), flush=True)
    selected = max(results, key=lambda r: r["selection_score"])
    write_json(ROOT / "baseline_selection.json", {"selected": selected, "candidates": results})
    return selected


def profile(steps):
    _, result = train_run("profile", SAC_CANDIDATES[0], 91, steps)
    write_json(ROOT / "profile.json", result)


def pipeline(pilot_steps, final_steps):
    if (ROOT / "test_results.json").exists():
        raise RuntimeError("A completed test evaluation already exists. Preserve it before a new experiment.")
    for split in ("train", "validation", "test"):
        write_json(ROOT / "manifests" / f"{split}.json", manifest(split))
    protocol = {
        "code_revision": code_revision(), "python": platform.python_version(),
        "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__, "device": "cpu", "torch_threads": 1,
        "baseline_candidate_count": len(PP_CANDIDATES),
        "sac_candidate_count": len(SAC_CANDIDATES),
        "pilot_seed": 101, "pilot_steps_per_candidate": pilot_steps,
        "final_steps_per_seed": final_steps, "final_seeds": list(FINAL_SEEDS),
        "selection_rule": "100 * success_rate + 10 * ordered_gate_completion - route_distance_rmse - 0.005 * elapsed_seconds",
        "validation_episodes_per_candidate": len(scenarios("validation")),
        "test_episodes_per_controller_per_stress": len(scenarios("test")),
        "tuning_note": "Four candidate configurations per method; equal validation scenarios. Computational costs are not equal.",
        "checkpoint_selection": "Use each run's final checkpoint. No test-based checkpoint selection.",
        "stresses": [asdict(s) for s in stresses()],
    }
    write_json(ROOT / "protocol.json", protocol)
    pp = baseline()
    pilot_results = []
    for index, config in enumerate(SAC_CANDIDATES):
        _, result = train_run(f"pilot_{index}", config, 101, pilot_steps)
        pilot_results.append(result)
    selected = max(pilot_results, key=lambda r: r["selection_score"])
    write_json(ROOT / "sac_selection.json", {"selected": selected, "candidates": pilot_results})
    final_runs = []
    for seed in FINAL_SEEDS:
        _, result = train_run(f"seed_{seed}", selected["config"], seed, final_steps)
        final_runs.append(result)
    # Every final seed is tested. One seed is chosen on validation for trace plots only.
    representative = max(final_runs, key=lambda r: r["selection_score"])["seed"]
    write_json(ROOT / "frozen_controllers.json", {
        "baseline": pp["config"], "sac_config": selected["config"],
        "runs": final_runs, "representative_seed_for_plots": representative,
        "frozen_before_test": True})
    controllers = {"Pure Pursuit": PurePursuit(**pp["config"])}
    controllers.update({f"SAC seed {r['seed']}": SACController(SAC.load(r["model_path"], device="cpu"))
                        for r in final_runs})
    all_rows, summaries = [], []
    for stress in stresses():
        for name, controller in controllers.items():
            trace_dir = None
            if stress.name == "nominal" and name in ("Pure Pursuit", f"SAC seed {representative}"):
                slug = "pure_pursuit" if name == "Pure Pursuit" else f"sac_seed_{representative}"
                trace_dir = ROOT / "traces" / slug
            rows = evaluate(controller, "test", stress, trace_dir=trace_dir)
            all_rows.extend([{**row, "controller": name} for row in rows])
            summary = {"controller": name, "stress": stress.name, **summarize(rows)}
            summaries.append(summary)
            print("TEST " + json.dumps(summary), flush=True)
    write_csv(ROOT / "test_episodes.csv", all_rows)
    write_json(ROOT / "test_results.json", {"summaries": summaries, "episodes": all_rows})
    # Fixed examples chosen by family/index, not by best test performance.
    for case in scenarios("test")[:4]:
        plot_trace({
            "Pure Pursuit": ROOT / "traces/pure_pursuit" / f"{case.name}.json",
            f"SAC seed {representative}": ROOT / f"traces/sac_seed_{representative}" / f"{case.name}.json",
        }, ROOT / "figures" / f"{case.name}.png", f"Unseen test route: {case.family}")
    learning_curve()
    write_json(ROOT / "progress.json", {"phase": "complete", "test_episodes": len(all_rows)})
    print("EXPERIMENT COMPLETE", flush=True)


def learning_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for seed in FINAL_SEEDS:
        path = ROOT / "training" / f"seed_{seed}" / "monitor.csv"
        data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1)
        data = np.atleast_1d(data)
        rewards = data["r"]
        window = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.cumsum(data["l"])[window - 1:], smoothed, label=f"Seed {seed}")
    ax.set(xlabel="Environment steps", ylabel="Episode return (20-episode moving mean)",
           title="SAC training returns: training routes only")
    ax.legend()
    ax.grid(alpha=0.2)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/training_returns.png", dpi=170)
    plt.close(fig)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "profile", "pipeline"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--pilot-steps", type=int, default=12000)
    parser.add_argument("--final-steps", type=int, default=80000)
    args = parser.parse_args()
    ROOT = args.output_dir
    if args.mode == "baseline":
        baseline()
    elif args.mode == "profile":
        profile(args.steps)
    else:
        pipeline(args.pilot_steps, args.final_steps)


if __name__ == "__main__":
    main()
