"""Follow-up study: does training with random command delay make SAC robust to it?

Run from the repo root on a separate branch:  python delay_study.py
Writes artifacts/delay_study/ and leaves the original study's artifacts untouched.
"""
import json
from pathlib import Path
import torch
from stable_baselines3 import SAC
from kinesis.env import TrackingEnv, stresses
from kinesis.controllers import SACController
from kinesis.evaluation import evaluate, summarize

OUT = Path("artifacts/delay_study")
VARIANTS = {"A_random_delay": 0.02, "B_random_delay_smoother": 0.2}   # action-change penalty weight
SEEDS = (7, 17, 27)
CONDITIONS = [s for s in stresses() if s.name in ("nominal", "delay_200ms")]


def main():
    torch.set_num_threads(1)
    results = []
    for variant, smooth in VARIANTS.items():
        for seed in SEEDS:
            env = TrackingEnv("train", random_delay=True, smooth_weight=smooth)
            model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                        learning_rate=0.0003, ent_coef="auto_0.1",
                        learning_starts=1000, buffer_size=100000, batch_size=128,
                        train_freq=4, gradient_steps=1, policy_kwargs={"net_arch": [64, 64]})
            model.learn(total_timesteps=80000)
            folder = OUT / variant / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            model.save(folder / "model.zip")
            for stress in CONDITIONS:
                rows = evaluate(SACController(model), "test", stress)
                summary = {"variant": variant, "seed": seed, "stress": stress.name, **summarize(rows)}
                results.append(summary)
                print(json.dumps(summary), flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
