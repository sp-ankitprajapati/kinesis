"""Run a saved controller and plot an inspected trajectory."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import SAC
from .controllers import PurePursuit, SACController
from .routes import scenarios
from .env import stresses
from .evaluation import rollout, plot_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["pure_pursuit", "sac"], default="pure_pursuit")
    parser.add_argument("--model", type=Path, default=Path("models/sac_selected.zip"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--stress", choices=[s.name for s in stresses()], default="nominal")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = scenarios(args.split)
    if not 0 <= args.scenario < len(cases):
        parser.error("Scenario index outside the selected manifest.")
    if args.controller == "sac":
        if not args.model.exists():
            parser.error("Model not found. Use a supplied checkpoint or train a reproduction.")
        controller = SACController(SAC.load(args.model, device="cpu"))
    else:
        frozen = Path("artifacts/frozen_controllers.json")
        config = json.loads(frozen.read_text())["baseline"] if frozen.exists() else {"lookahead": 0.8, "cruise_speed": 1.0}
        controller = PurePursuit(**config)
    stress = next(s for s in stresses() if s.name == args.stress)
    trace = args.output_dir / f"{args.controller}.json"
    metrics = rollout(controller, cases[args.scenario], stress, save_trace=trace)
    plot_trace({args.controller: trace}, args.output_dir / f"{args.controller}.png",
               f"{cases[args.scenario].name} | {stress.name}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
