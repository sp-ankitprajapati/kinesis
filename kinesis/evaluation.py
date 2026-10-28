"""Paired, deterministic scenario evaluation and standalone plots."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from .env import TrackingEnv, Stress
from .routes import scenarios


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(split):
    from dataclasses import asdict
    return [{**asdict(case), "name": case.name, "route_sha256": case.route().fingerprint()}
            for case in scenarios(split)]


def rollout(controller, case, stress=None, *, save_trace=None):
    env = TrackingEnv(case.split, stress)
    # Same observation-noise draw sequence for every controller on this scenario.
    obs, _ = env.reset(seed=45000 + case.index, options={"scenario": case})
    latencies = []
    while True:
        start = time.perf_counter_ns()
        action = controller.predict(obs)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    history = env.history
    errors = np.array([x["path_distance"] for x in history])
    commands = np.array([[x["v_command"], x["w_command"]] for x in history])
    slew = np.diff(np.vstack([[0, 0], commands]), axis=0) / env.motion.dt
    result = {
        "scenario": case.name, "family": case.family,
        "stress": (stress or Stress()).name, "success": info["success"], "reason": info["reason"],
        "completion": info["completion"], "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "max_error_m": float(errors.max()), "elapsed_s": info["elapsed_seconds"],
        "v_command_slew_rms_mps2": float(np.sqrt(np.mean(slew[:, 0]**2))),
        "w_command_slew_rms_radps2": float(np.sqrt(np.mean(slew[:, 1]**2))),
        "latency_median_ms": float(np.median(latencies)), "latency_p95_ms": float(np.percentile(latencies, 95)),
        "steps": len(history),
    }
    if save_trace:
        write_json(save_trace, {"metrics": result, "route": env.route.points.tolist(), "history": history})
    return result


def evaluate(controller, split, stress=None, *, trace_dir=None):
    return [rollout(controller, case, stress,
                    save_trace=Path(trace_dir) / f"{case.name}.json" if trace_dir else None)
            for case in scenarios(split)]


def summarize(rows):
    keys = ("completion", "rmse_m", "max_error_m", "elapsed_s",
            "v_command_slew_rms_mps2", "w_command_slew_rms_radps2",
            "latency_median_ms", "latency_p95_ms")
    return {"episodes": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            **{key: float(np.mean([r[key] for r in rows])) for key in keys}}


def selection_score(rows):
    # Completion dominates; tracking error breaks ties. Fixed before testing.
    summary = summarize(rows)
    return 100 * summary["success_rate"] + 10 * summary["completion"] - summary["rmse_m"] - 0.005 * summary["elapsed_s"]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(trace_paths, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for label, path in trace_paths.items():
        trace = json.loads(Path(path).read_text())
        route = np.asarray(trace["route"])
        history = trace["history"]
        if not axes[0].lines:
            axes[0].plot(route[:, 0], route[:, 1], "k--", label="Reference", linewidth=1.5)
        axes[0].plot([x["x"] for x in history], [x["y"] for x in history], label=label)
        axes[1].plot([x["elapsed_seconds"] for x in history],
                     [x["path_distance"] for x in history], label=label)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    axes[1].set(xlabel="Time (s)", ylabel="Distance to local route (m)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
