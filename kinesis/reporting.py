"""Generate measured results from saved experiments, without retuning."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from .evaluation import write_json

ROOT = Path("artifacts")


def load(name):
    return json.loads((ROOT / name).read_text())


def mean_sd(values, digits=3, scale=1):
    a = np.asarray(values, float) * scale
    return f"{a.mean():.{digits}f} +/- {a.std(ddof=1):.{digits}f}"


def main():
    data, protocol, frozen = load("test_results.json"), load("protocol.json"), load("frozen_controllers.json")
    rows, summaries = data["episodes"], data["summaries"]
    nominal = [r for r in summaries if r["stress"] == "nominal"]
    pp = next(r for r in nominal if r["controller"] == "Pure Pursuit")
    sac = [r for r in nominal if r["controller"].startswith("SAC")]
    lines = [
        "# Kinesis experimental results", "",
        "Ankit Prajapati | Simulated mobile-robot path tracking | 28 October 2026", "",
        "## Scope", "",
        "A bounded planar simulator compares Pure Pursuit with Stable-Baselines3 SAC. "
        "This is a simulation study with no physical robot or embedded deployment.", "",
        "## Protocol", "",
        f"- Training source checkpoint: {protocol['code_revision']}.",
        "- Python 3.12.11; CPU PyTorch 2.14.1; Stable-Baselines3 2.9.2; Gymnasium 1.4.0.",
        "- Cloud hardware: 2 vCPUs, approximately 8 GiB RAM; AMD EPYC 7763 host CPU.",
        "- 24 training, 12 validation and 24 test scenarios; route hashes check exact split overlap.",
        "- Training/validation use straight, sine and arc routes. Double-sine geometry is held out.",
        f"- Four candidates per method; SAC pilots use {protocol['pilot_steps_per_candidate']:,} steps each, seed 101.",
        f"- Three final SAC models use {protocol['final_steps_per_seed']:,} steps each, independent seeds 7, 17 and 27.",
        "- Candidate counts and validation scenarios match; computational tuning costs do not.",
        f"- Selected Pure Pursuit: {frozen['baseline']}.",
        f"- Selected SAC: {frozen['sac_config']}.",
        "- All controllers share observations, action limits and route-specific time allowances.",
        "- Ordered 0.5 m gates with 0.35 m tolerance prevent skipped routes from counting as success.",
        "", "## Nominal held-out results", "",
        "All models use the same 24 test scenarios. SAC values are mean +/- sample "
        "standard deviation across three training seeds, not confidence intervals.", "",
        "| Metric | Pure Pursuit | SAC across three seeds |", "| --- | ---: | ---: |",
    ]
    for label, key, scale, digits in [
        ("Success (%)", "success_rate", 100, 1),
        ("Ordered gate completion (%)", "completion", 100, 1),
        ("Tracking RMSE (m)", "rmse_m", 1, 3),
        ("Maximum error, episode mean (m)", "max_error_m", 1, 3),
        ("Elapsed time, all attempts (s)", "elapsed_s", 1, 2),
        ("Linear command slew RMS (m/s^2)", "v_command_slew_rms_mps2", 1, 2),
        ("Angular command slew RMS (rad/s^2)", "w_command_slew_rms_radps2", 1, 2),
        ("Mean episode median inference (ms)", "latency_median_ms", 1, 3),
        ("Mean episode p95 inference (ms)", "latency_p95_ms", 1, 3),
    ]:
        lines.append(f"| {label} | {pp[key] * scale:.{digits}f} | {mean_sd([r[key] for r in sac], digits, scale)} |")
    lines += ["", "Errors and elapsed times include early failures. Low values on "
              "short failed attempts do not demonstrate better control.", "",
              "| SAC model | Successes / 24 | RMSE (m) | Paired successes | PP time (s) | SAC time (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    paired_records = []
    for summary in sac:
        name = summary["controller"]
        sr = {r["scenario"]: r for r in rows if r["stress"] == "nominal" and r["controller"] == name}
        pr = {r["scenario"]: r for r in rows if r["stress"] == "nominal" and r["controller"] == "Pure Pursuit"}
        pairs = [(pr[k], v) for k, v in sr.items() if v["success"] and pr[k]["success"]]
        pt = float(np.mean([a["elapsed_s"] for a, _ in pairs])) if pairs else None
        st = float(np.mean([b["elapsed_s"] for _, b in pairs])) if pairs else None
        paired_records.append({"controller": name, "paired_successes": len(pairs), "pp_elapsed_s": pt, "sac_elapsed_s": st})
        pt_text, st_text = (f"{pt:.2f}", f"{st:.2f}") if pairs else ("n/a", "n/a")
        lines.append(f"| {name} | {sum(r['success'] for r in sr.values())} | {summary['rmse_m']:.3f} | {len(pairs)} | {pt_text} | {st_text} |")
    lines += ["", "## Stress tests", "",
              "Nominally trained models face one disturbance at a time. Each condition "
              "uses the same routes and initial states for every controller.", "",
              "| Condition | PP success (%) | SAC success (%) | PP RMSE (m) | SAC RMSE (m) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for stress in protocol["stresses"]:
        name = stress["name"]
        base = next(r for r in summaries if r["controller"] == "Pure Pursuit" and r["stress"] == name)
        learned = [r for r in summaries if r["controller"].startswith("SAC") and r["stress"] == name]
        lines.append(f"| {name} | {100 * base['success_rate']:.1f} | "
                     f"{mean_sd([r['success_rate'] for r in learned], 1, 100)} | {base['rmse_m']:.3f} | "
                     f"{mean_sd([r['rmse_m'] for r in learned])} |")
    lines += ["", "Noise: 0.03 m Gaussian position noise per coordinate and 3 degree heading noise. "
              "Delay: two 0.1 s control steps with observed command queue. Slow response: both "
              "time constants multiplied by 2.5. Noise does not simulate full localization.", "",
              "## Held-out route family", "",
              "| Nominal route family | PP success (%) | SAC success (%) | PP RMSE (m) | SAC RMSE (m) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    family_records = []
    for family in ("straight", "sine", "arc", "double_sine"):
        base = [r for r in rows if r["controller"] == "Pure Pursuit" and r["stress"] == "nominal" and r["family"] == family]
        groups = [[r for r in rows if r["controller"] == f"SAC seed {seed}" and r["stress"] == "nominal" and r["family"] == family] for seed in (7, 17, 27)]
        bs, be = float(np.mean([r["success"] for r in base])), float(np.mean([r["rmse_m"] for r in base]))
        ss = [float(np.mean([r["success"] for r in g])) for g in groups]
        se = [float(np.mean([r["rmse_m"] for r in g])) for g in groups]
        lines.append(f"| {family} | {100 * bs:.1f} | {mean_sd(ss, 1, 100)} | {be:.3f} | {mean_sd(se)} |")
        family_records.append({"family": family, "pp_success": bs, "sac_success": ss, "pp_rmse_m": be, "sac_rmse_m": se})
    lines += ["", "Each family has six test scenarios, a small sample.", "", "## Failure records", "",
              "| Controller | Condition | Failure reasons |", "| --- | --- | --- |"]
    for summary in summaries:
        counts = Counter(r["reason"] for r in rows if r["controller"] == summary["controller"] and r["stress"] == summary["stress"] and not r["success"])
        text = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"
        lines.append(f"| {summary['controller']} | {summary['stress']} | {text} |")
    minutes = sum(r["training_seconds"] for r in frozen["runs"]) / 60
    lines += ["", "## Resources and interpretation", "",
              f"Final training used {minutes:.2f} minutes summed over three learning calls. "
              "This excludes pilots, evaluation, setup and documentation. No GPU was used.", "",
              "- Three seeds give limited evidence about variability. No significance or universal superiority claim is made.",
              "- Reference previews and localization are simplified; the scenario distribution is finite.",
              "- Stresses are tested separately. Combined disturbances were not evaluated.",
              "- Episode latency percentiles are averaged, not pooled over all calls.",
              "- Latency is CPU wall time around prediction, not an embedded deadline or worst-case guarantee.",
              "- Wheel dynamics, slip, obstacles, collisions, physical sensing and safety validation are outside scope.",
              "- Selection was frozen before testing; test outcomes do not feed tuning.", "",
              "## Evidence", "",
              "- artifacts/protocol.json and manifests/: budgets, versions, source checkpoint, cases and route hashes.",
              "- baseline_selection.json and sac_selection.json: all validation candidates.",
              "- training/seed_*/run.json and monitor.csv: each final training record.",
              "- test_episodes.csv and test_results.json: all test attempts, including failures.",
              "- figures/: fixed examples and training-return plots.",
              "- docs/learning-report.md: concepts, code walkthrough and interview revision.", "",
              "SAC: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html",
              "Pure Pursuit: https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm"]
    Path("docs/results.md").write_text("\n".join(lines) + "\n")
    write_json(ROOT / "report_summary.json", {"nominal_baseline": pp, "nominal_sac": sac,
               "paired_success_times": paired_records, "families": family_records,
               "final_training_minutes": minutes, "representative_seed": frozen["representative_seed_for_plots"]})
    print("Wrote docs/results.md and artifacts/report_summary.json")


if __name__ == "__main__":
    main()
