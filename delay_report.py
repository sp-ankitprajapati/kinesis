"""Print the follow-up results next to the original study and save one chart."""
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("artifacts/delay_study/summary.json"))
groups = defaultdict(list)
for r in rows:
    groups[(r["variant"], r["stress"])].append(r)

print("Original SAC (RESULTS.md): nominal 72/72, RMSE 0.113 m, slew 5.91; delay_200ms 65/72, RMSE 0.175 m")
success = {}
for (variant, stress), rs in sorted(groups.items()):
    wins = sum(round(r["success_rate"] * r["episodes"]) for r in rs)
    total = sum(r["episodes"] for r in rs)
    rmse = sum(r["rmse_m"] for r in rs) / len(rs)
    slew = sum(r["w_command_slew_rms_radps2"] for r in rs) / len(rs)
    print(f"{variant:26s} {stress:12s} success {wins}/{total}  RMSE {rmse:.3f} m  steering slew {slew:.2f} rad/s^2")
    if stress == "delay_200ms":
        success[variant] = 100 * wins / total

labels = ["Pure Pursuit", "Original SAC", "A: random delay", "B: random delay, smoother"]
values = [100.0, 100 * 65 / 72, success.get("A_random_delay", 0.0), success.get("B_random_delay_smoother", 0.0)]
plt.figure(figsize=(7, 3.5))
plt.bar(labels, values, color=["0.75", "0.55", "tab:blue", "tab:blue"])
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.0f}%", ha="center")
plt.ylim(0, 110)
plt.ylabel("Success at 200 ms delay (%)")
plt.title("Training with random delay")
plt.tight_layout()
plt.savefig("artifacts/delay_study/delay_success.png", dpi=150)
print("saved artifacts/delay_study/delay_success.png")
