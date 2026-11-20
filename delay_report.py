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
        success[variant] = (wins, total)

labels = ["Pure Pursuit", "Original SAC", "A: random\ndelay", "B: random delay,\nsmoother"]
counts = [(24, 24), (65, 72), success.get("A_random_delay", (0, 1)), success.get("B_random_delay_smoother", (0, 1))]
values = [100 * wins / total for wins, total in counts]
plt.figure(figsize=(7, 3.8))
plt.bar(labels, values, color=["0.75", "0.55", "tab:blue", "tab:blue"])
for i, (wins, total) in enumerate(counts):
    plt.text(i, values[i] + 1, f"{wins}/{total}", ha="center")
plt.ylim(0, 110)
plt.ylabel("Success at 200 ms delay (%)")
plt.title("Training with random delay")
plt.tight_layout()
plt.savefig("artifacts/delay_study/delay_success.png", dpi=150)
print("saved artifacts/delay_study/delay_success.png")
