import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
with open(HERE / "cache" / "kshot_sweep_results.json") as f:
    r = json.load(f)

cfgs = r["configs"]
ks = [cfgs[k]["k"] for k in cfgs]
accs = [cfgs[k]["acc"] * 100 for k in cfgs]
f1s = [cfgs[k]["f1"] for k in cfgs]

fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax1.plot(ks, accs, "o-", color="#1f77b4", label="Accuracy (%)", linewidth=2, markersize=8)
ax1.axhline(25, ls=":", color="gray", alpha=0.5, label="Random 25%")
ax1.set_xlabel("k (number of Korean exemplars)")
ax1.set_ylabel("Accuracy (%)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_xticks(ks)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(ks, f1s, "s--", color="#d62728", label="Macro-F1", linewidth=2, markersize=7)
ax2.set_ylabel("Macro-F1", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

for k, a in zip(ks, accs):
    ax1.text(k, a + 1, f"{a:.1f}", ha="center", fontsize=9, color="#1f77b4")

plt.title("k-shot scaling on Korean FER AU (Qwen2.5-7B 4-bit)\nInverted-U: few > many")
fig.tight_layout()
fig.savefig(HERE / "cache" / "kshot_curve.png", dpi=140)
print(f"saved {HERE}/cache/kshot_curve.png")
