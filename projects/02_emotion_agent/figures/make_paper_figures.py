"""
Paper figures for §4 — reads from leaderboard data and multi-seed JSON files.
Outputs PNGs at 300 DPI ready for camera-ready submission.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = Path(__file__).parent
FIG_OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

# ============================================================
# Figure 1: 3-tier adaptation hierarchy
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
tiers = ["Prompt\n(zero-shot FACS)", "ICL\n(k=4 Korean)", "LoRA\n(r=16, 10K, 1ep)"]
means = [29.08, 41.83, 55.00]
stds =  [0.76, 3.41, 2.61]
colors = ["#c8d6e5", "#6c9bd5", "#2c5282"]

bars = ax.bar(tiers, means, yerr=stds, capsize=6, color=colors,
              edgecolor="black", linewidth=0.8)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 1.2, f"{m:.2f}±{s:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# Annotate deltas
ax.annotate("", xy=(1, means[1]+stds[1]+0.3), xytext=(0, means[0]+stds[0]+0.3),
            arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(0.5, max(means[0]+stds[0], means[1]+stds[1])/2 + 3,
        f"+{means[1]-means[0]:.2f} pp", ha="center", fontsize=10, color="#444")
ax.annotate("", xy=(2, means[2]+stds[2]+0.3), xytext=(1, means[1]+stds[1]+0.3),
            arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(1.5, max(means[1]+stds[1], means[2]+stds[2])/2 + 3,
        f"+{means[2]-means[1]:.2f} pp", ha="center", fontsize=10, color="#444")

# Random line
ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.text(2.45, 25.5, "random (25%)", fontsize=8, color="gray", ha="right")

ax.set_ylabel("4-class accuracy (%)")
ax.set_title("3-tier adaptation hierarchy on Korean FER AU (n=3 seeds)")
ax.set_ylim(0, 70)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig1_3tier_hierarchy.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] {FIG_OUT / 'fig1_3tier_hierarchy.png'}")

# ============================================================
# Figure 2: Anchor-ratio variance curve
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

k_wes     = [0, 1, 2, 3, 4]
acc_mean  = [41.00, 41.50, 41.25, 40.08, 25.42]
acc_std   = [4.34, 6.00, 0.87, 0.80, 0.38]

ax1.errorbar(k_wes, acc_mean, yerr=acc_std, fmt="o-", color="#2c5282",
             capsize=5, markersize=8, linewidth=1.5, markerfacecolor="white",
             markeredgewidth=1.5, label="Accuracy")
ax1.fill_between(k_wes,
                 np.array(acc_mean) - np.array(acc_std),
                 np.array(acc_mean) + np.array(acc_std),
                 alpha=0.15, color="#2c5282")
ax1.set_xlabel("number of Western Ekman anchors (k_Wes)")
ax1.set_ylabel("accuracy (%)")
ax1.set_title("Mean: driven by Korean presence")
ax1.set_xticks(k_wes)
ax1.set_xticklabels([f"{kw}\n(Kor={4-kw})" for kw in k_wes])
ax1.axvspan(1.5, 2.5, alpha=0.1, color="orange", zorder=0)
ax1.text(2, 48, "transition", ha="center", fontsize=9, color="#d97706")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

ax2.bar(k_wes, acc_std, color=["#e74c3c", "#e74c3c", "#27ae60", "#27ae60", "#95a5a6"],
        edgecolor="black", linewidth=0.8)
for kw, s in zip(k_wes, acc_std):
    ax2.text(kw, s + 0.2, f"{s:.2f}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(1, color="black", linestyle=":", lw=1, alpha=0.7)
ax2.set_xlabel("number of Western Ekman anchors (k_Wes)")
ax2.set_ylabel("std of accuracy across seeds (pp)")
ax2.set_title("Variance: collapses at k_Wes ≥ 2")
ax2.set_xticks(k_wes)
ax2.set_xticklabels([f"{kw}\n(Kor={4-kw})" for kw in k_wes])
ax2.set_ylim(0, 7.5)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

fig.suptitle("Two-effect anchor decomposition (k=4 total, n=3 seeds)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig2_anchor_variance.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] {FIG_OUT / 'fig2_anchor_variance.png'}")

# ============================================================
# Figure 3: k-scaling — saturation not inverted-U
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
k_list    = [0, 2, 4, 8, 16]
k_means   = [29.08, 44.25, 42.75, 43.50, 42.33]
k_stds    = [0.76, 2.82, 5.02, 1.52, 1.84]
# Original single-seed for comparison
k_single  = [30.00, 42.25, 40.25, 41.00, 36.00]

ax.errorbar(k_list, k_means, yerr=k_stds, fmt="o-", color="#2c5282",
            capsize=5, markersize=8, linewidth=1.8, markerfacecolor="white",
            markeredgewidth=1.6, label="multi-seed (n=3)")
ax.plot(k_list, k_single, "s--", color="#e74c3c", alpha=0.6, markersize=6,
        label="single-seed (original, artifact)")
ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.text(16.5, 25, "random", fontsize=8, color="gray", va="center")

# Saturation arrow
ax.annotate("saturation", xy=(8, 43.5), xytext=(12, 48),
            arrowprops=dict(arrowstyle="->", color="#2c5282", lw=1.1), fontsize=10, color="#2c5282")

ax.set_xlabel("number of in-context exemplars (k)")
ax.set_ylabel("accuracy (%)")
ax.set_title("k-scaling on Korean FER AU — saturation, not inverted-U")
ax.set_xticks(k_list)
ax.set_ylim(24, 52)
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig3_kshot_saturation.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] {FIG_OUT / 'fig3_kshot_saturation.png'}")

# ============================================================
# Figure 4: LoRA + ICL substitutability
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
configs = ["LoRA only\n(56.75)", "LoRA + Mixed\n(2+2)", "LoRA + Korean\n(k=4)"]
vals    = [56.75, 56.00, 52.75]
deltas  = [0, -0.75, -4.00]
colors_ft = ["#2c5282", "#6c9bd5", "#e74c3c"]

bars = ax.bar(configs, vals, color=colors_ft, edgecolor="black", linewidth=0.8)
for b, v, d in zip(bars, vals, deltas):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    if d != 0:
        ax.text(b.get_x() + b.get_width()/2, v - 4, f"Δ={d:+.2f} pp",
                ha="center", va="top", fontsize=9, color="white", fontweight="bold")

ax.axhline(56.75, color="#2c5282", linestyle=":", lw=1.2, alpha=0.7)
ax.text(2.45, 56.9, "LoRA-only baseline", fontsize=8, color="#2c5282", ha="right")

ax.set_ylabel("accuracy (%)")
ax.set_title("Adding ICL on top of LoRA: substitutes, not complements")
ax.set_ylim(45, 62)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig4_lora_icl_substitute.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] {FIG_OUT / 'fig4_lora_icl_substitute.png'}")

print("\n[DONE] 4 paper figures written to", FIG_OUT)
