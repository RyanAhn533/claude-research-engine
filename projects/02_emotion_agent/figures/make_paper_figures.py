"""
Paper figures for §4 — updated with cross-domain 3-tier + revised anchor decomp.
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
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 120,
})

# ============================================================
# Figure 1: 3-tier hierarchy (Korean FER — kept from original)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
tiers = ["Prompt\n(zero-shot)", "ICL\n(k=4)", "LoRA\n(r=16)"]
means = [29.08, 41.83, 55.00]
stds = [0.76, 3.41, 2.61]
colors = ["#c8d6e5", "#6c9bd5", "#2c5282"]
bars = ax.bar(tiers, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.8)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 1.2, f"{m:.2f}±{s:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.text(2.45, 25.5, "random (25%)", fontsize=8, color="gray", ha="right")
ax.set_ylabel("4-class accuracy (%)")
ax.set_title("3-tier adaptation hierarchy — Korean FER AU (n=3)")
ax.set_ylim(0, 70)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig1_3tier_korean_fer.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] fig1_3tier_korean_fer.png")

# ============================================================
# Figure 5 (NEW): Cross-domain 3-tier — main revised result
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

datasets = ["Korean FER AU\n(novel modality)", "IEMOCAP\n(English dialog)", "MELD\n(English TV)"]
t1 = [29.08, 47.83, 55.50]; t1_std = [0.76, 2.50, 3.27]
t2 = [41.83, 48.92, 55.17]; t2_std = [3.41, 4.25, 5.65]
t3 = [55.00, 70.00, 61.75]; t3_std = [2.61, 3.70, 1.15]

x = np.arange(len(datasets))
w = 0.25
ax.bar(x - w, t1, w, yerr=t1_std, capsize=4, label="T1 zero-shot", color="#c8d6e5", edgecolor="black", linewidth=0.7)
ax.bar(x, t2, w, yerr=t2_std, capsize=4, label="T2 ICL k=4", color="#6c9bd5", edgecolor="black", linewidth=0.7)
ax.bar(x + w, t3, w, yerr=t3_std, capsize=4, label="T3 LoRA", color="#2c5282", edgecolor="black", linewidth=0.7)

for i, (a1, a2, a3, s1, s2, s3) in enumerate(zip(t1, t2, t3, t1_std, t2_std, t3_std)):
    ax.text(i - w, a1 + s1 + 1.5, f"{a1:.1f}", ha="center", fontsize=8)
    ax.text(i,     a2 + s2 + 1.5, f"{a2:.1f}", ha="center", fontsize=8)
    ax.text(i + w, a3 + s3 + 1.5, f"{a3:.1f}", ha="center", fontsize=8, fontweight="bold")

ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.5)
ax.text(2.5, 25.8, "random", fontsize=7, color="gray", ha="right")
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("4-class accuracy (%)")
ax.set_title("Cross-domain 3-tier hierarchy — LoRA robust, ICL domain-specific")
ax.legend(loc="upper left")
ax.set_ylim(0, 85)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Annotate ICL gain per dataset
for i, (a1, a2) in enumerate(zip(t1, t2)):
    delta = a2 - a1
    col = "#27ae60" if delta > 2 else "#e74c3c" if delta < 0 else "#888"
    ax.text(i - w/2, max(a1, a2) + max(t1_std[i], t2_std[i]) + 5.5,
            f"ICL Δ={delta:+.1f}", ha="center", fontsize=9, color=col, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_OUT / "fig5_cross_domain_3tier.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] fig5_cross_domain_3tier.png")

# ============================================================
# Figure 2: Anchor variance decomposition — REVISED (exp_014 + 018 + 020)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

configs_full = [
    ("Kor4 (4-class)\nE", 41.00, 4.34, "#e74c3c", "4-class Korean"),
    ("Kor4 (3-class)\nH1", 40.67, 1.53, "#f39c12", "3-class Korean"),
    ("Kor2+Random2\nG1", 41.08, 2.50, "#9b59b6", "3-class, semantic diverse"),
    ("Kor2+Wes2\nD_mixed", 41.25, 0.87, "#27ae60", "3-class, Mixed anchor (Ekman)"),
    ("Wes4 (Ekman)\nC", 25.42, 0.38, "#888888", "0 Korean"),
]

x_pos = np.arange(len(configs_full))
means = [c[1] for c in configs_full]
stds = [c[2] for c in configs_full]
colors_list = [c[3] for c in configs_full]
labels = [c[0] for c in configs_full]

bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors_list, edgecolor="black", linewidth=0.8)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 1.3, f"σ={s:.2f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("accuracy (%)")
ax.set_title("Anchor variance decomposition — class redundancy + anchor origin + semantic content")
ax.set_ylim(15, 55)

# Annotations showing decomposition
ax.annotate("", xy=(1, 42), xytext=(0, 42), arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.3))
ax.text(0.5, 44, "Δσ=−2.81\n(class\nredundancy)", ha="center", fontsize=7, color="#e67e22")

ax.annotate("", xy=(3, 36), xytext=(1, 36), arrowprops=dict(arrowstyle="->", color="#16a085", lw=1.3))
ax.text(2, 34, "Δσ=−0.66\n(anchor\norigin)", ha="center", fontsize=7, color="#16a085")

ax.annotate("", xy=(3, 48), xytext=(2, 48), arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.3))
ax.text(2.5, 50, "Δσ=−1.63\n(semantic\ncontent)", ha="center", fontsize=7, color="#8e44ad")

ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig2_anchor_variance_revised.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] fig2_anchor_variance_revised.png")

# ============================================================
# Figure 3: k-scaling — Korean FER (saturation) + IEMOCAP (FLAT)
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 4.5))

k_list = [0, 2, 4, 8, 16]
kor_means = [29.08, 44.25, 42.75, 43.50, 42.33]
kor_stds = [0.76, 2.82, 5.02, 1.52, 1.84]

# IEMOCAP (exp_023: k=0,4,8)
k_list_iem = [0, 4, 8]
iem_means = [46.67, 46.17, 47.00]
iem_stds = [4.06, 6.39, 4.67]

ax.errorbar(k_list, kor_means, yerr=kor_stds, fmt="o-", color="#2c5282",
            capsize=5, markersize=8, linewidth=1.8, markerfacecolor="white",
            markeredgewidth=1.6, label="Korean FER AU")
ax.errorbar(k_list_iem, iem_means, yerr=iem_stds, fmt="s-", color="#e74c3c",
            capsize=5, markersize=8, linewidth=1.8, markerfacecolor="white",
            markeredgewidth=1.6, label="IEMOCAP (text)")
ax.axhline(25, color="gray", linestyle="--", lw=0.8, alpha=0.6)
ax.text(16.5, 25, "random", fontsize=8, color="gray", va="center")

# Annotations
ax.annotate("saturation", xy=(8, 43.5), xytext=(12, 36),
            arrowprops=dict(arrowstyle="->", color="#2c5282", lw=1.1), fontsize=9, color="#2c5282")
ax.annotate("flat (no ICL gain)", xy=(4, 46.2), xytext=(10, 52),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.1), fontsize=9, color="#e74c3c")

ax.set_xlabel("number of in-context exemplars (k)")
ax.set_ylabel("accuracy (%)")
ax.set_title("k-scaling across domains — ICL saturates on Korean FER, flat on IEMOCAP")
ax.set_xticks(k_list)
ax.set_ylim(20, 58)
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig3_kshot_cross_domain.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] fig3_kshot_cross_domain.png")

# ============================================================
# Figure 4: LoRA + ICL substitutability (kept)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
configs = ["LoRA only\n(56.75)", "LoRA + Mixed\n(2+2)", "LoRA + Korean\n(k=4)"]
vals = [56.75, 56.00, 52.75]
deltas = [0, -0.75, -4.00]
colors_ft = ["#2c5282", "#6c9bd5", "#e74c3c"]
bars = ax.bar(configs, vals, color=colors_ft, edgecolor="black", linewidth=0.8)
for b, v, d in zip(bars, vals, deltas):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    if d != 0:
        ax.text(b.get_x() + b.get_width()/2, v - 4, f"Δ={d:+.2f} pp", ha="center", va="top",
                fontsize=9, color="white", fontweight="bold")
ax.axhline(56.75, color="#2c5282", linestyle=":", lw=1.2, alpha=0.7)
ax.text(2.45, 56.9, "LoRA-only baseline", fontsize=8, color="#2c5282", ha="right")
ax.set_ylabel("accuracy (%)")
ax.set_title("Adding ICL on top of LoRA — substitutes, not complements")
ax.set_ylim(45, 62)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OUT / "fig4_lora_icl_substitute.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] fig4_lora_icl_substitute.png")

print("\n[DONE] 5 paper figures updated.")
