"""
exp_028: Hidden-representation cluster tightness — mechanism follow-up to exp_019.

Tests whether D_mixed's lower output variance (σ=0.87) vs E_kor4 (σ=4.34)
is explained at the HIDDEN-STATE level (last-layer representation before lm_head).

For each config, extract last-layer hidden state at the last input token
(generation position) for 80 test samples. Cluster by true label and compute:
  - within-class distance: avg distance to class centroid
  - between-class distance: avg distance from class centroid to other centroids
  - tightness = between / within (higher = better clustering)

Hypothesis: D_mixed yields tighter per-label clusters → more stable predictions
across seeds → lower σ. If tightness(D) > tightness(E), mechanism located.
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry", "happy", "neutral", "sad"]
KR_TO_EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}

AU_DESC = {
    "AU1": "inner brow raiser", "AU2": "outer brow raiser", "AU4": "brow lowerer",
    "AU5": "upper lid raiser", "AU6": "cheek raiser", "AU7": "lid tightener",
    "AU9": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip corner puller",
    "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher", "AU23": "lip tightener", "AU25": "lips part",
    "AU26": "jaw drop", "AU27": "mouth stretch",
}
KEY_AUS = list(AU_DESC.keys())

SEED = 42
N_SAMPLES = 80  # 20 per class


def au_to_text(row, threshold=50):
    active = []
    for au in KEY_AUS:
        if au in row and not pd.isna(row[au]) and row[au] >= threshold:
            active.append(f"{au} ({AU_DESC[au]})={row[au]:.0f}")
    if not active:
        scored = [(au, float(row[au])) for au in KEY_AUS if au in row and not pd.isna(row[au])]
        scored.sort(key=lambda x: -x[1])
        active = [f"{a} ({AU_DESC[a]})={v:.0f}" for a, v in scored[:3]]
    return ", ".join(active)


def format_au_dict(au_dict):
    parts = []
    for au, v in au_dict.items():
        if au in AU_DESC:
            parts.append(f"{au} ({AU_DESC[au]})={v}")
    return ", ".join(parts)


WESTERN_EKMAN = [
    (format_au_dict({"AU6": 80, "AU12": 85, "AU25": 60}), "happy"),
    (format_au_dict({"AU1": 70, "AU4": 65, "AU15": 75, "AU17": 55}), "sad"),
    (format_au_dict({"AU4": 80, "AU5": 65, "AU7": 78, "AU23": 72, "AU17": 60}), "angry"),
    (format_au_dict({"AU6": 10, "AU12": 15, "AU4": 12, "AU25": 20}), "neutral"),
]


def prepare():
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()

    pool_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pool_idx.extend(sub.sample(n=1, random_state=SEED).index.tolist())
    pool = df.loc[pool_idx].copy().reset_index(drop=True)
    pool["au_text"] = pool.apply(au_to_text, axis=1)
    remaining = df.drop(index=pool_idx)

    rng = np.random.default_rng(SEED)
    pieces = []
    per_class = N_SAMPLES // 4
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        pieces.append(sub.iloc[rng.choice(len(sub), size=per_class, replace=False)])
    test = pd.concat(pieces).reset_index(drop=True)

    kor_all = [(r["au_text"], r["emotion_en"]) for _, r in pool.iterrows()]
    D_mixed = kor_all[:2] + WESTERN_EKMAN[:2]
    E_kor4 = kor_all[:4]
    return test, {"D_mixed": D_mixed, "E_kor4": E_kor4}


def build_prompt(target_au_text, exemplars, origin_label):
    SYS_BASE = (
        "You are a facial emotion classifier using FACS action units (AUs). "
        "Given detected AU intensities (0-100) of a Korean subject, classify into: "
        "angry, happy, neutral, sad.\n"
    )
    sys_p = SYS_BASE
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} examples ({origin_label}) with correct labels:\n"
        for i, (au_text, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  AUs: {au_text}\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
    return [{"role": "system", "content": sys_p},
            {"role": "user", "content": f"Target AUs:\n{target_au_text}\n\nClassify emotion."}]


def extract_hidden(model, tokenizer, test_df, exemplars, origin_label, tag):
    hidden = []  # list of (vec, true_label)
    t0 = time.time()
    for i, row in test_df.iterrows():
        target = au_to_text(row)
        msgs = build_prompt(target, exemplars, origin_label)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        # last layer, last token
        h = out.hidden_states[-1][0, -1].float().cpu().numpy()
        hidden.append((h, row["emotion_en"]))
        del out
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"  [{tag}] {i+1}/{len(test_df)} {time.time()-t0:.1f}s")
    return hidden


def cluster_stats(hidden, labels=LABELS):
    by_label = defaultdict(list)
    for h, l in hidden:
        by_label[l].append(h)
    centroids = {l: np.mean(by_label[l], axis=0) for l in labels if by_label[l]}

    within = {}
    for l, vecs in by_label.items():
        if not vecs: continue
        c = centroids[l]
        dists = [np.linalg.norm(v - c) for v in vecs]
        within[l] = float(np.mean(dists))

    between = {}
    for l in labels:
        if l not in centroids: continue
        others = [centroids[l2] for l2 in labels if l2 != l and l2 in centroids]
        if not others: continue
        between[l] = float(np.mean([np.linalg.norm(centroids[l] - o) for o in others]))

    tightness = {l: (between[l] / within[l] if within[l] > 0 else 0.0)
                 for l in labels if l in within and l in between}

    return {
        "within_class_dist": within,
        "between_class_dist": between,
        "tightness_ratio": tightness,
        "within_overall": float(np.mean(list(within.values()))),
        "between_overall": float(np.mean(list(between.values()))),
        "tightness_overall": float(np.mean(list(tightness.values()))),
    }


test_df, ex_packs = prepare()
print(f"[info] test N={len(test_df)}, per-class dist: {test_df['emotion_en'].value_counts().to_dict()}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

results = {}
hidden_by_config = {}
for name, ex in ex_packs.items():
    origin = "Korean + Western mix" if name == "D_mixed" else "Korean subjects"
    print(f"\n=== {name} ===")
    hidden = extract_hidden(model, tokenizer, test_df, ex, origin, name)
    hidden_by_config[name] = hidden
    stats = cluster_stats(hidden)
    results[name] = stats
    print(f"  within-overall  = {stats['within_overall']:.3f}")
    print(f"  between-overall = {stats['between_overall']:.3f}")
    print(f"  tightness(between/within) = {stats['tightness_overall']:.4f}")
    print(f"  per-class within: {stats['within_class_dist']}")
    print(f"  per-class tightness: {stats['tightness_ratio']}")

# Comparison summary
tD = results["D_mixed"]["tightness_overall"]
tE = results["E_kor4"]["tightness_overall"]
print(f"\n=== SUMMARY ===")
print(f"D_mixed tightness = {tD:.4f}")
print(f"E_kor4  tightness = {tE:.4f}")
print(f"Ratio D/E = {tD/tE if tE > 0 else float('inf'):.3f}")
if tD > tE:
    print(f"→ D_mixed has TIGHTER clusters (mechanism confirmed at hidden-rep level)")
else:
    print(f"→ D_mixed does NOT have tighter clusters (mechanism is elsewhere)")

with open(OUT / "hidden_cluster_results.json", "w") as f:
    json.dump({
        "task": "Hidden-state last-layer cluster tightness — D_mixed vs E_kor4",
        "N_samples": len(test_df), "seed": SEED,
        "results": results,
        "tightness_comparison": {"D": tD, "E": tE, "ratio_D_over_E": float(tD / tE) if tE > 0 else None},
    }, f, indent=2)

# Figure: per-class within distance bar
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(LABELS)); w = 0.35
D_within = [results["D_mixed"]["within_class_dist"].get(l, 0) for l in LABELS]
E_within = [results["E_kor4"]["within_class_dist"].get(l, 0) for l in LABELS]
ax.bar(x - w/2, D_within, w, label="D_mixed (σ_out=0.87)", color="#27ae60", edgecolor="black", linewidth=0.7)
ax.bar(x + w/2, E_within, w, label="E_kor4 (σ_out=4.34)", color="#e74c3c", edgecolor="black", linewidth=0.7)
for i, (d, e) in enumerate(zip(D_within, E_within)):
    ax.text(i - w/2, d + 0.5, f"{d:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, e + 0.5, f"{e:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("within-class distance (lower = tighter cluster)")
ax.set_title(f"Hidden-rep cluster tightness — D (t={tD:.3f}) vs E (t={tE:.3f})")
ax.legend()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "hidden_cluster_within.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n[saved] cache/hidden_cluster_results.json + hidden_cluster_within.png")
