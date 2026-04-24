"""
exp_019: Attention entropy analysis — Paper B §5.2 mechanism evidence.

Tests whether Mixed anchors (D_mixed) cause more uniform attention over exemplars
than pure Korean (E_kor4), which would explain σ reduction mechanism.

For each config at seed=42:
  - 50 test samples (subset of exp_014's 400)
  - Forward pass with output_attentions=True
  - Extract mid-layer (layer 14/28) attention from last input token to exemplar tokens
  - Compute entropy over the 4 exemplar positions
  - Compare distributions

Output: per-config entropy list + mean ± std + histogram figure.

Hypothesis: D_mixed entropy higher (more uniform) than E_kor4 (more peaked) → σ↓.
"""
import os, sys, json, time, glob
from pathlib import Path
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
N_SAMPLES = 50
LAYER = 14  # mid layer of 28 in Qwen2.5-7B


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
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        pieces.append(sub.iloc[rng.choice(len(sub), size=min(N_SAMPLES // 4 + 1, len(sub)), replace=False)])
    test = pd.concat(pieces).reset_index(drop=True).head(N_SAMPLES)

    kor_all = [(r["au_text"], r["emotion_en"]) for _, r in pool.iterrows()]
    D_mixed = kor_all[:2] + WESTERN_EKMAN[:2]    # 2 Kor + 2 Wes (Ekman happy, sad)
    E_kor4 = kor_all[:4]                          # 4 Korean all-class

    return test, {"D_mixed": D_mixed, "E_kor4": E_kor4}


def build_prompt_with_marker(target_au_text, exemplars, origin_label):
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


def find_exemplar_token_ranges(tokenizer, prompt_text, n_exemplars=4):
    """Return list of (start, end) token indices for each 'Example i:' block."""
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    text = tokenizer.decode(tokens, skip_special_tokens=False)

    # Find character positions of "Example i:" headers
    markers = []
    for i in range(1, n_exemplars + 2):
        marker = f"Example {i}:"
        pos = text.find(marker)
        markers.append(pos if pos >= 0 else len(text))
    # n_exemplars + 1 to get final bound ("Now classify the target." region)

    # Map char positions to token indices using offset_mapping
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def char_to_tok(char_pos):
        for idx, (s, e) in enumerate(offsets):
            if s >= char_pos:
                return idx
        return len(offsets) - 1

    ranges = []
    for i in range(n_exemplars):
        start = char_to_tok(markers[i])
        end = char_to_tok(markers[i + 1])
        ranges.append((start, end))
    return ranges


def extract_entropy(model, tokenizer, test_df, exemplars, origin_label, tag):
    entropies = []
    per_exemplar_weights = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        target = au_to_text(row)
        msgs = build_prompt_with_marker(target, exemplars, origin_label)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        try:
            ex_ranges = find_exemplar_token_ranges(tokenizer, prompt_text, n_exemplars=len(exemplars))
        except Exception:
            continue

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)

        # out.attentions[LAYER]: (batch=1, heads, seq, seq)
        attn = out.attentions[LAYER][0].mean(dim=0)  # avg over heads: (seq, seq)
        last = attn[-1]  # last position attends to all: (seq,)

        # Sum attention mass over each exemplar range
        weights = []
        for (s, e) in ex_ranges:
            s = min(s, last.shape[0] - 1)
            e = min(e, last.shape[0])
            w = last[s:e].sum().item()
            weights.append(w)
        total = sum(weights) + 1e-9
        probs = np.array(weights) / total
        probs = np.clip(probs, 1e-9, 1.0)
        H = -np.sum(probs * np.log(probs))
        entropies.append(float(H))
        per_exemplar_weights.append(probs.tolist())

        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i+1}/{len(test_df)} {time.time()-t0:.1f}s")

    return entropies, per_exemplar_weights


test_df, ex_packs = prepare()
print(f"[info] test N={len(test_df)}  exemplar packs: {list(ex_packs.keys())}")

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
for name, ex in ex_packs.items():
    origin = "Korean + Western mix" if name == "D_mixed" else "Korean subjects"
    print(f"\n=== {name} ===")
    entropies, weights = extract_entropy(model, tokenizer, test_df, ex, origin, name)
    arr = np.array(entropies)
    results[name] = {
        "entropies": entropies,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_exemplar_weights_mean": np.mean(weights, axis=0).tolist() if weights else [],
    }
    print(f"  entropy: mean={arr.mean():.3f} ± {arr.std(ddof=1):.3f} (max=ln(4)={np.log(4):.3f})")
    print(f"  per-exemplar weight avg: {np.mean(weights, axis=0).round(3)}")

# Save JSON
with open(OUT / "attention_entropy_results.json", "w") as f:
    json.dump({
        "task": "Attention entropy over 4 exemplars, layer 14, seed=42",
        "N_samples": len(test_df), "layer": LAYER, "max_entropy_ln4": np.log(4),
        "results": results,
    }, f, indent=2)

# Figure: histograms
fig, ax = plt.subplots(figsize=(7, 4))
for name, color in [("D_mixed", "#27ae60"), ("E_kor4", "#e74c3c")]:
    ents = results[name]["entropies"]
    if ents:
        ax.hist(ents, bins=15, alpha=0.55, color=color,
                label=f"{name} (μ={results[name]['mean']:.3f}±{results[name]['std']:.3f})",
                edgecolor="black", linewidth=0.5)
ax.axvline(np.log(4), color="black", linestyle="--", lw=0.8, alpha=0.7)
ax.text(np.log(4) - 0.02, ax.get_ylim()[1] * 0.9, "uniform\n(ln 4)", ha="right", fontsize=8)
ax.set_xlabel("attention entropy over 4 exemplars (layer 14)")
ax.set_ylabel("count of test samples")
ax.set_title("Attention entropy: D_mixed (Mixed anchor) vs E_kor4 (pure Korean)")
ax.legend(loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "attention_entropy_hist.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n[saved] {OUT}/attention_entropy_results.json + attention_entropy_hist.png")
