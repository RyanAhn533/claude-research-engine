"""
exp_019b: Attention entropy across MULTIPLE layers (1, 7, 14, 21, 27).
Extends exp_019 to check whether mechanism shows at different depth.
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
N_SAMPLES = 40  # slightly fewer due to multi-layer memory
LAYERS = [1, 7, 14, 21, 27]  # 28 layer Qwen2.5-7B


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


def find_exemplar_token_ranges(tokenizer, prompt_text, n_exemplars=4):
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    markers = []
    for i in range(1, n_exemplars + 2):
        marker = f"Example {i}:"
        markers.append(prompt_text.find(marker) if prompt_text.find(marker) >= 0 else len(prompt_text))

    def char_to_tok(char_pos):
        for idx, (s, e) in enumerate(offsets):
            if s >= char_pos:
                return idx
        return len(offsets) - 1

    return [(char_to_tok(markers[i]), char_to_tok(markers[i + 1])) for i in range(n_exemplars)]


def extract_multi_layer_entropy(model, tokenizer, test_df, exemplars, origin_label, tag):
    per_layer_entropies = {L: [] for L in LAYERS}
    per_layer_weights = {L: [] for L in LAYERS}
    t0 = time.time()

    for i, row in test_df.iterrows():
        target = au_to_text(row)
        msgs = build_prompt(target, exemplars, origin_label)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            ex_ranges = find_exemplar_token_ranges(tokenizer, prompt_text, n_exemplars=len(exemplars))
        except Exception:
            continue

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)

        for L in LAYERS:
            attn = out.attentions[L][0].mean(dim=0)  # (seq, seq)
            last = attn[-1]
            weights = []
            for (s, e) in ex_ranges:
                s = min(s, last.shape[0] - 1); e = min(e, last.shape[0])
                weights.append(last[s:e].sum().item())
            total = sum(weights) + 1e-9
            probs = np.array(weights) / total
            probs = np.clip(probs, 1e-9, 1.0)
            H = float(-np.sum(probs * np.log(probs)))
            per_layer_entropies[L].append(H)
            per_layer_weights[L].append(probs.tolist())

        # free attention tensors
        del out
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i+1}/{len(test_df)} {time.time()-t0:.1f}s")

    return per_layer_entropies, per_layer_weights


test_df, ex_packs = prepare()
print(f"[info] test N={len(test_df)} layers={LAYERS}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
    attn_implementation="eager",
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB, total layers: {len(model.model.layers)}")

results = {}
for name, ex in ex_packs.items():
    origin = "Korean + Western mix" if name == "D_mixed" else "Korean subjects"
    print(f"\n=== {name} ===")
    ents, weights = extract_multi_layer_entropy(model, tokenizer, test_df, ex, origin, name)
    layer_stats = {}
    for L in LAYERS:
        arr = np.array(ents[L])
        layer_stats[f"L{L}"] = {
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "weights_mean": np.mean(weights[L], axis=0).tolist() if weights[L] else [],
        }
    results[name] = {"layers": layer_stats, "raw_entropies": {f"L{L}": ents[L] for L in LAYERS}}
    for L in LAYERS:
        s = layer_stats[f"L{L}"]
        print(f"  L{L:2d}: entropy={s['mean']:.3f} ± {s['std']:.3f}, weights={np.array(s['weights_mean']).round(3)}")

with open(OUT / "multi_layer_entropy_results.json", "w") as f:
    json.dump({
        "task": "Attention entropy across layers (1,7,14,21,27)",
        "N_samples": len(test_df), "layers": LAYERS, "max_entropy_ln4": float(np.log(4)),
        "results": results,
    }, f, indent=2)

# Figure
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, color in [("D_mixed", "#27ae60"), ("E_kor4", "#e74c3c")]:
    means = [results[name]["layers"][f"L{L}"]["mean"] for L in LAYERS]
    stds = [results[name]["layers"][f"L{L}"]["std"] for L in LAYERS]
    ax.errorbar(LAYERS, means, yerr=stds, fmt="o-", color=color,
                capsize=5, markersize=8, linewidth=1.8, markerfacecolor="white",
                markeredgewidth=1.6, label=name)
ax.axhline(np.log(4), color="black", linestyle="--", lw=0.8, alpha=0.6)
ax.text(27, np.log(4) - 0.05, "uniform (ln 4)", fontsize=8, color="black", ha="right")
ax.set_xlabel("transformer layer index (of 28)")
ax.set_ylabel("attention entropy over 4 exemplars")
ax.set_title("Multi-layer attention entropy — D_mixed vs E_kor4")
ax.set_xticks(LAYERS)
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "multi_layer_entropy.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n[saved] cache/multi_layer_entropy_results.json + multi_layer_entropy.png")
