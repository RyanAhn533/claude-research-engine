"""
exp_048 — per-emotion ICL gain (합본 척추, 재칼질).
가설 EA-H003: ICL은 happy(명확)에선 높은 절대정확도, 한국 부정감정(angry/sad, AU-ambiguous)에선
gain 있어도 절대 천장 낮음. = format-readability 복구하나 cultural AU-ambiguity는 못 넘음.
qwen7b 4-bit, balanced 100/class, 5 seed, zero vs icl_k4 (no-FACS numeric).
"""
import sys, json, glob
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HERE = Path(__file__).resolve(); E = HERE.parents[1]
sys.path.insert(0, str(E / "exp_035_multimodel_gating"))
sys.path.insert(0, str(E / "exp_038_format_intervention"))
import run_gating as G
import run_gating_batched as RGB
import intervene as IV

CACHE = "/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
SEEDS = [42, 123, 777, 1234, 2024]; BS = 32
EMOS = ["happy", "angry", "sad", "neutral"]


def per_emotion_acc(items, exemplars, model, tok, use_icl):
    y = [g for _, g in items]
    msgs = [IV.prompt_numeric(x if x else "neutral.", exemplars if use_icl else None) for x, _ in items]
    preds = []
    for b in range(0, len(msgs), BS):
        preds.extend(G.parse_output(r) for r in RGB.gen_batch(msgs[b:b+BS], model, tok))
    acc = {}
    for e in EMOS:
        idx = [i for i, t in enumerate(y) if t == e]
        acc[e] = float(np.mean([preds[i] == e for i in idx])) if idx else None
    return acc


def main():
    snap = sorted(glob.glob(f"{CACHE}/snapshots/*"))[0]
    print(f"[load] {snap}", flush=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

    z = {e: [] for e in EMOS}; ic = {e: [] for e in EMOS}
    for seed in SEEDS:
        items, ex = IV.load(seed, IV.numeric_fmt)
        za = per_emotion_acc(items, ex, model, tok, False)
        ia = per_emotion_acc(items, ex, model, tok, True)
        for e in EMOS: z[e].append(za[e]); ic[e].append(ia[e])
        print(f"  seed{seed}: " + " ".join(f"{e}:{za[e]:.2f}->{ia[e]:.2f}" for e in EMOS), flush=True)

    out = {}
    for e in EMOS:
        zm, im = float(np.mean(z[e])), float(np.mean(ic[e]))
        out[e] = {"zero": round(zm, 3), "icl": round(im, 3), "gain_pp": round((im - zm) * 100, 1)}
    neg = np.mean([out["angry"]["icl"], out["sad"]["icl"]])
    out["_ceiling_gap_happy_minus_neg"] = round(out["happy"]["icl"] - neg, 3)
    (HERE.parent / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== PER-EMOTION DONE ===")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
