"""
exp_049 — 멀티모델 per-emotion ICL 천장 + confusion (EA-H005).
happy≫negative ICL 절대 천장이 Qwen 외 모델에서도 재현되는지(#7) + 부정감정이 어디로 새는지(self-attack).
모델 순차 로드(4-bit), 각자 끝나면 free. 5 seed balanced, ICL k4 (no-FACS numeric).
"""
import sys, json, glob, gc
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HERE = Path(__file__).resolve(); E = HERE.parents[1]
sys.path.insert(0, str(E / "exp_035_multimodel_gating"))
sys.path.insert(0, str(E / "exp_038_format_intervention"))
import run_gating as G
import run_gating_batched as RGB
import intervene as IV

HUB = "/mnt/hdd/ajy/caches/huggingface/hub"
MODELS = {
    "qwen7b":  "Qwen--Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama--Llama-3.1-8B-Instruct",
    "mistral": "mistralai--Mistral-7B-Instruct-v0.3",
    "yi6b":    "01-ai--Yi-1.5-6B-Chat",
    "falcon7b":"tiiuae--Falcon3-7B-Instruct",
}
SEEDS = [42, 123, 777, 1234, 2024]; BS = 32
EMOS = ["happy", "angry", "sad", "neutral"]


def preds_for(items, exemplars, model, tok, use_icl):
    msgs = [IV.prompt_numeric(x if x else "neutral.", exemplars if use_icl else None) for x, _ in items]
    out = []
    for b in range(0, len(msgs), BS):
        out.extend(G.parse_output(r) for r in RGB.gen_batch(msgs[b:b+BS], model, tok))
    return out


def run_model(key, repo):
    snaps = glob.glob(f"{HUB}/models--{repo}/snapshots/*")
    if not snaps:
        return {"error": "no snapshot"}
    snap = snaps[0]
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    acc = {e: [] for e in EMOS}
    conf = {e: {p: 0 for p in EMOS + ["other"]} for e in EMOS}  # ICL confusion
    for seed in SEEDS:
        items, ex = IV.load(seed, IV.numeric_fmt)
        y = [g for _, g in items]
        pr = preds_for(items, ex, model, tok, True)
        for e in EMOS:
            idx = [i for i, t in enumerate(y) if t == e]
            acc[e].append(float(np.mean([pr[i] == e for i in idx])) if idx else None)
            for i in idx:
                conf[e][pr[i] if pr[i] in EMOS else "other"] += 1
    res = {e: round(float(np.mean(acc[e])), 3) for e in EMOS}
    neg = np.mean([res["angry"], res["sad"]])
    res["_gap_happy_minus_neg"] = round(res["happy"] - neg, 3)
    res["_icl_confusion"] = conf
    del model; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    out = {}
    for key, repo in MODELS.items():
        print(f"\n##### {key} #####", flush=True)
        try:
            out[key] = run_model(key, repo)
            r = out[key]
            if "error" not in r:
                print(f">>> {key}: happy={r['happy']} angry={r['angry']} sad={r['sad']} "
                      f"neutral={r['neutral']} GAP={r['_gap_happy_minus_neg']}", flush=True)
            (HERE.parent / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as ex:
            out[key] = {"error": str(ex)[:200]}
            print(f"!!! {key} FAIL: {ex}", flush=True)
    npass = sum(1 for k, r in out.items() if "error" not in r and r.get("_gap_happy_minus_neg", 0) > 0.15)
    print(f"\n=== EXP_049 DONE — {npass}/{len(MODELS)} 모델 gap>0.15 ===")
    print(json.dumps({k: (v.get("_gap_happy_minus_neg") if "error" not in v else "ERR")
                      for k, v in out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
