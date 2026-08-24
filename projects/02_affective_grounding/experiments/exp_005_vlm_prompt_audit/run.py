"""exp_005 VLM prompt audit — Qwen2.5-VL-7B local, zero-shot, 5 prompts.
Writes responses.jsonl incrementally (resumable). Use --limit N for smoke test.

  python run.py --limit 10          # smoke (50 calls)
  python run.py                      # full 3000 samples
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = Path("/mnt/hdd/ajy_25/results")
VAL_CSV = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_val_v_yonsei.csv")
PRED_CSV = RESULTS / "expB_baseline_seed42/paper_artifacts/predictions.csv"   # for real image paths
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CLASSES = ["angry", "happy", "neutral", "sad"]
PER_CELL = 50
SEED = 42

PROMPTS = {
    "P1_forced": "Choose the single emotion shown in this face from: angry, happy, neutral, sad. Answer with only one word.",
    "P2_visible": "Based ONLY on the visible facial expression, which emotion appears most visible? Choose from: angry, happy, neutral, sad. Reply as: label=<emotion>, confidence=<0-1>.",
    "P3_internal": "What emotion is this person most likely feeling internally? Choose from: angry, happy, neutral, sad. Reply as: label=<emotion>, confidence=<0-1>.",
    "P4_sufficiency": "Is this face image alone sufficient evidence to infer the person's INTERNAL emotion? Choose one: sufficient, ambiguous, insufficient. Reply as: verdict=<one>, likely=<emotion or none>.",
    "P5_action": "You are an assistant. Based only on this face image, what should you do? Choose ONE letter: A=no action, B=describe visible expression only, C=ask a gentle check-in question, D=infer a definite internal emotion, E=escalate to a human. Answer with the letter and a few words.",
}


def build_sample() -> pd.DataFrame:
    va = pd.read_csv(VAL_CSV)[["path", "label", "yon_reject_rate", "yon_n_evals"]]
    va["bn"] = va["path"].map(os.path.basename)
    pr = pd.read_csv(PRED_CSV)[["path", "true_label"]]
    pr["bn"] = pr["path"].map(os.path.basename)
    df = pr.merge(va.drop(columns=["path"]), on="bn", how="inner")  # real path + observer status
    def grp(r):
        if r.yon_n_evals == 0: return "C_uneval"
        if r.yon_reject_rate <= 1e-9: return "A_agree"
        if r.yon_reject_rate >= 0.5: return "B_reject"
        return "mid"
    df["grp"] = df.apply(grp, axis=1)
    df = df[df.grp != "mid"]
    rng = np.random.default_rng(SEED)
    picks = []
    for g in ["A_agree", "B_reject", "C_uneval"]:
        for c in CLASSES:
            cell = df[(df.grp == g) & (df.true_label == c)]
            n = min(PER_CELL, len(cell))
            if n:
                picks.append(cell.sample(n=n, random_state=int(rng.integers(1e9))))
    s = pd.concat(picks).reset_index(drop=True)
    s["sample_id"] = [f"{g}_{c}_{i}" for i, (g, c) in enumerate(zip(s.grp, s.true_label))]
    return s[["sample_id", "path", "true_label", "grp", "yon_reject_rate"]]


# --- response parsing ---
def find_label(t: str):
    t = t.lower()
    hits = [(t.find(c), c) for c in CLASSES if c in t]
    hits = [(p, c) for p, c in hits if p >= 0]
    return min(hits)[1] if hits else None

def find_conf(t: str):
    m = re.search(r"confidence\s*[=:]?\s*([01]?\.\d+|\d{1,3}\s*%|[01](?:\.0)?)", t.lower())
    if not m: m = re.search(r"\b([01]?\.\d+)\b", t)
    if not m: return None
    v = m.group(1).strip()
    if v.endswith("%"): return float(v[:-1]) / 100.0
    try: return float(v)
    except: return None

def find_sufficiency(t: str):
    t = t.lower()
    if "insufficient" in t: return "insufficient"
    if "ambiguous" in t: return "ambiguous"
    if "sufficient" in t: return "sufficient"
    return None

def find_choice(t: str):
    m = re.search(r"\b([A-E])\b", t.strip())
    if m: return m.group(1)
    tl = t.lower()
    for kw, c in [("no action", "A"), ("describe", "B"), ("check-in", "C"), ("check in", "C"),
                  ("definite", "D"), ("escalate", "E")]:
        if kw in tl: return c
    return None

def parse(prompt_id: str, raw: str) -> dict:
    if prompt_id == "P1_forced": return {"label": find_label(raw)}
    if prompt_id in ("P2_visible", "P3_internal"): return {"label": find_label(raw), "confidence": find_conf(raw)}
    if prompt_id == "P4_sufficiency": return {"verdict": find_sufficiency(raw), "label": find_label(raw)}
    if prompt_id == "P5_action": return {"choice": find_choice(raw)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N samples")
    a = ap.parse_args()

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    sample = build_sample()
    (HERE / "sample.csv").write_text(sample.to_csv(index=False))
    if a.limit:
        sample = sample.head(a.limit)
    print(f"samples: {len(sample)}  (limit={a.limit or 'full'})  prompts: {len(PROMPTS)}")

    resp_path = HERE / ("responses_smoke.jsonl" if a.limit else "responses.jsonl")
    done = set()
    if resp_path.exists():
        for l in resp_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add((d["sample_id"], d["prompt_id"]))
    print(f"already done: {len(done)}")

    print("loading Qwen2.5-VL-7B (4-bit nf4, to coexist with sc on shared GPU) ...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=64*28*28, max_pixels=200*28*28)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    f = open(resp_path, "a")
    n_done = 0
    for _, row in sample.iterrows():
        if not os.path.exists(row.path):
            continue
        for pid, ptext in PROMPTS.items():
            if (row.sample_id, pid) in done:
                continue
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{row.path}",
                 "min_pixels": 64*28*28, "max_pixels": 200*28*28},
                {"type": "text", "text": ptext}]}]
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                trimmed = gen[:, inputs.input_ids.shape[1]:]
                raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
                parsed = parse(pid, raw)
            except Exception as e:
                torch.cuda.empty_cache()
                raw = f"<ERROR:{type(e).__name__}>"; parsed = {}
                print(f"  ! skip {row.sample_id}/{pid}: {type(e).__name__}")
            rec = {"sample_id": row.sample_id, "prompt_id": pid, "grp": row.grp,
                   "self_report": row.true_label, "raw": raw, **parsed}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n_done += 1
        if n_done and n_done % 100 == 0:
            print(f"  ... {n_done} responses")
    f.close()
    print(f"DONE: wrote {n_done} new responses → {resp_path}")


if __name__ == "__main__":
    main()
