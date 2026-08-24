"""
Diagnostic: is qwen14b's +5.5pp familiar(iemocap) ICL gain a parsing artifact?
Hypothesis: 14B zero-shot outputs lack a clean 'LABEL:' line → parse falls back to
'neutral' → deflates zero-shot. ICL teaches format → gain is format-compliance, not emotion.
Measure fallback rate (no explicit LABEL: line) for zero_shot vs icl_k4.
"""
import sys, glob
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).parent))
import run_gating as G

HUB = G.HUB
LAB = G.LABELS

def parse_explicit(raw):
    """returns (label, had_explicit_LABEL_line)"""
    for line in raw.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("LABEL:"):
            lab = s.split(":", 1)[1].strip().lower()
            for l in LAB:
                if l in lab:
                    return l, True
    low = raw.lower()
    for l in LAB:
        if l in low:
            return l, False   # found by keyword scan, not explicit format
    return "neutral", False    # pure fallback

snap = glob.glob(f"{HUB}/models--Qwen--Qwen2.5-14B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(snap); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16).eval()
print("[load] 14b ok", flush=True)

novelty, loader, prompt_fn = G.DATASETS["iemocap"]
items, exemplars = loader(42)
items = items[:120]

def run(use_icl, tag):
    msgs = [prompt_fn(x if x else "neutral.", exemplars if use_icl else None) for x, _ in items]
    ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    explicit = 0; labs = []
    samples = []
    for b in range(0, len(ps), 32):
        enc = tok(ps[b:b+32], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        for g in out[:, enc.input_ids.shape[1]:]:
            raw = tok.decode(g, skip_special_tokens=True)
            lab, exp = parse_explicit(raw)
            explicit += int(exp); labs.append(lab)
            if len(samples) < 4: samples.append(raw.replace(chr(10), " ")[:120])
    print(f"\n[{tag}] explicit-LABEL rate = {explicit}/{len(items)} = {explicit/len(items)*100:.0f}%", flush=True)
    print(f"  pred dist: {dict(Counter(labs))}", flush=True)
    print(f"  sample raws: {samples}", flush=True)

run(False, "zero_shot")
run(True, "icl_k4")
print("\n[DONE] If zero_shot explicit-rate << icl_k4 → 14B familiar gain is FORMAT artifact, familiar stays flat.", flush=True)
