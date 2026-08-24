"""
exp_056: UNLABELED-exemplar control -> disentangle FORMAT GROUNDING from TASK RECOGNITION.

Motivation (reviewer-critical): our zero-shot prompt ALREADY names the task + label space
+ (for AU) FACS prototypes. So the TR = random_icl - zero from exp_040 does NOT cleanly
measure "task recognition" in the Pan&Gao sense (revealing task identity) -- the model
already knows the task at zero-shot. What demonstrations may add on a NOVEL format (AU) is
schema/FORMAT GROUNDING: how to map the unfamiliar representation onto the known task.

Add a 4th condition -> 3-way decomposition:
  zero       : no exemplars
  unlabeled  : exemplar INPUTS shown, LABEL lines removed ("Here are N examples:")
  random     : exemplars + DERANGED labels (no exemplar keeps its true label)
  gold       : exemplars + correct labels

  GROUNDING  = unlabeled - zero      (adapting the input schema to the known task)
  RECOG/LBLSP= random    - unlabeled (label-space / task cue beyond bare inputs)
  MAPPING(TL)= gold      - random    (learning the correct input->label mapping)

Prediction (format-grounding hypothesis): on AU(novel), GROUNDING >> 0 and is the bulk of
the gain; on IEMOCAP(familiar), GROUNDING ~ 0 (nothing to ground -- inputs already familiar).
If confirmed, the paper's mechanism is grounding, not task-identity recognition -> retitle.

AU prompt (prompt_au) MIRRORS exp_038 prompt_numeric exactly (only the label line +
"with correct labels" phrase are dropped in unlabeled mode). NOTE: prompt_text does NOT
carry exp_035 text_prompt's "Here are N labeled examples:" header, so exp_056 TEXT absolute
gains must NOT be cross-compared to exp_035 published numbers -- use the WITHIN-exp_056
decomposition only (all 4 conditions share this builder, so GR/RC/TL stay internally valid).
Batched 4-bit.
"""
import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "exp_035_multimodel_gating"))
sys.path.insert(0, str(HERE.parents[1] / "exp_038_format_intervention"))
import run_gating as G
import intervene as IV

OUT = HERE.parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS_FULL = [42, 123, 777, 2024, 31337, 1234, 9001]


# ---------- prompt builders (mirror exp_038 prompt_numeric / exp_035 text_prompt) ----------
def prompt_au(x, ex, unlabeled=False):
    sys_p = ("You are a facial emotion classifier using FACS action units (AUs). "
             "Given detected AU intensities (0-100) of a Korean subject, classify into: angry, happy, neutral, sad.\n")
    if ex:
        sys_p += (f"Here are {len(ex)} Korean examples:\n" if unlabeled
                  else f"Here are {len(ex)} Korean examples with correct labels:\n")
        for i, (t, l) in enumerate(ex):
            sys_p += f"\nExample {i+1}:\n  AUs: {t}\n" + ("" if unlabeled else f"  LABEL: {l}\n")
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    u = f"Target AUs:\n{x}\n\nClassify emotion." if ex else f"AUs: {x}\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": u}]


def prompt_text(x, ex, unlabeled=False):
    sys_p = ("You are an emotion classifier for spoken English utterances. "
             "Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.\n")
    if ex:
        if not unlabeled:
            sys_p += ""  # (exp_035 text prompt has no "with correct labels" preamble)
        for i, (t, l) in enumerate(ex):
            sys_p += f"\nExample {i+1}:\n  Utterance: \"{t}\"\n" + ("" if unlabeled else f"  LABEL: {l}\n")
        sys_p += "\nNow classify the target."
    sys_p += "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
    u = f"Utterance: \"{x}\"\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": u}]


def derange_labels(exemplars, seed):
    """Deranged labels: no exemplar keeps its own true label (avoids accidental-correct inflation)."""
    labs = [l for _, l in exemplars]; n = len(labs)
    rng = np.random.default_rng(seed + 9999)
    for _ in range(2000):
        perm = rng.permutation(n)
        if all(labs[perm[i]] != labs[i] for i in range(n)):
            return [(exemplars[i][0], labs[perm[i]]) for i in range(n)]
    # fallback: cyclic shift (guaranteed derangement when all labels distinct)
    return [(exemplars[i][0], labs[(i + 1) % n]) for i in range(n)]


@torch.inference_mode()
def run_cfg(items, ex, prompt_fn, model, tok, mode, bs=32):
    """mode in {zero, unlabeled, random, gold}. ex already carries the right labels for random/gold."""
    use_ex = None if mode == "zero" else ex
    unlab = (mode == "unlabeled")
    y = [g for _, g in items]
    msgs = [prompt_fn(x if x else "neutral.", use_ex, unlabeled=unlab) for x, _ in items]
    preds = []
    for b in range(0, len(msgs), bs):
        ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
        preds.extend(G.parse_output(tok.decode(g, skip_special_tokens=True)) for g in out[:, enc.input_ids.shape[1]:])
    return float(accuracy_score(y, preds))


def au_loader(seed):
    return IV.load(seed, IV.numeric_fmt)


DATASETS = {
    "au_nofacs": (au_loader, prompt_au),
    "iemocap":   (G.DATASETS["iemocap"][1], prompt_text),
    "meld":      (G.DATASETS["meld"][1], prompt_text),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="")  # empty -> SEEDS_FULL; else comma list (smoke: "42")
    ap.add_argument("--datasets", default="au_nofacs,iemocap")  # smoke default: 2 datasets
    args = ap.parse_args()
    SEEDS = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS_FULL
    dss = [d.strip() for d in args.datasets.split(",")]

    snap = glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model} seeds={SEEDS} datasets={dss}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16, trust_remote_code=True).eval()
    results = {}
    for ds in dss:
        loader, prompt_fn = DATASETS[ds]
        z = []; unl = []; rnd = []; gold = []
        for s in SEEDS:
            items, ex = loader(s)
            z.append(run_cfg(items, ex, prompt_fn, model, tok, "zero"))
            unl.append(run_cfg(items, ex, prompt_fn, model, tok, "unlabeled"))
            rnd.append(run_cfg(items, derange_labels(ex, s), prompt_fn, model, tok, "random"))
            gold.append(run_cfg(items, ex, prompt_fn, model, tok, "gold"))
        z, unl, rnd, gold = map(np.array, (z, unl, rnd, gold))
        GR = (unl.mean() - z.mean()) * 100      # format grounding
        RC = (rnd.mean() - unl.mean()) * 100     # recognition / label-space
        TL = (gold.mean() - rnd.mean()) * 100    # mapping learning
        gain = (gold.mean() - z.mean()) * 100
        results[ds] = {"zero": float(z.mean()*100), "unlabeled": float(unl.mean()*100),
                       "random": float(rnd.mean()*100), "gold": float(gold.mean()*100),
                       "gain_pp": float(gain), "GROUNDING_pp": float(GR), "RECOG_pp": float(RC), "TL_pp": float(TL),
                       "z_seeds": z.tolist(), "unl_seeds": unl.tolist(), "rnd_seeds": rnd.tolist(), "gold_seeds": gold.tolist()}
        print(f"  {args.model}/{ds}: zero={z.mean()*100:.1f} unl={unl.mean()*100:.1f} rand={rnd.mean()*100:.1f} gold={gold.mean()*100:.1f} "
              f"| gain={gain:+.1f} GROUND={GR:+.1f} RECOG={RC:+.1f} TL={TL:+.1f}", flush=True)
    json.dump({"model": args.model, "seeds": SEEDS, "results": results},
              open(OUT / f"unlabeled_{args.model}.json", "w"), indent=2)
    print(f"[SAVED] unlabeled_{args.model}.json", flush=True)
