"""
exp_055: GRADED format-novelty -> graded ICL gain (within-task causal dose-response).

Rescues the fertility contribution. exp_042 showed fertility only SEPARATES novel vs
familiar (between-format, within-family rho~0). Reviewer objection: "fertility is just a
novel/familiar label, not a graded predictor." This falsifies-or-confirms that objection
by varying novelty CONTINUOUSLY on ONE task (SST-2), holding content fixed:

  graded leetspeak: substitute a fraction p of eligible chars, p in {0,.25,.5,.75,1}.
  Higher p -> more token fragmentation -> higher fertility, same sentence & label.

For each p: zero / gold-ICL / random-ICL accuracy -> gain, TR, TL, and measured fertility.
Test: WITHIN one task, does ICL gain rise MONOTONICALLY with fertility across the 5 levels?
  - monotone (Spearman rho ~ +1 per model) => fertility is a GRADED causal predictor
    (upgrade: not merely a novel-vs-familiar cluster label). STRONGER contribution.
  - step/threshold (flat then jump) => honest boundary: novelty acts as a threshold, not
    a dose. Either way it is a publishable, decisive result.

Deterministic substitution (hashlib, cross-process stable). Batched 4-bit, greedy.
"""
import os, sys, json, glob, argparse, hashlib
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_035_multimodel_gating"))
import run_gating as G
os.environ.setdefault("HF_HOME", "/mnt/hdd/ajy/caches/huggingface")
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
LABELS = ["positive", "negative"]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "g": "9", "l": "1"}


def graded_leet(s, p):
    """Substitute fraction ~p of eligible chars, deterministic per (sentence, level)."""
    s = s.strip().lower()
    if p <= 0:
        return s
    seed = int.from_bytes(hashlib.md5(f"{s}|{p}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return "".join(LEET[c] if (c in LEET and rng.random() < p) else c for c in s)


_DS = None
def get_ds():
    global _DS
    if _DS is None:
        d = load_dataset("stanfordnlp/sst2", split="train")
        _DS = [(r["sentence"], "positive" if r["label"] == 1 else "negative")
               for r in d if len(r["sentence"].split()) >= 4]
    return _DS


def load(seed, p, n_test=100):
    data = get_ds(); rng = np.random.default_rng(seed)
    pos = [x for x in data if x[1] == "positive"]; neg = [x for x in data if x[1] == "negative"]
    def pick(pool, k): return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    ex = [(graded_leet(s, p), l) for s, l in pick(pos, 2) + pick(neg, 2)]
    test = [(graded_leet(s, p), l) for s, l in pick(pos, n_test) + pick(neg, n_test)]
    return test, ex


def prompt(x, ex):
    sys_p = "You are a sentiment classifier. Given a movie-review snippet, answer positive or negative.\n"
    if ex:
        sys_p += f"Here are {len(ex)} labeled examples:\n"
        for i, (t, l) in enumerate(ex):
            sys_p += f"\nExample {i+1}:\n  \"{t}\"\n  LABEL: {l}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <positive or negative>\nREASON: <one sentence>"
    user = f"Target:\n\"{x}\"\n\nClassify." if ex else f"\"{x}\"\n\nClassify."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]


def parse_bin(raw):
    for line in raw.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("LABEL:"):
            v = s.split(":", 1)[1].strip().lower()
            if "positive" in v or "pos" in v: return "positive"
            if "negative" in v or "neg" in v: return "negative"
    low = raw.lower()
    return "positive" if (low.count("positive") >= low.count("negative")) else "negative"


def rand_labels(ex, seed):
    rng = np.random.default_rng(seed + 9999)
    return [(t, LABELS[rng.integers(0, 2)]) for t, _ in ex]


@torch.inference_mode()
def run(items, ex, model, tok, use_icl, bs=32):
    y = [g for _, g in items]; msgs = [prompt(x, ex if use_icl else None) for x, _ in items]; pr = []
    for b in range(0, len(msgs), bs):
        ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
        pr.extend(parse_bin(tok.decode(g, skip_special_tokens=True)) for g in out[:, enc.input_ids.shape[1]:])
    return float(accuracy_score(y, pr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="42,123,777"); ap.add_argument("--ntest", type=int, default=100)
    args = ap.parse_args()
    SEEDS = [int(s) for s in args.seeds.split(",")]
    snap = glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model} seeds={SEEDS} ntest={args.ntest}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.float16, trust_remote_code=True).eval()
    res = {}
    for p in LEVELS:
        z = []; gold = []; rnd = []; fert = 0.0
        for s in SEEDS:
            items, ex = load(s, p, args.ntest)
            z.append(run(items, ex, model, tok, False))
            gold.append(run(items, ex, model, tok, True))
            rnd.append(run(items, rand_labels(ex, s), model, tok, True))
            if s == SEEDS[0]:
                toks = [len(tok(x).input_ids) for x, _ in items]; wds = [max(1, len(x.split())) for x, _ in items]
                fert = float(np.mean([t/w for t, w in zip(toks, wds)]))
        z, gold, rnd = np.array(z), np.array(gold), np.array(rnd)
        res[f"p{p}"] = {"p": p, "zero": float(z.mean()*100), "gold": float(gold.mean()*100),
                        "random": float(rnd.mean()*100), "gain_pp": float((gold.mean()-z.mean())*100),
                        "TR_pp": float((rnd.mean()-z.mean())*100), "TL_pp": float((gold.mean()-rnd.mean())*100),
                        "fertility": fert, "gain_seeds": ((gold-z)*100).tolist()}
        r = res[f"p{p}"]
        print(f"  p={p}: zero={r['zero']:.1f} gold={r['gold']:.1f} rand={r['random']:.1f} | "
              f"gain={r['gain_pp']:+.1f} TR={r['TR_pp']:+.1f} TL={r['TL_pp']:+.1f} fert={r['fertility']:.2f}", flush=True)
    # within-task monotonicity: Spearman(fertility, gain) across the 5 levels
    from scipy.stats import spearmanr
    ferts = [res[f"p{p}"]["fertility"] for p in LEVELS]; gains = [res[f"p{p}"]["gain_pp"] for p in LEVELS]
    rho, pv = spearmanr(ferts, gains)
    res["_within_task_spearman"] = {"rho": float(rho), "p": float(pv), "ferts": ferts, "gains": gains}
    json.dump({"model": args.model, "seeds": SEEDS, "results": res}, open(OUT/f"graded_{args.model}.json", "w"), indent=2)
    print(f"[MONOTONE] within-task Spearman(fertility,gain) rho={rho:+.3f} p={pv:.3f} "
          f"(rho~+1 => graded predictor; flat/step => threshold)", flush=True)
    print(f"[SAVED] graded_{args.model}.json", flush=True)
