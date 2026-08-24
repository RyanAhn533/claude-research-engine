"""
exp_040: Task-Recognition vs Task-Learning decomposition (Pan&Gao 2023 ACL; Min 2022 EMNLP).
The lit-scout's #1 graft to fix our missing positive mechanism.

For each (model, dataset): zero-shot, gold-label ICL, RANDOM-label ICL (exemplar labels
randomly reassigned). Then:
  TR = random_icl - zero      (task recognition: demos help even with wrong labels)
  TL = gold_icl   - random_icl(task learning: needs correct input->label mapping)
Prediction: AU(novel format) gain is mostly TL (random labels destroy it);
            IEMOCAP(familiar) gain is mostly TR (random labels are fine, TL~0).
=> TL becomes the POSITIVE scalar predicting ICL gain. Min-et-al random-label-is-fine
   replicates on text but BREAKS on AU = one-figure proof of modality gating.

AU uses no-FACS numeric prompt (exp_038). Batched 4-bit. 3 seeds.
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
# 7 seeds (was 3): first 5 match the clean gating panel for cross-experiment consistency,
# +2 more -> tighter paired CI on the TR/TL decomposition (reviewer stat-hardening).
SEEDS = [42, 123, 777, 2024, 31337, 1234, 9001]

def au_loader(seed): return IV.load(seed, IV.numeric_fmt)
DATASETS = {"au_nofacs": (au_loader, IV.prompt_numeric), "iemocap": (G.DATASETS["iemocap"][1], G.DATASETS["iemocap"][2])}

def randomize_labels(exemplars, seed):
    rng = np.random.default_rng(seed + 9999)
    return [(t, G.LABELS[rng.integers(0, len(G.LABELS))]) for (t, _) in exemplars]

@torch.inference_mode()
def run_cfg(items, ex, prompt_fn, model, tok, use_icl, bs=32):
    y=[g for _,g in items]; msgs=[prompt_fn(x if x else "neutral.", ex if use_icl else None) for x,_ in items]; preds=[]
    for b in range(0,len(msgs),bs):
        ps=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs[b:b+bs]]
        enc=tok(ps,return_tensors="pt",padding=True,truncation=True,max_length=2048).to(model.device)
        out=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id)
        preds.extend(G.parse_output(tok.decode(g,skip_special_tokens=True)) for g in out[:,enc.input_ids.shape[1]:])
    return float(accuracy_score(y,preds))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); args=ap.parse_args()
    snap=glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model}",flush=True)
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(snap,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(snap,quantization_config=bnb,device_map={"":0},torch_dtype=torch.float16,trust_remote_code=True).eval()
    results={}
    for ds,(loader,prompt_fn) in DATASETS.items():
        z=[]; gold=[]; rand=[]
        for s in SEEDS:
            items,ex=loader(s)
            z.append(run_cfg(items,ex,prompt_fn,model,tok,False))
            gold.append(run_cfg(items,ex,prompt_fn,model,tok,True))
            rand.append(run_cfg(items,randomize_labels(ex,s),prompt_fn,model,tok,True))
        z,gold,rand=np.array(z),np.array(gold),np.array(rand)
        TR=(rand.mean()-z.mean())*100; TL=(gold.mean()-rand.mean())*100; gain=(gold.mean()-z.mean())*100
        results[ds]={"zero":float(z.mean()*100),"gold_icl":float(gold.mean()*100),"random_icl":float(rand.mean()*100),
                     "gain_pp":float(gain),"TR_pp":float(TR),"TL_pp":float(TL),
                     "z_seeds":z.tolist(),"gold_seeds":gold.tolist(),"rand_seeds":rand.tolist()}
        print(f"  {args.model}/{ds}: zero={z.mean()*100:.1f} gold={gold.mean()*100:.1f} rand={rand.mean()*100:.1f} | gain={gain:+.1f} TR={TR:+.1f} TL={TL:+.1f}",flush=True)
    json.dump({"model":args.model,"seeds":SEEDS,"results":results}, open(OUT/f"tr_tl_{args.model}.json","w"), indent=2)
    print(f"[SAVED] tr_tl_{args.model}.json",flush=True)
