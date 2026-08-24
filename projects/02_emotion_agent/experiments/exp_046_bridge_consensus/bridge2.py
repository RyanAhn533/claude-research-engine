"""
Bridge 2 — consensus-stratified ICL (합본 척추 실험).

가설: ICL gain(포맷 복구)은 consensus 동의/거부 두 층 모두에서 나타나지만,
절대 정확도 천장은 거부(low-consensus)층에서 낮게 유지된다.
→ "ICL은 format-novelty를 고치지, label-ambiguity는 못 넘는다."
   = A(format gating) + B(consensus ambiguity)를 한 인과 스토리로 묶음.

입력: AU-as-text (no-FACS numeric, exp_038 IV.prompt_numeric).
층: is_selected==1 (consensus 동의) vs ==0 (거부) — AU parquet에 이미 있음.
모델: qwen7b 4-bit batched. 5 seed.
"""
import sys, json, glob, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HERE = Path(__file__).resolve()
E = HERE.parents[1]
sys.path.insert(0, str(E / "exp_035_multimodel_gating"))
sys.path.insert(0, str(E / "exp_038_format_intervention"))
import run_gating as G            # LABELS, KR_TO_EN, AU_PARQUET, parse_output
import run_gating_batched as RGB  # gen_batch, run_cfg (batched 23x)
import intervene as IV            # numeric_fmt, prompt_numeric (no-FACS)

CACHE = "/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
N_PER_CLASS = 100
SEEDS = [42, 123, 777, 1234, 2024]


def load_stratum(seed, is_sel):
    """consensus 층(is_sel) 테스트셋 + 공유 exemplar 구성 (IV.load와 동일 방식, 층만 다름)."""
    df = pd.read_parquet(G.AU_PARQUET)
    df = df[df["is_selected"] == is_sel].copy()
    df["emotion_en"] = df["emotion"].map(G.KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    rng = np.random.default_rng(seed)
    ex_idx = [df[df["emotion_en"] == l].iloc[rng.integers(0, len(df[df["emotion_en"] == l]))].name
              for l in G.LABELS]
    ex_df = df.loc[ex_idx]; rem = df.drop(index=ex_idx)
    rng2 = np.random.default_rng(seed)
    pieces = []
    for l in G.LABELS:
        sub = rem[rem["emotion_en"] == l]
        k = min(N_PER_CLASS, len(sub))
        pieces.append(sub.iloc[rng2.choice(len(sub), size=k, replace=False)])
    test = pd.concat(pieces).reset_index(drop=True)
    ex = [(IV.numeric_fmt(r), r["emotion_en"]) for _, r in ex_df.iterrows()]
    items = [(IV.numeric_fmt(r), r["emotion_en"]) for _, r in test.iterrows()]
    return items, ex


def acc(items, exemplars, model, tok, use_icl, bs):
    # run_gating_batched.run_cfg returns (acc, f1)
    a, _f1 = RGB.run_cfg(items, exemplars, IV.prompt_numeric, model, tok, use_icl, bs)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    snap = sorted(glob.glob(f"{CACHE}/snapshots/*"))[0]
    print(f"[load] {snap}", flush=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

    STRATA = {"agreed(is_sel=1)": 1, "rejected(is_sel=0)": 0}
    out = {}
    for sname, sval in STRATA.items():
        z_list, i_list = [], []
        for seed in SEEDS:
            items, ex = load_stratum(seed, sval)
            z = acc(items, ex, model, tok, False, args.bs)
            i = acc(items, ex, model, tok, True, args.bs)
            z_list.append(z); i_list.append(i)
            print(f"  {sname} seed{seed}: zero={z:.3f} icl={i:.3f} gain={(i-z)*100:+.1f}pp", flush=True)
        zm, im = float(np.mean(z_list)), float(np.mean(i_list))
        out[sname] = {"zero": zm, "icl": im, "gain_pp": (im - zm) * 100,
                      "zero_per_seed": z_list, "icl_per_seed": i_list}
        print(f">>> {sname}: zero={zm:.3f} icl={im:.3f} GAIN={(im-zm)*100:+.1f}pp", flush=True)
    (HERE.parent / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== BRIDGE2 DONE ===")
    print(json.dumps({k: {"zero": round(v["zero"],3), "icl": round(v["icl"],3),
                          "gain_pp": round(v["gain_pp"],1)} for k,v in out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
