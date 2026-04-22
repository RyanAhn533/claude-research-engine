"""
Text-only baseline on IEMOCAP + MELD 4-class.
Method: TF-IDF (word+char ngrams) + LogReg
Eval: 3-seed, stratified split (or official MELD split)
"""
import os, json, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

OUT = Path(__file__).parent
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

IEMOCAP_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet")
MELD_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_002_meld_preproc/cache/meld_4class.parquet")

SEED_LIST = [42, 123, 777]
LABEL_NAMES = ["ang", "hap", "neu", "sad"]


def load_dataset(name):
    if name == "IEMOCAP":
        df = pd.read_parquet(IEMOCAP_PATH)
        text = df["transcription"].fillna("").astype(str).values
        y = df["label"].astype(int).values
        split = None
    elif name == "MELD":
        df = pd.read_parquet(MELD_PATH)
        text = df["text"].fillna("").astype(str).values
        y = df["label_4"].astype(int).values
        split = df["split"].values
    return text, y, split


def tfidf_logreg(X_text_tr, y_tr, X_text_te, y_te):
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=50000, sublinear_tf=True, min_df=2,
    )
    Xtr = vec.fit_transform(X_text_tr)
    Xte = vec.transform(X_text_te)
    clf = LogisticRegression(
        max_iter=1000, C=1.0, n_jobs=-1,
        solver="liblinear", multi_class="auto",
    )
    clf.fit(Xtr, y_tr)
    yhat = clf.predict(Xte)
    acc = accuracy_score(y_te, yhat)
    f1 = f1_score(y_te, yhat, average="macro")
    return acc, f1


# ===== IEMOCAP: 3-seed stratified 80/20 =====
print("[IEMOCAP]")
text, y, _ = load_dataset("IEMOCAP")
print(f"  samples: {len(text)}  class dist: {Counter(y.tolist())}")
iem_results = []
for seed in SEED_LIST:
    t0 = time.time()
    X_tr, X_te, y_tr, y_te = train_test_split(
        text, y, test_size=0.2, random_state=seed, stratify=y
    )
    acc, f1 = tfidf_logreg(X_tr, y_tr, X_te, y_te)
    print(f"  seed={seed}  acc={acc*100:.2f}%  macro-F1={f1:.3f}  ({time.time()-t0:.1f}s)")
    iem_results.append({"seed": seed, "acc": float(acc), "f1": float(f1)})

iem_summary = {
    "acc_mean": float(np.mean([r["acc"] for r in iem_results])),
    "acc_std":  float(np.std([r["acc"] for r in iem_results])),
    "f1_mean":  float(np.mean([r["f1"] for r in iem_results])),
    "f1_std":   float(np.std([r["f1"] for r in iem_results])),
}
print(f"  MEAN acc={iem_summary['acc_mean']*100:.2f}% ± {iem_summary['acc_std']*100:.2f}  "
      f"F1={iem_summary['f1_mean']:.3f} ± {iem_summary['f1_std']:.3f}")

# ===== MELD: official train/dev/test split =====
print("\n[MELD]")
text_m, y_m, split_m = load_dataset("MELD")
tr_m = split_m == "train"
dev_m = split_m == "dev"
te_m = split_m == "test"
print(f"  samples: train={tr_m.sum()}  dev={dev_m.sum()}  test={te_m.sum()}")
print(f"  class dist train: {Counter(y_m[tr_m].tolist())}")

# Use train only, eval on test (official protocol)
meld_results = []
for seed in SEED_LIST:
    t0 = time.time()
    # seed only affects LR initialization; TF-IDF is deterministic
    np.random.seed(seed)
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=50000, sublinear_tf=True, min_df=2,
    )
    Xtr = vec.fit_transform(text_m[tr_m])
    Xte = vec.transform(text_m[te_m])
    clf = LogisticRegression(
        max_iter=1000, C=1.0, n_jobs=-1,
        solver="liblinear", random_state=seed,
    )
    clf.fit(Xtr, y_m[tr_m])
    yhat = clf.predict(Xte)
    acc = accuracy_score(y_m[te_m], yhat)
    f1 = f1_score(y_m[te_m], yhat, average="macro")
    print(f"  seed={seed}  acc={acc*100:.2f}%  macro-F1={f1:.3f}  ({time.time()-t0:.1f}s)")
    meld_results.append({"seed": seed, "acc": float(acc), "f1": float(f1)})

meld_summary = {
    "acc_mean": float(np.mean([r["acc"] for r in meld_results])),
    "acc_std":  float(np.std([r["acc"] for r in meld_results])),
    "f1_mean":  float(np.mean([r["f1"] for r in meld_results])),
    "f1_std":   float(np.std([r["f1"] for r in meld_results])),
}
print(f"  MEAN acc={meld_summary['acc_mean']*100:.2f}% ± {meld_summary['acc_std']*100:.2f}  "
      f"F1={meld_summary['f1_mean']:.3f} ± {meld_summary['f1_std']:.3f}")

# ===== Save =====
results = {
    "method": "TF-IDF (word 1-2gram, 50k feat) + LogReg",
    "IEMOCAP_3seed_80_20": {"seeds": iem_results, "summary": iem_summary},
    "MELD_official_split": {"seeds": meld_results, "summary": meld_summary},
    "random_baseline_4class": 0.25,
}
with open(CACHE / "text_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[done] {CACHE}/text_baseline_results.json")
