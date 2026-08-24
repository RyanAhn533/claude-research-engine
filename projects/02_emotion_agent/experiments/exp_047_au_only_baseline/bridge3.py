"""
Bridge 3 — AU-only classifier baseline (합본 보조).
목적: AU 신호 자체가 감정정보를 담지만, 한국 부정감정에서 구조적 ambiguity 천장이 남음을 보임.
      → LLM AU-as-text와 같은 split에서 비교 가능 + B의 "AU cosine 0.97-0.99" 천장 주장 보강.
방법: AU parquet의 41 AU 컬럼 = feature, 4-class 감정. LogReg, stratified split.
      consensus 층(is_selected)별 + emotion별 test acc 분해.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PQ = "/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet"
KR_TO_EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
HERE = Path(__file__).resolve()

df = pd.read_parquet(PQ)
au_cols = [c for c in df.columns if c.startswith("AU")]
df["y"] = df["emotion"].map(KR_TO_EN)
df = df[df["y"].notna()].copy()
X = df[au_cols].fillna(0).values
y = df["y"].values
sel = df["is_selected"].values

Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
    X, y, sel, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial").fit(sc.transform(Xtr), ytr)
pred = clf.predict(sc.transform(Xte))

out = {"n_features": len(au_cols), "n_train": len(ytr), "n_test": len(yte),
       "overall_acc": float(accuracy_score(yte, pred)),
       "macro_f1": float(f1_score(yte, pred, average="macro"))}
# per-emotion acc
out["per_emotion_acc"] = {}
for e in ["happy", "angry", "sad", "neutral"]:
    m = yte == e
    out["per_emotion_acc"][e] = float(accuracy_score(yte[m], pred[m])) if m.sum() else None
# per consensus stratum
out["per_stratum_acc"] = {}
for s, nm in [(1, "agreed"), (0, "rejected")]:
    m = str_te == s
    out["per_stratum_acc"][nm] = {"n": int(m.sum()),
                                  "acc": float(accuracy_score(yte[m], pred[m])) if m.sum() else None}

(HERE.parent / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("=== BRIDGE3 (AU-only LogReg) DONE ===")
print(json.dumps(out, indent=2, ensure_ascii=False))
