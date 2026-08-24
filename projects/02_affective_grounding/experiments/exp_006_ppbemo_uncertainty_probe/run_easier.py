"""exp_006 follow-up: easier binary targets (valence/arousal median split).
Reuses run.py pipeline. Decides if Track B is dead on PPB-Emo or just on 7-class.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import (feature_cols, ece, brier_multi, train_ensemble_predict, CSV, SEEDS, N_SPLITS)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def run_target(X, y, groups, name, seed):
    n_cls = 2
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), n_cls))
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr])); Xte = sc.transform(imp.transform(X[te]))
        oof[te] = train_ensemble_predict(Xtr, y[tr], Xte, n_cls, seed)
    pred = oof.argmax(1); ent = -(oof * np.log(oof + 1e-12)).sum(1)
    mis = (pred != y).astype(int)
    auroc = float(roc_auc_score(mis, ent)) if mis.sum() not in (0, len(mis)) else None
    return {
        "target": name, "seed": seed, "n_pos": int(y.sum()), "n_neg": int((1-y).sum()),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, oof[:, 1])),
        "ece": ece(oof, y), "nll": float(log_loss(y, oof, labels=[0, 1])),
        "err_unc_auroc": auroc,
        "oof_pos": oof[:, 1].tolist(), "pred": pred.tolist(), "ent": ent.tolist(), "mis": mis.tolist(), "y": y.tolist(),
    }


def main():
    df = pd.read_csv(CSV)
    fcols = feature_cols(df)
    X = df[fcols].apply(pd.to_numeric, errors="coerce").values
    groups = df["participant"].values
    n = len(df)

    targets = {
        "valence_bin": (df["valence"] > df["valence"].median()).astype(int).values,
        "arousal_bin": (df["arousal"] > df["arousal"].median()).astype(int).values,
    }
    print(f"X={X.shape} n={n} participants={df.participant.nunique()} chance=0.500")
    print(f"valence_bin n_pos={(targets['valence_bin']==1).sum()}, arousal_bin n_pos={(targets['arousal_bin']==1).sum()}")

    out = {}
    for tname, y in targets.items():
        per = [run_target(X, y, groups, tname, s) for s in SEEDS]
        # bootstrap CI on seed-0
        p0 = per[0]; rng = np.random.default_rng(0); B = 2000
        ent0 = np.array(p0["ent"]); mis0 = np.array(p0["mis"]); pred0 = np.array(p0["pred"]); y0 = np.array(p0["y"])
        boot = []
        for _ in range(B):
            idx = rng.integers(0, n, n); mb = mis0[idx]
            if mb.sum() not in (0, len(mb)):
                boot.append(roc_auc_score(mb, ent0[idx]))
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        aucs = [p["err_unc_auroc"] for p in per if p["err_unc_auroc"] is not None]
        bals = [p["bal_acc"] for p in per]
        aucm = float(np.mean(aucs)) if aucs else None
        # risk-coverage on seed-0
        order = np.argsort(ent0)
        def sel(cov):
            k = max(2, int(cov * n)); idx = order[:k]
            return float(balanced_accuracy_score(y0[idx], pred0[idx]))
        out[tname] = {
            "bal_acc_per_seed": bals, "bal_acc_mean": float(np.mean(bals)),
            "err_unc_auroc_per_seed": [p["err_unc_auroc"] for p in per],
            "err_unc_auroc_mean": aucm, "err_unc_auroc_ci95": ci,
            "ece_mean": float(np.mean([p["ece"] for p in per])),
            "auc_mean": float(np.mean([p["auc"] for p in per])),
            "risk_coverage": {"full": float(np.mean(bals)), "cov80": sel(0.8), "cov50": sel(0.5)},
            "supported_err_unc": bool(aucm is not None and aucm >= 0.60 and ci[0] > 0.5),
        }

    out["verdict"] = {
        "valence_bin_uncertainty_informative": out["valence_bin"]["supported_err_unc"],
        "arousal_bin_uncertainty_informative": out["arousal_bin"]["supported_err_unc"],
    }
    (HERE / "results_easier.json").write_text(json.dumps(out, indent=2, default=float))

    for t in ["valence_bin", "arousal_bin"]:
        r = out[t]
        print(f"\n=== {t} (chance 0.5) ===")
        print(f"  bal_acc seeds={r['bal_acc_per_seed']} mean={r['bal_acc_mean']:.3f}  AUC={r['auc_mean']:.3f}")
        print(f"  err-vs-unc AUROC: seeds={r['err_unc_auroc_per_seed']} mean={r['err_unc_auroc_mean']} CI={r['err_unc_auroc_ci95']}")
        print(f"  risk-cov bal_acc: full={r['risk_coverage']['full']:.3f} cov80={r['risk_coverage']['cov80']:.3f} cov50={r['risk_coverage']['cov50']:.3f}")
        print(f"  ECE={r['ece_mean']:.3f}  supported(unc informative)={r['supported_err_unc']}")


if __name__ == "__main__":
    main()
