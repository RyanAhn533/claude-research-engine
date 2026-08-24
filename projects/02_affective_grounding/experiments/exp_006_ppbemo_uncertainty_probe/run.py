"""exp_006 PPB-Emo driver-state uncertainty probe (Track B survival gate).
Subject-wise GroupKFold + deep-ensemble MLP -> OOF predictive uncertainty.
Decisive metric: error-vs-uncertainty AUROC (does uncertainty predict errors?).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss

HERE = Path(__file__).parent
CSV = Path("/home/ajy/04_driver_monitoring/02_DMS_PPBEMO_driving_behavior/PPB-EMO_codes/ppb-emo/features_all.csv")
SEEDS = [42, 123, 777]
N_SPLITS = 5
N_ENSEMBLE = 5


def feature_cols(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("eeg_") or cl.startswith("dbd_") or any(
            k in cl for k in ["delta", "theta", "alpha", "beta", "gamma"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier_multi(probs, y, n_cls):
    oh = np.eye(n_cls)[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def train_ensemble_predict(Xtr, ytr, Xte, n_cls, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    preds = []
    for m in range(N_ENSEMBLE):
        torch.manual_seed(seed * 100 + m)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(64, n_cls)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for ep in range(120):
            opt.zero_grad(); out = net(Xtr_t); loss = lossf(out, ytr_t)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(torch.softmax(net(Xte_t), 1).cpu().numpy())
    return np.mean(preds, axis=0)  # ensemble mean softmax


def run_seed(X, y, groups, n_cls, seed):
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
        "oof": oof, "pred": pred, "ent": ent, "mis": mis,
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "ece": ece(oof, y), "nll": float(log_loss(y, oof, labels=list(range(n_cls)))),
        "brier": brier_multi(oof, y, n_cls), "err_unc_auroc": auroc,
    }


def main():
    df = pd.read_csv(CSV)
    fcols = feature_cols(df)
    y_raw = df["category"].astype("category")
    y = y_raw.cat.codes.values; n_cls = len(y_raw.cat.categories)
    X = df[fcols].apply(pd.to_numeric, errors="coerce").values
    groups = df["participant"].values
    maj = pd.Series(y).value_counts(normalize=True).max()
    print(f"X={X.shape} classes={n_cls} groups={df.participant.nunique()} chance={1/n_cls:.3f} majority={maj:.3f}")

    per = [run_seed(X, y, groups, n_cls, s) for s in SEEDS]
    aurocs = [p["err_unc_auroc"] for p in per if p["err_unc_auroc"] is not None]
    bal = [p["bal_acc"] for p in per]

    # bootstrap CI on pooled seed-0 OOF (representative)
    rng = np.random.default_rng(0); p0 = per[0]; n = len(y); B = 2000
    boot_auroc, boot_bal = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m = p0["mis"][idx]
        if m.sum() not in (0, len(m)):
            boot_auroc.append(roc_auc_score(m, p0["ent"][idx]))
        boot_bal.append(balanced_accuracy_score(y[idx], p0["pred"][idx]))
    auroc_ci = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
    bal_ci = [float(np.percentile(boot_bal, 2.5)), float(np.percentile(boot_bal, 97.5))]

    # risk-coverage on seed-0
    order = np.argsort(p0["ent"])  # low uncertainty first
    def sel_bacc(cov):
        k = max(2, int(cov * n)); idx = order[:k]
        return float(balanced_accuracy_score(y[idx], p0["pred"][idx]))
    rc = {"full": p0["bal_acc"], "cov80": sel_bacc(0.8), "cov50": sel_bacc(0.5)}

    auroc_mean = float(np.mean(aurocs)) if aurocs else None
    THRESH = 0.60
    supported = bool(auroc_mean is not None and auroc_mean >= THRESH and auroc_ci[0] > 0.50)

    res = {
        "exp_id": "exp_006_ppbemo_uncertainty_probe", "n": n, "n_classes": n_cls,
        "n_features": len(fcols), "n_participants": int(df.participant.nunique()),
        "chance": 1 / n_cls, "majority": float(maj), "seeds": SEEDS,
        "bal_acc_per_seed": bal, "bal_acc_mean": float(np.mean(bal)), "bal_acc_ci95": bal_ci,
        "err_unc_auroc_per_seed": [p["err_unc_auroc"] for p in per],
        "err_unc_auroc_mean": auroc_mean, "err_unc_auroc_ci95": auroc_ci,
        "ece_mean": float(np.mean([p["ece"] for p in per])),
        "nll_mean": float(np.mean([p["nll"] for p in per])),
        "brier_mean": float(np.mean([p["brier"] for p in per])),
        "risk_coverage_balacc": rc, "threshold": THRESH,
        "supported": supported,
        "verdict": "uncertainty_informative" if supported else "uncertainty_uninformative",
    }
    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nbal_acc={res['bal_acc_mean']:.3f} (chance {res['chance']:.3f}, majority {maj:.3f}) CI{bal_ci}")
    print(f"ECE={res['ece_mean']:.3f} NLL={res['nll_mean']:.3f} Brier={res['brier_mean']:.3f}")
    print(f"*** err-vs-unc AUROC = {auroc_mean} CI{auroc_ci} (>=0.60 & CI_lo>0.5 => informative) ***")
    print(f"risk-coverage bal_acc: full={rc['full']:.3f} cov80={rc['cov80']:.3f} cov50={rc['cov50']:.3f}")
    print(f"VERDICT = {res['verdict']}")


if __name__ == "__main__":
    main()
