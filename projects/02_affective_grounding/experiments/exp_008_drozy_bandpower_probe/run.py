"""exp_008 DROZY drowsiness — spectral band-power features (smarter substrate retry).
Classical drowsiness signal (theta/alpha rise) → if THIS also fails subject-wise, Track B dies hard.
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss

HERE = Path(__file__).parent
SESS_DIR = Path("/home/ajy/04_driver_monitoring/03_DMS_sleep_classification/251219/data/sessions")
SEEDS = [42, 123, 777]
N_SPLITS = 5
N_ENSEMBLE = 5
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}


def bandpower(x, fs, lo, hi):
    # x: (T,) window. FFT-based band power.
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    p = (np.abs(np.fft.rfft(x, axis=-1)) ** 2) / n
    m = (f >= lo) & (f < hi)
    return p[..., m].sum(axis=-1)


def features(X, fs=512):
    # X: (W, T, C). Compute log band-power per channel per band → (W, C*B + extras)
    W, T, C = X.shape
    feats = []
    for b, (lo, hi) in BANDS.items():
        bp = np.stack([bandpower(X[:, :, c], fs, lo, hi) for c in range(C)], axis=1)  # (W, C)
        feats.append(np.log1p(bp))
    F = np.concatenate(feats, axis=1)  # (W, C*B)
    # ratios: theta/alpha, (theta+alpha)/beta — classical drowsiness markers per channel
    theta = features._bp(X, fs, *BANDS["theta"])
    alpha = features._bp(X, fs, *BANDS["alpha"])
    beta = features._bp(X, fs, *BANDS["beta"])
    ratio1 = np.log1p(theta / (alpha + 1e-9))
    ratio2 = np.log1p((theta + alpha) / (beta + 1e-9))
    F = np.concatenate([F, ratio1, ratio2], axis=1)
    return F


def _bp(X, fs, lo, hi):
    return np.stack([bandpower(X[:, :, c], fs, lo, hi) for c in range(X.shape[2])], axis=1)
features._bp = _bp


def load_all():
    files = sorted(glob.glob(str(SESS_DIR / "*.npz")))
    Xs, ys, sids, sessids = [], [], [], []
    for f in files:
        d = np.load(f); X = d["X"]; y = int(d["y"]); sess = str(d["session"])
        subj = int(re.match(r"(\d+)-", sess).group(1)); n = X.shape[0]
        Xs.append(X); ys.append(np.full(n, y, dtype=np.int64))
        sids.append(np.full(n, subj, dtype=np.int64))
        sessids.append(np.full(n, len(Xs) - 1, dtype=np.int64))
    return (np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys),
            np.concatenate(sids), np.concatenate(sessids))


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
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for ep in range(80):
            opt.zero_grad(); loss = lossf(net(Xtr_t), ytr_t); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(torch.softmax(net(Xte_t), 1).cpu().numpy())
    return np.mean(preds, axis=0)


def ece(probs, y, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0: e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def run_seed_target(F, y, subj, n_cls, seed, method="mlp_ens"):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), n_cls))
    for tr, te in gkf.split(F, y, subj):
        sc = StandardScaler().fit(F[tr]); Xtr = sc.transform(F[tr]); Xte = sc.transform(F[te])
        if method == "logreg":
            m = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial",
                                   class_weight="balanced", random_state=seed)
            m.fit(Xtr, y[tr]); p = m.predict_proba(Xte)
        else:
            p = train_ensemble_predict(Xtr, y[tr], Xte, n_cls, seed)
        oof[te] = p
    pred = oof.argmax(1); ent = -(oof * np.log(oof + 1e-12)).sum(1)
    mis = (pred != y).astype(int)
    auroc = float(roc_auc_score(mis, ent)) if mis.sum() not in (0, len(mis)) else None
    return {"oof": oof, "pred": pred, "ent": ent, "mis": mis,
            "bal_acc": float(balanced_accuracy_score(y, pred)),
            "ece": ece(oof, y), "nll": float(log_loss(y, oof, labels=list(range(n_cls)))),
            "err_unc_auroc": auroc}


def main():
    X, y3, subj, sess_idx = load_all()
    n_cls3 = int(y3.max()) + 1
    print(f"loaded {len(X)} windows, {len(set(subj))} subjects, classes={sorted(set(y3))}")
    print("extracting band-power features ...")
    F = features(X)
    print(f"features shape: {F.shape}")

    # also binary: alert (0) vs drowsy (1 or 2)
    y_bin = (y3 > 0).astype(np.int64)

    results = {"exp_id": "exp_008_drozy_bandpower_probe", "n_windows": int(len(y3)),
               "n_subjects": int(len(set(subj))), "n_features": int(F.shape[1])}

    for tname, y, n_cls, chance in [("3class", y3, 3, 1/3), ("binary", y_bin, 2, 0.5)]:
        per_method = {}
        for method in ["logreg", "mlp_ens"]:
            per = [run_seed_target(F, y, subj, n_cls, s, method=method) for s in SEEDS]
            aurocs = [p["err_unc_auroc"] for p in per if p["err_unc_auroc"] is not None]
            bals = [p["bal_acc"] for p in per]
            # bootstrap CI on seed-0
            p0 = per[0]; rng = np.random.default_rng(0); n = len(y); B = 2000
            boot_auroc = []
            for _ in range(B):
                idx = rng.integers(0, n, n); m = p0["mis"][idx]
                if m.sum() not in (0, len(m)):
                    boot_auroc.append(roc_auc_score(m, p0["ent"][idx]))
            ci = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
            # session-level voting (using seed-0)
            sess_true, sess_pred = [], []
            for sid in np.unique(sess_idx):
                m = (sess_idx == sid); sess_pred.append(p0["oof"][m].mean(0).argmax()); sess_true.append(int(y[m][0]))
            sess_bal = float(balanced_accuracy_score(sess_true, sess_pred))
            # risk-coverage
            order = np.argsort(p0["ent"])
            def sel(cov):
                k = max(2, int(cov * n)); idx = order[:k]
                return float(balanced_accuracy_score(y[idx], p0["pred"][idx]))
            per_method[method] = {
                "bal_acc_per_seed": bals, "bal_acc_mean": float(np.mean(bals)),
                "session_bal_acc": sess_bal,
                "err_unc_auroc_per_seed": [p["err_unc_auroc"] for p in per],
                "err_unc_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
                "err_unc_auroc_ci95": ci,
                "ece_mean": float(np.mean([p["ece"] for p in per])),
                "risk_coverage": {"full": float(np.mean(bals)), "cov80": sel(0.8), "cov50": sel(0.5)},
                "supported": bool(aurocs and np.mean(aurocs) >= 0.60 and ci[0] > 0.5
                                  and np.mean(bals) > chance + 0.05),
            }
        results[tname] = {"chance": chance, "methods": per_method}

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print()
    for tname, blk in [(k, results[k]) for k in ["3class", "binary"]]:
        print(f"=== {tname} (chance {blk['chance']:.3f}) ===")
        for method, r in blk["methods"].items():
            print(f"  {method:8} win_balacc={r['bal_acc_mean']:.3f} sess_balacc={r['session_bal_acc']:.3f}  "
                  f"err-unc AUROC={r['err_unc_auroc_mean']} CI{r['err_unc_auroc_ci95']}  "
                  f"risk-cov:{r['risk_coverage']}  supported={r['supported']}")
    # decisive call: any (target, method) pass with bal_acc > chance + 0.05 AND AUROC >= 0.60
    any_pass = any(results[t]["methods"][m]["supported"] for t in ["3class", "binary"] for m in ["logreg", "mlp_ens"])
    results["verdict"] = "any_substrate_alive" if any_pass else "all_dead"
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nVERDICT = {results['verdict']}")


if __name__ == "__main__":
    main()
