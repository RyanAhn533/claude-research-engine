"""
gate_c.py — paired-delta statistical gate.

External review §3 problem 4 fix:
v1 rule was `bootstrap_CI_lower > baseline_mean` — unpaired comparison.
v2 rule is `paired_delta_bootstrap_CI_lower > 0` — proper paired test.

Two stages:
- prototype:    n_seeds ≥ 3, mean/std only, directional decision
- paper_ready:  n_seeds ≥ 7, paired CI + Cohen's d + Wilcoxon + TOST

scipy is required for paper_ready. prototype works on numpy alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Literal, Sequence

import numpy as np


Verdict = Literal["improvement_candidate", "neutral", "failed", "pending"]


@dataclass
class GateCResult:
    verdict: Verdict
    stage: str
    n_seeds: int
    paired_delta_mean: float | None = None
    paired_delta_ci_95: tuple[float, float] | None = None
    cohen_d_paired: float | None = None
    wilcoxon_p: float | None = None
    tost_passed: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("paired_delta_ci_95") is not None:
            d["paired_delta_ci_95"] = list(d["paired_delta_ci_95"])
        return d


def _bootstrap_ci_mean(
    samples: np.ndarray, n: int = 10_000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_samples = len(samples)
    if n_samples == 0:
        return (math.nan, math.nan)
    idx = rng.integers(0, n_samples, size=(n, n_samples))
    boot = samples[idx].mean(axis=1)
    alpha = 1 - ci
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _cohens_d_paired(deltas: np.ndarray) -> float:
    sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    if sd == 0:
        return 0.0
    return float(deltas.mean() / sd)


def evaluate(
    *,
    new_results: Sequence[float],
    baseline_results: Sequence[float],
    stage: Literal["prototype", "paper_ready"],
    delta_threshold: float | None = None,
    cohen_d_threshold: float = 0.5,
    p_threshold: float = 0.05,
    higher_is_better: bool = True,
) -> GateCResult:
    """
    new_results and baseline_results MUST be paired (same seed → same index).
    """
    if len(new_results) != len(baseline_results):
        raise ValueError(
            f"paired comparison requires equal lengths: new={len(new_results)} baseline={len(baseline_results)}"
        )

    n = len(new_results)
    arr_new = np.asarray(new_results, dtype=float)
    arr_base = np.asarray(baseline_results, dtype=float)
    deltas = arr_new - arr_base
    if not higher_is_better:
        deltas = -deltas  # so "improvement" always means positive delta

    # ── prototype stage ──
    if stage == "prototype":
        if n < 3:
            return GateCResult("failed", stage, n, reason=f"n_seeds={n} < 3 (prototype minimum)")
        mean = float(deltas.mean())
        if mean > 0:
            return GateCResult("improvement_candidate", stage, n,
                               paired_delta_mean=mean,
                               reason=f"directional only: mean delta={mean:.3f}")
        return GateCResult("neutral", stage, n,
                           paired_delta_mean=mean,
                           reason=f"directional only: mean delta={mean:.3f}")

    # ── paper_ready stage ──
    if stage == "paper_ready":
        if n < 7:
            return GateCResult("failed", stage, n, reason=f"n_seeds={n} < 7 (paper_ready minimum)")

        try:
            from scipy import stats as sps
        except ImportError:
            return GateCResult("failed", stage, n, reason="scipy required for paper_ready; pip install scipy")

        mean = float(deltas.mean())
        ci_lo, ci_hi = _bootstrap_ci_mean(deltas)
        d = _cohens_d_paired(deltas)

        # Wilcoxon signed-rank
        try:
            wstat, p = sps.wilcoxon(arr_new, arr_base)
            p = float(p)
        except Exception as e:
            p = math.nan

        # TOST against pre-registered delta_threshold (equivalence margin)
        tost_pass = None
        if delta_threshold is not None and delta_threshold > 0:
            # Symmetric TOST: reject if both one-sided tests at α=0.05 reject
            # We adapt: claim improvement iff lower CI > 0 AND |mean delta| > delta_threshold
            # (delta_threshold is the smallest effect of practical interest)
            tost_pass = bool(ci_lo > 0 and abs(mean) >= delta_threshold)

        # Decision rule (external review §3 problem 4 fix)
        is_improvement = (
            ci_lo > 0
            and d > cohen_d_threshold
            and (not math.isnan(p) and p < p_threshold)
        )
        if delta_threshold is not None and delta_threshold > 0:
            is_improvement = is_improvement and (tost_pass or abs(mean) >= delta_threshold)

        verdict: Verdict = "improvement_candidate" if is_improvement else "neutral"
        # explicit "failed" only if statistically worse
        if ci_hi < 0:
            verdict = "failed"

        return GateCResult(
            verdict, stage, n,
            paired_delta_mean=mean,
            paired_delta_ci_95=(ci_lo, ci_hi),
            cohen_d_paired=d,
            wilcoxon_p=p if not math.isnan(p) else None,
            tost_passed=tost_pass,
            reason=(
                f"paired_delta_mean={mean:.3f}, ci_95=[{ci_lo:.3f},{ci_hi:.3f}], "
                f"cohen_d={d:.3f}, wilcoxon_p={p:.4f}"
            ),
        )

    return GateCResult("failed", stage, n, reason=f"unknown stage: {stage}")
