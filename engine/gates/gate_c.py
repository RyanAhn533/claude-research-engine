"""
gate_c.py — paired-delta statistical gate.

External review §3 problem 4 fix:
v1 rule was `bootstrap_CI_lower > baseline_mean` — unpaired comparison.
v2 rule is `paired_delta_bootstrap_CI_lower > 0` — proper paired test.

External review §3 problem 4 (post-v2) fix:
The previous `tost_passed` field was *not* a true TOST. It computed
"lower CI > 0 AND |mean delta| ≥ delta_threshold", which is a
**practical-effect threshold** test, NOT an equivalence test (TOST proper).

This version exposes both:
  - `practical_effect_passed`  — the original gate (improvement ≥ Δ)
  - `tost_equivalent`          — true two one-sided t-tests for equivalence
                                  to a band [-Δ, +Δ] (used when claim is
                                  "no meaningful difference", not improvement)

For backward compatibility, `tost_passed` is kept as an alias of
`practical_effect_passed` (the original semantic) and marked deprecated.

Two stages:
- prototype:    n_seeds ≥ 3, mean/std only, directional decision
- paper_ready:  n_seeds ≥ 7, paired CI + Cohen's d + Wilcoxon
                + practical-effect threshold + (optional) true TOST

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
    # Renamed from `tost_passed` (external review fix). Original semantic was
    # "improvement is at least delta_threshold AND CI lower > 0" — a practical
    # effect threshold, NOT a TOST.
    practical_effect_passed: bool | None = None
    # True two one-sided t-tests for equivalence within ±delta_threshold band.
    # Only computed when delta_threshold is provided.
    tost_equivalent: bool | None = None
    tost_equivalent_p_lo: float | None = None
    tost_equivalent_p_hi: float | None = None
    # Deprecated alias — kept for backward-compat with schemas/fixtures
    # that still reference `tost_passed`. Always mirrors `practical_effect_passed`.
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

        # Practical-effect threshold (NOT TOST — see module docstring)
        practical_pass = None
        if delta_threshold is not None and delta_threshold > 0:
            practical_pass = bool(ci_lo > 0 and abs(mean) >= delta_threshold)

        # True TOST: two one-sided t-tests for equivalence within ±delta_threshold
        tost_eq = None
        p_lo_tost = None
        p_hi_tost = None
        if delta_threshold is not None and delta_threshold > 0 and n >= 2:
            sd = float(deltas.std(ddof=1))
            if sd > 0:
                se = sd / math.sqrt(n)
                # H1_lower: mean > -delta_threshold  → t = (mean - (-Δ))/se
                t_lo = (mean - (-delta_threshold)) / se
                # H1_upper: mean < +delta_threshold  → t = (mean - Δ)/se
                t_hi = (mean - delta_threshold) / se
                df = n - 1
                p_lo_tost = float(1 - sps.t.cdf(t_lo, df))    # right tail (mean > -Δ)
                p_hi_tost = float(sps.t.cdf(t_hi, df))        # left tail  (mean < +Δ)
                tost_eq = bool(p_lo_tost < p_threshold and p_hi_tost < p_threshold)

        # Decision rule (improvement claim)
        is_improvement = (
            ci_lo > 0
            and d > cohen_d_threshold
            and (not math.isnan(p) and p < p_threshold)
        )
        if delta_threshold is not None and delta_threshold > 0:
            is_improvement = is_improvement and (practical_pass or abs(mean) >= delta_threshold)

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
            practical_effect_passed=practical_pass,
            tost_equivalent=tost_eq,
            tost_equivalent_p_lo=p_lo_tost,
            tost_equivalent_p_hi=p_hi_tost,
            tost_passed=practical_pass,  # deprecated alias — same value
            reason=(
                f"paired_delta_mean={mean:.3f}, ci_95=[{ci_lo:.3f},{ci_hi:.3f}], "
                f"cohen_d={d:.3f}, wilcoxon_p={p:.4f}, "
                f"practical_effect_passed={practical_pass}, tost_equivalent={tost_eq}"
            ),
        )

    return GateCResult("failed", stage, n, reason=f"unknown stage: {stage}")
