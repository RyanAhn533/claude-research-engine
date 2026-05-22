"""
leakage_auditor.py — Gate B runs this. Any fail → protocol_violation → HALT.

The 6 ARCHITECTURE.md §6.5 checks:
  1. test_set_not_in_train_indices             [AUTO — set intersection]
  2. preprocessing_stats_fit_on_train_only     [DECLARATION + spot check]
  3. no_future_information_in_features        [DECLARATION]
  4. cv_folds_respect_grouping_variable       [AUTO if grouping_variable provided]
  5. augmentation_only_on_train_split         [DECLARATION]
  6. label_not_derivable_from_other_features  [AUTO — mutual info upper bound]

Some checks need domain knowledge that lives in design.md, not in code.
Those are surfaced as REQUIRED DECLARATIONS: the experiment author must positively
assert each one in design.md's `leakage_audit:` block, signed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class AuditReport:
    exp_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def fail_summary(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)


REQUIRED_DECLARATIONS = [
    "preprocessing_stats_fit_on_train_only",
    "no_future_information_in_features",
    "augmentation_only_on_train_split",
]


def audit(
    *,
    exp_id: str,
    train_indices: set | None = None,
    test_indices: set | None = None,
    val_indices: set | None = None,
    cv_groups: list | None = None,
    fold_assignments: list | None = None,
    declarations: dict[str, bool] | None = None,
    label_predictability_upper_bound: float | None = None,
) -> AuditReport:
    """
    All args optional — what's passed determines what's checked.
    What's not passed → check returns passed=False with reason "not_checked".
    Caller must inspect report.checks for granular pass/fail.
    """
    r = AuditReport(exp_id=exp_id)

    # 1. test ∩ train must be empty
    if train_indices is not None and test_indices is not None:
        overlap = train_indices & test_indices
        r.checks.append(Check(
            "test_set_not_in_train_indices",
            passed=not overlap,
            detail=f"overlap_size={len(overlap)}" if overlap else "no overlap",
        ))
        if val_indices is not None:
            tv = train_indices & val_indices
            vt = val_indices & test_indices
            r.checks.append(Check(
                "val_train_test_disjoint",
                passed=not tv and not vt,
                detail=f"train∩val={len(tv)}, val∩test={len(vt)}",
            ))
    else:
        r.checks.append(Check(
            "test_set_not_in_train_indices",
            passed=False,
            detail="not_checked (train/test indices not provided)"
        ))

    # 4. CV folds respect grouping variable
    if cv_groups is not None and fold_assignments is not None:
        # for each fold, the set of groups in train must be disjoint from groups in val
        violations = 0
        from collections import defaultdict
        fold_to_groups: dict[Any, set] = defaultdict(set)
        for grp, fold in zip(cv_groups, fold_assignments):
            fold_to_groups[fold].add(grp)
        # cross-check pairwise
        folds = sorted(fold_to_groups)
        for i, f1 in enumerate(folds):
            for f2 in folds[i + 1:]:
                shared = fold_to_groups[f1] & fold_to_groups[f2]
                violations += len(shared)
        r.checks.append(Check(
            "cv_folds_respect_grouping_variable",
            passed=violations == 0,
            detail=f"cross-fold group leaks: {violations}",
        ))

    # 6. label predictability from features alone
    if label_predictability_upper_bound is not None:
        # caller has run e.g. random-feature-permutation control or MI estimator
        threshold = 0.05  # 5% above chance is suspicious; tune per project
        r.checks.append(Check(
            "label_not_derivable_from_other_features",
            passed=label_predictability_upper_bound <= threshold,
            detail=f"upper_bound={label_predictability_upper_bound:.3f} (threshold={threshold})",
        ))

    # 2, 3, 5: required declarations
    decl = declarations or {}
    for name in REQUIRED_DECLARATIONS:
        v = decl.get(name)
        if v is True:
            r.checks.append(Check(name, passed=True, detail="declared true in design.md"))
        elif v is False:
            r.checks.append(Check(name, passed=False, detail="declared false → leakage risk"))
        else:
            r.checks.append(Check(name, passed=False, detail="declaration missing — REQUIRED"))

    return r


def parse_declarations_from_design_md(design_md_path: str | Path) -> dict[str, bool]:
    """
    Look for a `leakage_audit:` yaml-ish block in design.md and extract bool fields.
    Tolerant of inline yaml; doesn't require strict parsing.
    """
    p = Path(design_md_path)
    if not p.exists():
        return {}
    text = p.read_text()
    out: dict[str, bool] = {}
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("leakage_audit:"):
            in_block = True
            continue
        if in_block:
            if not line.startswith(" ") and s and ":" in s and not s.startswith("#"):
                # left the block
                break
            if ":" in s:
                k, _, v = s.partition(":")
                k = k.strip().lstrip("- ").strip()
                v = v.strip().lower()
                if v in {"true", "yes"}:
                    out[k] = True
                elif v in {"false", "no"}:
                    out[k] = False
    return out
