"""
hashing.py — deterministic hashing for engine.

Two uses:
1. config_fingerprint(config_dict) — for paper_tried dedup.
2. row_hash(row_dict) — for append-only hash chain.

Determinism guarantee: sorted keys, UTF-8, json.dumps with separators.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_fingerprint(config: dict) -> str:
    """
    sha256 of canonical config dict. Used by paper_tried.method_globally_blocked dedup.
    Same (method_id, fingerprint) pair = same experiment. Different config = retryable.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config_fingerprint expects dict, got {type(config).__name__}")
    h = hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def row_hash(row: dict, exclude: tuple[str, ...] = ("prev_hash", "row_id")) -> str:
    """
    Hash of a row excluding fields that depend on the chain itself.
    Used to compute the next row's prev_hash.
    """
    clean = {k: v for k, v in row.items() if k not in exclude}
    h = hashlib.sha256(_canonical(clean).encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def data_hash(payload: bytes) -> str:
    """For reproducibility_manifest.data_hash."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
