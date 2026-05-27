#!/usr/bin/env python3
"""
schema_freeze 검증.
1. 8개 schema 자체가 draft-07 valid한지
2. 각 fixture가 해당 schema를 통과하는지
3. negative test (invalid fixture가 reject되는지)
"""
import json
import sys
from pathlib import Path
from jsonschema import Draft7Validator, ValidationError

SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
FIX_DIR = Path(__file__).parent / "fixtures"

SCHEMAS = [
    "state",
    "task",
    "experiment",
    "hypothesis",
    "reproducibility_manifest",
    "paper_tried_entry",
    "claim",
    "permission",
]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    failures = []
    print(f"== schema self-validation ({len(SCHEMAS)} schemas) ==")
    for name in SCHEMAS:
        sp = SCHEMA_DIR / f"{name}.schema.json"
        schema = load(sp)
        try:
            Draft7Validator.check_schema(schema)
            print(f"  [ok ]  {name}.schema.json")
        except Exception as e:
            print(f"  [FAIL] {name}.schema.json: {e}")
            failures.append((name, "self", str(e)))

    print(f"\n== fixture validation ({len(SCHEMAS)} positive cases) ==")
    for name in SCHEMAS:
        schema = load(SCHEMA_DIR / f"{name}.schema.json")
        fixture = load(FIX_DIR / f"{name}.valid.json")
        v = Draft7Validator(schema)
        errs = list(v.iter_errors(fixture))
        if not errs:
            print(f"  [ok ]  {name}.valid.json")
        else:
            for e in errs:
                print(f"  [FAIL] {name}.valid.json: {e.message} at {list(e.absolute_path)}")
                failures.append((name, "valid", e.message))

    print(f"\n== negative test (invalid fixtures must reject) ==")
    for name in SCHEMAS:
        inv_path = FIX_DIR / f"{name}.invalid.json"
        if not inv_path.exists():
            continue
        schema = load(SCHEMA_DIR / f"{name}.schema.json")
        fixture = load(inv_path)
        v = Draft7Validator(schema)
        errs = list(v.iter_errors(fixture))
        if errs:
            print(f"  [ok ]  {name}.invalid.json → rejected ({len(errs)} err)")
        else:
            print(f"  [FAIL] {name}.invalid.json → wrongly ACCEPTED")
            failures.append((name, "invalid", "wrongly accepted"))

    print()
    if failures:
        print(f"!! {len(failures)} failure(s)")
        for f in failures:
            print(f"   {f}")
        sys.exit(1)
    print("ALL PASS — schemas are frozen-ready")


if __name__ == "__main__":
    main()
