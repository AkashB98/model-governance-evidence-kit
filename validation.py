"""Data-quality gate: reports issues, never raises on bad data."""

from __future__ import annotations

from typing import Dict, List

from generator import FIELDS, WINDOWS

_RANGES = {"dti_ratio": (0.0, 1.0), "income_volatility": (0.0, 1.0),
           "credit_utilization": (0.0, 1.0), "score": (0.0, 1.0)}


def validate_records(records: List[Dict]) -> List[str]:
    issues: List[str] = []
    if not isinstance(records, list):
        return ["records must be a list"]
    seen = set()
    for i, r in enumerate(records):
        rid = r.get("record_id", f"#{i}")
        if not isinstance(r, dict):
            issues.append(f"{rid}: record is not an object")
            continue
        for field in FIELDS:
            if field not in r:
                issues.append(f"{rid}: missing field {field}")
        if r.get("record_id") in seen:
            issues.append(f"{rid}: duplicate record_id")
        seen.add(r.get("record_id"))
        if "window" in r and r["window"] not in WINDOWS:
            issues.append(f"{rid}: window must be one of {WINDOWS}")
        for field, (lo, hi) in _RANGES.items():
            v = r.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                if field in r:
                    issues.append(f"{rid}: {field} must be a number")
            elif not (lo <= v <= hi):
                issues.append(f"{rid}: {field}={v} outside [{lo}, {hi}]")
        label = r.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            if "label" in r:
                issues.append(f"{rid}: label must be 0 or 1 (not bool)")
    return issues
