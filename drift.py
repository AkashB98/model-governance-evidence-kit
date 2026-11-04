"""Drift: PSI (score) and CSI (each feature) vs a reference window.

PSI — population stability index; CSI — characteristic stability
index, the same formula applied to one input feature instead of the
score. For bins b:  PSI = sum_b (cur% - ref%) * ln(cur% / ref%), with
every share floored at ``EPSILON`` so an empty bin cannot divide by
zero. Verdicts: < 0.10 pass, 0.10-0.25 warn, > 0.25 fail.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

EPSILON = 0.0001
WARN_THRESHOLD = 0.10
FAIL_THRESHOLD = 0.25

SCORE_EDGES = [0.10 * i for i in range(1, 10)]
FEATURE_EDGES = {
    "dti_ratio": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    "income_volatility": [0.08, 0.14, 0.20, 0.26, 0.32, 0.38, 0.46],
    "credit_utilization": [0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
}
FEATURES = list(FEATURE_EDGES)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def shares(values: List[float], edges: Sequence[float]) -> List[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[_bin_index(v, edges)] += 1
    total = len(values)
    return [c / total if total else 0.0 for c in counts]


def psi_from_shares(ref: Sequence[float], cur: Sequence[float],
                    epsilon: float = EPSILON) -> float:
    total = 0.0
    for r, c in zip(ref, cur):
        rs, cs = max(r, epsilon), max(c, epsilon)
        total += (cs - rs) * math.log(cs / rs)
    return round(total, 6)


def psi(ref_values: List[float], cur_values: List[float],
        edges: Sequence[float]) -> float:
    return psi_from_shares(shares(ref_values, edges), shares(cur_values, edges))


def verdict(value: float) -> str:
    if value > FAIL_THRESHOLD:
        return "fail"
    if value >= WARN_THRESHOLD:
        return "warn"
    return "pass"


def drift_report(records: List[Dict]) -> Dict:
    ref = [r for r in records if r.get("window") == "reference"]
    cur = [r for r in records if r.get("window") == "current"]
    if not ref or not cur:
        raise ValueError("drift requires both reference and current windows")
    score_psi = psi([r["score"] for r in ref], [r["score"] for r in cur],
                    SCORE_EDGES)
    features = {}
    for feat in FEATURES:
        value = psi([r[feat] for r in ref], [r[feat] for r in cur],
                    FEATURE_EDGES[feat])
        features[feat] = {"csi": value, "verdict": verdict(value)}
    all_values = [score_psi] + [f["csi"] for f in features.values()]
    worst = "pass"
    for v in [verdict(x) for x in all_values]:
        if v == "fail":
            worst = "fail"
            break
        if v == "warn":
            worst = "warn"
    return {
        "score_psi": score_psi,
        "score_verdict": verdict(score_psi),
        "features": features,
        "verdict": worst,
        "reference_count": len(ref),
        "current_count": len(cur),
        "thresholds": {"warn": WARN_THRESHOLD, "fail": FAIL_THRESHOLD,
                       "epsilon": EPSILON},
    }
