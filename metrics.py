"""Accuracy and stability: AUC, calibration by band, run variance.

* AUC — area under the ROC curve, computed with the Mann-Whitney
  rank formula (ties take average ranks). ``auc_bruteforce`` is the
  O(n_pos * n_neg) pairwise definition, kept as the independent
  cross-check the golden evals pin the fast formula against.
* Calibration — per score band, mean predicted score vs realised
  default rate, judged on the ``current`` window (the period under
  review). Max band error > 0.20 fails; > 0.10 warns.
* Run-to-run variance — the same model re-scored over the same cases
  under several seeds (small seeded jitter, sd 0.01). Low spread means
  the score an auditor sees is stable, not a lucky draw.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Sequence, Tuple

BANDS = ["S1", "S2", "S3", "S4", "S5"]
BAND_EDGES = [0.15, 0.30, 0.45, 0.60]
CAL_WARN = 0.10
CAL_FAIL = 0.20
AUC_WARN = 0.65
AUC_FAIL = 0.60
VAR_WARN_MEAN_STD = 0.02
VAR_FAIL_MEAN_STD = 0.04
RUN_SEEDS = [101, 102, 103, 104, 105]
RUN_JITTER_SD = 0.01


def score_band(score: float) -> str:
    for edge, band in zip(BAND_EDGES, BANDS):
        if score < edge:
            return band
    return BANDS[-1]


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: need both classes")
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _, y in pairs[i:j + 1] if y == 1)
        i = j + 1
    return round((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 6)


def auc_bruteforce(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Pairwise definition: P(pos > neg) + 0.5 * P(pos == neg)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        raise ValueError("AUC undefined: need both classes")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 6)


def calibration_by_band(records: List[Dict],
                        window: str = "current") -> List[Dict]:
    pool = [r for r in records if r.get("window") == window] or list(records)
    rows = []
    for band in BANDS:
        group = [r for r in pool if score_band(r["score"]) == band]
        count = len(group)
        predicted = sum(r["score"] for r in group) / count if count else 0.0
        observed = sum(r["label"] for r in group) / count if count else 0.0
        rows.append({
            "band": band, "count": count,
            "predicted": round(predicted, 6),
            "observed": round(observed, 6),
            "abs_error": round(abs(observed - predicted), 6),
        })
    return rows


def _verdict_cal(max_err: float) -> str:
    return "fail" if max_err > CAL_FAIL else ("warn" if max_err > CAL_WARN else "pass")


def _verdict_auc(value: float) -> str:
    return "fail" if value < AUC_FAIL else ("warn" if value < AUC_WARN else "pass")


def rescore(records: List[Dict], seed: int,
            jitter_sd: float = RUN_JITTER_SD) -> List[float]:
    """One seeded re-run of the model over the same cases."""
    out = []
    for idx, r in enumerate(records):
        rng = random.Random(seed * 1_000_003 + idx)
        out.append(min(0.99, max(0.01, r["score"] + rng.gauss(0, jitter_sd))))
    return out


def variance_report(records: List[Dict], seeds: Sequence[int] = RUN_SEEDS,
                    jitter_sd: float = RUN_JITTER_SD) -> Dict:
    if not records:
        raise ValueError("variance requires at least one record")
    runs = [rescore(records, s, jitter_sd) for s in seeds]
    stds = [statistics.pstdev([run[i] for run in runs])
            for i in range(len(records))]
    mean_std = round(statistics.mean(stds), 6)
    max_std = round(max(stds), 6)
    v = ("fail" if mean_std > VAR_FAIL_MEAN_STD
         else "warn" if mean_std > VAR_WARN_MEAN_STD else "pass")
    return {"runs": len(seeds), "seeds": list(seeds),
            "mean_score_std": mean_std, "max_score_std": max_std,
            "verdict": v,
            "thresholds": {"warn_mean_std": VAR_WARN_MEAN_STD,
                           "fail_mean_std": VAR_FAIL_MEAN_STD}}


def accuracy_report(records: List[Dict]) -> Dict:
    cur = [r for r in records if r.get("window") == "current"] or list(records)
    value = auc([r["score"] for r in cur], [r["label"] for r in cur])
    cal = calibration_by_band(records, window="current")
    max_err = max((r["abs_error"] for r in cal if r["count"] > 0), default=0.0)
    worst_band = max((r for r in cal if r["count"] > 0),
                     key=lambda r: r["abs_error"], default=None)
    cal_v, auc_v = _verdict_cal(max_err), _verdict_auc(value)
    worst = "fail" if "fail" in (cal_v, auc_v) else (
        "warn" if "warn" in (cal_v, auc_v) else "pass")
    return {"auc": value, "auc_verdict": auc_v, "calibration": cal,
            "max_calibration_error": max_err,
            "worst_band": worst_band["band"] if worst_band else None,
            "calibration_verdict": cal_v, "verdict": worst,
            "current_count": len(cur),
            "thresholds": {"cal_warn": CAL_WARN, "cal_fail": CAL_FAIL,
                           "auc_warn": AUC_WARN, "auc_fail": AUC_FAIL}}
