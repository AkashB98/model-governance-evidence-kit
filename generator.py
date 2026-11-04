"""Deterministic synthetic scored-portfolio generator.

SYNTHETIC DATA ONLY. No real borrower data is ever used.

The eval harness scores agent outputs per case; this produces periodic
model-level validation evidence for an auditor.

Each record is one scored case: features, a model ``score`` (predicted
probability of default), and the realised ``label`` (1 = defaulted).
Records come in two windows — ``reference`` (the baseline an auditor
approved) and ``current`` (the period under review).

Two defects are planted in the default (seeded) portfolio so the
validation checks have something real to find:

* **Drift:** in the ``current`` window the ``dti_ratio`` distribution
  shifts up (+0.18 in mean) and scores shift with it (+0.06). PSI/CSI
  against the reference window must flag this.
* **Miscalibration:** in the ``current`` window, cases whose score
  falls in band S3 (0.30-0.45) default at ~85% instead of ~37%.
  Calibration by score band must catch that band.

Pass ``drift=False`` / ``miscalibrated=False`` for a clean control
portfolio: same generator, no planted defects.
"""

from __future__ import annotations

import random
from typing import Dict, List

WINDOWS = ["reference", "current"]
FIELDS = ["record_id", "window", "dti_ratio", "income_volatility",
          "credit_utilization", "score", "label"]

DEFAULT_SEED = 42
DEFAULT_PER_WINDOW = 800
DRIFT_DTI_SHIFT = 0.18
DRIFT_SCORE_SHIFT = 0.06
BAD_BAND = (0.30, 0.45)
BAD_BAND_TRUE_RATE = 0.85


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def generate_portfolio(per_window: int = DEFAULT_PER_WINDOW,
                       seed: int = DEFAULT_SEED,
                       drift: bool = True,
                       miscalibrated: bool = True) -> List[Dict]:
    """Return synthetic scored records, deterministic for ``seed``."""
    if per_window < 0:
        raise ValueError("per_window must be >= 0")
    rng = random.Random(seed)
    records: List[Dict] = []
    seq = 0
    for window in WINDOWS:
        is_current = window == "current"
        for _ in range(per_window):
            seq += 1
            dti_mean = 0.32 + (DRIFT_DTI_SHIFT if (is_current and drift) else 0.0)
            dti = round(_clamp(rng.gauss(dti_mean, 0.10), 0.05, 0.92), 4)
            vol = round(_clamp(rng.gauss(0.22, 0.08), 0.02, 0.70), 4)
            util = round(_clamp(rng.gauss(0.45, 0.15), 0.01, 0.99), 4)

            propensity = (0.08 + (dti - 0.32) * 0.9
                          + (vol - 0.22) * 0.6 + (util - 0.45) * 0.4)
            propensity = _clamp(propensity, 0.02, 0.92)

            score_raw = propensity + rng.gauss(0, 0.05)
            if is_current and drift:
                score_raw += DRIFT_SCORE_SHIFT
            score = round(_clamp(score_raw, 0.01, 0.99), 4)

            true_p = propensity
            if (miscalibrated and is_current
                    and BAD_BAND[0] <= score < BAD_BAND[1]):
                true_p = BAD_BAND_TRUE_RATE
            label = 1 if rng.random() < true_p else 0

            records.append({
                "record_id": f"MG-{seq:05d}",
                "window": window,
                "dti_ratio": dti,
                "income_volatility": vol,
                "credit_utilization": util,
                "score": score,
                "label": label,
            })
    return records


def records_to_jsonl(records: List[Dict]) -> str:
    import json
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)


def records_to_csv(records: List[Dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    for r in records:
        writer.writerow({k: r[k] for k in FIELDS})
    return buf.getvalue()
