import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mk(**kw):
    base = {"record_id": "t1", "window": "current", "dti_ratio": 0.32,
            "income_volatility": 0.22, "credit_utilization": 0.45,
            "score": 0.20, "label": 0}
    base.update(kw)
    return base
