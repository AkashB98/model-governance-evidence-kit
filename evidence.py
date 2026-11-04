"""Evidence pack assembly: the auditor-facing artifact.

The eval harness scores agent outputs per case; this produces periodic
model-level validation evidence for an auditor.

An evidence pack is a pure, deterministic function of its input
records: a fixed timestamp, sorted-keys canonical JSON, and SHA-256
hashes of the inputs mean two runs over the same records produce
byte-identical packs — an auditor can re-run and compare hashes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List

from drift import drift_report
from metrics import accuracy_report, variance_report
from validation import validate_records

MODEL_NAME = "synthetic-credit-score-v1"
FIXED_TIMESTAMP = "2026-11-04T00:00:00Z"
SCHEMA_VERSION = "1.0"


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def input_hash(records: List[Dict]) -> str:
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    return sha256_text(lines)


def _combine(verdicts: List[str]) -> str:
    if "fail" in verdicts:
        return "fail"
    if "warn" in verdicts:
        return "warn"
    return "pass"


def build_evidence(records: List[Dict]) -> Dict:
    issues = validate_records(records)
    if issues:
        raise ValueError(f"invalid records: {issues[:3]}")
    drift = drift_report(records)
    accuracy = accuracy_report(records)
    variance = variance_report(records)
    checks = {
        "drift": {"verdict": drift["verdict"], "detail": drift},
        "accuracy": {"verdict": accuracy["verdict"], "detail": accuracy},
        "variance": {"verdict": variance["verdict"], "detail": variance},
    }
    overall = _combine([c["verdict"] for c in checks.values()])
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "generated_at": FIXED_TIMESTAMP,
        "deterministic": True,
        "record_count": len(records),
        "input_sha256": input_hash(records),
        "checks": checks,
        "verdicts": {k: c["verdict"] for k, c in checks.items()},
        "overall_verdict": overall,
    }


def evidence_json(evidence: Dict) -> str:
    return canonical(evidence) + "\n"


def evidence_hash(evidence: Dict) -> str:
    return sha256_text(evidence_json(evidence))


def render_markdown(evidence: Dict) -> str:
    d = evidence["checks"]["drift"]["detail"]
    a = evidence["checks"]["accuracy"]["detail"]
    v = evidence["checks"]["variance"]["detail"]
    lines = [
        "# Model Validation Evidence Pack",
        "",
        "The eval harness scores agent outputs per case; this produces "
        "periodic model-level validation evidence for an auditor.",
        "",
        f"- **Model:** {evidence['model_name']}",
        f"- **Generated at (fixed for determinism):** {evidence['generated_at']}",
        f"- **Records:** {evidence['record_count']}",
        f"- **Input SHA-256:** `{evidence['input_sha256']}`",
        f"- **Overall verdict: {evidence['overall_verdict'].upper()}**",
        "",
        "## Checks",
        "",
        f"### Drift — {evidence['verdicts']['drift'].upper()}",
        "",
        f"- Score PSI: {d['score_psi']} ({d['score_verdict']})",
    ]
    for feat, row in sorted(d["features"].items()):
        lines.append(f"- {feat} CSI: {row['csi']} ({row['verdict']})")
    lines += [
        f"- Windows: reference n={d['reference_count']}, current n={d['current_count']}",
        "",
        f"### Accuracy — {evidence['verdicts']['accuracy'].upper()}",
        "",
        f"- AUC (current window): {a['auc']} ({a['auc_verdict']})",
        f"- Max calibration error: {a['max_calibration_error']} "
        f"(worst band {a['worst_band']}, {a['calibration_verdict']})",
        "",
        "| Band | n | Predicted | Observed | Abs. error |",
        "|---|---|---|---|---|",
    ]
    for row in a["calibration"]:
        lines.append(f"| {row['band']} | {row['count']} | {row['predicted']} "
                     f"| {row['observed']} | {row['abs_error']} |")
    lines += [
        "",
        f"### Run-to-run variance — {evidence['verdicts']['variance'].upper()}",
        "",
        f"- Runs: {v['runs']} (seeds {v['seeds']})",
        f"- Mean score std: {v['mean_score_std']}; max score std: {v['max_score_std']}",
        "",
    ]
    return "\n".join(lines)
