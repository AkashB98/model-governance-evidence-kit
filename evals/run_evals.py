"""Golden evals for model-governance-evidence-kit.

SYNTHETIC DATA ONLY. Deterministic: eval_report.json must be
byte-identical across runs (no timestamps, ports, or temp paths).

G1 PSI flags the planted drift. G2 PSI is clean on the undrifted
control. G3 calibration catches the planted bad band. G4 AUC matches
brute-force pairwise on a small fixture. G5 evidence pack hash and
verdict determinism (byte-identical across runs). G6-G10 pin the
generator, variance stability, the check gate, the serve round-trip,
and validation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from drift import drift_report  # noqa: E402
from evidence import build_evidence, evidence_json  # noqa: E402
from generator import generate_portfolio, records_to_jsonl  # noqa: E402
from metrics import accuracy_report, auc, auc_bruteforce, variance_report  # noqa: E402
from validation import validate_records  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: model-governance-evidence-kit ==")
    records = generate_portfolio()
    control = generate_portfolio(drift=False, miscalibrated=False)
    ev = build_evidence(records)

    drift = drift_report(records)
    check("G1 PSI flags planted drift (score PSI > 0.25, dti CSI > 0.25)",
          drift["score_psi"] == 2.612705 and drift["score_verdict"] == "fail"
          and drift["features"]["dti_ratio"]["csi"] == 2.924964
          and drift["verdict"] == "fail",
          f"score_psi={drift['score_psi']}")

    cdrift = drift_report(control)
    check("G2 PSI clean on undrifted control (PSI < 0.10, pass)",
          cdrift["score_psi"] == 0.008779 and cdrift["verdict"] == "pass",
          f"control_psi={cdrift['score_psi']}")

    acc = accuracy_report(records)
    s3 = next(r for r in acc["calibration"] if r["band"] == "S3")
    check("G3 calibration catches planted bad band (S3 err > 0.20, fail)",
          s3["abs_error"] == 0.49476 and acc["worst_band"] == "S3"
          and acc["calibration_verdict"] == "fail",
          f"s3_error={s3['abs_error']}")

    small = [r for r in generate_portfolio(per_window=25)
             if r["window"] == "current"]
    fast = auc([r["score"] for r in small], [r["label"] for r in small])
    slow = auc_bruteforce([r["score"] for r in small], [r["label"] for r in small])
    hand = auc([0.9, 0.4, 0.5, 0.1], [1, 1, 0, 0])
    check("G4 AUC matches brute-force on small fixture (+ hand case 0.75)",
          fast == slow and hand == 0.75
          and auc_bruteforce([0.9, 0.4, 0.5, 0.1], [1, 1, 0, 0]) == 0.75,
          f"fixture_auc={fast}")

    ev2 = build_evidence(generate_portfolio())
    json_bytes = evidence_json(ev)
    md5 = hashlib.md5(json_bytes.encode()).hexdigest()
    check("G5 evidence determinism (byte-identical, verdicts pinned)",
          json_bytes == evidence_json(ev2)
          and ev["input_sha256"] == ev2["input_sha256"]
          and ev["overall_verdict"] == "fail"
          and ev["verdicts"] == {"drift": "fail", "accuracy": "fail",
                                 "variance": "pass"},
          f"evidence_md5={md5}")

    text = records_to_jsonl(records)
    gen_md5 = hashlib.md5(text.encode()).hexdigest()
    check("G6 generator determinism + headline AUC pinned",
          text == records_to_jsonl(generate_portfolio())
          and acc["auc"] == 0.777225,
          f"jsonl_md5={gen_md5} auc={acc['auc']}")

    var = variance_report(records)
    unstable = variance_report(records, jitter_sd=0.20)
    check("G7 variance stable on seeded runs, fails under large jitter",
          var["verdict"] == "pass" and var["mean_score_std"] == 0.007989
          and unstable["verdict"] == "fail",
          f"mean_std={var['mean_score_std']}")

    import tempfile
    tmp = tempfile.mkdtemp(prefix="mgov-evals-")
    clean_path = os.path.join(tmp, "clean.jsonl")
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(records_to_jsonl(control))
    codes = (_run_cli(["check"]), _run_cli(["check", "--input", clean_path]),
             _run_cli(["check", "--input", "/tmp/no-such-mgov-xyz.json"]))
    check("G8 CLI check exit codes 1 (planted) / 0 (control) / 2 (missing)",
          codes == (EXIT_FAIL, EXIT_OK, EXIT_ERROR), f"codes={codes}")

    reset_state(records)
    server = make_server(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        def _post(payload):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/validate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        serve_ok = _post({}) == ev and _post({"records": records}) == ev
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("G9 serve round-trip matches local evidence", serve_ok)

    bad = dict(records[0], label=7)
    check("G10 validation flags defect; seeded + control packs pinned",
          validate_records([bad]) != [] and validate_records(records) == []
          and build_evidence(control)["overall_verdict"] == "pass")

    passed = sum(1 for r in RESULTS if r["passed"])
    total_n = len(RESULTS)
    doc = {"passed": passed, "total": total_n,
           "all_passed": passed == total_n, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_n} evals passed -> evals/eval_report.json")
    return 0 if passed == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
