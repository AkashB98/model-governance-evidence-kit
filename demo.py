"""End-to-end demo. SYNTHETIC DATA ONLY. Offline; serve is loopback."""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main as cli_main, make_server, reset_state  # noqa: E402
from drift import drift_report  # noqa: E402
from evidence import build_evidence, evidence_json, render_markdown  # noqa: E402
from generator import generate_portfolio  # noqa: E402
from metrics import accuracy_report, auc, auc_bruteforce, variance_report  # noqa: E402
from validation import validate_records  # noqa: E402


def _request(port, path, payload=None, method="GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate seeded portfolio (planted drift + bad band) ==")
    records = generate_portfolio()
    assert records == generate_portfolio(), "generator must be deterministic"
    assert validate_records(records) == []
    print(f"   {len(records)} records, clean")

    print("\n== 2. drift: planted shift flagged, control clean ==")
    drift = drift_report(records)
    control = drift_report(generate_portfolio(drift=False, miscalibrated=False))
    print(f"   seeded score PSI {drift['score_psi']} ({drift['score_verdict']}), "
          f"dti CSI {drift['features']['dti_ratio']['csi']}")
    print(f"   control score PSI {control['score_psi']} ({control['score_verdict']})")
    assert drift["verdict"] == "fail" and control["verdict"] == "pass"

    print("\n== 3. accuracy: AUC vs brute force, bad band caught ==")
    small = generate_portfolio(per_window=25)
    cur = [r for r in small if r["window"] == "current"]
    fast = auc([r["score"] for r in cur], [r["label"] for r in cur])
    slow = auc_bruteforce([r["score"] for r in cur], [r["label"] for r in cur])
    assert fast == slow, (fast, slow)
    acc = accuracy_report(records)
    s3 = next(r for r in acc["calibration"] if r["band"] == "S3")
    print(f"   AUC {acc['auc']}; S3 pred {s3['predicted']} obs {s3['observed']} "
          f"err {s3['abs_error']}")
    assert acc["calibration_verdict"] == "fail" and s3["abs_error"] > 0.20

    print("\n== 4. variance: seeded re-runs are stable ==")
    var = variance_report(records)
    print(f"   {var['runs']} runs, mean std {var['mean_score_std']} -> {var['verdict']}")
    assert var["verdict"] == "pass"

    print("\n== 5. evidence pack: deterministic, verdict FAIL on seeded data ==")
    ev = build_evidence(records)
    assert evidence_json(ev) == evidence_json(build_evidence(records))
    assert ev["overall_verdict"] == "fail"
    assert "Overall verdict: FAIL" in render_markdown(ev)
    clean_ev = build_evidence(generate_portfolio(drift=False, miscalibrated=False))
    print(f"   seeded overall={ev['overall_verdict']} {ev['verdicts']}; "
          f"control overall={clean_ev['overall_verdict']}")
    assert clean_ev["overall_verdict"] == "pass", clean_ev["verdicts"]

    print("\n== 6. check gate exit codes 0 / 1 / 2 ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mgov-demo-")
    clean_path = os.path.join(tmp, "clean.jsonl")
    from generator import records_to_jsonl
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(records_to_jsonl(
            generate_portfolio(drift=False, miscalibrated=False)))
    planted_code = cli_main(["check"])
    clean_code = cli_main(["check", "--input", clean_path])
    err_code = cli_main(["check", "--input", "/tmp/no-such-mgov-xyz.json"])
    print(f"   planted={planted_code}, clean={clean_code}, missing={err_code}")
    assert planted_code == 1 and clean_code == 0 and err_code == 2

    print("\n== 7. live localhost serve round-trip ==")
    reset_state(records)
    server = make_server(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served = _request(port, "/validate", payload={}, method="POST")
        assert status == 200 and served == ev
        status, served2 = _request(port, "/validate",
                                   payload={"records": records}, method="POST")
        assert status == 200 and served2 == ev
        print("   POST /validate (seeded + explicit) matches local evidence")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, drift, accuracy, variance, evidence, "
          "check and serve verified.")


if __name__ == "__main__":
    main()
