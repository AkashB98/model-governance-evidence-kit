"""CLI: generate | validate | evidence | check | demo | serve.

SYNTHETIC DATA ONLY. Exit codes for ``check``: 0 = pass/warn,
1 = validation fail, 2 = usage / IO / data-quality error.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence import build_evidence, canonical, evidence_json, render_markdown  # noqa: E402
from generator import generate_portfolio, records_to_csv, records_to_jsonl  # noqa: E402
from validation import validate_records  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def _coerce(r: Dict) -> Dict:
    out = dict(r)
    for f in ("dti_ratio", "income_volatility", "credit_utilization", "score"):
        if isinstance(out.get(f), str):
            try:
                out[f] = float(out[f])
            except ValueError:
                pass
    if isinstance(out.get("label"), str):
        try:
            out["label"] = int(out["label"])
        except ValueError:
            pass
    return out


def load_records(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return []
    if path.endswith(".csv"):
        return [_coerce(dict(row)) for row in csv.DictReader(io.StringIO(text))]
    if stripped.startswith("["):
        return [_coerce(dict(x)) for x in json.loads(stripped)]
    return [_coerce(json.loads(line)) for line in stripped.splitlines() if line.strip()]


def _load_or_error(path: Optional[str]) -> Optional[List[Dict]]:
    if path is None:
        return generate_portfolio()
    try:
        return load_records(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _evidence_or_error(records):
    issues = validate_records(records)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        return None
    try:
        return build_evidence(records)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def cmd_generate(args) -> int:
    records = generate_portfolio(per_window=args.per_window, seed=args.seed,
                                 drift=not args.no_drift,
                                 miscalibrated=not args.no_miscalibration)
    fmt = args.format or ("csv" if (args.out and args.out.endswith(".csv")) else "jsonl")
    text = records_to_csv(records) if fmt == "csv" else records_to_jsonl(records)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"generated {len(records)} synthetic records (seed={args.seed}) -> {args.out}")
    else:
        print(text, end="")
    return EXIT_OK


def cmd_validate(args) -> int:
    records = _load_or_error(args.input)
    if records is None:
        return EXIT_ERROR
    ev = _evidence_or_error(records)
    if ev is None:
        return EXIT_ERROR
    print(canonical({"verdicts": ev["verdicts"],
                     "overall_verdict": ev["overall_verdict"],
                     "checks": ev["checks"]}))
    return EXIT_OK


def cmd_evidence(args) -> int:
    records = _load_or_error(args.input)
    if records is None:
        return EXIT_ERROR
    ev = _evidence_or_error(records)
    if ev is None:
        return EXIT_ERROR
    text = render_markdown(ev) if args.format == "md" else evidence_json(ev)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"evidence pack ({args.format}) -> {args.out} "
              f"verdict={ev['overall_verdict']}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return EXIT_OK


def cmd_check(args) -> int:
    records = _load_or_error(args.input)
    if records is None:
        return EXIT_ERROR
    issues = validate_records(records)
    if issues:
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    ev = build_evidence(records)
    d, a = ev["checks"]["drift"]["detail"], ev["checks"]["accuracy"]["detail"]
    print(f"records={ev['record_count']} score_psi={d['score_psi']} "
          f"auc={a['auc']} max_calibration_error={a['max_calibration_error']} "
          f"verdicts={ev['verdicts']}")
    if ev["overall_verdict"] == "fail":
        print(f"check FAIL: overall verdict fail ({ev['verdicts']})")
        return EXIT_FAIL
    print(f"check PASS (overall={ev['overall_verdict']})")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


class _State:
    def __init__(self):
        self.records: List[Dict] = generate_portfolio()


_STATE = _State()


def reset_state(records: Optional[List[Dict]] = None) -> None:
    _STATE.records = list(records) if records is not None else generate_portfolio()


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        if self.path != "/validate":
            return self._send_json(404, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if isinstance(body, dict) and "records" in body:
                records = [_coerce(dict(x)) for x in body["records"]]
                issues = validate_records(records)
                if issues:
                    return self._send_json(400, {"error": "invalid records",
                                                 "issues": issues[:5]})
                _STATE.records = records
                return self._send_json(200, build_evidence(records))
            if isinstance(body, dict) and not body:
                return self._send_json(200, build_evidence(_STATE.records))
            raise ValueError("expected {} or {'records': [...]}")
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  (POST /validate, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-governance-evidence-kit",
        description="Periodic model-level validation evidence (synthetic data).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic scored portfolio")
    g.add_argument("--per-window", type=int, default=800)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default=None)
    g.add_argument("--format", choices=["csv", "jsonl"], default=None)
    g.add_argument("--no-drift", action="store_true")
    g.add_argument("--no-miscalibration", action="store_true")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="run drift/accuracy/variance checks (JSON)")
    v.add_argument("--input", default=None)
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("evidence", help="build the evidence pack (JSON or Markdown)")
    e.add_argument("--input", default=None)
    e.add_argument("--format", choices=["json", "md"], default="json")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_evidence)

    c = sub.add_parser("check", help="CI gate: exit 0 pass / 1 fail / 2 error")
    c.add_argument("--input", default=None)
    c.set_defaults(func=cmd_check)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="localhost API: POST /validate, GET /health")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
