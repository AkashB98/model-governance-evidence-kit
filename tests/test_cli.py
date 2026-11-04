import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

from tests.helpers import mk  # noqa: F401
from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, load_records, main, make_server, reset_state
from generator import generate_portfolio, records_to_jsonl


def run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return main(argv)


class TestCli(unittest.TestCase):
    def test_check_seeded_fails_gate(self):
        self.assertEqual(run_cli(["check"]), EXIT_FAIL)

    def test_check_control_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clean.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(records_to_jsonl(
                    generate_portfolio(drift=False, miscalibrated=False)))
            self.assertEqual(run_cli(["check", "--input", path]), EXIT_OK)

    def test_check_missing_file_is_error(self):
        self.assertEqual(run_cli(["check", "--input", "/tmp/nope-mgov.json"]),
                         EXIT_ERROR)

    def test_check_bad_data_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([mk(label=9)], fh)
            self.assertEqual(run_cli(["check", "--input", path]), EXIT_ERROR)

    def test_generate_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.jsonl")
            self.assertEqual(run_cli(["generate", "--per-window", "10",
                                      "--out", path]), EXIT_OK)
            self.assertEqual(len(load_records(path)), 20)

    def test_evidence_md_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("json", "md"):
                path = os.path.join(tmp, f"ev.{fmt}")
                self.assertEqual(run_cli(["evidence", "--format", fmt,
                                          "--out", path]), EXIT_OK)
                with open(path, encoding="utf-8") as fh:
                    self.assertTrue(fh.read().strip())

    def test_validate_command(self):
        self.assertEqual(run_cli(["validate"]), EXIT_OK)


class TestServe(unittest.TestCase):
    def _serve(self):
        server = make_server(0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, server.server_address[1]

    def _req(self, port, path, payload=None, method="GET"):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def test_health_and_validate_round_trip(self):
        from evidence import build_evidence
        records = generate_portfolio()
        reset_state(records)
        server, port = self._serve()
        try:
            self.assertEqual(self._req(port, "/health"), (200, {"status": "ok"}))
            status, served = self._req(port, "/validate", {}, "POST")
            self.assertEqual(status, 200)
            self.assertEqual(served, build_evidence(records))
            status, served2 = self._req(port, "/validate",
                                        {"records": records}, "POST")
            self.assertEqual(served2, build_evidence(records))
        finally:
            server.shutdown()
            server.server_close()
            reset_state()

    def test_unknown_endpoint_404(self):
        server, port = self._serve()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._req(port, "/nope")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
