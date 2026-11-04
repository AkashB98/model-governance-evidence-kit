"""No-network enforcement: with sockets blocked, the full pipeline
must still work. (``serve`` binds loopback on demand only.)"""

import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Blocker:
    def __init__(self, *a, **k):
        raise RuntimeError("network access blocked in hermetic test")


class TestNoNetwork(unittest.TestCase):
    def setUp(self):
        self._real_socket = socket.socket
        socket.socket = _Blocker
        import urllib.request as urlreq
        self._real_urlopen = urlreq.urlopen
        urlreq.urlopen = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("network access blocked in hermetic test"))

    def tearDown(self):
        socket.socket = self._real_socket
        import urllib.request as urlreq
        urlreq.urlopen = self._real_urlopen

    def test_full_pipeline_offline(self):
        from evidence import build_evidence
        from generator import generate_portfolio
        from validation import validate_records
        records = generate_portfolio()
        self.assertEqual(validate_records(records), [])
        self.assertEqual(build_evidence(records)["overall_verdict"], "fail")

    def test_no_module_opens_sockets_at_import(self):
        for mod in ("generator", "drift", "metrics", "evidence",
                    "validation", "cli", "demo"):
            sys.modules.pop(mod, None)
        import cli, demo, drift, evidence, generator, metrics, validation  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
