# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import http.server
import json
import os
import tempfile
import threading
import unittest
import zipfile

import numpy as np

from _util import mod, PKG


class TestRegistryAndCli(unittest.TestCase):
    def test_registry_filter(self):
        K = mod("kits")
        PI = mod("plugin_info")
        keys = [k.KEY for k in K.all_kits()]
        if PI.KIT_KEYS:
            self.assertEqual(keys, list(PI.KIT_KEYS))
        else:
            self.assertEqual(len(keys), 6)
        old = PI.KIT_KEYS
        try:
            PI.KIT_KEYS = ["floodstudy"]
            self.assertEqual([k.KEY for k in K.all_kits()], ["floodstudy"])
        finally:
            PI.KIT_KEYS = old
        for k in K.all_kits():
            self.assertTrue(k.NAME and k.KEY and isinstance(k.PARAMS, list))
            names = [p["name"] for p in k.PARAMS]
            self.assertEqual(len(names), len(set(names)), k.KEY)

    def test_cli_list_and_bad_kit(self):
        import contextlib
        import io
        CLI = mod("cli")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(CLI.main(["list"]), 0)
        keys = [k.KEY for k in mod("kits").all_kits()]
        self.assertTrue(all(k in out.getvalue() for k in keys))
        with self.assertRaises(KeyError):
            CLI.main(["no_such_kit", "--bbox", "0", "0", "1", "1"])


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = {"n": 0}

    def do_GET(self):
        self.hits["n"] += 1
        if self.path.startswith("/busy") and self.hits["n"] == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        body = json.dumps({"ok": True, "n": self.hits["n"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestDownloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_retry_on_429_then_cache(self):
        D = mod("core.data")
        _Handler.hits["n"] = 0
        url = f"{self.base}/busy?x={os.getpid()}-{id(self)}"
        a = D.get_json(url)
        self.assertTrue(a["ok"])
        self.assertEqual(_Handler.hits["n"], 2)           # one 429, one success
        b = D.get_json(url)                                # served from the disk cache
        self.assertEqual(_Handler.hits["n"], 2)
        self.assertEqual(a, b)
        c = D.get_json(url, cache=0)                       # opt out: live request
        self.assertEqual(_Handler.hits["n"], 3)
        self.assertEqual(c["n"], 3)
        p = D._cache_path(url)
        if p and os.path.exists(p):
            os.remove(p)

    def test_unreachable_raises_kiterror(self):
        D = mod("core.data")
        with self.assertRaises(D.KitError):
            D.get_json("http://127.0.0.1:9/never", tries=1, cache=0, timeout=2)


class TestReport(unittest.TestCase):
    def test_writes_html_docx_pdf(self):
        A = mod("core.aoi")
        R = mod("core.report")
        M = mod("core.maps")
        aoi = A.AOI(bbox=(77.0, 28.4, 77.05, 28.45), name="Test área")
        g = A.Grid(aoi, 100)
        arr = np.where(g.mask, (np.arange(g.rows * g.cols).reshape(g.shape) % 3) + 1, 0).astype(np.uint8)
        cls = {1: ("One", "#1f77b4"), 2: ("Two", "#ff7f0e"), 3: ("Three", "#2ca02c")}
        verdict = R.grade([(2, 2, "check one passed"), (1, 2, "check two partly")])
        rep = R.Report("Test Kit", "A tiny report", aoi.name, verdict, "#2f6b4f", aoi, g)
        spec = M.MapSpec(arr, g, "Test map", classes=cls, source="synthetic")
        rep.summary([("Area", "1 km²"), ("Temperature", "41.9 °C")], ["A point with ₂ and ° symbols."], spec)
        rep.study_area(spec, [("Note", "unicode: é, ü, ñ, ç, Ω")])
        rep.data([("Synthetic", "test", "100 m", "2026", "CC0")])
        rep.method([("Step", "Nothing real.")])
        rep.heading("Results")
        rep.table(["A", "B"], [["1", "2"]], "A table")
        rep.bars([("x", 1.0, "#333333"), ("y", -2.0, "#999999")], "Bars", "u", signed=True)
        rep.line([("s", [1, 2, 3], "#2166ac")], ["a", "b", "c"], "Line", "u")
        rep.validation([("c", "r", "m")])
        rep.limits(["limit"])
        rep.conclusions(["done"])
        rep.references(["ref"])
        rep.proof([("x", "id", "2026-01-01", "https://example.org")], "python -m x")
        rep.files(["report.html"])
        with tempfile.TemporaryDirectory() as d:
            docs = rep.write(d, 1.0)
            if docs.get("pdf_job"):
                R.write_deferred_pdf(docs)
            self.assertTrue(os.path.getsize(docs["html"]) > 1000)
            with zipfile.ZipFile(docs["docx"]) as z:
                self.assertIn("word/document.xml", z.namelist())
                self.assertIn("A tiny report", z.read("word/document.xml").decode("utf-8"))
            if docs.get("pdf"):
                with open(docs["pdf"], "rb") as fh:
                    self.assertEqual(fh.read(4), b"%PDF")
            else:
                self.skipTest("Qt not available: PDF skipped")


if __name__ == "__main__":
    unittest.main()
