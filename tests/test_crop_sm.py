# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Crop soil moisture (NASA POWER GWETROOT/GWETTOP): parsing, anomalies, season series, correlation, offline."""
import unittest

import numpy as np

from _util import mod


def _fake_power(n_years=40, seed=3, dry_year=2012):
    rng = np.random.default_rng(seed)
    root, top = {}, {}
    for y in range(1981, 1981 + n_years):
        for m in range(1, 13):
            base = 0.6 + 0.1 * np.sin(m / 12 * 2 * np.pi)
            v = base + rng.normal(0, 0.03) - (0.15 if (y == dry_year and 5 <= m <= 8) else 0)
            root[f"{y}{m:02d}"] = round(float(v), 3)
            top[f"{y}{m:02d}"] = round(float(v - 0.05), 3)
        root[f"{y}13"] = 0.6                              # annual mean key must be ignored
    root["198101"] = -999.0                               # missing value must be ignored
    return {"properties": {"parameter": {"GWETROOT": root, "GWETTOP": top}}}


class TestSoilMoisture(unittest.TestCase):
    def setUp(self):
        self.SL = mod("kits.crop.soil")
        self._orig = (self.SL.get_json, self.SL.trace)
        self.SL.get_json = lambda url, timeout=60: _fake_power()
        self.SL.trace = lambda **k: None

    def tearDown(self):
        self.SL.get_json, self.SL.trace = self._orig

    def test_monthly_parsing_skips_annual_and_missing(self):
        m = self.SL.monthly((-88.75, 36.62, -88.65, 36.70), (-88.7, 36.66))
        self.assertNotIn((1981, 13), m)
        self.assertNotIn("root", m[(1981, 1)])            # -999 dropped, surface kept
        self.assertIn("top", m[(1981, 1)])
        self.assertTrue(all(0 <= d.get("root", 0.5) <= 1 for d in m.values()))

    def test_anomalies_are_standardised_and_find_the_dry_year(self):
        m = self.SL.monthly((-88.75, 36.62, -88.65, 36.70), (-88.7, 36.66))
        z = self.SL.anomalies(m, "root")
        jul = np.array([v for (y, mm), v in z.items() if mm == 7])
        self.assertAlmostEqual(float(jul.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(jul.std()), 1.0, places=6)
        self.assertLess(z[(2012, 7)], -2.5)               # planted drought stands out
        s = self.SL.season_series(m, [5, 6, 7, 8], "root")
        self.assertEqual(min(s, key=lambda y: s[y]["z"]), 2012)
        self.assertEqual(self.SL.sm_class(z[(2012, 7)]), "extremely dry")

    def test_correlation_needs_eight_common_years(self):
        a = {y: float(y) for y in range(2000, 2010)}
        b = {y: 2.0 * y + 1 for y in range(2000, 2010)}
        r, n = self.SL.corr(a, b)
        self.assertAlmostEqual(r, 1.0, places=9)
        self.assertEqual(n, 10)
        r, n = self.SL.corr({1: 1.0, 2: 2.0}, {1: 1.0, 2: 3.0})
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
