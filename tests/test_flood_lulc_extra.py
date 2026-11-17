# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Offline tests for the flood response helpers (safe ground, vulnerability, guidance, plain errors) and LULC texture."""
import unittest

import numpy as np

from _util import mod

RS = mod("kits.flood.response")
CO = mod("kits.lulc.composite")


class FakeGrid:
    """Minimal grid: north up, 1 pixel = res metres, lon = col, lat = -row (only used for round trips)."""
    def __init__(self, rows, cols, res=10.0):
        self.rows, self.cols, self.res = rows, cols, res
        self.shape = (rows, cols)
        self.mask = np.ones(self.shape, bool)
        self.px_km2 = res * res / 1e6

    def lonlat_to_pixel(self, lon, lat):
        return int(round(-lat)), int(round(lon))

    def pixel_to_lonlat(self, r, c):
        return float(c), float(-r)


class TestSafeGround(unittest.TestCase):
    def setUp(self):
        # valley: low flooded strip in the middle (cols 40-59), ground rises 0.2 m per column away from it
        self.g = FakeGrid(100, 100, 10.0)
        cols = np.arange(100)
        self.dem = np.tile(np.abs(cols - 49.5) * 0.2, (100, 1)).astype(np.float32)
        self.flood = np.zeros(self.g.shape, bool)
        self.flood[:, 40:60] = True
        self.perm = np.zeros(self.g.shape, bool)
        self.ever = self.flood.copy()

    def test_safe_is_dry_and_high(self):
        safe, k, level = RS.safe_ground(self.g, self.flood, self.perm, self.ever, self.dem)
        self.assertEqual(k, 10)
        self.assertTrue(safe.any())
        big = RS.expand(safe.astype(np.uint8), k, self.g.shape) == 1
        self.assertFalse((big & self.flood).any())                     # never inside the flood
        self.assertTrue((self.dem[big] >= self.dem[self.flood].max() + 1.0 - 1.0).all())

    def test_nearest_direction(self):
        res = RS.evacuation(self.g, [dict(lon=50.0, lat=-50.0, people_flooded=100.0, kind="settlement")], None,
                            self.flood, self.perm, self.ever, self.dem)
        r = res["rows"][0]
        self.assertIsNotNone(r["safe_km"])
        self.assertIn(r["bearing"], ("E", "W"))
        self.assertLess(r["safe_km"], 0.5)
        self.assertGreater(res["safe_km2"], 0)

    def test_no_safe_ground_on_flat_flooded_land(self):
        flat = np.zeros(self.g.shape, np.float32)
        allf = np.ones(self.g.shape, bool)
        res = RS.evacuation(self.g, [dict(lon=50.0, lat=-50.0, people_flooded=10.0, kind="settlement")], None,
                            allf, self.perm, allf, flat)
        self.assertIsNone(res["rows"][0]["safe_km"])


class TestVulnerabilityGuidance(unittest.TestCase):
    def test_vulnerability_range(self):
        g = FakeGrid(60, 60, 30.0)
        built = np.random.default_rng(1).random(g.shape).astype(np.float32)
        hand = np.tile(np.linspace(0, 10, 60), (60, 1)).astype(np.float32)
        dist = np.tile(np.linspace(0, 5000, 60), (60, 1)).T.astype(np.float32)
        v, comps = RS.vulnerability(g, built, hand, dist, None)
        self.assertEqual(len(comps), 2)
        self.assertTrue(np.nanmin(v) >= 0 and np.nanmax(v) <= 1)
        # far from safe ground and on low ground -> more vulnerable than near and high
        self.assertGreater(np.nanmean(v[-10:, :10]), np.nanmean(v[:10, -10:]))

    def test_guidance_and_plain_error(self):
        R = {"more": {"evac": {"rows": [dict(lat=26.7, lon=67.7, people_flooded=1200.0, safe_km=2.5, bearing="NE")]},
                      "isolated_people": 300.0, "rp_exposure": [dict(rp=100, km2=12.0, pop=5000.0)]},
             "hydro": {"q": {"quant": {2: 1000.0, 5: 2000.0, 10: 3000.0}, "loc": (67.8, 26.8)}}}
        pts = RS.guidance(R)
        txt = " ".join(pts)
        self.assertIn("Priority settlements", txt)
        self.assertIn("Early-warning", txt)
        self.assertIn("No-build zone", txt)
        self.assertIn("general guidance", pts[-1])
        self.assertIn("busy", RS.plain_error("HTTP Error 429: Too Many Requests"))
        self.assertIn("did not answer", RS.plain_error("The read operation timed out"))


class TestTexture(unittest.TestCase):
    def test_glcm_like(self):
        smooth = np.ones((40, 40), np.float32) * 0.3
        rough = np.random.default_rng(2).random((40, 40)).astype(np.float32)
        (c1, h1, s1), names = CO.texture_extended(smooth + np.linspace(0, 1e-3, 40)[None, :], smooth)
        (c2, h2, s2), _ = CO.texture_extended(rough, rough)
        self.assertEqual(len(names), 3)
        self.assertLess(np.nanmean(c1), np.nanmean(c2))     # rough surface: more contrast
        self.assertGreater(np.nanmean(h1), np.nanmean(h2))  # and less homogeneity
        self.assertLess(np.nanmean(s1), np.nanmean(s2))


if __name__ == "__main__":
    unittest.main()
