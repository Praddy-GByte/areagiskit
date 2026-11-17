# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import os
import tempfile
import unittest

import numpy as np

from _util import mod


class TestUTCI(unittest.TestCase):
    def setUp(self):
        self.U = mod("kits.urban.utci")

    def test_reference_values(self):
        # values from pythermalcomfort 4.8 (utci(tdb, tr, v, rh)), which implements the same Bröde et al. (2012) polynomial
        self.assertAlmostEqual(float(self.U.utci(25.0, 25.0, 1.0, 50.0)), 24.6, delta=0.06)
        self.assertAlmostEqual(float(self.U.utci(35.0, 55.0, 2.0, 40.0)), 39.8, delta=0.06)
        self.assertEqual(len(self.U._TERMS), 210)

    def test_sun_hotter_than_shade(self):
        el = self.U.solar_elevation(28.45, 136, 12.5)
        self.assertGreater(el, 75)
        sun = self.U.mrt(42.0, 700.0, 220.0, 930.0, 415.0, el)
        shade = self.U.mrt(42.0, 0.0, 0.0, 0.0, 415.0, -1.0)
        self.assertGreater(sun, shade + 10)
        self.assertGreater(self.U.utci(42.0, sun, 3.0, 20.0), self.U.utci(42.0, shade, 3.0, 20.0))


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.N = mod("kits.urban.network")

    def test_straight_road_and_grid_detour(self):
        roads = np.zeros((200, 200), bool)
        roads[::20, :] = True
        roads[:, ::20] = True
        d = self.N.road_distance(roads, [(100, 100)], 30.0)
        self.assertAlmostEqual(float(d[100, 150]), 50 * 30.0, places=3)        # along the same road
        self.assertAlmostEqual(float(d[120, 120]), 40 * 30.0, delta=30.0)       # Manhattan path (one corner cut at a junction), not diagonal (848 m)
        self.assertTrue(np.isinf(d[105, 105]))                                  # off-road cell not reached by the search

    def test_snap_and_unreachable(self):
        roads = np.zeros((50, 50), bool)
        roads[10, :] = True
        d = self.N.road_distance(roads, [(13, 5)], 10.0, snap_m=100.0)
        self.assertAlmostEqual(float(d[10, 5]), 30.0, places=3)                # snap distance added
        none = self.N.road_distance(roads, [(40, 40)], 10.0, snap_m=50.0)
        self.assertTrue(np.isinf(none).all())


class TestTwoSFCAWithDistances(unittest.TestCase):
    def test_precomputed_matches_euclidean(self):
        HX = mod("kits.health.extra")
        dem = np.array([[0.0, 0.0], [1000.0, 0.0], [3000.0, 0.0]])
        sup = np.array([[500.0, 0.0], [2500.0, 0.0]])
        D = np.sqrt(((dem[:, None, :] - sup[None]) ** 2).sum(-1))
        a1, r1 = HX.two_step_fca(dem, [100, 200, 300], sup, [1, 2], 2000.0)
        a2, r2 = HX.two_step_fca(dem, [100, 200, 300], sup, [1, 2], 2000.0, D=D)
        np.testing.assert_allclose(a1, a2)
        # a longer road distance can only lower access
        a3, _ = HX.two_step_fca(dem, [100, 200, 300], sup, [1, 2], 2000.0, D=D * 1.5)
        self.assertLessEqual(a3.sum(), a1.sum() + 1e-12)


class TestForecast(unittest.TestCase):
    def setUp(self):
        self.F = mod("kits.health.forecast")

    def test_recovers_trend_and_flags_outbreak(self):
        rng = np.random.default_rng(5)
        n, T = 80, 8
        P = np.tile(rng.uniform(5000, 50000, n)[:, None], (1, T))
        mu = P * 0.002 * np.exp(rng.normal(0, 0.3, n))[:, None] * np.exp(0.05 * np.arange(T))[None]
        C = rng.poisson(mu).astype(float)
        m = self.F.fit(C, P)
        self.assertAlmostEqual(np.log(np.exp(m["b"])), 0.05, delta=0.02)
        C[:5, -1] = rng.poisson(mu[:5, -1] * 3)
        ew = self.F.early_warning(C, P)
        self.assertGreaterEqual(int(ew["alert"][:5].sum()), 4)
        self.assertLessEqual(int(ew["alert"][5:].sum()), 6)
        fc = self.F.forecast_next(C, P)
        self.assertTrue(np.all(fc["lo"] <= fc["mean"]) and np.all(fc["mean"] <= fc["hi"]))


class TestWealth(unittest.TestCase):
    def test_read_and_paint(self):
        W = mod("kits.urban.wealth")
        A = mod("core.aoi")
        aoi = A.AOI(bbox=(77.0, 28.40, 77.10, 28.50), name="t")
        g = A.Grid(aoi, 100)
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as fh:
            fh.write("latitude,longitude,rwi,error\n28.45,77.05,1.2,0.3\n28.42,77.02,-0.5,0.3\n10.0,10.0,9,0.1\n")
        try:
            pts = W.read_points(path, aoi.bbox)
            self.assertEqual(len(pts), 2)                                       # the far-away point is dropped
            grid = W.to_grid(pts, g)
            r, c = g.lonlat_to_pixel(77.05, 28.45)
            self.assertAlmostEqual(float(grid[r, c]), 1.2, places=5)
            r2, c2 = g.lonlat_to_pixel(77.05 + 0.009, 28.45)                   # ~0.9 km east: same 2.4 km tile
            self.assertAlmostEqual(float(grid[r2, c2]), 1.2, places=5)
            pts2, note = W.fetch(aoi.bbox, user_csv=path)
            self.assertEqual(note, "your file")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
