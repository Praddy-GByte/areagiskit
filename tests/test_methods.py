# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import datetime as dt
import unittest

import numpy as np

from _util import mod


class TestFloodDesign(unittest.TestCase):
    def setUp(self):
        self.DS = mod("kits.flood.design")
        # a valley: HAND rises 0.1 m per column away from a river in column 0
        self.hand = np.tile(np.arange(60) * 0.1, (40, 1)).astype(float)
        self.mask = np.ones_like(self.hand, bool)
        self.perm = np.zeros_like(self.mask)

    def test_calibrate_recovers_stage(self):
        obs = self.hand <= 2.0
        cal = self.DS.calibrate(self.hand, obs, self.mask, self.mask, self.perm)
        self.assertAlmostEqual(cal["stage_area"], 2.0, delta=0.11)
        self.assertGreater(cal["csi_area"], 0.95)

    def test_stage_for_area_and_depth(self):
        px = 0.01                              # km² per pixel
        st = self.DS.stage_for_area(self.hand, self.mask, self.perm, km2=40 * 21 * px, px_km2=px)
        self.assertAlmostEqual(st, 2.0, delta=0.11)
        d = self.DS.depth_map(self.hand, self.mask, self.perm, st)
        self.assertAlmostEqual(np.nanmax(d), st, delta=0.02)
        self.assertTrue(np.isnan(d[:, -1]).all())

    def test_csi(self):
        a = np.zeros((4, 4), bool); a[:2] = True
        self.assertEqual(self.DS.csi(a, a, np.ones_like(a)), 1.0)
        self.assertEqual(self.DS.csi(a, ~a, np.ones_like(a)), 0.0)


class TestHeatIndex(unittest.TestCase):
    def test_noaa_table(self):
        CM = mod("kits.urban.climate")
        f2c = lambda f: (f - 32) * 5 / 9
        # NOAA/NWS heat index chart values (°F): (T, RH) -> HI
        for t_f, rh, hi_f in ((90, 70, 106), (100, 40, 109), (86, 50, 89), (80, 40, 80), (104, 55, 137)):
            got = float(CM.heat_index(f2c(t_f), rh))
            self.assertAlmostEqual(got, f2c(hi_f), delta=f2c(hi_f + 2) - f2c(hi_f), msg=(t_f, rh))

    def test_mann_kendall(self):
        CM = mod("kits.urban.climate")
        s, p = CM.mann_kendall_slope(list(range(20)))
        self.assertAlmostEqual(s, 1.0)
        self.assertLess(p, 0.001)


class TestPhenology(unittest.TestCase):
    def test_single_and_double_season(self):
        SE = mod("kits.crop.series")
        T = 36
        t = np.arange(T)
        single = 0.2 + 0.6 * np.exp(-0.5 * ((t - 18) / 3.5) ** 2)
        double = 0.2 + 0.6 * np.exp(-0.5 * ((t - 10) / 2.5) ** 2) + 0.6 * np.exp(-0.5 * ((t - 27) / 2.5) ** 2)
        bare = np.full(T, 0.15)
        S = np.stack([single, double, bare], -1).reshape(T, 1, 3).astype(np.float32)
        dates = [dt.date(2021, 1, 1) + dt.timedelta(days=10 * i) for i in range(T)]
        out = SE.phenology(S, dates)
        self.assertEqual(out["n"].ravel().tolist(), [1, 2, 0])


class TestRunout(unittest.TestCase):
    def test_debris_runs_downhill_and_stops(self):
        RO = mod("kits.landslide.runout")
        res = 30.0
        rows, cols = 40, 20
        # slope 30° for 15 rows, then flat valley floor
        z = np.zeros((rows, cols))
        drop = res * np.tan(np.radians(30))
        for r in range(rows):
            z[r] = (15 - r) * drop if r < 15 else 0.0
        z += 0.001 * np.arange(cols)               # tiny tilt so flow has a direction across
        src = np.zeros((rows, cols), bool)
        src[1, 8:12] = True
        mask = np.ones_like(src)
        out = RO.runout(z, src, mask, res, angle_deg=11.0)
        reach = out["reach"]
        self.assertTrue(reach[14, 8:12].any())     # reaches the slope foot
        far = np.flatnonzero(reach.any(1)).max()
        # energy line at 11°: travel ≤ drop / tan(11°)
        max_cells = 15 * drop / np.tan(np.radians(11)) / res
        self.assertLessEqual(far, 1 + max_cells + 2)
        self.assertFalse(reach[:1].any() and not src[:1].any())
        self.assertGreaterEqual(np.nanmax(out["v"]), 0)


if __name__ == "__main__":
    unittest.main()
