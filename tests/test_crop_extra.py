# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Crop water and drought methods: incomplete gamma, SPI on synthetic gamma rainfall, VHI arithmetic, dry spells, irrigation."""
import datetime as dt
import math
import unittest

import numpy as np

from _util import mod


class TestSPI(unittest.TestCase):
    def setUp(self):
        self.WA = mod("kits.crop.water")

    def test_gamma_cdf_exponential(self):
        # alpha = 1 is the exponential distribution: P = 1 - exp(-x/beta)
        for x in (0.1, 1.0, 3.0, 10.0):
            self.assertAlmostEqual(self.WA.gamma_cdf(x, 1.0, 2.0), 1 - math.exp(-x / 2.0), places=9)

    def test_gamma_cdf_chi2(self):
        # chi-square with 4 df = gamma(2, 2); P(X <= 9.488) = 0.95
        self.assertAlmostEqual(self.WA.gamma_cdf(9.487729, 2.0, 2.0), 0.95, places=5)

    def test_spi_is_standard_normal_on_gamma_rain(self):
        rng = np.random.default_rng(5)
        monthly = {}
        for y in range(1951, 2021):                       # 70 years
            for m in range(1, 13):
                monthly[(y, m)] = float(rng.gamma(2.0 + m / 6, 30.0))
        s = self.WA.spi(monthly, scales=(1, 3))
        v1 = np.array(list(s[1].values()))
        self.assertGreater(len(v1), 800)
        self.assertAlmostEqual(float(v1.mean()), 0.0, delta=0.08)
        self.assertAlmostEqual(float(v1.std()), 1.0, delta=0.08)
        self.assertAlmostEqual(float((v1 <= -1).mean()), 0.159, delta=0.03)   # Φ(-1)
        v3 = np.array(list(s[3].values()))
        self.assertAlmostEqual(float(v3.std()), 1.0, delta=0.1)

    def test_spi_mixed_distribution_with_dry_months(self):
        rng = np.random.default_rng(7)
        monthly = {(y, m): (0.0 if (m in (12, 1) and rng.random() < 0.4) else float(rng.gamma(1.5, 20.0)))
                   for y in range(1960, 2021) for m in range(1, 13)}
        s = self.WA.spi(monthly, scales=(1,))[1]
        jan = np.array([v for (y, m), v in s.items() if m == 1])
        self.assertTrue(np.isfinite(jan).all())
        zero = [v for (y, m), v in s.items() if m == 1 and monthly[(y, m)] == 0]
        self.assertTrue(zero and max(zero) < 0)             # a rainless month is below normal
        self.assertEqual(self.WA.spi_class(-1.6), "severe drought")


class TestVHI(unittest.TestCase):
    def setUp(self):
        self.WA = mod("kits.crop.water")

    def test_vhi_arithmetic(self):
        ys = list(range(2001, 2011))
        nd = {y: np.full((2, 2), 0.3 + 0.04 * i, np.float32) for i, y in enumerate(ys)}     # greener every year
        ls = {y: np.full((2, 2), 40.0 - 1.0 * i, np.float32) for i, y in enumerate(ys)}     # cooler every year
        r = self.WA.vhi(nd, ls, np.ones((2, 2), bool))
        self.assertAlmostEqual(r["rows"][2001]["vci"], 0.0, places=4)
        self.assertAlmostEqual(r["rows"][2001]["tci"], 0.0, places=4)
        self.assertAlmostEqual(r["rows"][2010]["vhi"], 100.0, places=4)
        self.assertAlmostEqual(r["rows"][2005]["vhi"], 0.5 * 4 / 9 * 100 * 2, places=3)
        self.assertEqual(r["rows"][2001]["drought_pct"], 100.0)
        self.assertEqual(self.WA.vhi_class(35), "mild drought")


class TestIrrigation(unittest.TestCase):
    def setUp(self):
        self.WA = mod("kits.crop.water")

    def test_dry_spell_and_green_crop(self):
        start = dt.date(2021, 1, 1)
        days = [start - dt.timedelta(days=90) + dt.timedelta(days=i) for i in range(460)]
        rain = np.array([5.0 if d.month < 5 else 0.0 for d in days])        # wet until April, then no rain
        centers = [start + dt.timedelta(days=10 * i + 5) for i in range(36)]
        S = np.full((36, 2, 3), 0.3, np.float32)
        S[15:24, 0, :] = 0.8                                                  # row 0 green in summer (irrigated)
        crop = np.ones((2, 3), bool)
        cls, info = self.WA.irrigated(S, crop, centers, days, rain)
        self.assertGreater(info["n_dry"], 5)
        self.assertTrue((cls[0] == 1).all() and (cls[1] == 2).all())
        # humid year: no dry spell -> not determinable, except paddy rice
        rice = np.zeros((2, 3), bool)
        rice[1, 0] = True
        cls2, info2 = self.WA.irrigated(S, crop, centers, days, np.full(len(days), 4.0), rice=rice)
        self.assertEqual(info2["n_dry"], 0)
        self.assertEqual(int(cls2[1, 0]), 1)
        self.assertEqual(int(cls2[0, 0]), 3)


if __name__ == "__main__":
    unittest.main()
