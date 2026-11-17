# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import unittest

import numpy as np

from _util import mod


class TestUrbanExtra(unittest.TestCase):
    def setUp(self):
        self.SF = mod("kits.urban.surface")
        self.CM = mod("kits.urban.climate")

    def test_liang_coefficients(self):
        # a grey surface reflecting 0.20 in every band: 0.20 × (sum of weights) − 0.0018
        a = self.SF.liang_albedo(*(np.full(3, 0.2),) * 5)
        self.assertAlmostEqual(float(a[0]), 0.2 * 1.016 - 0.0018, places=6)
        # weights as published by Liang (2001)
        self.assertEqual((self.SF.LIANG["blue"], self.SF.LIANG["red"], self.SF.LIANG["nir08"]), (0.356, 0.130, 0.373))

    def test_wbgt_bom_example(self):
        # Bureau of Meteorology formula by hand: T 30 °C, RH 50% -> e = 21.14 hPa, WBGT = 29.26 °C
        e = float(self.CM.vapour_pressure(30.0, 50.0))
        self.assertAlmostEqual(e, 21.14, delta=0.02)
        self.assertAlmostEqual(float(self.CM.wbgt_simple(30.0, 50.0)), 29.26, delta=0.02)
        # hotter and more humid air is more stressful
        self.assertGreater(self.CM.wbgt_simple(35, 70), self.CM.wbgt_simple(35, 30))

    def test_apparent_temperature(self):
        # AT = T + 0.33 e − 0.70 ws − 4.00 ; T 30, RH 50 (e 21.14), wind 2 m/s -> 31.58
        self.assertAlmostEqual(float(self.CM.apparent_temp(30.0, 50.0, 2.0)), 30 + 0.33 * 21.14 - 1.4 - 4.0, delta=0.02)

    def test_slope_se(self):
        x = np.arange(50.0)
        y = 2.0 * x + 1 + np.sin(x)
        b, se = self.SF.slope_se(x, y)
        self.assertAlmostEqual(b, 2.0, delta=0.05)
        self.assertGreater(se, 0)


if __name__ == "__main__":
    unittest.main()
