# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Landslide additions: SVM and gradient boosting, lithology grouping, Poisson hazard, guidance."""
import types
import unittest

import numpy as np

from _util import mod


def _auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(3000, 4))
        # non-linear truth: a ring (needs a kernel or trees)
        self.y = ((self.X[:, 0] ** 2 + self.X[:, 1] ** 2) > 1.4).astype(float)
        self.tr = rng.random(3000) < 0.7

    def test_svm_learns_nonlinear(self):
        ML = mod("kits.landslide.ml")
        s = ML.svm_model(self.X[self.tr], self.y[self.tr], self.X[~self.tr])
        self.assertGreater(_auc(self.y[~self.tr], s), 0.9)

    def test_gbm_learns_nonlinear(self):
        ML = mod("kits.landslide.ml")
        p, m = ML.gbm_model(self.X[self.tr], self.y[self.tr], self.X[~self.tr])
        self.assertGreater(_auc(self.y[~self.tr], p), 0.9)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


class TestGeology(unittest.TestCase):
    def test_grouping(self):
        GE = mod("kits.landslide.geology")
        self.assertEqual(GE.group_of("medium fine biotite granite"), 6)
        self.assertEqual(GE.group_of("gravel sand silt"), 1)
        self.assertEqual(GE.group_of("dacite & rhyolite pyroclastic rocks"), 3)
        self.assertEqual(GE.group_of("crystalline metamorphic rocks"), 8)
        self.assertEqual(GE.group_of("limestone and dolomite"), 5)
        self.assertEqual(GE.group_of(""), 9)


class TestHazard(unittest.TestCase):
    def test_poisson_rates_rise_with_class(self):
        HZ = mod("kits.landslide.hazard")
        AOI = mod("core.aoi")
        grid = AOI.Grid(AOI.AOI(bbox=(10.0, 45.0, 10.03, 45.03), name="t"), 30)
        shape = grid.shape
        nb = shape[1] // 5
        cls = np.clip(np.arange(shape[1]) // nb + 1, 1, 5).astype(np.uint8)[None, :].repeat(shape[0], 0)
        mask = grid.mask
        cls[~mask] = 0
        act = {}
        rng = np.random.default_rng(1)
        for y in (2020, 2021, 2022, 2023):
            a = np.zeros(shape, bool)
            for c, n in ((3, 1), (4, 3), (5, 8)):          # more scars in more susceptible bands
                cols = np.flatnonzero(cls[0] == c)
                for _ in range(n):
                    r0, c0 = rng.integers(5, shape[0] - 8), rng.choice(cols[2:-3])
                    a[r0:r0 + 2, c0:c0 + 2] = True
            act[y] = a
        out = HZ.temporal(act, cls, mask, grid)
        rates = {r["cls"]: r["rate"] for r in out["rows"]}
        self.assertEqual(rates[1], 0)
        self.assertLess(rates[3], rates[5])
        r5 = [r for r in out["rows"] if r["cls"] == 5][0]
        self.assertAlmostEqual(r5["p"][10], 1 - np.exp(-r5["rate"] * 10), places=9)
        self.assertTrue(set(np.unique(out["map"][mask])) <= {1, 2, 3, 4, 5})

    def test_guidance_minimal(self):
        HZ = mod("kits.landslide.hazard")
        g = types.SimpleNamespace(mask=np.ones((10, 10), bool), px_km2=0.0009)
        cls = np.full((10, 10), 5, np.uint8)
        R = dict(g=g, sus=dict(cls=cls), runout=None, fs_cls=np.full((10, 10), 2, np.uint8), rain={}, exp={})
        pts = HZ.guidance(R)
        self.assertTrue(any("very high susceptibility" in p for p in pts))
        self.assertTrue(any("rain-gauge" in p for p in pts))


if __name__ == "__main__":
    unittest.main()
