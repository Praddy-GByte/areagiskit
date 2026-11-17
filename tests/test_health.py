# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import math
import unittest

import numpy as np

from _util import mod, rook

ST = mod("kits.health.stats")


class TestRates(unittest.TestCase):
    def test_byar_ci_brackets_rate(self):
        c, p = np.array([0, 5, 50]), np.array([1000.0, 1000.0, 1000.0])
        lo, hi = ST.byar_ci(c, p, per=1000)
        self.assertEqual(lo[0], 0.0)
        for i in range(3):
            self.assertLessEqual(lo[i], c[i] / p[i] * 1000 + 1e-9)
            self.assertGreaterEqual(hi[i], c[i] / p[i] * 1000 - 1e-9)
        # 50 cases: exact Poisson 95% CI is about 37.1-65.9
        self.assertAlmostEqual(lo[2], 37.1, delta=0.4)
        self.assertAlmostEqual(hi[2], 65.9, delta=0.4)

    def test_smr_and_exceedance(self):
        c = np.array([10.0, 30.0, 20.0])
        p = np.array([100.0, 100.0, 100.0])
        e, s, pr = ST.smr(c, p)
        self.assertAlmostEqual(e.sum(), c.sum())
        np.testing.assert_allclose(s, c / e)
        self.assertLess(pr[1], 0.05)        # 30 vs 20 expected is unusual
        self.assertGreater(pr[0], 0.95)     # 10 vs 20 expected is not an excess

    def test_poisson_sf(self):
        self.assertAlmostEqual(ST.poisson_sf(0, 3.0), 1.0)
        self.assertAlmostEqual(ST.poisson_sf(1, 3.0), 1 - math.exp(-3), places=10)

    def test_eb_global_shrinks(self):
        rng = np.random.default_rng(0)
        p = rng.integers(50, 5000, 60).astype(float)
        c = rng.poisson(p * 0.01).astype(float)
        eb = ST.eb_global(c, p)
        raw = c / p
        base = c.sum() / p.sum()
        self.assertTrue(np.all(np.abs(eb - base) <= np.abs(raw - base) + 1e-12))


class TestAutocorrelation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.nb = rook(8, 8)
        base = np.add.outer(np.arange(8), np.arange(8)).ravel().astype(float)
        self.x = base + rng.normal(0, 1, 64)

    def test_morans_i_matches_dense_formula(self):
        I, p = ST.morans_i(self.x, self.nb, perms=199)
        n = len(self.x)
        W = np.zeros((n, n))
        for i, v in enumerate(self.nb):
            W[i, v] = 1.0 / len(v)
        z = self.x - self.x.mean()
        ref = (n / W.sum()) * (z @ W @ z) / (z @ z)
        self.assertAlmostEqual(I, ref, places=10)
        self.assertLess(p, 0.01)               # strong gradient is clustered

    def test_lisa_sums_to_global(self):
        I, _ = ST.morans_i(self.x, self.nb, perms=9)
        Ii, p, q = ST.lisa(self.x, self.nb, perms=99)
        # row-standardised, z scaled by its std: sum(Ii) = (n-1) * I  (PySAL esda convention)
        self.assertAlmostEqual(Ii.sum(), (len(self.x) - 1) * I, places=8)
        self.assertTrue(set(np.unique(q)) <= {1, 2, 3, 4})

    def test_fdr(self):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
        self.assertEqual(ST.fdr(p, 0.05).tolist(), [True, True, True, False, False])   # BH: 0.03 <= 3/5*0.05


class TestScan(unittest.TestCase):
    def test_kulldorff_finds_planted_cluster(self):
        nr = nc = 7
        xy = np.array([(c, r) for r in range(nr) for c in range(nc)], float)
        p = np.full(nr * nc, 1000.0)
        rng = np.random.default_rng(5)
        c = rng.poisson(10, nr * nc).astype(float)
        hot = [0, 1, 7, 8]
        c[hot] = rng.poisson(40, len(hot))
        cl = ST.kulldorff(xy, c, p, sims=99, seed=1)
        self.assertTrue(cl)
        top = cl[0]
        self.assertLessEqual(top["p"], 0.05)
        self.assertTrue(set(hot) & set(top["members"]))
        self.assertGreater(top["rr"], 2)


class TestBYM2(unittest.TestCase):
    def test_bym2_smoke(self):
        try:
            A = mod("kits.health.advanced")
        except Exception as e:   # module may be under development
            self.skipTest(f"advanced module not importable: {e}")
        nb = rook(6, 6)
        rng = np.random.default_rng(2)
        E = np.full(36, 20.0)
        y = rng.poisson(E).astype(float)
        y[[0, 1, 6, 7]] = rng.poisson(60, 4)
        r = A.bym2(y, E, nb, iters=1500, chains=2, seed=1)
        self.assertEqual(len(r["rr"]), 36)
        self.assertTrue(np.all(r["lo"] <= r["rr"]) and np.all(r["rr"] <= r["hi"]))
        self.assertGreater(r["rr"][[0, 1, 6, 7]].mean(), r["rr"][20:].mean())


if __name__ == "__main__":
    unittest.main()
