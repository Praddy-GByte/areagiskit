# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import unittest

import numpy as np

from _util import mod, rook

HX = mod("kits.health.extra")


class TestGWPR(unittest.TestCase):
    def test_recovers_varying_coefficient(self):
        rng = np.random.default_rng(1)
        n = 120
        xy = rng.uniform(0, 100, (n, 2)) * 1000
        x1 = rng.normal(size=n)
        b1 = 0.1 + 0.6 * xy[:, 0] / 1e5
        pop = rng.uniform(5000, 50000, n)
        y = rng.poisson(pop * 1e-3 * np.exp(b1 * x1)).astype(float)
        f = HX.gwpr(y, x1[:, None], np.log(pop * 1e-3), xy)
        self.assertGreater(np.corrcoef(f["beta"][:, 1], b1)[0, 1], 0.85)
        self.assertLess(f["aicc"], f["global_aicc"])

    def test_constant_effect_prefers_wide_kernel(self):
        rng = np.random.default_rng(2)
        n = 100
        xy = rng.uniform(0, 100, (n, 2)) * 1000
        x1 = rng.normal(size=n)
        pop = rng.uniform(5000, 50000, n)
        y = rng.poisson(pop * 1e-3 * np.exp(0.4 * x1)).astype(float)
        f = HX.gwpr(y, x1[:, None], np.log(pop * 1e-3), xy)
        self.assertGreaterEqual(f["aicc"], f["global_aicc"] - 2)
        self.assertAlmostEqual(float(np.median(f["beta"][:, 1])), 0.4, delta=0.1)


class TestKnox(unittest.TestCase):
    def test_points_cluster_detected(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(0, 50, (150, 2))
        t = rng.uniform(0, 365, 150)
        xy[:25] = rng.normal(25, 1.0, (25, 2))
        t[:25] = rng.uniform(100, 110, 25)
        r = HX.knox_points(xy, t, 3, 14, sims=199)
        self.assertLessEqual(r["p"], 0.05)
        self.assertGreater(r["ratio"], 1.5)

    def test_areal_null_and_cluster(self):
        rng = np.random.default_rng(4)
        nb = rook(8, 8)
        pop = rng.uniform(1000, 5000, 64)
        C = rng.poisson(pop[:, None] * 2e-3 * np.ones(6)).astype(float)
        self.assertGreater(HX.knox_areal(C, nb, sims=199)["p"], 0.01)
        cl = [27, 28, 35, 36]
        C[np.ix_(cl, [3, 4])] = rng.poisson(pop[cl][:, None] * 2e-3 * 3)
        self.assertLessEqual(HX.knox_areal(C, nb, sims=199)["p"], 0.05)


class TestKDE(unittest.TestCase):
    def test_planted_excess_found(self):
        rng = np.random.default_rng(5)
        mask = np.ones((60, 60), bool)
        pop = np.full((60, 60), 100.0)
        prob = pop.ravel() / pop.sum()
        rc = np.zeros(3600)
        np.add.at(rc, rng.choice(3600, 300, p=prob), 1)
        rc = rc.reshape(60, 60)
        rc[28:33, 28:33] += 6           # 150 extra cases in a 5x5 block
        r = HX.kde_risk(rc, pop, mask, 100.0, 300.0, sims=39)
        self.assertEqual(r["cls"][30, 30], 1)
        self.assertLess((r["cls"] == 1).mean(), 0.1)


class Test2SFCA(unittest.TestCase):
    def test_hand_example(self):
        A, R = HX.two_step_fca([[0, 0], [10, 0]], [100, 300], [[0, 0], [10, 0]], [10, 10], 15, decay="binary")
        np.testing.assert_allclose(R, [0.025, 0.025])
        np.testing.assert_allclose(A, [0.05, 0.05])
        A, R = HX.two_step_fca([[0, 0], [10, 0]], [100, 300], [[0, 0], [10, 0]], [10, 10], 5, decay="binary")
        np.testing.assert_allclose(R, [0.1, 10 / 300])
        np.testing.assert_allclose(A, [0.1, 10 / 300])


if __name__ == "__main__":
    unittest.main()
