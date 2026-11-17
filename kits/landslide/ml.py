# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Two more machine-learning susceptibility models common in the literature, in numpy only:
- Support Vector Machine with an RBF kernel, approximated with random Fourier features (Rahimi & Recht 2007) and fitted as a
  regularised least-squares SVM (Suykens & Vandewalle 1999); the decision value is the susceptibility score.
- Gradient-boosted decision trees for binary log-loss (Friedman 2001), shallow trees on quantile-binned factors
  (the histogram approach of LightGBM/XGBoost-hist)."""

import math

import numpy as np


# ------------------------------------------------------------------ SVM (RBF, random Fourier features)
def svm_model(Xtr, ytr, Xall, D=1000, lam=1e-4, seed=0, chunk=200_000):
    """Kernel width 1/n_factors, weak regularisation and standardised factors clipped at ±4 SD (distances and stream power are
    heavy-tailed); chosen on random hold-outs at Hiroshima, Sikkim and Gapyeong as the best balance across the three."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    a = np.clip((Xtr - mu) / sd, -4, 4)
    rng = np.random.default_rng(seed)
    gamma = 1.0 / a.shape[1]
    W = rng.normal(0, math.sqrt(2 * gamma), (a.shape[1], D))
    u = rng.uniform(0, 2 * math.pi, D)
    fa = math.sqrt(2 / D) * np.cos(a @ W + u)
    t = ytr * 2.0 - 1.0
    A = fa.T @ fa + lam * len(fa) * np.eye(D)
    beta = np.linalg.solve(A, fa.T @ t)
    out = np.empty(len(Xall), np.float32)
    for i in range(0, len(Xall), chunk):
        b = np.clip((Xall[i:i + chunk] - mu) / sd, -4, 4)
        out[i:i + chunk] = (math.sqrt(2 / D) * np.cos(b @ W + u)) @ beta
    return out


# ------------------------------------------------------------------ gradient boosting
class _Tree:
    """Depth-limited regression tree on binned features, fitted to gradients/hessians (Newton step leaves)."""

    def __init__(self, depth, min_leaf, lam):
        self.depth, self.min_leaf, self.lam = depth, min_leaf, lam
        self.feat, self.thr, self.left, self.right, self.value = [], [], [], [], []

    def _new(self):
        for lst in (self.feat, self.thr, self.left, self.right, self.value):
            lst.append(-1 if lst is not self.value else 0.0)
        return len(self.feat) - 1

    def fit(self, Xb, g, h, nbins):
        root = self._new()
        stack = [(root, np.arange(len(g)), 0)]
        while stack:
            node, idx, d = stack.pop()
            G, Hs = g[idx].sum(), h[idx].sum()
            self.value[node] = -G / (Hs + self.lam)
            if d >= self.depth or len(idx) < 2 * self.min_leaf:
                continue
            best = (0.0, -1, -1)
            parent = G * G / (Hs + self.lam)
            for f in range(Xb.shape[1]):
                xb = Xb[idx, f]
                gs = np.bincount(xb, g[idx], nbins).cumsum()
                hs = np.bincount(xb, h[idx], nbins).cumsum()
                cs = np.bincount(xb, None, nbins).cumsum()
                ok = (cs >= self.min_leaf) & (cs <= len(idx) - self.min_leaf)
                if not ok.any():
                    continue
                gl, hl = gs, hs
                gr, hr = G - gs, Hs - hs
                gain = gl * gl / (hl + self.lam) + gr * gr / (hr + self.lam) - parent
                gain[~ok] = -np.inf
                b = int(np.argmax(gain))
                if gain[b] > best[0]:
                    best = (float(gain[b]), f, b)
            if best[1] < 0:
                continue
            _, f, b = best
            m = Xb[idx, f] <= b
            L, Rn = self._new(), self._new()
            self.feat[node], self.thr[node], self.left[node], self.right[node] = f, b, L, Rn
            stack.append((L, idx[m], d + 1))
            stack.append((Rn, idx[~m], d + 1))
        self.feat, self.thr = np.array(self.feat), np.array(self.thr)
        self.left, self.right, self.value = np.array(self.left), np.array(self.right), np.array(self.value)
        return self

    def predict(self, Xb):
        node = np.zeros(len(Xb), np.int64)
        for _ in range(self.depth + 1):
            f = self.feat[node]
            leaf = f < 0
            if leaf.all():
                break
            go_left = np.zeros(len(Xb), bool)
            nl = ~leaf
            go_left[nl] = Xb[np.flatnonzero(nl), f[nl]] <= self.thr[node[nl]]
            node = np.where(leaf, node, np.where(go_left, self.left[node], self.right[node]))
        return self.value[node]


class GradientBoosting:
    def __init__(self, n_trees=150, lr=0.1, depth=3, min_leaf=20, lam=1.0, nbins=32, subsample=0.8, seed=0):
        self.n_trees, self.lr, self.depth, self.min_leaf, self.lam = n_trees, lr, depth, min_leaf, lam
        self.nbins, self.subsample, self.seed = nbins, subsample, seed

    def _bin(self, X):
        Xb = np.empty(X.shape, np.int64)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return np.minimum(Xb, self.nbins - 1)

    def fit(self, X, y):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.nbins + 1)[1:-1])) for j in range(X.shape[1])]
        Xb = self._bin(X)
        p0 = min(max(y.mean(), 1e-3), 1 - 1e-3)
        self.base = math.log(p0 / (1 - p0))
        F = np.full(len(y), self.base)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            p = 1 / (1 + np.exp(-F))
            g, h = p - y, np.maximum(p * (1 - p), 1e-6)
            sel = rng.random(len(y)) < self.subsample
            t = _Tree(self.depth, self.min_leaf, self.lam).fit(Xb[sel], g[sel], h[sel], self.nbins)
            F += self.lr * t.predict(Xb)
            self.trees.append(t)
        return self

    def decision(self, X, chunk=200_000):
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), chunk):
            Xb = self._bin(X[i:i + chunk])
            F = np.full(len(Xb), self.base)
            for t in self.trees:
                F += self.lr * t.predict(Xb)
            out[i:i + chunk] = F
        return out

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.decision(X)))


def gbm_model(Xtr, ytr, Xall, seed=0):
    m = GradientBoosting(seed=seed).fit(Xtr, ytr.astype(float))
    return m.predict_proba(Xall), m
