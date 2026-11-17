# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""A small, fast Random Forest in numpy (no scikit-learn needed inside QGIS).

Features are binned to 32 quantile bins, so every split search is a bincount.
Trees are grown on bootstrap samples with sqrt(features) candidates per node
(Breiman 2001). Prediction is vectorised over all pixels.
"""

import numpy as np

NB = 32


class RandomForest:
    def __init__(self, n_trees=40, max_depth=14, min_leaf=3, seed=0):
        self.n_trees, self.max_depth, self.min_leaf = n_trees, max_depth, min_leaf
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ binning
    def _fit_bins(self, X):
        self.edges = []
        for f in range(X.shape[1]):
            q = np.unique(np.nanquantile(X[:, f], np.linspace(0, 1, NB + 1)[1:-1]))
            self.edges.append(q)

    def _bin(self, X):
        out = np.empty(X.shape, np.uint8)
        for f in range(X.shape[1]):
            out[:, f] = np.searchsorted(self.edges[f], X[:, f], side="right")
        return out

    # ------------------------------------------------------------ training
    def fit(self, X, y, classes):
        self.classes = np.asarray(classes)
        K = len(classes)
        yi = np.searchsorted(self.classes, y)
        self._fit_bins(X)
        Xb = self._bin(X)
        n, F = Xb.shape
        mtry = max(1, int(np.sqrt(F)))
        self.trees = []
        imp = np.zeros(F)
        oob_votes = np.zeros((n, K))
        for _ in range(self.n_trees):
            boot = self.rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), boot)
            tree, tim = self._grow(Xb, yi, boot, K, F, mtry)
            self.trees.append(tree)
            imp += tim
            if oob.size:
                oob_votes[oob] += self._predict_tree(tree, Xb[oob])
        self.importance = imp / imp.sum() if imp.sum() else imp
        seen = oob_votes.sum(1) > 0
        self.oob_accuracy = float((oob_votes[seen].argmax(1) == yi[seen]).mean()) if seen.any() else None
        return self

    def _grow(self, Xb, yi, idx, K, F, mtry):
        feat, thr, left, right, prob = [], [], [], [], []
        imp = np.zeros(F)

        def new_node():
            feat.append(-1)
            thr.append(0)
            left.append(-1)
            right.append(-1)
            prob.append(None)
            return len(feat) - 1

        root = new_node()
        stack = [(root, idx, 0)]
        while stack:
            node, ids, depth = stack.pop()
            counts = np.bincount(yi[ids], minlength=K).astype(np.float64)
            nn = counts.sum()
            prob[node] = counts / nn
            if depth >= self.max_depth or nn < 2 * self.min_leaf or counts.max() == nn:
                continue
            g_parent = 1 - ((counts / nn) ** 2).sum()
            best = (0.0, -1, -1)
            for f in self.rng.choice(F, mtry, replace=False):
                h = np.bincount(Xb[ids, f].astype(np.int64) * K + yi[ids], minlength=(NB + 1) * K).reshape(NB + 1, K)
                cl = np.cumsum(h, axis=0)[:-1]
                nl = cl.sum(1)
                cr = counts - cl
                nr = nn - nl
                ok = (nl >= self.min_leaf) & (nr >= self.min_leaf)
                if not ok.any():
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    gl = 1 - ((cl / nl[:, None]) ** 2).sum(1)
                    gr = 1 - ((cr / nr[:, None]) ** 2).sum(1)
                gain = g_parent - (nl * gl + nr * gr) / nn
                gain[~ok] = -1
                t = int(np.argmax(gain))
                if gain[t] > best[0]:
                    best = (float(gain[t]), int(f), t)
            if best[1] < 0:
                continue
            f, t = best[1], best[2]
            imp[f] += best[0] * nn
            mask = Xb[ids, f] <= t
            li, ri = ids[mask], ids[~mask]
            feat[node], thr[node] = f, t
            ln, rn = new_node(), new_node()
            left[node], right[node] = ln, rn
            stack.append((ln, li, depth + 1))
            stack.append((rn, ri, depth + 1))
        tree = (np.array(feat), np.array(thr, np.uint8), np.array(left), np.array(right),
                np.stack([p if p is not None else np.full(K, 1 / K) for p in prob]))
        return tree, imp

    def _predict_tree(self, tree, Xb):
        feat, thr, left, right, prob = tree
        node = np.zeros(len(Xb), np.int64)
        active = feat[node] >= 0
        while active.any():
            ia = np.flatnonzero(active)
            nd = node[ia]
            go_left = Xb[ia, feat[nd]] <= thr[nd]
            node[ia] = np.where(go_left, left[nd], right[nd])
            active[ia] = feat[node[ia]] >= 0
        return prob[node]

    def predict_proba(self, X, chunk=400_000):
        out = np.zeros((len(X), len(self.classes)), np.float32)
        for s in range(0, len(X), chunk):
            Xb = self._bin(X[s:s + chunk])
            acc = np.zeros((len(Xb), len(self.classes)))
            for t in self.trees:
                acc += self._predict_tree(t, Xb)
            out[s:s + chunk] = acc / len(self.trees)
        return out

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes[p.argmax(1)], p.max(1)
