# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Drivers of urban growth, ecosystem service value, LST relations and future simulation."""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------- drivers


def logistic(X, y, iters=50):
    """Logistic regression by IRLS. X without intercept (standardised). Returns beta, se."""
    Xd = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        z = np.clip(Xd @ b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = Xd.T @ (Xd * W[:, None]) + 1e-6 * np.eye(Xd.shape[1])
        g = Xd.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    cov = np.linalg.inv(H)
    return b, np.sqrt(np.diag(cov))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def density(mask, radius):
    k = 2 * radius + 1
    m = np.pad(mask.astype(np.float32), radius)
    c = np.cumsum(np.cumsum(m, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def drivers(built1, built2, valid, layers, seed=3, n=6000):
    """Which factors explain where new built-up land appeared between two dates.
    layers: dict name -> array. Returns table rows, AUC, and a suitability map."""
    cand = valid & ~built1
    yv = (built2 & cand)
    pos = np.flatnonzero(yv.ravel())
    neg = np.flatnonzero((cand & ~built2).ravel())
    if pos.size < 50 or neg.size < 50:
        return None
    rng = np.random.default_rng(seed)
    pos = rng.choice(pos, min(n, pos.size), replace=False)
    neg = rng.choice(neg, min(n, neg.size), replace=False)
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    names = list(layers)
    X = np.column_stack([np.nan_to_num(layers[k].ravel()[idx]) for k in names])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.7 * len(y))], perm[int(0.7 * len(y)):]
    b, se = logistic(Xs[tr], y[tr])
    s_te = np.column_stack([np.ones(len(te)), Xs[te]]) @ b
    a = auc(y[te], s_te)
    rows = []
    for i, k in enumerate(names):
        z = b[i + 1] / se[i + 1] if se[i + 1] > 0 else 0
        p = math.erfc(abs(z) / math.sqrt(2))
        rows.append(dict(name=k, coef=float(b[i + 1]), odds=float(math.exp(b[i + 1])), z=float(z), p=p))
    full = np.column_stack([np.nan_to_num(layers[k].ravel()) for k in names])
    suit = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(full)), (full - mu) / sd]) @ b, -30, 30)))
    return dict(rows=rows, auc=a, suitability=suit.reshape(built1.shape).astype(np.float32), n=len(y))


# --------------------------------------------------------------------- ecosystem services

# Costanza et al. (1997) global mean values, US$ per ha per year (1994 US$)
ESV_COEF = {1: ("Lakes/rivers", 8498), 2: ("Urban", 0), 3: ("Forest", 969), 4: ("Cropland", 92),
            5: ("Grass/rangelands", 232), 6: ("Barren", 0), 7: ("Wetlands", 14785), 8: ("Ice/rock", 0)}
TROPICAL_FOREST = 2007
TEMPERATE_FOREST = 302


def esv(areas_km2, lat):
    coef = {k: v[1] for k, v in ESV_COEF.items()}
    coef[3] = TROPICAL_FOREST if abs(lat) <= 23.5 else TEMPERATE_FOREST
    by = {k: areas_km2.get(k, 0) * 100 * coef[k] for k in coef}  # km² → ha
    return sum(by.values()), by, coef


def esv_sensitivity(areas_km2, lat):
    """Coefficient of sensitivity for ±50% change of each class's value coefficient."""
    base, _, coef = esv(areas_km2, lat)
    out = {}
    for k in coef:
        if coef[k] == 0 or areas_km2.get(k, 0) == 0 or base == 0:
            continue
        adj = base + areas_km2[k] * 100 * coef[k] * 0.5
        out[k] = ((adj - base) / base) / 0.5
    return out


# --------------------------------------------------------------------- LST


def pearson(a, b, mask, n=30000, seed=5):
    v = mask & np.isfinite(a) & np.isfinite(b)
    idx = np.flatnonzero(v.ravel())
    if idx.size < 100:
        return None
    idx = np.random.default_rng(seed).choice(idx, min(n, idx.size), replace=False)
    x, y = a.ravel()[idx], b.ravel()[idx]
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------- Markov + CA


def markov(T):
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.nan_to_num(T / T.sum(1, keepdims=True))
    for i in range(len(P)):
        if P[i].sum() == 0:
            P[i, i] = 1
    return P


def ca_allocate(cur, P, classes, suit_built, steps=1, seed=11):
    """Cellular-automata allocation of Markov transition quantities.
    Transitions are placed where the target class is most suitable: logistic suitability
    for built-up, neighbourhood share of the target class for the others."""
    rng = np.random.default_rng(seed)
    out = cur.copy()
    Pn = np.linalg.matrix_power(P, steps)
    idx = {c: i for i, c in enumerate(classes)}
    counts = {c: int((cur == c).sum()) for c in classes}
    demands = []
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i != j:
                n = int(round(counts[ci] * Pn[i, j]))
                if n > 0:
                    demands.append((n, ci, cj))
    demands.sort(reverse=True)
    taken = np.zeros(cur.shape, bool)
    neigh = {cj: density(cur == cj, 2) for cj in classes}
    noise = rng.random(cur.shape).astype(np.float32) * 0.05
    for n, ci, cj in demands:
        cand = (cur == ci) & ~taken
        if not cand.any():
            continue
        score = neigh[cj] + noise
        if cj == 2 and suit_built is not None:
            score = suit_built * (0.5 + neigh[cj]) + noise
        flat = np.flatnonzero(cand.ravel())
        s = score.ravel()[flat]
        k = min(n, flat.size)
        pick = flat[np.argpartition(-s, k - 1)[:k]]
        out.ravel()[pick] = cj
        taken.ravel()[pick] = True
    return out


def simulation_skill(start, actual, sim, classes):
    """Overall agreement, kappa, figure of merit and quantity/allocation disagreement."""
    v = (start > 0) & (actual > 0) & (sim > 0)
    a, s, o = actual[v], sim[v], start[v]
    oa = float((a == s).mean())
    k = len(classes)
    lut = np.full(256, -1, np.int64)
    for i, c in enumerate(classes):
        lut[c] = i
    m = np.bincount(lut[s] * k + lut[a], minlength=k * k).reshape(k, k).astype(float) / v.sum()
    pe = (m.sum(0) * m.sum(1)).sum()
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0
    Q = 0.5 * np.abs(m.sum(1) - m.sum(0)).sum()
    A = (1 - oa) - Q
    obs_ch, sim_ch = a != o, s != o
    hits = (obs_ch & sim_ch & (a == s)).sum()
    wrong = (obs_ch & sim_ch & (a != s)).sum()
    misses = (obs_ch & ~sim_ch).sum()
    false = (~obs_ch & sim_ch).sum()
    fom = hits / max(hits + wrong + misses + false, 1)
    # null model: predict no change at all
    null_oa = float((a == o).mean())
    return dict(oa=oa, kappa=float(kappa), Q=float(Q), A=float(max(A, 0)), fom=float(fom), null_oa=null_oa)
