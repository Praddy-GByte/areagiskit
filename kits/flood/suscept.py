# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Flood susceptibility: conditioning factors, automatic flood inventory, four models, validation.

Models: Frequency Ratio (bivariate statistics), AHP (expert weights, independent of the inventory),
Logistic Regression (multivariate statistics) and Random Forest (machine learning), plus their ensemble.
"""

import numpy as np

from ...core.raster import neighbours
from ..lulc.models import auc, logistic
from ..lulc.rf import RandomForest

# name, unit, direction of AHP score (+1 higher value = more susceptible), AHP importance (1-9)
FACTORS = [
    ("hand", "Height above nearest drainage", "m", -1, 9),
    ("dist_drain", "Distance to drainage", "m", -1, 8),
    ("elev", "Elevation", "m", -1, 7),
    ("dist_water", "Distance to permanent water", "m", -1, 6),
    ("slope", "Slope", "°", -1, 6),
    ("dd", "Drainage density", "km/km²", +1, 5),
    ("tpi", "Topographic position (TPI)", "m", -1, 4),
    ("lc", "Land cover", "class", 0, 4),
    ("clay", "Soil clay content", "%", +1, 3),
    ("curv", "Plan curvature (concavity)", "1/100 m", -1, 2),
    ("built", "Built-up surface share", "%", +1, 2),
]
LC_SCORE = {1: 5, 2: 1, 4: 5, 5: 4, 7: 4, 8: 3, 9: 1, 10: 3, 11: 3}  # IO classes: water, trees, flooded veg., crops, built, bare, snow, cloud, rangeland
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48}


def _box_mean(a, r):
    from .depth import _box_mean as bm
    return bm(a, r)


def factors(grid, dem, slope, hand, occ, lc, built, clay):
    """Dict key -> array (float), only those that could be computed."""
    F = {}
    F["elev"] = dem
    F["slope"] = slope
    if hand is not None:
        F["hand"] = hand
        drain = np.nan_to_num(hand, nan=99) <= 1.0
        if drain.any():
            F["dist_drain"] = grid.distance_to(drain)
            r = max(1, int(500 / grid.res))
            F["dd"] = _box_mean(drain.astype(np.float32), r) / grid.res * 1000.0  # km of channel per km² (approx.)
    if occ is not None:
        perm = np.nan_to_num(occ, nan=0) >= 75
        if perm.any():
            F["dist_water"] = grid.distance_to(perm)
    r1 = max(1, int(500 / grid.res))
    F["tpi"] = dem - _box_mean(dem, r1)
    z = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    lap = (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z) / (grid.res ** 2)
    F["curv"] = -lap * 1e4  # positive = concave hollow
    if lc is not None:
        F["lc"] = lc.astype(np.float32)
    if built is not None:
        F["built"] = np.nan_to_num(built) * 100
    if clay is not None:
        F["clay"] = clay
    for k in F:
        F[k] = np.where(grid.mask, F[k], np.nan).astype(np.float32)
    return F


def inventory(flood_n, obs_n, perm, excluded, mask, min_obs=20):
    """Flood (1) where GFM saw flood at least once; non-flood (0) where never flooded in many looks."""
    ok = mask & ~perm & ~excluded
    pos = ok & (flood_n >= 1)
    neg = ok & (flood_n == 0) & (obs_n >= min_obs)
    return pos, neg


def sample(pos, neg, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    ip, ineg = np.flatnonzero(pos.ravel()), np.flatnonzero(neg.ravel())
    k = min(n, ip.size, ineg.size)
    if k < 50:
        return None
    sp, sn = rng.choice(ip, k, replace=False), rng.choice(ineg, k, replace=False)
    idx = np.concatenate([sp, sn])
    y = np.concatenate([np.ones(k, np.int8), np.zeros(k, np.int8)])
    perm = rng.permutation(idx.size)
    return idx[perm], y[perm]


# ------------------------------------------------------------------ frequency ratio
def quantile_bins(a, mask, k=8):
    v = a[mask & np.isfinite(a)]
    if v.size == 0:
        return None
    e = np.unique(np.nanpercentile(v, np.linspace(0, 100, k + 1)))
    if e.size < 3:
        return None
    e[0], e[-1] = -np.inf, np.inf
    return e


def classes_of(key, a, mask):
    if key == "lc":
        return np.nan_to_num(a, nan=0).astype(np.int32), None
    e = quantile_bins(a, mask)
    if e is None:
        return None, None
    c = np.digitize(np.nan_to_num(a, nan=np.nanmedian(a)), e[1:-1]).astype(np.int32) + 1
    return c, e


def frequency_ratio(F, idx_tr, y_tr, mask):
    """Returns FR tables, FR-encoded layers and the FR susceptibility index (sum of FR)."""
    tables, enc = {}, {}
    pos_idx = idx_tr[y_tr == 1]
    for k, a in F.items():
        c, e = classes_of(k, a, mask)
        if c is None:
            continue
        cl = c.ravel()
        area = np.bincount(cl[mask.ravel()], minlength=cl.max() + 1).astype(float)
        fl = np.bincount(cl[pos_idx], minlength=cl.max() + 1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = np.where(area > 0, (fl / max(fl.sum(), 1)) / (area / area.sum()), 0)
        rows = []
        for i in range(1, len(area)):
            if area[i] == 0:
                continue
            if e is not None:
                lo, hi = e[i - 1], e[i]
                lab = (f"< {hi:,.1f}" if not np.isfinite(lo) else f"≥ {lo:,.1f}" if not np.isfinite(hi)
                       else f"{lo:,.1f} – {hi:,.1f}")
            else:
                lab = str(i)
            rows.append(dict(cls=i, label=lab, area=area[i] / area.sum() * 100, flood=fl[i] / max(fl.sum(), 1) * 100, fr=float(fr[i])))
        tables[k] = rows
        enc[k] = np.where(mask, fr[c], np.nan).astype(np.float32)
    lsi = np.nansum(np.stack(list(enc.values())), axis=0)
    lsi[~mask] = np.nan
    return tables, enc, lsi


# ------------------------------------------------------------------ AHP
def ahp_weights(keys):
    imp = {k: s for k, _, _, _, s in FACTORS}
    s = np.array([imp[k] for k in keys], float)
    n = len(s)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            d = s[i] - s[j]
            A[i, j] = d + 1 if d >= 0 else 1 / (1 - d)
    vals, vecs = np.linalg.eig(A)
    m = np.argmax(vals.real)
    w = np.abs(vecs[:, m].real)
    w /= w.sum()
    lam = vals[m].real
    ci = (lam - n) / (n - 1) if n > 1 else 0
    cr = ci / RI.get(n, 1.49) if RI.get(n, 0) else 0
    return dict(zip(keys, w)), A, float(cr), float(lam)


def ahp_score(k, a, mask):
    """Rate a factor 1 (low susceptibility) – 5 (high) with fixed, literature-style rules."""
    v = np.nan_to_num(a, nan=np.nanmedian(a) if np.isfinite(a).any() else 0)
    if k == "hand":
        s = np.select([v < 2, v < 5, v < 10, v < 20], [5, 4, 3, 2], 1)
    elif k in ("dist_drain",):
        s = np.select([v < 100, v < 250, v < 500, v < 1000], [5, 4, 3, 2], 1)
    elif k == "dist_water":
        s = np.select([v < 250, v < 500, v < 1000, v < 2500], [5, 4, 3, 2], 1)
    elif k == "slope":
        s = np.select([v < 2, v < 5, v < 10, v < 20], [5, 4, 3, 2], 1)
    elif k == "clay":
        s = np.select([v >= 40, v >= 30, v >= 20, v >= 10], [5, 4, 3, 2], 1)
    elif k == "built":
        s = np.select([v >= 50, v >= 30, v >= 15, v >= 5], [5, 4, 3, 2], 1)
    elif k == "lc":
        s = np.zeros(v.shape)
        for c, sc in LC_SCORE.items():
            s[v == c] = sc
        s[s == 0] = 3
    else:  # relative factors: quintiles within the area
        q = np.nanpercentile(a[mask], [20, 40, 60, 80])
        s = np.digitize(v, q) + 1
        direction = {d[0]: d[3] for d in FACTORS}[k]
        if direction < 0:
            s = 6 - s
    return np.where(mask, s, np.nan).astype(np.float32)


def ahp(F, mask):
    keys = [k for k, *_ in FACTORS if k in F]
    w, A, cr, lam = ahp_weights(keys)
    idx = np.zeros(mask.shape, np.float32)
    for k in keys:
        idx += w[k] * np.nan_to_num(ahp_score(k, F[k], mask))
    idx[~mask] = np.nan
    return idx, w, A, cr, lam, keys


# ------------------------------------------------------------------ LR and RF
def design(F, enc, keys, idx):
    """Model matrix: continuous factors as values, land cover FR-encoded."""
    cols = []
    for k in keys:
        a = enc["lc"] if (k == "lc" and "lc" in enc) else F[k]
        cols.append(a.ravel()[idx])
    X = np.column_stack(cols).astype(np.float64)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    return X


def full_matrix(F, enc, keys, mask):
    idx = np.flatnonzero(mask.ravel())
    return design(F, enc, keys, idx), idx


def logistic_model(Xtr, ytr, Xall):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    b, se = logistic((Xtr - mu) / sd, ytr.astype(float))
    z = np.clip(np.column_stack([np.ones(len(Xall)), (Xall - mu) / sd]) @ b, -30, 30)
    from math import erfc, sqrt
    zs = b / np.maximum(se, 1e-12)
    p = [erfc(abs(v) / sqrt(2)) for v in zs]
    return 1 / (1 + np.exp(-z)), b, se, np.array(p)


def rf_model(Xtr, ytr, Xall, keys, seed=0):
    m = RandomForest(n_trees=60, max_depth=12, min_leaf=5, seed=seed).fit(Xtr, ytr, [0, 1])
    p = m.predict_proba(Xall)[:, 1]
    return p, m


def permutation_importance(model, X, y, keys, seed=3):
    rng = np.random.default_rng(seed)
    base = auc(y, model.predict_proba(X)[:, 1])
    out = {}
    for j, k in enumerate(keys):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[k] = max(0.0, base - auc(y, model.predict_proba(Xp)[:, 1]))
    return out


def vif(X, keys):
    out = {}
    for j, k in enumerate(keys):
        y = X[:, j]
        O = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(O, y, rcond=None)
        res = y - O @ beta
        r2 = 1 - res.var() / max(y.var(), 1e-12)
        out[k] = 1 / max(1 - r2, 1e-6)
    return out


def roc(y, s, n=60):
    t = np.quantile(s, np.linspace(1, 0, n))
    P, N = max(y.sum(), 1), max((1 - y).sum(), 1)
    tpr = [float(((s >= v) & (y == 1)).sum() / P) for v in t]
    fpr = [float(((s >= v) & (y == 0)).sum() / N) for v in t]
    return fpr, tpr


def norm01(a, mask):
    v = a[mask & np.isfinite(a)]
    if v.size == 0:
        return a
    lo, hi = np.percentile(v, [1, 99])
    return np.where(mask, np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), np.nan).astype(np.float32)


def kmeans_breaks(a, mask, k=5, seed=0, n=20000):
    """Natural breaks (1-D k-means, close to Jenks) for k classes."""
    v = a[mask & np.isfinite(a)]
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    v = np.sort(rng.choice(v, min(n, v.size), replace=False))
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(100):
        lab = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        nc = np.array([v[lab == i].mean() if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(nc, c):
            break
        c = np.sort(nc)
    return (c[:-1] + c[1:]) / 2


def classify5(a, mask):
    b = kmeans_breaks(a, mask)
    out = np.zeros(mask.shape, np.uint8)
    if b is None:
        return out, None
    out[mask & np.isfinite(a)] = (np.digitize(a, b) + 1)[mask & np.isfinite(a)]
    return out, b


def class_density(cls5, flood, mask):
    rows = []
    tot_f = max((flood & mask).sum(), 1)
    tot = max(mask.sum(), 1)
    for c in range(1, 6):
        m = (cls5 == c) & mask
        a = m.sum()
        f = (m & flood).sum()
        rows.append(dict(cls=c, area=a / tot * 100, flood=f / tot_f * 100,
                         density=(f / tot_f) / max(a / tot, 1e-9)))
    return rows
