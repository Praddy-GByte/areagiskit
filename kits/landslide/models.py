# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Landslide susceptibility models: Frequency Ratio, Information Value, Weights of Evidence, Index of Entropy,
AHP (expert weights), Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting and their ensemble,
with spatial cross-validation."""

import numpy as np

from ..flood import suscept as S
from ..lulc.models import auc

# key, name, unit, AHP importance (1-9), AHP direction (+1 higher value -> more susceptible, 0 categorical/rule)
FACTORS = [
    ("slope", "Slope", "°", 9, +1),
    ("relief", "Local relief (300 m)", "m", 6, +1),
    ("lc", "Land cover", "class", 6, 0),
    ("ndvi", "Vegetation (NDVI)", "", 5, -1),
    ("dist_stream", "Distance to streams", "m", 5, -1),
    ("twi", "Topographic wetness index", "", 5, +1),
    ("plan", "Plan curvature", "1/100 m", 4, -1),
    ("profile", "Profile curvature", "1/100 m", 4, -1),
    ("aspect", "Aspect", "class", 3, 0),
    ("elev", "Elevation", "m", 3, 0),
    ("spi", "Stream power index", "", 4, +1),
    ("tri", "Terrain ruggedness", "m", 4, +1),
    ("tpi", "Topographic position", "m", 3, -1),
    ("dd", "Drainage density", "km/km²", 4, +1),
    ("clay", "Soil clay", "%", 3, +1),
    ("sand", "Soil sand", "%", 2, -1),
    ("dist_fault", "Distance to active faults", "m", 5, -1),
    ("dist_road", "Distance to roads", "m", 5, -1),
    ("hand", "Height above drainage", "m", 3, 0),
    ("geol", "Lithology (Macrostrat)", "class", 6, 0),
    ("pga", "Earthquake shaking hazard (PGA, 475-year)", "g", 5, +1),
    ("pga_event", "Event shaking (USGS ShakeMap PGA)", "g", 7, +1),
]
NAMES = {k: (n, u) for k, n, u, _, _ in FACTORS}
CATEGORICAL = {"lc", "aspect", "geol"}
# Smooth regional factors (5.5 km / ~1 km grids) act as a location proxy in data-driven models and inflate spatial
# cross-validation (Sikkim: +0.03-0.07 AUC with 1 km blocks, ~0 with 5 km blocks), so they enter the bivariate tables,
# FR/IV/WoE/IoE and AHP, but not LR/RF/SVM/GBM.
CONTEXT_ONLY = {"pga", "pga_event"}
GEOL_SCORE = {1: 4, 2: 5, 3: 4, 4: 3, 5: 2, 6: 3, 7: 2, 8: 3, 9: 3}   # weathering-prone and weak units score higher
LC_SCORE = {1: 1, 2: 2, 4: 1, 5: 3, 7: 4, 8: 5, 9: 1, 10: 3, 11: 4}


def aspect_classes(aspect, slope):
    a = np.nan_to_num(aspect)
    c = (((a + 22.5) % 360) // 45 + 1).astype(np.float32)   # 1 N, 2 NE, ... 8 NW
    c[np.nan_to_num(slope) < 2] = 9                        # flat
    return c


ASPECT = {1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W", 8: "NW", 9: "Flat"}


def tables(F, pos_idx, mask, k=8):
    """Class tables (pixels, landslide pixels) for every factor."""
    out, cls = {}, {}
    for key, a in F.items():
        if key in CATEGORICAL:
            c = np.nan_to_num(a, nan=0).astype(np.int32)
            e = None
        else:
            c, e = S.classes_of(key, a, mask)
            if c is None:
                continue
        cl = c.ravel()
        n = int(cl.max()) + 1
        area = np.bincount(cl[mask.ravel()], minlength=n).astype(float)
        ls = np.bincount(cl[pos_idx], minlength=n).astype(float)
        rows = []
        for i in range(1, n):
            if area[i] == 0:
                continue
            if e is not None:
                lo, hi = e[i - 1], e[i]
                lab = (f"< {hi:,.1f}" if not np.isfinite(lo) else f"≥ {lo:,.1f}" if not np.isfinite(hi) else f"{lo:,.1f} – {hi:,.1f}")
            elif key == "aspect":
                lab = ASPECT.get(i, str(i))
            elif key == "geol":
                from .geology import NAMES as _GN
                lab = _GN.get(i, str(i))
            else:
                lab = str(i)
            rows.append(dict(cls=i, label=lab, npx=area[i], nls=ls[i]))
        out[key], cls[key] = rows, c
    return out, cls


def bivariate(T, C, mask):
    """FR, IV, WoE (contrast) and Index-of-Entropy indices from class tables. Returns dict of maps and per-factor stats."""
    maps = {m: np.zeros(mask.shape, np.float32) for m in ("fr", "iv", "woe", "ioe")}
    stats = {}
    for key, rows in T.items():
        tot_px = sum(r["npx"] for r in rows)
        tot_ls = max(sum(r["nls"] for r in rows), 1)
        frs = []
        lut_fr, lut_iv, lut_w = {}, {}, {}
        for r in rows:
            p_ls = r["nls"] / tot_ls
            p_px = r["npx"] / tot_px
            fr = p_ls / max(p_px, 1e-12)
            r["fr"] = fr
            r["iv"] = float(np.log(max(fr, 1e-3)))
            # weights of evidence
            n1, n2 = r["nls"], r["npx"] - r["nls"]
            n3, n4 = tot_ls - n1, (tot_px - tot_ls) - n2
            wp = np.log(((n1 + 0.5) / (tot_ls + 1)) / ((n2 + 0.5) / (tot_px - tot_ls + 1)))
            wm = np.log(((n3 + 0.5) / (tot_ls + 1)) / ((n4 + 0.5) / (tot_px - tot_ls + 1)))
            r["w_plus"], r["w_minus"], r["contrast"] = float(wp), float(wm), float(wp - wm)
            lut_fr[r["cls"]], lut_iv[r["cls"]], lut_w[r["cls"]] = fr, r["iv"], r["contrast"]
            frs.append(fr)
        frs = np.array(frs)
        pij = frs / max(frs.sum(), 1e-12)
        H = -np.sum(pij[pij > 0] * np.log2(pij[pij > 0]))
        Hmax = np.log2(max(len(frs), 2))
        I = (Hmax - H) / Hmax
        W = I * frs.mean()
        stats[key] = dict(entropy_weight=float(W), info=float(I), fr_range=float(frs.max() - frs.min()) if len(frs) else 0)
        c = C[key]
        n = int(c.max()) + 1
        for name, lut in (("fr", lut_fr), ("iv", lut_iv), ("woe", lut_w)):
            arr = np.zeros(n, np.float32)
            for kk, v in lut.items():
                arr[kk] = v
            maps[name] += arr[c]
        arr = np.zeros(n, np.float32)
        for kk, v in lut_fr.items():
            arr[kk] = v * W
        maps["ioe"] += arr[c]
    for m in maps:
        maps[m][~mask] = np.nan
    return maps, stats


def ahp_score(key, a, mask):
    v = np.nan_to_num(a, nan=np.nanmedian(a) if np.isfinite(a).any() else 0)
    if key == "slope":
        s = np.select([v >= 35, v >= 25, v >= 15, v >= 8], [5, 4, 3, 2], 1)
    elif key == "lc":
        s = np.zeros(v.shape)
        for c, sc in LC_SCORE.items():
            s[v == c] = sc
        s[s == 0] = 3
    elif key == "aspect":
        s = np.where(v == 9, 1, 3)
    elif key == "geol":
        s = np.zeros(v.shape)
        for c, sc in GEOL_SCORE.items():
            s[v == c] = sc
        s[s == 0] = 3
    elif key == "dist_fault":
        s = np.select([v < 500, v < 1000, v < 2000, v < 5000], [5, 4, 3, 2], 1)
    elif key == "dist_road":
        s = np.select([v < 100, v < 250, v < 500, v < 1000], [5, 4, 3, 2], 1)
    elif key == "dist_stream":
        s = np.select([v < 50, v < 100, v < 200, v < 500], [5, 4, 3, 2], 1)
    elif key in ("elev", "hand"):
        q = np.nanpercentile(a[mask], [20, 40, 60, 80])
        mid = np.digitize(v, q) + 1
        s = np.select([mid == 3, (mid == 2) | (mid == 4)], [5, 3], 2)   # mid-slope elevations fail most
    else:
        q = np.nanpercentile(a[mask], [20, 40, 60, 80])
        s = np.digitize(v, q) + 1
        d = {k: dd for k, _, _, _, dd in FACTORS}[key]
        if d < 0:
            s = 6 - s
    return np.where(mask, s, np.nan).astype(np.float32)


def ahp(F, mask):
    keys = [k for k, *_ in FACTORS if k in F]
    imp = {k: s for k, _, _, s, _ in FACTORS}
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
    ci = (lam - n) / (n - 1)
    cr = ci / S.RI.get(n, 1.59)
    idx = np.zeros(mask.shape, np.float32)
    for k, wk in zip(keys, w):
        idx += wk * np.nan_to_num(ahp_score(k, F[k], mask))
    idx[~mask] = np.nan
    return idx, dict(zip(keys, w)), float(cr), float(lam), keys


def design(F, keys, idx, enc):
    cols = []
    for k in keys:
        a = enc[k] if k in CATEGORICAL else F[k]
        cols.append(a.ravel()[idx])
    X = np.column_stack(cols).astype(np.float64)
    med = np.nanmedian(X, axis=0)
    return np.where(np.isfinite(X), X, med)


def block_ids(shape, block_px):
    r, c = np.indices(shape)
    return (r // block_px) * 10000 + (c // block_px)


def spatial_cv(F, keys, enc, pos, neg, mask, block_px, k=5, seed=3):
    """k-fold cross-validation with whole spatial blocks held out (no neighbouring pixels in train and test)."""
    rng = np.random.default_rng(seed)
    bid = block_ids(mask.shape, block_px)
    ub = np.unique(bid[mask])
    rng.shuffle(ub)
    fold_of = {b: i % k for i, b in enumerate(ub)}
    fold = np.vectorize(fold_of.get, otypes=[np.int64])(bid.ravel()[np.flatnonzero(mask.ravel())])
    fmap = np.full(mask.size, -1, np.int64)
    fmap[np.flatnonzero(mask.ravel())] = fold
    from . import ml as ML
    res = {"lr": [], "rf": [], "svm": [], "gbm": []}
    for f in range(k):
        tr_p = pos & (fmap.reshape(mask.shape) != f)
        tr_n = neg & (fmap.reshape(mask.shape) != f)
        te_p = pos & (fmap.reshape(mask.shape) == f)
        te_n = neg & (fmap.reshape(mask.shape) == f)
        a = S.sample(tr_p, tr_n, n=3000, seed=f)
        b = S.sample(te_p, te_n, n=1500, seed=f + 10)
        if a is None or b is None:
            continue
        Xtr, Xte = design(F, keys, a[0], enc), design(F, keys, b[0], enc)
        p_lr, *_ = S.logistic_model(Xtr, a[1], Xte)
        p_rf, _ = S.rf_model(Xtr, a[1], Xte, keys, seed=f)
        res["lr"].append(auc(b[1], p_lr))
        res["rf"].append(auc(b[1], p_rf))
        res["svm"].append(auc(b[1], ML.svm_model(Xtr, a[1], Xte, seed=f)))
        res["gbm"].append(auc(b[1], ML.gbm_model(Xtr, a[1], Xte, seed=f)[0]))
    return {m: (float(np.mean(v)), float(np.std(v)), len(v)) for m, v in res.items() if v}


def success_rate(index, pos, mask, n=50):
    """Success-rate curve: share of landslide pixels captured in the top x% most susceptible area. AUC of the curve."""
    v = index[mask & np.isfinite(index)]
    t = np.quantile(v, np.linspace(1, 0, n + 1))
    tot = max((pos & np.isfinite(index)).sum(), 1)
    x = np.linspace(0, 1, n + 1)
    y = np.array([((index >= q) & pos).sum() / tot for q in t])
    return x.tolist(), y.tolist(), float(np.trapz(y, x))
