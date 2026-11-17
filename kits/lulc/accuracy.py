# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Accuracy assessment.

* Confusion matrix, overall accuracy, Cohen's kappa, producer's / user's accuracy, F1.
* Area-adjusted accuracy and area estimates with 95% confidence intervals from a
  stratified random reference sample (Olofsson et al. 2014, Remote Sensing of Environment 148).
* A ready-to-label stratified sample design, so a user can add true reference labels.
"""

import numpy as np


def confusion(y_true, y_pred, classes):
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((k, k), np.int64)  # rows = map (predicted), cols = reference
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[p], idx[t]] += 1
    return m


def metrics(m):
    n = m.sum()
    if n == 0:
        return None
    oa = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = np.diag(m) / m.sum(1)
        pa = np.diag(m) / m.sum(0)
        f1 = 2 * ua * pa / (ua + pa)
    return dict(oa=float(oa), kappa=float(kappa), ua=ua, pa=pa, f1=f1, n=int(n))


def olofsson(m, map_area_px, px_area):
    """Area-adjusted estimates. m: sample confusion (rows map, cols reference) counts;
    map_area_px: pixel count of each map class (same order)."""
    W = map_area_px / map_area_px.sum()
    ni = m.sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (W[:, None] * m / ni[:, None])
    p = np.nan_to_num(p)
    area_prop = p.sum(0)
    se = np.sqrt(np.nansum(np.where(ni[:, None] > 1, (W[:, None] * p - p ** 2) / (ni[:, None] - 1), 0), axis=0))
    total = map_area_px.sum() * px_area
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = np.diag(p) / p.sum(1)
        pa = np.diag(p) / area_prop
    return dict(oa=float(np.trace(p)), ua=ua, pa=pa, area=area_prop * total, ci=1.96 * se * total,
                area_map=W * total)


def sample_design(cmap, mask, classes, total=500, min_per_class=50, seed=7):
    """Stratified random sample: proportional allocation with a floor for rare classes."""
    rng = np.random.default_rng(seed)
    counts = {c: int(((cmap == c) & mask).sum()) for c in classes}
    tot = sum(counts.values()) or 1
    out = []
    for c in classes:
        if counts[c] == 0:
            continue
        n = max(min_per_class, int(round(total * counts[c] / tot)))
        idx = np.flatnonzero(((cmap == c) & mask).ravel())
        out += [(int(i), c) for i in rng.choice(idx, size=min(n, idx.size), replace=False)]
    return out


def agreement(a, b, classes):
    """Map-to-map agreement on pixels where both have a class."""
    v = (a > 0) & (b > 0)
    if not v.any():
        return None
    m = confusion(b[v], a[v], classes)
    r = metrics(m)
    r["matrix"] = m
    return r
