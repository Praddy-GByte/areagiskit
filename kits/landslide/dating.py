# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Date each automatic landslide scar to the interval between two satellite views (month-level), so rainfall thresholds can
be built even when an inventory has no dates. For each scar pixel: the last clear Sentinel-2 view in which the slope was still
green and the first view in which it was bare. The landslide happened somewhere in between."""

import datetime as dt

import numpy as np

from .detect import rdndvi


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


def date_scars(m, greenest_prev, prev, cur, thr=-15.0, green_thr=-7.5, keep_bare=0.8, pixels=False):
    """m: scars found when year y is compared with year y-1 (so the slide happened in y-1 after its green views, or early in
    y); greenest_prev: greenest NDVI of y-1; prev/cur: (dates, stack) for y-1 and y. Over the joined sequence of views, the
    slide falls between the LAST clearly green view and the next clear view, provided the slope stays bare afterwards
    (≥ keep_bare of later clear views). Seasonal leaf-off before the slide does not matter: it is followed by green views again.
    Returns {(start, end): pixels}."""
    if not m.any() or cur is None:
        return ({}, []) if pixels else {}
    ref = greenest_prev[m].astype(np.float32)
    dates = list(prev[0]) + list(cur[0]) if prev is not None else list(cur[0])
    S = np.concatenate([prev[1][:, m], cur[1][:, m]], 0) if prev is not None else cur[1][:, m]
    S = S.astype(np.float32)
    with np.errstate(all="ignore"):
        r = rdndvi(ref[None, :], S)
    fin = np.isfinite(S)
    green = fin & (r > green_thr)
    bare = fin & (r <= thr)
    K = S.shape[0]
    has_g = green.any(0)
    L = np.where(has_g, K - 1 - green[::-1].argmax(0), -1)
    out = {}
    idx = np.arange(K)[:, None]
    after = idx > L[None, :]
    n_after = (fin & after).sum(0)
    b_after = (bare & after).sum(0)
    okp = has_g & (n_after >= 1) & (b_after >= keep_bare * np.maximum(n_after, 1))
    nxt = np.where((fin & after).any(0), (fin & after).argmax(0), -1)
    per = []
    for l, e, o in zip(L, nxt, okp):
        if not o or e < 0:
            per.append(None)
            continue
        key = (str(dates[l]), str(dates[e]))
        out[key] = out.get(key, 0) + 1
        per.append(key)
    return (out, per) if pixels else out


def windows(dated, px_m2, min_px=5):
    """Merge the per-year dictionaries into a list of dated landslide windows with area (≥ min_px pixels)."""
    rows = []
    for (s, e), n in dated.items():
        if n < min_px:
            continue
        a, b = _d(s), _d(e)
        rows.append(dict(start=a, end=b, days=(b - a).days, px=n, km2=n * px_m2 / 1e6))
    return sorted(rows, key=lambda r: r["end"])


def rain_thresholds(days, rain, wins, scene_dates, ks=(1, 3, 7, 15, 30)):
    """Rain within each dated window (the trigger is assumed to be the wettest k days in the window), empirical 5% thresholds,
    and a skill test: do intervals between views that produced new scars hold more rain than intervals that did not?"""
    from ..lulc.models import auc
    idx = {d: i for i, d in enumerate(days)}
    r = np.nan_to_num(np.asarray(rain, float))

    def wmax(a, b, k):
        i0, i1 = idx.get(a + dt.timedelta(days=1)), idx.get(b)
        if i0 is None or i1 is None or i1 < i0:
            return None
        c = np.convolve(r, np.ones(k), mode="full")[:len(r)]
        return float(c[i0:i1 + 1].max())
    cum = {k: [wmax(w["start"], w["end"], k) for w in wins] for k in ks}
    rows = []
    for k in ks:
        v = np.array([x for x in cum[k] if x is not None])
        if v.size < 3:
            continue
        thr5 = float(np.percentile(v, 5))
        allc = np.convolve(r, np.ones(k), mode="full")[:len(r)]
        rows.append(dict(days=k, median=float(np.median(v)), thr5=thr5, exceed=float((allc >= thr5).mean() * 100)))
    # skill: all consecutive view intervals; positive = an interval ending a dated window with new scars
    sd = sorted({_d(s) for s in scene_dates})
    ends = {w["end"] for w in wins}
    y, s3 = [], []
    for a, b in zip(sd[:-1], sd[1:]):
        if (b - a).days > 70:
            continue
        m = wmax(a, b, 3)
        if m is None:
            continue
        y.append(1 if b in ends else 0)
        s3.append(m)
    skill = None
    if sum(y) >= 2 and len(y) - sum(y) >= 2:
        skill = dict(auc=auc(np.array(y), np.array(s3)), n_pos=int(sum(y)), n_neg=int(len(y) - sum(y)))
    med_days = float(np.median([w["days"] for w in wins])) if wins else None
    return dict(n=len(wins), rows=rows, skill=skill, median_window_days=med_days)
