# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Sentinel-2 vegetation time series for a crop year: 10-day composites, gap filling, smoothing and phenology."""

import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import item_date, search, warp_items, warp_s2

CLEAR = [4, 5, 6, 7, 11]   # SCL: vegetation, bare soil, water, unclassified, snow


def windows(start, n=36, step=10):
    return [(start + dt.timedelta(days=i * step), start + dt.timedelta(days=(i + 1) * step - 1)) for i in range(n)]


def _read(items, grid):
    scl = warp_items(items, "SCL", grid, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
    ok = np.isin(scl, CLEAR) & grid.mask
    if ok.sum() < 0.02 * grid.mask.sum():
        return None
    rd = lambda b: warp_s2(items, b, grid)
    r, n = rd("B04"), rd("B08")
    with np.errstate(all="ignore"):
        v = (n - r) / (n + r)
    v[~ok | ~np.isfinite(v) | (n + r <= 0.02) | (v < -1) | (v > 1)] = np.nan
    return v.astype(np.float16)


def ndvi_series(aoi, grid, start, n=36, step=10, max_cloud=70, threads=4, progress=None):
    """NDVI for each 10-day window (least cloudy date in the window). Returns (stack T x H x W float16, dates, used ids)."""
    end = start + dt.timedelta(days=n * step)
    items = search("sentinel-2-l2a", aoi.bbox, str(start), str(end), query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=1000)
    by_day = {}
    for it in items:
        by_day.setdefault(item_date(it), []).append(it)
    W = windows(start, n, step)
    pick = []
    for a, b in W:
        cand = [(d, v) for d, v in by_day.items() if a <= dt.date.fromisoformat(d) <= b]
        if cand:
            d, v = min(cand, key=lambda kv: np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]]))
            pick.append((d, v))
        else:
            pick.append((None, None))
    done = [0]

    def job(p):
        d, v = p
        if d is None:
            return None
        try:
            return _read(v, grid)
        except Exception:
            return None
    out = []
    with ThreadPoolExecutor(threads) as ex:
        for r in ex.map(job, pick):
            done[0] += 1
            if progress:
                progress(done[0], len(pick))
            out.append(r if r is not None else np.full(grid.shape, np.nan, np.float16))
    centers = [a + (b - a) / 2 for a, b in W]
    return np.stack(out), centers, [d for d, _ in pick]


def fill_smooth(Y, mask, chunk=400_000):
    """Linear gap filling along time, then a Savitzky-Golay-like smoothing (window 5, quadratic)."""
    T = Y.shape[0]
    flat = Y.reshape(T, -1)
    out = np.full(flat.shape, np.nan, np.float32)
    idx = np.flatnonzero(mask.ravel())
    t = np.arange(T, dtype=np.float32)
    sg = np.array([-3, 12, 17, 12, -3], np.float32) / 35.0
    for s in range(0, idx.size, chunk):
        ii = idx[s:s + chunk]
        y = flat[:, ii].astype(np.float32)
        y[(y < -1) | (y > 1)] = np.nan
        ok = np.isfinite(y)
        n_ok = ok.sum(0)
        good = n_ok >= max(6, T // 5)
        # linear interpolation per column via forward/backward fill of indices
        yy = y.copy()
        last = np.where(ok, t[:, None], -1).astype(np.int32)
        np.maximum.accumulate(last, axis=0, out=last)
        nxt = np.where(ok, t[:, None], T).astype(np.int32)
        nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
        cols = np.arange(y.shape[1])
        lv = np.where(last >= 0, y[np.clip(last, 0, T - 1), cols], np.nan)
        nv = np.where(nxt < T, y[np.clip(nxt, 0, T - 1), cols], np.nan)
        w = np.where((last >= 0) & (nxt < T), (t[:, None] - last) / np.maximum(nxt - last, 1), 0)
        interp = np.where(np.isfinite(lv) & np.isfinite(nv), lv + (nv - lv) * w, np.where(np.isfinite(lv), lv, nv))
        yy = np.where(ok, y, interp)
        # upper-envelope tweak: clouds left in the data only lower NDVI, so replace big single dips
        p = np.pad(yy, ((2, 2), (0, 0)), mode="edge")
        sm = sum(sg[k] * p[k:k + T] for k in range(5))
        dip = (yy < sm - 0.12)
        yy = np.where(dip, sm, yy)
        p = np.pad(yy, ((2, 2), (0, 0)), mode="edge")
        sm = sum(sg[k] * p[k:k + T] for k in range(5))
        sm = np.clip(sm, -1, 1)
        sm[:, ~good] = np.nan
        out[:, ii] = sm
    return out.reshape(Y.shape)


def phenology(S, dates, min_amp=0.20, min_peak=0.50, max_seasons=3, min_rel=0.85, strong=0.70, min_len=9, min_drop=0.30):
    """Per pixel: number of crop seasons and, for each, start / peak / end window index, peak NDVI and integrated NDVI.
    A season is a peak at least `min_amp` above the lowest NDVI on both sides and with NDVI >= min_peak. Extra seasons must
    also reach `min_rel` of the strongest season's peak: weaker green-ups (winter weeds, cover crops, volunteer growth before
    planting) are not counted as separate crops, unless the green-up itself is strong (peak >= `strong`). Extra seasons must also last
    at least `min_len` windows (90 days), which drops cover crops ended in spring. Finally, between the main season and an
    extra one NDVI must fall at least `min_drop` below the lower of the two peaks: two real crops are separated by a harvest and
    (nearly) bare soil, while cover crops, regrowth and weeds that green up a standing or unharvested field are not. Tuned on
    USDA CDL Arkansas 2021 and Punjab rice-wheat; tested unseen on western Kentucky and California (CDL 2021)."""
    T, H, W = S.shape
    Y = S.reshape(T, -1)
    nseas = np.zeros(Y.shape[1], np.uint8)
    res = {k: np.full((max_seasons, Y.shape[1]), np.nan, np.float32) for k in ("sos", "pos", "eos", "peak", "integral", "amp")}
    finite = np.isfinite(Y).all(0)
    prev = np.vstack([Y[:1], Y[:-1]])
    nxt = np.vstack([Y[1:], Y[-1:]])
    is_peak = (Y >= prev) & (Y > nxt) & (Y >= min_peak) & finite[None]
    # local max over ±3 windows (one month)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(np.nan_to_num(Y, nan=-1), ((3, 3), (0, 0)), mode="edge")
    wmax = sliding_window_view(pad, 7, axis=0).max(-1)
    is_peak &= Y >= wmax - 1e-6
    cols = np.flatnonzero(is_peak.any(0))
    for c in cols:
        y = Y[:, c]
        pk = np.flatnonzero(is_peak[:, c])
        seasons = []
        for p in pk:
            lo_l = y[:p + 1].min() if p > 0 else y[p]
            l = int(np.argmin(y[max(0, p - 15):p + 1])) + max(0, p - 15)
            r = int(np.argmin(y[p:min(T, p + 16)])) + p
            amp = y[p] - max(y[l], y[r])
            if amp < min_amp:
                continue
            base_l, base_r = y[l], y[r]
            thr_l = base_l + 0.2 * (y[p] - base_l)
            thr_r = base_r + 0.2 * (y[p] - base_r)
            sos = l + int(np.argmax(y[l:p + 1] >= thr_l))
            eos = p + int(np.argmax(y[p:r + 1] <= thr_r)) if (y[p:r + 1] <= thr_r).any() else r
            integ = float(np.clip(y[sos:eos + 1] - min(base_l, base_r), 0, None).sum())
            seasons.append((p, sos, eos, float(y[p]), integ, float(amp)))
        # merge peaks that share a season (closer than 6 windows = 2 months)
        merged = []
        for s in sorted(seasons, key=lambda s: -s[3]):
            if all(abs(s[0] - m[0]) >= 6 for m in merged):
                merged.append(s)
        if merged:
            top = max(m[3] for m in merged)
            best = max(merged, key=lambda m: m[3])

            def apart(m):
                # between two real crops the field goes (nearly) bare: NDVI must drop >= min_drop below the lower peak
                a, b = sorted((best[0], m[0]))
                return min(m[3], best[3]) - float(np.nanmin(y[a:b + 1])) >= min_drop
            merged = [m for m in merged if m is best or ((m[3] >= min_rel * top or m[3] >= strong) and m[2] - m[1] >= min_len
                                                          and apart(m))]
        merged = sorted(merged, key=lambda s: s[0])[:max_seasons]
        nseas[c] = len(merged)
        for i, (p, sos, eos, pv, integ, amp) in enumerate(merged):
            res["pos"][i, c], res["sos"][i, c], res["eos"][i, c] = p, sos, eos
            res["peak"][i, c], res["integral"][i, c], res["amp"][i, c] = pv, integ, amp
    out = {k: v.reshape(max_seasons, H, W) for k, v in res.items()}
    out["n"] = nseas.reshape(H, W)
    out["_start"] = dates[0] - dt.timedelta(days=4) if len(dates) else None   # start of the first 10-day window
    return out
