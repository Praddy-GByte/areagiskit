# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Crop groups without field data (clustering of NDVI calendars), rice from Sentinel-1 radar, supervised crop maps
with your field points, and management zones inside fields."""

import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import item_date, search, warp_items

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUP_COLS = ["#e6550d", "#31a354", "#3182bd", "#fdae6b", "#a1d99b", "#9ecae1", "#756bb1", "#bcbddc", "#636363", "#d9d9d9"]


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        C2 = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
        if np.allclose(C2, C, atol=1e-4):
            break
        C = C2
    return C, lab


def calendar_groups(S, crop, centers, k=6, n_sample=40000, seed=0):
    """Cluster cropland pixels by their smoothed NDVI calendar. Returns (group map, profiles, labels)."""
    T, H, W = S.shape
    Y = S.reshape(T, -1)
    idx = np.flatnonzero(crop.ravel() & np.isfinite(Y).all(0))
    if idx.size < k * 50:
        return None, None, None
    rng = np.random.default_rng(seed)
    sm = rng.choice(idx, min(n_sample, idx.size), replace=False)
    C, _ = kmeans(Y[:, sm].T.astype(np.float32), k, seed=seed)
    gmap = np.zeros(H * W, np.uint8)
    for s in range(0, idx.size, 300_000):
        ii = idx[s:s + 300_000]
        d = ((Y[:, ii].T[:, None, :] - C[None]) ** 2).sum(-1)
        gmap[ii] = d.argmin(1) + 1
    from .series import phenology
    P = phenology(C.T[:, :, None].astype(np.float32), centers)   # same season rules as the per-pixel map
    labels = []
    for j, c in enumerate(C):
        n = int(P["n"][j, 0])
        if n == 0:
            lab = "Low or no crop growth (fallow, sparse)" if c.max() < 0.5 else "Perennial / continuous green (orchard, sugarcane, grass)"
        else:
            names = [MONTHS[centers[int(P["pos"][i, j, 0])].month - 1] for i in range(n)]
            lab = ("Single season, peak " if n == 1 else "Double season, peaks " if n == 2 else "Triple season, peaks ") + " + ".join(names)
            if c.min() > 0.45:
                lab = "Perennial / continuous green (orchard, sugarcane, grass)"
        labels.append(lab)
    return gmap.reshape(H, W), C, labels


def s1_series(aoi, grid, start, n=30, step=12, threads=4):
    """Sentinel-1 VH (dB) every 12 days on the most common orbit. Returns stack and dates."""
    end = start + dt.timedelta(days=n * step)
    items = search("sentinel-1-rtc", aoi.bbox, str(start), str(end), max_items=2000)
    if not items:
        return None, []
    orbits = {}
    for it in items:
        orbits.setdefault(it["properties"].get("sat:relative_orbit"), []).append(it)
    orb = max(orbits, key=lambda o: len(orbits[o]))
    by = {}
    for it in orbits[orb]:
        by.setdefault(item_date(it), []).append(it)
    keys = sorted(by)

    def job(k):
        try:
            a = warp_items(by[k], "vh", grid, resample="average", dtype=gdal.GDT_Float32, src_nodata=-32768, dst_nodata=np.nan)
            a = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-6)), np.nan)
            return a.astype(np.float16)
        except Exception:
            return None
    with ThreadPoolExecutor(threads) as ex:
        arrs = list(ex.map(job, keys))
    ok = [(k, a) for k, a in zip(keys, arrs) if a is not None]
    if len(ok) < 6:
        return None, []
    return np.stack([a for _, a in ok]), [k for k, _ in ok]


def rice(vh, crop, dates=None, pheno=None, min_db=-22.0, rise_db=3.0, before=30, after=60, look=60):
    """Paddy rice: the field is flooded (VH <= min_db) close to the start of an optical crop season (from 30 days before
    to 60 days after the Sentinel-2 season start), and radar backscatter rises by at least `rise_db` within `look` days as
    the crop grows. Tying the flood to the season start removes winter-wet bare soil. Tested against USDA CDL 2021:
    California (water-seeded rice) found 89%, correct 82%; Arkansas (drill-seeded, flooded late) is mostly missed.
    Without `pheno` it falls back to the lowest-VH rule (Nguyen et al. 2016)."""
    V = vh.astype(np.float32)
    T = V.shape[0]
    if pheno is None or dates is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tmin = np.argmin(np.where(np.isfinite(V), V, 99), axis=0)
            vmin = np.nanmin(V, axis=0)
            aft = np.full(vmin.shape, -99.0, np.float32)
            for t in range(T - 1):
                aft = np.where(tmin == t, np.nanmax(V[t + 1:t + 9], axis=0), aft)
        return crop & (vmin <= min_db) & ((aft - vmin) >= 6.0)
    d = [dt.date.fromisoformat(str(x)) for x in dates]
    sos = pheno["sos"]                                     # window index of each season start
    out = np.zeros(V.shape[1:], bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(T):
            fl = V[t] <= min_db
            if not fl.any():
                continue
            j = [k for k in range(t + 1, T) if 0 < (d[k] - d[t]).days <= look]
            if not j:
                continue
            rise = (np.nanmax(V[j], axis=0) - V[t]) >= rise_db
            near = np.zeros_like(fl)
            for k in range(sos.shape[0]):
                # season start in days since the series start (10-day windows, centre of window)
                dd = (d[t] - d[0]).days + (d[0] - pheno["_start"]).days - (sos[k] * 10 + 5)
                near |= np.isfinite(dd) & (dd >= -before) & (dd <= after)
            out |= fl & rise & near
    return crop & out


def management_zones(ndvi_peak, field_ids, n_fields, k=3):
    """Within each field: low / medium / high productivity zones (terciles of peak NDVI)."""
    z = np.zeros(ndvi_peak.shape, np.uint8)
    stats = []
    for f in range(1, n_fields + 1):
        m = (field_ids == f) & np.isfinite(ndvi_peak)
        if m.sum() < 10:
            stats.append(None)
            continue
        v = ndvi_peak[m]
        q = np.percentile(v, [100 / 3, 200 / 3])
        z[m] = np.digitize(v, q) + 1
        stats.append(dict(mean=float(v.mean()), cv=float(v.std() / max(v.mean(), 1e-6) * 100), low_share=float((v < q[0]).mean() * 100)))
    return z, stats
