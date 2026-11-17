# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Every-year Landsat series (light composites) for trends and disturbance dating.

For each year: up to 4 clear dates spread over the year, read at 60 m, median NDVI, NBR
and surface temperature. Trends: Mann-Kendall test and Sen's slope per pixel.
Disturbance: year of the largest sustained drop in NDVI (LandTrendr-like, simplified).
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import trace, vsi, warp
from .composite import ROY, scenes

TREND_CLASSES = {1: ("Significant decrease", "#b2182b"), 2: ("Decrease (not significant)", "#f4a582"),
                 3: ("No trend", "#f2f0e8"), 4: ("Increase (not significant)", "#92c5de"), 5: ("Significant increase", "#2166ac")}


def light_year(aoi, grid, year, max_cloud=50):
    groups = scenes(aoi, year, list(range(1, 13)), max_cloud=max_cloud, max_dates=4)
    if not groups:
        return None
    for g in groups:
        trace(g)

    def rd(its, asset, dtype, resample, nod):
        hrefs = [vsi(i["assets"][asset]["href"]) for i in its if asset in i["assets"]]
        return warp(hrefs, grid, resample=resample, dtype=dtype, src_nodata=nod, dst_nodata=nod) if hrefs else None

    def one(its):
        plat = its[0]["properties"]["platform"]
        qa = rd(its, "qa_pixel", gdal.GDT_UInt16, "near", 1)
        good = ((qa & 0b111111) == 0) & grid.mask
        out = {}
        for b in ("red", "nir08", "swir22"):
            a = rd(its, b, gdal.GDT_Float32, "average", 0)
            r = a * 2.75e-05 - 0.2
            if plat not in ("landsat-8", "landsat-9"):
                r = ROY[b][0] + ROY[b][1] * r
            r[~good | (a == 0)] = np.nan
            out[b] = r
        asset = "lwir11" if "lwir11" in its[0]["assets"] else "lwir"
        t = rd(its, asset, gdal.GDT_Float32, "average", 0)
        c = t * 0.00341802 + 149.0 - 273.15
        c[~good | (t == 0) | (c < -40) | (c > 80)] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            nd = (out["nir08"] - out["red"]) / (out["nir08"] + out["red"])
            nbr = (out["nir08"] - out["swir22"]) / (out["nir08"] + out["swir22"])
        return nd, nbr, c

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, groups))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nd = np.nanmedian(np.stack([r[0] for r in res]), 0)
        nbr = np.nanmedian(np.stack([r[1] for r in res]), 0)
        lst = np.nanmedian(np.stack([r[2] for r in res]), 0)
    for a in (nd, nbr, lst):
        a[~grid.mask] = np.nan
    return dict(ndvi=nd.astype(np.float32), nbr=nbr.astype(np.float32), lst=lst.astype(np.float32), n=len(groups))


def series(aoi, grid, y0, y1, progress=None, max_cloud=50):
    years, nd, nbr, lst = [], [], [], []
    yrs = list(range(y0, y1 + 1))

    def job(y):
        try:
            return y, light_year(aoi, grid, y, max_cloud)
        except Exception:
            return y, None

    done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        for y, r in ex.map(job, yrs):
            done += 1
            if progress:
                progress(done / len(yrs), f"Annual series: {y}")
            if r is None:
                continue
            years.append(y)
            nd.append(r["ndvi"])
            nbr.append(r["nbr"])
            lst.append(r["lst"])
    return np.array(years), np.stack(nd), np.stack(nbr), np.stack(lst)


def mk_sen(Y, t, chunk=60000):
    """Mann-Kendall Z, p and Sen's slope per pixel. Y: (T, rows, cols) with NaN gaps."""
    T = Y.shape[0]
    flat = Y.reshape(T, -1)
    npx = flat.shape[1]
    Z = np.full(npx, np.nan, np.float32)
    P = np.full(npx, np.nan, np.float32)
    S = np.full(npx, np.nan, np.float32)
    ii, jj = np.triu_indices(T, 1)
    dt = (t[jj] - t[ii]).astype(np.float32)
    for s in range(0, npx, chunk):
        x = flat[:, s:s + chunk]
        ok = np.isfinite(x)
        n = ok.sum(0)
        d = x[jj] - x[ii]
        sgn = np.sign(np.nan_to_num(d))
        sgn[~(ok[jj] & ok[ii])] = 0
        stat = sgn.sum(0)
        var = n * (n - 1) * (2 * n + 5) / 18.0
        z = np.where(stat > 0, (stat - 1) / np.sqrt(np.maximum(var, 1)),
                     np.where(stat < 0, (stat + 1) / np.sqrt(np.maximum(var, 1)), 0))
        p = _erfc_vec(np.abs(z) / math.sqrt(2))
        slope = np.nanmedian(np.where(ok[jj] & ok[ii], d / dt[:, None], np.nan), axis=0)
        good = n >= max(6, T // 2)
        Z[s:s + chunk] = np.where(good, z, np.nan)
        P[s:s + chunk] = np.where(good, p, np.nan)
        S[s:s + chunk] = np.where(good, slope, np.nan)
    sh = Y.shape[1:]
    return Z.reshape(sh), P.reshape(sh), S.reshape(sh)


def _erfc_vec(x):
    # Abramowitz-Stegun 7.1.26
    t = 1 / (1 + 0.3275911 * x)
    y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return (y * np.exp(-x * x)).astype(np.float32)


def trend_classes(Z, P, mask):
    c = np.zeros(Z.shape, np.uint8)
    v = mask & np.isfinite(Z)
    sig = np.nan_to_num(P, nan=1) <= 0.05
    c[v & (Z < 0) & sig] = 1
    c[v & (Z < 0) & ~sig] = 2
    c[v & (Z == 0)] = 3
    c[v & (Z > 0) & ~sig] = 4
    c[v & (Z > 0) & sig] = 5
    return c


def disturbance(nd, years, min_drop=0.15, win=2):
    """Year (first year after) of the largest sustained NDVI drop, and its size."""
    T = len(years)
    best = np.zeros(nd.shape[1:], np.float32)
    when = np.zeros(nd.shape[1:], np.int16)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(win, T - win + 1):
            pre = np.nanmean(nd[k - win:k], 0)
            post = np.nanmean(nd[k:k + win], 0)
            drop = np.nan_to_num(pre - post)
            upd = drop > best
            best[upd] = drop[upd]
            when[upd] = years[k]
    when[best < min_drop] = 0
    return when, best
