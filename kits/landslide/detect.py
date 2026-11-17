# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Automatic landslide inventory from Sentinel-2: sudden loss of vegetation on slopes
(relative difference NDVI, as in HazMapper; Scheip & Wegmann 2021, NHESS 21, 1495)."""

import datetime as dt
import warnings

import numpy as np
from osgeo import gdal

from ...core.data import item_date, search, warp_items, warp_s2
from ...core.raster import neighbours
from ...core.net import skipped

CREDIT_S2 = "Sentinel-2 L2A (Copernicus), Microsoft Planetary Computer."


def ndvi_composite(aoi, grid, start, end, max_cloud=60, max_scenes=12, stat="median", keep=False):
    """Per-pixel median NDVI (and MNDWI) of clear observations between two dates. stat="max" gives the greenest-pixel
    composite from the clearest scene of each month (as HazMapper does): it does not depend on which season happened to be
    cloud-free, so year-to-year comparisons are fair in places with leaf-off winters or snow."""
    items = search("sentinel-2-l2a", aoi.bbox, str(start), str(end), query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=200)
    if not items:
        return None, None, []
    days = {}
    for it in items:
        days.setdefault(item_date(it), []).append(it)
    def cc(kv):
        return np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]])
    if stat == "max":
        best = {}
        for kv in days.items():
            mth = kv[0][:7]
            if mth not in best or cc(kv) < cc(best[mth]):
                best[mth] = kv
        order = sorted(best.values(), key=cc)[:max_scenes]
    else:
        order = sorted(days.items(), key=cc)[:max_scenes]
    nd, mw, used = [], [], []
    for day, its in order:
        try:
            scl = warp_items(its, "SCL", grid, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
            ok = np.isin(scl, [4, 5, 6, 7, 11])
            if ok.sum() < 0.05 * grid.mask.sum():
                continue
            def rd(b):
                return warp_s2(its, b, grid)
            r, n, g, s = rd("B04"), rd("B08"), rd("B03"), rd("B11")
        except Exception as exc:
            skipped(exc)
            continue
        v = (n - r) / (n + r)
        w = (g - s) / (g + s)
        bad = ~ok | (n + r <= 0.02) | ~np.isfinite(v) | (v < -1) | (v > 1)   # haze, shadows or broken pixels
        v[bad] = np.nan
        w[bad | (w < -1) | (w > 1)] = np.nan
        nd.append(v.astype(np.float32))
        mw.append(w.astype(np.float32))
        used.append(day)
    if not nd:
        return (None, None, [], None) if keep else (None, None, [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = np.nanmax if stat == "max" else np.nanmedian
        out = (f(np.stack(nd), 0), np.nanmedian(np.stack(mw), 0), sorted(used))
    if keep:   # every scene, in date order, as float16 (for dating scars to the month)
        o = np.argsort(used)
        return out + (([used[i] for i in o], np.stack([nd[i] for i in o]).astype(np.float16)),)
    return out


def rdndvi(pre, post):
    a, b = np.clip(pre, 0, 1), np.clip(post, 0, 1)
    with np.errstate(all="ignore"):
        return (b - a) / np.sqrt(a + b + 1e-6) * 100


def scars(pre, post, mndwi_post, slope, lc, mask, thr=-15.0, min_pre=0.4, min_slope=10.0, min_px=4, hand=None, min_hand=5.0):
    """Boolean map of new bare scars on slopes."""
    r = rdndvi(pre, post)
    m = (r <= thr) & (pre >= min_pre) & (np.nan_to_num(slope) >= min_slope) & mask
    if mndwi_post is not None:
        m &= ~(np.nan_to_num(mndwi_post, nan=-1) > 0.1)   # not new water
    if lc is not None:
        m &= ~np.isin(lc, [5, 7])                         # not cropland harvest or construction
    if hand is not None:
        m &= np.nan_to_num(hand, nan=99) >= min_hand      # not river banks and channels scoured by floods
    m &= neighbours(m, 1) >= 3                            # remove speckle
    # drop objects smaller than min_px (approximate: dense neighbourhood)
    m &= neighbours(m, 2) >= min_px
    return m, r


def event_inventory(aoi, grid, date, slope, lc, post_days=90, hand=None, **kw):
    """Compare the season after the event with the SAME season one year earlier, so leaf fall, harvest and
    seasonal greening are not mistaken for landslides."""
    a, b = date + dt.timedelta(days=3), date + dt.timedelta(days=post_days)
    post, mw, d_post = ndvi_composite(aoi, grid, a, b)
    pre, _, d_pre = ndvi_composite(aoi, grid, a - dt.timedelta(days=365), b - dt.timedelta(days=365))
    if pre is None or post is None:
        return None
    m, r = scars(pre, post, mw, slope, lc, grid.mask, hand=hand, **kw)
    seen = np.isfinite(pre) & np.isfinite(post) & grid.mask
    return dict(map=m & seen, rd=r, seen=seen, pre_dates=d_pre, post_dates=d_post, ndvi_pre=pre, ndvi_post=post)
