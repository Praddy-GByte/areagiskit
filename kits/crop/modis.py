# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Long records from MODIS (2000-today): NDVI climatology, anomalies, Vegetation Condition Index, and burned cropland."""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import search, warp_items

CREDIT_MODIS = ("MODIS Terra/Aqua Vegetation Indices 16-day 250 m (MOD13Q1/MYD13Q1 v6.1) and Burned Area monthly 500 m "
                "(MCD64A1 v6.1), NASA LP DAAC, via Microsoft Planetary Computer.")


def _date(it):
    return dt.date.fromisoformat((it["properties"].get("start_datetime") or it["properties"]["datetime"])[:10])


def ndvi_record(aoi, grid, crop_mask, start_year=2001, end=None, threads=8, progress=None):
    """Mean cropland NDVI for every 16-day MODIS composite (Terra and Aqua). Returns sorted [(date, mean NDVI)] and
    a per-pixel stack is not kept (memory)."""
    end = end or dt.date.today()
    items = []
    for y in range(start_year, end.year + 1):
        items += search("modis-13Q1-061", aoi.bbox, f"{y}-01-01", f"{y}-12-31", max_items=500)
    groups = {}
    for it in items:
        groups.setdefault(_date(it), []).append(it)
    keys = sorted(groups)
    done = [0]

    def job(k):
        try:
            v = warp_items(groups[k], "250m_16_days_NDVI", grid, resample="average", dtype=gdal.GDT_Float32,
                           src_nodata=-3000, dst_nodata=np.nan) * 1e-4
            q = warp_items(groups[k], "250m_16_days_pixel_reliability", grid, resample="near", dtype=gdal.GDT_Int16,
                           src_nodata=-1, dst_nodata=-1)
            v[(q < 0) | (q > 1)] = np.nan
            m = crop_mask & np.isfinite(v)
            if m.sum() < 0.2 * max(crop_mask.sum(), 1):
                return k, None, None
            return k, float(np.nanmean(v[m])), v.astype(np.float16)
        except Exception:
            return k, None, None
    out, last = [], {}
    with ThreadPoolExecutor(threads) as ex:
        for k, mean, arr in ex.map(job, keys):
            done[0] += 1
            if progress and done[0] % 20 == 0:
                progress(done[0], len(keys))
            if mean is not None:
                out.append((k, mean))
                last[k] = arr
    return out, last


def climatology(record):
    """Day-of-year (16-day bins) mean and std of cropland NDVI."""
    bins = {}
    for d, v in record:
        b = (d.timetuple().tm_yday - 1) // 16
        bins.setdefault(b, []).append((d.year, v))
    return {b: (float(np.mean([v for _, v in xs])), float(np.std([v for _, v in xs])), float(np.min([v for _, v in xs])),
                float(np.max([v for _, v in xs]))) for b, xs in bins.items()}


def yearly(record, months=None):
    """Integrated (sum) and peak NDVI per year over the given months (default all)."""
    ys = {}
    for d, v in record:
        if months and d.month not in months:
            continue
        ys.setdefault(d.year, []).append(v)
    return {y: dict(peak=float(max(v)), integral=float(np.sum(v)), n=len(v)) for y, v in ys.items() if len(v) >= (len(months) if months else 20)}


def vci(current, history_min, history_max):
    """Vegetation Condition Index (Kogan 1995): 0 = worst on record, 100 = best on record."""
    with np.errstate(all="ignore"):
        return np.clip((current - history_min) / np.maximum(history_max - history_min, 1e-6) * 100, 0, 100)


def burned(aoi, grid, crop_mask, start_year=2001, end=None, threads=8):
    """Burned cropland (km²) per month from MCD64A1. Returns {(year, month): km2} and the burn-count raster."""
    end = end or dt.date.today()
    items = search("modis-64A1-061", aoi.bbox, f"{start_year}-01-01", str(end), max_items=2000)
    groups = {}
    for it in items:
        d = _date(it)
        groups.setdefault((d.year, d.month), []).append(it)

    def job(k):
        try:
            a = warp_items(groups[k], "Burn_Date", grid, resample="near", dtype=gdal.GDT_Int16, src_nodata=-2, dst_nodata=-2)
            return k, (a > 0) & crop_mask
        except Exception:
            return k, None
    out = {}
    count = np.zeros(grid.shape, np.uint16)
    with ThreadPoolExecutor(threads) as ex:
        for k, b in ex.map(job, sorted(groups)):
            if b is None:
                continue
            out[k] = float(b.sum() * grid.px_km2)
            count += b
    return out, count
