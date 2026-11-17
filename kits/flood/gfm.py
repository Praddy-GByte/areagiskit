# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Copernicus Global Flood Monitoring (GFM): the official Sentinel-1 flood archive (2015-today).

Read anonymously from the EODC STAC catalogue. GFM combines three independent flood algorithms
(DLR, TU Wien, LIST) into an ensemble flood extent for every Sentinel-1 pass worldwide.
"""

import datetime as dt
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr

from ...core.data import KitError, TRACE, get_json, warp

STAC = "https://stac.eodc.eu/api/v1/search"
ITEM = "https://stac.eodc.eu/api/v1/collections/GFM/items/{i}"
CREDIT_GFM = ("Copernicus Emergency Management Service, Global Flood Monitoring (GFM), ensemble flood extent "
              "(DLR, TU Wien, LIST algorithms), © European Union; served by EODC (stac.eodc.eu).")
_lock = threading.Lock()


def _trace(items):
    with _lock:
        for it in items:
            rec = ("GFM", it["id"], it["properties"]["datetime"][:10], ITEM.format(i=it["id"]))
            if rec not in TRACE:
                TRACE.append(rec)


def search(aoi, start, end, progress=None):
    """All GFM items whose data footprint overlaps the area. start/end: dates (searched year by year in parallel)."""
    s, e = dt.date.fromisoformat(str(start)[:10]), dt.date.fromisoformat(str(end)[:10])
    spans = []
    a = s
    while a <= e:
        b = min(dt.date(a.year, 12, 31), e)
        spans.append((a, b))
        a = b + dt.timedelta(days=1)
    with ThreadPoolExecutor(min(8, len(spans))) as ex:
        parts = list(ex.map(lambda ab: _search(aoi, ab[0], ab[1]), spans))
    return [it for p in parts for it in p]


def _search(aoi, start, end):
    x0, y0, x1, y1 = aoi.bbox
    url = (f"{STAC}?collections=GFM&bbox={x0},{y0},{x1},{y1}"
           f"&datetime={start}T00:00:00Z/{end}T23:59:59Z&limit=200")
    out, pages = [], 0
    while url:
        d = get_json(url, timeout=90)
        out += d.get("features", [])
        url = next((lk["href"] for lk in d.get("links", []) if lk.get("rel") == "next"), None)
        pages += 1
        if pages > 100:
            break
    geom = aoi.geom
    keep = []
    for it in out:
        try:
            g = ogr.CreateGeometryFromJson(json.dumps(it["geometry"]))
            if g is not None and g.Intersects(geom):
                keep.append(it)
        except Exception:
            keep.append(it)
    return keep


def passes(items):
    """Group tiles of the same satellite pass: {datetime: [items]} sorted by time."""
    g = defaultdict(list)
    for it in items:
        g[it["properties"]["datetime"][:13]].append(it)  # one satellite pass = same date and hour
    return dict(sorted(g.items()))


def read(items, grid, asset="ensemble_flood_extent", resample="max"):
    """Read one pass onto the grid. Returns uint8: 0 dry, 1 flood, 255 not observed."""
    srcs = ["/vsicurl/" + it["assets"][asset]["href"] for it in items if asset in it["assets"]]
    if not srcs:
        return None
    gdal.ErrorReset()
    a = warp(srcs, grid, resample=resample, dtype=gdal.GDT_Byte, src_nodata=255, dst_nodata=255)
    if gdal.GetLastErrorType() >= gdal.CE_Failure:  # a block failed to download: do not trust a partial map
        msg = gdal.GetLastErrorMsg()
        gdal.ErrorReset()
        raise IOError(msg)
    a[~grid.mask] = 255
    _trace(items)
    return a


def read_retry(items, grid, tries=4, **kw):
    import time
    for attempt in range(tries):
        try:
            return read(items, grid, **kw)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 + 4 * attempt)


def footprint_cover(items, aoi):
    """Share of the area inside the radar footprints of one pass (no download needed)."""
    u = None
    for it in items:
        try:
            g = ogr.CreateGeometryFromJson(json.dumps(it["geometry"]))
        except Exception:
            return 1.0
        if g is None:
            continue
        u = g if u is None else u.Union(g)
    if u is None:
        return 0.0
    i = u.Intersection(aoi.geom)
    return i.GetArea() / max(aoi.geom.GetArea(), 1e-12) if i is not None else 0.0


def _minute(items):
    t = min(it["properties"]["datetime"] for it in items)
    return int(t[11:13]) * 60 + int(t[14:16]) + int(t[17:19]) / 60.0


def useful_passes(keys, groups, grid, n_mask, min_cover, threads, cancelled=None, probes=3):
    """Sentinel-1 repeats the same orbits at the same time of day. Read a few passes of each time-of-day
    slot; skip slots whose orbit never covers the area. Returns (keys to read, probed arrays, n skipped)."""
    mins = {k: _minute(groups[k]) for k in keys}
    slots = {}
    for k in sorted(keys, reverse=True):
        slots.setdefault(int(mins[k] // 2), []).append(k)
    probe = []
    for s, ks in slots.items():
        probe += ks[:probes] + ks[len(ks) // 2:len(ks) // 2 + 1]
    probe = sorted(set(probe))

    def job(k):
        if cancelled and cancelled():
            return k, None
        try:
            return k, read_retry(groups[k], grid)
        except Exception:
            return k, None
    probed, good_min = {}, []
    with ThreadPoolExecutor(threads) as ex:
        for k, a in ex.map(job, probe):
            if a is None:
                continue
            cov = (a != 255).sum() / n_mask
            if cov >= min_cover * 0.5:
                good_min.append(mins[k])
                probed[k] = a
    good_min = np.array(good_min)
    keep = [k for k in keys if good_min.size and np.abs(good_min - mins[k]).min() <= 2.5]
    keep_set = set(keep)
    probed = {k: v for k, v in probed.items() if k in keep_set}
    return keep, probed, len(keys) - len(keep)


def history(aoi, grid, groups, min_cover=0.3, threads=16, progress=None, cancelled=None):
    """Read every pass. Returns dict with per-pass flooded area and per-pixel statistics."""
    n_mask = max(int(grid.mask.sum()), 1)
    keys = [k for k in groups if footprint_cover(groups[k], aoi) >= min_cover]
    keys, probed, n_skipped = useful_passes(keys, groups, grid, n_mask, min_cover, threads, cancelled)
    flood_n = np.zeros(grid.shape, np.uint16)
    obs_n = np.zeros(grid.shape, np.uint16)
    first = np.full(grid.shape, 0, np.int32)      # first flood date (days since 1970); 0 = never
    last = np.full(grid.shape, 0, np.int32)
    years = sorted({int(k[:4]) for k in keys})
    year_any = {y: np.zeros(grid.shape, bool) for y in years}
    year_obs = {y: np.zeros(grid.shape, bool) for y in years}
    series = []
    done = [0]

    failed = [0]

    def job(k):
        if k in probed:
            return k, probed.pop(k)
        if cancelled and cancelled():
            return k, None
        try:
            return k, read_retry(groups[k], grid)
        except Exception:
            failed[0] += 1
            return k, None

    with ThreadPoolExecutor(threads) as ex:
        for k, a in ex.map(job, keys):
            done[0] += 1
            if progress and done[0] % 10 == 0:
                progress(done[0], len(keys))
            if a is None:
                continue
            valid = a != 255
            cover = valid.sum() / n_mask
            if cover < min_cover:
                continue
            fl = a == 1
            d = (dt.date.fromisoformat(k[:10]) - dt.date(1970, 1, 1)).days
            flood_n += fl
            obs_n += valid
            new = fl & (first == 0)
            first[new] = d
            last[fl] = d
            y = int(k[:4])
            year_any[y] |= fl
            year_obs[y] |= valid
            series.append(dict(time=k, date=k[:10], cover=float(cover), flood_px=int(fl.sum()),
                               valid_px=int(valid.sum()), items=[i["id"] for i in groups[k]]))
    if not series:
        raise KitError("No Copernicus GFM flood observations could be read for this area.")
    return dict(flood_n=flood_n, obs_n=obs_n, first=first, last=last, year_any=year_any, year_obs=year_obs,
                series=sorted(series, key=lambda s: s["time"]), years=years, failed=failed[0], tried=len(keys),
                skipped=n_skipped)


def event_stack(groups, keys, grid, threads=8):
    """Read a short list of passes at full detail. Returns [(time, array)]."""
    def job(k):
        try:
            return k, read_retry(groups[k], grid)
        except Exception:
            return k, None
    with ThreadPoolExecutor(threads) as ex:
        return [(k, a) for k, a in ex.map(job, keys) if a is not None]


def aux(groups, key, grid, asset):
    try:
        a = read_retry(groups[key], grid, asset=asset, resample="near")
        return a
    except Exception:
        return None
