# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""10 m (or 20 m) land cover map of the latest year from Sentinel-2, trained on pixels where two independent 10 m maps
(IO/Esri annual and ESA WorldCover) agree and the IO map did not change between 2021 and 2023."""

import warnings

import numpy as np
from osgeo import gdal

from ...core import datasets as D
from ...core.aoi import Grid
from ...core.data import item_date, search, warp_items, warp_s2
from . import classify as C

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
IO_LAST = 2023


def _composite(aoi, grid, year, max_cloud=50, max_scenes=10):
    items = search("sentinel-2-l2a", aoi.bbox, f"{year}-01-01", f"{year}-12-31",
                   query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=300)
    days = {}
    for it in items:
        days.setdefault(item_date(it), []).append(it)
    cc = lambda kv: np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]])
    # spread through the year: clearest scene of each month first, then the next clearest
    best = {}
    for kv in sorted(days.items(), key=cc):
        best.setdefault(kv[0][:7], kv)
    order = sorted(best.values(), key=cc)[:max_scenes]
    stacks = {b: [] for b in BANDS}
    used = []
    for day, its in order:
        try:
            scl = warp_items(its, "SCL", grid, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
            ok = np.isin(scl, [4, 5, 6, 7, 11])
            if ok.sum() < 0.05 * grid.mask.sum():
                continue
            got = {}
            for b in BANDS:
                a = warp_s2(its, b, grid)
                a[~ok] = np.nan
                got[b] = a.astype(np.float32)
        except Exception:
            continue
        for b in BANDS:
            stacks[b].append(got[b])
        used.append(day)
    if len(used) < 3:
        return None, used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = {b: np.nanmedian(np.stack(stacks[b]), 0) for b in BANDS}
        nd = np.stack([(n - r) / (n + r) for n, r in zip(stacks["B08"], stacks["B04"])])
        out = dict(med, NDVI=np.nanmedian(nd, 0), NDVI_p10=np.nanpercentile(nd, 10, 0), NDVI_p90=np.nanpercentile(nd, 90, 0))
    del stacks, nd
    g, s, n, s2 = out["B03"], out["B11"], out["B08"], out["B12"]
    out["MNDWI"] = (g - s) / (g + s)
    out["NDBI"] = (s - n) / (s + n)
    out["BSI"] = ((s + out["B04"]) - (n + out["B02"])) / ((s + out["B04"]) + (n + out["B02"]))
    return out, used


def s2_lulc(aoi, year, progress=None):
    """Return dict(map, grid, res, classes, oa, kappa, f1, agree_io, agree_wc, scenes, areas) or None."""
    step = progress or (lambda f, m: None)
    res = 10 if aoi.area_km2 <= 300 else 20
    grid = Grid(aoi, res, max_pixels=4_000_000)
    step(0.05, f"Sentinel-2 {grid.res:.0f} m composite {year}")
    comp, used = _composite(aoi, grid, year)
    if comp is None:
        return None
    step(0.45, "Training labels where IO/Esri and WorldCover agree")
    io_items = lambda y: [i for i in search("io-lulc-annual-v02", aoi.bbox, f"{y}-01-01", f"{y}-12-31") if item_date(i)[:4] == str(y)]
    rd_io = lambda y: C._lut(C.IO_TO)[warp_items(io_items(y), "data", grid, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)]
    io21, io23 = rd_io(2021), rd_io(IO_LAST)
    wc = C._lut(C.WC_TO)[warp_items(search("esa-worldcover", aoi.bbox, "2021-01-01", "2021-12-31"), "map", grid,
                                    resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)]
    lab = np.where((io21 == wc) & (io21 == io23) & (io21 > 0) & grid.mask, io21, 0).astype(np.uint8)
    elev = D.dem(grid)
    slope, _ = D.slope_aspect(elev, grid)
    names = list(BANDS) + ["NDVI", "NDVI_p10", "NDVI_p90", "MNDWI", "NDBI", "BSI"]
    F = np.stack([comp[k] for k in names] + [elev.astype(np.float32), slope.astype(np.float32)])
    del comp
    step(0.6, "Random forest classification")
    r = C.classify_year(F, lab, np.ones(grid.shape, bool), grid, seed=year, n_trees=40)
    del F
    m = C.majority(r["map"], grid.mask)
    yt, yp = r["test"]
    oa = float((yt == yp).mean())
    ks = sorted(set(yt.tolist()) | set(yp.tolist()))
    pe = sum((yt == k).mean() * (yp == k).mean() for k in ks)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else None
    f1 = {}
    for k in ks:
        tp = ((yt == k) & (yp == k)).sum()
        p_, r_ = tp / max((yp == k).sum(), 1), tp / max((yt == k).sum(), 1)
        f1[k] = float(2 * p_ * r_ / (p_ + r_)) if p_ + r_ else 0.0
    v = (m > 0) & (wc > 0)
    agree_wc = float((m[v] == wc[v]).mean()) if v.any() else None
    same = io_items(year) if year <= IO_LAST else []
    agree_io = None
    if same:
        ioy = C._lut(C.IO_TO)[warp_items(same, "data", grid, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)]
        v = (m > 0) & (ioy > 0)
        agree_io = float((m[v] == ioy[v]).mean()) if v.any() else None
    px = grid.px_km2
    areas = {k: float((m == k).sum() * px) for k in C.CLASSES if (m == k).any()}
    step(1.0, "Done")
    return dict(map=m, grid=grid, res=grid.res, classes=C.CLASSES, oa=oa, kappa=kappa, f1=f1, agree_io=agree_io,
                agree_wc=agree_wc, scenes=used, areas=areas, n_train=int((lab > 0).sum()), year=year)
