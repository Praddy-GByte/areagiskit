# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Yield estimate and automatic field boundaries.

Yield. Three levels, from always-available to calibrated:
  1. Relative yield index per pixel / field / zone from the season-integrated Sentinel-2 NDVI of the main season (area under
     the NDVI curve between sowing and harvest, above the bare-soil base). 100 = the median cropland of the area.
     Season-integrated NDVI tracks absorbed light and biomass, the standard remote-sensing yield proxy (Tucker et al. 1980;
     Lobell 2013). It is a ranking within the area, not tonnes.
  2. Yield anomaly per year since 2000 from MODIS gross primary production (MOD17A2HGF, 8-day, 500 m, gap-filled), summed over
     the crop season and expressed as % of the median year. GPP integrates weather (light, water stress, heat) and is the
     best open, global, long-record productivity measure (Running et al. 2004; Guan et al. 2016).
  3. Yield in t/ha only when the user gives yield statistics for zones (e.g. district yields): a linear calibration of the
     zone mean index against reported yield, with R², RMSE and leave-one-out error (Lobell et al. 2015 'scalable' approach,
     simplified to a single predictor).

Field boundaries. Edges between fields show as sharp spatial changes of NDVI on at least some dates of the year (different
crops, sowing dates, field margins, roads, hedges). The mean gradient magnitude of the smoothed NDVI time series is
thresholded (Otsu) inside cropland; the remaining interiors are connected regions = fields (GDAL polygonize, 4-connected),
smaller than `min_ha` dropped. numpy + GDAL only (no scipy). Similar in spirit to time-series edge methods
(Watkins & van Niekerk 2019; Waldner & Diakogiannis 2020), without deep learning.
"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr, osr

from ...core.data import search, warp_items

CREDIT_GPP = ("MODIS Terra Gross Primary Productivity 8-day gap-filled 500 m (MOD17A2HGF v6.1), Running & Zhao, "
              "NASA LP DAAC, via Microsoft Planetary Computer.")


# ------------------------------------------------------------------ yield index (Sentinel-2)
def main_integral(P):
    """Integrated NDVI of each pixel's strongest season (largest integral), and the sum over all seasons."""
    integ = P["integral"]
    best = np.nanmax(np.where(np.isfinite(integ), integ, -1), 0)
    best = np.where(best >= 0, best, np.nan).astype(np.float32)
    tot = np.nansum(np.where(np.isfinite(integ), integ, 0), 0).astype(np.float32)
    tot[~np.isfinite(best)] = np.nan
    return best, tot


def relative_index(v, mask):
    """100 = median of the masked values."""
    ok = mask & np.isfinite(v)
    if ok.sum() < 20:
        return np.full(v.shape, np.nan, np.float32), None
    med = float(np.median(v[ok]))
    out = np.where(ok, v / max(med, 1e-6) * 100, np.nan).astype(np.float32)
    return out, med


def season_window(P, crop, centers):
    """Day-of-year window (start, end) covering the crop seasons: median sowing of the earliest season to median harvest of the
    latest season, from the Sentinel-2 phenology. Returns (start_date, end_date) as real dates in the crop year."""
    sos, eos = P["sos"], P["eos"]
    s_idx, e_idx = [], []
    for i in range(sos.shape[0]):
        m = crop & np.isfinite(sos[i])
        if m.sum() > 0.05 * crop.sum():
            s_idx.append(np.nanmedian(sos[i][m]))
            e_idx.append(np.nanmedian(eos[i][m]))
    if not s_idx:
        return centers[0], centers[-1]
    a = centers[int(np.clip(round(min(s_idx)), 0, len(centers) - 1))]
    b = centers[int(np.clip(round(max(e_idx)), 0, len(centers) - 1))]
    return a, b


def _date(it):
    return dt.date.fromisoformat((it["properties"].get("start_datetime") or it["properties"]["datetime"])[:10])


def gpp_record(aoi, grid, crop_mask, window, start_year=2001, end=None, threads=8, progress=None):
    """Season GPP (gC/m², mean over cropland) for every year since `start_year`, summing the 8-day composites whose dates
    fall in the same day-of-year window as `window` (a pair of dates; may run over New Year). Terra only (record from 2000).
    Returns {year: dict(gpp, n)} where `year` is the year the season starts, plus the per-year 500 m season GPP maps."""
    end = end or dt.date.today()
    a, b = window
    span = (b - a).days
    if span <= 0:
        span += 365
    items = []
    for y in range(start_year, end.year + 1):
        s = dt.date(y, a.month, min(a.day, 28))
        e = s + dt.timedelta(days=span)
        if s > end:
            break
        items += [(y, it) for it in search("modis-17A2HGF-061", aoi.bbox, str(s), str(min(e, end)), max_items=200)
                  if it["id"].startswith("MOD")]
    groups, seen = {}, {}
    for y, it in sorted(items, key=lambda yi: yi[1]["id"]):     # one item per date and tile: the latest production wins
        tile = it["id"].split(".")[2]
        seen[(y, _date(it), tile)] = it
    for (y, d, _), it in seen.items():
        groups.setdefault((y, d), []).append(it)
    keys = sorted(groups)
    done = [0]

    def job(k):
        try:
            v = warp_items(groups[k], "Gpp_500m", grid, resample="near", dtype=gdal.GDT_Int16, src_nodata=32767, dst_nodata=32767)
            v = v.astype(np.float32)
            v[(v < 0) | (v > 30000)] = np.nan
            return k, v * 0.1          # 0.0001 kgC/m² -> gC/m² per 8 days
        except Exception:
            return k, None
    acc, cnt = {}, {}
    with ThreadPoolExecutor(threads) as ex:
        for k, v in ex.map(job, keys):
            done[0] += 1
            if progress and done[0] % 20 == 0:
                progress(done[0], len(keys))
            if v is None:
                continue
            y = k[0]
            if y not in acc:
                acc[y] = np.zeros(grid.shape, np.float32)
                cnt[y] = np.zeros(grid.shape, np.int16)
            ok = np.isfinite(v)
            acc[y][ok] += v[ok]
            cnt[y][ok] += 1
    full = max(int(np.median(cnt[y][crop_mask])) for y in cnt) if cnt and crop_mask.any() else 0
    out, maps = {}, {}
    for y in sorted(acc):
        n = int(np.median(cnt[y][crop_mask])) if crop_mask.any() else 0
        if full == 0 or n < 0.8 * full:          # incomplete season (current year, missing composites)
            continue
        m = np.where(cnt[y] >= 0.8 * full, acc[y] / np.maximum(cnt[y], 1) * full, np.nan)
        sel = crop_mask & np.isfinite(m)
        if sel.sum() < 3:
            continue
        out[y] = dict(gpp=float(np.nanmean(m[sel])), n=n)
        maps[y] = m.astype(np.float32)
    med = float(np.median([v["gpp"] for v in out.values()])) if out else None
    # data errors: some years of MOD17A2HGF on the archive hold near-zero GPP (seen for 2022: values ~1% of normal)
    bad = [y for y in out if med and out[y]["gpp"] < 0.4 * med]
    for y in bad:
        out.pop(y)
        maps.pop(y, None)
    gpp_record.dropped = bad
    med = float(np.median([v["gpp"] for v in out.values()])) if out else None
    for y in out:
        out[y]["anomaly"] = (out[y]["gpp"] / med - 1) * 100 if med else None
    return out, maps, med


# ------------------------------------------------------------------ calibration to t/ha
def calibrate(x, y):
    """Least-squares y = a + b x with R², RMSE and leave-one-out RMSE / MAPE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else None
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    loo = []
    for i in range(n):
        k = np.arange(n) != i
        if np.ptp(x[k]) == 0:
            continue
        bb, aa = np.polyfit(x[k], y[k], 1)
        loo.append(aa + bb * x[i] - y[i])
    loo = np.array(loo)
    return dict(a=float(a), b=float(b), r2=r2, rmse=rmse, n=n,
                loo_rmse=float(np.sqrt((loo ** 2).mean())) if loo.size else None,
                loo_mape=float(np.mean(np.abs(loo) / np.maximum(np.abs(y[:loo.size]), 1e-9)) * 100) if loo.size else None,
                mean_y=float(y.mean()))


# ------------------------------------------------------------------ field boundaries
def edge_strength(S, crop):
    """Mean gradient magnitude of the NDVI time series (central differences), only dates with data."""
    T, H, W = S.shape
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.int16)
    for t in range(T):
        a = S[t].astype(np.float32)
        gx = np.zeros_like(a)
        gy = np.zeros_like(a)
        gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / 2
        gy[1:-1, :] = (a[2:, :] - a[:-2, :]) / 2
        g = np.hypot(gx, gy)
        ok = np.isfinite(g)
        acc[ok] += g[ok]
        cnt[ok] += 1
    e = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    e[~crop] = np.nan
    return e


def otsu(v, bins=256):
    v = v[np.isfinite(v)]
    if v.size < 100:
        return None
    hi = np.percentile(v, 99.5)
    h, edges = np.histogram(np.clip(v, 0, hi), bins=bins, range=(0, hi))
    w = np.cumsum(h)
    mu = np.cumsum(h * edges[:-1])
    tot, mt = w[-1], mu[-1]
    with np.errstate(all="ignore"):
        sb = (mt * w / tot - mu) ** 2 / (w * (tot - w))
    return float(edges[int(np.nanargmax(sb))])


def segment_fields(S, crop, grid, min_ha=0.5, max_fields=20000):
    """Field ids (0 = edge / not a field) from NDVI time-series edges. Returns (ids, n, threshold, polygons)."""
    e = edge_strength(S, crop)
    thr = otsu(e[crop])
    if thr is None:
        return np.zeros(crop.shape, np.int32), 0, None, []
    interior = crop & np.isfinite(e) & (e < thr)
    # label connected interiors with GDAL polygonize -> rasterize back with ids
    H, W = crop.shape
    drv = gdal.GetDriverByName("MEM")
    src = drv.Create("", W, H, 1, gdal.GDT_Byte)
    src.SetGeoTransform(grid.gt)
    src.SetProjection(grid.wkt)
    src.GetRasterBand(1).WriteArray(interior.astype(np.uint8))
    vds = ogr.GetDriverByName("Memory").CreateDataSource("f")
    srs = osr.SpatialReference(wkt=grid.wkt)
    lyr = vds.CreateLayer("f", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(src.GetRasterBand(1), src.GetRasterBand(1), lyr, 0, [])
    min_m2 = min_ha * 1e4
    geoms = []
    for f in lyr:
        gm = f.GetGeometryRef()
        if f.GetField("v") == 1 and gm.GetArea() >= min_m2:
            geoms.append(gm.Clone())
    geoms.sort(key=lambda g: -g.GetArea())
    geoms = geoms[:max_fields]
    ids = np.zeros((H, W), np.int32)
    if geoms:
        dst = drv.Create("", W, H, 1, gdal.GDT_Int32)
        dst.SetGeoTransform(grid.gt)
        dst.SetProjection(grid.wkt)
        mv = ogr.GetDriverByName("Memory").CreateDataSource("g")
        ml = mv.CreateLayer("g", srs, ogr.wkbPolygon)
        ml.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        for i, g in enumerate(geoms):
            ft = ogr.Feature(ml.GetLayerDefn())
            ft.SetField("id", i + 1)
            ft.SetGeometry(g)
            ml.CreateFeature(ft)
        gdal.RasterizeLayer(dst, [1], ml, options=["ATTRIBUTE=id"])
        ids = dst.GetRasterBand(1).ReadAsArray().astype(np.int32)
    ids[~interior] = 0
    return ids, len(geoms), thr, geoms


def field_stats(ids, n, layers):
    """Per-field mean of each layer (dict name -> array), pixel count. Vectorised with bincount."""
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n + 1).astype(float)
    out = {"count": cnt}
    for k, a in layers.items():
        v = a.ravel().astype(float)
        ok = np.isfinite(v) & (flat > 0)
        s = np.bincount(flat[ok], weights=v[ok], minlength=n + 1)
        c = np.bincount(flat[ok], minlength=n + 1)
        with np.errstate(all="ignore"):
            out[k] = np.where(c > 0, s / c, np.nan)
        ss = np.bincount(flat[ok], weights=v[ok] ** 2, minlength=n + 1)
        with np.errstate(all="ignore"):
            out[k + "_sd"] = np.sqrt(np.maximum(np.where(c > 0, ss / c, np.nan) - out[k] ** 2, 0))
    return out


def zones_within_fields(v, ids, n, k=3):
    """Low / medium / high terciles of v inside every field (vectorised)."""
    flat = ids.ravel()
    val = v.ravel()
    ok = np.flatnonzero((flat > 0) & np.isfinite(val))
    z = np.zeros(flat.shape, np.uint8)
    if ok.size == 0:
        return z.reshape(ids.shape), np.zeros(n + 1)
    o = ok[np.lexsort((val[ok], flat[ok]))]
    f = flat[o]
    start = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
    sizes = np.diff(np.r_[start, f.size])
    rank = np.arange(f.size) - np.repeat(start, sizes)
    frac = rank / np.repeat(np.maximum(sizes, 1), sizes)
    z[o] = (np.minimum((frac * k).astype(int), k - 1) + 1).astype(np.uint8)
    z[o[np.repeat(sizes, sizes) < 10]] = 0
    low = np.bincount(f[z[o] == 1], minlength=n + 1) / np.maximum(np.bincount(f, minlength=n + 1), 1) * 100
    return z.reshape(ids.shape), low


def cdl_purity(ids, n, cdl):
    """Share of each field's pixels in its majority CDL class (1.0 = one crop). Returns array (n+1) and majority class."""
    flat = ids.ravel()
    c = cdl.ravel().astype(np.int64)
    ok = flat > 0
    key = flat[ok].astype(np.int64) * 256 + c[ok]
    u, cnt = np.unique(key, return_counts=True)
    fid, cls = u // 256, u % 256
    tot = np.bincount(fid, weights=cnt, minlength=n + 1)
    best = np.zeros(n + 1)
    bcls = np.zeros(n + 1, int)
    order = np.lexsort((cnt, fid))
    for i in order:          # last one per field wins = largest count
        best[fid[i]] = cnt[i]
        bcls[fid[i]] = cls[i]
    with np.errstate(all="ignore"):
        return np.where(tot > 0, best / tot, np.nan), bcls


def write_fields_gpkg(path, geoms, grid, stats, names):
    """Write field polygons (local equal-area CRS) with attributes to a GeoPackage."""
    import os
    if os.path.exists(path):
        os.remove(path)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    srs = osr.SpatialReference(wkt=grid.wkt)
    lyr = ds.CreateLayer("fields", srs, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("field_id", ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
    for nm in names:
        lyr.CreateField(ogr.FieldDefn(nm, ogr.OFTReal))
    for i, g in enumerate(geoms):
        ft = ogr.Feature(lyr.GetLayerDefn())
        ft.SetField("field_id", i + 1)
        ft.SetField("area_ha", round(g.GetArea() / 1e4, 3))
        for nm in names:
            v = stats[nm][i + 1]
            if np.isfinite(v):
                ft.SetField(nm, round(float(v), 4))
        ft.SetGeometry(g)
        lyr.CreateFeature(ft)
    ds = None
    return path
