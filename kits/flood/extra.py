# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""More flood analyses: settlements affected, cut-off settlements, hazard and risk classes, hot spots,
crop damage from Sentinel-2, recession rate and the discharge-flood area relation."""

import datetime as dt
import math

import numpy as np
from osgeo import gdal, ogr

from ...core.raster import neighbours
from ...core.data import warp_s2
from ...core.net import skipped


# ------------------------------------------------------------------ connected components (GDAL, no SciPy)
def label(mask, grid, eight=True):
    """Connected components of a boolean mask. Returns (labels int32, count)."""
    drv = gdal.GetDriverByName("MEM")
    src = drv.Create("", grid.cols, grid.rows, 1, gdal.GDT_Byte)
    src.SetGeoTransform(grid.gt)
    src.SetProjection(grid.wkt)
    src.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    mds = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = mds.CreateLayer("p", srs=grid.aoi.srs)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(src.GetRasterBand(1), src.GetRasterBand(1), lyr, 0, ["8CONNECTED=8"] if eight else [])
    geoms = [f.GetGeometryRef().Clone() for f in lyr]
    if not geoms:
        return np.zeros(mask.shape, np.int32), 0
    lab = grid.rasterize(geoms, values=list(range(1, len(geoms) + 1)))
    return (lab * mask).astype(np.int32), len(geoms)


# ------------------------------------------------------------------ settlements
def settlements(built, pop, flood, grid, min_built=0.10, min_cells=5):
    """Settlements = connected patches of built-up land (GHSL share >= min_built). For each: area, people, share flooded."""
    m = (np.nan_to_num(built) >= min_built) & grid.mask
    lab, n = label(m, grid)
    if n == 0:
        return lab, []
    flat = lab.ravel()
    cnt = np.bincount(flat, minlength=n + 1)
    fl = np.bincount(flat, weights=flood.ravel().astype(float), minlength=n + 1)
    pp = np.bincount(flat, weights=np.nan_to_num(pop).ravel(), minlength=n + 1) if pop is not None else np.zeros(n + 1)
    ppf = np.bincount(flat, weights=(np.nan_to_num(pop) * flood).ravel(), minlength=n + 1) if pop is not None else np.zeros(n + 1)
    rows = []
    rr, cc = np.indices(lab.shape)
    sr = np.bincount(flat, weights=rr.ravel(), minlength=n + 1)
    sc = np.bincount(flat, weights=cc.ravel(), minlength=n + 1)
    for i in range(1, n + 1):
        if cnt[i] < min_cells:
            continue
        lon, lat = grid.pixel_to_lonlat(int(sr[i] / cnt[i]), int(sc[i] / cnt[i]))
        rows.append(dict(id=i, area_km2=float(cnt[i] * grid.px_km2), people=float(pp[i]), flooded_pct=float(fl[i] / cnt[i] * 100),
                         people_flooded=float(ppf[i]), lon=lon, lat=lat))
    return lab, rows


def isolated(flood, observed, perm, built, grid, min_area_km2=0.01):
    """Dry land completely surrounded by flood water (cut off): dry patches that do not touch the area boundary
    and whose surrounding cells are at least 70% flood water (not radar gaps or permanent water)."""
    dry = observed & ~flood & ~perm & grid.mask
    lab, n = label(dry, grid, eight=False)
    if n == 0:
        return np.zeros(flood.shape, bool), []
    edge = np.zeros(flood.shape, bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge |= grid.mask & (neighbours(~grid.mask, 1) > 0)
    touching = np.zeros(n + 1, bool)
    touching[np.unique(lab[edge & (lab > 0)])] = True
    tot = np.zeros(n + 1)
    fl = np.zeros(n + 1)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(np.roll(lab, dy, 0), dx, 1)
        sel = (nb > 0) & ~dry & grid.mask
        tot += np.bincount(nb[sel], minlength=n + 1)
        fl += np.bincount(nb[sel], weights=flood[sel].astype(float), minlength=n + 1)
    size = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    bsum = np.bincount(lab.ravel(), weights=np.nan_to_num(built).ravel(), minlength=n + 1) if built is not None else np.zeros(n + 1)
    with np.errstate(all="ignore"):
        ok = (~touching) & (tot > 0) & (fl / np.maximum(tot, 1) >= 0.7) & (size * grid.px_km2 >= min_area_km2)
    ok[0] = False
    out = ok[lab] & (lab > 0)
    rr, cc = np.indices(lab.shape)
    sr = np.bincount(lab.ravel(), weights=rr.ravel(), minlength=n + 1)
    sc = np.bincount(lab.ravel(), weights=cc.ravel(), minlength=n + 1)
    rows = []
    for i in np.flatnonzero(ok):
        lon, lat = grid.pixel_to_lonlat(int(sr[i] / size[i]), int(sc[i] / size[i]))
        rows.append(dict(area_km2=float(size[i] * grid.px_km2), built_share=float(bsum[i] / size[i] * 100), lon=lon, lat=lat))
    return out, sorted(rows, key=lambda r: (-r["built_share"] * r["area_km2"], -r["area_km2"]))


# ------------------------------------------------------------------ hazard and risk
HAZARD = {1: ("Low", "#fde0c5"), 2: ("Moderate", "#f9a66c"), 3: ("High", "#e8562a"), 4: ("Very high", "#8e1d12")}


def hazard_classes(depth, duration, flood):
    """Depth x duration hazard (after common practice: deeper and longer floods do more harm)."""
    d = np.nan_to_num(depth, nan=0.0)
    t = np.nan_to_num(duration, nan=0.0)
    ds = np.select([d >= 2, d >= 1, d >= 0.5], [3, 2, 1], 0)
    ts = np.select([t >= 14, t >= 7, t >= 3], [3, 2, 1], 0)
    s = ds + ts
    h = np.select([s >= 5, s >= 3, s >= 1], [4, 3, 2], 1).astype(np.uint8)
    h[~flood] = 0
    return h


RISK = {1: ("Very low", "#2f7a4d"), 2: ("Low", "#9cc96b"), 3: ("Moderate", "#f2d36b"), 4: ("High", "#ec8b3a"), 5: ("Very high", "#b8322a")}


def risk(hazard01, pop, built, mask, vulnerability=None):
    """Risk = hazard x exposure (x vulnerability), each scaled 0-1; returns index and 5 classes by quantiles of non-zero risk."""
    def scale(a):
        if a is None:
            return np.zeros(mask.shape, np.float32)
        a = np.nan_to_num(a, nan=0.0)
        v = a[mask & (a > 0)]
        if v.size == 0:
            return np.zeros_like(a)
        hi = np.percentile(v, 98)
        return np.clip(a / max(hi, 1e-9), 0, 1)
    ex = np.maximum(scale(pop), scale(built))
    r = np.nan_to_num(hazard01) * ex
    if vulnerability is not None:
        r = r * (0.5 + 0.5 * scale(vulnerability))
    r[~mask] = np.nan
    v = r[mask & (r > 0)]
    cls = np.zeros(mask.shape, np.uint8)
    if v.size:
        q = np.percentile(v, [50, 75, 90, 97])
        cls[mask] = 1
        cls[mask & (r > 0)] = (np.digitize(r, q) + 1)[mask & (r > 0)]
    return r.astype(np.float32), cls


# ------------------------------------------------------------------ hot spots (Getis-Ord Gi*)
def gistar(values, mask, cell_px, radius_cells=1):
    """Gi* z-scores on a coarse grid (cell_px x cell_px pixels) with a square neighbourhood."""
    rows, cols = values.shape
    R, C = rows // cell_px, cols // cell_px
    if R < 3 or C < 3:
        return None, None
    v = np.nan_to_num(values[:R * cell_px, :C * cell_px]).reshape(R, cell_px, C, cell_px)
    m = mask[:R * cell_px, :C * cell_px].reshape(R, cell_px, C, cell_px)
    cnt = m.sum((1, 3))
    x = np.where(cnt > cell_px * cell_px * 0.5, (v * m).sum((1, 3)) / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(x)
    n = ok.sum()
    if n < 10:
        return None, None
    xb, s = np.nanmean(x), np.nanstd(x)
    k = 2 * radius_cells + 1
    xp = np.pad(np.nan_to_num(x), radius_cells)
    op = np.pad(ok.astype(float), radius_cells)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(xp, (k, k)).sum((-1, -2))
    wn = sliding_window_view(op, (k, k)).sum((-1, -2))
    with np.errstate(all="ignore"):
        z = (sw - xb * wn) / (s * np.sqrt((n * wn - wn ** 2) / (n - 1)))
    z[~ok] = np.nan
    return z, (R, C)


def expand(z, cell_px, shape):
    out = np.full(shape, np.nan, np.float32)
    R, C = z.shape
    out[:R * cell_px, :C * cell_px] = np.repeat(np.repeat(z, cell_px, 0), cell_px, 1)
    return out


GISTAR = {1: ("Cold spot 99%", "#2166ac"), 2: ("Cold spot 95%", "#67a9cf"), 3: ("Not significant", "#f2f2f2"),
          4: ("Hot spot 95%", "#ef8a62"), 5: ("Hot spot 99%", "#b2182b")}


def gi_classes(z):
    c = np.zeros(z.shape, np.uint8)
    f = np.isfinite(z)
    c[f] = 3
    c[f & (z >= 1.96)] = 4
    c[f & (z >= 2.58)] = 5
    c[f & (z <= -1.96)] = 2
    c[f & (z <= -2.58)] = 1
    return c


# ------------------------------------------------------------------ recession, extent-discharge
def recession(timeline):
    """Mean fall in flooded area per day after the peak (km²/day) and days to halve."""
    pk = max(timeline, key=lambda t: t["km2"])
    after = [t for t in timeline if t["day"] > pk["day"] and t["obs"] >= 80]
    if not after or pk["km2"] <= 0:
        return None
    half = next((t for t in after if t["km2"] <= pk["km2"] / 2), None)
    last = after[-1]
    rate = (pk["km2"] - last["km2"]) / max(last["day"] - pk["day"], 1)
    return dict(rate=rate, half_days=(half["day"] - pk["day"]) if half else None, peak=pk["km2"], until=last["day"] - pk["day"])


def extent_vs_discharge(series, q_dates, q_vals, px_km2, min_cover=0.9):
    """Flooded area on each well-covered radar pass vs river discharge that day."""
    idx = {d: i for i, d in enumerate(q_dates)}
    pts = []
    for s in series:
        d = dt.date.fromisoformat(s["date"])
        if s["cover"] >= min_cover and d in idx and np.isfinite(q_vals[idx[d]]):
            pts.append((float(q_vals[idx[d]]), s["flood_px"] * px_km2 / s["cover"]))
    if len(pts) < 15:
        return None
    q = np.array([p[0] for p in pts])
    a = np.array([p[1] for p in pts])
    rho = _spearman(q, a)
    # simple piecewise view: median area in discharge bins
    bins = np.quantile(q, [0, 0.5, 0.75, 0.9, 0.97, 1.0])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (q >= lo) & (q <= hi)
        if m.any():
            rows.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), median_km2=float(np.median(a[m])), max_km2=float(a[m].max())))
    # log-linear fit on high flows (area vs discharge above the median)
    hi_m = q >= np.median(q)
    fit = None
    if hi_m.sum() >= 8 and np.ptp(q[hi_m]) > 0:
        b = np.polyfit(q[hi_m], a[hi_m], 1)
        # only a rising relation is usable for prediction (more water, more flooded area)
        if b[0] > 0 and (rho is None or rho >= 0.3):
            fit = dict(slope=float(b[0]), intercept=float(b[1]))
    return dict(points=pts, rho=rho, bins=rows, fit=fit)


def _spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ crop damage from Sentinel-2
def crop_ndvi_change(aoi, grid, peak, crop, flood, search, warp_items, item_date):
    """NDVI on flooded vs non-flooded cropland, before (up to 60 days) and after (15-75 days) the flood."""
    def comp(a, b):
        items = search("sentinel-2-l2a", aoi.bbox, str(a), str(b), query={"eo:cloud_cover": {"lt": 40}}, max_items=60)
        if not items:
            return None, None
        days = {}
        for it in items:
            days.setdefault(item_date(it), []).append(it)
        stack = []
        used = []
        for day, its in sorted(days.items(), key=lambda kv: np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]]))[:4]:
            try:
                scl = warp_items(its, "SCL", grid, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
                r, n = warp_s2(its, "B04", grid), warp_s2(its, "B08", grid)
            except Exception as exc:
                skipped(exc)
                continue
            nd = (n - r) / (n + r)
            nd[~np.isin(scl, [4, 5])] = np.nan
            stack.append(nd)
            used.append(day)
        if not stack:
            return None, None
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmedian(np.stack(stack), 0), used
    before, db = comp(peak - dt.timedelta(days=60), peak - dt.timedelta(days=5))
    after, da = comp(peak + dt.timedelta(days=15), peak + dt.timedelta(days=75))
    if before is None or after is None:
        return None
    out = {}
    for lab, m in (("flooded", crop & flood), ("not flooded", crop & ~flood)):
        ok = m & np.isfinite(before) & np.isfinite(after)
        if ok.sum() < 50:
            return None
        out[lab] = dict(before=float(np.median(before[ok])), after=float(np.median(after[ok])), n=int(ok.sum()))
    ok = crop & flood & np.isfinite(before) & np.isfinite(after)
    loss = (before - after)[ok]
    out["loss_share"] = float((loss > 0.2).mean() * 100)
    out["dates_before"], out["dates_after"] = db, da
    out["change"] = np.where(crop & np.isfinite(before) & np.isfinite(after), after - before, np.nan).astype(np.float32)
    return out
