# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""People, green space, access, air (NO2) and heat vulnerability for the AreaGIS Kit: Urban."""

import datetime as dt
import os
import tempfile
import urllib.request
import warnings

import numpy as np
from osgeo import gdal

from ...core.data import UA, item_date, search, trace, vsi, warp
from ...core.net import open_url, skipped

HRSL = "https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt"
CREDIT_HRSL = ("Population: Meta (Facebook) Data for Good & CIESIN, High Resolution Settlement Layer / population density maps "
               "(30 m, ~2020), CC BY 4.0, via AWS Open Data.")
CREDIT_S5P = "Sentinel-5P TROPOMI NO2 Level-2 (Copernicus data, ESA/KNMI), via Microsoft Planetary Computer."


# ------------------------------------------------------------------ people
def population_hrsl(grid):
    """People per analysis cell from Meta's 30 m population map."""
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    trace(url=HRSL, label="meta-hrsl-population")
    try:
        a = warp(["/vsicurl/" + HRSL], grid, resample="sum", dtype=gdal.GDT_Float32, src_nodata=np.nan, dst_nodata=-1)
    except Exception:
        return None
    a = np.where(a < 0, 0, a).astype(np.float32)
    a[~grid.mask] = np.nan
    return a


# ------------------------------------------------------------------ blocks
def block(arr, k, fn="mean"):
    """Aggregate an array into k x k blocks (edges dropped). Returns (coarse array, (R, C))."""
    R, C = arr.shape[0] // k, arr.shape[1] // k
    v = arr[:R * k, :C * k].reshape(R, k, C, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}[fn](v, axis=(1, 3))
    return out, (R, C)


def expand(coarse, k, shape):
    out = np.full(shape, np.nan, np.float32)
    R, C = coarse.shape
    out[:R * k, :C * k] = np.repeat(np.repeat(coarse, k, 0), k, 1)
    return out


def zscore(x, m):
    v = x[m & np.isfinite(x)]
    if v.size < 3 or v.std() == 0:
        return np.zeros_like(x)
    return (x - v.mean()) / v.std()


def quintiles(x, m):
    v = x[m & np.isfinite(x)]
    out = np.zeros(x.shape, np.uint8)
    if v.size < 10:
        return out
    q = np.percentile(v, [20, 40, 60, 80])
    f = m & np.isfinite(x)
    out[f] = np.digitize(x[f], q) + 1
    return out


# ------------------------------------------------------------------ green
def box_mean(a, r):
    """Mean over a (2r+1) x (2r+1) window, ignoring NaN (integral images)."""
    v = np.nan_to_num(a).astype(np.float64)
    m = np.isfinite(a).astype(np.float64)
    def win(x):
        p = np.pad(x, ((r + 1, r), (r + 1, r)))
        c = p.cumsum(0).cumsum(1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]
    with np.errstate(all="ignore"):
        return (win(v) / win(m)).astype(np.float32)


def green_metrics(tree10, green10, grid10, grid, pop, urban, built):
    """Tree canopy and green space from 10 m masks; 3-30-300 parts; green space per person; distance to green >= 0.5 ha."""
    from ..flood.extra import label
    from .heat import to_grid
    tree10 = tree10.astype(np.float32)
    green10 = green10.astype(np.float32)
    tree = to_grid(np.where(grid10.mask, tree10, np.nan), grid10, grid)        # fraction of each 30 m cell
    green = to_grid(np.where(grid10.mask, green10, np.nan), grid10, grid)
    gmask = np.nan_to_num(green) >= 0.5
    lab, n = label(gmask, grid)
    sizes = np.bincount(lab.ravel())
    big = np.zeros(grid.shape, bool)
    if n:
        ok = np.flatnonzero(sizes >= max(1, int(np.ceil(5000 / (grid.res ** 2))))) # >= 0.5 ha
        ok = ok[ok > 0]
        big = np.isin(lab, ok)
    dist = grid.distance_to(big) if big.any() else np.full(grid.shape, np.inf, np.float32)
    out = dict(tree=tree, green=green, green_patches=big, dist_green=dist)
    if urban is not None and urban.any():
        u = urban & np.isfinite(tree)
        out["canopy_urban"] = float(np.nanmean(tree[u]) * 100)
        out["green_urban"] = float(np.nanmean(green[u]) * 100)
        if pop is not None:
            people = float(np.nansum(pop[urban]))
            g_m2 = float(np.nansum(green[urban]) * grid.res ** 2)
            out["green_per_person"] = g_m2 / max(people, 1)
            out["people_urban"] = people
    if pop is not None:
        res = grid.mask & (np.nan_to_num(pop) > 0)
        tot = float(np.nansum(pop[res]))
        out["pop_300m_green"] = float(np.nansum(pop[res & (dist <= 300)]) / max(tot, 1) * 100)
        # 30%: share of people living in 300 m neighbourhoods with at least 30% tree canopy
        k = max(1, int(round(300 / grid.res)))
        tc, _ = block(np.where(grid.mask, tree, np.nan), k)
        tcx = expand(tc, k, grid.shape)
        out["canopy_300m"] = tcx
        out["pop_30pct_canopy"] = float(np.nansum(pop[res & (np.nan_to_num(tcx) >= 0.3)]) / max(tot, 1) * 100)
    return out


# ------------------------------------------------------------------ access
def access(grid, seeds, pop, limit_km, roads=None):
    """Distance (km) to the nearest facility and people beyond the limit. With a road mask the distance follows roads
    (kits/urban/network.py); otherwise straight line."""
    d_line = grid.distance_to(seeds) / 1000.0
    d_line[~grid.mask] = np.nan
    d, how = d_line, "straight line"
    if roads is not None and roads.any():
        from .network import network_distance_km, detour_ratio
        rr, cc = np.nonzero(seeds)
        dn = network_distance_km(grid, roads, list(zip(rr.tolist(), cc.tolist())))
        if dn is not None:
            d, how = np.where(np.isfinite(dn), dn, d_line), "along roads"
    d = d.astype(np.float32)
    res = grid.mask & (np.nan_to_num(pop) > 0) if pop is not None else None
    out = dict(dist=d, how=how)
    if how == "along roads":
        out["detour"] = detour_ratio(d, d_line, grid.mask)
    if res is not None:
        tot = float(np.nansum(pop[res]))
        out["pop_far"] = float(np.nansum(pop[res & (d > limit_km)]))
        out["pop_far_pct"] = out["pop_far"] / max(tot, 1) * 100
    return out


# ------------------------------------------------------------------ air
def no2(aoi, grid_c, days=24, months_back=12, progress=None):
    """Tropospheric NO2 (µmol/m²) from Sentinel-5P near-real-time granules on `days` dates spread over the last year.
    Each granule (~60 MB) is downloaded, read and deleted. Returns (mean map, monthly means {YYYY-MM: value}, dates used)."""
    end = dt.date.today() - dt.timedelta(days=7)
    start = end - dt.timedelta(days=30 * months_back)
    want = [start + dt.timedelta(days=int(i * (end - start).days / max(days, 1))) for i in range(days)]
    acc = np.zeros(grid_c.shape, np.float64)
    cnt = np.zeros(grid_c.shape, np.float64)
    monthly, used = {}, []
    tmp = tempfile.mkdtemp(prefix="areagiskit_s5p_")
    x0, y0, x1, y1 = aoi.bbox
    for i, d in enumerate(want):
        if progress:
            progress(i, len(want))
        try:
            items = [it for it in search("sentinel-5p-l2-netcdf", aoi.bbox, str(d), str(d + dt.timedelta(days=3)), max_items=60)
                     if it["properties"].get("s5p:product_name") == "no2" and it["properties"].get("s5p:processing_mode") == "NRTI"]
        except Exception as exc:
            skipped(exc)
            continue
        if not items:
            continue
        it = items[0]
        url = vsi(it["assets"]["no2"]["href"]).replace("/vsicurl/", "")
        trace([it])
        fn = os.path.join(tmp, "g.nc")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with open_url(req, timeout=120) as r, open(fn, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            def rd(v):
                return gdal.Open(f'NETCDF:"{fn}":/PRODUCT/{v}').ReadAsArray().astype(np.float64)
            lat, lon, val, qa = rd("latitude"), rd("longitude"), rd("nitrogendioxide_tropospheric_column"), rd("qa_value")
        except Exception as exc:
            skipped(exc)
            continue
        finally:
            try:
                os.remove(fn)
            except OSError:
                pass
        if lat.ndim == 3:
            lat, lon, val, qa = lat[0], lon[0], val[0], qa[0]
        if qa.max() > 1.5:
            qa = qa / 100.0
        m = (qa >= 0.75) & (val < 1e3) & (lon >= x0 - 0.1) & (lon <= x1 + 0.1) & (lat >= y0 - 0.1) & (lat <= y1 + 0.1)
        if m.sum() < 3:
            continue
        # each coarse cell takes the nearest satellite pixel (pixels ~5.5 x 3.5 km) if its centre is within 4 km
        v = val[m] * 1e6
        pts = np.array([aoi.to_local.TransformPoint(float(a), float(b))[:2] for a, b in zip(lon[m], lat[m])])
        if not hasattr(grid_c, "_xy"):
            cc, rr = np.meshgrid(np.arange(grid_c.cols), np.arange(grid_c.rows))
            grid_c._xy = (grid_c.gt[0] + (cc + 0.5) * grid_c.res, grid_c.gt[3] - (rr + 0.5) * grid_c.res)
        gx, gy = grid_c._xy
        d2 = (gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2
        j = d2.argmin(-1)
        near = d2.min(-1) <= 4000.0 ** 2
        day_sum = np.where(near, v[j], 0.0)
        day_n = near.astype(np.float64)
        if day_n.sum() == 0:
            continue
        acc += day_sum
        cnt += day_n
        key = item_date(it)[:7]
        monthly.setdefault(key, []).append(float(v.mean()))
        used.append(item_date(it))
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    if not used:
        return None, {}, []
    with np.errstate(all="ignore"):
        mean = np.where(cnt > 0, acc / cnt, np.nan).astype(np.float32)
    return mean, {k: float(np.mean(v)) for k, v in sorted(monthly.items())}, used
