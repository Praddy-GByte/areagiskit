# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Urban heat: Landsat surface temperature, MODIS day/night temperature, surface urban heat island, cooling of parks and water."""

import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import item_date, search, trace, vsi, warp

CREDIT_MODIS_LST = ("MODIS Land Surface Temperature MOD11A1/MOD11A2 and MYD11A2 v6.1 (1 km, Terra 10:30/22:30, Aqua 13:30/01:30), "
                    "NASA LP DAAC, via Microsoft Planetary Computer.")


def to_grid(arr, src, dst, alg="average", nodata=np.nan):
    """Resample an array on grid `src` to grid `dst` (same projection) with GDAL."""
    drv = gdal.GetDriverByName("MEM")
    a = drv.Create("", src.cols, src.rows, 1, gdal.GDT_Float32)
    a.SetGeoTransform(src.gt)
    a.SetProjection(src.wkt)
    b = a.GetRasterBand(1)
    b.SetNoDataValue(-9999.0)
    b.WriteArray(np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32))
    o = gdal.Warp("", a, format="MEM", outputBounds=dst.bounds, xRes=dst.res, yRes=dst.res, dstSRS=dst.wkt,
                  resampleAlg=alg, srcNodata=-9999.0, dstNodata=-9999.0, outputType=gdal.GDT_Float32)
    out = o.GetRasterBand(1).ReadAsArray()
    return np.where(out == -9999.0, nodata, out).astype(np.float32)


# ------------------------------------------------------------------ Landsat
def landsat_days(bbox, start, end, max_cloud=40):
    items = search("landsat-c2-l2", bbox, str(start), str(end),
                   query={"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8", "landsat-9"]}}, max_items=500)
    by = {}
    for it in items:
        by.setdefault(item_date(it), []).append(it)
    return by


def lst_scene(items, grid):
    """Landsat Collection 2 surface temperature (°C) for one day; clouds, shadows and fill removed."""
    trace(items)
    st = warp([vsi(i["assets"]["lwir11"]["href"]) for i in items], grid, resample="bilinear",
              dtype=gdal.GDT_Float32, src_nodata=0, dst_nodata=np.nan)
    qa = warp([vsi(i["assets"]["qa_pixel"]["href"]) for i in items], grid, resample="near",
              dtype=gdal.GDT_UInt16, src_nodata=1, dst_nodata=1)
    bad = (qa & 0b11111) > 0
    c = st * 0.00341802 + 149.0 - 273.15
    return np.where(bad | ~np.isfinite(c) | (c < -40) | (c > 80), np.nan, c).astype(np.float32)


def months_between(year, months):
    out = []
    for m in months:
        y = year - (1 if (m >= 10 and min(months) <= 3) else 0)   # e.g. Dec–Feb summer spans two years
        last = (dt.date(y + (m == 12), (m % 12) + 1, 1) - dt.timedelta(days=1)).day
        out.append((dt.date(y, m, 1), dt.date(y, m, last)))
    return out


def season_lst(grid, year, months, n=8, min_valid=0.6, threads=4):
    """Median hot-season LST from up to `n` of the clearest Landsat 8/9 days. Returns (lst, dates, per-date arrays)."""
    by = {}
    for a, b in months_between(year, months):
        by.update(landsat_days(grid.aoi.bbox, a, b))
    order = sorted(by, key=lambda d: np.mean([i["properties"]["eo:cloud_cover"] for i in by[d]]))

    def one(d):
        try:
            return d, lst_scene(by[d], grid)
        except Exception:
            return d, None
    got = []
    with ThreadPoolExecutor(threads) as ex:
        for d, a in ex.map(one, order[:n + 4]):
            if a is not None and np.isfinite(a[grid.mask]).mean() >= min_valid:
                got.append((d, a))
            if len(got) >= n:
                break
    if not got:
        return None, [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lst = np.nanmedian(np.stack([a for _, a in got]), 0)
    lst[~grid.mask] = np.nan
    return lst.astype(np.float32), sorted(d for d, _ in got), got


def yearly_suhi(aoi, grid_c, urban_c, rural_c, months, years, n=3, progress=None):
    """Hot-season Landsat LST for each year on a coarse grid: urban mean, rural mean and SUHI. Returns {year: dict}."""
    out = {}
    for i, y in enumerate(years):
        if progress:
            progress(i, len(years), y)
        try:
            lst, dates, _ = season_lst(grid_c, y, months, n=n, min_valid=0.5)
        except Exception:
            continue
        if lst is None:
            continue
        u, r = lst[urban_c & np.isfinite(lst)], lst[rural_c & np.isfinite(lst)]
        if u.size < 5 or r.size < 5:
            continue
        out[y] = dict(urban=float(u.mean()), rural=float(r.mean()), suhi=float(u.mean() - r.mean()), n=len(dates))
    return out


# ------------------------------------------------------------------ MODIS
def _modis(items, grid, var):
    a = warp([vsi(i["assets"][f"LST_{var}_1km"]["href"]) for i in items if f"LST_{var}_1km" in i["assets"]], grid,
             resample="near", dtype=gdal.GDT_Float32, src_nodata=0, dst_nodata=np.nan)
    q = warp([vsi(i["assets"][f"QC_{var}"]["href"]) for i in items if f"QC_{var}" in i["assets"]], grid,
             resample="near", dtype=gdal.GDT_Int16, src_nodata=-1, dst_nodata=-1)
    ok = ((q & 3) <= 1) & (((q >> 6) & 3) <= 1) & (q >= 0)    # good or other quality, LST error <= 2 K
    return np.where(ok & np.isfinite(a), a * 0.02 - 273.15, np.nan).astype(np.float32)


def modis_season(grid_k, year, months, collection="modis-11A2-061", platforms=("terra", "aqua"), threads=6):
    """Mean hot-season MODIS LST (day and night) for one year at 1 km. Returns dict var -> array."""
    items = []
    for a, b in months_between(year, months):
        items += search(collection, grid_k.aoi.bbox, str(a), str(b), max_items=200)
    by = {}
    for it in items:
        plat = "aqua" if it["id"].upper().startswith("MYD") else "terra"
        if plat in platforms:
            by.setdefault((plat, item_date(it)), []).append(it)
    out = {}
    for var in ("Day", "Night"):
        def job(k):
            try:
                return _modis(by[k], grid_k, var)
            except Exception:
                return None
        with ThreadPoolExecutor(threads) as ex:
            arrs = [a for a in ex.map(job, sorted(by)) if a is not None]
        if arrs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[var] = np.nanmean(np.stack(arrs), 0)
    return out


def modis_day(grid_k, day):
    """Terra MODIS daytime LST (10:30 overpass, like Landsat) for one date, for cross-checking Landsat."""
    items = [i for i in search("modis-11A1-061", grid_k.aoi.bbox, day, day, max_items=10) if i["id"].upper().startswith("MOD")]
    if not items:
        return None
    try:
        return _modis(items, grid_k, "Day")
    except Exception:
        return None


def compare(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    x, y = a[m].astype(float), b[m].astype(float)
    return dict(n=int(m.sum()), r=float(np.corrcoef(x, y)[0, 1]), bias=float((x - y).mean()), rmse=float(np.sqrt(((x - y) ** 2).mean())),
                ubrmse=float(np.std(x - y)))


# ------------------------------------------------------------------ heat island geometry
def urban_rural(built, grid, elev=None, water=None, core_frac=0.5, cell_m=500, max_dz=50.0):
    """Urban footprint: cells (cell_m) with at least `core_frac` built-up, largest connected clusters. Rural reference: a ring
    around it of about the same area, not built-up, not water, within ±max_dz m of the urban median elevation (Peng et al. 2012).
    Returns (urban mask, rural mask, ring width m, note)."""
    from ..flood.extra import label
    k = max(1, int(round(cell_m / grid.res)))
    R, C = grid.rows // k, grid.cols // k
    if R < 3 or C < 3:
        return None, None, None, "Area too small for an urban–rural comparison."
    f = built[:R * k, :C * k].reshape(R, k, C, k).mean((1, 3))
    core = np.zeros(grid.shape, bool)
    core[:R * k, :C * k] = np.repeat(np.repeat(f >= core_frac, k, 0), k, 1)
    core &= grid.mask
    if core.sum() * grid.px_km2 < 1.0:
        return None, None, None, "Less than 1 km² of dense built-up land: no urban core to compare with countryside."
    lab, n = label(core, grid)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= max(sizes.max() * 0.1, 1.0 / grid.px_km2)) + 1
    urban = np.isin(lab, keep)
    d = grid.distance_to(urban)
    ua = urban.sum()
    rural = None
    width = None
    note = ""
    for w in range(500, 20001, 250):
        ring = (d > 0) & (d <= w) & grid.mask & ~built.astype(bool)
        if water is not None:
            ring &= ~water
        if elev is not None:
            ue = np.nanmedian(elev[urban])
            ring &= np.abs(np.nan_to_num(elev, nan=1e9) - ue) <= max_dz
        if ring.sum() >= ua:
            rural, width = ring, w
            break
    if rural is None:
        ring = (d > 0) & grid.mask & ~built.astype(bool)
        if water is not None:
            ring &= ~water
        if ring.sum() * grid.px_km2 < 0.5:
            return urban, None, None, "No countryside inside the area to compare with; enlarge the area around the city."
        rural, width = ring, None
        note = "The countryside ring is smaller than the city; SUHI uses all non-built land in the area."
    return urban, rural, width, note


def distance_profile(lst, patches, grid, within, step_m=30, max_m=480):
    """Mean LST in rings around patches (green or water): returns (distances, mean LST, cooling intensity, cooling distance)."""
    d = grid.distance_to(patches)
    xs, ys = [0], [float(np.nanmean(lst[patches & np.isfinite(lst)])) if (patches & np.isfinite(lst)).any() else np.nan]
    for a in range(0, max_m, step_m):
        m = (d > a) & (d <= a + step_m) & within & np.isfinite(lst)
        if m.sum() < 30:
            break
        xs.append(a + step_m)
        ys.append(float(lst[m].mean()))
    ys = np.array(ys)
    if len(ys) < 4 or not np.isfinite(ys[0]):
        return xs, ys.tolist(), None, None
    # cooling distance: first ring after which temperature rises by less than 0.1 °C per step (Yu et al. 2017, first turning point)
    cd = xs[-1]
    for i in range(1, len(ys) - 1):
        if ys[i + 1] - ys[i] < 0.1:
            cd = xs[i]
            break
    i = xs.index(cd)
    return xs, ys.tolist(), float(ys[i] - ys[0]), float(cd)
