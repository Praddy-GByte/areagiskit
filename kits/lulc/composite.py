# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Cloud-free, sensor-harmonised Landsat composites for any year since 1984.

Landsat 5, 7, 8 and 9 Collection 2 Level-2 surface reflectance, read only for the
study area, cloud/shadow/snow masked with the QA band, ETM+/TM reflectance
harmonised to OLI (Roy et al. 2016), then the per-pixel median of the season.
Also gives surface temperature (°C) and spectral indices.
"""

import calendar
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ...core.data import KitError, item_date, search, trace, vsi, warp

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
# Roy et al. (2016) ETM+ -> OLI OLS coefficients (intercept, slope), also used for TM
ROY = {"blue": (0.0003, 0.8474), "green": (0.0088, 0.8483), "red": (0.0061, 0.9047),
       "nir08": (0.0412, 0.8462), "swir16": (0.0254, 0.8937), "swir22": (0.0172, 0.9071)}
INDEX_INFO = {
    "NDVI": "Vegetation greenness",
    "SAVI": "Vegetation, corrected for soil",
    "NDWI": "Open water (McFeeters)",
    "MNDWI": "Open water, built-up suppressed (Xu)",
    "NDBI": "Built-up",
    "BSI": "Bare soil",
    "NDMI": "Vegetation moisture",
}


def _months_range(year, months):
    for m in months:
        last = calendar.monthrange(year, m)[1]
        yield f"{year}-{m:02d}-01", f"{year}-{m:02d}-{last}"


def scenes(aoi, year, months, max_cloud=60, max_dates=16):
    items = []
    for a, b in _months_range(year, months):
        items += search("landsat-c2-l2", aoi.bbox, a, b, query={"eo:cloud_cover": {"lt": max_cloud},
                        "platform": {"in": ["landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9"]}},
                        max_items=200)
    by = {}
    for it in items:
        by.setdefault((item_date(it), it["properties"]["platform"]), []).append(it)

    def pen(key, its):
        d, plat = key
        p = 0 if plat in ("landsat-8", "landsat-9") else (1 if plat in ("landsat-4", "landsat-5") else
                                                          (3 if d >= "2003-06-01" else 1))
        return (p, float(np.mean([i["properties"]["eo:cloud_cover"] for i in its])))

    ranked = sorted(by.items(), key=lambda kv: pen(*kv))
    # spread over the year: the best date of each month first, so crops' green season is seen
    chosen, months_seen = [], set()
    for key, its in ranked:
        m = key[0][5:7]
        if m not in months_seen:
            chosen.append(its)
            months_seen.add(m)
    for key, its in ranked:
        if len(chosen) >= max_dates:
            break
        if its not in chosen:
            chosen.append(its)
    return chosen[:max_dates]


def composite(aoi, grid, year, months, progress=None, max_cloud=60, max_dates=16):
    """Return dict with bands (6, rows, cols) reflectance, lst (°C), n_obs, dates, platforms."""
    groups = scenes(aoi, year, months, max_cloud, max_dates)
    if not groups:
        raise KitError(f"No usable Landsat scenes for {year} in months {months}. Try more months or a higher cloud limit.")
    for g in groups:
        trace(g)
    n = len(groups)

    def read(its, asset, dtype, resample, nod):
        hrefs = [vsi(i["assets"][asset]["href"]) for i in its if asset in i["assets"]]
        if not hrefs:
            return None
        return warp(hrefs, grid, resample=resample, dtype=dtype, src_nodata=nod, dst_nodata=nod)

    def masks(its):
        qa = read(its, "qa_pixel", gdal.GDT_UInt16, "near", 1)
        bad = (qa & 0b111111) > 0  # fill, dilated cloud, cirrus, cloud, shadow, snow
        return ~bad & grid.mask

    with ThreadPoolExecutor(max_workers=6) as ex:
        good = list(ex.map(masks, groups))
    plats = [its[0]["properties"]["platform"] for its in groups]
    out = np.full((6,) + grid.shape, np.nan, np.float32)
    keep = {}
    done = [0]
    total = 6 + 1
    for b, band in enumerate(BANDS):
        def one(k):
            a = read(groups[k], band, gdal.GDT_Float32, "bilinear", 0)
            if a is None:
                return None
            r = a * 2.75e-05 - 0.2
            if plats[k] not in ("landsat-8", "landsat-9"):
                i0, s = ROY[band]
                r = i0 + s * r
            r[~good[k] | (a == 0)] = np.nan
            return r.astype(np.float32)
        with ThreadPoolExecutor(max_workers=6) as ex:
            stack = np.stack([x if x is not None else np.full(grid.shape, np.nan, np.float32)
                              for x in ex.map(one, range(n))])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[b] = np.nanmedian(stack, axis=0)
        if band in ("red", "nir08", "green", "swir16"):
            keep[band] = stack
        del stack
        done[0] += 1
        if progress:
            progress(done[0] / total, f"{year}: {NAMES[b]} band from {n} dates")

    def lst_one(k):
        asset = "lwir11" if "lwir11" in groups[k][0]["assets"] else "lwir"
        a = read(groups[k], asset, gdal.GDT_Float32, "bilinear", 0)
        if a is None:
            return None
        c = a * 0.00341802 + 149.0 - 273.15
        c[~good[k] | (a == 0) | (c < -40) | (c > 80)] = np.nan
        return c.astype(np.float32)
    with ThreadPoolExecutor(max_workers=6) as ex:
        ls = [x for x in ex.map(lst_one, range(n)) if x is not None]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lst = np.nanmedian(np.stack(ls), axis=0) if ls else np.full(grid.shape, np.nan, np.float32)
        nobs = np.sum(np.stack(good), axis=0).astype(np.int16)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nd = (keep["nir08"] - keep["red"]) / (keep["nir08"] + keep["red"])
        mn = (keep["green"] - keep["swir16"]) / (keep["green"] + keep["swir16"])
        ndvi_p10, ndvi_p90 = np.nanpercentile(nd, 10, axis=0), np.nanpercentile(nd, 90, axis=0)
        mndwi_p90 = np.nanpercentile(mn, 90, axis=0)
    del keep, nd, mn
    out = np.clip(out, -0.05, 1.2)
    out[:, ~grid.mask] = np.nan
    lst[~grid.mask] = np.nan
    if progress:
        progress(1.0, f"{year}: composite ready")
    seasonal = {"NDVI_p10": np.clip(ndvi_p10, -1, 1).astype(np.float32),
                "NDVI_p90": np.clip(ndvi_p90, -1, 1).astype(np.float32),
                "MNDWI_p90": np.clip(mndwi_p90, -1, 1).astype(np.float32)}
    for v in seasonal.values():
        v[~grid.mask] = np.nan
    return {"bands": out, "lst": lst, "nobs": nobs, "seasonal": seasonal, "dates": sorted(its[0]["properties"]["datetime"][:10] for its in groups),
            "platforms": sorted(set(plats)), "year": year}


def indices(b):
    blue, green, red, nir, sw1, sw2 = b
    with np.errstate(divide="ignore", invalid="ignore"):
        d = {
            "NDVI": (nir - red) / (nir + red),
            "SAVI": 1.5 * (nir - red) / (nir + red + 0.5),
            "NDWI": (green - nir) / (green + nir),
            "MNDWI": (green - sw1) / (green + sw1),
            "NDBI": (sw1 - nir) / (sw1 + nir),
            "BSI": ((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue)),
            "NDMI": (nir - sw1) / (nir + sw1),
        }
    return {k: np.clip(v, -1, 1).astype(np.float32) for k, v in d.items()}


def _box(a, r):
    """Mean over a (2r+1)x(2r+1) window with edge padding (integral image)."""
    p = np.pad(a, r + 1, mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return ((s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k))[: a.shape[0], : a.shape[1]].astype(np.float32)


def texture_extended(nir, ndvi, r=2, levels=8):
    """GLCM-like texture without scipy/skimage: NIR is quantised to `levels` grey levels; for horizontal and vertical
    neighbour pairs, contrast = (i-j)² and homogeneity = 1/(1+(i-j)²) are averaged over a (2r+1)² window
    (Haralick et al. 1973, mean over the two offsets). Plus local standard deviation of NDVI over the same window."""
    x = np.nan_to_num(nir, nan=np.nanmedian(nir) if np.isfinite(nir).any() else 0)
    edges = np.nanpercentile(x, np.linspace(0, 100, levels + 1)[1:-1])
    q = np.digitize(x, edges).astype(np.float32)
    dh = np.zeros_like(q)
    dv = np.zeros_like(q)
    dh[:, :-1] = q[:, 1:] - q[:, :-1]
    dv[:-1, :] = q[1:, :] - q[:-1, :]
    con = _box((dh ** 2 + dv ** 2) / 2, r)
    hom = _box((1 / (1 + dh ** 2) + 1 / (1 + dv ** 2)) / 2, r)
    v = np.nan_to_num(ndvi, nan=0.0)
    m = _box(v, r)
    sd = np.sqrt(np.maximum(_box(v * v, r) - m * m, 0)).astype(np.float32)
    bad = ~np.isfinite(nir)
    for a in (con, hom, sd):
        a[bad] = np.nan
    return [con, hom, sd], ["NIR contrast (GLCM-like)", "NIR homogeneity (GLCM-like)", "NDVI local std (5x5)"]


def features(comp, elev=None, slope=None, extended_texture=False):
    """Feature stack (F, rows, cols) for classification, plus names."""
    b = comp["bands"]
    ix = indices(b)
    feats = [b[i] for i in range(6)] + [ix[k] for k in ("NDVI", "MNDWI", "NDBI", "BSI", "NDMI", "SAVI")]
    names = NAMES + ["NDVI", "MNDWI", "NDBI", "BSI", "NDMI", "SAVI"]
    # texture: local variability of NIR (buildings and roads make a rough surface)
    from numpy.lib.stride_tricks import sliding_window_view
    nir = np.nan_to_num(b[3], nan=np.nanmedian(b[3]) if np.isfinite(b[3]).any() else 0)
    w = sliding_window_view(np.pad(nir, 1, mode="edge"), (3, 3))
    tex = w.std(axis=(-1, -2)).astype(np.float32)
    tex[~np.isfinite(b[3])] = np.nan
    feats.append(tex)
    names.append("NIR texture")
    if extended_texture:
        tx, tn = texture_extended(b[3], ix["NDVI"])
        feats += tx
        names += tn
    for k, v in comp["seasonal"].items():
        feats.append(v)
        names.append(k.replace("_p", " p"))
        ix[k] = v
    if elev is not None:
        feats.append(elev.astype(np.float32))
        names.append("Elevation")
    if slope is not None:
        feats.append(slope.astype(np.float32))
        names.append("Slope")
    return np.stack(feats), names, ix
