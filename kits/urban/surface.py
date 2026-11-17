# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Surface albedo (Landsat 8/9), Local Climate Zones (global 100 m LCZ map) and a greening scenario."""

import warnings

import numpy as np
from osgeo import gdal

from ...core.data import trace, vsi, warp
from . import heat as HT

CREDIT_LCZ = ("Demuzere, M. et al. (2022). A global map of local climate zones to support earth system modelling and urban-scale "
              "environmental science. Earth System Science Data 14, 3835–3873. Map v3 on Zenodo (doi:10.5281/zenodo.8419340), CC BY 4.0.")
LCZ_URL = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif"

# Liang (2001) broadband shortwave albedo, coefficients for the Landsat TM/ETM+ bands that correspond to OLI B2, B4, B5, B6, B7
LIANG = dict(blue=0.356, red=0.130, nir08=0.373, swir16=0.085, swir22=0.072, const=-0.0018)

LCZ = {1: ("1 Compact high-rise", "#8c0000"), 2: ("2 Compact mid-rise", "#d10000"), 3: ("3 Compact low-rise", "#ff0000"),
       4: ("4 Open high-rise", "#bf4d00"), 5: ("5 Open mid-rise", "#ff6600"), 6: ("6 Open low-rise", "#ff9955"),
       7: ("7 Lightweight low-rise", "#faee05"), 8: ("8 Large low-rise", "#bcbcbc"), 9: ("9 Sparsely built", "#ffccaa"),
       10: ("10 Heavy industry", "#555555"), 11: ("A Dense trees", "#006a00"), 12: ("B Scattered trees", "#00aa00"),
       13: ("C Bush, scrub", "#648525"), 14: ("D Low plants", "#b9db79"), 15: ("E Bare rock or paved", "#000000"),
       16: ("F Bare soil or sand", "#fbf7ae"), 17: ("G Water", "#6a6aff")}


def liang_albedo(blue, red, nir, swir1, swir2):
    return (LIANG["blue"] * blue + LIANG["red"] * red + LIANG["nir08"] * nir + LIANG["swir16"] * swir1
            + LIANG["swir22"] * swir2 + LIANG["const"])


def _scene_albedo(items, grid):
    trace(items)
    sr = {}
    for b in ("blue", "red", "nir08", "swir16", "swir22"):
        a = warp([vsi(i["assets"][b]["href"]) for i in items], grid, resample="average", dtype=gdal.GDT_Float32,
                 src_nodata=0, dst_nodata=np.nan)
        sr[b] = a * 0.0000275 - 0.2
    qa = warp([vsi(i["assets"]["qa_pixel"]["href"]) for i in items], grid, resample="near", dtype=gdal.GDT_UInt16,
              src_nodata=1, dst_nodata=1)
    bad = (qa & 0b11111) > 0
    al = liang_albedo(sr["blue"], sr["red"], sr["nir08"], sr["swir16"], sr["swir22"])
    return np.where(bad | ~np.isfinite(al) | (al < 0) | (al > 1), np.nan, al).astype(np.float32)


def season_albedo(grid, year, months, dates):
    """Median albedo of the same clear Landsat days used for surface temperature."""
    by = {}
    for a, b in HT.months_between(year, months):
        by.update(HT.landsat_days(grid.aoi.bbox, a, b))
    arrs = []
    for d in dates:
        if d in by:
            try:
                arrs.append(_scene_albedo(by[d], grid))
            except Exception:
                continue
    if not arrs:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        al = np.nanmedian(np.stack(arrs), 0)
    al[~grid.mask] = np.nan
    return al.astype(np.float32)


def lcz_map(grid):
    """Global 100 m Local Climate Zone map (Demuzere et al. 2022), read by HTTP range requests (no download of the full file)."""
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    trace(url=LCZ_URL, label="lcz-global-v3")
    a = warp(["/vsicurl/" + LCZ_URL], grid, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
    a[~grid.mask] = 0
    return a


def slope_se(x, y):
    """Least-squares slope and its standard error."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = np.sqrt(np.sum(res ** 2) / max(n - 2, 1) / max(np.sum((x - x.mean()) ** 2), 1e-12))
    return float(b), float(se)
