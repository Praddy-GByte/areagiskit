# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Ready-made readers for the open datasets the kits share."""

import math

import numpy as np
from osgeo import gdal

from .data import KitError, exists, search, trace, warp, warp_items, item_date

# Impact Observatory / Microsoft / Esri annual 10 m land cover classes and colours
LULC = {
    1: ("Water", "#419bdf"),
    2: ("Trees", "#397d49"),
    4: ("Flooded vegetation", "#7a87c6"),
    5: ("Crops", "#e49635"),
    7: ("Built area", "#c4281b"),
    8: ("Bare ground", "#a59b8f"),
    9: ("Snow/ice", "#a8ebff"),
    10: ("Clouds", "#616161"),
    11: ("Rangeland", "#e3e2c3"),
}
LULC_YEARS = list(range(2017, 2024))
WC_TO_IO = {10: 2, 20: 11, 30: 11, 40: 5, 50: 7, 60: 8, 70: 9, 80: 1, 90: 4, 95: 4, 100: 11}

CREDIT_LULC = "Land cover: Impact Observatory, Microsoft and Esri, 10 m Annual LULC v02, CC BY 4.0."
CREDIT_WC = "ESA WorldCover 2021 © ESA WorldCover project, CC BY 4.0."
CREDIT_DEM = "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA."
CREDIT_HAND = "Global 30m Height Above Nearest Drainage (HAND), © 2022 Alaska Satellite Facility, CC0, accessed from https://registry.opendata.aws/glo-30-hand; produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018."
CREDIT_JRC = "Global Surface Water: Pekel et al. (2016), EC JRC / Google, CC BY 4.0."
CREDIT_S1 = "Sentinel-1 RTC: contains modified Copernicus Sentinel data, processed by Microsoft (Catalyst)."
CREDIT_S2 = "Sentinel-2 L2A: contains modified Copernicus Sentinel data."
CREDIT_LS = "Landsat Collection 2 Level-2: courtesy of the U.S. Geological Survey."
CREDIT_OSM = "Facilities: © OpenStreetMap contributors, ODbL."
CREDIT_PC = "Data served by Microsoft Planetary Computer; HAND by ASF on AWS Open Data. No account needed."


def lulc(grid, year):
    items = search("io-lulc-annual-v02", grid.aoi.bbox, f"{year}-01-01", f"{year}-12-31")
    items = [i for i in items if item_date(i)[:4] == str(year)]
    if not items:
        raise KitError(f"No land cover map found for {year} in this area.")
    a = warp_items(items, "data", grid, resample="near" if grid.res <= 10 else "mode",
                   dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
    a[~grid.mask] = 0
    return a, [i["id"] for i in items]


def worldcover(grid):
    items = search("esa-worldcover", grid.aoi.bbox, "2021-01-01", "2021-12-31")
    if not items:
        return None
    a = warp_items(items, "map", grid, resample="near" if grid.res <= 10 else "mode",
                   dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
    lut = np.zeros(256, np.uint8)
    for k, v in WC_TO_IO.items():
        lut[k] = v
    out = lut[a]
    out[~grid.mask] = 0
    return out


def dem(grid):
    items = search("cop-dem-glo-30", grid.aoi.bbox, "2000-01-01", "2030-12-31")
    if not items:
        raise KitError("No elevation data (Copernicus DEM) for this area.")
    a = warp_items(items, "data", grid, resample="bilinear", dtype=gdal.GDT_Float32,
                   src_nodata=-32767, dst_nodata=np.nan)
    a[~grid.mask] = np.nan
    return a


def _hand_tiles(bbox):
    x0, y0, x1, y1 = bbox
    urls = []
    for lat in range(math.floor(y0), math.floor(y1) + 1):
        for lon in range(math.floor(x0), math.floor(x1) + 1):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            urls.append(f"https://glo-30-hand.s3.amazonaws.com/v1/2021/"
                        f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_HAND.tif")
    return [u for u in urls if exists(u)]


def hand(grid):
    """Height Above Nearest Drainage (m), 30 m, from ASF's global GLO-30 HAND."""
    urls = _hand_tiles(grid.aoi.bbox)
    if not urls:
        raise KitError("No HAND data for this area (open ocean or outside coverage).")
    for u in urls:
        trace(url=u, label="glo-30-hand")
    a = warp(["/vsicurl/" + u for u in urls], grid, resample="bilinear",
             dtype=gdal.GDT_Float32, dst_nodata=np.nan)
    a[~grid.mask] = np.nan
    return a


def water_occurrence(grid):
    """JRC Global Surface Water occurrence 1984-2021, percent of time with water."""
    items = search("jrc-gsw", grid.aoi.bbox, "1984-01-01", "2030-12-31")
    if not items:
        return None
    a = warp_items(items, "occurrence", grid, resample="average", dtype=gdal.GDT_Float32,
                   src_nodata=255, dst_nodata=np.nan)
    a = np.where(np.isnan(a), 0, a)
    a[~grid.mask] = np.nan
    return a


def slope_aspect(elev, grid):
    """Slope (degrees) and aspect from an elevation array on the grid."""
    z = np.where(np.isfinite(elev), elev, np.nanmean(elev) if np.isfinite(elev).any() else 0)
    gy, gx = np.gradient(z, grid.res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
    slope[~np.isfinite(elev)] = np.nan
    return slope, aspect


def relief(elev, radius_px):
    """Local relief: max - min elevation in a square window (box filter via cumulative max)."""
    from numpy.lib.stride_tricks import sliding_window_view
    z = np.where(np.isfinite(elev), elev, np.nanmin(elev) if np.isfinite(elev).any() else 0)
    k = 2 * radius_px + 1
    zp = np.pad(z, radius_px, mode="edge")
    w = sliding_window_view(zp, (k, k))
    # reduce in two passes to keep memory down
    mx = w.max(axis=-1).max(axis=-1)
    mn = w.min(axis=-1).min(axis=-1)
    r = (mx - mn).astype(np.float32)
    r[~np.isfinite(elev)] = np.nan
    return r
