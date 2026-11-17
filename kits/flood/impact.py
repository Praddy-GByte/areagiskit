# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Exposure, damage and the independent GloFAS river flood hazard model."""

import json
import math

import numpy as np
from osgeo import gdal, ogr

from ...core.data import get_json, trace, warp

GLOFAS = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/"
CREDIT_GLOFAS = ("Global river flood hazard maps v2.1 (GloFAS, LISFLOOD-FP), Copernicus Emergency Management Service, "
                 "© European Union 1995-2026, ODbL 1.0. Baugh, C. et al. (2024) Global river flood hazard maps.")
_TILES = None


def _tiles():
    global _TILES
    if _TILES is None:
        _TILES = get_json(GLOFAS + "tile_extents.geojson", timeout=60)["features"]
    return _TILES


def glofas(grid, rp=100):
    """Modelled river flood depth (m) for a return period; NaN = not flooded in the model."""
    try:
        x0, y0, x1, y1 = grid.aoi.bbox
        box = ogr.CreateGeometryFromWkt(f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))")
        srcs = []
        for f in _tiles():
            g = ogr.CreateGeometryFromJson(json.dumps(f["geometry"]))
            if g.Intersects(box):
                n = f"ID{f['properties']['id']}_{f['properties']['name']}_RP{rp}_depth.tif"
                u = f"{GLOFAS}RP{rp}/{n}"
                trace(url=u, label=f"glofas-rp{rp}")
                srcs.append("/vsicurl/" + u)
        if not srcs:
            return None
        a = warp(srcs, grid, resample="bilinear", dtype=gdal.GDT_Float32, src_nodata=-9999, dst_nodata=np.nan)
    except Exception:
        return None
    a = np.where(a > 0, a, np.nan)
    a[~grid.mask] = np.nan
    return a


# --------------------------------------------------------------------- damage
# Huizinga, J., de Moel, H., Szewczyk, W. (2017). Global flood depth-damage functions. JRC105688.
DEPTHS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6], float)
RESIDENTIAL = {
    "Europe": [0, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95, 1.00],
    "North America": [0, 0.44, 0.58, 0.68, 0.78, 0.85, 0.92, 0.96, 1.00],
    "South America": [0, 0.49, 0.71, 0.84, 0.95, 0.98, 1.00, 1.00, 1.00],
    "Asia": [0, 0.33, 0.49, 0.62, 0.72, 0.87, 0.93, 0.98, 1.00],
    "Africa": [0, 0.22, 0.38, 0.53, 0.64, 0.82, 0.90, 0.96, 1.00],
    "Oceania": [0, 0.48, 0.64, 0.71, 0.79, 0.93, 0.97, 0.98, 1.00],
}
AGRICULTURE = {
    "Asia": [0, 0.17, 0.37, 0.51, 0.56, 0.69, 0.83, 0.97, 1.00],
    "Africa": [0, 0.24, 0.47, 0.74, 0.92, 1.00, 1.00, 1.00, 1.00],
    "North America": [0.02, 0.27, 0.48, 0.56, 0.61, 0.76, 0.88, 0.95, 1.00],
}


def continent(lon, lat):
    if lat > 35 and -25 <= lon <= 45:
        return "Europe"
    if lon < -30:
        return "North America" if lat > 13 else "South America"
    if lat < -10 and lon > 110:
        return "Oceania"
    if lon < 52 and lat < 37 and not (lon > 34 and lat > 12):
        return "Africa"
    return "Asia"


def damage_factor(depth, curve):
    d = np.nan_to_num(depth, nan=0.0)
    f = np.interp(np.clip(d, 0, 6), DEPTHS, np.asarray(curve, float))
    return np.where(d > 0, f, 0.0).astype(np.float32)


def agri_curve(cont):
    if cont in AGRICULTURE:
        return AGRICULTURE[cont], cont
    avg = np.mean([AGRICULTURE[k] for k in AGRICULTURE], axis=0)
    return list(avg), "mean of Asia, Africa and North America (no curve published for " + cont + ")"


def road_length(roads, grid, flood):
    """km of major road on flooded cells (roads: lists of lon/lat vertices)."""
    if not roads:
        return None
    tot = wet = 0.0
    for line in roads:
        pts = []
        for lon, lat in line:
            x, y, _ = grid.aoi.to_local.TransformPoint(lon, lat)
            pts.append((x, y))
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(xb - xa, yb - ya)
            n = max(1, int(seg / grid.res))
            for i in range(n):
                t = (i + 0.5) / n
                x, y = xa + (xb - xa) * t, ya + (yb - ya) * t
                c = int((x - grid.gt[0]) / grid.res)
                r = int((grid.gt[3] - y) / grid.res)
                if 0 <= r < grid.rows and 0 <= c < grid.cols and grid.mask[r, c]:
                    tot += seg / n
                    if flood[r, c]:
                        wet += seg / n
    return tot / 1000, wet / 1000
