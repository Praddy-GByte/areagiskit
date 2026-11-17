# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Context layers for landslide studies: active faults (GEM), roads (OpenStreetMap), vegetation (Sentinel-2)."""

import json
import os
import urllib.request

import numpy as np
from osgeo import ogr

from ...core.data import UA, trace

GEM = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson"
CREDIT_GEM = ("Active faults: GEM Global Active Faults Database (Styron & Pagani 2020, Earthquake Spectra 36, 160–180), "
              "CC BY-SA 4.0.")
CACHE = os.path.join(os.path.expanduser("~"), ".areagiskit_cache")


def _gem():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "gem_active_faults.geojson")
    if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
        req = urllib.request.Request(GEM, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=180) as r, open(p + ".part", "wb") as fh:
            fh.write(r.read())
        os.replace(p + ".part", p)
    trace(url=GEM, label="gem-active-faults")
    return p


def fault_distance(grid, search_km=50):
    """Distance (m) to the nearest mapped active fault, and the faults near the area (lon/lat lines)."""
    p = _gem()
    x0, y0, x1, y1 = grid.aoi.bbox
    d = search_km / 111.0
    ds = ogr.Open(p)
    lyr = ds.GetLayer(0)
    lyr.SetSpatialFilterRect(x0 - d, y0 - d, x1 + d, y1 + d)
    lines, names = [], []
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        gl = g.Clone()
        gl.Transform(grid.aoi.to_local)
        lines.append(gl)
        names.append(f.GetField("name") if f.GetFieldIndex("name") >= 0 else "")
    if not lines:
        return None, []
    seed = grid.rasterize(lines, all_touched=True) > 0
    if not seed.any():
        # faults outside the grid: distance to the nearest one, computed from geometry
        pts = np.argwhere(grid.mask)[::max(1, grid.mask.sum() // 2000)]
        best = np.inf
        for r, c in pts:
            x = grid.gt[0] + (c + 0.5) * grid.res
            y = grid.gt[3] - (r + 0.5) * grid.res
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint_2D(x, y)
            best = min(best, min(l.Distance(pt) for l in lines))
        out = np.full(grid.shape, best, np.float32)
        out[~grid.mask] = np.nan
        return out, [n for n in names if n]
    dist = grid.distance_to(seed).astype(np.float32)
    dist[~grid.mask] = np.nan
    return dist, [n for n in names if n]


def road_distance(grid, roads):
    if not roads:
        return None
    geoms = []
    for line in roads:
        if len(line) < 2:
            continue
        g = ogr.Geometry(ogr.wkbLineString)
        for lon, lat in line:
            x, y, _ = grid.aoi.to_local.TransformPoint(lon, lat)
            g.AddPoint_2D(x, y)
        geoms.append(g)
    if not geoms:
        return None
    seed = grid.rasterize(geoms, all_touched=True) > 0
    d = grid.distance_to(seed).astype(np.float32)
    d[~grid.mask] = np.nan
    return d
