# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Distance along roads (OpenStreetMap) instead of straight lines.

Roads are burnt into the study grid; a Dijkstra search runs over road cells (8-neighbour steps, so junctions are where
roads touch on the grid) from the facilities, each snapped to its nearest road cell. Every other cell then gets the road
distance of its nearest road cell plus the straight walk to that cell. Distances only, with no speeds, one-way streets or
turn rules. If roads cannot be read, callers fall back to straight-line distance and say so."""

import heapq
import json
import math
import urllib.parse
import urllib.request

import numpy as np
from osgeo import ogr

from ...core.data import UA, KitError
from ...core.osm import ENDPOINTS
from ...core.net import open_url

ROAD_CLASSES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|road"


def fetch_roads(bbox, timeout=40, buffer_deg=0.02):
    """All drivable/walkable road ways in and around the area as lists of (lon, lat)."""
    x0, y0, x1, y1 = bbox
    b = f"{y0 - buffer_deg},{x0 - buffer_deg},{y1 + buffer_deg},{x1 + buffer_deg}"
    q = f'[out:json][timeout:{timeout}];way["highway"~"^({ROAD_CLASSES})(_link)?$"]({b});out geom;'
    last = None
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": UA})
            with open_url(req, timeout=timeout + 15) as r:
                d = json.load(r)
            return [[(p["lon"], p["lat"]) for p in e.get("geometry", [])] for e in d.get("elements", []) if e.get("geometry")]
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise KitError(f"OpenStreetMap roads could not be read ({last}).")


def road_mask(grid, lines):
    """Burn road polylines (lon, lat) into the grid."""
    geoms = []
    for ln in lines:
        if len(ln) < 2:
            continue
        g = ogr.Geometry(ogr.wkbLineString)
        for lon, lat in ln:
            x, y, _ = grid.aoi.to_local.TransformPoint(lon, lat)
            g.AddPoint_2D(x, y)
        geoms.append(g)
    if not geoms:
        return np.zeros(grid.shape, bool)
    return grid.rasterize(geoms, all_touched=True)


_STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def road_distance(roads, sources_rc, res, snap_m=2000.0):
    """Shortest distance (m) along road cells from any source cell. Sources are snapped to the nearest road cell within
    `snap_m` (searching a square window); the snap distance is added. Returns float32 array (inf off-road / unreached)."""
    R, C = roads.shape
    dist = np.full(roads.shape, np.inf, np.float64)
    heap = []
    w = max(1, int(math.ceil(snap_m / res)))
    rr_all, cc_all = np.nonzero(roads)
    if rr_all.size == 0:
        return dist.astype(np.float32)
    for r, c in sources_rc:
        if not (0 <= r < R and 0 <= c < C):
            continue
        r0, r1, c0, c1 = max(r - w, 0), min(r + w + 1, R), max(c - w, 0), min(c + w + 1, C)
        sub = roads[r0:r1, c0:c1]
        if not sub.any():
            continue
        sr, sc = np.nonzero(sub)
        d = np.hypot(sr + r0 - r, sc + c0 - c) * res
        i = int(np.argmin(d))
        tr, tc = int(sr[i] + r0), int(sc[i] + c0)
        if d[i] < dist[tr, tc]:
            dist[tr, tc] = d[i]
            heapq.heappush(heap, (d[i], tr, tc))
    diag = res * math.sqrt(2.0)
    while heap:
        d0, r, c = heapq.heappop(heap)
        if d0 > dist[r, c]:
            continue
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and roads[nr, nc]:
                nd = d0 + (diag if dr and dc else res)
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist.astype(np.float32)


def network_distance_km(grid, roads, sources_rc, snap_m=2000.0):
    """Distance (km) from every cell to the nearest source along roads: walk to the nearest road cell, then follow roads."""
    from ..flood.depth import nearest_value
    rd = road_distance(roads, sources_rc, grid.res, snap_m)
    reached = np.isfinite(rd)
    if not reached.any():
        return None
    on_road = nearest_value(np.where(reached, rd, np.nan).astype(np.float32), reached)
    walk = grid.distance_to(reached)
    out = (on_road + walk) / 1000.0
    out[~grid.mask] = np.nan
    return out.astype(np.float32)


def detour_ratio(net_km, straight_km, mask):
    """Median ratio of road to straight-line distance (a plausibility check; typically 1.2–1.6 in cities)."""
    ok = mask & np.isfinite(net_km) & np.isfinite(straight_km) & (straight_km > 0.5)
    if ok.sum() < 20:
        return None
    return float(np.median(net_km[ok] / straight_km[ok]))
