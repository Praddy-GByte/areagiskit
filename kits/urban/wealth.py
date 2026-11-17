# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Relative Wealth Index (Meta Data for Good; Chi et al. 2022): estimated relative wealth of 2.4 km tiles in 93 low- and
middle-income countries, from satellite imagery and connectivity data. Downloaded per country from the Humanitarian
Data Exchange (HDX), no account. Licence CC BY-NC 4.0: non-commercial use only.

The country file is chosen by intersecting the study area with Natural Earth country outlines, cached on disk after the
first download, and painted onto the study grid as 2.4 km squares (zoom-14 Bing tiles). A user can also pass a CSV with
latitude, longitude and rwi columns (for example a file downloaded by hand from HDX)."""

import csv
import io
import json
import math
import os
import tempfile
import time
import urllib.request

import numpy as np
from osgeo import ogr

from ...core.data import UA, get_json, trace

HDX_PKG = "https://data.humdata.org/api/3/action/package_show?id=relative-wealth-index"
NE_COUNTRIES = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
CREDIT_RWI = ("Relative Wealth Index: Chi, G., Fang, H., Chatterjee, S., Blumenstock, J. E. (2022). Microestimates of wealth for all "
              "low- and middle-income countries. PNAS 119(3), e2113658119. Meta Data for Good, via HDX. CC BY-NC 4.0 (non-commercial).")
ALIASES = {"India": "India and Pakistan", "Pakistan": "India and Pakistan", "eSwatini": "Swaziland", "Republic of the Congo": "Republic of Congo",
           "Dem. Rep. Congo": "Democratic Republic of the Congo", "Ivory Coast": "Côte d'Ivoire", "Laos": "Laos", "Vietnam": "Vietnam",
           "Moldova": "Moldova", "North Macedonia": "North Macedonia", "Turkey": "Turkey", "Cabo Verde": "Cape Verde"}


def _cache_dir():
    d = os.path.join(tempfile.gettempdir(), "areagiskit_cache")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _download(url, dest, tries=4):
    """Download to `dest`. HDX sometimes answers 202 with an empty body (a bot check); retry with back-off."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) > 100:
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dest)
                return dest
            last = RuntimeError("empty answer (the download service asked for a browser check)")
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(min(5 * 2 ** i, 40))
    raise RuntimeError(f"HDX download failed: {last}")


def countries(bbox):
    js = get_json(NE_COUNTRIES, cache=30 * 86400)
    x0, y0, x1, y1 = bbox
    box = ogr.CreateGeometryFromWkt(f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))")
    out = []
    for f in js["features"]:
        g = ogr.CreateGeometryFromJson(json.dumps(f["geometry"]))
        if g is not None and g.Intersects(box):
            p = f["properties"]
            out.append(p.get("NAME") or p.get("ADMIN"))
    if not out:   # coarse outlines miss small coastal areas: take the nearest country
        best = None
        c = ogr.CreateGeometryFromWkt(f"POINT({(x0 + x1) / 2} {(y0 + y1) / 2})")
        for f in js["features"]:
            g = ogr.CreateGeometryFromJson(json.dumps(f["geometry"]))
            if g is not None:
                d = g.Distance(c)
                if best is None or d < best[0]:
                    best = (d, f["properties"].get("NAME"))
        if best and best[0] < 1.0:
            out.append(best[1])
    return out


def _resources():
    js = get_json(HDX_PKG, cache=30 * 86400)
    res = {}
    for r in js["result"]["resources"]:
        n = r.get("name", "")
        if n.endswith("_relative_wealth_index.csv") or n.endswith("_relative_wealth_index.csv.csv"):
            res.setdefault(n.split("_relative_wealth_index")[0].strip(), r["url"])
    return res


def read_points(path, bbox, margin=0.05):
    x0, y0, x1, y1 = bbox
    pts = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            try:
                lat, lon, w = float(row["latitude"]), float(row["longitude"]), float(row["rwi"])
            except (KeyError, ValueError, TypeError):
                continue
            if x0 - margin <= lon <= x1 + margin and y0 - margin <= lat <= y1 + margin:
                pts.append((lon, lat, w))
    return pts


def fetch(bbox, user_csv=None):
    """Return (points, source_note) or (None, reason)."""
    if user_csv:
        pts = read_points(user_csv, bbox)
        return (pts, "your file") if pts else (None, "no RWI points of your file fall in the area")
    try:
        names = countries(bbox)
        res = _resources()
    except Exception as exc:  # noqa: BLE001
        return None, f"the country list or HDX catalogue could not be read ({exc})"
    files = []
    for n in names:
        key = ALIASES.get(n, n)
        url = res.get(key) or next((u for k, u in res.items() if k.lower() == key.lower()), None)
        if url:
            files.append((key, url))
    if not files:
        return None, ("the Relative Wealth Index covers low- and middle-income countries only"
                      + (f"; none for {', '.join(names)}" if names else ""))
    cd = _cache_dir() or tempfile.mkdtemp()
    pts, used = [], []
    for key, url in dict(files).items():
        dest = os.path.join(cd, "rwi_" + os.path.basename(url.split("?")[0]))
        try:
            if not (os.path.exists(dest) and os.path.getsize(dest) > 100):
                _download(url, dest)
            trace(url=url, label="meta-rwi")
            pts += read_points(dest, bbox)
            used.append(key)
        except Exception as exc:  # noqa: BLE001
            return None, f"HDX could not be reached for {key} ({exc}); download the file by hand from data.humdata.org and give it as 'wealth file'"
    if not pts:
        return None, "no RWI tiles fall in this area"
    return pts, f"HDX file for {', '.join(used)}"


def to_grid(points, grid):
    """Paint each 2.4 km tile (zoom-14 Bing tile, width 2.445 km × cos(lat) on the ground) onto the grid."""
    out = np.full(grid.shape, np.nan, np.float32)
    if not points:
        return out
    lat0 = float(np.mean([p[1] for p in points]))
    half = 0.5 * 2445.98 * math.cos(math.radians(lat0)) / grid.res
    h = max(1, int(round(half)))
    for lon, lat, w in points:
        r, c = grid.lonlat_to_pixel(lon, lat)
        r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, grid.rows), max(c - h, 0), min(c + h + 1, grid.cols)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = w
    out[~grid.mask] = np.nan
    return out
