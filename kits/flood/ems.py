# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Independent expert reference: Copernicus EMS Rapid Mapping flood delineations.

When a Copernicus Emergency Management Service activation covers the study area, its
analyst-made 'observed flood extent' polygons are downloaded and compared with the maps.
"""

import datetime as dt
import math
import os
import tempfile
import urllib.request

import numpy as np
from osgeo import ogr, osr

from ...core.data import UA, get_json, trace

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/"
CREDIT_EMS = ("Copernicus Emergency Management Service (© European Union), Rapid Mapping delineation products "
              "(observed flood extent), https://rapidmapping.emergency.copernicus.eu.")


def _list():
    out, url = [], API + "public-activations-info/?limit=100&offset=0"
    while url and len(out) < 3000:
        d = get_json(url, timeout=60)
        out += d.get("results", [])
        url = d.get("next")
    return out


def find(aoi, date=None, days=40, max_km=400):
    """Flood activations near the area (and the event date if given), most relevant first."""
    try:
        acts = _list()
    except Exception:
        return []
    cx, cy = aoi.center
    res = []
    for a in acts:
        if a.get("category") != "Flood":
            continue
        try:
            px, py = [float(v) for v in a["centroid"].split("(")[1].rstrip(")").split()]
        except Exception:
            continue
        dkm = math.hypot((px - cx) * 111 * math.cos(math.radians(cy)), (py - cy) * 111)
        if dkm > max_km:
            continue
        et = dt.date.fromisoformat(a["eventTime"][:10])
        if date is not None and abs((et - date).days) > days:
            continue
        res.append((abs((et - date).days) if date else 0, dkm, a))
    return [a for *_, a in sorted(res, key=lambda r: (r[0], r[1]))]


def products(code, aoi):
    """Delineation products whose extent overlaps the study area: [(product, aoi_name, overlap_share)]."""
    d = get_json(API + f"public-activations/?code={code}", timeout=60)
    if not d.get("results"):
        return []
    act = d["results"][0]
    out = []
    for a in act.get("aois", []):
        for p in a.get("products", []):
            if p.get("type") != "DEL" or not p.get("downloadPath"):
                continue
            g = ogr.CreateGeometryFromWkt(p.get("extent") or a.get("extent") or "")
            if g is None:
                continue
            g.AssignSpatialReference(aoi.geom.GetSpatialReference())
            if not g.Intersects(aoi.geom):
                continue
            inter = g.Intersection(aoi.geom)
            share = inter.GetArea() / max(aoi.geom.GetArea(), 1e-12)
            img = p.get("images") or [{}]
            out.append(dict(code=code, aoi=a.get("name"), url=p["downloadPath"], share=share,
                            time=img[0].get("acquisitionTime"), sensor=img[0].get("sensorName"),
                            monitoring=p.get("monitoringNumber", 0), extent=g.ExportToWkt()))
    return sorted(out, key=lambda r: -r["share"])


def load(product, grid, workdir=None):
    """Download a delineation product and rasterize observed flood polygons. Returns (flood, in_product_area)."""
    workdir = workdir or tempfile.mkdtemp(prefix="ems_")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, os.path.basename(product["url"]))
    req = urllib.request.Request(product["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as fh:
        fh.write(r.read())
    trace(url=product["url"], label="copernicus-ems")
    from osgeo import gdal
    import zipfile
    names = zipfile.ZipFile(path).namelist()
    shp = [n for n in names if "observedEvent" in n and "previous" not in n.lower() and n.endswith(".shp")]
    aoi_shp = [n for n in names if "areaOfInterest" in n and n.endswith(".shp")]
    if not shp:
        return None, None
    ct_cache = {}

    def geoms(name, flood_only):
        ds = ogr.Open("/vsizip/" + path + "/" + name)
        lyr = ds.GetLayer(0)
        src = lyr.GetSpatialRef()
        key = src.ExportToWkt() if src else "4326"
        if key not in ct_cache:
            s = src or osr.SpatialReference()
            if not src:
                s.ImportFromEPSG(4326)
            s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ct_cache[key] = osr.CoordinateTransformation(s, grid.aoi.srs)
        out = []
        for f in lyr:
            if flood_only:
                # keep standing water only: 'Flooded area'; drop 'Flood trace' (wet soil, debris) and
                # 'Previous flooded area' (an earlier date in monitoring products)
                txt = " ".join(str(f.GetField(i) or "") for i in range(f.GetFieldCount())).lower()
                if "previous" in txt or "trace" in txt:
                    continue
                if "flooded area" not in txt and "inundat" not in txt:
                    continue
            g = f.GetGeometryRef()
            if g is None:
                continue
            g = g.Clone()
            g.Transform(ct_cache[key])
            out.append(g)
        return out

    fl = geoms(shp[0], True)
    area = geoms(aoi_shp[0], False) if aoi_shp else []
    flood = grid.rasterize(fl) > 0 if fl else np.zeros(grid.shape, bool)
    inside = (grid.rasterize(area) > 0) if area else np.ones(grid.shape, bool)
    return flood & grid.mask, inside & grid.mask


def scores(pred, ref, where):
    """Agreement of a flood map with a reference flood map inside `where`."""
    p, r = pred[where], ref[where]
    tp = float((p & r).sum())
    fp = float((p & ~r).sum())
    fn = float((~p & r).sum())
    tn = float((~p & ~r).sum())
    n = tp + fp + fn + tn
    if n == 0 or (tp + fn) == 0:
        return None
    oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(csi=tp / max(tp + fp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                pa=tp / max(tp + fn, 1), ua=tp / max(tp + fp, 1), oa=oa,
                kappa=(oa - pe) / (1 - pe) if pe < 1 else 0, bias=(tp + fp) / max(tp + fn, 1))
