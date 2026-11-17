# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Earthquake shaking as a landslide factor.

1. Long-term shaking hazard: peak ground acceleration (PGA, fraction of g) with a 10% chance of being exceeded in 50 years
   (475-year return period) on rock, from the GEM Global Seismic Hazard Map v2023.1 (3 arc-minutes, about 5.5 km). The
   GeoTIFF is read straight from its Zenodo zip by HTTP range requests (no account, no full download).
   Licence: CC BY-NC-SA 4.0: free for non-commercial use with credit; commercial use needs a licence from GEM.
2. Event shaking: if the study has an event date and USGS has a ShakeMap for an earthquake of magnitude 5 or more within
   300 km in the days around it, the ShakeMap PGA grid (USGS, public domain) is used as an event factor.

At 5.5 km a hazard grid only varies across larger study areas, so the factor is dropped automatically when fewer than
4 source cells cover the area or when PGA changes by less than 0.03 g inside it (it would then only add a smooth trend)."""

import datetime as dt
import math
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal

from ...core.data import get_json, trace, warp

GEM_ZIP = "https://zenodo.org/records/8409647/files/GEM-GSHM_PGA-475y-rock_v2023.zip?download=1"
GEM_TIF = "v2023_1_pga_475_rock_3min.tif"
GEM_CELL_DEG = 0.05
CREDIT_GEM = ("Earthquake hazard: GEM Global Seismic Hazard Map v2023.1, PGA with 10% probability of exceedance in 50 years on rock "
              "(Pagani, M. et al. 2023, GEM Foundation; doi:10.5281/zenodo.8409647). CC BY-NC-SA 4.0: non-commercial use only.")
CREDIT_SHAKEMAP = ("Event shaking: USGS ShakeMap (Worden, C. B. et al. 2020, ShakeMap Manual, USGS), via the USGS earthquake catalogue "
                   "API; public domain.")
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _cells_across(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) / GEM_CELL_DEG, (y1 - y0) / GEM_CELL_DEG


def hazard_pga(grid, min_cells=4, min_range=0.03):
    """GEM 475-year PGA (g) on the study grid. Returns (array or None, info)."""
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".zip,.tif,.tiff")
    src = "/vsizip/{/vsicurl/" + GEM_ZIP + "}/" + GEM_TIF
    trace(url=GEM_ZIP, label="gem-gshm-pga475-v2023.1")
    a = warp([src], grid, resample="bilinear", dtype=gdal.GDT_Float32, dst_nodata=np.nan)
    a = np.where((a >= 0) & (a < 5), a, np.nan).astype(np.float32)
    a[~grid.mask] = np.nan
    if not np.isfinite(a[grid.mask]).any():
        return None, dict(reason="no data")
    v = a[grid.mask & np.isfinite(a)]
    nx, ny = _cells_across(grid.aoi.bbox)
    info = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), cells=float(nx * ny), source="GEM 2023.1")
    if nx * ny < min_cells or (info["max"] - info["min"]) < min_range:
        info["dropped"] = True
        info["reason"] = (f"the hazard grid is about 5.5 km and covers this area with only {nx * ny:.1f} cells"
                          if nx * ny < min_cells else
                          f"PGA hardly changes inside the area ({info['min']:.2f}–{info['max']:.2f} g)")
        return None, info
    return a, info


# ------------------------------------------------------------------ USGS ShakeMap for an earthquake-triggered event
def find_shakemap(event_date, bbox, days_before=3, days_after=1, min_mag=5.0, radius_km=300):
    """Largest earthquake with a ShakeMap near the area around the event date. Returns event dict or None."""
    if not event_date:
        return None
    d = event_date if isinstance(event_date, dt.date) else dt.date.fromisoformat(str(event_date)[:10])
    x0, y0, x1, y1 = bbox
    q = dict(format="geojson", starttime=str(d - dt.timedelta(days=days_before)), endtime=str(d + dt.timedelta(days=days_after + 1)),
             minmagnitude=min_mag, latitude=f"{(y0 + y1) / 2:.4f}", longitude=f"{(x0 + x1) / 2:.4f}", maxradiuskm=radius_km,
             producttype="shakemap", orderby="magnitude")
    js = get_json(FDSN + "?" + urllib.parse.urlencode(q), timeout=60)
    feats = js.get("features") or []
    if not feats:
        return None
    f = feats[0]
    p = f["properties"]
    return dict(id=f["id"], mag=p.get("mag"), place=p.get("place"), time=dt.datetime.utcfromtimestamp(p["time"] / 1000).isoformat()[:16],
                detail=p.get("detail"), url=p.get("url"), lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1])


def parse_grid_xml(text):
    """ShakeMap grid.xml -> (lon0, lat_top, dlon, dlat, PGA array in g, top row first)."""
    root = ET.fromstring(text)
    ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    tag = (lambda t: f"s:{t}") if ns else (lambda t: t)
    spec = root.find(tag("grid_specification"), ns)
    nlon, nlat = int(spec.get("nlon")), int(spec.get("nlat"))
    lon_min, lat_max = float(spec.get("lon_min")), float(spec.get("lat_max"))
    dlon, dlat = float(spec.get("nominal_lon_spacing")), float(spec.get("nominal_lat_spacing"))
    fields = {f.get("name").upper(): int(f.get("index")) - 1 for f in root.findall(tag("grid_field"), ns)}
    data = np.array(root.find(tag("grid_data"), ns).text.split(), float).reshape(-1, len(fields))
    pga = data[:, fields["PGA"]] / 100.0                 # %g -> g
    lon, lat = data[:, fields["LON"]], data[:, fields["LAT"]]
    col = np.rint((lon - lon_min) / dlon).astype(int)
    row = np.rint((lat_max - lat) / dlat).astype(int)
    out = np.full((nlat, nlon), np.nan, np.float32)
    ok = (row >= 0) & (row < nlat) & (col >= 0) & (col < nlon)
    out[row[ok], col[ok]] = pga[ok]
    return lon_min, lat_max, dlon, dlat, out


def event_pga(grid, ev):
    """ShakeMap PGA of event `ev` (from find_shakemap) on the study grid, or None."""
    import urllib.request
    det = get_json(ev["detail"], timeout=60)
    sm = (det.get("properties", {}).get("products", {}).get("shakemap") or [None])[0]
    if not sm:
        return None
    c = sm.get("contents", {}).get("download/grid.xml")
    if not c:
        return None
    url = c["url"]
    trace(url=url, label=f"usgs-shakemap-{ev['id']}")
    req = urllib.request.Request(url, headers={"User-Agent": "QGIS-Study-Kits (open-source QGIS plugin)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        lon0, lat1, dlon, dlat, a = parse_grid_xml(r.read())
    drv = gdal.GetDriverByName("MEM").Create("", a.shape[1], a.shape[0], 1, gdal.GDT_Float32)
    drv.SetGeoTransform((lon0 - dlon / 2, dlon, 0, lat1 + dlat / 2, 0, -dlat))
    from osgeo import osr
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    drv.SetProjection(sr.ExportToWkt())
    drv.GetRasterBand(1).WriteArray(np.nan_to_num(a, nan=-1))
    drv.GetRasterBand(1).SetNoDataValue(-1)
    out = gdal.Warp("", drv, format="MEM", outputBounds=grid.bounds, xRes=grid.res, yRes=grid.res, dstSRS=grid.wkt,
                    resampleAlg="bilinear", dstNodata=-1).ReadAsArray().astype(np.float32)
    out[out < 0] = np.nan
    out[~grid.mask] = np.nan
    if not np.isfinite(out[grid.mask]).any() or np.nanstd(out[grid.mask]) < 0.005:
        return None
    return out
