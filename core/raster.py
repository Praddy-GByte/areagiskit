# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Writing results that open styled in any GIS, plus PNG previews for reports."""

import json
import math
import os
import struct
import zlib

import numpy as np
from osgeo import gdal

# Colour ramps (low -> high). Chosen to stay readable for colour-blind readers.
RAMPS = {
    "ndvi": ["#a5682a", "#d9b56b", "#f2ecc4", "#9cc96b", "#3f8f3a", "#0f5323"],
    "heat": ["#2c4f8c", "#6c95c4", "#f2efe4", "#f2a65e", "#d1442d", "#7a1414"],
    "risk": ["#2f7a4d", "#9cc96b", "#f2d36b", "#ec8b3a", "#b8322a"],
    "depth": ["#e8f1f8", "#9cc2e0", "#4a8cc4", "#1d5a96", "#0b2f5c"],
    "elev": ["#2f6b4f", "#9cba7f", "#efe3b3", "#c49b6b", "#8a6248", "#f2f2f2"],
    "seq": ["#f4efe6", "#d6c3a3", "#a88a63", "#6e5638", "#3b2c1c"],
    "gray": ["#0d0d0d", "#4d4d4d", "#9a9a9a", "#e6e6e6"],
}


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp_colors(name, n):
    stops = [hex2rgb(c) for c in RAMPS[name]]
    out = []
    for i in range(n):
        t = i / max(n - 1, 1) * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        f = t - k
        out.append(tuple(int(round(stops[k][j] + (stops[k + 1][j] - stops[k][j]) * f)) for j in range(3)))
    return out


def _create(path, grid, dtype, nodata):
    ds = gdal.GetDriverByName("GTiff").Create(
        path, grid.cols, grid.rows, 1, dtype,
        options=["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=" + ("3" if dtype == gdal.GDT_Float32 else "2")])
    ds.SetGeoTransform(grid.gt)
    ds.SetProjection(grid.wkt)
    b = ds.GetRasterBand(1)
    if nodata is not None:
        b.SetNoDataValue(nodata)
    return ds, b


def write_classes(path, arr, grid, classes):
    """Byte raster with an embedded colour table and class names (QGIS, ArcGIS, ...).

    classes: {value: (name, '#rrggbb')}; 0 is no-data."""
    ds = gdal.GetDriverByName("GTiff").Create(path, grid.cols, grid.rows, 1, gdal.GDT_Byte,
                                              options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(grid.gt)
    ds.SetProjection(grid.wkt)
    b = ds.GetRasterBand(1)
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0, 0, 0, 0))
    names = [""] * 256
    for v, (n, col) in classes.items():
        ct.SetColorEntry(int(v), hex2rgb(col) + (255,))
        names[int(v)] = n
    b.SetRasterColorTable(ct)
    b.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    b.SetCategoryNames(names)
    b.SetNoDataValue(0)
    b.WriteArray(arr.astype(np.uint8))
    ds.FlushCache()
    ds = None
    return path


def write_float(path, arr, grid, ramp="seq", vmin=None, vmax=None, unit="", nodata=-9999.0):
    """Float raster plus a .qml style that QGIS loads automatically."""
    a = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    ds, b = _create(path, grid, gdal.GDT_Float32, nodata)
    b.WriteArray(a)
    if unit:
        b.SetUnitType(unit)
    ds.FlushCache()
    ds = None
    vals = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.percentile(vals, 2)) if vals.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(vals, 98)) if vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1
    cols = RAMPS[ramp]
    items = "".join(
        f'<item alpha="255" value="{vmin + (vmax - vmin) * i / (len(cols) - 1):.4f}" '
        f'label="{vmin + (vmax - vmin) * i / (len(cols) - 1):.2f}{(" " + unit) if unit else ""}" color="{c}"/>'
        for i, c in enumerate(cols))
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0" styleCategories="Symbology">
 <pipe>
  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" nodataColor="" classificationMin="{vmin}" classificationMax="{vmax}" alphaBand="-1">
   <rastershader>
    <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" minimumValue="{vmin}" maximumValue="{vmax}" labelPrecision="2">
     {items}
    </colorrampshader>
   </rastershader>
  </rasterrenderer>
 </pipe>
</qgis>"""
    with open(os.path.splitext(path)[0] + ".qml", "w", encoding="utf-8") as fh:
        fh.write(qml)
    return path, (vmin, vmax)


def write_geojson(path, features):
    """features: list of (lon, lat) points or GeoJSON geometry dicts, with properties."""
    fc = {"type": "FeatureCollection", "features": []}
    for geom, props in features:
        if isinstance(geom, tuple):
            geom = {"type": "Point", "coordinates": [round(geom[0], 6), round(geom[1], 6)]}
        fc["features"].append({"type": "Feature", "geometry": geom, "properties": props})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    return path


# ---------------------------------------------------------------- previews

def _png(index, palette, alpha):
    h, w = index.shape
    raw = b"".join(b"\x00" + index[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", b"".join(bytes(p) for p in palette))
            + chunk(b"tRNS", bytes(alpha))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def _step(arr, max_px):
    return max(1, int(math.ceil(max(arr.shape) / max_px)))


def preview_classes(arr, classes, max_px=900):
    s = _step(arr, max_px)
    a = arr[::s, ::s].astype(np.uint8)
    pal = [(255, 255, 255)] * 256
    alpha = [255] * 256
    alpha[0] = 0
    for v, (_, col) in classes.items():
        pal[int(v)] = hex2rgb(col)
    return _png(a, pal, alpha), a.shape[::-1]


def preview_float(arr, ramp, vmin, vmax, mask=None, max_px=900):
    s = _step(arr, max_px)
    a = arr[::s, ::s]
    valid = np.isfinite(a)
    if mask is not None:
        valid &= mask[::s, ::s]
    t = np.clip((np.nan_to_num(a, nan=vmin) - vmin) / ((vmax - vmin) or 1), 0, 1)
    idx = np.where(valid, 1 + np.round(t * 254), 0).astype(np.uint8)
    pal = [(255, 255, 255)] + ramp_colors(ramp, 255)
    alpha = [0] + [255] * 255
    return _png(idx, pal, alpha), idx.shape[::-1]


def ramp_css(ramp):
    return "linear-gradient(90deg," + ",".join(RAMPS[ramp]) + ")"


def neighbours(mask, radius=1):
    """Count of True cells in the (2r+1)^2 window around each cell (including itself)."""
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * radius + 1
    m = np.pad(mask.astype(np.uint8), radius)
    return sliding_window_view(m, (k, k)).sum(axis=(-1, -2))


def otsu(values, bins=256):
    v = values[np.isfinite(values)]
    if v.size < 100:
        return None
    hist, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * mids)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(between))])
