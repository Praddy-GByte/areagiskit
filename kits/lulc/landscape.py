# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Landscape metrics (FRAGSTATS definitions, McGarigal & Marks 1995), urban expansion,
terrain zones and zonal statistics."""

import math

import numpy as np
from osgeo import gdal, ogr

# --------------------------------------------------------------------- patches


def patches(cmap, grid):
    """Patches (8-connected) via GDAL polygonize. Returns list of (class, area_cells, perimeter_cells)."""
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", grid.cols, grid.rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, grid.rows, 0, -1))  # cell units
    b = ds.GetRasterBand(1)
    b.WriteArray(cmap)
    b.SetNoDataValue(0)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("p")
    lyr = mem.CreateLayer("p")
    lyr.CreateField(ogr.FieldDefn("c", ogr.OFTInteger))
    gdal.Polygonize(b, b, lyr, 0, ["8CONNECTED=8"])
    out = []
    for f in lyr:
        g = f.GetGeometryRef()
        out.append((f.GetField(0), g.GetArea(), g.Boundary().Length()))
    return out


def _adjacency(cmap, classes):
    """Like/unlike adjacency counts (rook, each pair counted once)."""
    k = len(classes)
    lut = np.full(256, -1, np.int64)
    for i, c in enumerate(classes):
        lut[c] = i
    g = np.zeros((k, k), np.int64)
    for a, b in ((cmap[:, :-1], cmap[:, 1:]), (cmap[:-1, :], cmap[1:, :])):
        ia, ib = lut[a.ravel()], lut[b.ravel()]
        ok = (ia >= 0) & (ib >= 0)
        g += np.bincount(ia[ok] * k + ib[ok], minlength=k * k).reshape(k, k)
    return g + g.T - np.diag(np.diag(g))  # symmetric; like adjacencies on the diagonal once


def _max_like(A):
    n = int(math.floor(math.sqrt(A)))
    m = A - n * n
    if m == 0:
        return 2 * n * (n - 1)
    if m <= n:
        return 2 * n * (n - 1) + 2 * m - 1
    return 2 * n * (n - 1) + 2 * m - 2


def metrics(cmap, grid, classes):
    """Class-level and landscape-level metrics for one map."""
    cell_ha = grid.res * grid.res / 1e4
    valid = cmap > 0
    A_tot = valid.sum() * cell_ha
    P = patches(cmap, grid)
    g = _adjacency(cmap, classes)
    cls = {}
    for i, c in enumerate(classes):
        ps = [(a, p) for cc, a, p in P if cc == c]
        n_cells = int((cmap == c).sum())
        if n_cells == 0:
            continue
        areas = np.array([a for a, _ in ps]) * cell_ha
        perims = np.array([p for _, p in ps])
        edge_cells = (g[i].sum() - g[i, i])  # unlike adjacencies of class i
        cls[c] = dict(
            PLAND=n_cells * cell_ha / A_tot * 100,
            NP=len(ps),
            PD=len(ps) / A_tot * 100,                           # patches per 100 ha
            LPI=areas.max() / A_tot * 100 if len(areas) else 0,
            MPS=areas.mean() if len(areas) else 0,              # ha
            ED=edge_cells * grid.res / A_tot,                   # m per ha
            MSI=float(np.mean(0.25 * perims / np.sqrt(np.array([a for a, _ in ps])))) if ps else 0,
            AI=g[i, i] / _max_like(n_cells) * 100 if _max_like(n_cells) > 0 else 100,
        )
    p = np.array([cls[c]["PLAND"] / 100 for c in cls])
    shdi = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    m = len(cls)
    gs = g.astype(float)
    rows = gs.sum(1, keepdims=True)
    P_i = np.array([(cmap == c).sum() for c in classes], float) / max(valid.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = P_i[:, None] * np.where(rows > 0, gs / rows, 0)
        contag = (1 + np.nansum(np.where(q > 0, q * np.log(q), 0)) / (2 * math.log(m))) * 100 if m > 1 else 100
    total_edge = (g.sum() - np.trace(g)) / 2 * grid.res
    land = dict(NP=len(P), PD=len(P) / A_tot * 100,
                LPI=max((a for _, a, _ in P), default=0) * cell_ha / A_tot * 100,
                ED=total_edge / A_tot, SHDI=shdi, SHEI=shdi / math.log(m) if m > 1 else 0, CONTAG=contag,
                AI=float(sum(cls[c]["PLAND"] / 100 * cls[c]["AI"] for c in cls)))
    return cls, land


# --------------------------------------------------------------------- urban expansion

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def urban_metrics(b1, b2, y1, y2, valid_area_km2, px):
    u1, u2 = b1.sum() * px, b2.sum() * px
    n = y2 - y1
    agr = ((u2 / u1) ** (1 / n) - 1) * 100 if u1 > 0 else None
    ueii = (u2 - u1) / (valid_area_km2 * n) * 100
    return dict(u1=u1, u2=u2, growth=u2 - u1, agr=agr, ueii=ueii)


def centre(built, grid):
    """Centre of the earliest built-up area (mean position of its pixels)."""
    rr, cc = np.nonzero(built)
    if rr.size == 0:
        return grid.rows // 2, grid.cols // 2
    return int(rr.mean()), int(cc.mean())


def sectors(new, ctr, px):
    rr, cc = np.nonzero(new)
    ang = (np.degrees(np.arctan2(cc - ctr[1], -(rr - ctr[0]))) + 360 + 22.5) % 360
    s = (ang // 45).astype(int)
    return np.bincount(s, minlength=8) * px


def rings(builts, ctr, grid, ring_km=1.0, max_rings=15):
    rr, cc = np.indices(grid.shape)
    d = np.hypot((rr - ctr[0]) * grid.res, (cc - ctr[1]) * grid.res) / 1000
    ring = (d // ring_km).astype(int)
    out = []
    for b in builts:
        dens = []
        for k in range(max_rings):
            sel = (ring == k) & grid.mask
            if sel.sum() < 50:
                break
            dens.append(b[sel].mean() * 100)
        out.append(dens)
    return out


def lei(old, new, grid):
    """Landscape expansion index (Liu et al. 2010) for each new built-up patch.
    Returns per-pixel type map (1 infilling, 2 edge-expansion, 3 outlying) and areas."""
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", grid.cols, grid.rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((0, 1, 0, grid.rows, 0, -1))
    b = ds.GetRasterBand(1)
    b.WriteArray(new.astype(np.uint8))
    mem = ogr.GetDriverByName("Memory").CreateDataSource("n")
    lyr = mem.CreateLayer("n")
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(b, b, lyr, 0, ["8CONNECTED=8"])
    feats = [f for f in lyr if f.GetField(0) == 1]
    if not feats:
        return np.zeros(grid.shape, np.uint8), {1: 0, 2: 0, 3: 0}
    rd = drv.Create("", grid.cols, grid.rows, 1, gdal.GDT_Int32)
    rd.SetGeoTransform((0, 1, 0, grid.rows, 0, -1))
    pd = drv.Create("", grid.cols, grid.rows, 1, gdal.GDT_Int32)
    pd.SetGeoTransform((0, 1, 0, grid.rows, 0, -1))
    rl = mem.CreateLayer("r")
    rl.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    pl = mem.CreateLayer("p")
    pl.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    for i, f in enumerate(feats, start=1):
        g = f.GetGeometryRef()
        ring = g.Buffer(1.0, 1).Difference(g)
        for L, geom in ((rl, ring), (pl, g)):
            nf = ogr.Feature(L.GetLayerDefn())
            nf.SetGeometry(geom)
            nf.SetField("id", i)
            L.CreateFeature(nf)
    gdal.RasterizeLayer(rd, [1], rl, options=["ATTRIBUTE=id", "ALL_TOUCHED=TRUE"])
    gdal.RasterizeLayer(pd, [1], pl, options=["ATTRIBUTE=id"])
    rid = rd.GetRasterBand(1).ReadAsArray().ravel()
    pid = pd.GetRasterBand(1).ReadAsArray()
    n = len(feats) + 1
    ring_px = np.bincount(rid, minlength=n).astype(float)
    ring_old = np.bincount(rid, weights=old.ravel().astype(float), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(ring_px > 0, ring_old / ring_px * 100, 0)
    typ = np.where(L >= 50, 1, np.where(L > 0, 2, 3)).astype(np.uint8)
    typ[0] = 0
    tmap = typ[pid]
    tmap[~new] = 0
    px = grid.px_km2
    return tmap, {t: float((tmap == t).sum() * px) for t in (1, 2, 3)}


# --------------------------------------------------------------------- terrain and zones

SLOPE_BINS = [(0, 3, "0–3°"), (3, 8, "3–8°"), (8, 15, "8–15°"), (15, 25, "15–25°"), (25, 90, "> 25°")]


def by_zones(cmap, zones, labels, classes, px):
    """Area of each class inside each zone label (zones: int array, 0 = outside)."""
    out = []
    for z, name in labels:
        sel = zones == z
        out.append((name, {c: float(((cmap == c) & sel).sum() * px) for c in classes}))
    return out


def elevation_zones(elev, mask, n=5):
    v = elev[mask & np.isfinite(elev)]
    if v.size == 0:
        return np.zeros(elev.shape, np.int32), []
    edges = np.unique(np.round(np.percentile(v, np.linspace(0, 100, n + 1))))
    z = np.digitize(np.nan_to_num(elev, nan=-1e9), edges[1:-1]) + 1
    z[~mask] = 0
    labels = [(i + 1, f"{edges[i]:.0f}–{edges[i + 1]:.0f} m") for i in range(len(edges) - 1)]
    return z.astype(np.int32), labels


def slope_zones(slope, mask):
    z = np.zeros(slope.shape, np.int32)
    s = np.nan_to_num(slope, nan=-1)
    for i, (a, b, _) in enumerate(SLOPE_BINS, start=1):
        z[(s >= a) & (s < b) & mask] = i
    return z, [(i, lab) for i, (_, _, lab) in enumerate(SLOPE_BINS, start=1)]
