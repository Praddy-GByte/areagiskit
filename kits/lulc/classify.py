# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Automatic training and classification for any year.

1. Reference labels: where two independent 2021 maps (IO/Esri annual land cover and
   ESA WorldCover) agree, and the 30 m cell is at least 7/9 pure.
2. Label migration: for another year, keep only reference pixels whose spectral
   indices did not change between that year's and the 2021 composite (stable pixels).
3. Random Forest per year on the year's own Landsat features; 30% of samples held out.
4. 3×3 majority filter, and a temporal rule that built-up land does not flicker.
"""

import types

import numpy as np
from osgeo import gdal
from numpy.lib.stride_tricks import sliding_window_view

from ...core.data import KitError, item_date, search, warp, warp_items
from .rf import RandomForest

CLASSES = {
    1: ("Water bodies", "#3a8fd9"),
    2: ("Built-up", "#c4281b"),
    3: ("Forest / tree cover", "#2f6b3a"),
    4: ("Cropland", "#e3a33b"),
    5: ("Grass / scrub", "#c9c27a"),
    6: ("Barren / wasteland", "#a59b8f"),
    7: ("Wetland", "#7a87c6"),
    8: ("Snow / ice", "#a8ebff"),
}
IO_TO = {1: 1, 7: 2, 2: 3, 5: 4, 11: 5, 8: 6, 4: 7, 9: 8}
WC_TO = {80: 1, 50: 2, 10: 3, 95: 3, 40: 4, 20: 5, 30: 5, 100: 5, 60: 6, 90: 7, 70: 8}
CCI_TO = {}
for v in (10, 11, 12, 20, 30, 40):
    CCI_TO[v] = 4
for v in (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100):
    CCI_TO[v] = 3
for v in (110, 120, 121, 122, 130, 140):
    CCI_TO[v] = 5
for v in (150, 151, 152, 153, 200, 201, 202):
    CCI_TO[v] = 6
for v in (160, 170, 180):
    CCI_TO[v] = 7
CCI_TO[190] = 2
CCI_TO[210] = 1
CCI_TO[220] = 8

REF_YEAR = 2021


def _lut(m):
    lut = np.zeros(256, np.uint8)
    for k, v in m.items():
        lut[k] = v
    return lut


def fine_grid(grid, factor=3):
    g = types.SimpleNamespace()
    g.bounds, g.wkt, g.res = grid.bounds, grid.wkt, grid.res / factor
    g.cols, g.rows = grid.cols * factor, grid.rows * factor
    return g


def reference_labels(grid):
    """30 m labels where IO 2021 and WorldCover 2021 agree on ≥7 of the 9 10-m pixels."""
    fg = fine_grid(grid)
    io_items = [i for i in search("io-lulc-annual-v02", grid.aoi.bbox, f"{REF_YEAR}-01-01", f"{REF_YEAR}-12-31")
                if item_date(i)[:4] == str(REF_YEAR)]
    wc_items = search("esa-worldcover", grid.aoi.bbox, "2021-01-01", "2021-12-31")
    if not io_items or not wc_items:
        raise KitError("Reference land cover maps (2021) are not available for this area.")
    io = _lut(IO_TO)[warp_items(io_items, "data", fg, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)]
    wc = _lut(WC_TO)[warp_items(wc_items, "map", fg, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)]
    agree = np.where((io == wc) & (io > 0), io, 0).astype(np.uint8)
    blk = agree.reshape(grid.rows, 3, grid.cols, 3).transpose(0, 2, 1, 3).reshape(grid.rows, grid.cols, 9)
    counts = np.stack([(blk == k).sum(-1) for k in CLASSES], -1)
    lab = np.array(list(CLASSES), np.uint8)[counts.argmax(-1)]
    pure = counts.max(-1) >= 7
    lab = np.where(pure & grid.mask, lab, 0).astype(np.uint8)
    # agreement statistics between the two maps (useful for the report)
    both = (io > 0) & (wc > 0)
    agreement = float((io[both] == wc[both]).mean()) if both.any() else None
    io30 = np.array(list(CLASSES), np.uint8)[np.stack([(io.reshape(grid.rows, 3, grid.cols, 3) == k).sum((1, 3)) for k in CLASSES], -1).argmax(-1)]
    return lab, agreement, io30


GHSL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
        "GHS_BUILT_S_E{e}_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_BUILT_S_E{e}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip/"
        "GHS_BUILT_S_E{e}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.tif")
CREDIT_GHSL = ("GHS-BUILT-S R2023A: Pesaresi, M., Politis, P. (2023). GHS-BUILT-S R2023A - GHS built-up surface grid, "
               "derived from Sentinel-2 composite and Landsat, multitemporal (1975-2030). European Commission, JRC. CC BY 4.0.")


def ghsl_epoch(year):
    return int(min(max(round(year / 5) * 5, 1975), 2020))


def ghsl_fraction(grid, year):
    """Share of each 30 m cell's surrounding 100 m cell covered by buildings, from GHSL (observed epochs 1975-2020)."""
    from osgeo import osr
    from ...core.data import trace
    e = ghsl_epoch(year)
    moll = osr.SpatialReference()
    moll.SetFromUserInput("ESRI:54009")
    ct = osr.CoordinateTransformation(grid.aoi.wgs, moll)
    x0, y0, x1, y1 = grid.aoi.bbox
    tiles = set()
    for lon in (x0, x1, (x0 + x1) / 2):
        for lat in (y0, y1, (y0 + y1) / 2):
            mx, my, _ = ct.TransformPoint(lon, lat)
            tiles.add((int((9000000 - my) // 1e6) + 1, int((mx + 18041000) // 1e6) + 1))
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".zip,.tif,.TIF,.tiff")
    srcs = []
    for r, c in sorted(tiles):
        u = GHSL.format(e=e, r=r, c=c)
        trace(url=u.split(".zip/")[0] + ".zip", label=f"ghsl-built-s-{e}")
        srcs.append("/vsizip//vsicurl/" + u)
    try:
        a = warp(srcs, grid, resample="near", dtype=gdal.GDT_Float32, src_nodata=65535, dst_nodata=np.nan)
    except Exception:
        return None, e
    frac = a / 10000.0
    frac[~grid.mask] = np.nan
    return frac, e


def cci_map(grid, year):
    yy = min(max(year, 1992), 2020)
    items = [i for i in search("esa-cci-lc", grid.aoi.bbox, f"{yy}-01-01", f"{yy}-12-31") if item_date(i)[:4] == str(yy)]
    if not items:
        return None, None
    a = warp_items(items, "lccs_class", grid, resample="near", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
    return _lut(CCI_TO)[a], yy


def stable_mask(ix, ix_ref):
    """Pixels whose median and seasonal spectral behaviour match the reference year."""
    return ((np.abs(ix["NDVI"] - ix_ref["NDVI"]) < 0.08) & (np.abs(ix["MNDWI"] - ix_ref["MNDWI"]) < 0.10)
            & (np.abs(ix["NDBI"] - ix_ref["NDBI"]) < 0.08)
            & (np.abs(ix["NDVI_p90"] - ix_ref["NDVI_p90"]) < 0.10)
            & (np.abs(ix["NDVI_p10"] - ix_ref["NDVI_p10"]) < 0.10))


def sample(labels, where, per_class=2500, min_class=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(set(np.unique(labels).tolist()) - {0}):
        idx = np.flatnonzero(((labels == k) & where).ravel())
        if idx.size < min_class:
            continue
        out.append(rng.choice(idx, size=min(per_class, idx.size), replace=False))
    return np.concatenate(out) if out else np.array([], np.int64)


def relaxed_mask(ix, ix_ref, factor=2.0):
    return ((np.abs(ix["NDVI"] - ix_ref["NDVI"]) < 0.08 * factor) & (np.abs(ix["MNDWI"] - ix_ref["MNDWI"]) < 0.10 * factor)
            & (np.abs(ix["NDBI"] - ix_ref["NDBI"]) < 0.08 * factor)
            & (np.abs(ix["NDVI_p90"] - ix_ref["NDVI_p90"]) < 0.10 * factor))


def fill_missing(lab_y, where, lab, relaxed, ix, min_class=30, jrc_occ=None):
    """Classes that have reference labels but too few stable pixels this year get a second chance:
    water from a spectral rule (MNDWI > 0.1 and NDVI < 0.15), other classes with a relaxed stability test.
    Returns notes describing what was done."""
    notes = []
    for k in CLASSES:
        if (lab == k).sum() < min_class:
            continue
        if ((lab_y == k) & where).sum() >= min_class:
            continue
        if k == 1:
            if jrc_occ is not None:
                rule = (np.nan_to_num(jrc_occ) >= 50) & (ix["MNDWI"] > 0)
            else:
                rule = (ix["MNDWI"] > 0.3) & (ix["NDVI"] < 0.1) & (ix["MNDWI_p90"] > 0.3)
            if rule.sum() >= min_class:
                lab_y[rule] = 1
                where |= rule
                notes.append((k, "spectral water rule"))
                continue
        rel = (lab == k) & relaxed
        if rel.sum() >= min_class:
            lab_y[rel] = k
            where |= rel
            notes.append((k, "relaxed stability test"))
        else:
            notes.append((k, "not mappable this year (no training pixels)"))
    return notes


def year_labels(lab, stable, ghsl, year):
    """Training labels and where to sample them for one year.

    Non-built classes migrate where the pixel is spectrally stable. Built-up migrates
    only where GHSL shows at least 20% building cover in that year's epoch, because dry
    bare land and built-up look alike in Landsat and urban growth must not be assumed
    backwards."""
    lab_y = lab.copy()
    where = stable.copy()
    if year < REF_YEAR:
        built_ok = (np.nan_to_num(ghsl) >= 0.2) if ghsl is not None else np.zeros_like(stable)
        b = lab == 2
        lab_y[b & ~built_ok] = 0
        where = (where & ~b) | (b & built_ok)
    return lab_y, where


def classify_year(feat, labels, stable, grid, seed=0, n_trees=40, min_class=30, min_total=200, keep_samples=False):
    """Return dict(map, conf, proba, test=(y_true, y_pred), importance, oob, counts[, samples])."""
    F = feat.shape[0]
    X_all = feat.reshape(F, -1).T
    valid = np.isfinite(X_all).all(1) & grid.mask.ravel()
    where = stable & valid.reshape(grid.shape)
    idx = sample(labels, where, seed=seed, min_class=min_class)
    if idx.size < min_total:
        raise KitError("Too few training pixels for this year; the area may have changed almost completely, "
                       "or too few training points were given.")
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    y = labels.ravel()
    classes = sorted(set(y[idx].tolist()))
    rf = RandomForest(n_trees=n_trees, seed=seed).fit(X_all[tr], y[tr], classes)
    yp_te, _ = rf.predict(X_all[te])
    pred = np.zeros(X_all.shape[0], np.uint8)
    conf = np.full(X_all.shape[0], np.nan, np.float32)
    vi = np.flatnonzero(valid)
    pr = rf.predict_proba(X_all[vi])
    pred[vi] = np.asarray(classes)[pr.argmax(1)]
    conf[vi] = pr.max(1)
    proba = np.zeros((len(classes), X_all.shape[0]), np.float16)
    proba[:, vi] = pr.T
    counts = {k: int((y[idx] == k).sum()) for k in classes}
    out = dict(map=pred.reshape(grid.shape), conf=conf.reshape(grid.shape), proba=proba.reshape((len(classes),) + grid.shape),
               test=(y[te], yp_te), importance=rf.importance, oob=rf.oob_accuracy, counts=counts, classes=classes)
    if keep_samples:
        out["samples"] = (X_all[tr].copy(), y[tr].copy(), X_all[te].copy(), y[te].copy(), yp_te)
    return out


def majority(cmap, mask):
    k = np.array(sorted(set(np.unique(cmap).tolist()) - {0}), np.uint8)
    if k.size == 0:
        return cmap
    m = np.pad(cmap, 1)
    win = sliding_window_view(m, (3, 3))
    counts = np.stack([(win == c).sum((-1, -2)) for c in k], -1)
    out = k[counts.argmax(-1)]
    keep = counts.max(-1) < 4  # no clear majority: keep the original class
    out = np.where(keep, cmap, out)
    out[~mask | (cmap == 0)] = 0
    return out.astype(np.uint8)


def ghsl_constraint(res, ghsl, built_codes=(2,), ndvi_p90=None, p_keep=0.7, green=0.30):
    """Remove built-up in years up to 2020 where GHSL shows under 5% building cover AND the evidence for built-up is
    weak: the classifier's built-up probability is below p_keep, or the pixel had a green season (NDVI 90th
    percentile above `green`). Tuned on Las Vegas against USGS LCMAP; see validation report."""
    if ghsl is None:
        return 0
    cls = np.asarray(res["classes"])
    if not np.isin(cls, built_codes).any():
        return 0
    pb = sum(res["proba"][list(cls).index(b)].astype(np.float32) for b in built_codes if b in cls)
    weak = pb < p_keep
    if ndvi_p90 is not None:
        weak |= np.nan_to_num(ndvi_p90) > green
    bad = np.isin(res["map"], built_codes) & (np.nan_to_num(ghsl, nan=1) < 0.05) & weak
    if not bad.any():
        return 0
    pr = res["proba"][:, bad].astype(np.float32)
    for b in built_codes:
        if b in cls:
            pr[list(cls).index(b)] = -1
    res["map"][bad] = cls[pr.argmax(0)]
    return int(bad.sum())


def water_constraint(res, jrc_occ, mndwi, year):
    """Water outside anywhere that was ever water in 1984–2021 (JRC Global Surface Water) and without a water signal
    is reassigned to the next most likely class."""
    if jrc_occ is None or year > 2021:
        return 0
    cls = np.asarray(res["classes"])
    if 1 not in cls:
        return 0
    bad = (res["map"] == 1) & (np.nan_to_num(jrc_occ, nan=0) <= 0) & (np.nan_to_num(mndwi, nan=-1) < 0.1)
    if not bad.any():
        return 0
    pr = res["proba"][:, bad].astype(np.float32)
    pr[list(cls).index(1)] = -1
    res["map"][bad] = cls[pr.argmax(0)]
    return int(bad.sum())


def no_urban_reversal(results, years):
    """Urban land rarely returns to other uses. For a pixel built-up at one date but not the next:
    if it is built-up again at any later date, the gap is an omission and is filled with built-up;
    if it is never built-up again, the early built-up is removed (next most likely class)."""
    changed = 0
    for i in range(len(years) - 1):
        cur = results[years[i]]
        later = np.zeros(cur["map"].shape, bool)
        for j in range(i + 2, len(years)):
            later |= results[years[j]]["map"] == 2
        nxt = results[years[i + 1]]
        gap = (cur["map"] == 2) & (nxt["map"] != 2) & (nxt["map"] > 0)
        fill = gap & later
        nxt["map"][fill] = 2
        changed += int(fill.sum())
    for i in range(len(years) - 2, -1, -1):
        cur, nxt = results[years[i]], results[years[i + 1]]
        bad = (cur["map"] == 2) & (nxt["map"] != 2) & (nxt["map"] > 0)
        if not bad.any():
            continue
        cls = np.asarray(cur["classes"])
        pr = cur["proba"][:, bad].astype(np.float32)
        if 2 in cls:
            pr[list(cls).index(2)] = -1
        cur["map"][bad] = cls[pr.argmax(0)]
        changed += int(bad.sum())
    return changed


def temporal_rule(maps):
    """Built-up that is present before and after a date is set to built-up at that date too."""
    fixed = 0
    for i in range(1, len(maps) - 1):
        before = (maps[i - 1] == 2)
        after = np.zeros_like(before)
        for j in range(i + 1, len(maps)):
            after |= maps[j] == 2
        sel = before & after & (maps[i] != 2) & (maps[i] > 0)
        fixed += int(sel.sum())
        maps[i][sel] = 2
    return maps, fixed
