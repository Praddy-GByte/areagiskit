# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Additional LULC analyses: population and SDG 11.3.1, sprawl, ecological quality (RSEI),
urban heat (UTFVI), carbon storage, habitat quality, landscape ecological risk and gradient."""

import math

import numpy as np
from osgeo import gdal, osr

from ...core.data import trace, warp
from ...core.net import skipped

# ======================================================================= population / SDG 11.3.1

GHSL_POP = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E{e}_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E{e}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip/"
            "GHS_POP_E{e}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.tif")
CREDIT_POP = ("GHS-POP R2023A: Schiavina, M., Freire, S., Carioli, A., MacManus, K. (2023). GHS-POP R2023A - GHS population "
              "grid multitemporal (1975-2030). European Commission, JRC. CC BY 4.0.")


def _ghsl_tiles(aoi):
    moll = osr.SpatialReference()
    moll.SetFromUserInput("ESRI:54009")
    ct = osr.CoordinateTransformation(aoi.wgs, moll)
    x0, y0, x1, y1 = aoi.bbox
    t = set()
    for lon in (x0, x1, (x0 + x1) / 2):
        for lat in (y0, y1, (y0 + y1) / 2):
            mx, my, _ = ct.TransformPoint(lon, lat)
            t.add((int((9000000 - my) // 1e6) + 1, int((mx + 18041000) // 1e6) + 1))
    return sorted(t)


def population(grid, year):
    """People per analysis cell from GHS-POP (nearest 5-year epoch, 1975-2030). Returns (array, epoch)."""
    e = int(min(max(round(year / 5) * 5, 1975), 2030))
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".zip,.tif,.TIF,.tiff")
    srcs = []
    for r, c in _ghsl_tiles(grid.aoi):
        u = GHSL_POP.format(e=e, r=r, c=c)
        trace(url=u.split(".zip/")[0] + ".zip", label=f"ghsl-pop-{e}")
        srcs.append("/vsizip//vsicurl/" + u)
    try:
        a = warp(srcs, grid, resample="bilinear", dtype=gdal.GDT_Float32, src_nodata=-200, dst_nodata=np.nan)
    except Exception:
        return None, e
    a = np.where(a < 0, 0, a) * (grid.res * grid.res) / (100 * 100)  # people per 100 m cell → per analysis cell
    a[~grid.mask] = np.nan
    return a.astype(np.float32), e


def sdg_1131(built_km2, pop, years):
    """UN SDG indicator 11.3.1 for each interval: LCR, PGR and LCRPGR, plus built-up per person."""
    rows = []
    for (a, b) in zip(years[:-1], years[1:]):
        n = b - a
        u1, u2, p1, p2 = built_km2[a], built_km2[b], pop[a], pop[b]
        lcr = math.log(u2 / u1) / n if u1 > 0 and u2 > 0 else None
        pgr = math.log(p2 / p1) / n if p1 > 0 and p2 > 0 else None
        ratio = lcr / pgr if (lcr is not None and pgr not in (None, 0)) else None
        rows.append(dict(period=f"{a}–{b}", lcr=lcr, pgr=pgr, ratio=ratio,
                         m2_per_person_start=u1 * 1e6 / p1 if p1 > 0 else None,
                         m2_per_person_end=u2 * 1e6 / p2 if p2 > 0 else None))
    return rows


# ======================================================================= sprawl

def shannon(values):
    v = np.asarray(values, float)
    v = v[v > 0]
    if v.size < 2:
        return 0.0, 0.0
    p = v / v.sum()
    h = float(-(p * np.log(p)).sum())
    return h, h / math.log(len(values))


def compactness(patches_list, res):
    """Area-weighted compactness (2√(πA)/P) and area-weighted fractal dimension (2 ln(P/4)/ln A) of built-up patches."""
    ps = [(a, p) for a, p in patches_list if a >= 2]
    if not ps:
        return None, None
    A = np.array([a for a, _ in ps], float) * res * res
    P = np.array([p for _, p in ps], float) * res
    w = A / A.sum()
    comp = float((w * (2 * np.sqrt(np.pi * A) / P)).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        fd = 2 * np.log(P / 4) / np.log(A)
    fd = np.where(np.isfinite(fd), fd, 1.0)
    return comp, float((w * fd).sum())


# ======================================================================= RSEI

TC_WET_OLI = (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559)  # Baig et al. 2014, reflectance


def _norm(a, mask):
    v = a[mask & np.isfinite(a)]
    if v.size < 50:
        return np.full(a.shape, np.nan, np.float32)
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((a - lo) / ((hi - lo) or 1), 0, 1).astype(np.float32)


def rsei(bands, lst, ndvi, mask):
    """Remote Sensing Ecological Index (Xu 2013): PC1 of greenness, wetness, heat and dryness."""
    blue, green, red, nir, sw1, sw2 = bands
    wet = sum(c * b for c, b in zip(TC_WET_OLI, bands))
    with np.errstate(divide="ignore", invalid="ignore"):
        si = ((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue))
        a = 2 * sw1 / (sw1 + nir)
        b_ = nir / (nir + red) + green / (green + sw1)
        ibi = (a - b_) / (a + b_)
        mndwi = (green - sw1) / (green + sw1)
    ndbsi = (si + ibi) / 2
    land = mask & (np.nan_to_num(mndwi, nan=1) <= 0) & np.isfinite(lst)
    comps = [_norm(ndvi, land), _norm(wet, land), _norm(lst, land), _norm(ndbsi, land)]
    ok = land & np.all([np.isfinite(c) for c in comps], axis=0)
    if ok.sum() < 100:
        return None
    X = np.stack([c[ok] for c in comps], 1)
    Xc = X - X.mean(0)
    w, v = np.linalg.eigh(np.cov(Xc.T))
    pc1 = v[:, np.argmax(w)]
    if pc1[0] < 0:  # orient so greenness loads positively (higher = better ecology)
        pc1 = -pc1
    contrib = float(w.max() / w.sum() * 100)
    s = np.full(mask.shape, np.nan, np.float32)
    s[ok] = Xc @ pc1
    out = _norm(s, ok)
    out[~ok] = np.nan
    return dict(rsei=out, loadings=dict(zip(["Greenness (NDVI)", "Wetness (TC)", "Heat (LST)", "Dryness (NDBSI)"], pc1.tolist())),
                pc1_share=contrib, mean=float(np.nanmean(out)))


RSEI_CLASSES = {1: ("Poor (0–0.2)", "#8c2d04"), 2: ("Fair (0.2–0.4)", "#e6550d"), 3: ("Moderate (0.4–0.6)", "#fdd0a2"),
                4: ("Good (0.6–0.8)", "#74c476"), 5: ("Excellent (0.8–1)", "#00441b")}


def classify_01(a, mask):
    c = np.zeros(a.shape, np.uint8)
    v = mask & np.isfinite(a)
    c[v] = 1 + np.minimum((a[v] * 5).astype(int), 4)
    return c


# ======================================================================= urban heat

UTFVI_CLASSES = {1: ("None (< 0)", "#2c7bb6"), 2: ("Weak (0–0.005)", "#abd9e9"), 3: ("Middle (0.005–0.010)", "#ffffbf"),
                 4: ("Strong (0.010–0.015)", "#fdae61"), 5: ("Stronger (0.015–0.020)", "#f46d43"), 6: ("Strongest (> 0.020)", "#a50026")}
UTFVI_EEI = {1: "Excellent", 2: "Good", 3: "Normal", 4: "Bad", 5: "Worse", 6: "Worst"}


def utfvi(lst, mask):
    """Urban thermal field variance index (Zhang et al. 2006) on surface temperature in kelvin."""
    tk = lst + 273.15
    m = np.nanmean(tk[mask])
    u = (tk - m) / tk
    c = np.zeros(lst.shape, np.uint8)
    v = mask & np.isfinite(u)
    br = [0, 0.005, 0.010, 0.015, 0.020]
    c[v] = 1 + np.digitize(u[v], br)
    return u.astype(np.float32), c


# ======================================================================= carbon

# Indicative IPCC Tier-1 style carbon densities (Mg C/ha): above-ground, below-ground, soil, dead matter.
CARBON_TROPICAL = {1: (0, 0, 0, 0), 2: (2, 1, 20, 0), 3: (100, 25, 60, 5), 4: (5, 1, 50, 0), 5: (8, 8, 45, 1),
                   6: (1, 0, 20, 0), 7: (20, 10, 120, 2), 8: (0, 0, 0, 0)}
CARBON_TEMPERATE = {1: (0, 0, 0, 0), 2: (2, 1, 25, 0), 3: (80, 20, 80, 8), 4: (5, 1, 60, 0), 5: (6, 10, 60, 1),
                    6: (1, 0, 25, 0), 7: (20, 10, 150, 2), 8: (0, 0, 0, 0)}


def carbon(cmap, px_km2, lat):
    t = CARBON_TROPICAL if abs(lat) <= 23.5 else CARBON_TEMPERATE
    ha = px_km2 * 100
    by = {c: float((cmap == c).sum() * ha * sum(v)) for c, v in t.items()}
    dens = np.zeros(cmap.shape, np.float32)
    for c, v in t.items():
        dens[cmap == c] = sum(v)
    return sum(by.values()), by, dens, t


# ======================================================================= habitat quality

HABITAT = {1: 0.7, 2: 0.0, 3: 1.0, 4: 0.3, 5: 0.7, 6: 0.1, 7: 1.0, 8: 0.2}
THREATS = {2: ("Built-up", 1.0, 8.0, "exp"), 4: ("Cropland", 0.7, 4.0, "lin")}
SENS = {1: {2: 0.8, 4: 0.5}, 3: {2: 1.0, 4: 0.6}, 4: {2: 0.4, 4: 0.0}, 5: {2: 0.7, 4: 0.5}, 6: {2: 0.2, 4: 0.1},
        7: {2: 0.9, 4: 0.6}, 8: {2: 0.1, 4: 0.0}, 2: {2: 0.0, 4: 0.0}}


def _decay_conv(mask, res, dmax_km, kind):
    """Sum of decayed influence of threat cells within dmax (FFT convolution), normalised to 0-1."""
    r = int(math.ceil(dmax_km * 1000 / res))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(x, y) * res / 1000
    k = np.exp(-2.99 * d / dmax_km) if kind == "exp" else np.clip(1 - d / dmax_km, 0, 1)
    k[d > dmax_km] = 0
    k /= k.sum()
    H, W = mask.shape
    sh = (H + 2 * r, W + 2 * r)
    f = np.fft.rfft2(mask.astype(np.float32), sh) * np.fft.rfft2(k, sh)
    out = np.fft.irfft2(f, sh)[r:r + H, r:r + W]
    return np.clip(out, 0, 1).astype(np.float32)


def habitat_quality(cmap, mask, res, z=2.5):
    """Simplified InVEST habitat quality model (Sharp et al. 2020)."""
    impact = {t: _decay_conv(cmap == t, res, dmax, kind) for t, (_, w, dmax, kind) in THREATS.items()}
    wsum = sum(w for _, w, _, _ in THREATS.values())
    D = np.zeros(cmap.shape, np.float32)
    for c, s in SENS.items():
        sel = cmap == c
        if not sel.any():
            continue
        acc = np.zeros(sel.sum(), np.float32)
        for t, (_, w, _, _) in THREATS.items():
            acc += (w / wsum) * impact[t][sel] * s.get(t, 0)
        D[sel] = acc
    k = 0.5 * float(D[mask].max()) if D[mask].max() > 0 else 0.5
    H = np.zeros(cmap.shape, np.float32)
    for c, h in HABITAT.items():
        H[cmap == c] = h
    q = H * (1 - D ** z / (D ** z + k ** z))
    q[~mask | (cmap == 0)] = np.nan
    return q.astype(np.float32), D


# ======================================================================= landscape ecological risk

VULN = {2: 1, 3: 2, 5: 3, 4: 4, 7: 5, 1: 6, 6: 7, 8: 7}


def ecological_risk(cmap, patches_list, grid, cell_km=1.0):
    """Landscape ecological risk index (after Xie et al. 2013): disturbance × vulnerability, area-weighted per cell."""
    A = (cmap > 0).sum()
    classes = sorted(set(np.unique(cmap)) - {0})
    npatch = {c: sum(1 for cc, a, _ in patches_list if cc == c) for c in classes}
    N = sum(npatch.values()) or 1
    cpx = max(2, int(round(cell_km * 1000 / grid.res)))
    R_, C_ = grid.rows // cpx, grid.cols // cpx
    sub = cmap[:R_ * cpx, :C_ * cpx].reshape(R_, cpx, C_, cpx)
    vsum = sum(VULN.get(c, 4) for c in classes)
    Ri = {}
    for c in classes:
        Ai = (cmap == c).sum()
        if Ai == 0:
            continue
        Ci = npatch[c] / Ai
        Ni = 0.5 * math.sqrt(npatch[c] / A) / (Ai / A)
        Q = float(((sub == c).any(axis=(1, 3))).mean())
        Di = (Q + npatch[c] / N) / 4 + (Ai / A) / 2
        Ei = 0.5 * Ci + 0.3 * Ni + 0.2 * Di
        Ri[c] = Ei * VULN.get(c, 4) / vsum
    tot = (sub > 0).sum(axis=(1, 3)).astype(float)
    eri = np.zeros((R_, C_))
    for c, r in Ri.items():
        eri += (sub == c).sum(axis=(1, 3)) * r
    with np.errstate(divide="ignore", invalid="ignore"):
        eri = np.where(tot > cpx * cpx * 0.5, eri / tot, np.nan)
    return eri, cpx, Ri


ERI_CLASSES = {1: ("Low", "#1a9850"), 2: ("Lower", "#91cf60"), 3: ("Medium", "#fee08b"), 4: ("Higher", "#fc8d59"), 5: ("High", "#d73027")}


def expand_cells(arr, cpx, shape, dtype=np.float32, fill=np.nan):
    out = np.full(shape, fill, dtype)
    r, c = arr.shape
    out[:r * cpx, :c * cpx] = np.kron(arr, np.ones((cpx, cpx))).astype(dtype)
    return out


# ======================================================================= gradient

def ring_ids(grid, ctr, ring_km, max_rings=15):
    rr, cc = np.indices(grid.shape)
    d = np.hypot((rr - ctr[0]) * grid.res, (cc - ctr[1]) * grid.res) / 1000
    ring = (d // ring_km).astype(int) + 1
    ring[~grid.mask] = 0
    ring[ring > max_rings] = 0
    return ring


def gradient_metrics(cmap, ring, grid, classes, metrics_fn, min_px=500):
    rows = []
    for k in range(1, ring.max() + 1):
        sel = ring == k
        if sel.sum() < min_px:
            break
        m = np.where(sel, cmap, 0).astype(np.uint8)
        try:
            cls, land = metrics_fn(m, grid, classes)
        except Exception as exc:
            skipped(exc)
            continue
        rows.append(dict(ring=k, built=cls.get(2, {}).get("PLAND", 0.0), PD=land["PD"], ED=land["ED"],
                         SHDI=land["SHDI"], LPI=land["LPI"]))
    return rows
