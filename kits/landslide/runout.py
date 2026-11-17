# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Where landslide debris travels: Flow-R style runout (Horton et al. 2013, NHESS 13, 869).

Spreading: Holmgren (1994) multiple flow direction (exponent x) combined with Gamma (2000) persistence, so debris prefers
to keep its direction. Reach: simplified friction law on an energy line. Kinetic energy per unit mass changes from cell to
cell as E_j = E_i + g·Δz − g·Δl·tan φ (φ = travel angle, angle of reach); the flow stops where E would fall below zero,
and E is capped at ½·v_max² so velocities stay realistic. Pure numpy + one Python pass over cells in descending elevation
(each cell is final before it passes debris on, so mass is conserved along the way)."""

import math

import numpy as np

G = 9.81
# Gamma (2000) persistence weights by turning angle 0, 45, 90, 135, 180 degrees
PERSIST = (1.0, 0.8, 0.4, 0.0, 0.0)
# 8 directions: (dr, dc); index order lets the turning angle be min(|a-b|, 8-|a-b|) * 45 degrees
DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

CLASSES = {1: ("Source area", "#67000d"), 2: ("Frequent reach (≥ 10%)", "#cb181d"),
           3: ("Possible reach (1–10%)", "#fb6a4a"), 4: ("Rare reach (< 1%)", "#fcbba1")}


def sources(inv, cls, slope, mask, min_slope_scar=15.0, min_slope_sus=30.0):
    """Release cells: detected scars on slopes ≥ 15° plus very-high susceptibility on slopes ≥ 30° (tested on Hiroshima 2018:
    wider release rules flag far more land without catching more debris)."""
    s = np.nan_to_num(slope)
    src = (inv & (s >= min_slope_scar)) | ((cls == 5) & (s >= min_slope_sus))
    return src & mask


def runout(dem, src, mask, res, angle_deg=11.0, vmax=15.0, exponent=4.0, pmin=1e-4, progress=None):
    """Returns dict(p, v, reach, cls, n_src): p = share of released debris passing each cell (summed over sources and
    clipped to 1, so it is a relative frequency), v = maximum velocity (m/s), reach = cells debris can reach."""
    rows, cols = dem.shape
    R, C = rows + 2, cols + 2
    z = np.full((R, C), np.nan, np.float64)
    z[1:-1, 1:-1] = np.where(mask, dem, np.nan)
    ok = np.isfinite(z)
    zf = np.where(ok, z, np.inf)                       # outside the area counts as a wall
    tanphi = math.tan(math.radians(angle_deg))
    emax = 0.5 * vmax * vmax
    off = [dr * C + dc for dr, dc in DIRS]
    dl = [res * (math.sqrt(2) if dr and dc else 1.0) for dr, dc in DIRS]
    flat_z = zf.ravel()
    # base Holmgren weights at each cell (towards lower neighbours only)
    W = np.zeros((8, R * C), np.float64)
    DZ = np.zeros((8, R * C), np.float64)
    for k, (dr, dc) in enumerate(DIRS):
        zn = _shift(zf, dr, dc)
        with np.errstate(invalid="ignore"):
            drop = zf - zn
        good = ok & np.isfinite(drop) & (drop > 0)
        W[k] = np.where(good, (np.where(good, drop, 0) / dl[k]) ** exponent, 0.0).ravel()
        DZ[k] = np.where(good, drop, 0.0).ravel()
    pers = np.array([[PERSIST[min(abs(a - b), 8 - abs(a - b))] for b in range(8)] for a in range(8)])
    P = np.zeros(R * C)
    E = np.full(R * C, -1.0)
    BEST = np.zeros(R * C)
    DIN = np.full(R * C, -1, np.int8)
    s = np.zeros((R, C), bool)
    s[1:-1, 1:-1] = src
    sidx = np.flatnonzero(s.ravel())
    P[sidx] = 1.0
    E[sidx] = 0.0
    order = np.flatnonzero(ok.ravel())
    order = order[np.argsort(-flat_z[order], kind="stable")]
    Wt = W.T.copy()
    DZt = DZ.T.copy()
    n = len(order)
    step = max(1, n // 20)
    for q, i in enumerate(order):
        if progress and q % step == 0:
            progress(q / n)
        p = P[i]
        if p < pmin:
            continue
        if p > 1.0:
            p = 1.0
            P[i] = 1.0
        w = Wt[i]
        d = DIN[i]
        if d >= 0:
            w = w * pers[d]
            if w.sum() <= 0:          # persistence blocks every downhill way: fall back to plain spreading
                w = Wt[i]
        tot = w.sum()
        if tot <= 0:
            continue
        e = E[i]
        dz = DZt[i]
        for k in range(8):
            wk = w[k]
            if wk <= 0:
                continue
            ej = e + G * dz[k] - G * dl[k] * tanphi
            if ej < 0:
                continue
            if ej > emax:
                ej = emax
            j = i + off[k]
            c = p * wk / tot
            P[j] += c
            if ej > E[j]:
                E[j] = ej
            if c > BEST[j]:
                BEST[j] = c
                DIN[j] = k
    P = np.minimum(P, 1.0).reshape(R, C)[1:-1, 1:-1]
    E = E.reshape(R, C)[1:-1, 1:-1]
    reach = (P >= pmin) & (E >= 0) & mask
    v = np.where(reach, np.sqrt(np.maximum(E, 0) * 2), np.nan)
    pr = np.where(reach, P, 0.0)
    cls = np.zeros(dem.shape, np.uint8)
    cls[reach & (pr < 0.01)] = 4
    cls[reach & (pr >= 0.01)] = 3
    cls[reach & (pr >= 0.10)] = 2
    cls[src & mask] = 1
    return dict(p=np.where(mask, pr, np.nan).astype(np.float32), v=v.astype(np.float32), reach=reach | (src & mask), cls=cls,
                n_src=int((src & mask).sum()), angle=angle_deg, vmax=vmax, exponent=exponent)


def _shift(a, dr, dc):
    """out[r, c] = a[r + dr, c + dc] (value of the neighbour in direction (dr, dc)); outside = +inf."""
    out = np.full(a.shape, np.inf)
    R, C = a.shape
    r0, r1 = max(0, -dr), R - max(0, dr)
    c0, c1 = max(0, -dc), C - max(0, dc)
    out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out

