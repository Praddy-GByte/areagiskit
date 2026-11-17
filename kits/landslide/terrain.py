# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Terrain hydrology without extra libraries: depression filling (priority flood), D8 flow directions,
flow accumulation, stream network and the indices used in landslide studies (TWI, SPI, STI, curvatures, TPI, TRI)."""

import heapq
import math

import numpy as np

# D8 neighbour offsets: E, SE, S, SW, W, NW, N, NE
DR = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1])
DIST = np.array([1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2), 1, math.sqrt(2)])


def fill(dem, mask, eps=1e-3):
    """Priority-flood depression filling with a tiny gradient (Barnes et al. 2014) so every cell drains."""
    H, W = dem.shape
    z = np.where(mask & np.isfinite(dem), dem, np.nan).astype(np.float64)
    done = ~np.isfinite(z)
    out = z.copy()
    pq = []
    # seeds: valid cells on the edge of the valid area
    nb_invalid = np.zeros((H, W), bool)
    for dr, dc in zip(DR, DC):
        sh = np.ones((H, W), bool)
        r0, r1 = max(dr, 0), H + min(dr, 0)
        c0, c1 = max(dc, 0), W + min(dc, 0)
        sh[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = done[r0:r1, c0:c1]
        nb_invalid |= sh
    seeds = np.argwhere(~done & nb_invalid)
    for r, c in seeds:
        heapq.heappush(pq, (out[r, c], int(r), int(c)))
        done[r, c] = True
    flat = out.ravel()
    dflat = done.ravel()
    push, pop = heapq.heappush, heapq.heappop
    while pq:
        zc, r, c = pop(pq)
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if 0 <= rr < H and 0 <= cc < W:
                i = rr * W + cc
                if not dflat[i]:
                    dflat[i] = True
                    if flat[i] <= zc:
                        flat[i] = zc + eps
                    push(pq, (flat[i], rr, cc))
    return out


def d8(z, res):
    """Receiver index (flat) of each cell, -1 for outlets, and the local slope to it (m/m)."""
    H, W = z.shape
    best = np.full((H, W), 0.0)
    rec = np.full((H, W), -1, np.int64)
    zp = np.pad(z, 1, constant_values=np.nan)
    idx = np.arange(H * W).reshape(H, W)
    for k in range(8):
        nz = zp[1 + DR[k]:1 + DR[k] + H, 1 + DC[k]:1 + DC[k] + W]
        drop = (z - nz) / (DIST[k] * res)
        better = np.isfinite(drop) & (drop > best)
        best = np.where(better, drop, best)
        rr = np.clip(np.arange(H)[:, None] + DR[k], 0, H - 1)
        cc = np.clip(np.arange(W)[None, :] + DC[k], 0, W - 1)
        rec = np.where(better, rr * W + cc, rec)
    rec[~np.isfinite(z)] = -1
    return rec.ravel(), best


def accumulate(z, rec):
    """Number of upslope cells draining through each cell (including itself)."""
    n = z.size
    acc = np.ones(n, np.float64)
    zf = z.ravel()
    order = np.argsort(np.where(np.isfinite(zf), -zf, np.inf), kind="stable")
    rec_o = rec[order]
    acc_l = acc  # numpy array indexed in python loop
    for i, r in zip(order.tolist(), rec_o.tolist()):
        if r >= 0:
            acc_l[r] += acc_l[i]
    acc[~np.isfinite(zf)] = np.nan
    return acc.reshape(z.shape)


def indices(dem, mask, res):
    """All terrain factors. Returns dict of arrays (float32) and the stream mask."""
    z = fill(dem, mask)
    rec, sl = d8(z, res)
    acc = accumulate(z, rec)
    a = acc * res  # specific catchment area per unit contour width (m)
    zz = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    gy, gx = np.gradient(zz, res)
    tanb = np.hypot(gx, gy)
    tanb_c = np.maximum(tanb, 0.001)
    twi = np.log(a / tanb_c)
    spi = np.log1p(a * tanb_c)
    sti = (a / 22.13) ** 0.6 * (np.sin(np.arctan(tanb_c)) / 0.0896) ** 1.3
    # curvatures (Zevenbergen & Thorne 1987)
    p = np.pad(zz, 1, mode="edge")
    z1, z2, z3 = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    z4, z5, z6 = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    z7, z8, z9 = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    L = res
    D = ((z4 + z6) / 2 - z5) / L ** 2
    E = ((z2 + z8) / 2 - z5) / L ** 2
    F = (-z1 + z3 + z7 - z9) / (4 * L ** 2)
    G = (-z4 + z6) / (2 * L)
    Hh = (z2 - z8) / (2 * L)
    g2 = G ** 2 + Hh ** 2 + 1e-12
    profile = -2 * (D * G ** 2 + E * Hh ** 2 + F * G * Hh) / g2 * 100
    plan = 2 * (D * Hh ** 2 + E * G ** 2 - F * G * Hh) / g2 * 100
    # roughness
    win = np.stack([z1, z2, z3, z4, z6, z7, z8, z9])
    tri = np.sqrt(((win - z5) ** 2).mean(0))
    streams = acc * res * res >= 1e6 * 0.5  # channels draining at least 0.5 km²
    out = dict(twi=twi, spi=spi, sti=sti, plan=plan, profile=profile, tri=tri, acc_km2=acc * res * res / 1e6)
    for k in out:
        out[k] = np.where(mask & np.isfinite(dem), out[k], np.nan).astype(np.float32)
    return out, streams & mask
