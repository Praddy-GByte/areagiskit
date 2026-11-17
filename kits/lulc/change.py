# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Change analysis.

* Transition (from-to) matrices.
* Gain, loss, net change, swap and total change per class (Pontius et al. 2004).
* Annual rate of change (Puyravaud 2003) and percent change.
* Intensity analysis at interval, category and transition level (Aldwaik & Pontius 2012).
* Change frequency, persistence, and change hot spots on a regular grid.
"""

import math

import numpy as np


def transition(a, b, classes, px):
    k = len(classes)
    idx = np.zeros(256, np.int64) - 1
    for i, c in enumerate(classes):
        idx[c] = i
    v = (a > 0) & (b > 0)
    ia, ib = idx[a[v]], idx[b[v]]
    ok = (ia >= 0) & (ib >= 0)
    m = np.bincount(ia[ok] * k + ib[ok], minlength=k * k).reshape(k, k).astype(float)
    return m * px  # km², rows = from (earlier), cols = to (later)


def components(T):
    """Gain, loss, net, swap, total (km²) per class from a transition matrix."""
    d = np.diag(T)
    gain = T.sum(0) - d
    loss = T.sum(1) - d
    net = gain - loss
    swap = 2 * np.minimum(gain, loss)
    return dict(gain=gain, loss=loss, net=net, swap=swap, total=gain + loss, start=T.sum(1), end=T.sum(0))


def annual_rate(a1, a2, years):
    """Puyravaud (2003): r = (1/(t2-t1)) ln(A2/A1) x 100, % per year."""
    if a1 <= 0 or a2 <= 0 or years <= 0:
        return None
    return 100.0 / years * math.log(a2 / a1)


def intensity(Ts, years):
    """Intensity analysis (Aldwaik & Pontius 2012).

    Ts: transition matrices (km²) for consecutive intervals; years: the time points.
    Returns interval-, category- and transition-level results."""
    total = Ts[0].sum()
    durations = [years[i + 1] - years[i] for i in range(len(Ts))]
    changes = [T.sum() - np.trace(T) for T in Ts]
    S = [c / total / d * 100 for c, d in zip(changes, durations)]
    U = sum(changes) / total / (years[-1] - years[0]) * 100
    cat = []
    for T, d, s in zip(Ts, durations, S):
        c = components(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = c["gain"] / d / c["end"] * 100
            L = c["loss"] / d / c["start"] * 100
        cat.append(dict(gain=np.nan_to_num(G), loss=np.nan_to_num(L), uniform=s))
    return dict(interval=S, uniform=U, durations=durations, category=cat)


def transition_intensity(T, d, n):
    """Transition level for the gain of class index n in one interval."""
    start = T.sum(1)
    gain_n = T[:, n].sum() - T[n, n]
    others = start.sum() - start[n]
    W = gain_n / d / others * 100 if others > 0 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(start > 0, T[:, n] / d / start * 100, 0)
    R[n] = np.nan
    return R, W


def frequency(maps):
    f = np.zeros(maps[0].shape, np.uint8)
    for a, b in zip(maps[:-1], maps[1:]):
        f += ((a != b) & (a > 0) & (b > 0)).astype(np.uint8)
    return f


def grid_change(changed, valid, cell_px):
    """Share of changed pixels in square blocks of cell_px × cell_px pixels."""
    r, c = changed.shape
    R, C = r // cell_px, c // cell_px
    ch = changed[:R * cell_px, :C * cell_px].reshape(R, cell_px, C, cell_px).sum((1, 3)).astype(float)
    va = valid[:R * cell_px, :C * cell_px].reshape(R, cell_px, C, cell_px).sum((1, 3)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(va > cell_px * cell_px * 0.5, ch / va, np.nan)
    return share


def gi_star_grid(x):
    """Getis-Ord Gi* on a grid with queen neighbours (including self). Returns z-scores."""
    v = np.isfinite(x)
    n = v.sum()
    if n < 10:
        return np.full(x.shape, np.nan)
    xb, s = np.nanmean(x), np.nanstd(x)
    xp = np.pad(np.where(v, x, 0), 1)
    vp = np.pad(v.astype(float), 1)
    sx = sum(xp[1 + dr:1 + dr + x.shape[0], 1 + dc:1 + dc + x.shape[1]] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    w = sum(vp[1 + dr:1 + dr + x.shape[0], 1 + dc:1 + dc + x.shape[1]] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (sx - xb * w) / (s * np.sqrt((n * w - w * w) / (n - 1)))
    z[~v] = np.nan
    return z
