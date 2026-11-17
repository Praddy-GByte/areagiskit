# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Flood water depth from a flood extent and a DEM (FwDET method, Cohen et al. 2018/2019).

The water surface at every flooded cell is taken from the elevation of the nearest flood
boundary cell (where water meets dry land), smoothed, and the ground elevation is subtracted.
"""

import numpy as np

from ...core.raster import neighbours


def _box_mean(a, r):
    """Mean of finite values in a (2r+1)^2 window (integral image)."""
    ok = np.isfinite(a)
    v = np.where(ok, a, 0.0).astype(np.float64)
    k = 2 * r + 1

    def box(x):
        p = np.pad(x, ((r + 1, r), (r + 1, r)))
        c = p.cumsum(0).cumsum(1)
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    s, n = box(v), box(ok.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, s / n, np.nan)


def nearest_value(values, seed, limit=None):
    """Value of the nearest seed cell for every cell (Euclidean allocation)."""
    try:
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(~seed, return_distances=False, return_indices=True)
        return values[idx[0], idx[1]]
    except Exception:
        out = np.where(seed, values, np.nan)
        filled = seed.copy()
        target = limit if limit is not None else np.ones_like(seed)
        shifts = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for _ in range(4000):
            todo = target & ~filled
            if not todo.any():
                break
            grew = False
            for dy, dx in shifts:
                src = np.roll(np.roll(filled, dy, 0), dx, 1)
                val = np.roll(np.roll(out, dy, 0), dx, 1)
                take = todo & src & ~filled
                if take.any():
                    out[take] = val[take]
                    filled |= take
                    grew = True
            if not grew:
                break
        return out


def depth(flood, valid, dem, permanent=None, smooth_px=3):
    """Returns (depth m, water surface elevation m, number of boundary cells used)."""
    dry = valid & ~flood & np.isfinite(dem)
    if permanent is not None:
        dry &= ~permanent
    # radar flood maps are speckled: single dry pixels inside a flood are not shorelines.
    # A shoreline is where water meets land that is mostly dry in its 5 x 5 neighbourhood.
    solid = dry & (neighbours(dry, 2) >= 18)
    edge = flood & (neighbours(solid, 1) > 0) & np.isfinite(dem)
    if edge.sum() < 10:
        return None, None, int(edge.sum())
    e = np.where(edge, dem, np.nan)
    wse = nearest_value(np.nan_to_num(e, nan=0.0), edge, limit=flood)
    wse = np.where(flood, wse, np.nan)
    wse = np.where(flood, _box_mean(wse, smooth_px), np.nan)
    d = np.where(flood & np.isfinite(dem), np.clip(wse - dem, 0, None), np.nan)
    return d.astype(np.float32), wse.astype(np.float32), int(edge.sum())
