# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Physically based slope stability: the infinite-slope model (Skempton & DeLory 1957; as used in SINMAP and SHALSTAB).

FS = [c + (γ - m·γw)·z·cos²β·tanφ] / [γ·z·sinβ·cosβ]
c = soil cohesion + root cohesion (kPa), φ = friction angle, z = soil depth (m), m = share of the soil column that is
saturated (0 dry ... 1 fully saturated), γ = 18 kN/m³ soil, γw = 9.81 kN/m³ water. Soil strength is estimated from
SoilGrids clay and sand (typical values for the texture), root cohesion from land cover. These are textbook values, not
site measurements: the map shows where slopes are mechanically weakest, not a design calculation.
"""

import numpy as np

GAMMA, GAMMA_W = 18.0, 9.81
ROOT = {2: 5.0, 4: 2.0, 5: 1.0, 7: 0.0, 8: 0.0, 11: 2.0, 1: 0.0}
CLASSES = {1: ("Least stable: model FS < 1 even when dry", "#7f0000"), 2: ("Less stable: model FS < 1 when saturated", "#d7301f"),
           3: ("Less stable: model FS < 1 when half saturated", "#fc8d59"), 4: ("Marginal: model FS 1–1.5 when saturated", "#fdcc8a"),
           5: ("More stable: model FS ≥ 1.5 when saturated", "#91cf60")}
# FS uses textbook soil strengths from global soil maps, not site tests: classes rank slopes, they do not say a slope will fail.


def soil_strength(clay, sand):
    """Cohesion (kPa) and friction angle (degrees) from texture; defaults where SoilGrids is missing."""
    if clay is None or sand is None:      # SoilGrids not reachable: typical loam everywhere
        shape = np.shape(sand if clay is None else clay) if (clay is not None or sand is not None) else None
        if shape is None:
            return 3.0, 31.0
        clay = np.full(shape, 25.0) if clay is None else clay
        sand = np.full(shape, 40.0) if sand is None else sand
    cl = np.nan_to_num(clay, nan=25.0)
    sa = np.nan_to_num(sand, nan=40.0)
    c = np.select([cl >= 40, cl >= 25, sa >= 70], [12.0, 6.0, 1.0], 3.0)
    phi = np.select([cl >= 40, cl >= 25, sa >= 70], [22.0, 27.0, 34.0], 31.0)
    return c, phi


def soil_depth_map(plan_curv, base=1.5, k=0.5):
    """Soil depth that varies with plan curvature around a base depth (base × 0.5 … base × 1.5).
    Tested on three inventories (Hiroshima 2018, Sikkim, Gapyeong 2025): physical-model AUC 0.51→0.59, 0.71→0.71,
    0.59→0.61 against a constant depth; a slope-based depth rule helped Hiroshima but hurt Sikkim, so it is not used."""
    pc = np.nan_to_num(np.asarray(plan_curv, float))
    sc = np.nanpercentile(np.abs(pc), 90) or 1.0
    return np.clip(base * (1 + k * np.clip(pc / sc, -1, 1)), 0.3, 5.0).astype(np.float32)


def factor_of_safety(slope_deg, clay, sand, lc, depth=1.5, m=0.0):
    c_soil, phi = soil_strength(clay, sand)
    root = np.zeros(slope_deg.shape, np.float32)
    if lc is not None:
        for k, v in ROOT.items():
            root[lc == k] = v
    b = np.radians(np.clip(np.nan_to_num(slope_deg), 0.5, 70))
    num = (c_soil + root) + (GAMMA - m * GAMMA_W) * depth * np.cos(b) ** 2 * np.tan(np.radians(phi))
    den = GAMMA * depth * np.sin(b) * np.cos(b)
    fs = num / den
    fs[~np.isfinite(slope_deg)] = np.nan
    return fs.astype(np.float32)


def classes(fs_dry, fs_half, fs_sat):
    c = np.zeros(fs_dry.shape, np.uint8)
    ok = np.isfinite(fs_dry)
    c[ok] = 5
    c[ok & (fs_sat < 1.5)] = 4
    c[ok & (fs_half < 1)] = 3
    c[ok & (fs_sat < 1) & (fs_half >= 1)] = 2
    c[ok & (fs_dry < 1)] = 1
    return c
