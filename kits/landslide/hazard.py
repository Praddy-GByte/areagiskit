# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Temporal landslide hazard from the yearly satellite activity scan, and rule-based general guidance.

Hazard: for each susceptibility class, landslides (connected new-scar patches) per km² per year over the scanned years give a
Poisson rate λ; the chance of at least one landslide in a 1 km² area within T years is 1 − exp(−λT) (Guzzetti et al. 2005).
The record is short (a few years of Sentinel-2) and misses landslides smaller than a pixel, so these are minimum rates."""

import numpy as np

from ..flood.extra import label

PERIODS = (1, 10, 50)
HAZ = {1: ("Very low (< 5% in 50 years per km²)", "#1a9850"), 2: ("Low (5–20%)", "#91cf60"), 3: ("Moderate (20–50%)", "#fee08b"),
       4: ("High (50–80%)", "#fc8d59"), 5: ("Very high (> 80%)", "#d73027")}


def temporal(activity, cls5, mask, grid, min_px=2):
    """activity: {year: bool scar mask of NEW scars that year}. Returns dict(rows, years, map, total_rate) or None."""
    yrs = sorted(activity)
    if len(yrs) < 2:
        return None
    n_years = len(yrs)
    counts = {c: 0 for c in range(1, 6)}
    areas_hit = {c: 0.0 for c in range(1, 6)}
    for y in yrs:
        lab, n = label(activity[y] & mask, grid)
        if not n:
            continue
        sz = np.bincount(lab.ravel())
        for i in np.flatnonzero(sz >= min_px):
            if i == 0:
                continue
            m = lab == i
            cl = np.bincount(cls5[m], minlength=6)[1:].argmax() + 1   # class of most of the scar
            counts[int(cl)] += 1
            areas_hit[int(cl)] += float(m.sum() * grid.px_km2)
    rows = []
    hz = np.zeros(mask.shape, np.uint8)
    for c in range(1, 6):
        a = float(((cls5 == c) & mask).sum() * grid.px_km2)
        if a <= 0:
            continue
        lam = counts[c] / n_years / a
        p = {T: float(1 - np.exp(-lam * T)) for T in PERIODS}
        rows.append(dict(cls=c, area=a, n=counts[c], rate=lam, p=p, hit_km2=areas_hit[c]))
        p50 = p[50]
        hz[(cls5 == c) & mask] = 1 if p50 < 0.05 else 2 if p50 < 0.2 else 3 if p50 < 0.5 else 4 if p50 < 0.8 else 5
    tot = sum(counts.values()) / n_years
    return dict(rows=rows, years=yrs, n_years=n_years, map=hz, per_year=tot)


def guidance(R):
    """Short, general 'what to do' points derived from the results. Not a substitute for a site investigation."""
    out = []
    sus = R.get("sus") or {}
    g = R["g"]
    cls = sus.get("cls")
    run = R.get("runout") or {}
    if cls is not None:
        vh = float(((cls == 5) & g.mask).sum() * g.px_km2)
        if vh > 0:
            out.append(f"Avoid new houses, roads cuts and water tanks on the {vh:.1f} km² of very high susceptibility; where building cannot be "
                       "avoided, ask for a geotechnical site check first.")
    if run.get("reach") is not None and run["reach"].any():
        out.append("Keep the runout zones below steep slopes (map 'Where debris travels') free of new housing and schools; existing buildings "
                   "there are the first to warn and evacuate in very heavy rain.")
    fsc = R.get("fs_cls")
    if fsc is not None:
        u = float(np.isin(fsc, [2, 3])[g.mask].mean() * 100)
        if u >= 1:
            out.append(f"{u:.0f}% of the area becomes unstable only when the soil is wet: surface drainage (lined drains, no water on slopes), "
                       "keeping or planting deep-rooted trees and avoiding toe cuts help most there.")
    rain = R.get("rain") or {}
    thr = rain.get("threshold") or rain.get("sat_threshold")
    if thr:
        out.append("Use the rainfall threshold table as a starting point for a local early warning: compare forecast rain for the next 1–3 days "
                   "with the lowest totals that triggered landslides here.")
    else:
        out.append("Set up a simple rain-gauge early warning with local records: the satellite record is too short for a reliable threshold here.")
    hs = (R.get("exp") or {}).get("gi")
    if hs is not None and np.isin(hs, [4, 5]).any():
        out.append("Inspect the landslide hot spots (map 'Clusters of landslide scars') after every monsoon; repeated scars there point to "
                   "active slopes that may need monitoring.")
    out.append("Check every automatic scar in the field or on high-resolution imagery before using these maps for official zoning.")
    return out
