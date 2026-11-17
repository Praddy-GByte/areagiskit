# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Soil moisture from NASA POWER (MERRA-2 land model), account-free.

GWETROOT = root-zone soil wetness (0 = completely dry, 1 = saturated; top ~1 m), GWETTOP = surface soil wetness (top 5 cm).
Monthly values since 1981 at up to four points over the area are averaged; anomalies are standardised per calendar month
(z = (x - mean) / sd over all years with data), so a value of -1 means drier than about 84% of years for that month.
"""

import datetime as dt

import numpy as np

from ...core.data import get_json, trace

POWER_MONTHLY = "https://power.larc.nasa.gov/api/temporal/monthly/point"
CREDIT_SM = ("Soil moisture: NASA POWER monthly GWETROOT (root-zone) and GWETTOP (surface) soil wetness from the MERRA-2 land "
             "surface model (Gelaro et al. 2017), https://power.larc.nasa.gov (public, no account).")


def _points(bbox, center):
    x0, y0, x1, y1 = bbox
    if (x1 - x0) > 0.6 or (y1 - y0) > 0.6:
        return [((x0 * 3 + x1) / 4, (y0 * 3 + y1) / 4), ((x0 + x1 * 3) / 4, (y0 * 3 + y1) / 4),
                ((x0 * 3 + x1) / 4, (y0 + y1 * 3) / 4), ((x0 + x1 * 3) / 4, (y0 + y1 * 3) / 4)]
    return [center]


def monthly(bbox, center, end_year=None):
    """{(year, month): {"root": v, "top": v}} averaged over the points; values in 0..1."""
    end_year = end_year or dt.date.today().year
    acc = {}
    for lon, lat in _points(bbox, center):
        url = (f"{POWER_MONTHLY}?parameters=GWETROOT,GWETTOP&community=AG&longitude={lon:.4f}&latitude={lat:.4f}"
               f"&start=1981&end={end_year}&format=JSON")
        js = get_json(url, timeout=120)
        trace(url=url.split("&format")[0], label="nasa-power-soil-wetness")
        par = js["properties"]["parameter"]
        for key, name in (("GWETROOT", "root"), ("GWETTOP", "top")):
            for k, v in par.get(key, {}).items():
                y, m = int(k[:4]), int(k[4:6])
                if m < 1 or m > 12 or v is None or v < 0:          # month 13 is the annual mean; -999 = missing
                    continue
                acc.setdefault((y, m), {}).setdefault(name, []).append(float(v))
    return {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in acc.items()}


def anomalies(mon, var="root", min_years=15):
    """Standardised anomaly per calendar month: {(y, m): z}."""
    out = {}
    for m in range(1, 13):
        vals = {y: d[var] for (y, mm), d in mon.items() if mm == m and var in d}
        if len(vals) < min_years:
            continue
        a = np.array(list(vals.values()))
        mu, sd = float(a.mean()), float(a.std())
        if sd <= 1e-9:
            continue
        for y, v in vals.items():
            out[(y, m)] = (v - mu) / sd
    return out


def season_series(mon, months, var="root"):
    """Mean wetness over the crop-season months for each year with all months present, and its z-score."""
    by = {}
    for (y, m), d in mon.items():
        if m in months and var in d:
            by.setdefault(y, []).append(d[var])
    full = {y: float(np.mean(v)) for y, v in by.items() if len(v) == len(months)}
    if len(full) < 5:
        return {}
    a = np.array(list(full.values()))
    mu, sd = float(a.mean()), float(a.std()) or 1.0
    return {y: dict(wet=v, z=(v - mu) / sd) for y, v in sorted(full.items())}


def sm_class(z):
    return ("extremely dry" if z <= -2 else "severely dry" if z <= -1.5 else "moderately dry" if z <= -1
            else "near normal" if z < 1 else "wet")


def corr(a, b):
    """Pearson r over common keys (None when fewer than 8)."""
    ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if len(ks) < 8:
        return None, len(ks)
    x, y = np.array([a[k] for k in ks], float), np.array([b[k] for k in ks], float)
    if x.std() == 0 or y.std() == 0:
        return None, len(ks)
    return float(np.corrcoef(x, y)[0, 1]), len(ks)
