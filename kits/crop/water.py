# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Crop water and drought: VHI (VCI + TCI from MODIS land surface temperature), SPI from NASA POWER rainfall, MODIS
evapotranspiration and a water-productivity proxy, and an indicative irrigated vs rainfed cropland map.

VHI. Kogan (1995, 1997): VCI = (NDVI − NDVImin)/(NDVImax − NDVImin)·100 and TCI = (BTmax − BT)/(BTmax − BTmin)·100 over the
same calendar window in all years; VHI = 0.5·VCI + 0.5·TCI. VHI < 40 = drought (< 30 moderate, < 20 severe, < 10 extreme).
Here NDVI is MODIS 250 m (MOD13Q1/MYD13Q1) and BT is MODIS daytime land surface temperature (MOD11A2, 1 km).

SPI. McKee et al. (1993): k-month rainfall totals are fitted by a gamma distribution for each calendar month (Thom 1958
maximum-likelihood approximation), months with no rain handled by the mixed distribution H = q + (1 − q)·G(x), and H is
transformed to a standard normal deviate. SPI ≤ −1 moderate, ≤ −1.5 severe, ≤ −2 extreme drought.

ET and water productivity. MODIS MOD16A3GF yearly evapotranspiration (500 m, Running et al. 2017). Water productivity
proxy = crop-season GPP (gC/m²) / annual ET (mm), a relative 'crop per drop' index, not grain per m³.

Irrigated vs rainfed (indicative). A crop that stays green through a real dry spell has water from somewhere other than rain.
For every 10-day period the previous 60 days of rainfall (NASA POWER, area mean) are summed; periods with < 60 mm in 60 days
and < 25 mm in 30 days are 'dry windows' (a growing crop needs roughly 3–5 mm per day). Cropland whose smoothed Sentinel-2 NDVI
is ≥ 0.60 in a dry window, or that is flooded paddy rice, is marked likely irrigated (in the spirit of the greenness rules of
MIrAD-US and LANID, Pervez & Brown 2010; Xie & Lark 2021). Where the crop year has no dry window (humid climates), greenness
cannot separate irrigated from rainfed crops and the map says 'not determinable'.
"""

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist

import numpy as np
from osgeo import gdal

from ...core.data import search, warp_items

CREDIT_LST = ("MODIS Terra Land Surface Temperature 8-day 1 km (MOD11A2 v6.1), Wan, Hook & Hulley, NASA LP DAAC, via Microsoft "
              "Planetary Computer.")
CREDIT_ET = ("MODIS Terra/Aqua Net Evapotranspiration yearly gap-filled 500 m (MOD16A3GF/MYD16A3GF v6.1), Running, Mu & Guo, "
             "NASA LP DAAC, via Microsoft Planetary Computer.")
CREDIT_LANID = ("Xie, Y., Lark, T. J. (2021). Mapping annual irrigation from Landsat imagery and environmental variables across the "
                "conterminous United States. Remote Sensing of Environment 260, 112445 (LANID-US, Zenodo 10.5281/zenodo.5548555).")

_ND = NormalDist()


# ------------------------------------------------------------------ incomplete gamma (numpy only, Numerical Recipes)
def _gser(a, x):
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(500):
        ap += 1
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_cdf(x, alpha, beta):
    """Regularised lower incomplete gamma P(alpha, x/beta)."""
    if x <= 0:
        return 0.0
    z = x / beta
    return _gser(alpha, z) if z < alpha + 1 else 1.0 - _gcf(alpha, z)


def gamma_fit(v):
    """Thom (1958) MLE approximation for positive values. Returns (alpha, beta) or None."""
    v = np.asarray(v, float)
    v = v[v > 0]
    if v.size < 8:
        return None
    m = v.mean()
    A = math.log(m) - np.mean(np.log(v))
    if A <= 0:
        return None
    a = (1 + math.sqrt(1 + 4 * A / 3)) / (4 * A)
    return a, m / a


# ------------------------------------------------------------------ SPI
def spi(monthly, scales=(1, 3, 6), min_years=20):
    """monthly: {(year, month): mm} with complete months. Returns {scale: {(year, month): spi}}."""
    keys = sorted(monthly)
    if not keys:
        return {}
    idx = {k: i for i, k in enumerate(keys)}
    out = {}
    for k in scales:
        acc = {}
        for (y, m) in keys:
            prev = []
            yy, mm = y, m
            for _ in range(k):
                if (yy, mm) not in monthly:
                    break
                prev.append(monthly[(yy, mm)])
                mm -= 1
                if mm == 0:
                    yy, mm = yy - 1, 12
            if len(prev) == k:
                acc[(y, m)] = float(sum(prev))
        res = {}
        for cm in range(1, 13):
            vals = [(kk, v) for kk, v in acc.items() if kk[1] == cm]
            if len(vals) < min_years:
                continue
            arr = np.array([v for _, v in vals])
            q = float((arr <= 0).mean())
            fit = gamma_fit(arr)
            if fit is None:
                continue
            a, b = fit
            for kk, v in vals:
                H = q + (1 - q) * gamma_cdf(v, a, b)
                H = min(max(H, 1e-6), 1 - 1e-6)
                res[kk] = _ND.inv_cdf(H)
        out[k] = res
    return out


def spi_class(s):
    if s is None:
        return "–"
    return ("extreme drought" if s <= -2 else "severe drought" if s <= -1.5 else "moderate drought" if s <= -1 else
            "near normal" if s < 1 else "moderately wet" if s < 1.5 else "very wet" if s < 2 else "extremely wet")


# ------------------------------------------------------------------ VHI
def _date(it):
    return dt.date.fromisoformat((it["properties"].get("start_datetime") or it["properties"]["datetime"])[:10])


def season_year(d, window):
    """Year in which the season containing date d starts (window = (start_date, end_date) of one crop year), or None."""
    a, b = window
    span = (b - a).days % 365 or 365
    for y in (d.year, d.year - 1):
        try:
            s = dt.date(y, a.month, min(a.day, 28))
        except ValueError:
            continue
        if s <= d <= s + dt.timedelta(days=span):
            return y
    return None


def season_ndvi(arrs, window):
    """Mean NDVI per pixel for each season year from the 16-day MODIS arrays {date: float16 array}."""
    acc, cnt = {}, {}
    for d, a in arrs.items():
        y = season_year(d, window)
        if y is None:
            continue
        a = a.astype(np.float32)
        if y not in acc:
            acc[y], cnt[y] = np.zeros(a.shape, np.float32), np.zeros(a.shape, np.int16)
        ok = np.isfinite(a)
        acc[y][ok] += a[ok]
        cnt[y][ok] += 1
    if not cnt:
        return {}
    full = max(int(np.median(c[c > 0])) if (c > 0).any() else 0 for c in cnt.values())
    return {y: np.where(cnt[y] >= max(1, 0.6 * full), acc[y] / np.maximum(cnt[y], 1), np.nan).astype(np.float32)
            for y in acc if (cnt[y] > 0).any() and np.median(cnt[y][cnt[y] > 0]) >= 0.6 * full}


def season_lst(aoi, grid, window, years, per_year=4, threads=8, progress=None):
    """Mean daytime LST (°C) per pixel for each season year, from `per_year` MOD11A2 composites spread through the season."""
    a, b = window
    span = (b - a).days % 365 or 365
    jobs = []
    for y in years:
        s = dt.date(y, a.month, min(a.day, 28))
        e = s + dt.timedelta(days=span)
        its = [it for it in search("modis-11A2-061", aoi.bbox, str(s), str(e), max_items=200) if it["id"].startswith("MOD")]
        by = {}
        for it in its:
            by.setdefault(_date(it), []).append(it)
        ds = sorted(by)
        if not ds:
            continue
        pick = [ds[int(round(i))] for i in np.linspace(0, len(ds) - 1, min(per_year, len(ds)))]
        jobs += [(y, by[d]) for d in sorted(set(pick))]
    done = [0]

    def job(j):
        y, its = j
        try:
            v = warp_items(its, "LST_Day_1km", grid, resample="near", dtype=gdal.GDT_UInt16, src_nodata=0, dst_nodata=0)
            v = v.astype(np.float32)
            v[v == 0] = np.nan
            return y, v * 0.02 - 273.15
        except Exception:
            return y, None
    acc, cnt = {}, {}
    with ThreadPoolExecutor(threads) as ex:
        for y, v in ex.map(job, jobs):
            done[0] += 1
            if progress and done[0] % 10 == 0:
                progress(done[0], len(jobs))
            if v is None:
                continue
            if y not in acc:
                acc[y], cnt[y] = np.zeros(v.shape, np.float32), np.zeros(v.shape, np.int16)
            ok = np.isfinite(v)
            acc[y][ok] += v[ok]
            cnt[y][ok] += 1
    return {y: np.where(cnt[y] > 0, acc[y] / np.maximum(cnt[y], 1), np.nan).astype(np.float32) for y in acc}


def vhi(ndvi_y, lst_y, crop_mask, min_years=8):
    """VCI, TCI and VHI per year (cropland means) and the per-pixel VHI maps. Needs >= min_years common years."""
    ys = sorted(set(ndvi_y) & set(lst_y))
    if len(ys) < min_years:
        return None
    N = np.stack([ndvi_y[y] for y in ys])
    T = np.stack([lst_y[y] for y in ys])
    with np.errstate(all="ignore"):
        nmin, nmax = np.nanmin(N, 0), np.nanmax(N, 0)
        tmin, tmax = np.nanmin(T, 0), np.nanmax(T, 0)
        VCI = np.clip((N - nmin) / np.maximum(nmax - nmin, 1e-6) * 100, 0, 100)
        TCI = np.clip((tmax - T) / np.maximum(tmax - tmin, 1e-6) * 100, 0, 100)
    VHI = 0.5 * VCI + 0.5 * TCI
    rows = {}
    for i, y in enumerate(ys):
        m = crop_mask & np.isfinite(VHI[i])
        if m.sum() < 3:
            continue
        rows[y] = dict(vci=float(np.nanmean(VCI[i][m])), tci=float(np.nanmean(TCI[i][m])), vhi=float(np.nanmean(VHI[i][m])),
                       drought_pct=float((VHI[i][m] < 40).mean() * 100), lst=float(np.nanmean(T[i][m])), ndvi=float(np.nanmean(N[i][m])))
    return dict(years=ys, rows=rows, maps={y: VHI[i] for i, y in enumerate(ys)})


def vhi_class(v):
    return "extreme drought" if v < 10 else "severe drought" if v < 20 else "moderate drought" if v < 30 else "mild drought" if v < 40 else "no drought"


# ------------------------------------------------------------------ ET
def et_record(aoi, grid, crop_mask, start_year=2001, end=None, threads=8):
    """Yearly ET (mm) over cropland and per-pixel maps from MOD16A3GF (Terra; Aqua where Terra is missing)."""
    end = end or dt.date.today()
    items = search("modis-16A3GF-061", aoi.bbox, f"{start_year}-01-01", f"{end.year}-12-31", max_items=500)
    by = {}
    for it in items:
        y = _date(it).year
        sat = it["id"][:3]
        by.setdefault((y, sat), []).append(it)
    keys = {}
    for (y, sat), its in by.items():
        if sat == "MOD" or (y, "MOD") not in by:
            keys[y] = its

    def job(y):
        try:
            v = warp_items(keys[y], "ET_500m", grid, resample="near", dtype=gdal.GDT_UInt16, src_nodata=65535, dst_nodata=65535)
            v = v.astype(np.float32)
            v[v >= 32700] = np.nan          # fill values (water, urban, barren)
            return y, v * 0.1               # kg/m²/yr = mm/yr
        except Exception:
            return y, None
    out, maps = {}, {}
    with ThreadPoolExecutor(threads) as ex:
        for y, v in ex.map(job, sorted(keys)):
            if v is None:
                continue
            m = crop_mask & np.isfinite(v)
            if m.sum() < 3:
                continue
            out[y] = float(np.nanmean(v[m]))
            maps[y] = v
    return out, maps


# ------------------------------------------------------------------ irrigated vs rainfed
def dry_windows(centers, days, rain, win60=60, win30=30, max60=60.0, max30=25.0):
    """Boolean per 10-day center: previous 60 days < max60 mm and previous 30 days < max30 mm (area-mean rainfall)."""
    idx = {d: i for i, d in enumerate(days)}
    r = np.nan_to_num(np.asarray(rain, float))
    cs = np.concatenate([[0.0], np.cumsum(r)])
    out, tot60 = [], []
    for c in centers:
        if isinstance(c, (tuple, list)):          # (start, end) of a 10-day window: judge at its end
            c = c[1]
        if isinstance(c, dt.datetime):
            c = c.date()
        i = idx.get(c)
        if i is None or i < win60:
            out.append(False)
            tot60.append(None)
            continue
        s60 = cs[i + 1] - cs[i + 1 - win60]
        s30 = cs[i + 1] - cs[i + 1 - win30]
        out.append(bool(s60 < max60 and s30 < max30))
        tot60.append(float(s60))
    return np.array(out), tot60


def irrigated(S, crop, centers, days, rain, rice=None, green=0.60):
    """Classes: 0 outside cropland, 1 likely irrigated, 2 likely rainfed, 3 not determinable (no dry window)."""
    dry, tot60 = dry_windows(centers, days, rain)
    cls = np.zeros(crop.shape, np.uint8)
    cls[crop] = 3
    irr = np.zeros(crop.shape, bool)
    if dry.any():
        g = np.nanmax(np.where(dry[:, None, None], S, -1), 0) >= green
        irr |= g
        cls[crop] = 2
    if rice is not None:
        irr |= rice
    cls[crop & irr] = 1
    return cls, dict(dry_periods=[str(c) for c, d in zip(centers, dry) if d], n_dry=int(dry.sum()), rain60=tot60)


IRR_CLASSES = {1: ("Likely irrigated", "#2c7fb8"), 2: ("Likely rainfed", "#d9b365"), 3: ("Not determinable (no dry spell)", "#cccccc")}
