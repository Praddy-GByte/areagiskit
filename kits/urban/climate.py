# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Air temperature and heat stress (NASA POWER, MERRA-2), fine particulate matter PM2.5 (Copernicus CAMS via Open-Meteo)
and building height (JRC GHS-BUILT-H). All open, no account."""

import datetime as dt
import math

import numpy as np
from osgeo import gdal, osr

from ...core.data import get_json, trace, warp

CREDIT_POWER = ("NASA POWER (Prediction Of Worldwide Energy Resources), MERRA-2 based hourly and daily meteorology, "
                "https://power.larc.nasa.gov. Public, no restrictions.")
CREDIT_CAMS = ("Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition forecasts/reanalysis, via the "
               "Open-Meteo Air Quality API (CC BY 4.0; free for non-commercial use).")
CREDIT_GHSH = ("GHS-BUILT-H R2023A: Pesaresi, M., Politis, P. (2023). GHS-BUILT-H R2023A - GHS building height, derived from "
               "AW3D30, SRTM30, and Sentinel-2 composite (2018). European Commission, JRC. CC BY 4.0.")

GHSH = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/"
        "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip/"
        "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.tif")


# ------------------------------------------------------------------ heat index
def heat_index(t_c, rh):
    """NOAA heat index (Rothfusz regression with the NWS adjustments), °C in, °C out."""
    t = np.asarray(t_c, float) * 9 / 5 + 32
    rh = np.asarray(rh, float)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 6.83783e-3 * t * t
          - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
    with np.errstate(invalid="ignore"):
        adj1 = np.where((rh < 13) & (t >= 80) & (t <= 112), (13 - rh) / 4 * np.sqrt(np.maximum(17 - np.abs(t - 95), 0) / 17), 0)
        adj2 = np.where((rh > 85) & (t >= 80) & (t <= 87), (rh - 85) / 10 * (87 - t) / 5, 0)
    hi = hi - adj1 + adj2
    out = np.where((simple + t) / 2 < 80, simple, hi)
    return (out - 32) * 5 / 9


def vapour_pressure(t_c, rh):
    """Water vapour pressure (hPa) from air temperature (°C) and relative humidity (%), as used by the Bureau of Meteorology."""
    return np.asarray(rh, float) / 100.0 * 6.105 * np.exp(17.27 * np.asarray(t_c, float) / (237.7 + np.asarray(t_c, float)))


def wbgt_simple(t_c, rh):
    """Australian Bureau of Meteorology simplified wet-bulb globe temperature (°C) for average sun and light wind:
    WBGT = 0.567·Ta + 0.393·e + 3.94 (e in hPa). It ignores the actual sunshine and wind, so it is an approximation."""
    return 0.567 * np.asarray(t_c, float) + 0.393 * vapour_pressure(t_c, rh) + 3.94


def apparent_temp(t_c, rh, ws):
    """Steadman (1994) apparent temperature in the shade, Bureau of Meteorology form: AT = Ta + 0.33·e − 0.70·ws − 4.00."""
    return np.asarray(t_c, float) + 0.33 * vapour_pressure(t_c, rh) - 0.70 * np.asarray(ws, float) - 4.00


def _power(kind, lat, lon, start, end, params):
    u = (f"https://power.larc.nasa.gov/api/temporal/{kind}/point?parameters={params}&community=RE"
         f"&longitude={lon:.4f}&latitude={lat:.4f}&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON")
    trace(url=u.split("&format")[0], label=f"nasa-power-{kind}")
    js = get_json(u, timeout=120)
    return js["properties"]["parameter"]


def air_heat(lat, lon, year, months):
    """Hot-season air temperature and heat stress at the area centre (≈50 km reanalysis cell), plus the 1981–today trend."""
    start = dt.date(year, min(months), 1)
    last = max(months)
    end = (dt.date(year + (last == 12), last % 12 + 1, 1) - dt.timedelta(days=1))
    end = min(end, dt.date.today() - dt.timedelta(days=7))
    P = _power("hourly", lat, lon, start, end,
               "T2M,RH2M,WS10M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,ALLSKY_SFC_LW_DWN")
    keys = sorted(k for k in P["T2M"] if P["T2M"][k] > -900 and P["RH2M"].get(k, -999) > -900)
    T = np.array([P["T2M"][k] for k in keys])
    H = np.array([P["RH2M"][k] for k in keys])
    HI = heat_index(T, H)
    WS = np.array([max(P.get("WS10M", {}).get(k, 0.0), 0.0) for k in keys])
    WB = wbgt_simple(T, H)
    AT = apparent_temp(T, H, WS)
    days = sorted({k[:8] for k in keys})
    by = {d: [i for i, k in enumerate(keys) if k[:8] == d] for d in days}
    tmax = np.array([T[v].max() for v in by.values()])
    tmin = np.array([T[v].min() for v in by.values()])
    himax = np.array([HI[v].max() for v in by.values()])
    out = dict(start=start.isoformat(), end=end.isoformat(), n_days=len(days),
               tmax_mean=float(tmax.mean()), tmin_mean=float(tmin.mean()), tmax_abs=float(tmax.max()),
               hi_max=float(HI.max()), hi_mean_dmax=float(himax.mean()),
               days_40=int((tmax >= 40).sum()), days_45=int((tmax >= 45).sum()),
               warm_nights=int((tmin >= 25).sum()), tropical_nights=int((tmin >= 20).sum()),
               hours_caution=int(((HI >= 27) & (HI < 32)).sum()), hours_extreme_caution=int(((HI >= 32) & (HI < 41)).sum()),
               hours_danger=int(((HI >= 41) & (HI < 54)).sum()), hours_extreme_danger=int((HI >= 54).sum()),
               daily=[(d, float(a), float(b), float(c)) for d, a, b, c in zip(days, tmax, tmin, himax)],
               wbgt_max=float(WB.max()), wbgt_hours_28=int((WB >= 28).sum()), wbgt_hours_31=int((WB >= 31).sum()),
               wbgt_days_28=int(sum((WB[v] >= 28).any() for v in by.values())),
               at_max=float(AT.max()), at_hours_40=int((AT >= 40).sum()), n_hours=len(keys))
    # UTCI with a mean radiant temperature for a person standing in the sun (open ground)
    try:
        from . import utci as UT
        rad = {k: np.array([P.get(k, {}).get(x, np.nan) for x in keys], float)
               for k in ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_LW_DWN")}
        good = np.all([np.isfinite(v) & (v > -900) for v in rad.values()], axis=0)
        if good.mean() > 0.8:
            el = np.array([UT.solar_elevation(lat, dt.date(int(k[:4]), int(k[4:6]), int(k[6:8])).timetuple().tm_yday,
                                              int(k[8:10]) + 0.5) for k in keys])
            sw = {k: np.where(good, np.maximum(v, 0), 0.0) for k, v in rad.items()}
            lw = np.where(good, rad["ALLSKY_SFC_LW_DWN"], 5.670374e-8 * (T + 273.15) ** 4 * 0.8)
            TMRT = UT.mrt(T, sw["ALLSKY_SFC_SW_DNI"], sw["ALLSKY_SFC_SW_DIFF"], sw["ALLSKY_SFC_SW_DWN"], lw, el)
            U_sun = UT.utci(T, TMRT, WS, H)
            U_shade = UT.utci(T, UT.mrt(T, 0 * T, 0 * T, 0 * T, lw, -1 + 0 * el), WS, H)
            out["utci"] = dict(max_sun=float(U_sun.max()), max_shade=float(U_shade.max()),
                               hours_sun=UT.stress_hours(U_sun), hours_shade=UT.stress_hours(U_shade),
                               days_38_sun=int(sum((U_sun[v] >= 38).any() for v in by.values())),
                               days_38_shade=int(sum((U_shade[v] >= 38).any() for v in by.values())),
                               tmrt_max=float(TMRT.max()),
                               daily=[(d, float(U_sun[v].max()), float(U_shade[v].max())) for d, v in by.items()])
    except Exception as ex:
        out["utci_err"] = str(ex)
    # trend of hot-season mean Tmax and Tmin since 1981 (daily data)
    try:
        y1 = min(year, dt.date.today().year - 1)
        D = _power("daily", lat, lon, dt.date(1981, 1, 1), dt.date(y1, 12, 31), "T2M_MAX,T2M_MIN")
        ys, mx, mn = [], [], []
        for y in range(1981, y1 + 1):
            ks = [k for k in D["T2M_MAX"] if k[:4] == str(y) and int(k[4:6]) in months and D["T2M_MAX"][k] > -900]
            if len(ks) < 20:
                continue
            ys.append(y)
            mx.append(float(np.mean([D["T2M_MAX"][k] for k in ks])))
            mn.append(float(np.mean([D["T2M_MIN"][k] for k in ks if D["T2M_MIN"][k] > -900])))
        out["trend"] = dict(years=ys, tmax=mx, tmin=mn)
    except Exception as ex:
        out["trend_err"] = str(ex)
    return out


# ------------------------------------------------------------------ PM2.5
WHO_24H, WHO_YEAR = 15.0, 5.0


def _points(bbox):
    x0, y0, x1, y1 = bbox
    pts = [((y0 + y1) / 2, (x0 + x1) / 2)]
    if x1 - x0 > 0.4 or y1 - y0 > 0.4:      # bigger than one CAMS cell: sample the corners too
        pts += [(y0 + (y1 - y0) * a, x0 + (x1 - x0) * b) for a in (0.2, 0.8) for b in (0.2, 0.8)]
    return pts


def pm25(bbox, year):
    """Daily PM2.5 for the year (and the years before, from 2021) at up to five points; returns area statistics."""
    today = dt.date.today()
    seen, series = set(), []
    yrs = list(range(max(2021, year - 4), year + 1))
    for lat, lon in _points(bbox):
        s = dt.date(yrs[0], 1, 1)
        e = min(dt.date(year, 12, 31), today - dt.timedelta(days=1))
        u = (f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4f}&longitude={lon:.4f}"
             f"&hourly=pm2_5&start_date={s}&end_date={e}&timezone=auto")
        js = get_json(u, timeout=120)
        key = (round(js["latitude"], 2), round(js["longitude"], 2))
        if key in seen:
            continue
        seen.add(key)
        trace(url=u, label="cams-pm25")
        t, v = js["hourly"]["time"], js["hourly"]["pm2_5"]
        series.append((key, t, v))
    if not series:
        return None
    # daily means averaged over the distinct CAMS cells
    daily = {}
    for _, t, v in series:
        acc = {}
        for ti, vi in zip(t, v):
            if vi is not None:
                acc.setdefault(ti[:10], []).append(vi)
        for d, a in acc.items():
            if len(a) >= 18:
                daily.setdefault(d, []).append(float(np.mean(a)))
    days = sorted(daily)
    val = np.array([np.mean(daily[d]) for d in days])
    yr = np.array([int(d[:4]) for d in days])
    cur = yr == year
    if cur.sum() < 30:
        return None
    mo = {}
    for d, x in zip(days, val):
        if d.startswith(str(year)):
            mo.setdefault(int(d[5:7]), []).append(x)
    per_year = {int(y): float(val[yr == y].mean()) for y in np.unique(yr) if (yr == y).sum() >= 300}
    return dict(cells=len(series), year=year, n_days=int(cur.sum()), mean=float(val[cur].mean()),
                p98=float(np.percentile(val[cur], 98)), max=float(val[cur].max()),
                days_who=int((val[cur] > WHO_24H).sum()), days_35=int((val[cur] > 35).sum()), days_60=int((val[cur] > 60).sum()),
                times_who=float(val[cur].mean() / WHO_YEAR), month={m: float(np.mean(a)) for m, a in sorted(mo.items())},
                per_year=per_year, daily=list(zip(days, val.tolist())))


# ------------------------------------------------------------------ building height
def building_height(grid):
    """Average gross building height (m) per 100 m cell, 2018 (GHS-BUILT-H). Returns array on `grid` or None."""
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
        u = GHSH.format(r=r, c=c)
        trace(url=u.split(".zip/")[0] + ".zip", label="ghs-built-h-2018")
        srcs.append("/vsizip//vsicurl/" + u)
    try:
        a = warp(srcs, grid, resample="near", dtype=gdal.GDT_Float32, src_nodata=65535, dst_nodata=np.nan)
    except Exception:
        return None
    a[~grid.mask] = np.nan
    return a


def mann_kendall_slope(y):
    """Sen's slope per year and two-sided Mann-Kendall p (normal approximation, no ties correction)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 8:
        return None, None
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    sl = np.median([(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)])
    return float(sl), float(p)
