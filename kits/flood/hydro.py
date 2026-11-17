# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Why it flooded: rainfall (NASA POWER) and river discharge (GloFAS reanalysis), return periods and trends.

Both services are free and need no account:
* NASA POWER daily precipitation (MERRA-2 based, corrected), 1981-today, ~0.5°.
* GloFAS v4 river discharge reanalysis via the Open-Meteo flood API, 1984-today, ~5 km river network.
"""

import datetime as dt
import math
import urllib.parse
import urllib.request
import json

import numpy as np

from ...core.data import UA, trace

POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
FLOODAPI = "https://flood-api.open-meteo.com/v1/flood"
CREDIT_POWER = ("Rainfall: NASA Prediction Of Worldwide Energy Resources (POWER) Project, daily PRECTOTCORR (MERRA-2), "
                "https://power.larc.nasa.gov (public, no account).")
CREDIT_GLOFAS_Q = ("River discharge: GloFAS v4 reanalysis (Copernicus Emergency Management Service, Harrigan et al. 2020), "
                   "served by Open-Meteo flood API (CC BY 4.0).")


def _get(url, timeout=90):
    import time
    for i, wait in enumerate((0, 10, 30)):
        time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:      # free APIs answer 429 when many requests come from one address
            if "429" not in str(e) or i == 2:
                raise


def rainfall(lon, lat, start, end):
    """Daily rainfall (mm) at a point: (dates list, values array)."""
    q = urllib.parse.urlencode(dict(parameters="PRECTOTCORR", community="AG", longitude=f"{lon:.4f}", latitude=f"{lat:.4f}",
                                    start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"), format="JSON"))
    url = POWER + "?" + q
    d = _get(url, timeout=120)
    trace(url=POWER + f"?parameters=PRECTOTCORR&longitude={lon:.3f}&latitude={lat:.3f}", label="nasa-power")
    p = d["properties"]["parameter"]["PRECTOTCORR"]
    days = sorted(p)
    v = np.array([p[k] for k in days], float)
    v[v < 0] = np.nan
    return [dt.date(int(k[:4]), int(k[4:6]), int(k[6:])) for k in days], v


def area_rainfall(aoi, start, end):
    """Mean daily rainfall over the area from up to 4 points (POWER cells are ~50 km)."""
    x0, y0, x1, y1 = aoi.bbox
    pts = [aoi.center]
    if (x1 - x0) > 0.6 or (y1 - y0) > 0.6:
        pts = [((x0 * 3 + x1) / 4, (y0 * 3 + y1) / 4), ((x0 + x1 * 3) / 4, (y0 * 3 + y1) / 4),
               ((x0 * 3 + x1) / 4, (y0 + y1 * 3) / 4), ((x0 + x1 * 3) / 4, (y0 + y1 * 3) / 4)]
    series = []
    days = None
    for lon, lat in pts:
        d, v = rainfall(lon, lat, start, end)
        days = d if days is None else days
        if len(v) == len(days):
            series.append(v)
    with np.errstate(all="ignore"):
        return days, np.nanmean(np.stack(series), axis=0)


def discharge(points, start, end):
    """GloFAS daily discharge (m³/s) for several (lon, lat). Returns dict point -> (dates, values)."""
    q = urllib.parse.urlencode(dict(latitude=",".join(f"{p[1]:.4f}" for p in points),
                                    longitude=",".join(f"{p[0]:.4f}" for p in points),
                                    start_date=str(start), end_date=str(end), daily="river_discharge"))
    d = _get(FLOODAPI + "?" + q, timeout=120)
    trace(url=FLOODAPI + "?daily=river_discharge", label="glofas-discharge")
    d = d if isinstance(d, list) else [d]
    out = {}
    for p, r in zip(points, d):
        t = [dt.date.fromisoformat(x) for x in r["daily"]["time"]]
        v = np.array([np.nan if x is None else x for x in r["daily"]["river_discharge"]], float)
        out[p] = (t, v, (r.get("longitude"), r.get("latitude")))
    return out


def main_river(aoi, start, end, n=5):
    """Pick the GloFAS cell inside the area with the largest mean discharge (the main river)."""
    x0, y0, x1, y1 = aoi.bbox
    pts = []
    for i in range(n):
        for j in range(n):
            lon, lat = x0 + (x1 - x0) * (i + 0.5) / n, y0 + (y1 - y0) * (j + 0.5) / n
            pts.append((lon, lat))
    from osgeo import ogr
    inside = []
    for lon, lat in pts:
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(lon, lat)
        if aoi.geom.Contains(g):
            inside.append((lon, lat))
    pts = inside or [aoi.center]
    res = discharge(pts, start, end)
    best = max(res.items(), key=lambda kv: np.nanmean(kv[1][1]) if np.isfinite(kv[1][1]).any() else -1)
    return best[0], best[1]


# ------------------------------------------------------------------ statistics
def annual_max(dates, values, window=1):
    """Annual maxima of the `window`-day running sum (rain) or mean (window=1 for discharge)."""
    v = np.nan_to_num(values, nan=0.0)
    if window > 1:
        c = np.convolve(v, np.ones(window), mode="full")[:len(v)]
    else:
        c = v
    years = {}
    for d, x in zip(dates, c):
        years.setdefault(d.year, []).append(x)
    full = {y: max(xs) for y, xs in years.items() if len(xs) >= 330}
    return full


def gumbel(maxima):
    """Gumbel (EV1) by method of moments. Returns (mu, beta)."""
    x = np.array(list(maxima), float)
    if x.size < 8:
        return None
    s = x.std(ddof=1)
    beta = s * math.sqrt(6) / math.pi
    mu = x.mean() - 0.5772 * beta
    return mu, beta


def return_period(value, fit):
    if fit is None or value is None:
        return None
    mu, beta = fit
    F = math.exp(-math.exp(-(value - mu) / max(beta, 1e-9)))
    return 1 / max(1 - F, 1e-6)


def quantile(T, fit):
    mu, beta = fit
    return mu - beta * math.log(-math.log(1 - 1 / T))


def gof_ks(maxima, fit):
    """Kolmogorov-Smirnov distance of the Gumbel fit (smaller is better) and the 5% critical value."""
    x = np.sort(np.array(list(maxima), float))
    n = x.size
    mu, beta = fit
    F = np.exp(-np.exp(-(x - mu) / beta))
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    D = float(max(np.abs(F - emp_hi).max(), np.abs(F - emp_lo).max()))
    return D, 1.36 / math.sqrt(n)


def mann_kendall(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return None
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return dict(z=float(z), p=float(p), slope=float(np.median(slopes)), n=int(n))


def empirical_rp(values, x):
    """Empirical return period (Weibull plotting position) of x among annual values."""
    v = np.sort(np.asarray(values, float))[::-1]
    n = v.size
    rank = int((v >= x).sum())
    return (n + 1) / max(rank, 1) if n else None
