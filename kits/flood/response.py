# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Flood response helpers: nearest safe high ground for evacuation planning, a proxy-based vulnerability index, rule-based
general guidance, and plain-language wording of technical errors. Numpy only.

Safe ground (on ~100 m blocks): never seen flooded in the radar history, not flooded in this event, not permanent water,
and at least `margin` m above the highest flooded ground within ~1 km (so it would stay dry if water rose a little).
Distances are straight lines: there is no road network in open data that works offline everywhere, so travel routes are
not modelled."""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _block(a, k, how="mean"):
    r, c = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:r, :c].reshape(r // k, k, c // k, k)
    with np.errstate(all="ignore"):
        if how == "max":
            return np.nanmax(b, axis=(1, 3))
        if how == "any":
            return b.any(axis=(1, 3))
        return np.nanmean(b, axis=(1, 3))


def _box_max(a, r):
    p = np.pad(a, r, mode="constant", constant_values=-np.inf)
    return sliding_window_view(p, (2 * r + 1, 2 * r + 1)).max(axis=(-1, -2))


def hg_to_eg(arr_h, HG, EG):
    """Nearest-neighbour transfer from the history grid to the event grid (same origin)."""
    r = np.minimum(((np.arange(EG.rows) + 0.5) * EG.res / HG.res).astype(int), HG.rows - 1)
    c = np.minimum(((np.arange(EG.cols) + 0.5) * EG.res / HG.res).astype(int), HG.cols - 1)
    return arr_h[np.ix_(r, c)]


def safe_ground(EG, flood, perm, ever_flooded, dem, block_m=100.0, margin=1.0, window_m=1000.0):
    """Return (safe_blocks bool, k, level) on a coarse grid of ~block_m cells."""
    k = max(1, int(round(block_m / EG.res)))
    valid = _block(EG.mask.astype(np.float32), k) > 0.5
    fl = _block(flood.astype(np.float32), k) > 0
    pm = _block(perm.astype(np.float32), k) > 0
    ev = _block(ever_flooded.astype(np.float32), k) > 0
    z = _block(np.where(np.isfinite(dem), dem, np.nan).astype(np.float32), k)
    zf = np.where(fl & np.isfinite(z), z, -np.inf)
    w = max(1, int(round(window_m / (k * EG.res))))
    level = _box_max(zf, w)                                   # highest flooded ground nearby
    has_level = np.isfinite(level)
    high_enough = np.where(has_level, z >= level + margin, True)
    safe = valid & ~fl & ~pm & ~ev & np.isfinite(z) & high_enough
    return safe, k, level


def nearest(points_rc, safe, k, res):
    """For each (row, col) on the fine grid: distance (m), bearing and target (row, col) of the nearest safe block."""
    rr, cc = np.nonzero(safe)
    if rr.size == 0:
        return [None] * len(points_rc)
    ty, tx = (rr + 0.5) * k, (cc + 0.5) * k
    out = []
    for r, c in points_rc:
        d2 = (ty - r) ** 2 + (tx - c) ** 2
        i = int(np.argmin(d2))
        dist = math.sqrt(float(d2[i])) * res
        ang = (math.degrees(math.atan2(tx[i] - c, -(ty[i] - r))) + 360) % 360
        out.append(dict(dist_m=dist, bearing=DIRS[int((ang + 22.5) // 45) % 8], target_rc=(float(ty[i]), float(tx[i]))))
    return out


def evacuation(EG, settlements, lab, flood, perm, ever_flooded, dem, isolated=None, pop=None, roads=None):
    """Rows for settlements with people in flood water (and cut-off places): nearest safe ground (straight line), and the
    distance along roads that are not under water when an OpenStreetMap road mask on the event grid is given."""
    safe, k, level = safe_ground(EG, flood, perm, ever_flooded, dem)
    road_km = None
    if roads is not None and roads.any() and safe.any():
        try:
            from ..urban.network import network_distance_km
            sr, sc = np.nonzero(safe)
            src = list(zip(((sr + 0.5) * k).astype(int).tolist(), ((sc + 0.5) * k).astype(int).tolist()))
            road_km = network_distance_km(EG, roads & ~flood, src, snap_m=1000.0)
        except Exception:
            road_km = None
    rows = []
    pts = []
    for s in settlements:
        r, c = EG.lonlat_to_pixel(s["lon"], s["lat"])
        pts.append((r, c))
    res = nearest(pts, safe, k, EG.res)
    for s, n in zip(settlements, res):
        row = dict(s)
        if n:
            tlon, tlat = EG.pixel_to_lonlat(int(n["target_rc"][0]), int(n["target_rc"][1]))
            row.update(safe_km=n["dist_m"] / 1000, bearing=n["bearing"], safe_lon=tlon, safe_lat=tlat)
            if road_km is not None:
                r_, c_ = EG.lonlat_to_pixel(s["lon"], s["lat"])
                if 0 <= r_ < EG.rows and 0 <= c_ < EG.cols and np.isfinite(road_km[r_, c_]):
                    row["road_km"] = float(road_km[r_, c_])
        else:
            row.update(safe_km=None, bearing=None, safe_lon=None, safe_lat=None)
        rows.append(row)
    # distance to safe ground for every fine cell, on the coarse grid (for the vulnerability index)
    dist = None
    if safe.any():
        try:
            from scipy import ndimage
            dist = ndimage.distance_transform_edt(~safe) * k * EG.res
        except Exception:
            dist = None
    safe_km2 = float(safe.sum() * (k * EG.res) ** 2 / 1e6)
    return dict(rows=rows, safe=safe, k=k, safe_km2=safe_km2, dist_coarse=dist, roads=road_km is not None)


def expand(coarse, k, shape, fill=0):
    out = np.full(shape, fill, dtype=coarse.dtype)
    big = np.kron(coarse, np.ones((k, k), dtype=coarse.dtype))
    out[:big.shape[0], :big.shape[1]] = big[:shape[0], :shape[1]]
    return out


def _rank01(a, mask):
    a = np.asarray(a, float)
    v = a[mask & np.isfinite(a)]
    out = np.full(a.shape, np.nan, np.float32)
    if v.size < 10 or np.nanmax(v) == np.nanmin(v):
        return out
    lo, hi = np.percentile(v, [2, 98])
    out[mask] = np.clip((a[mask] - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def vulnerability(HG, built, hand, dist_safe_m=None, dist_health_m=None, wealth=None):
    """Proxy vulnerability 0-1 on the history grid: buildings on very low ground (HAND < 2 m) nearby, distance to safe
    high ground and distance to a health facility. Population is NOT included (it is exposure, used separately in risk)."""
    mask = HG.mask
    comps = {}
    if built is not None and hand is not None:
        low = (np.nan_to_num(hand, nan=99) < 2).astype(np.float32) * np.nan_to_num(built)
        from .depth import _box_mean
        comps["Buildings on very low ground (HAND < 2 m) within 300 m"] = _rank01(_box_mean(low, max(1, int(round(300 / HG.res)))), mask)
    if dist_safe_m is not None:
        comps["Distance to safe high ground"] = _rank01(dist_safe_m, mask)
    if dist_health_m is not None:
        comps["Distance to a health facility"] = _rank01(dist_health_m, mask)
    if wealth is not None and np.isfinite(wealth[mask]).mean() > 0.5:
        comps["Low relative wealth (Meta RWI, 2.4 km)"] = _rank01(-wealth, mask)
    if not comps:
        return None, []
    st = np.stack([c for c in comps.values()])
    with np.errstate(all="ignore"):
        v = np.nanmean(st, axis=0)
    v[~mask] = np.nan
    return v.astype(np.float32), list(comps)


# ------------------------------------------------------------------ plain language
def plain_error(err):
    """Turn an exception text into a sentence a non-programmer understands (technical detail kept in brackets)."""
    e = str(err or "")
    el = e.lower()
    if "429" in el or "too many requests" in el or "busy" in el:
        why = "the free data service was busy (too many requests); running the kit again later usually works"
    elif "timed out" in el or "timeout" in el:
        why = "the data service did not answer in time; check the internet connection and run the kit again"
    elif "name resolution" in el or "unreachable" in el or "connection" in el or "urlopen" in el:
        why = "the data service could not be reached from this computer (internet, proxy or firewall)"
    elif "no data" in el or "not available" in el or "no sentinel" in el or "no reference" in el:
        why = "the needed satellite data does not exist for this place and date"
    elif "too few" in el:
        why = "there was too little data in this area to fit the model"
    else:
        why = "an unexpected problem occurred"
    short = e.replace("\n", " ")[:140]
    return f"{why} (technical detail: {short})" if short else why


# ------------------------------------------------------------------ guidance
def guidance(R, fmt=lambda v: f"{v:,.0f}"):
    """Rule-based, general 'what to do' points from this study's own numbers. Not a substitute for local engineering."""
    X = R.get("more") or {}
    pts = []
    ev = X.get("evac") or {}
    rows = [r for r in ev.get("rows", []) if r.get("people_flooded", 0) > 0]
    if rows:
        top = sorted(rows, key=lambda r: -r["people_flooded"])[:5]
        names = "; ".join(f"{r['lat']:.3f}, {r['lon']:.3f} ({fmt(r['people_flooded'])} people"
                          + (f", safe ground {r['safe_km']:.1f} km {r['bearing']}" if r.get("safe_km") is not None else "") + ")"
                          for r in top)
        pts.append(f"<b>Priority settlements for warning and evacuation planning</b>: {names}. Plan routes on the ground; "
                   "distances here are straight lines.")
    if X.get("isolated_people"):
        pts.append(f"<b>Cut-off places</b>: about {fmt(X['isolated_people'])} people lived on dry islands at the peak; pre-position boats, "
                   "food and medicine there before the next flood season.")
    H = (R.get("hydro") or {}).get("q") or {}
    if H.get("quant"):
        q = H["quant"]
        pts.append(f"<b>Early-warning levels</b> at the main river cell ({H['loc'][1]:.3f}, {H['loc'][0]:.3f}): alert at "
                   f"{fmt(q[2])} m³/s (2-year flow), warning at {fmt(q[5])} m³/s (5-year), danger at {fmt(q[10])} m³/s (10-year). "
                   "Check the GloFAS forecast against these each day in the flood season.")
    fc = (R.get("hydro") or {}).get("forecast")
    if fc and fc.get("level") in ("medium", "high"):
        pts.append(f"<b>Current forecast</b>: some ensemble members exceed the {'10' if fc['level'] == 'high' else '2'}-year flow "
                   f"by {fc['peak_max'][0]}; review preparedness now.")
    if X.get("gi") is not None and X.get("gi_cell_m"):
        n_hot = int((X["gi"] == 1).sum())
        if n_hot:
            pts.append("<b>Drainage and embankment checks</b>: the flood hot-spot map (99% confidence clusters) shows where water "
                       "returns most often; inspect drains, culverts and embankments there first.")
    rp = {r["rp"]: r for r in (X.get("rp_exposure") or [])}
    if 100 in rp:
        r = rp[100]
        pts.append(f"<b>No-build zone</b>: the modelled 100-year river flood zone covers {fmt(r['km2'])} km²"
                   + (f" with {fmt(r['pop'])} people already living in it" if r.get("pop") else "") +
                   "; avoid new construction there and require raised plinths for existing buildings.")
    enc = X.get("encroach") or []
    if len(enc) >= 2:
        z0 = (X.get("encroach_zones") or [None])[0]
        a, b = enc[0].get(z0, (None, None))[0], enc[-1].get(z0, (None, None))[0]
        if a is not None and b is not None and b > a * 1.1:
            pts.append(f"<b>Encroachment</b>: building footprint in land flooded at least once grew from {a:.2f} to {b:.2f} km² "
                       f"({enc[0]['year']}–{enc[-1]['year']}); enforce the flood-zone rules in planning permissions.")
    if R.get("fac"):
        n = sum(1 for f in R["fac"] if f[4])
        if n:
            pts.append(f"<b>Facilities</b>: {n} schools or health facilities stood in or next to flood water; plan alternatives "
                       "or flood-proofing for them.")
    if not pts:
        pts.append("No specific priorities could be derived from the data available in this run.")
    pts.append("These points are general guidance computed from satellite data and open models. Final decisions need local "
               "engineers, the disaster management authority and field verification.")
    return pts
