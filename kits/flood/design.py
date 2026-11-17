# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Design-flood inundation and depth from HAND (Height Above Nearest Drainage), calibrated on the observed flood.

Water rises to a stage h above the drainage network; every cell with HAND ≤ h is wet, with depth h − HAND (Nobre et al. 2016,
Zheng et al. 2018). The stage is set so the wet area matches the flooded area that the river-flow / flood-area relation
predicts for a 10- or 100-year flow; the same rule is first tested on the observed event (critical success index)."""

import datetime as dt
import urllib.parse

import numpy as np

STAGES = np.round(np.arange(0.1, 30.01, 0.1), 2)


def to_grid(arr_e, EG, HG):
    """Nearest-neighbour transfer of a boolean map from the event grid to the history grid (same origin)."""
    r = np.minimum(((np.arange(HG.rows) + 0.5) * HG.res / EG.res).astype(int), EG.rows - 1)
    c = np.minimum(((np.arange(HG.cols) + 0.5) * HG.res / EG.res).astype(int), EG.cols - 1)
    return arr_e[np.ix_(r, c)]


def _valid(hand, mask, perm):
    return mask & np.isfinite(hand) & (hand <= STAGES[-1]) & ~perm


def calibrate(hand, observed, observed_valid, mask, perm):
    """Best stage against the observed flood (max CSI) and the stage whose wet area equals the observed area."""
    v = _valid(hand, mask, perm) & observed_valid
    if observed[v].sum() < 50:
        return None
    h = hand[v]
    o = observed[v]
    best = (0, None, None)
    area_h = None
    n_obs = o.sum()
    for s in STAGES:
        w = h <= s
        tp = (w & o).sum()
        fp = (w & ~o).sum()
        fn = n_obs - tp
        csi = tp / max(tp + fp + fn, 1)
        if csi > best[0]:
            best = (csi, s, (tp / max(n_obs, 1), tp / max(w.sum(), 1)))
        if area_h is None and w.sum() >= n_obs:
            area_h = s
            tp_a, wa = tp, w.sum()
            csi_a = csi
            rec_a, prec_a = tp_a / max(n_obs, 1), tp_a / max(wa, 1)
    if best[1] is None or area_h is None:
        return None
    return dict(stage_best=float(best[1]), csi_best=float(best[0]), recall_best=float(best[2][0]), precision_best=float(best[2][1]),
                stage_area=float(area_h), csi_area=float(csi_a), recall_area=float(rec_a), precision_area=float(prec_a),
                obs_km2=None)


def floodplain_calibrate(hand, slope, dperm, observed, observed_valid, mask, perm, max_stage=15.0):
    """Choose the floodplain (distance to rivers/lakes wide enough to show as permanent water, and gentle ground) and the
    water stage above drainage that best reproduce the observed flood (critical success index). Mountain streams have
    near-zero HAND too, so a single stage over the whole area floods every headwater channel; limiting the model to the
    floodplain of the main rivers removes most of those false alarms."""
    base = mask & np.isfinite(hand) & ~perm
    ov = base & observed_valid
    o_all = observed & ov
    n_obs = int(o_all.sum())
    if n_obs < 50:
        return None
    edges = np.round(np.arange(0.1, max_stage + 0.001, 0.1), 2)
    best = None
    opts = [(None, None)] + [(d, s) for d in (2000, 1000, 500) for s in (None, 5, 3)]
    opts += [(None, 5), (None, 3)]
    for dmax, smax in opts:
        ok = base.copy()
        if dmax is not None:
            if dperm is None:
                continue
            ok &= dperm <= dmax
        if smax is not None:
            ok &= np.nan_to_num(slope, nan=90) <= smax
        h = hand[ok & ov]
        o = o_all[ok & ov]
        tp = np.searchsorted(np.sort(h[o]), edges, side="right")
        allw = np.searchsorted(np.sort(h), edges, side="right")
        fp = allw - tp
        fn = n_obs - tp
        csi_ = tp / np.maximum(tp + fp + fn, 1)
        i = int(np.argmax(csi_))
        if best is None or csi_[i] > best["csi"] + 0.005:
            wet_all = int(((hand <= edges[i]) & ok).sum())
            best = dict(dmax=dmax, smax=smax, stage=float(edges[i]), csi=float(csi_[i]), recall=float(tp[i] / n_obs),
                        precision=float(tp[i] / max(allw[i], 1)), k=float(wet_all / n_obs), ok=ok)
    if best is None:
        return None
    sl = np.nan_to_num(slope, nan=0)[o_all]
    best["flood_slope"] = float(np.median(sl)) if sl.size else 0.0
    parts = []
    if best["dmax"] is not None:
        parts.append(f"within {best['dmax']:,} m of rivers and lakes wide enough to be mapped as permanent water")
    if best["smax"] is not None:
        parts.append(f"on ground gentler than {best['smax']}°")
    best["label"] = " and ".join(parts) if parts else "the whole area"
    return best


def stage_in(hand, ok, km2, px_km2):
    h = np.sort(hand[ok])
    n = int(round(km2 / px_km2))
    if n <= 0 or h.size == 0:
        return None
    return float(min(h[min(n, h.size) - 1], 30.0))


def depth_in(hand, ok, stage):
    v = ok & (hand <= stage)
    d = np.full(hand.shape, np.nan, np.float32)
    d[v] = np.maximum(stage - hand[v], 0.01)
    return d


def stage_for_area(hand, mask, perm, km2, px_km2):
    v = _valid(hand, mask, perm)
    h = np.sort(hand[v])
    n = int(round(km2 / px_km2))
    if n <= 0 or h.size == 0:
        return None
    return float(h[min(n, h.size) - 1])


def depth_map(hand, mask, perm, stage):
    v = _valid(hand, mask, perm) & (hand <= stage)
    d = np.full(hand.shape, np.nan, np.float32)
    d[v] = np.maximum(stage - hand[v], 0.01)
    return d


def csi(a, b, valid):
    a, b = a & valid, b & valid
    tp = (a & b).sum()
    return float(tp / max(tp + (a & ~b).sum() + (~a & b).sum(), 1))


# ------------------------------------------------------------------ river-flow forecast (GloFAS via Open-Meteo)
def forecast(get, lon, lat, days=210):
    q = urllib.parse.urlencode(dict(latitude=f"{lat:.4f}", longitude=f"{lon:.4f}", forecast_days=days,
                                    daily="river_discharge,river_discharge_median,river_discharge_max,river_discharge_p75,river_discharge_p25,river_discharge_min"))
    url = "https://flood-api.open-meteo.com/v1/flood?" + q
    import time
    for i, wait in enumerate((0, 10, 30, 60)):
        time.sleep(wait)
        try:
            d = get(url)
            break
        except Exception as e:      # the free API answers 429 when many requests come from one address
            if "429" not in str(e) or i == 3:
                raise
    t = [dt.date.fromisoformat(x) for x in d["daily"]["time"]]
    def arr(k):
        return np.array([np.nan if x is None else x for x in d["daily"].get(k, [None] * len(t))], float)
    return url, dict(dates=t, control=arr("river_discharge"), median=arr("river_discharge_median"), max=arr("river_discharge_max"),
                     p75=arr("river_discharge_p75"), p25=arr("river_discharge_p25"), min=arr("river_discharge_min"))
