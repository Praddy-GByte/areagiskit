# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Report chapter for the AreaGIS Kit: Flood."""

import datetime as dt

import numpy as np

from ...core import datasets as D
from ...core.maps import MapSpec, render_bars, render_grouped_bars, render_line, render_multi_line
from ...core.report import Report, fmt, grade
from . import ems as EMS
from . import gfm as G
from . import impact as IM
from . import suscept as S
from . import response as RS

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _n(v, d=0):
    return "–" if v is None else fmt(v, d)


def _roc_png(series):
    from ...core import maps as M
    if not M.HAVE_MPL:
        return None
    fig, ax = M._chart_fig(5.2, 4.2)
    for name, fpr, tpr, col in series:
        ax.plot(fpr, tpr, color=col, lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], color=M.RULE, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate (1 − specificity)", fontsize=7.5, color=M.MUTED)
    ax.set_ylabel("True positive rate (sensitivity)", fontsize=7.5, color=M.MUTED)
    ax.grid(color=M.RULE, lw=0.4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return M._png(fig)


def _rgb_png(before, after, mask, title):
    from ...core import maps as M
    if not M.HAVE_MPL:
        return None
    sc = lambda a: np.clip((np.nan_to_num(a.astype(np.float32), nan=-25) + 25) / 25, 0, 1)
    img = np.dstack([sc(before), sc(after), sc(after)])
    img[~mask] = 1
    step = max(1, int(max(img.shape[:2]) / 1600))
    img = img[::step, ::step]
    h, w = img.shape[:2]
    fig, ax = M._chart_fig(6.3, 6.3 * h / max(w, 1))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=M.INK, loc="left")
    return M._png(fig)


def _dd(s):
    d = dt.date.fromisoformat(str(s)[:10])
    return f"{d.day} {d:%b %Y}"


def build(kit, R, aoi, out_dir, files, layers):
    from ..floodstudy import CMP, EVENT, FREQ, MODELS, SUSC
    from ..lulc.classify import CREDIT_GHSL
    from ..lulc.compare import CREDIT_SOIL
    from ..lulc.extra import CREDIT_POP
    EG, HG, H = R["EG"], R["HG"], R["H"]
    pk = R["peak"]["date"]
    imp = R["impact"]
    sus = R.get("sus")
    src_gfm = "Copernicus GFM (Sentinel-1)"

    # ------------------------------------------------------------ validation checks → verdict
    checks, vrows = [], []
    if R.get("ems"):
        e = R["ems"][0]
        s = e.get("combined") or e["gfm"]
        if s:
            checks.append((3 if s["csi"] >= 0.6 else 2 if s["csi"] >= 0.45 else 1 if s["csi"] >= 0.3 else 0, 3,
                           f"Expert Copernicus EMS map {e['code']} ({e['aoi']}): critical success index {s['csi']:.2f}, F1 {s['f1']:.2f}."))
            vrows.append(("Expert map (Copernicus EMS " + e["code"] + ")",
                          f"CSI {s['csi']:.2f}, F1 {s['f1']:.2f}, flood found {s['pa'] * 100:.0f}%, correct {s['ua'] * 100:.0f}%",
                          f"Analysts mapped the flood from a {e['sensor'] or 'satellite'} image of {e['image'][:10]}; compared with the GFM pass of "
                          f"{e['gfm_date']} ({e['gap']} day gap). CSI above 0.6 is good for flood maps."))
    else:
        vrows.append(("Expert map (Copernicus EMS)", "No activation found for this place and date",
                      "Copernicus EMS maps only events it was asked to map; none overlaps this area within 30 days."))
    if R.get("own") and R["own"]["scores"]:
        s = R["own"]["scores"]
        checks.append((2 if s["f1"] >= 0.7 else 1 if s["f1"] >= 0.5 else 0, 2,
                       f"Independent re-detection from raw radar agrees with GFM: F1 {s['f1']:.2f}, kappa {s['kappa']:.2f}."))
        vrows.append(("Independent re-detection", f"F1 {s['f1']:.2f}, kappa {s['kappa']:.2f}",
                      "A separate change-detection method on the raw radar images (this kit's own algorithm) against the GFM ensemble."))
    if R.get("optical"):
        o = R["optical"]
        checks.append((2 if o["confirmed"] >= 70 else 1 if o["confirmed"] >= 50 else 0, 2,
                       f"A Sentinel-2 optical image of {o['date']} sees water at {o['confirmed']:.0f}% of radar water pixels."))
        vrows.append(("Optical satellite", f"{o['confirmed']:.0f}% of radar water confirmed (kappa {o['kappa']:.2f})",
                      "Different physics (sunlight, MNDWI index) from a different satellite."))
    if sus:
        r = sus["res"]["ens"]
        checks.append((2 if r["auc_holdout"] >= 0.85 else 1 if r["auc_holdout"] >= 0.75 else 0, 2,
                       f"Susceptibility (ensemble) predicts held-out flood pixels with AUC {r['auc_holdout']:.2f}."))
        if r.get("auc_temporal") is not None:
            checks.append((2 if r["auc_temporal"] >= 0.75 else 1 if r["auc_temporal"] >= 0.65 else 0, 2,
                           f"Trained on floods up to {R['temporal']['split']}, it predicts places first flooded later with AUC {r['auc_temporal']:.2f}."))
        if r.get("auc_glofas") is not None:
            checks.append((1 if r["auc_glofas"] >= 0.7 else 0, 1,
                           f"It agrees with the independent GloFAS 100-year flood model (AUC {r['auc_glofas']:.2f})."))
    if R.get("user_ref") and R["user_ref"].get("max"):
        s = R["user_ref"]["max"]
        checks.append((3 if s["csi"] >= 0.6 else 2 if s["csi"] >= 0.45 else 1 if s["csi"] >= 0.3 else 0, 3,
                       f"Your observed flood extent: CSI {s['csi']:.2f}, F1 {s['f1']:.2f}."))
    pk_cov = R["peak_obs"].sum() / max(EG.mask.sum(), 1) * 100
    checks.append((1 if pk_cov >= 90 else 0, 1, f"Radar observed {pk_cov:.0f}% of the area at the flood peak."))
    checks.append((1 if len(H["series"]) >= 100 else 0, 1, f"{len(H['series'])} radar passes form the flood history."))
    verdict = grade(checks)
    event_checked = bool(R.get("ems") or (R.get("own") and R["own"]["scores"]) or R.get("optical") or R.get("user_ref"))
    if not event_checked and verdict["level"] == "High":
        verdict["level"] = "Medium"
        verdict["reasons"].append("The event map could not be checked against any independent source (no expert map, "
                                  "no independent re-detection, no clear optical image), so confidence is capped at Medium.")
    if imp["peak"]["km2"] < 0.5:
        verdict["reasons"].append("Radar saw almost no flood water on the peak date. Either there was no flood here on the radar "
                                  "dates, or it was missed (short flood, water under trees or in towns).")

    # ------------------------------------------------------------ summary
    ttl = f"Flood study, {R['H']['years'][0]}–{R['H']['years'][-1]}; event of {_dd(pk)}"
    rep = Report(kit.NAME, ttl, aoi.name, verdict, "#1f5fa8", aoi, EG)
    peak_map = MapSpec(R["ev_map"], EG, f"Flood extent, {_dd(pk)}", classes=EVENT, subtitle=f"{aoi.name} · {src_gfm}",
                       source=G.CREDIT_GFM[:90] + "…")
    mi, pi = imp["max"], imp["peak"]
    results = [("Flooded at the peak", f"{fmt(pi['km2'])} km² ({_dd(pk)})"),
               ("Maximum extent during the event", f"{fmt(mi['km2'])} km²")]
    if mi.get("pop") is not None:
        results.append(("People living in flooded area", f"{_n(mi['pop'])}"))
    if mi.get("built_m2") is not None:
        results.append(("Building footprint flooded", f"{fmt(mi['built_m2'] / 1e6, 3)} km²"))
    if mi["lc"].get(5):
        results.append(("Cropland flooded", f"{fmt(mi['lc'][5])} km²"))
    if R.get("depth_stats") and R["depth_stats"]["median"] is not None:
        if R.get("depth_reliable"):
            results.append(("Water depth (median / 90th pct.)", f"{R['depth_stats']['median']:.1f} m / {R['depth_stats']['p90']:.1f} m"))
            results.append(("Flood water volume", f"{fmt(R['vol_mcm'], 1)} million m³"))
        else:
            results.append(("Water depth", "not reliable here (flat land; see section 4.2)"))
    results.append((f"Area flooded at least once {H['years'][0]}–{H['years'][-1]}", f"{fmt(R['ever_km2'])} km²"))
    Hy0, X0 = R.get("hydro") or {}, R.get("more") or {}
    if Hy0.get("rain_rp") and Hy0["rain_rp"].get(3) and Hy0["rain_rp"][3].get("rp"):
        r3 = Hy0['rain_rp'][3]
        results.append(("3-day rain before the peak", f"{r3['value']:.0f} mm ({_rpy(r3['rp'], r3.get('n'))})"))
    if Hy0.get("q") and Hy0["q"].get("rp"):
        results.append(("Peak river flow", f"{Hy0['q']['peak'][1]:,.0f} m³/s ({_rpy(Hy0['q']['rp'], len(Hy0['q'].get('am') or {}))})"))
    if X0.get("settlements"):
        results.append(("Settlements with flood water", f"{len(X0['settlements'])}"))
    if X0.get("isolated_rows"):
        results.append(("Inhabited places cut off by water", f"{sum(1 for r in X0['isolated_rows'] if r['built_share'] >= 5)}" + (f" ({_n(X0.get('isolated_people'))} people)" if X0.get("isolated_people") else "")))
    if sus:
        vh = next(r for r in sus["exposure"] if r["cls"] == 5)
        results.append(("Very high flood susceptibility", f"{fmt(vh['km2'])} km²"))
        results.append(("Susceptibility model accuracy (AUC)", f"{sus['res']['ens']['auc_holdout']:.2f}"))
    if R.get("ems") and (R["ems"][0].get("combined") or R["ems"][0]["gfm"]):
        results.append(("Agreement with expert EMS map (CSI)", f"{(R['ems'][0].get('combined') or R['ems'][0]['gfm'])['csi']:.2f}"))
    points = [f"On <b>{_dd(pk)}</b> radar saw <b>{fmt(pi['km2'])} km²</b> of flood water "
              f"({pi['km2'] / max(aoi.area_km2, 1e-9) * 100:.1f}% of the area); across the event (−15 to +45 days) "
              f"<b>{fmt(mi['km2'])} km²</b> was under water at some time."
              + (" The date was chosen automatically as the largest flood in the archive." if R["event_auto"] else "")]
    if mi.get("pop") is not None:
        points.append(f"About <b>{_n(mi['pop'])} people</b> live in the flooded area (GHS-POP {R['pop_epoch']}).")
    if R.get("depth_stats") and R["depth_stats"]["median"] is not None and R.get("depth_reliable"):
        points.append(f"Typical water depth was <b>{R['depth_stats']['median']:.1f} m</b>; the flood held about "
                      f"<b>{fmt(R['vol_mcm'], 1)} million m³</b> of water.")
    points.append(f"Since {H['years'][0]}, <b>{fmt(R['ever_km2'])} km²</b> has flooded at least once and "
                  f"<b>{fmt(R['recurrent_km2'])} km²</b> floods in at least half of the years.")
    if R.get("trend"):
        t = R["trend"]
        points.append(f"The yearly flooded area shows {'an <b>increasing</b>' if t['slope'] > 0 else 'a <b>decreasing</b>'} trend of "
                      f"{fmt(abs(t['slope']))} km²/year ({'significant' if t['p'] < 0.05 else 'not significant'}, Mann-Kendall p = {t['p']:.2f}).")
    if sus:
        best = max(("fr", "ahp", "lr", "rf"), key=lambda k: sus["res"][k]["auc_holdout"])
        points.append(f"Of four susceptibility models, <b>{dict((k, n) for k, n, _ in MODELS)[best]}</b> performed best "
                      f"(AUC {sus['res'][best]['auc_holdout']:.2f}).")
    rep.summary(results, points, peak_map)

    rep.study_area(MapSpec(np.where(EG.mask, R["L"]["E"]["dem"], np.nan), EG, "Study area: elevation", ramp="elev",
                           vmin=float(np.nanpercentile(R["L"]["E"]["dem"], 2)), vmax=float(np.nanpercentile(R["L"]["E"]["dem"], 98)),
                           unit="m", subtitle=aoi.name, source="Copernicus DEM GLO-30"),
                   [("Flood event studied", _dd(pk) + (" (largest on record, chosen automatically)" if R["event_auto"] else "")),
                    ("Flood history", f"{H['years'][0]}–{H['years'][-1]}, {len(H['series'])} radar passes"
                     + ("" if not (R["params"].get("months") or "").strip() else f", months {R['params']['months']}")),
                    ("History grid", f"{HG.res:.0f} m"), ("Event grid", f"{EG.res:.0f} m")])

    rows = [("Copernicus GFM ensemble flood extent", "Flood water on every radar pass; event, history, inventory", "20 m",
             f"{len(H['series'])} passes {H['series'][0]['date']} to {H['series'][-1]['date']}", "Copernicus / CEMS terms (free, attribution)"),
            ("Sentinel-1 RTC (VV, VH)", "Independent re-detection of the event", "10 m",
             (R["own"]["date"] + "; reference " + ", ".join(R["own"]["refs"])) if R.get("own") else "not used", "Copernicus data"),
            ("Sentinel-2 L2A", "Optical cross-check", "10–20 m", R["optical"]["date"] if R.get("optical") else "none clear", "Copernicus data"),
            ("Copernicus DEM GLO-30", "Depth, slope, elevation, TPI, curvature", "30 m", "2011–2015", "Copernicus DEM licence"),
            ("ASF GLO-30 HAND", "Height above drainage, drainage network", "30 m", "2021", "CC0"),
            ("JRC Global Surface Water", "Permanent water", "30 m", "1984–2021", "CC BY 4.0"),
            ("IO/Esri annual land cover", "What was flooded; land cover factor", "10 m", str(R["lc_year"]), "CC BY 4.0"),
            ("GHS-BUILT-S R2023A", "Building footprint flooded; built-up factor", "100 m", str(R["ghsl_epoch"]), "CC BY 4.0"),
            ("GHS-POP R2023A", "People in the flood", "100 m", str(R["pop_epoch"]), "CC BY 4.0"),
            ("SoilGrids 2.0 clay", "Soil factor", "250 m", "2020", "CC BY 4.0"),
            ("GloFAS river flood hazard v2.1", "Independent model of 10/100/500-year floods", "90 m", "2026 release", "ODbL 1.0"),
            ("OpenStreetMap", "Roads, schools, health facilities", "vector", "today", "ODbL 1.0"),
            ("NASA POWER daily rainfall", "Event rain, return periods, trends", "~50 km", "1981–today", "Public, no account"),
            ("GloFAS v4 river discharge", "Hydrograph, flood frequency, flood–flow relation", "~5 km river network", "1984–today", "CC BY 4.0 (Open-Meteo)"),
            ("Copernicus EMS Rapid Mapping", "Expert flood delineation for validation", "vector",
             ", ".join(e["code"] for e in R.get("ems", [])) or "none for this event", "Copernicus terms")]
    rep.data(rows)
    rep.method([
        ("Official flood maps", "Copernicus Global Flood Monitoring (GFM) runs three independent flood algorithms (DLR, TU Wien, LIST) on every "
         "Sentinel-1 radar image and keeps flood where at least two agree. Every pass over the area since the start year was read."),
        ("Flood classifier", "The final flood map comes from a small convolutional neural network (U-Net) that looks at each pixel and "
         "its surroundings: Sentinel-1 VV and VH after the flood, their change from the same season a year earlier and from 2–8 weeks "
         "before, the three GFM algorithms and GFM likelihood, height above drainage, slope, permanent water and land cover. It was "
         "trained on 52 floods mapped by Copernicus EMS analysts from other satellites (ICEYE, COSMO-SkyMed, TerraSAR-X, PAZ, SPOT, "
         "WorldView) in 20 countries. Tested on floods it had never seen (leave-event-out cross-validation), its mean critical success "
         "index is 0.46, against 0.39 for GFM alone and 0.44 for GFM plus simple change detection; it finds 69% of the expert-mapped "
         "flood (GFM 49%) with 61% of mapped pixels confirmed (GFM 69%). If the classifier cannot run, GFM plus change detection is used."),
        ("Event", "The peak is the pass with the largest flood share near the given date (or the largest flood on record). Passes within "
         "±2 days are combined to fill gaps between orbits. All passes from 15 days before to 45 days after give the timeline, the maximum "
         "extent and the duration (days between the first and last time each place was seen flooded)."),
        ("Depth and volume", "FwDET method (Cohen et al. 2018, 2019): the water surface is the ground height where the flood meets dry land, "
         "carried to every flooded cell from its nearest edge and smoothed; depth = water surface − ground height."),
        ("Impact and damage", "Flood and depth are overlaid on population (GHS-POP), building footprint (GHS-BUILT-S), land cover and "
         "OpenStreetMap roads and facilities. Damage uses the JRC global depth–damage curves for the continent (Huizinga et al. 2017)."),
        ("History", "For each pixel: how often it was flooded (share of radar looks), in how many years, and the yearly flooded area with a "
         "Mann-Kendall trend test and Sen's slope."),
        ("Susceptibility", "Eleven conditioning factors are derived automatically. The flood inventory is automatic: pixels seen flooded in the "
         "archive versus pixels never flooded in at least 20 radar looks. Four models are built (Frequency Ratio, AHP with fixed expert "
         "weights, Logistic Regression and Random Forest) and averaged into an ensemble, classified into five classes by natural breaks."),
        ("Validation", "Random 30% hold-out (AUC), temporal test (trained on early years, tested on places first flooded later), comparison "
         "with the GloFAS hydrodynamic model, independent re-detection from raw radar, an optical image, and expert Copernicus EMS maps."),
        ("Design floods", "HAND stage model: the stage is found whose wet area (HAND ≤ stage) equals the area predicted for a 2/10/25/100-year "
         "flow by the flow–area relation of past radar floods; depth = stage − HAND. The same rule is first scored against the observed flood."),
        ("Forecast", "GloFAS ensemble river-flow forecast (median, 25th/75th percentile, min and max of the members) at the main river cell, "
         "compared with the 2- and 10-year flows from the Gumbel fit of the 1984–today record."),
        ("Hidden flood", "Sentinel-1 VV at least 3 dB brighter than the dry reference while VH does not get brighter (double bounce between water "
         "and stems or walls), within 1 km of open flood water, on low ground (HAND ≤ 10 m), in trees, shrubs, grass, crops, wetland or built-up "
         "areas. Plain VV brightening was tested and rejected: on the Gudbrandsdalen 2023 flood it was no better than chance against the EMS map."),
        ("Floodplain for design floods", "The HAND model is limited to the floodplain that best reproduces the observed flood: chosen among "
         "the whole area, ground within 500 m, 1 km or 2 km of rivers and lakes mapped as permanent water, and slopes under 3° or 5°."),
    ])

    # ------------------------------------------------------------ 4.1 event
    rep.heading("The flood event")
    if imp["peak"]["km2"] < 0.5:
        rep.text("<b>Radar saw almost no flood water here on the radar dates around this event.</b> Either the flood did not reach this "
                 "area on those days, it drained before the satellite passed, or the water was hidden under trees, crops or between buildings. "
                 "The history and susceptibility sections remain valid.")
    if R.get("own_err"):
        rep.text("The independent re-detection could not run: " + RS.plain_error(R["own_err"]) + ".")
    if not R["event_auto"]:
        ev = R["ev"]
        dates = sorted({dt.date.fromisoformat(s["date"]) for s in H["series"]})
        before = [d for d in dates if d <= ev]
        after = [d for d in dates if d >= ev]
        if before and after and (after[0] - before[-1]).days >= 5:
            rep.text(f"<b>Note:</b> Sentinel-1 did not image this area between {_dd(before[-1])} and {_dd(after[0])}, around the date you gave "
                     f"({_dd(ev)}). The highest water may have fallen in this gap, so the peak shown here can be smaller than the real one. "
                     "Commercial radar satellites used by Copernicus EMS may have seen more (see the validation section).")
    rep.text(f"Bright blue is flood water on {_dd(pk)}; dark blue is water that is normally there; grey marks places the radar cannot "
             "judge (radar shadow, dense forest or dense town, flagged by GFM) or did not see.")
    rep.map(peak_map, f"Flood extent at the peak, {_dd(pk)}.")
    tl = R["timeline"]
    if len(tl) >= 3:
        rep.line([("Flooded area (km²)", [t["km2"] for t in tl], "#1f6fd1")], [t["date"][5:] for t in tl],
                 "Flooded area on each radar pass around the event (month-day). Low values can also mean a pass covered only part of the area.",
                 "km²")
        rep.table(["Date", "Days from peak", "Area observed", "Flooded km²", "Flooded share of observed"],
                  [[t["date"], f"{t['day']:+d}", f"{t['obs']:.0f}%", fmt(t["km2"]), f"{t['share']:.1f}%"] for t in tl],
                  "Event timeline from every Sentinel-1 pass")
    mx = np.where(EG.mask, np.where(R["max_ext"], 1, np.where(R["obs_any"], 3, 4)), 0).astype(np.uint8)
    rep.map(MapSpec(mx, EG, "Maximum flood extent during the event", classes={1: EVENT[1], 3: EVENT[3], 4: EVENT[4]},
                    subtitle="Any pass from −15 to +45 days", source=src_gfm), "Everything that was flooded at some time during the event.")
    dur = np.where(R["max_ext"], R["duration"], np.nan)
    if np.isfinite(dur).any():
        rep.text(f"Water stayed for a median of <b>{np.nanmedian(dur):.0f} days</b>; "
                 f"{(dur >= 14).sum() * EG.px_km2:,.1f} km² stayed flooded for two weeks or more.")
        rep.map(MapSpec(dur, EG, "Flood duration", ramp="depth", vmin=0, vmax=float(max(np.nanpercentile(dur, 98), 6)), unit="days",
                        subtitle="Days between first and last time seen flooded", source=src_gfm), "How long the water stayed.")

    # ------------------------------------------------------------ 4.1b why it flooded: rainfall and discharge
    Hy = R.get("hydro") or {}
    if Hy.get("rain_window") or Hy.get("q"):
        rep.heading("Why it flooded: rainfall and river flow")
    if Hy.get("rain_window"):
        rw = Hy["rain_window"]
        rep.figure(render_grouped_bars([d.strftime("%d %b") if i % 5 == 0 else "" for i, (d, _) in enumerate(rw)],
                                       [("Daily rainfall (mm)", [v for _, v in rw], "#2f6fb0")], "mm"),
                   f"Daily rainfall over the area from 30 days before to 10 days after {_dd(pk)} (NASA POWER).")
        t = Hy["rain_tot"]
        rep.text("Rain that fell before the peak: " + ", ".join(f"{k}-day <b>{t[k]:.0f} mm</b>" for k in (1, 3, 5, 7, 15, 30)) + ".")
        if Hy.get("rain_rp"):
            rows = []
            for k, r in Hy["rain_rp"].items():
                rows.append([f"{k}-day", f"{r['value']:.0f}", _rpt(r["rp"], r.get("n")), " / ".join(f"{r['q'][T]:.0f}" for T in (2, 10, 25, 100)),
                             f"{r['ks']:.2f} ({'fits' if r['ks'] < r['ks_crit'] else 'poor fit'})"])
            rep.table(["Rain total", "This event (mm)", "Return period (years)", "2 / 10 / 25 / 100-year rain (mm)", "Gumbel fit (KS D)"], rows,
                      f"How rare was the rain? Gumbel fit to the yearly maxima {min(Hy['rain_annual'])}–{max(Hy['rain_annual'])} "
                      "(Kolmogorov–Smirnov test at 5%).")
        if Hy.get("rain_annual"):
            ys = sorted(Hy["rain_annual"])
            rep.line([("Yearly rainfall (mm)", [Hy["rain_annual"][y] for y in ys], "#2f6fb0"),
                      ("Wettest day ×10 (mm)", [Hy["rain_annual_max"][y] * 10 for y in ys], "#c4281b")],
                     [str(y) if (y - ys[0]) % 5 == 0 else "" for y in ys], "Yearly rainfall and the wettest day of each year.", "mm")
            tt, tm = Hy.get("rain_trend_total"), Hy.get("rain_trend_max")
            if tt and tm:
                rep.text(f"Trend (Mann–Kendall, Sen's slope): yearly rainfall {tt['slope']:+.1f} mm/year (p = {tt['p']:.2f}); "
                         f"wettest day {tm['slope']:+.2f} mm/year (p = {tm['p']:.2f}). " +
                         ("Rainfall extremes are significantly increasing. " if tm["p"] < 0.05 and tm["slope"] > 0 else "") +
                         ("Values below p = 0.05 are statistically significant." if min(tt["p"], tm["p"]) >= 0.05 else ""))
        if Hy.get("rain_month"):
            rm = Hy["rain_month"]
            mon = R["monthly"]
            rep.figure(render_grouped_bars(MON, [("Mean monthly rain (mm)", [rm.get(m, 0) for m in range(1, 13)], "#2f6fb0"),
                                                 ("Flooded share ×100", [(mon[m][0] or 0) * 100 for m in range(1, 13)], "#1f6fd1")], ""),
                       "Rain season and flood season side by side.")
    elif Hy.get("rain_err"):
        rep.text("Rainfall data (NASA POWER) could not be read during this run.")
    q = Hy.get("q")
    if q:
        win = q["window"]
        rep.line([("Discharge (m³/s)", [v for _, v in win], "#0b2f5c")], [d.strftime("%d %b") if i % 10 == 0 else "" for i, (d, _) in enumerate(win)],
                 f"River discharge at the main river cell ({q['loc'][1]:.3f}°N, {q['loc'][0]:.3f}°E), GloFAS reanalysis.", "m³/s")
        txt = (f"Peak discharge <b>{q['peak'][1]:,.0f} m³/s</b> on {_dd(q['peak'][0])} (long-term mean {q['mean']:,.0f} m³/s).")
        if q.get("rp"):
            txt += f" Return period <b>{_rpy(q['rp'], len(q.get('am') or {}))}</b> (Gumbel), {_n(q['rp_emp'], 0)} years from the ranked record."
        if q.get("days_above"):
            txt += f" Flow stayed above the 2-year flood for {q['days_above'][2]} days and above the 5-year flood for {q['days_above'][5]} days."
        rep.text(txt)
        if q.get("quant"):
            rep.table(["Return period (years)", "2", "5", "10", "25", "50", "100"],
                      [["Discharge (m³/s)"] + [f"{q['quant'][T]:,.0f}" for T in (2, 5, 10, 25, 50, 100)]],
                      f"Flood frequency of the river (Gumbel fit to {len(q['am'])} yearly maxima; KS D {q['ks'][0]:.2f}, "
                      f"critical {q['ks'][1]:.2f})")
        if q.get("trend"):
            t = q["trend"]
            rep.text(f"Yearly peak flow trend: {t['slope']:+,.1f} m³/s per year (Mann–Kendall p = {t['p']:.3f}, {t['n']} years)."
                     " GloFAS is a model driven by weather reanalysis; trends partly reflect the model and its inputs.")
    elif Hy.get("q_err"):
        rep.text("River discharge (GloFAS) could not be read during this run.")
    qr = Hy.get("qrel")
    if qr:
        rep.table(["Discharge (m³/s)", "Radar passes", "Median flooded km²", "Largest flooded km²"],
                  [[f"{b['lo']:,.0f}–{b['hi']:,.0f}", b["n"], fmt(b["median_km2"]), fmt(b["max_km2"])] for b in qr["bins"]],
                  f"Flooded area on each radar pass against river flow that day (Spearman ρ = {qr['rho']:.2f}).")
        if qr.get("pred"):
            rep.text("From the straight-line fit on high flows, the river would flood roughly " +
                     ", ".join(f"<b>{fmt(v)} km²</b> at the {T}-year flow" for T, v in qr["pred"].items()) +
                     ". This is an extrapolation from observed floods, not a hydraulic model.")
    if Hy.get("event_area_rp"):
        rep.text(f"Among {len([1 for _, _, n in R['annual'] if n >= 20])} years of radar records, this flood ranks "
                 f"<b>#{Hy['event_rank']}</b> by area; as a yearly maximum it would be expected about once in "
                 f"<b>{Hy['event_area_rp']:.0f} years</b> (empirical, short record).")

    # ------------------------------------------------------------ 4.2 depth
    if R["depth"] is not None:
        rep.heading("Water depth and volume")
        rep.text(f"Depth was estimated from {R['n_edge']:,} flood-edge cells. Median depth {R['depth_stats']['median']:.1f} m, 90th percentile "
                 f"{R['depth_stats']['p90']:.1f} m; total volume about {fmt(R['vol_mcm'], 1)} million m³. The elevation model is a surface "
                 "model (it includes buildings and trees), so depths in towns and forests are less reliable (±0.5–1 m)."
                 + ("" if R.get("depth_reliable") else
                    " <b>Depth is not reliable here because the land is very flat:</b> "
                    + (f"the flooded ground is only {_n(R.get('dem_drop'), 1)} m lower than the dry ground within 500 m, "
                       if (R.get("dem_drop") or 0) >= 0 else
                       f"in the elevation model the flooded ground is not even lower than the dry ground within 500 m (it is "
                       f"{_n(-R.get('dem_drop'), 1)} m higher), ")
                    + "a difference within the elevation model's own error. The water surface cannot be traced "
                    "from the shoreline on such flat land, so read the depths below as a minimum only. For real depths use river gauge "
                    "readings, flood marks on walls or field reports.")
                 + (" Most flooded patches here are narrow, so nearly every flooded cell is an edge cell and its depth is set close "
                    "to zero by the method: treat these depths as a lower bound." if R["depth_stats"]["median"] < 0.1 else ""))
        rep.map(MapSpec(R["depth"], EG, "Flood water depth", ramp="depth", vmin=0, vmax=float(max(np.nanpercentile(R["depth"], 98), 1)),
                        unit="m", subtitle="FwDET from flood edge and Copernicus DEM", source=src_gfm + "; Copernicus DEM"),
                "Estimated water depth (maximum extent).")
        rep.table(["Depth", "Area km²", "People", "Building footprint m²", "Cropland km²"],
                  [[r["label"], fmt(r["km2"]), _n(r["pop"]), _n(r["built_m2"]), _n(r["crop"], 2)] for r in R["depth_rows"]],
                  "Who and what was flooded, by water depth")

    # ------------------------------------------------------------ 4.3 impact
    rep.heading("What was affected")
    lcr = [(k, v) for k, v in sorted(mi["lc"].items(), key=lambda kv: -kv[1]) if v > 0.005 and k in D.LULC]
    if lcr:
        rep.table(["Land cover", "Flooded at peak km²", "Flooded at any time km²", "Share of this land cover"],
                  [[D.LULC[k][0], fmt(pi["lc"].get(k, 0)), fmt(v), f"{v / max((R['L']['E']['lc'] == k).sum() * EG.px_km2, 1e-9) * 100:.0f}%"]
                   for k, v in lcr], f"Flooded land by land cover ({R['lc_year']} map)", [D.LULC[k][1] for k, _ in lcr])
        rep.bars([(D.LULC[k][0], v, D.LULC[k][1]) for k, v in lcr], "Flooded area by land cover (maximum extent).", "km²")
    rows = []
    if mi.get("pop") is not None:
        rows.append(["People living in flooded area", _n(pi["pop"]), _n(mi["pop"])])
    if mi.get("built_m2") is not None:
        rows.append(["Building footprint flooded (m²)", _n(pi["built_m2"]), _n(mi["built_m2"])])
    if R.get("roads"):
        tot, wet = R["roads"]
        rows.append(["Major roads flooded (km)", "–", f"{wet:,.1f} of {tot:,.1f}"])
    if R.get("fac") is not None:
        n_hit = sum(1 for f in R["fac"] if f[4])
        rows.append(["Schools and health facilities in or next to flood", "–", f"{n_hit} of {len(R['fac'])}"])
    if rows:
        rep.table(["Exposure", "At the peak", "During the event"], rows, "People, buildings and infrastructure exposed")
    if R.get("fac"):
        hit = [f for f in R["fac"] if f[4]][:25]
        if hit:
            rep.table(["Facility", "Type", "Location"], [[f[2] or "(no name)", f[3], f"{f[1]:.4f}, {f[0]:.4f}"] for f in hit],
                      "Schools and health facilities in or next to flood water (OpenStreetMap)")
    elif R.get("osm_err"):
        rep.text("OpenStreetMap could not be reached during this run, so roads and facilities are not included.")
    if R.get("damage"):
        dm = R["damage"]
        txt = (f"Damage uses the JRC depth–damage curves for <b>{dm['continent']}</b> (residential buildings) and "
               f"agriculture ({dm['ag_src']}). The fully-damaged equivalent is "
               f"<b>{_n(dm['built_eq_m2'])} m²</b> of building footprint and <b>{fmt(dm['crop_eq_m2'] / 1e6, 2)} km²</b> of cropland.")
        if dm.get("money_built") is not None or dm.get("money_crop") is not None:
            txt += (f" With the values you gave, the estimated direct damage is "
                    f"<b>{_n((dm.get('money_built') or 0) + (dm.get('money_crop') or 0))} {dm['currency']}</b>"
                    f" (buildings {_n(dm.get('money_built'))}, crops {_n(dm.get('money_crop'))}).")
        else:
            txt += " Enter a value per m² for buildings and crops to convert this into money."
        rep.text(txt)

    # ------------------------------------------------------------ 4.3b settlements, cut-off places, hazard, crop damage
    X = R.get("more") or {}
    if R["L"]["E"].get("pop") is None or R.get("built_fallback"):
        rep.text("<b>Note:</b> the JRC server for population and building data (GHS-POP, GHS-BUILT-S) could not be reached during this "
                 "run. People counts and the encroachment trend are missing" +
                 (", and buildings come from the built-up class of the land cover map instead (less detailed)" if R.get("built_fallback") else "") +
                 ". Run the kit again later for the full numbers.")
    if R.get("more_err"):
        rep.text("Some of the additional analyses could not run: " + RS.plain_error(R["more_err"]) + ".")
    if X.get("recession"):
        r = X["recession"]
        rep.text(f"After the peak the water receded by about <b>{r['rate']:.1f} km² per day</b>"
                 + (f"; the flooded area halved within {r['half_days']} days." if r.get("half_days") else
                    f"; it had not halved {r['until']} days after the peak."))
    if X.get("settlements"):
        rep.heading("Settlements affected")
        st = X["settlements"]
        has_pop = R["L"]["E"].get("pop") is not None
        rep.text(f"{X['n_settlements']} built-up settlements were found automatically (connected built-up land, "
                 f"{'land cover built-up class' if R.get('built_fallback') else 'GHS-BUILT-S'}); "
                 f"<b>{len(st)}</b> had flood water inside them.")
        rep.table(["#", "Location (lat, lon)", "Area km²", "People", "Flooded %", "People in flood water"],
                  [[i + 1, f"{r['lat']:.4f}, {r['lon']:.4f}", fmt(r["area_km2"]), _n(r["people"]) if has_pop else "–", f"{r['flooded_pct']:.0f}",
                    _n(r["people_flooded"]) if has_pop else "–"]
                   for i, r in enumerate(st[:25])], "Most affected settlements (full list in settlements_affected.csv)")
    if X.get("isolated_rows") is not None:
        rows = X["isolated_rows"]
        if rows:
            rep.heading("Places cut off by water")
            n_inh = sum(1 for r in rows if r["built_share"] >= 5)
            rep.text(f"<b>{len(rows)}</b> dry islands (at least 0.01 km²) were completely surrounded by flood water on {_dd(pk)}; "
                     f"<b>{n_inh}</b> of them have buildings"
                     + (f", home to about <b>{_n(X.get('isolated_people'))} people</b>" if X.get("isolated_people") else "") +
                     ". These need boats or air access; relief planners should check them first.")
            iso_map = np.where(EG.mask, np.where(X["isolated"], 1, np.where(R["peak_ext"], 2, 3)), 0).astype(np.uint8)
            rep.map(MapSpec(iso_map, EG, "Dry land cut off by flood water", classes={1: ("Cut off", "#d7301f"), 2: ("Flood water", "#1f6fd1"),
                                                                                        3: ("Other land", "#ece9df")}, source="This study"),
                    "Islands of dry land surrounded by water.")
            rep.table(["#", "Location (lat, lon)", "Area km²", "Built-up share"],
                      [[i + 1, f"{r['lat']:.4f}, {r['lon']:.4f}", fmt(r["area_km2"], 3), f"{r['built_share']:.0f}%"] for i, r in enumerate(rows[:20])],
                      "Largest cut-off places with buildings first (full list in cut_off_places.csv)")
    EV = X.get("evac")
    if EV and EV.get("rows"):
        rep.heading("Safe high ground for evacuation planning")
        rows_ev = sorted(EV["rows"], key=lambda r: -(r.get("people_flooded") or 0))
        reach = [r for r in rows_ev if r["safe_km"] is not None]
        far = [r for r in reach if r["safe_km"] > 5]
        rep.text(f"<b>{fmt(EV['safe_km2'])} km²</b> of land was never seen flooded since {H['years'][0]}, stayed dry in this event and lies at "
                 "least 1 m above the highest flooded ground within about 1 km. For each settlement with people in flood water (and each "
                 "inhabited cut-off place) the table gives the straight-line distance and direction to the nearest such ground"
                 + (f"; {len(far)} of them are more than 5 km away" if far else "") +
                 (". The 'by dry road' column follows OpenStreetMap roads that were not under water at the peak; routes still need checking "
                  "on the ground (bridges, embankments and road damage are not known)." if EV.get("roads") else
                  ". Roads could not be read from OpenStreetMap this time, so distances are straight lines: actual routes will be longer and "
                  "may cross water, so check them on the ground."))
        if R.get("evac_map") is not None:
            from ..floodstudy import SAFE as _SAFE
            rep.map(MapSpec(R["evac_map"], EG, "Safe high ground", classes=_SAFE, source="This study (GFM history, Copernicus DEM)"),
                    "Never-flooded ground above nearby flood level; lines to it are in evacuation_to_safe_ground.geojson.")
        rep.table(["#", "Type", "Location (lat, lon)", "People in flood water", "Safe ground (km)", "By dry road (km)", "Direction"],
                  [[i + 1, r["kind"], f"{r['lat']:.4f}, {r['lon']:.4f}", _n(r.get("people_flooded")) if r["kind"] == "settlement" else "cut off",
                    "none in area" if r["safe_km"] is None else f"{r['safe_km']:.1f}",
                    f"{r['road_km']:.1f}" if r.get("road_km") is not None else "–", r["bearing"] or "–"] for i, r in enumerate(rows_ev[:25])],
                  "Nearest safe high ground (full list in evacuation_to_safe_ground.csv)")
        if not reach:
            rep.text("No ground in the study area meets the safe-ground rule (the area is flat and has flooded widely). Look for raised "
                     "embankments, roads and buildings outside the area, which the 30 m elevation model cannot resolve.")
    elif X.get("evac_err"):
        rep.text("Safe ground could not be mapped: " + RS.plain_error(X["evac_err"]) + ".")

    if X.get("hazard") is not None:
        from .extra import HAZARD
        rep.heading("Flood hazard (depth and duration)")
        h = X["hazard"]
        rep.map(MapSpec(h, EG, "Flood hazard from depth and duration", classes=HAZARD, source="This study"),
                "Deeper and longer floods are more hazardous (depth classes 0.5/1/2 m, duration 3/7/14 days).")
        rep.table(["Hazard", "Area km²"], [[HAZARD[c][0], fmt((h == c).sum() * EG.px_km2)] for c in HAZARD], "Area in each hazard class",
                  [HAZARD[c][1] for c in HAZARD])
    if X.get("crop"):
        c = X["crop"]
        rep.heading("Crop damage seen by Sentinel-2")
        rep.text(f"Vegetation index (NDVI) on cropland before the flood ({', '.join(c['dates_before'])}) and 2–10 weeks after "
                 f"({', '.join(c['dates_after'])}): flooded cropland went from {c['flooded']['before']:.2f} to {c['flooded']['after']:.2f}, "
                 f"cropland that stayed dry from {c['not flooded']['before']:.2f} to {c['not flooded']['after']:.2f}. "
                 f"<b>{c['loss_share']:.0f}%</b> of flooded cropland lost more than 0.2 NDVI. The difference between flooded and dry "
                 "cropland is the flood's effect; the change on dry cropland is the normal season.")
        rep.map(MapSpec(c["change"], EG, "Change in crop greenness (NDVI) after the flood", ramp="ndvi", vmin=-0.5, vmax=0.5, unit="ΔNDVI",
                        source="Copernicus Sentinel-2"), "Red = cropland that lost greenness, green = gained.")
    elif X.get("crop_err"):
        rep.text("Crop damage from Sentinel-2 could not be assessed: " + RS.plain_error(X["crop_err"]) + ".")

    # ------------------------------------------------------------ 4.4 history
    rep.heading(f"Flood history {H['years'][0]}–{H['years'][-1]}")
    rep.text((f"{H['skipped']} catalogue entries belong to orbits that do not reach this area and were skipped. " if H.get("skipped") else "")
             + f"{H.get('tried', len(H['series']))} radar passes were read; {len(H['series'])} saw at least 30% of the area and were used"
             + (f" ({H['failed']} could not be downloaded)" if H.get("failed") else "") + f". {fmt(R['ever_km2'])} km² flooded at least once; "
             f"{fmt(R['recurrent_km2'])} km² flooded in at least half of the years: these are the chronic flood areas.")
    rep.map(MapSpec(R["freq_cls"], HG, "How many times each place was seen flooded", classes=FREQ,
                    subtitle=f"{len(H['series'])} Sentinel-1 passes", source=src_gfm), "Flood count from the radar archive.")
    yf = np.where(R["years_flooded"] > 0, R["years_flooded"], np.nan).astype(np.float32)
    rep.map(MapSpec(yf, HG, "Number of years with flooding", ramp="depth", vmin=1, vmax=max(R["n_years"], 2), unit="years",
                    subtitle=f"out of {R['n_years']} years", source=src_gfm), "Flood recurrence.")
    rep.figure(render_grouped_bars([str(y) for y, _, _ in R["annual"]],
                                   [("Area flooded at least once (km²)", [a for _, a, _ in R["annual"]], "#1f6fd1")], "km²"),
               "Area flooded at least once in each year." + (" The current year is incomplete." if R["annual"][-1][0] == dt.date.today().year else ""))
    rep.table(["Year", "Area flooded km²", "Radar passes"], [[y, fmt(a), n] for y, a, n in R["annual"]], "Yearly flooded area")
    mon = R["monthly"]
    rep.figure(render_grouped_bars(MON, [("Mean flooded share of observed area (%)", [mon[m][0] or 0 for m in range(1, 13)], "#2f6fb0")], "%"),
               "Flood season: average share of the observed area under flood water, by month.")
    if R["top"]:
        rep.table(["Rank", "Date", "Flooded km²", "Area observed"],
                  [[i + 1, _dd(s["date"]), fmt(s["km2"]), f"{s['cover'] * 100:.0f}%"] for i, s in enumerate(R["top"])],
                  "Largest floods seen by radar (one per 20 days)")
    if R.get("trend"):
        t = R["trend"]
        rep.text(f"Trend test on {t['n']} complete years: Mann-Kendall Z = {t['z']:.2f}, p = {t['p']:.3f}; Sen's slope "
                 f"{t['slope']:+,.2f} km²/year. " + ("The trend is statistically significant." if t["p"] < 0.05 else
                                                     "The change is not statistically significant; a longer record is needed."))

    # ------------------------------------------------------------ 4.5 susceptibility
    if sus:
        F = R["factors"]
        rep.heading("Flood susceptibility")
        rep.text(f"Flood inventory (automatic): {fmt(R['inv']['pos_km2'])} km² seen flooded versus {fmt(R['inv']['neg_km2'])} km² never "
                 "flooded in at least 20 radar looks. 4,000 points of each were sampled; 70% trained the models and 30% tested them.")
        names = {k: (n, u) for k, n, u, *_ in S.FACTORS}
        for k, ramp in (("hand", "depth"), ("dist_drain", "seq"), ("elev", "elev"), ("slope", "seq")):
            if k in F:
                a = F[k]
                rep.map(MapSpec(a, HG, f"Factor: {names[k][0]}", ramp=ramp, vmin=float(np.nanpercentile(a, 2)),
                                vmax=float(np.nanpercentile(a, 98)), unit=names[k][1], source="Derived (see data table)"),
                        f"Conditioning factor: {names[k][0].lower()}.")
        rep.text("Other factors used: " + ", ".join(names[k][0].lower() for k in sus["keys"] if k not in ("hand", "dist_drain", "elev", "slope"))
                 + ". All factor rasters are in the results folder.")
        vif = sus["vif"]
        rep.table(["Factor", "VIF", "Multicollinearity"],
                  [[names[k][0], f"{v:.1f}", "none" if v < 5 else "moderate" if v < 10 else "high"] for k, v in vif.items()],
                  "Multicollinearity check (variance inflation factor; below 10 is acceptable)")
        # FR tables for top 4 factors by FR range
        rng_fr = {k: max(r["fr"] for r in rows) - min(r["fr"] for r in rows) for k, rows in sus["tables"].items() if rows}
        for k in sorted(rng_fr, key=lambda k: -rng_fr[k])[:4]:
            rows = sus["tables"][k]
            lab = (lambda r: D.LULC.get(r["cls"], (str(r["cls"]),))[0]) if k == "lc" else (lambda r: r["label"])
            rep.table(["Class", "Area %", "Flood %", "Frequency ratio"],
                      [[lab(r), f"{r['area']:.1f}", f"{r['flood']:.1f}", f"{r['fr']:.2f}"] for r in rows],
                      f"Frequency ratio: {names[k][0]} ({names[k][1]}). FR > 1 means floods are more common than average.")
        w = sus["weights"]
        rep.table(["Factor", "AHP weight"], [[names[k][0], f"{w[k]:.3f}"] for k in sus["ahp_keys"]],
                  f"AHP weights from the pairwise comparison matrix (λmax {sus['lam']:.2f}, consistency ratio {sus['cr']:.3f}; "
                  f"{'acceptable, below 0.10' if sus['cr'] < 0.1 else 'above 0.10, interpret with care'})")
        lr = sus["lr"]
        rep.table(["Term", "Coefficient (standardised)", "Std. error", "p-value"],
                  [["Intercept", f"{lr['b'][0]:+.3f}", f"{lr['se'][0]:.3f}", f"{lr['p'][0]:.3g}"]]
                  + [[names[k][0], f"{lr['b'][i + 1]:+.3f}", f"{lr['se'][i + 1]:.3f}", f"{lr['p'][i + 1]:.3g}"] for i, k in enumerate(sus["keys"])],
                  "Logistic regression (land cover entered as its frequency ratio)")
        imp_rows = sorted(sus["rf_imp"].items(), key=lambda kv: -kv[1])
        rep.bars([(names[k][0], v, "#238b45") for k, v in imp_rows], "Random Forest: drop in AUC when each factor is scrambled (importance).", "AUC drop")
        cls_counts = SUSC
        rep.map(MapSpec(sus["cls"], HG, "Flood susceptibility (ensemble of four models)", classes=cls_counts, subtitle=aoi.name,
                        source="This study"), "Flood susceptibility zones.")
        for k in ("fr", "ahp", "lr", "rf"):
            nm = dict((a, b) for a, b, _ in MODELS)[k]
            rep.map(MapSpec(sus["maps"][k], HG, f"Susceptibility index: {nm}", ramp="risk", vmin=0, vmax=1, unit="index",
                            source="This study"), f"Susceptibility from {nm}.")
        res = sus["res"]
        ser = []
        for k, nm, col in MODELS:
            fpr, tpr = res[k]["roc"]
            ser.append((f"{nm} (AUC {res[k]['auc_holdout']:.2f})", fpr, tpr, col))
        rep.figure(_roc_png(ser),
                   "ROC curves on the 30% hold-out sample. The closer to the top-left, the better.")
        rep.table(["Model", "AUC hold-out", "AUC later floods", "AUC vs GloFAS model"],
                  [[nm, f"{res[k]['auc_holdout']:.3f}", _n(res[k].get('auc_temporal'), 3), _n(res[k].get('auc_glofas'), 3)] for k, nm, _ in MODELS],
                  "Model performance. AUC 0.5 = random, 0.7–0.8 acceptable, 0.8–0.9 excellent, above 0.9 outstanding (Hosmer & Lemeshow).")
        dens = sus["density"]
        rep.table(["Class", "Share of area", "Share of observed floods", "Flood density"],
                  [[SUSC[r["cls"]][0], f"{r['area']:.1f}%", f"{r['flood']:.1f}%", f"{r['density']:.2f}"] for r in dens],
                  "Observed floods in each class. Density should rise steadily from very low to very high.",
                  [SUSC[r["cls"]][1] for r in dens])
        ex = sus["exposure"]
        rep.table(["Class", "Area km²", "People", "Building footprint km²", "Cropland km²"],
                  [[SUSC[r["cls"]][0], fmt(r["km2"]), _n(r["pop"]), _n(r["built"], 3), _n(r["crop"], 2)] for r in ex],
                  "Exposure in each susceptibility class", [SUSC[r["cls"]][1] for r in ex])
    elif R.get("sus_err"):
        rep.heading("Flood susceptibility")
        rep.text("Susceptibility could not be modelled: " + RS.plain_error(R["sus_err"]) + ".")

    # ------------------------------------------------------------ 4.5b risk and hot spots
    if X.get("risk") is not None:
        from .extra import RISK, GISTAR
        rep.heading("Flood risk (hazard × exposure)")
        rep.text(f"Risk combines where floods are likely ({X['risk_basis']}) with how many people and buildings are there "
                 "(GHS-POP, GHS-BUILT-S). Classes are quantiles of the risk index among places with any risk.")
        rep.map(MapSpec(X["risk"], HG, "Flood risk", classes=RISK, source="This study"), "Flood risk to people and buildings.")
        rep.table(["Risk", "Area km²", "People"], [[RISK[r["cls"]][0], fmt(r["km2"]), _n(r["pop"])] for r in X["risk_rows"]],
                  "Area and people in each risk class", [RISK[r["cls"]][1] for r in X["risk_rows"]])
    if X.get("vuln") is not None:
        from .extra import RISK
        rep.heading("Vulnerability (proxy-based) and risk")
        rep.text("Who would find it hardest to cope, estimated from open data only: " + "; ".join(X["vuln_comps"]) + ". "
                 "Each is ranked 0–1 and averaged. "
                 + ("Wealth comes from Meta's Relative Wealth Index (2.4 km tiles; CC BY-NC 4.0, non-commercial use only). "
                    if any("wealth" in c for c in X["vuln_comps"]) else
                    (f"Relative wealth was not added: {X['rwi_note']}. " if X.get("rwi_note") else ""))
                 + "Age and house type are not in open global data, so this is a proxy index; "
                 "add census data by ward for a full vulnerability assessment. Risk below = hazard × exposure × (0.5 + 0.5 × vulnerability).")
        rep.map(MapSpec(X["vuln"], HG, "Flood vulnerability (proxy index)", ramp="risk", vmin=0, vmax=1, unit="0–1", source="This study"),
                "Higher = more " + ", ".join(c[0].lower() + c[1:] for c in X["vuln_comps"]) + ".")
        rep.table(["Risk incl. vulnerability", "Area km²", "People"], [[RISK[r["cls"]][0], fmt(r["km2"]), _n(r["pop"])] for r in X["risk_v_rows"]],
                  "Area and people in each risk class when vulnerability is included", [RISK[r["cls"]][1] for r in X["risk_v_rows"]])
    elif X.get("vuln_err"):
        rep.text("The vulnerability index could not be computed: " + RS.plain_error(X["vuln_err"]) + ".")
    if X.get("gi") is not None:
        from .extra import GISTAR
        gi = X["gi"]
        rep.heading("Flood hot spots (Getis-Ord Gi*)")
        rep.text(f"Flood frequency was averaged in {X['gi_cell_m']:.0f} m cells and tested for clustering. Hot spots are places where "
                 "frequent flooding is clustered beyond chance (95% and 99% confidence).")
        rep.map(MapSpec(gi, HG, "Hot spots of flood frequency", classes=GISTAR, source="This study"), "Statistically significant flood clusters.")

    if X.get("encroach"):
        rep.heading("Encroachment: building and population growth in flood zones")
        zn = X["encroach_zones"]
        rows = []
        for r in X["encroach"]:
            rows.append([r["year"]] + [f"{_n(r[z][0], 2)} km² / {_n(r[z][1])}" for z in zn])
        rep.table(["Year"] + zn, rows, "Building footprint and people inside flood zones over time (GHS-BUILT-S, GHS-POP)")
        first, last = X["encroach"][0], X["encroach"][-1]
        z0 = zn[0]
        if first[z0][0] and last[z0][0]:
            rep.text(f"Building footprint in places that flood grew from {first[z0][0]:.2f} km² ({first['year']}) to "
                     f"<b>{last[z0][0]:.2f} km²</b> ({last['year']}), ×{last[z0][0] / max(first[z0][0], 1e-9):.1f}.")

    # ------------------------------------------------------------ 4.6 GloFAS
    g100 = R["glofas"].get(100)
    if g100 is not None:
        rep.heading("Comparison with the GloFAS river flood model")
        rep.text("GloFAS hazard maps come from a hydrological and hydraulic model (LISFLOOD, LISFLOOD-FP), not from satellites, so they are an "
                 "independent view. They cover rivers with catchments above about 5,000 km² only; local drainage and rain flooding are not modelled.")
        rep.map(MapSpec(g100, HG, "GloFAS 100-year river flood depth (model)", ramp="depth", vmin=0, vmax=5, unit="m",
                        source="GloFAS v2.1, CEMS"), "Modelled 100-year flood.")
        if X.get("rp_exposure"):
            rep.table(["Modelled flood", "Area km²", "Deeper than 1 m (km²)", "People", "Building footprint km²"],
                      [[f"{r['rp']}-year", fmt(r["km2"]), fmt(r["deep"]), _n(r["pop"]), _n(r["built"], 3)] for r in X["rp_exposure"]],
                      "Exposure in the modelled river flood zones (GloFAS)")
        if R.get("glofas_obs"):
            go = R["glofas_obs"]
            rep.table(["Model flood zone", "Share of observed flooding inside it"],
                      [[f"{rp}-year", _n(go[rp], 0) + ("%" if go[rp] is not None else "")] for rp in (10, 100, 500)],
                      "How much of the flooding seen by radar falls inside the modelled river flood zones")

    # ------------------------------------------------------------ design floods (HAND model)
    DSG = R.get("design")
    if DSG and (DSG.get("rows") or DSG.get("no_targets")):
        c = DSG["cal"]
        rep.heading("Flood depth for 10- and 100-year river flows (calibrated HAND model)")
        rep.text(f"Water is raised to a stage above the drainage network; every place in the floodplain lower than that stage (Height Above "
                 f"Nearest Drainage) is wet, with depth = stage − HAND (Nobre et al. 2016; Zheng et al. 2018). The floodplain and the stage are "
                 f"chosen to reproduce this flood: {c['label']}, stage {c['stage']:.1f} m. Against the observed flood this gives a critical "
                 f"success index of <b>{c['csi']:.2f}</b> ({c['recall'] * 100:.0f}% of the observed flood found, {c['precision'] * 100:.0f}% of the "
                 "modelled flood observed). "
                 + ("The flooded area for each design flow comes from the relation between river flow and flooded area seen by radar, "
                    "scaled by the same model."
                    if DSG.get("rows") else "Design floods were not computed: past radar floods show no clear rise of flooded area with river "
                    "flow here, so the area of a 10- or 100-year flood cannot be estimated from them."))
        if DSG.get("rows"):
            rep.table(["Flow", "Stage above drainage (m)", "Area km²", "Median depth (m)", "90th pct depth (m)", "People", "Building km²", "Match with GloFAS (CSI)"],
                      [[f"{r['T']}-year", f"{r['stage']:.1f}", fmt(r["km2"]), f"{r['d_med']:.1f}", f"{r['d_p90']:.1f}", _n(r["pop"]), _n(r["built"], 3),
                        f"{r['csi_glofas']:.2f}" if r.get("csi_glofas") is not None else "–"] for r in DSG["rows"]],
                      "Design floods from the calibrated HAND model")
            Tm = max(DSG["maps"])
            rep.map(MapSpec(DSG["maps"][Tm], HG, f"{Tm}-year flood depth (HAND model)", ramp="depth", vmin=0, vmax=3, unit="m",
                            source="ASF HAND, GFM, GloFAS discharge"), f"Modelled depth for the {Tm}-year river flow.")

    if DSG and DSG.get("unusable"):
        c = DSG["cal"]
        rep.heading("Flood depth for 10- and 100-year river flows (calibrated HAND model)")
        if c.get("reason") == "flat":
            rep.text(f"Not computed here: the land is so flat that the elevation model does not show the flooded ground as lower than the "
                     f"dry land around it (median slope in the flood {c['flood_slope']:.1f}°), so height above drainage cannot give "
                     f"reliable depths, even though a {c['stage']:.1f} m stage outlines the flood fairly well (critical success index "
                     f"{c['csi']:.2f}). Use the GloFAS model maps and the flood history instead.")
        else:
            rep.text(f"Not computed here: even inside the best floodplain ({c['label']}) the height-above-drainage model reproduces the "
                     f"observed flood poorly (best critical success index {c['csi']:.2f}; found {c['recall'] * 100:.0f}%, correct "
                     f"{c['precision'] * 100:.0f}%). Narrow valleys, embankments or radar shadow can cause this. Use the GloFAS model maps "
                     "and the flood history instead.")
    elif R.get("design_err"):
        rep.text("Design-flood depths were not computed: " + RS.plain_error(R["design_err"]) + ".")

    # ------------------------------------------------------------ forecast
    FC = (R.get("hydro") or {}).get("forecast")
    if FC:
        rep.heading("River flow forecast for the coming months")
        lvl = {"high": "above the 10-year flow in some ensemble members", "medium": "above the 2-year (bank-full) flow in some members",
               "low": "below the 2-year flow in all members"}[FC["level"]]
        n_ = len(FC["dates"])
        step_ = max(1, n_ // 60)
        rep.line([("Median", FC["median"][::step_], "#2166ac"), ("75th pct", FC["p75"][::step_], "#67a9cf"), ("Highest member", FC["max"][::step_], "#b2182b"),
                  ("2-year flow", [FC["Q"][2]] * len(FC["dates"][::step_]), "#999999"), ("10-year flow", [FC["Q"][10]] * len(FC["dates"][::step_]), "#555555")],
                 [d[5:] for d in FC["dates"][::step_]], f"GloFAS river flow forecast issued {FC['issued']} (ensemble statistics).", "m³/s")
        txt = (f"Over the next {n_} days the forecast is <b>{lvl}</b>. Median peak {FC['peak_med'][1]:,.0f} m³/s on {FC['peak_med'][0]} "
               f"(~{_rp(FC['rp_med'])}-year); highest member {FC['peak_max'][1]:,.0f} m³/s on {FC['peak_max'][0]} (~{_rp(FC['rp_max'])}-year). "
               f"Days above the 2-year flow: median {FC['days_med_q2']}, 75th percentile {FC['days_p75_q2']}, highest member {FC['days_max_q2']}.")
        if FC.get("area_med") is not None:
            txt += (f" From the flow–area relation of past floods, that means roughly {fmt(FC['area_med'])} km² (median) and up to "
                    f"{fmt(FC['area_max'])} km² (highest member) under water.")
        rep.text(txt + " Long-range forecasts lose skill after 2–4 weeks; treat later months as a seasonal outlook, not a warning.")

    elif (R.get("hydro") or {}).get("forecast_err"):
        rep.heading("River flow forecast for the coming months")
        rep.text(f"The GloFAS forecast could not be read this time ({R['hydro']['forecast_err']}). Run again later; the free forecast "
                 "service limits how many requests come from one address.")

    # ------------------------------------------------------------ hidden flood
    FVv = R.get("fv")
    if FVv and (FVv["veg_km2"] + FVv["urb_km2"]) == 0:
        rep.heading("Flood hidden under trees, crops and between buildings")
        rep.text("No radar brightening typical of water under trees, crops or between buildings was found next to the open flood water.")
    elif R.get("fv_err"):
        rep.text("The hidden-flood check could not run: " + RS.plain_error(R["fv_err"]) + ".")
    elif FVv and (FVv["veg_km2"] + FVv["urb_km2"]) > 0:
        rep.heading("Flood hidden under trees, crops and between buildings")
        from ..floodstudy import FV as _FV
        rep.map(MapSpec(FVv["map"], EG, "Possible hidden flood (radar brightening)", classes=_FV, source="Sentinel-1 RTC; ESA WorldCover"),
                "Where VV radar got ≥ 3 dB brighter while VH did not, within 1 km of open flood water: water under tree crowns or crops, or between walls (double bounce).")
        rep.text(f"Besides {fmt(FVv['open_km2'])} km² of open water, about <b>{fmt(FVv['veg_km2'])} km²</b> under trees or crops and "
                 f"<b>{fmt(FVv['urb_km2'])} km²</b> in built-up areas show the radar brightening typical of hidden flood water"
                 + (f" (about {_n(FVv['pop'])} people live there)" if FVv.get("pop") else "") + ". These are candidates: wet crops after rain, "
                 "harvest and new buildings can brighten radar too, so they are not added to the flood totals.")
        chk = [e["fv_check"] for e in (R.get("ems") or []) if e.get("fv_check")]
        if chk:
            c_ = chk[0]
            lift = c_["precision"] / max(c_["base"], 1e-9)
            rep.text(f"Check against the expert map: {c_['precision'] * 100:.0f}% of these candidates are flooded in the EMS map, against "
                     f"{c_['base'] * 100:.0f}% of comparable land next to the flood ({lift:.1f}×). "
                     + ("The signal adds information here." if lift >= 1.2 else "The signal is weak here; treat the candidates with care."))

    # ------------------------------------------------------------ 4.7 independent detection
    if R.get("own"):
        o = R["own"]
        rep.heading("Independent re-detection from raw radar")
        s = o["scores"]
        rep.text(f"This kit's own change-detection method was run on the raw Sentinel-1 image of {o['date']} against a dry reference from "
                 f"the previous year ({', '.join(o['refs'])}), threshold {o['thr']:.1f} dB, and compared with the GFM pass of {o['gfm_date']}. "
                 + (f"Agreement: F1 {s['f1']:.2f}, CSI {s['csi']:.2f}, kappa {s['kappa']:.2f}; this method found {fmt(o['own_km2'])} km² vs GFM {fmt(o['gfm_km2'])} km²."
                    if s else "Too little flood to compare."))
        if R.get("own_pair"):
            b, a = R["own_pair"]
            rep.figure(_rgb_png(b, a, EG.mask, f"Radar change composite: red = before ({', '.join(o['refs'])}), green/blue = {o['date']}"),
                       "Sentinel-1 VV before/after composite. New water shows as red (bright before, dark after); unchanged land is grey; "
                       "permanent water is black.")
        rep.map(MapSpec(o["map"], EG, f"Re-detection ({o['date']}) vs GFM ({o['gfm_date']})", classes=CMP, source="Sentinel-1 RTC; GFM"),
                "Two independent flood algorithms compared.")
    if R.get("optical"):
        o = R["optical"]
        rep.text(f"A Sentinel-2 optical image of {o['date']} independently sees water at {o['confirmed']:.0f}% of the radar water pixels "
                 f"(overall agreement {o['agree']:.0f}%, kappa {o['kappa']:.2f}).")

    # ------------------------------------------------------------ 4.8 EMS
    if R.get("ems"):
        rep.heading("Validation with expert Copernicus EMS maps")
        for e in R["ems"]:
            s = e.get("combined") or e["gfm"]
            rep.text(f"Copernicus EMS activation <b>{e['code']}</b> ({e['aoi']}) mapped the flood by hand from a {e['sensor'] or ''} image of "
                     f"{e['image'][:10]}. It is compared with {'this study’s flood map' if e.get('combined') else 'the GFM pass of ' + e['gfm_date']} inside the EMS mapping area, excluding permanent "
                     "water and places radar cannot see."
                     + (f" Critical success index <b>{s['csi']:.2f}</b>, F1 {s['f1']:.2f}, flood found {s['pa'] * 100:.0f}%, "
                        f"correct when flagged {s['ua'] * 100:.0f}%, area ratio {s['bias']:.2f}." if s else ""))
            rep.map(MapSpec(e["map"], EG, f"GFM ({e['gfm_date']}) vs EMS {e['code']} ({e['image'][:10]})", classes=CMP,
                            source="Copernicus EMS; GFM"), f"Satellite flood map against the expert map {e['code']}.")
        rows = []
        for e in R["ems"]:
            for lab, s in (("This study (flood classifier)" if R.get("combined") == "cnn" else "This study (GFM + second detector)", e.get("combined")), ("GFM ensemble alone", e["gfm"]),
                           ("Second detector alone", e.get("own"))):
                if s:
                    rows.append([f"{e['code']} {e['image'][:10]} ({e['sensor'] or ''}); GFM {e['gfm_date']}", lab, f"{s['csi']:.2f}", f"{s['f1']:.2f}", f"{s['pa'] * 100:.0f}%", f"{s['ua'] * 100:.0f}%", f"{s['kappa']:.2f}"])
        if rows:
            rep.table(["Reference (image date, sensor)", "Map", "CSI", "F1", "Found", "Correct", "Kappa"], rows,
                      "Agreement with expert maps. 'Found' = share of the expert's flood that the map also shows; "
                      "'Correct' = share of the map's flood that the expert also mapped.")
    elif R.get("ems_acts"):
        rep.heading("Copernicus EMS")
        rep.text("Activations found nearby (" + ", ".join(f"{c} {n} {d}" for c, n, d in R["ems_acts"]) +
                 ") had no delineation product overlapping this area.")
    if R.get("user_ref"):
        rep.heading("Validation with your observed flood")
        rows = [[lab, f"{s['csi']:.2f}", f"{s['f1']:.2f}", f"{s['pa'] * 100:.0f}%", f"{s['ua'] * 100:.0f}%"]
                for lab, s in (("Peak extent", R["user_ref"]["peak"]), ("Maximum extent", R["user_ref"]["max"])) if s]
        rep.table(["Map", "CSI", "F1", "Found", "Correct"], rows, "Agreement with the flood extent you supplied")

    # ------------------------------------------------------------ 4.9 zones
    if R.get("zones"):
        rep.heading("Flood by zone")
        rep.table(["Zone", "Area km²", "Flooded km²", "Flooded %", "People in flood", "Median depth m"],
                  [[z["name"], fmt(z["area"]), fmt(z["flood"]), f"{z['share']:.1f}", _n(z["pop"]), _n(z["depth"], 2)] for z in R["zones"][:40]],
                  "Flood statistics by zone (maximum extent)")

    # ------------------------------------------------------------ validation, limits, conclusions
    if sus:
        r = sus["res"]["ens"]
        vrows.append(("Susceptibility hold-out", f"AUC {r['auc_holdout']:.2f}", "30% of inventory points not used for training."))
        if r.get("auc_temporal") is not None:
            vrows.append(("Susceptibility, later floods", f"AUC {r['auc_temporal']:.2f}",
                          f"Trained on floods up to {R['temporal']['split']}; tested on {fmt(R['temporal']['late_km2'])} km² first flooded after that."))
        if r.get("auc_glofas") is not None:
            vrows.append(("Susceptibility vs GloFAS model", f"AUC {r['auc_glofas']:.2f}", "Independent hydrodynamic model of the 100-year flood."))
    vrows.append(("Radar coverage at the peak", f"{pk_cov:.0f}% of the area", "Share seen by radar on the peak date(s)."))
    vrows.append(("Your own check", "50 random points in validation_points.geojson",
                  "Check against photos or field reports and fill the 'correct' column."))
    if R.get("design"):
        c = R["design"]["cal"]
        vrows.append(("HAND flood model vs this flood", f"CSI {c['csi']:.2f} (found {c['recall'] * 100:.0f}%, correct {c['precision'] * 100:.0f}%)"
                      + (" (not used)" if R["design"].get("unusable") else ""),
                      "Floodplain and stage fitted to the observed flood; used for design floods only when CSI ≥ 0.25."))
    rep.heading("What to do: general guidance")
    rep.bullets(RS.guidance(R, fmt=lambda v: _n(v)))
    rep.validation(vrows)
    rep.limits([
        "Design-flood depths use one water stage above drainage for the whole floodplain (HAND model); backwater, levees and embankments are not "
        "modelled, so treat them as a screening map, not an engineering flood study.",
        "The river-flow forecast is the GloFAS ensemble at one river cell; its skill drops after 2–4 weeks.",
        "Radar misses water under dense trees, between tall buildings and in very shallow water among crops; GFM marks such places as not observable.",
        "Sentinel-1 passes every 6–12 days, so a short flash flood can be missed and the true peak may have been larger.",
        "Depth comes from a surface elevation model with buildings and trees; it is an estimate (typically ±0.5–1 m), not a measurement.",
        "Population and buildings are 100 m grids; numbers are estimates for planning, not counts.",
        "The automatic flood inventory covers floods seen since " + str(H["years"][0]) + "; rarer, larger floods are not in it. Rainfall is not a factor.",
        "Damage curves are continental averages; local construction and crop calendars change real losses.",
    ])
    rep.conclusions(points + [f"Overall confidence: <b>{verdict['level']}</b>."])
    from . import hydro as HY
    rep.references([HY.CREDIT_POWER, HY.CREDIT_GLOFAS_Q, "Gumbel, E. J. (1958). Statistics of Extremes. Columbia University Press.",
                    "Getis, A., Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. Geographical Analysis 24, 189–206.",
                    G.CREDIT_GFM, EMS.CREDIT_EMS, IM.CREDIT_GLOFAS, D.CREDIT_S1, D.CREDIT_S2, D.CREDIT_DEM, D.CREDIT_HAND,
                    D.CREDIT_JRC, D.CREDIT_LULC, CREDIT_GHSL, CREDIT_POP, CREDIT_SOIL, D.CREDIT_OSM,
                    "Salamon, P. et al. (2021). The new, systematic global flood monitoring product of the Copernicus Emergency Management Service. IGARSS 2021, 1053–1056.",
                    "Cohen, S. et al. (2018). Estimating floodwater depths from flood inundation maps and topography. JAWRA 54(4), 847–858.",
                    "Cohen, S. et al. (2019). The Floodwater Depth Estimation Tool (FwDET v2.0). Remote Sensing 11(19), 2167.",
                    "Huizinga, J., de Moel, H., Szewczyk, W. (2017). Global flood depth-damage functions. JRC Technical Report JRC105688.",
                    "Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.",
                    "Lee, S., Pradhan, B. (2007). Landslide hazard mapping using frequency ratio and logistic regression. Landslides 4, 33–41.",
                    "Tehrany, M. S., Pradhan, B., Jebur, M. N. (2014). Flood susceptibility mapping using a novel ensemble weights-of-evidence and SVM. J. Hydrology 512, 332–343.",
                    "Breiman, L. (2001). Random Forests. Machine Learning 45, 5–32.",
                    "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica 13, 245–259; Sen, P. K. (1968). JASA 63, 1379–1389.",
                    "Hosmer, D. W., Lemeshow, S. (2000). Applied Logistic Regression, 2nd ed. Wiley.",
                    "Nobre, A. D. et al. (2011). Height Above the Nearest Drainage. Journal of Hydrology 404, 13–29.",
                    "Nobre, A. D. et al. (2016). HAND contour: a new proxy predictor of inundation extent. Hydrological Processes 30, 320–333.",
                    "Zheng, X. et al. (2018). River channel geometry and rating curve estimation using height above the nearest drainage. JAWRA 54, 785–806.",
                    "Tsyganskaya, V. et al. (2018). Detection of temporary flooded vegetation using Sentinel-1 time series data. Remote Sensing 10, 1286.",
                    "Pulvirenti, L. et al. (2016). Use of SAR data for detecting floodwater in urban and agricultural areas: the role of the interferometric coherence. IEEE TGRS 54, 1532–1544.",
                    "Alfieri, L. et al. (2013). GloFAS – global ensemble streamflow forecasting and flood early warning. HESS 17, 1161–1175."])
    srcs = kit.sources()
    gfm_recs = [r for r in srcs if r[0] == "GFM"]
    if gfm_recs:
        import os
        pth = os.path.join(out_dir, "proof_gfm_items.csv")
        with open(pth, "w", encoding="utf-8") as fh:
            fh.write("item_id,date,stac_url\n")
            for _, i, d, u in gfm_recs:
                fh.write(f"{i},{d},{u}\n")
        files.append(pth)
        x0, y0, x1, y1 = aoi.bbox
        srcs = [r for r in srcs if r[0] != "GFM"] + [(
            "GFM", f"{len(gfm_recs)} flood-extent tiles (full list: proof_gfm_items.csv)",
            f"{gfm_recs[0][2]} … {gfm_recs[-1][2]}" if len(gfm_recs) > 1 else gfm_recs[0][2],
            f"{G.STAC}?collections=GFM&bbox={x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}")]
    rep.proof(srcs, kit.reproduce_command())
    rep.files(files + ["report.html", "report.docx", "report.pdf"])
    docs = rep.write(out_dir, kit.seconds())
    kit.step(100, "Done")
    return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers,
                summary=f"Peak {pk}: {fmt(pi['km2'])} km² flooded; event maximum {fmt(mi['km2'])} km².")


def _rp(x):
    return "–" if x is None else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


def _rpt(rp, n=None):
    """Return period for display. Beyond about three times the record length a Gumbel fit is pure extrapolation, so show a bound."""
    if rp is None or not np.isfinite(rp):
        return "–"
    if n and rp > 3 * n:
        return f">{3 * n:.0f} (beyond the {n}-year record)"
    return f"{rp:.1f}" if rp < 10 else f"{rp:.0f}"


def _rpy(rp, n=None):
    """'~3.0-year', or '>126-year, beyond the 42-year record' when a Gumbel fit is extrapolated far past the data."""
    if rp is None or not np.isfinite(rp):
        return "return period not estimated"
    if n and rp > 3 * n:
        return f">{3 * n:.0f}-year, beyond the {n}-year record"
    return f"~{rp:.1f}-year" if rp < 10 else f"~{rp:.0f}-year"
