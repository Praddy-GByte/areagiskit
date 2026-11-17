# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Report chapter for the AreaGIS Kit: Landslide."""

import datetime as dt

import numpy as np

from ...core import datasets as D
from ...core.maps import MapSpec, render_grouped_bars
from ...core.report import Report, fmt, grade
from ..flood import extra as FX
from ..flood import hydro as HY
from . import context as CX
from . import detect as DT
from . import models as LM
from . import stability as ST
from . import runout as RO
from . import seismic as SM

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _n(v, d=0):
    return "–" if v is None else fmt(v, d)


def _curve_png(series, xlabel, ylabel, diag=True):
    from ...core import maps as M
    if not M.HAVE_MPL:
        return None
    fig, ax = M._chart_fig(5.2, 4.2)
    for name, x, y, col in series:
        ax.plot(x, y, color=col, lw=1.5, label=name)
    if diag:
        ax.plot([0, 1], [0, 1], color=M.RULE, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=7.5, color=M.MUTED)
    ax.set_ylabel(ylabel, fontsize=7.5, color=M.MUTED)
    ax.grid(color=M.RULE, lw=0.4)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    return M._png(fig)


def build(kit, R, aoi, out_dir, files, layers):
    from ..landslidestudy import MODELS, SUSC
    g, F, sus = R["g"], R["F"], R["sus"]
    names = LM.NAMES
    mlab = {k: n for k, n, _ in MODELS}
    res = sus.get("res") or {}

    # ------------------------------------------------------------ verdict
    checks, vrows = [], []
    if res.get("ens"):
        a = res["ens"]["auc_holdout"]
        checks.append((2 if a >= 0.85 else 1 if a >= 0.75 else 0, 2, f"Ensemble susceptibility AUC on held-out landslides: {a:.2f}."))
        vrows.append(("Random hold-out (30%)", f"AUC {a:.2f}", "Landslide and stable pixels not used for training."))
    if sus.get("cv"):
        for m, (mu, sd, n) in sus["cv"].items():
            checks.append((2 if mu >= 0.8 else 1 if mu >= 0.7 else 0, 2, f"Spatial cross-validation ({mlab[m]}): AUC {mu:.2f} ± {sd:.2f}."))
            vrows.append((f"Spatial blocks, {n}-fold ({mlab[m]})", f"AUC {mu:.2f} ± {sd:.2f}",
                          f"Whole {sus['cv_block_m'] / 1000:.0f} km blocks held out, so neighbouring pixels never sit in both training and test."))
    if R.get("temporal"):
        t = R["temporal"]
        best = max(t["auc"].values())
        checks.append((2 if best >= 0.75 else 1 if best >= 0.65 else 0, 2, f"Trained on scars before {t['split']}, predicts later scars with AUC up to {best:.2f}."))
        vrows.append(("Future landslides", ", ".join(f"{mlab[k]} {v:.2f}" for k, v in t["auc"].items()),
                      f"Models built only from scars that appeared before {t['split']}, tested on scars that appeared later."))
    if res.get("ens", {}).get("auc_user") is not None:
        a = res["ens"]["auc_user"]
        checks.append((3 if a >= 0.85 else 2 if a >= 0.75 else 1 if a >= 0.65 else 0, 3, f"Your own landslide inventory (never used for training): AUC {a:.2f}."))
        vrows.append(("Your inventory (independent)", f"AUC {a:.2f}", "The strongest test: landslides mapped by someone else, unseen by the models."))
    ro = R.get("runout")
    if ro and ro.get("val"):
        v = ro["val"]
        vrows.append(("Runout vs your inventory", f"{v['hit']:.0f}% of mapped debris inside the predicted runout zone, which covers {v['area']:.0f}% of the area"
                      + (f"; AUC {v['auc']:.2f} (susceptibility alone {v['auc_sus']:.2f})" if v.get("auc") is not None else ""),
                      "Only landslide cells outside the release areas are tested, i.e. where debris travelled or stopped."))
        if v.get("g_hit") is not None:
            vrows.append(("Runout on gentle land (< 15°)", f"{v['g_hit']:.0f}% of mapped debris on valley floors and fans inside the zone, "
                          f"which covers {v['g_area']:.0f}% of gentle land",
                          "Where people usually live; slope failure models alone do not reach here."))
        checks.append((1 if v["hit"] >= 2 * v["area"] and v["hit"] >= 50 else 0, 1,
                       f"The runout zone holds {v['hit']:.0f}% of mapped debris on {v['area']:.0f}% of the land."))
    hc = R.get("hindcast") or {}
    if hc.get("auc"):
        best = max(hc["auc"].values())
        checks.append((3 if best >= 0.85 else 2 if best >= 0.75 else 1 if best >= 0.65 else 0, 3,
                       f"Map built only from landslides seen before the event predicts the event's scars with AUC up to {best:.2f}."))
        vrows.append(("Hindcast of the event (pre-event data only)", ", ".join(f"{mlab.get(k, 'Ensemble' if k == 'ens' else k)} {v:.2f}" for k, v in hc["auc"].items())
                      + f"; {hc['evt_in_high']:.0f}% of event scars in High/Very high zones covering {hc['high_share']:.0f}% of the land",
                      "Models trained only on scars that appeared before the event year, then tested on the event itself."))
        if hc.get("debris_in_runout") is not None:
            vrows.append(("Hindcast runout", f"{hc['debris_in_runout']:.0f}% of the event's debris inside the runout zone predicted from pre-event "
                          f"data, which covers {hc['runout_share']:.0f}% of the land", "Release cells: old scars and very-high pre-event susceptibility."))
    dd = R.get("dated") or {}
    st_ = (R.get("rain") or {}).get("sat_threshold") or {}
    if st_.get("skill"):
        s = st_["skill"]
        vrows.append(("Rain in view intervals with new scars", f"AUC {s['auc']:.2f} ({s['n_pos']} intervals with scars, {s['n_neg']} without)",
                      "Do the gaps between satellite views that produced new scars hold more 3-day rain than gaps that did not? 0.5 = no link."))
    if R.get("fs_auc") is not None:
        vrows.append(("Physical model vs landslides", f"AUC {R['fs_auc']:.2f}", "Lower factor of safety where landslides are; no training involved."))
    if sus.get("only_ahp"):
        checks.append((0, 2, "No landslide inventory was available, so only the expert-weight (AHP) model and the physical model could be built."))
    # plausibility: automatic scars covering a large share of steep land are almost surely seasonal or logging artefacts
    steep = int((g.mask & (np.nan_to_num(R["F"]["slope"]) >= 10)).sum()) if "slope" in R["F"] else 0
    share = float(R["inv"].sum() / max(steep, 1) * 100) if R.get("inv") is not None else 0.0
    if (R.get("train_src") or "").startswith("automatic") and share > 3:
        checks.append((0, 3, f"Automatic scars cover {share:.1f}% of steep land; real landslides rarely exceed 1–3%, so many "
                             f"scars are probably logging, fire, seasonal or cloud artefacts."))
    verdict = grade(checks) if checks else dict(level="Low", reasons=["No validation was possible."], score=0)
    if (R.get("train_src") or "").startswith("automatic") and share > 3 and verdict.get("level") != "Low":
        verdict = dict(verdict, level="Low", reasons=list(verdict.get("reasons", [])) + [
            f"Capped at Low: automatic scars cover {share:.1f}% of steep land, far more than real landslides usually do, so the "
            f"inventory (and every accuracy computed from it) is probably inflated by plantation, logging or seasonal changes."])
    # an independent inventory is the strongest test: it caps the grade
    au = res.get("ens", {}).get("auc_user")
    if au is not None:
        cap = "Low" if au < 0.65 else "Medium" if au < 0.75 else None
        order = {"Low": 0, "Medium": 1, "High": 2}
        if cap and order[verdict["level"]] > order[cap]:
            verdict["level"] = cap
            verdict["reasons"].insert(0, f"Capped at {cap}: on your independent inventory the map scores AUC {au:.2f}.")
    if (R.get("train_src") or "").startswith("automatic") and share > 3 and verdict["level"] == "High":
        verdict["level"] = "Medium"

    # ------------------------------------------------------------ summary
    ttl = "Landslide study" + (f"; event of {R['ev']:%d %b %Y}" if R.get("ev") else "")
    rep = Report(kit.NAME, ttl, aoi.name, verdict, "#7f2704", aoi, g)
    smap = MapSpec(sus["cls"], g, "Landslide susceptibility", classes=SUSC, subtitle=aoi.name, source="This study")
    X = R.get("exp") or {}
    results = []
    inv_km2 = float(R["inv"].sum() * g.px_km2)
    results.append(("Landslide scars found by Sentinel-2", f"{fmt(inv_km2, 2)} km²"))
    if R.get("event"):
        results.append((f"New scars after {R['ev']:%d %b %Y}", f"{fmt(R['event']['map'].sum() * g.px_km2, 2)} km²"))
    hv = [r for r in X.get("by_class", []) if r["cls"] >= 4]
    if hv:
        results.append(("High + very high susceptibility", f"{fmt(sum(r['km2'] for r in hv))} km² ({sum(r['km2'] for r in hv) / max(aoi.area_km2, 1e-9) * 100:.0f}%)"))
        if hv[0]["pop"] is not None:
            results.append(("People living there", _n(sum(r["pop"] or 0 for r in hv))))
    if res.get("ens"):
        results.append(("Model accuracy (AUC, hold-out)", f"{res['ens']['auc_holdout']:.2f}"))
    if sus.get("cv"):
        results.append(("Spatial cross-validation (AUC)", " / ".join(f"{mlab[m]} {v[0]:.2f}" for m, v in sus["cv"].items())))
    if ro:
        results.append(("Runout zone (where debris can travel)", f"{fmt(ro['km2'], 2)} km²" + (f", {_n(ro['pop'])} people" if ro.get("pop") is not None else "")))
    us = (R["fs_cls"] <= 2) & g.mask
    results.append(("Least stable slopes when saturated (model)", f"{fmt(us.sum() * g.px_km2)} km²"))
    points = [f"Sentinel-2 found <b>{fmt(inv_km2, 2)} km²</b> of new bare scars on slopes" +
              (f" over {len(R['activity'])} years" if R["activity"] else "") + "."]
    if hv:
        points.append(f"<b>{fmt(sum(r['km2'] for r in hv))} km²</b> is highly or very highly susceptible"
                      + (f", home to about <b>{_n(sum(r['pop'] or 0 for r in hv))} people</b>." if hv[0]["pop"] is not None else "."))
    if res:
        best = max((k for k in res if k != "ens"), key=lambda k: res[k]["auc_holdout"])
        points.append(f"Of {len([k for k in res if k != 'ens'])} models, <b>{mlab[best]}</b> fitted best (AUC {res[best]['auc_holdout']:.2f}); "
                      f"the ensemble scored {res['ens']['auc_holdout']:.2f}.")
    if sus.get("rf_imp"):
        top = sorted(sus["rf_imp"].items(), key=lambda kv: -kv[1])[:3]
        points.append("Strongest factors: " + ", ".join(names[k][0].lower() for k, _ in top) + ".")
    if ro and ro.get("pop") is not None:
        points.append(f"Debris from these slopes can reach <b>{fmt(ro['km2'], 2)} km²</b> below them (up to {ro['max_reach_m']:,.0f} m), "
                      f"where about <b>{_n(ro['pop'])} people</b> live.")
    rain = R.get("rain") or {}
    if rain.get("rp") and rain["rp"].get(3) and rain["rp"][3].get("rp"):
        points.append(f"The 3-day rain before the event in the ~50 km NASA POWER cell was {rain['rp'][3]['value']:.0f} mm (a ~{rain['rp'][3]['rp']:.0f}-year value "
                      f"for that cell). Rain gauges on the slopes can record several times more, so this does not measure the local trigger.")
    rep.summary(results, points, smap)

    rep.study_area(MapSpec(F["elev"], g, "Study area: elevation", ramp="elev", vmin=float(np.nanpercentile(F["elev"], 2)),
                           vmax=float(np.nanpercentile(F["elev"], 98)), unit="m", subtitle=aoi.name, source="Copernicus DEM GLO-30"),
                   [("Mean slope", f"{np.nanmean(F['slope']):.1f}°"), ("Slopes steeper than 25°", f"{(np.nan_to_num(F['slope']) >= 25)[g.mask].mean() * 100:.0f}% of the area")])
    rows = [("Copernicus DEM GLO-30", "Slope, aspect, curvature, flow, TWI, SPI, streams, relief", "30 m", "2011–2015", "Copernicus DEM licence"),
            ("Sentinel-2 L2A", "Landslide scars (vegetation loss), NDVI factor", "10–20 m",
             f"{len(R['activity'])} yearly comparisons" + (", event before/after" if R.get("event") else ""), "Copernicus data"),
            ("IO/Esri land cover", "Land cover factor", "10 m", str(R["lc_year"]), "CC BY 4.0"),
            ("SoilGrids 2.0", "Clay and sand; soil strength", "250 m", "2020", "CC BY 4.0"),
            ("GEM Global Active Faults", "Distance to faults", "vector", "2019", "CC BY-SA 4.0"),
            ("OpenStreetMap", "Roads, schools, health facilities", "vector", "today", "ODbL 1.0"),
            ("ASF GLO-30 HAND", "Height above drainage", "30 m", "2021", "CC0"),
            ("NASA POWER", "Rainfall", "~50 km", "1981–today", "Public"),
            ("Macrostrat geologic maps", "Lithology factor", ", ".join(sorted((R.get("geol_info") or {}).get("scales", {}) or ["–"])) + " scale",
             "compiled maps", "CC BY 4.0"),
            ("GHS-POP, GHS-BUILT-S", "People and buildings at risk", "100 m", f"{X.get('pop_epoch', '')} / {X.get('ghsl_epoch', '')}", "CC BY 4.0")]
    if R.get("pga_info"):
        rows.append(("GEM Global Seismic Hazard Map v2023.1", "Earthquake shaking hazard (PGA 475-year)"
                     + ("" if "pga" in F else " (checked, not used)"), "~5.5 km", "2023", "CC BY-NC-SA 4.0 (non-commercial)"))
    if R.get("shakemap"):
        rows.append((f"USGS ShakeMap {R['shakemap']['id']}", "Event shaking (PGA)", "~1 km", R["shakemap"]["time"][:10], "Public domain"))
    if R.get("user") is not None:
        rows.append(("Your landslide inventory", "Independent validation" if R["train_src"] != "your inventory" else "Training and validation",
                     "vector", "", "yours"))
    rep.data(rows)
    rep.method([
        ("Automatic landslide inventory", "The greenest Sentinel-2 vegetation (maximum NDVI, one clear scene per month) of each year is compared "
         "with the year before; a sharp relative drop (rdNDVI ≤ −15, Scheip & Wegmann 2021) on a vegetated slope steeper than 10°, outside cropland and built-up land, marks a new "
         "bare scar. For an event, the season after it is compared with the same season a year earlier, so leaf fall and harvest are not "
         "mistaken for landslides."),
        ("Conditioning factors", f"{len(F)} factors from open data. Depressions in the DEM are filled (priority flood), flow is routed "
         "with D8, and TWI, SPI, curvatures and streams (≥ 0.5 km² catchment) are derived."),
        ("Models", "Bivariate statistics (Frequency Ratio, Information Value, Weights of Evidence, Index of Entropy), AHP with fixed "
         "expert weights, Logistic Regression, Random Forest, a Support Vector Machine (RBF kernel via random Fourier features, "
         "least-squares fit) and Gradient Boosting (150 depth-3 trees on binned factors, log-loss); the ensemble is their average. Absence points are taken more than "
         "100 m from any scar on slopes above 5°."),
        ("Validation", "Random 30% hold-out, spatial block cross-validation (1 km blocks), success-rate curves, a temporal test (old "
         "scars predict new ones) and, if you supply one, your own inventory as an independent test."),
        ("Physical stability", "Infinite-slope factor of safety for dry, half-saturated and saturated soil, with soil strength from "
         "SoilGrids texture and root cohesion from land cover. Soil depth varies with plan curvature around the chosen depth "
         "(×0.5 to ×1.5); on three test inventories this raised the physical model's AUC from 0.51 to 0.59 (Hiroshima), 0.59 to 0.61 "
         "(Gapyeong) and left Sikkim unchanged (0.71)."),
        ("Risk", "Susceptibility × people and buildings (GHS), roads and facilities (OpenStreetMap); hot spots with Getis-Ord Gi*."),
        ("Landslide dates from satellite", "Each scar is dated between the last Sentinel-2 view in which its slope was still green and the next clear "
         "view, provided it stays bare afterwards (seasonal leaf-off is ignored because green views follow it). Rain in each dated window gives "
         "empirical thresholds even without a dated inventory."),
        ("Runout", f"Flow-R approach (Horton et al. 2013): release cells are scars on slopes ≥ 15° and very-high susceptibility on slopes ≥ 30°; "
         f"debris spreads by Holmgren multiple flow direction (exponent 4) with Gamma persistence, and stops on an energy line with travel "
         f"angle {R['params'].get('runout_angle') or 11.0}° (velocity capped at 15 m/s)."),
    ])

    # ------------------------------------------------------------ inventory
    rep.heading("Landslides found by Sentinel-2")
    det = np.where(g.mask, np.where(R["inv"], 1, 2), 0).astype(np.uint8)
    rep.map(MapSpec(det, g, "Landslide scars detected", classes={1: ("Landslide scar", "#7f2704"), 2: ("Other land", "#ece9df")},
                    subtitle="New bare ground on vegetated slopes", source="Sentinel-2"), "Automatic landslide inventory.")
    if R.get("event"):
        e = R["event"]
        rep.text(f"Event of {R['ev']:%d %b %Y}: <b>{fmt(e['map'].sum() * g.px_km2, 3)} km²</b> of new scars. Images after the event: "
                 f"{', '.join(e['post_dates'][:6])}{'…' if len(e['post_dates']) > 6 else ''}; same season a year earlier: "
                 f"{', '.join(e['pre_dates'][:6])}{'…' if len(e['pre_dates']) > 6 else ''}.")
        if e.get("recovered_km2"):
            rep.text(f"{fmt(e['recovered_km2'], 2)} km² of first-detected event scars were dropped because they were green again a year later "
                     f"(pruned or harvested plantations recover; real landslide scars stay bare much longer).")
    elif R.get("event_err"):
        rep.text("The event could not be analysed: " + R["event_err"][:150])
    if R.get("recovered"):
        tot_ = sum(R["recovered"].values())
        rep.text(f"Recovery test: {fmt(tot_, 2)} km² of yearly scars were dropped because the ground was green again the following year "
                 f"(by year: " + ", ".join(f"{y} {fmt(v, 2)} km²" for y, v in sorted(R["recovered"].items())) + "). Tea pruning, plantation "
                 f"harvest and short-lived clearing look like landslides in one year but recover; real scars do not recover this fast.")
    if R["activity"]:
        ys = sorted(R["activity"])
        vals = [R["activity"][y].sum() * g.px_km2 for y in ys]
        rep.figure(render_grouped_bars([str(y) for y in ys], [("New scar area (km²)", vals, "#7f2704")], "km²"),
                   "New landslide scars found each year (each year compared with the one before).")
        mk = HY.mann_kendall(vals)
        if mk:
            rep.text(f"Trend in yearly landslide activity: {mk['slope']:+.3f} km²/year (Mann–Kendall p = {mk['p']:.2f}; "
                     f"{'significant' if mk['p'] < 0.05 else 'not significant with this short record'}).")
    if dd.get("windows"):
        W = dd["windows"]
        big = sorted(W, key=lambda w: -w["km2"])[:8]
        rep.text(f"<b>Dates.</b> {dd['n_px'] / max(dd['n_scar_px'], 1) * 100:.0f}% of yearly scar cells could be dated to the gap between two "
                 f"satellite views (median gap {np.median([w['days'] for w in W]):.0f} days). The year map uses these dates, so a slide in "
                 f"July is credited to its own year, not to the next yearly comparison.")
        rep.table(["Last green view", "First bare view", "Gap (days)", "Area km²"],
                  [[f"{w['start']:%d %b %Y}", f"{w['end']:%d %b %Y}", str(w["days"]), fmt(w["km2"], 3)] for w in big],
                  "Largest dated landslide windows (full list in landslide_dates_from_satellite.csv)")
    if R.get("user") is not None:
        rep.text(f"Your inventory: {R.get('user_n', 0)} landslides, {fmt(R['user'].sum() * g.px_km2, 3)} km² on the grid."
                 + (" It was kept out of training and used as an independent test." if R["train_src"] != "your inventory" else ""))
    rep.text("Automatic scars are candidates: road cuts, quarries, logging and fires also remove vegetation. Check them with the "
             "validation points before use in official mapping.")

    # ------------------------------------------------------------ factors
    rep.heading("Conditioning factors")
    gi_ = R.get("geol_info") or {}
    if "geol" in F:
        rep.text("Lithology comes from the Macrostrat compilation of geological maps (" + ", ".join(f"{k} scale: {v} points" for k, v in
                 gi_.get("scales", {}).items()) + "), grouped into broad rock classes; its frequency ratio is in the factor tables.")
    elif gi_.get("single") or gi_.get("coarse"):
        rep.text("Lithology was checked in the Macrostrat geological map compilation, but here only the 1:20 million world map covers the area "
                 "(one rock class), so it cannot separate safer from weaker ground and is left out.")
    pi_ = R.get("pga_info") or {}
    if "pga" in F:
        rep.text(f"Earthquake shaking hazard (GEM 2023 map, peak ground acceleration with a 10% chance of being exceeded in 50 years) "
                 f"ranges from {pi_.get('min', 0):.2f} to {pi_.get('max', 0):.2f} g here. It is used in the frequency-ratio tables, the "
                 "bivariate models and AHP, but not in the machine-learning models: at 5.5 km it is so smooth that those models would use it "
                 "to learn location rather than shaking, which makes their scores look better than they are.")
    elif pi_.get("dropped"):
        rep.text(f"Earthquake shaking hazard was checked (GEM 2023 map, {pi_.get('min', 0):.2f}–{pi_.get('max', 0):.2f} g) but left out: "
                 f"{pi_.get('reason', '')}, so it cannot separate safer from more dangerous slopes inside this area.")
    elif R.get("pga_err"):
        rep.text(f"The earthquake shaking map could not be read this time ({R['pga_err'][:120]}).")
    sm_ = R.get("shakemap")
    if sm_:
        if "pga_event" in F:
            rep.text(f"USGS ShakeMap of the M{sm_['mag']} earthquake {sm_['place']} ({sm_['time']} UTC): shaking here "
                     f"{sm_.get('min', 0):.2f}–{sm_.get('max', 0):.2f} g, used as an event factor (tables, bivariate models, AHP).")
        else:
            rep.text(f"USGS found the M{sm_['mag']} earthquake {sm_['place']} ({sm_['time']} UTC) near the event date, but its ShakeMap "
                     "barely varies inside this area, so it is not used.")
    for k, ramp in (("slope", "seq"), ("twi", "depth"), ("dist_stream", "seq"), ("ndvi", "ndvi")):
        if k in F:
            rep.map(MapSpec(F[k], g, f"Factor: {names[k][0]}", ramp=ramp, vmin=float(np.nanpercentile(F[k], 2)),
                            vmax=float(np.nanpercentile(F[k], 98)), unit=names[k][1], source="Derived"), f"{names[k][0]}.")
    rep.text("All factors (" + ", ".join(names[k][0].lower() for k in F) + ") are in the results folder as GeoTIFFs.")
    if sus.get("vif"):
        rep.table(["Factor", "VIF", "Multicollinearity"],
                  [[names[k][0], f"{v:.1f}", "none" if v < 5 else "moderate" if v < 10 else "high"] for k, v in sus["vif"].items()],
                  "Multicollinearity check (variance inflation factor; below 10 is acceptable)")
    if sus.get("tables"):
        st = sus["stats"]
        for k in sorted(st, key=lambda k: -st[k]["fr_range"])[:5]:
            rows = sus["tables"][k]
            lab = (lambda r: D.LULC.get(r["cls"], (str(r["cls"]),))[0]) if k == "lc" else (lambda r: r["label"])
            rep.table(["Class", "Area %", "Landslide %", "FR", "IV", "W+", "W−", "Contrast"],
                      [[lab(r), f"{r['npx'] / sum(x['npx'] for x in rows) * 100:.1f}", f"{r['nls'] / max(sum(x['nls'] for x in rows), 1) * 100:.1f}",
                        f"{r['fr']:.2f}", f"{r['iv']:+.2f}", f"{r['w_plus']:+.2f}", f"{r['w_minus']:+.2f}", f"{r['contrast']:+.2f}"] for r in rows],
                      f"Bivariate statistics: {names[k][0]} ({names[k][1]}). FR > 1 and positive contrast mean more landslides than average.")
        rep.table(["Factor", "Index of Entropy weight", "Information coefficient"],
                  [[names[k][0], f"{v['entropy_weight']:.3f}", f"{v['info']:.3f}"] for k, v in sorted(st.items(), key=lambda kv: -kv[1]["entropy_weight"])],
                  "Index of Entropy factor weights")

    # ------------------------------------------------------------ models
    rep.heading("Susceptibility models")
    w = sus["weights"]
    rep.table(["Factor", "AHP weight"], [[names[k][0], f"{w[k]:.3f}"] for k in sus["ahp_keys"]],
              f"AHP weights (λmax {sus['lam']:.2f}, consistency ratio {sus['cr']:.3f}; {'acceptable' if sus['cr'] < 0.1 else 'above 0.10'})")
    if sus.get("lr"):
        lr = sus["lr"]
        rep.table(["Term", "Coefficient (std.)", "Std. error", "p-value"],
                  [["Intercept", f"{lr['b'][0]:+.3f}", f"{lr['se'][0]:.3f}", f"{lr['p'][0]:.3g}"]] +
                  [[names[k][0], f"{lr['b'][i + 1]:+.3f}", f"{lr['se'][i + 1]:.3f}", f"{lr['p'][i + 1]:.3g}"] for i, k in enumerate(sus["keys"])],
                  "Logistic regression (categorical factors entered as their frequency ratio)")
        imp = sorted(sus["rf_imp"].items(), key=lambda kv: -kv[1])
        rep.bars([(names[k][0], v, "#238b45") for k, v in imp], "Random Forest factor importance (drop in AUC when scrambled).", "AUC drop")
    rep.map(smap, "Landslide susceptibility (ensemble of the models).")
    for k in ("fr", "woe", "ahp", "lr", "rf", "gbm"):
        if k in sus["maps"] and k != "ens":
            rep.map(MapSpec(sus["maps"][k], g, f"Susceptibility index: {mlab[k]}", ramp="risk", vmin=0, vmax=1, source="This study"), f"{mlab[k]}.")
    if res:
        rep.figure(_curve_png([(f"{mlab[k]} ({res[k]['auc_holdout']:.2f})", *res[k]["roc"], c) for k, _, c in MODELS if k in res],
                              "False positive rate", "True positive rate"), "ROC curves on the hold-out sample.")
        rep.figure(_curve_png([(f"{mlab[k]} ({res[k]['success'][2]:.2f})", res[k]["success"][0], res[k]["success"][1], c) for k, _, c in MODELS if k in res],
                              "Share of area, most susceptible first", "Share of landslides captured"),
                   "Success-rate curves: how many landslides fall in the most susceptible part of the area.")
        cols = ["Model", "AUC hold-out", "Success-rate AUC"] + (["AUC your inventory"] if any("auc_user" in v for v in res.values()) else [])
        rep.table(cols, [[mlab[k], f"{res[k]['auc_holdout']:.3f}", f"{res[k]['success'][2]:.3f}"] +
                         ([_n(res[k].get("auc_user"), 3)] if len(cols) == 4 else []) for k, _, _ in MODELS if k in res],
                  "Model performance (0.7–0.8 acceptable, 0.8–0.9 excellent, above 0.9 outstanding)")
    if sus.get("density"):
        dn = sus["density"]
        rep.table(["Class", "Share of area", "Share of landslides", "Landslide density"],
                  [[SUSC[r["cls"]][0], f"{r['area']:.1f}%", f"{r['flood']:.1f}%", f"{r['density']:.2f}"] for r in dn],
                  "Landslides in each susceptibility class (density should rise steadily)", [SUSC[r["cls"]][1] for r in dn])
    elif sus.get("only_ahp"):
        rep.text("No landslide inventory was available (no scars found and none supplied), so the map uses AHP expert weights only and "
                 "cannot be validated statistically. Add your inventory for a full study.")

    # ------------------------------------------------------------ stability
    rep.heading("Physical slope stability")
    rep.text(f"Infinite-slope model with {R['params'].get('soil_depth') or 1.5} m of soil. Soil strength comes from typical values for the "
             "SoilGrids texture, root strength from land cover; these are textbook values, so the map ranks slopes rather than giving design "
             "safety factors." + (f" Lower safety factor matches mapped landslides with AUC {R['fs_auc']:.2f}." if R.get("fs_auc") is not None else ""))
    rep.map(MapSpec(R["fs_cls"], g, "Slope stability (factor of safety)", classes=ST.CLASSES, source="This study"), "Slope stability classes.")
    rep.table(["Class", "Area km²"], [[ST.CLASSES[c][0], fmt((R["fs_cls"] == c).sum() * g.px_km2)] for c in ST.CLASSES],
              "Area by stability class", [ST.CLASSES[c][1] for c in ST.CLASSES])

    # ------------------------------------------------------------ rainfall
    if rain:
        rep.heading("Rainfall")
        if rain.get("window"):
            rw = rain["window"]
            rep.figure(render_grouped_bars([d.strftime("%d %b") if i % 5 == 0 else "" for i, (d, _) in enumerate(rw)],
                                           [("Daily rain (mm)", [v for _, v in rw], "#2f6fb0")], "mm"), "Daily rainfall before and after the event (NASA POWER).")
            t = rain["tot"]
            rep.text("Rain before the event: " + ", ".join(f"{k}-day <b>{t[k]:.0f} mm</b>" for k in (1, 3, 5, 7, 15, 30)) + ".")
            if rain.get("rp"):
                rep.table(["Rain total", "Event (mm)", "Return period (years)", "2 / 10 / 25 / 100-year (mm)", "Gumbel fit"],
                          [[f"{k}-day", f"{r['value']:.0f}", _n(r["rp"], 1), " / ".join(f"{r['q'][T]:.0f}" for T in (2, 10, 25, 100)),
                            f"KS {r['ks']:.2f} ({'fits' if r['ks'] < r['crit'] else 'poor'})"] for k, r in rain["rp"].items()],
                          "How rare was the triggering rain?")
        if rain.get("threshold"):
            th = rain["threshold"]
            rep.table(["Rain over", "Median before landslides (mm)", "5% threshold (mm)", "Days a year above it"],
                      [[f"{r['days']} days", f"{r['median']:.0f}", f"{r['thr5']:.0f}", f"{r['exceed'] * 3.65:.0f}"] for r in th["rows"]],
                      f"Empirical rainfall thresholds from {th['n']} dated landslides in your inventory (5% of landslides occurred below the threshold).")
        stt = rain.get("sat_threshold")
        if stt and stt.get("rows"):
            rep.table(["Rain over", "Median in dated windows (mm)", "5% threshold (mm)", "Days a year above it"],
                      [[f"{r['days']} days", f"{r['median']:.0f}", f"{r['thr5']:.0f}", f"{r['exceed'] * 3.65:.0f}"] for r in stt["rows"]],
                      f"Rainfall thresholds from {stt['n']} landslide windows dated by Sentinel-2 (wettest spell inside each window)")
            sk = stt.get("skill")
            rep.text("Dating is only to the gap between two views (median "
                     f"{stt['median_window_days']:.0f} days), so these thresholds are upper bounds of the triggering rain."
                     + (f" Gaps with new scars held more 3-day rain than gaps without (AUC {sk['auc']:.2f})." if sk and sk["auc"] >= 0.6 else
                        f" Rain does not clearly separate gaps with and without new scars here (AUC {sk['auc']:.2f}); treat the thresholds with care." if sk else ""))
        if rain.get("month"):
            act_m = [0] * 12
            rep.figure(render_grouped_bars(MON, [("Mean monthly rain (mm)", [rain["month"].get(m, 0) for m in range(1, 13)], "#2f6fb0")], "mm"),
                       "Rain season.")
        if rain.get("trend_max"):
            tm, tt = rain["trend_max"], rain["trend_total"]
            rep.text(f"Since 1981: yearly rainfall {tt['slope']:+.1f} mm/year (p = {tt['p']:.2f}); wettest day {tm['slope']:+.2f} mm/year (p = {tm['p']:.2f}).")

    # ------------------------------------------------------------ runout
    if ro:
        rep.heading("Where debris travels (runout)")
        rep.map(MapSpec(ro["cls"], g, "Landslide runout", classes=RO.CLASSES, subtitle=f"Travel angle {ro['angle']:.0f}°, Flow-R method",
                        source="This study"), "Release areas and the land debris can reach.")
        vv = np.where(np.isfinite(ro["v"]), ro["v"], np.nan)
        rep.map(MapSpec(vv, g, "Maximum debris velocity", ramp="risk", vmin=0, vmax=15, unit="m/s", source="This study"),
                "Highest velocity the debris can reach in each cell (energy line).")
        rows = [("Release area", f"{fmt(ro['src_km2'], 2)} km²"), ("Runout zone below it", f"{fmt(ro['km2'], 2)} km²"),
                ("Longest reach from a release cell", f"{ro['max_reach_m']:,.0f} m"),
                ("Runout land with velocity ≥ 5 m/s (destructive)", f"{fmt(ro['fast_km2'], 2)} km²")]
        if ro.get("pop") is not None:
            rows.append(("People in the runout zone", _n(ro["pop"])))
        if ro.get("built_km2") is not None:
            rows.append(("Building footprint in the runout zone", f"{fmt(ro['built_km2'], 3)} km²"))
        if ro.get("roads_km"):
            rows.append(("Major roads in the runout zone", f"{ro['roads_km'][1]:,.1f} km"))
        if ro.get("facilities") is not None:
            rows.append(("Schools / health facilities in the runout zone", str(len(ro["facilities"]))))
        rep.table(["Measure", "Value"], [list(r) for r in rows], "Runout summary")
        if ro.get("settlements"):
            rep.table(["#", "Location (lat, lon)", "People", "% in runout", "People in runout"],
                      [[i + 1, f"{r['lat']:.4f}, {r['lon']:.4f}", _n(r["people"]), f"{r['flooded_pct']:.0f}", _n(r["people_flooded"])]
                       for i, r in enumerate(ro["settlements"][:20])], "Settlements in the path of debris (full list in settlements_in_runout.csv)")
        rep.text("Susceptibility says where slopes fail; runout says where the debris goes, which is often where people live (valley floors, "
                 "fans). Travel angle 11° is a common regional value for debris flows; lower values (8–10°) give longer, more cautious reaches."
                 + (f" Against your inventory, {ro['val']['hit']:.0f}% of mapped debris lies inside the zone." if ro.get("val") else ""))
    hc = R.get("hindcast") or {}
    if hc and (hc.get("auc") or hc.get("err")):
        rep.heading("Hindcast: would a map made before the event have warned?")
        if hc.get("err"):
            rep.text(hc["err"])
        else:
            yy = hc["years"]
            vis = np.zeros(g.shape, np.uint8)
            hi = (hc["cls"] >= 4) & g.mask
            vis[hi] = 1
            vis[hc["reach"] & ~hi & g.mask] = 2
            evm = R["event"]["map"] & g.mask
            warned = hi | hc["reach"]
            vis[evm & warned] = 3
            vis[evm & ~warned] = 4
            HCL = {1: ("Pre-event High/Very high susceptibility", "#fdae61"), 2: ("Pre-event runout reach", "#fee08b"),
                   3: ("Event scar inside warned area", "#1a9850"), 4: ("Event scar outside warned area", "#d73027")}
            try:
                from ...core.raster import write_classes
                import os as _os
                files.append(write_classes(_os.path.join(out_dir, "hindcast_pre_event_warning.tif"), vis, g, HCL))
            except Exception:
                pass
            rep.map(MapSpec(vis, g, "Hindcast of the event", classes=HCL,
                            subtitle=f"Warning zones from landslides seen {yy[0]}–{yy[1]} only; event scars {R['ev']}", source="This study"),
                    "Orange and yellow: what a map made before the event would have flagged. Green: event scars that fell inside it. Red: missed.")
            rows = [("Landslides used to build the pre-event map", f"{hc['pre_km2']:.2f} km² of scars seen {yy[0]}–{yy[1]}"),
                    ("New event scars used to test it", f"{hc['evt_km2']:.2f} km²")]
            for k, v in hc["auc"].items():
                rows.append((f"AUC on event scars, {mlab.get(k, 'Ensemble' if k == 'ens' else k)}", f"{v:.2f}"))
            rows += [("Event scars in High/Very high zones", f"{hc['evt_in_high']:.0f}% (zones cover {hc['high_share']:.0f}% of the land)"),
                     ("Event scars in Very high zone", f"{hc['evt_in_vhigh']:.0f}% (zone covers {hc['vhigh_share']:.0f}% of the land)"),
                     ("Class at the event's highest scar (source)", SUSC.get(hc["crown_cls"], ("–",))[0]),
                     ("Highest event scar (lat, lon, elevation)", f"{hc['crown'][1]:.4f}, {hc['crown'][0]:.4f}, {hc['crown'][2]:,.0f} m"),
                     ("Longest debris travel seen from the source", f"{hc['evt_length_m']:,.0f} m")]
            if hc.get("debris_in_runout") is not None:
                rows.append(("Event debris inside the pre-event runout zone", f"{hc['debris_in_runout']:.0f}% (zone covers {hc['runout_share']:.0f}% of the land)"))
            if hc.get("g_debris_in_runout") is not None:
                rows.append(("…on gentle land (< 15°, where people live)", f"{hc['g_debris_in_runout']:.0f}% (zone covers {hc['g_runout_share']:.0f}% of gentle land)"))
            if hc.get("built_hit_km2") is not None:
                rows.append(("Buildings hit by the event (footprint)", f"{hc['built_hit_km2']:.3f} km²"))
                rows.append(("…of which inside the pre-event runout zone", f"{hc['built_hit_in_runout']:.0f}%"))
                rows.append(("…inside any pre-event warning zone", f"{hc['built_hit_in_high']:.0f}%"))
            if hc.get("vmax_track") is not None:
                rows.append(("Highest modelled debris speed on the event path", f"{hc['vmax_track']:.1f} m/s (model capped at 15 m/s)"))
            rep.table(["Measure", "Value"], [list(r) for r in rows], "Hindcast: pre-event map tested on the event")
            rep.text("This is the strictest test a susceptibility map can face: everything is rebuilt from data that existed before the event, "
                     "then compared with what actually happened. A good result means the method would have helped; it does not mean the timing "
                     "of the event could have been forecast, which needs rainfall warnings.")

    # ------------------------------------------------------------ exposure and risk
    if X:
        rep.heading("People, buildings and roads at risk")
        if X.get("built_fallback"):
            rep.text("<b>Note:</b> the JRC building server was unreachable; buildings come from the land cover map.")
        rep.table(["Susceptibility", "Area km²", "People", "Building footprint km²"],
                  [[SUSC[r["cls"]][0], fmt(r["km2"]), _n(r["pop"]), _n(r["built"], 3)] for r in X["by_class"]],
                  "Exposure by susceptibility class", [SUSC[r["cls"]][1] for r in X["by_class"]])
        if X.get("roads_km"):
            tot, hi = X["roads_km"]
            rep.text(f"<b>{hi:,.1f} km</b> of the {tot:,.1f} km of major roads cross high or very high susceptibility.")
        if X.get("facilities") is not None:
            hit = [f for f in X["facilities"] if f[4] >= 4]
            rep.text(f"{len(hit)} of {len(X['facilities'])} schools and health facilities stand on high or very high susceptibility.")
            if hit:
                rep.table(["Facility", "Type", "Location"], [[f[2] or "(no name)", f[3], f"{f[1]:.4f}, {f[0]:.4f}"] for f in hit[:25]],
                          "Facilities on highly susceptible ground (OpenStreetMap)")
        if X.get("settlements"):
            st = X["settlements"]
            rep.table(["#", "Location (lat, lon)", "Area km²", "People", "% high/very high", "People there"],
                      [[i + 1, f"{r['lat']:.4f}, {r['lon']:.4f}", fmt(r["area_km2"]), _n(r["people"]), f"{r['flooded_pct']:.0f}", _n(r["people_flooded"])]
                       for i, r in enumerate(st[:25])], f"Settlements on susceptible ground ({len(st)} of {X['n_settlements']}; full list in settlements_at_risk.csv)")
        if X.get("risk") is not None:
            rep.map(MapSpec(X["risk"], g, "Landslide risk (susceptibility × exposure)", classes=FX.RISK, source="This study"), "Landslide risk.")
            rep.table(["Risk", "Area km²", "People"], [[FX.RISK[r["cls"]][0], fmt(r["km2"]), _n(r["pop"])] for r in X["risk_rows"]],
                      "Area and people by risk class", [FX.RISK[r["cls"]][1] for r in X["risk_rows"]])
        if X.get("gi") is not None:
            rep.map(MapSpec(X["gi"], g, "Landslide hot spots (Gi*)", classes=FX.GISTAR, source="This study"),
                    f"Clusters of landslide scars ({X['gi_cell']:.0f} m cells, 95% and 99% confidence).")
    # ------------------------------------------------------------ hazard over time
    HZr = R.get("hazard")
    if HZr and HZr.get("rows"):
        from .hazard import HAZ
        rep.heading("Landslide hazard over time")
        rep.text(f"From {HZr['n_years']} years of satellite scans ({HZr['years'][0]}–{HZr['years'][-1]}), about {HZr['per_year']:.1f} new "
                 "landslides per year were seen in this area. Counting them in each susceptibility class gives a yearly rate per km² and, "
                 "assuming landslides arrive at random (Poisson), the chance of at least one landslide in a 1 km² area in 1, 10 and 50 years. "
                 "The record is short and misses small slides, so read these as minimum chances.")
        rep.table(["Susceptibility", "Area km²", "Landslides seen", "Per km² per year", "1 year", "10 years", "50 years"],
                  [[SUSC[r["cls"]][0], fmt(r["area"]), str(r["n"]), f"{r['rate']:.3f}", f"{r['p'][1] * 100:.0f}%", f"{r['p'][10] * 100:.0f}%",
                    f"{r['p'][50] * 100:.0f}%"] for r in HZr["rows"]],
                  "Chance of at least one landslide in a 1 km² area", [SUSC[r["cls"]][1] for r in HZr["rows"]])
        rep.map(MapSpec(HZr["map"], g, "Landslide hazard: chance in 50 years", classes=HAZ, source="Sentinel-2 activity + this study"),
                "Susceptibility classes coloured by the chance of at least one landslide per km² in 50 years.")
        n_rows = [r for r in HZr["rows"] if r["n"] > 0]
        if len(n_rows) >= 2:
            rates = [r["rate"] for r in sorted(HZr["rows"], key=lambda r: r["cls"])]
            ok = all(b >= a * 0.8 for a, b in zip(rates, rates[1:]))
            vrows.append(("Hazard rises with susceptibility", "yes" if ok else "not steadily",
                          "Landslides per km² per year should grow from very low to very high susceptibility."))
    elif R.get("hazard_err"):
        rep.text(f"Hazard over time could not be computed: {R['hazard_err']}")

    # ------------------------------------------------------------ guidance
    if R.get("guidance"):
        rep.heading("What to do (general guidance)")
        rep.text("These points follow from the maps above. They are general good practice for landslide-prone areas, not engineering "
                 "advice; decisions on a particular site need a geotechnical investigation.")
        rep.bullets(R["guidance"])

    if R.get("zones"):
        rep.heading("Landslide hazard by zone")
        rep.table(["Zone", "Area km²", "High/very high %", "Scars km²", "People in high/very high"],
                  [[z["name"], fmt(z["area"]), f"{z['high']:.0f}", fmt(z["ls"], 3), _n(z["pop_high"])] for z in R["zones"][:40]],
                  "Statistics by zone")

    vrows.append(("Your own check", "50 random points in validation_points.geojson", "Look at them on imagery and fill the 'correct' column."))
    rep.validation(vrows)
    rep.limits([
        "Sentinel-2 starts in 2015–2017 and needs cloud-free views; old, small (< ~400 m²) or revegetated landslides are missed.",
        "Vegetation loss is not always a landslide: road cuts, quarries, logging, fire and landslides in bare land are possible confusions.",
        "Lithology comes from Macrostrat: detailed (1:50k–1:1M) in the USA, Japan, parts of Europe and Australia, but only 1:20 million "
        "elsewhere, where it is left out.",
        "Earthquake shaking hazard (GEM 2023) is a ~5.5 km map, free for non-commercial use only; it is left out of small areas and of "
        "the machine-learning models. Set 'Earthquake shaking factor' off for commercial work.",
        "Hazard over time rests on only a few years of satellite scans and assumes landslides arrive at random; it gives minimum chances.",
        "The DEM is 30 m and a surface model; very small slopes and road cuts are not resolved.",
        "Soil strength and depth in the stability model are textbook values, not field tests.",
        "People and buildings are 100 m grids: estimates for planning, not counts.",
        "Slow-moving landslides (mm–cm per year) need radar interferometry (InSAR); that needs Sentinel-1 SLC processing or an "
        "account-based service (e.g. EGMS for Europe), so it is not part of this one-click study.",
        "Runout uses one travel angle for the whole area and a 30 m DEM; it shows where debris can go, not depth or impact pressure. "
        "It is strongest for debris from scars already seen by the satellite; from susceptible slopes alone it is a rougher guide.",
        "Satellite dating is only to the gap between two clear views (weeks to months in cloudy seasons).",
    ])
    rep.conclusions(points + [f"Overall confidence: <b>{verdict['level']}</b>."])
    rep.references([
        DT.CREDIT_S2, D.CREDIT_DEM, D.CREDIT_HAND, D.CREDIT_LULC, CX.CREDIT_GEM, D.CREDIT_OSM, HY.CREDIT_POWER,
        SM.CREDIT_GEM, SM.CREDIT_SHAKEMAP,
        "Scheip, C. M., Wegmann, K. W. (2021). HazMapper: a global open-source natural hazard mapping application in Google Earth Engine. NHESS 21, 1495–1511.",
        "Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal depression-filling algorithm. Computers & Geosciences 62, 117–127.",
        "Beven, K., Kirkby, M. (1979). A physically based, variable contributing area model of basin hydrology. Hydrol. Sci. Bull. 24, 43–69.",
        "Lee, S., Pradhan, B. (2007). Landslide hazard mapping at Selangor, Malaysia using frequency ratio and logistic regression. Landslides 4, 33–41.",
        "Bonham-Carter, G. F. (1994). Geographic Information Systems for Geoscientists. Pergamon (weights of evidence).",
        "Yin, K. L., Yan, T. Z. (1988). Statistical prediction models for slope instability of metamorphosed rocks (information value).",
        "Bednarik, M. et al. (2010). Landslide hazard and risk assessment: a case study (index of entropy). Natural Hazards 53, 199–214.",
        "Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.",
        "Breiman, L. (2001). Random Forests. Machine Learning 45, 5–32.",
        "Rahimi, A., Recht, B. (2007). Random features for large-scale kernel machines. NIPS 20.",
        "Suykens, J. A. K., Vandewalle, J. (1999). Least squares support vector machine classifiers. Neural Processing Letters 9, 293–300.",
        "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29, 1189–1232.",
        "Guzzetti, F. et al. (2005). Probabilistic landslide hazard assessment at the basin scale. Geomorphology 72, 272–299.",
        "Peters, S. E., Husson, J. M., Czaplewski, J. (2018). Macrostrat: a platform for geological data integration. G-cubed 19, 1393–1409.",
        "Chung, C.-J., Fabbri, A. G. (2003). Validation of spatial prediction models for landslide hazard mapping. Natural Hazards 30, 451–472.",
        "Pack, R. T., Tarboton, D. G., Goodwin, C. N. (1998). The SINMAP approach to terrain stability mapping.",
        "Brenning, A. (2005). Spatial prediction models for landslide hazards: review, comparison and evaluation. NHESS 5, 853–862.",
        "Horton, P., Jaboyedoff, M., Rudaz, B., Zimmermann, M. (2013). Flow-R, a model for susceptibility mapping of debris flows and other gravitational hazards at a regional scale. NHESS 13, 869–885.",
        "Holmgren, P. (1994). Multiple flow direction algorithms for runoff modelling in grid based elevation models. Hydrological Processes 8, 327–334.",
        "Gamma, P. (2000). dfwalk – Ein Murgang-Simulationsprogramm zur Gefahrenzonierung. Geographica Bernensia G66.",
        "Corominas, J. (1996). The angle of reach as a mobility index for small and large landslides. Canadian Geotechnical Journal 33, 260–271.",
        "Getis, A., Ord, J. K. (1992). The analysis of spatial association. Geographical Analysis 24, 189–206.",
    ])
    rep.proof(kit.sources(), kit.reproduce_command())
    rep.files(files + ["report.html", "report.docx", "report.pdf"])
    docs = rep.write(out_dir, kit.seconds())
    kit.step(100, "Done")
    return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers,
                summary=f"{fmt(inv_km2, 2)} km² of landslide scars; susceptibility built with {len(res) or 1} models.")
