# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Report chapter for the AreaGIS Kit: Health."""

import numpy as np

from ...core import datasets as D
from ...core.aoi import Grid
from ...core.maps import MapSpec
from ...core.report import Report, fmt, grade


def _n(v, d=1):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt(v, d)


from . import stats as ST


def build(kit, R, aoi, out_dir, files, layers):
    from ..healthstudy import ENV, HOT, LISA, RATE5
    I, u, per = R["I"], np.flatnonzero(R["use"]), R["per"]
    dis = R["disease"]
    ids = [I["ids"][k] for k in u]
    perlab = f"per {int(per):,}"
    z, gp, hcls, gfdr = R["gi"]
    Ii, lp, lcls = R["lisa"]
    mi, mp = R["moran"]
    sig_scan = [c for c in R["scan"] if c["p"] <= 0.05]

    # ------------------------------------------------------------ verdict
    checks, vrows = [], []
    nu, nt = len(u), len(I["ids"])
    checks.append((2 if nu == nt and not I["unmatched"] else 1 if nu / nt >= 0.9 else 0, 2,
                   f"{nu} of {nt} areas have usable cases and population" + (f"; {I['unmatched']} had no matching CSV row." if I["unmatched"] else ".")))
    vrows.append(("Data completeness", f"{nu} of {nt} areas", "Areas without cases or population are left out, never filled in."))
    if mi is not None:
        checks.append((2 if mp <= 0.05 else 1, 2, f"Moran's I = {mi:.2f} (p = {mp:.3f}): "
                       + ("cases cluster in space beyond chance." if mp <= 0.05 else "no clear clustering beyond chance; read hot spots with caution.")))
        vrows.append(("Clustering beyond chance (999 shuffles)", f"Moran's I {mi:.2f}, p = {mp:.3f}", "p ≤ 0.05: similar rates really sit together."))
    kmin = int(R["params"].get("suppress_below") or 0)
    supp = np.array([kmin > 0 and 0 < R["C"][u[i]] < kmin for i in range(len(u))], bool)
    def cs(i):
        return f"< {kmin}" if supp[i] else fmt(R["C"][u[i]], 0)
    def rw(i, txt):
        return "hidden (few cases)" if supp[i] else txt
    if kmin > 0:
        vrows.append(("Privacy", f"{int(supp.sum())} areas with fewer than {kmin} cases",
                      "Their counts and crude rates are hidden in tables and exported files; smoothed estimates are shown."))
    hh = set(np.flatnonzero(lcls == 1))
    hot = set(np.flatnonzero(hcls <= 2))
    scan_m = set(np.flatnonzero(R["scan_member"] > 0))
    if hot:
        agree = len(hot & (hh | scan_m)) / len(hot) * 100
        checks.append((2 if agree >= 70 else 1 if agree >= 40 else 0, 2,
                       f"{agree:.0f}% of Gi* hot spots are also LISA high-high or scan-cluster areas: three methods agree."))
        vrows.append(("Agreement of three cluster methods", f"{agree:.0f}% of Gi* hot spots confirmed by LISA or the scan statistic",
                      "Hot spots found by different mathematics are robust."))
    if R.get("robust") is not None:
        vrows.append(("Independent of smoothing", f"{R['robust']:.0f}% of hot spots also found with raw rates", "Hot spots that do not depend on the smoothing method."))
    h, fa, T, k = R["power"]
    checks.append((2 if h / T >= 0.8 and fa / T <= 0.1 else 1 if h / T >= 0.5 else 0, 2,
                   f"On your own map, a planted cluster of {k} areas with double risk was found in {h} of {T} trials; "
                   f"false clusters appeared in {fa} of {T} trials with no cluster."))
    vrows.append(("Planted-cluster test on your map", f"found {h}/{T}; false alarms {fa}/{T}",
                  f"Shows whether a real cluster of about {k} areas with double risk could be detected with these populations at all."))
    if R.get("resid_moran") and R["resid_moran"][0] is not None:
        ri, rp = R["resid_moran"]
        vrows.append(("Spatial pattern left after the drivers", f"Moran's I of residuals {ri:.2f} (p = {rp:.3f})",
                      "If still significant, unmeasured local factors (or reporting) also matter."))
    if R.get("pop_est") is not None and not R.get("pop_from_meta"):
        pu = R["P"][u]
        pe = R["pop_est"][u]
        ok = (pe > 0) & np.isfinite(pu)
        if ok.sum() >= 5:
            r = float(np.corrcoef(np.log1p(pu[ok]), np.log1p(pe[ok]))[0, 1])
            ratio = float(pe[ok].sum() / pu[ok].sum())
            checks.append((1 if r >= 0.8 else 0, 1, f"Your population agrees with Meta's 30 m population map (r = {r:.2f} on log scale)."))
            vrows.append(("Your population vs Meta 30 m population", f"r = {r:.2f}; Meta total ×{ratio:.2f} of yours",
                          "A check for typing errors or mismatched area IDs. A total far from ×1 is expected when your denominator is a subgroup (births, children, women)."))
    B = R.get("bym")
    if B:
        ok = B["rhat_max"] <= 1.1
        checks.append((1 if ok else 0, 1, f"Bayesian model (BYM2) converged: largest split R-hat {B['rhat_max']:.3f}"
                       + (" (≤ 1.1)." if ok else " (> 1.1: treat its results as indicative).")))
        vrows.append(("Bayesian model convergence (BYM2)", f"split R-hat ≤ {B['rhat_max']:.3f}; {B['rhat_rr_share_ok'] * 100:.0f}% of areas ≤ 1.1",
                      f"{B['chains']} MCMC chains × {B['iters']:,} iterations; values near 1.00 mean the chains agree."))
        r = float(np.corrcoef(B["rr"], R["ebg"])[0, 1])
        vrows.append(("Bayesian vs empirical Bayes rates", f"r = {r:.2f}", "Two different smoothing methods; high agreement means the map does not hinge on one method."))
    A = R.get("age")
    if A:
        vrows.append(("Age standardisation", f"crude vs age-standardised rate r = {A['r_crude_vs_asr']:.2f}" if A.get("r_crude_vs_asr") is not None else "–",
                      "Low agreement means age structure differs a lot between areas, so crude rates mislead."))
        if A["covered"] < 0.999:
            vrows.append(("Age range covered", f"{A['covered'] * 100:.0f}% of the WHO standard population",
                          "Your groups do not cover all ages; the standard is rescaled to the ages you gave."))
    PTc = R.get("pts")
    if PTc and PTc.get("cls") is not None:
        share_ = float((PTc["cls"] == 1).sum() / max((PTc["cls"] > 0).sum(), 1) * 100)
        vrows.append(("Risk surface plausibility", f"{share_:.0f}% of the area flagged significantly high",
                      "Under 25% is plausible; much more means the cases do not follow the population layer."))
        checks.append((1 if share_ <= 25 else 0, 1, f"{share_:.0f}% of the area is flagged as significantly high on the case-point risk surface."))
    if R.get("gwpr"):
        GWc = R["gwpr"]
        vrows.append(("Local vs global regression", f"AICc {GWc['aicc']:.1f} vs {GWc['global_aicc']:.1f} ({GWc['nn']} neighbours)",
                      "AICc lower by more than 3 means the effects really vary across the map; otherwise trust the global model."))
    verdict = grade(checks)

    # ------------------------------------------------------------ maps
    g = Grid(aoi, 50, max_pixels=1_500_000)
    gl = [I["gl"][k] for k in u]

    def amap(vals):
        return g.rasterize(gl, values=[int(v) for v in vals]).astype(np.uint8)
    hot_map = MapSpec(amap(hcls), g, f"{dis}: hot and cold spots (Gi*)", classes=HOT, subtitle=f"{aoi.name} · {nu} areas", source="Your data")
    q = np.percentile(R["ebg"], [20, 40, 60, 80])
    rate_map = MapSpec(amap(1 + np.digitize(R["ebg"], q)), g, f"{dis}: smoothed rate {perlab} (quintiles)",
                       classes={k + 1: (f"{'≤' if k < 4 else '>'} {fmt(q[min(k, 3)], 1)}", RATE5[k]) for k in range(5)}, source="Your data")

    # ------------------------------------------------------------ summary
    rep = Report(kit.NAME, f"{dis}: small-area study", aoi.name, verdict, "#9e1b16", aoi, None)
    results = [("Overall rate", f"{fmt(R['base'], 2)} {perlab}"), ("Areas analysed", str(nu)),
               ("Gi* hot spots (95%+)", f"{int((hcls <= 2).sum())} ({int(gfdr.sum())} after FDR)"),
               ("Scan clusters (p ≤ 0.05)", str(len(sig_scan)))]
    if mi is not None:
        results.append(("Clustering (Moran's I)", f"{mi:.2f}, p = {mp:.3f}"))
    if B:
        results.append(("Areas with P(risk > average) > 95% (BYM2)", str(int((B["p_gt1"] > 0.95).sum()))))
    if A:
        results.append(("Age-standardised rate (WHO standard)", f"{fmt(A['overall'][0], 2)} {perlab}"))
    if R.get("st_scan") is not None:
        results.append(("Space-time clusters (p ≤ 0.05)", str(sum(1 for c_ in R["st_scan"] if c_["p"] <= 0.05))))
    if R.get("pop_far") is not None:
        results.append((f"People > {float(R['params'].get('access_km') or 5):g} km from a facility", f"{fmt(R['pop_far'], 0)} ({R['pop_far_pct']:.0f}%)"))
    pts = [f"Overall {dis.lower()} rate <b>{fmt(R['base'], 2)} {perlab}</b> ({int(R['C'][u].sum()):,} cases, {int(np.nansum(R['P'][u])):,} people"
           + (", population estimated from Meta's 30 m map" if R.get("pop_from_meta") else "") + ")."]
    if sig_scan:
        c0 = sig_scan[0]
        pts.append(f"The scan statistic finds a cluster of <b>{len(c0['members'])} areas</b> with <b>{c0['rr']:.2f}×</b> the risk of the rest "
                   f"({c0['obs']:.0f} cases where {c0['exp']:.1f} were expected, p = {c0['p']:.3f}).")
    if A:
        pts.append(f"Age-standardised rate (WHO World Standard Population): <b>{fmt(A['overall'][0], 2)} {perlab}</b> "
                   f"(95% CI {fmt(A['overall'][1], 2)}–{fmt(A['overall'][2], 2)}); SMR, smoothing and clusters use age-adjusted expected cases.")
    if B:
        pts.append(f"The Bayesian BYM2 model gives <b>{int((B['p_gt1'] > 0.95).sum())}</b> areas a probability above 95% of higher-than-average risk; "
                   f"{B['phi'] * 100:.0f}% of the area-to-area variation is spatially structured.")
    sig_st = [c_ for c_ in (R.get("st_scan") or []) if c_["p"] <= 0.05]
    if sig_st:
        c1 = sig_st[0]
        ks = R["trend_keys"]
        pts.append(f"The space-time scan finds <b>{len(c1['members'])} areas</b> in <b>{ks[c1['t0']]}–{ks[c1['t1']]}</b> with "
                   f"<b>{c1['rr']:.2f}×</b> the risk expected for those periods (p = {c1['p']:.3f}).")
    pts.append(f"<b>{int((hcls <= 2).sum())}</b> areas are Gi* hot spots and <b>{int((lcls == 1).sum())}</b> are LISA high-high clusters.")
    sig_d = [(k, v) for k, v in (R.get("drivers") or {}).items() if v.get("rho") is not None]
    if R.get("glm"):
        sig = [r for r in R["glm"]["rows"][1:] if r[4] < 0.05]
        lab = dict(ENV)
        if sig:
            pts.append("Drivers (Poisson regression, per standard deviation): " + "; ".join(
                f"<b>{lab.get(nm, nm)}</b> ×{rr:.2f}" for nm, rr, a, b, pv in sig) + ".")
        else:
            pts.append("None of the satellite-measured factors explains the rates significantly once the others are accounted for.")
    if R.get("trend_mk"):
        t = R["trend_mk"]
        pts.append(f"Over {len(R['trend_keys'])} periods the overall rate {'rose' if t['slope'] > 0 else 'fell'} "
                   f"({'significant' if t['p'] < 0.05 else 'not significant'}, p = {t['p']:.2f}); {R['n_rising']} areas rise significantly.")
    if R.get("emerging") is not None:
        pts.append(f"<b>{int(R['emerging'].sum())}</b> emerging hot spots (hot in the latest period, not earlier) and "
                   f"<b>{int(R['persistent'].sum())}</b> persistent ones.")
    rep.summary(results, pts, hot_map)
    rep.study_area(rate_map, [("Areas in the layer", str(nt)), ("Areas with usable data", str(nu)),
                              ("Total cases", f"{int(R['C'][u].sum()):,}"), ("Total population", f"{int(np.nansum(R['P'][u])):,}")])
    rows = [("Your area boundaries", "Units of analysis, neighbours", "polygons", "as supplied", "yours"),
            ("Your case counts" + (" and population" if not R.get("pop_from_meta") else ""), "Rates", "per area", "as supplied", "yours")]
    if R.get("pop_grid") is not None:
        rows.append(("Meta 30 m population (HRSL)", "Population check, density, population-weighted access" + (", population" if R.get("pop_from_meta") else ""),
                     "30 m", "~2020", "CC BY 4.0"))
    if R.get("env"):
        rows += [("ESA WorldCover 2021", "Built-up and water share", "10 m", "2021", "CC BY 4.0"),
                 ("Copernicus DEM", "Elevation", "30 m", "2021", "Copernicus"),
                 ("MODIS NDVI / LST", "Greenness, day surface temperature", "250 m / 1 km", "last full year", "NASA, public")]
    if R.get("fac"):
        rows.append(("Health facilities", "Access", "points", "as supplied" if R["fac_src"] != "OpenStreetMap" else "today", R["fac_src"]))
    if R.get("pts"):
        rows.append(("Your case points", "Density, risk surface" + (", Knox test" if R["pts"].get("knox") else ""), "points", "as supplied", "yours"))
    rep.data(rows)
    rep.method([
        ("Rates", f"Cases ÷ population × {int(per):,}, with exact-Poisson 95% intervals (Byar). Expected cases = overall rate × population; "
         "SMR = observed ÷ expected; probability of excess = chance of seeing at least this many cases if the area had the overall rate."),
        ("Smoothing", "Global empirical Bayes pulls unreliable small-area rates towards the overall rate; spatial empirical Bayes pulls them "
         "towards the neighbourhood rate (Marshall 1991)."),
        ("Age standardisation", "With age groups: direct age-standardised rates weighted by the WHO World Standard Population 2000–2025 "
         "(Ahmad 2001; groups of any width get the summed 5-year weights) with Fay–Feuer gamma 95% intervals, and indirect "
         "standardisation: expected cases from the study's own age-specific rates, so SMR, smoothing, clusters, the scan and the "
         "regression all compare like with like."),
        ("Bayesian disease mapping", "BYM2 model (Riebler 2016): cases ~ Poisson(expected × relative risk), log risk = intercept + "
         "σ(√(1−φ)·unstructured + √φ·scaled ICAR spatial effect on the neighbour graph). PC prior on σ (P(σ > 1) = 0.01), uniform φ. "
         "Metropolis-within-Gibbs MCMC, 2 chains; posterior relative risk, 95% credible interval, probability of risk above average, "
         "and the spatial share φ. Convergence by split R-hat."),
        ("Space-time scan", "With 3 or more periods: Kulldorff's space-time scan (Poisson): cylinders of circles (up to 50% of the "
         "population) × time windows (up to 50% of the periods), expected cases adjusted for each period's overall rate, Monte Carlo p-values."),
        ("Bayesian model with covariates", "When the environment is read: the same BYM2 model with the standardised satellite factors as "
         "covariates (log scale, N(0,1) priors per SD); rate ratio per SD with 95% credible intervals, and how much of the area-to-area "
         "variation the factors explain."),
        ("Space-time permutation scan", "With 3 or more periods: Kulldorff's space-time permutation model (2005) on the cases only: expected "
         "= area total × period total ÷ grand total; Monte Carlo by shuffling the period of each case."),
        ("Neighbours", "Areas sharing a border or corner (queen contiguity); areas without neighbours are linked to their two nearest areas."),
        ("Clusters", "Global Moran's I and local Moran (LISA, Anselin 1995) with 999 conditional permutations; Getis-Ord Gi* with false "
         "discovery rate control; Kulldorff's circular spatial scan statistic (Poisson, up to 50% of the population, Monte Carlo p-values)."),
        ("Drivers", "For every area, satellites give population density, built-up share, elevation, greenness, day surface temperature, "
         "surface water and distance to care. Spearman correlation, Geodetector q (Wang 2010) and Poisson regression with a population "
         "offset (quasi-Poisson errors; factors correlated above 0.8 dropped) measure their link with the rates."),
        ("Trends", "With periods: rate per period, Mann–Kendall trend overall and per area, and emerging hot spots (Gi* hot in the latest "
         "period but not in the first half)."),
        ("Power test", "On your own boundaries and populations, a cluster of double risk is planted 20 times and searched for with the scan "
         "statistic; 20 data sets with no cluster measure false alarms."),
        ("Local drivers (GWPR)", "Geographically weighted Poisson regression (Nakaya et al. 2005) with the same standardised factors and "
         "offset: an adaptive bisquare kernel over the k nearest areas, k chosen by AICc; local rate ratios, local t-values and a comparison "
         "with the global model by AICc."),
        ("Knox test", "Space-time interaction (Knox 1964): pairs of cases close in space and in time, against Monte Carlo shuffles of the "
         "time labels that keep each place's and each period's totals. For areas: same or neighbouring area and same or next period."),
        ("Case points", "If case locations are given: Gaussian kernel density (bandwidth by leave-one-out likelihood cross-validation over fractions of Scott's rule) of cases and of population; their "
         "ratio is the relative-risk surface (1 = average), with tolerance contours from 99 simulations where cases follow the population "
         "(Kelsall & Diggle 1995)."),
        ("Early warning and forecast", "log E[cases] = log population + area effect + common trend per period, fitted by maximum "
         "likelihood with quasi-Poisson over-dispersion. Alert: fit without the latest period, flag areas whose latest count exceeds the "
         "upper 99% prediction bound and is at least 3 (in the spirit of Farrington et al. 1996). Forecast: refit on all periods, next "
         "period with a 95% prediction interval. Needs at least 3 periods."),
        ("2SFCA access", "Two-step floating catchment area (Luo & Wang 2003) with a Gaussian distance decay inside the chosen distance: each "
         "facility's capacity (or 1 per facility) is shared among the population that can reach it, then summed for every place; mapped "
         "per area as a population-weighted mean per 10,000 people."),
    ])

    # ------------------------------------------------------------ chapters
    rep.heading("Rates and excess risk")
    rep.map(rate_map, f"Smoothed {dis.lower()} rate by area.")
    top = np.argsort(-R["ebg"])[:15]
    rep.table(["Area", "Cases", "Expected", "SMR", f"Rate {perlab} (95% CI)", "Smoothed", "P(excess by chance)"],
              [[ids[i], cs(i), fmt(R["exp"][i], 1), rw(i, f"{R['smr'][i]:.2f}"),
                rw(i, f"{fmt(R['raw'][i], 1)} ({fmt(R['lo'][i], 1)}–{fmt(R['hi'][i], 1)})"), fmt(R["ebg"][i], 1),
                rw(i, f"{R['pex'][i]:.3f}")] for i in top], "Areas with the highest smoothed rates")
    if kmin > 0 and supp.any():
        rep.text(f"Privacy rule: {int(supp.sum())} areas have fewer than {kmin} cases. Their counts and crude rates are hidden here and in the "
                 f"exported files, because small numbers in small areas can identify individual patients. Smoothed rates are shown instead.")
    rep.heading("Clusters and hot spots")
    rep.map(hot_map, "Getis-Ord Gi* on smoothed rates.")
    rep.map(MapSpec(amap(lcls), g, f"{dis}: local Moran (LISA)", classes=LISA, source="Your data"),
            "High-high clusters, low-low clusters and outliers (p ≤ 0.05).")
    if R["scan"]:
        sm = np.zeros(nu, int)
        for k2, cl in enumerate(R["scan"][:5]):
            sm[cl["members"]] = k2 + 1 if cl["p"] <= 0.05 else 0
        cols = ["#9e1b16", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"]
        cls = {k2 + 1: (f"Cluster {k2 + 1} (RR {cl['rr']:.2f}, p {cl['p']:.3f})", cols[k2]) for k2, cl in enumerate(R["scan"][:5]) if cl["p"] <= 0.05}
        if cls:
            cls[0] = ("Not in a significant cluster", "#ece9df")
            rep.map(MapSpec(amap(sm), g, f"{dis}: Kulldorff scan clusters", classes=cls, source="Your data"),
                    f"Most likely and secondary clusters (Monte Carlo, {R['scan_sims']} replications).")
        rep.table(["Cluster", "Areas", "Observed", "Expected", "Relative risk", "LLR", "p"],
                  [[str(k2 + 1), str(len(cl["members"])), fmt(cl["obs"], 0), fmt(cl["exp"], 1), f"{cl['rr']:.2f}", f"{cl['llr']:.1f}",
                    f"{cl['p']:.3f}"] for k2, cl in enumerate(R["scan"][:5])], "Spatial scan statistic (Kulldorff 1997)")
    if A:
        rep.heading("Age-standardised rates")
        qa = np.percentile(A["asr"], [20, 40, 60, 80])
        rep.map(MapSpec(amap(1 + np.digitize(A["asr"], qa)), g, f"{dis}: age-standardised rate {perlab} (WHO standard)",
                        classes={k + 1: (f"{'≤' if k < 4 else '>'} {fmt(qa[min(k, 3)], 1)}", RATE5[k]) for k in range(5)}, source="Your data"),
                "Direct age-standardised rate by area (quintiles).")
        rep.table(["Age group", "Cases", "Population", f"Rate {perlab}", "WHO weight"],
                  [[lab, fmt(cc, 0), fmt(pp, 0), fmt(cc / pp * per if pp else float("nan"), 2), f"{ww:.3f}"]
                   for (lab, _, _), cc, pp, ww in zip(A["groups"], A["crude_by_group"], A["pop_by_group"], A["w"])],
                  "Cases, population and rate by age group")
        top_a = np.argsort(-A["asr"])[:15]
        rep.table(["Area", "Crude rate", "Age-standardised (95% CI)", "Age-adjusted SMR (95% CI)"],
                  [[ids[i], rw(i, fmt(R["raw"][i], 1)), rw(i, f"{fmt(A['asr'][i], 1)} ({fmt(A['lo'][i], 1)}–{fmt(A['hi'][i], 1)})"),
                    rw(i, f"{A['smr'][i]:.2f} ({A['smr_lo'][i]:.2f}–{A['smr_hi'][i]:.2f})")] for i in top_a],
                  "Areas with the highest age-standardised rates")
    if B:
        rep.heading("Bayesian disease mapping (BYM2)")
        rrc = np.digitize(B["rr"], [0.67, 0.83, 0.95, 1.05, 1.2, 1.5])
        RR7 = {1: ("< 0.67", "#2c7bb6"), 2: ("0.67–0.83", "#74add1"), 3: ("0.83–0.95", "#c6dbef"), 4: ("0.95–1.05", "#f2efe6"),
               5: ("1.05–1.2", "#fdae61"), 6: ("1.2–1.5", "#e6550d"), 7: ("> 1.5", "#a63603")}
        rep.map(MapSpec(amap(rrc + 1), g, f"{dis}: relative risk (BYM2 posterior mean)", classes=RR7, source="Your data"),
                "Relative risk against the study average after Bayesian smoothing with spatial and unstructured effects.")
        pc = np.digitize(B["p_gt1"], [0.05, 0.2, 0.8, 0.95])
        PR = {1: ("< 5%: clearly lower", "#2c7bb6"), 2: ("5–20%", "#abd9e9"), 3: ("20–80%: unclear", "#f2efe6"),
              4: ("80–95%", "#fdae61"), 5: ("> 95%: clearly higher", "#a63603")}
        rep.map(MapSpec(amap(pc + 1), g, f"{dis}: probability that risk is above average", classes=PR, source="Your data"),
                "Posterior probability P(relative risk > 1). Areas above 95% are the most reliable high-risk areas.")
        top_b = np.argsort(-B["p_gt1"] - 1e-6 * B["rr"])[:15]
        rep.table(["Area", "Cases", "Expected", "Relative risk (95% credible interval)", "P(risk > average)"],
                  [[ids[i], cs(i), fmt(R["exp"][i], 1), f"{B['rr'][i]:.2f} ({B['lo'][i]:.2f}–{B['hi'][i]:.2f})",
                    f"{B['p_gt1'][i]:.2f}"] for i in top_b], "Areas most likely to have higher-than-average risk (BYM2)")
        rep.text(f"Spatial share of the variation φ = {B['phi']:.2f} (95% CrI {B['phi_ci'][0]:.2f}–{B['phi_ci'][1]:.2f}); "
                 f"σ = {B['sigma']:.2f} ({B['sigma_ci'][0]:.2f}–{B['sigma_ci'][1]:.2f}). {B['draws']:,} posterior draws, "
                 f"largest split R-hat {B['rhat_max']:.3f}.")
    elif R.get("bym_err"):
        rep.text(f"The Bayesian model could not be fitted: {R['bym_err']}")
    BC = R.get("bym_cov")
    if BC and BC.get("beta"):
        lab_ = dict(ENV)
        bt = BC["beta"]
        rep.heading("Bayesian model with environmental factors")
        rep.table(["Factor", "Rate ratio per SD (posterior mean)", "95% credible interval", "P(effect > 0)", "R-hat"],
                  [[lab_.get(nm, nm), f"{bt['irr'][j]:.2f}", f"{bt['irr_lo'][j]:.2f}–{bt['irr_hi'][j]:.2f}", f"{bt['p_pos'][j]:.2f}",
                    f"{bt['rhat'][j]:.3f}"] for j, nm in enumerate(BC["names"])],
                  "BYM2 with the satellite factors as covariates (standardised)")
        rep.text(f"With the factors in the model the remaining area-to-area variation σ falls from {R['bym']['sigma']:.2f} to "
                 f"{BC['sigma']:.2f} ({BC['sigma_drop'] * 100:.0f}% less); φ = {BC['phi']:.2f}. A credible interval that excludes 1 "
                 "means the factor is associated with the rate after allowing for spatial clustering, which the plain regression does not.")
    elif R.get("bym_cov_err"):
        rep.text(f"The Bayesian model with covariates could not be fitted: {R['bym_cov_err']}")
    if R.get("st_scan") is not None:
        rep.heading("Space-time clusters")
        ks = R["trend_keys"]
        if R["st_scan"]:
            stm = np.zeros(nu, int)
            cols = ["#9e1b16", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"]
            stc = {}
            for k2, cl in enumerate(R["st_scan"][:5]):
                if cl["p"] <= 0.05:
                    stm[cl["members"]] = k2 + 1
                    stc[k2 + 1] = (f"Cluster {k2 + 1}: {ks[cl['t0']]}–{ks[cl['t1']]} (RR {cl['rr']:.2f})", cols[k2])
            if stc:
                stc[0] = ("Not in a significant cluster", "#ece9df")
                rep.map(MapSpec(amap(stm), g, f"{dis}: space-time clusters", classes=stc, source="Your data"),
                        f"Areas and periods with more cases than each period's rate predicts (Monte Carlo, {R['st_sims']} replications).")
            rep.table(["Cluster", "Areas", "Periods", "Observed", "Expected", "Relative risk", "p"],
                      [[str(k2 + 1), str(len(cl["members"])), f"{ks[cl['t0']]}–{ks[cl['t1']]}", fmt(cl["obs"], 0), fmt(cl["exp"], 1),
                        f"{cl['rr']:.2f}", f"{cl['p']:.3f}"] for k2, cl in enumerate(R["st_scan"][:5])],
                      "Space-time scan statistic (Kulldorff 1998, Poisson, time-trend adjusted)")
        if R.get("st_note"):
            rep.text(R["st_note"])
        SP = R.get("stp_scan")
        if SP:
            rep.table(["Cluster", "Areas", "Periods", "Observed", "Expected", "Ratio", "p"],
                      [[str(k2 + 1), str(len(cl["members"])), f"{ks[cl['t0']]}–{ks[cl['t1']]}", fmt(cl["obs"], 0), fmt(cl["exp"], 1),
                        f"{cl['rr']:.2f}", f"{cl['p']:.3f}"] for k2, cl in enumerate(SP[:3])],
                      "Space-time permutation scan (Kulldorff 2005, cases only)")
            both = [cl for cl in SP if cl["p"] <= 0.05]
            strong = [cl for cl in both if cl["rr"] >= 1.2]
            msg = ("The permutation scan uses the cases alone (no population), so it cannot be fooled by wrong or outdated "
                   "population figures. " + (f"It finds {len(both)} significant space-time cluster(s), {len(strong)} with a ratio of 1.2 or more."
                                              if both else "It finds no significant space-time cluster."))
            if len(both) > len(strong):
                msg += (" Clusters with a ratio below 1.2 are statistically significant only because the data are large; they are "
                        "small shifts, often from populations that grew or aged differently over time (this scan uses crude counts, "
                        "not age-adjusted ones). Trust the Poisson space-time scan above for size, this one for timing.")
            rep.text(msg)
    if R.get("env"):
        rep.heading("Environment and drivers")
        E, lab = R["env"], dict(ENV)
        t = []
        for k2, l2 in ENV:
            if k2 in R.get("drivers", {}):
                dv = R["drivers"][k2]
                t.append([l2, _n(float(np.nanmedian(E[k2][u])), 1), _n(dv["rho"], 2), _n(dv["q"], 2)])
        if t:
            rep.table(["Factor", "Median over areas", "Spearman ρ with rate", "Geodetector q"], t, "How each factor relates to the rate")
        if R.get("glm"):
            f = R["glm"]
            rep.table(["Factor", "Rate ratio per SD", "95% CI", "p"],
                      [[lab.get(nm, nm), f"{rr:.2f}", f"{a:.2f}–{b:.2f}", f"{pv:.3f}"] for nm, rr, a, b, pv in f["rows"][1:]],
                      f"Poisson regression (pseudo-R² {f['r2']:.2f}, dispersion {f['dispersion']:.1f})" if f.get("r2") is not None else "Poisson regression")
            if R.get("dropped"):
                rep.text("Left out because they move together with another factor (|r| ≥ 0.8): " + ", ".join(lab.get(d, d) for d in R["dropped"]) + ".")
    GW = R.get("gwpr")
    if GW:
        lab = dict(ENV)
        rep.heading("Local drivers: geographically weighted Poisson regression")
        better = GW["aicc"] < GW["global_aicc"] - 3
        rows_ = []
        for j, nm in enumerate(GW["names"]):
            b = GW["beta"][:, j + 1]
            tv = b / np.maximum(GW["se"][:, j + 1], 1e-12)
            rows_.append([lab.get(nm, nm), f"{np.exp(np.percentile(b, 10)):.2f}", f"{np.exp(np.median(b)):.2f}", f"{np.exp(np.percentile(b, 90)):.2f}",
                          f"{np.exp(GW['global_beta'][j + 1]):.2f}", f"{(tv >= 1.96).mean() * 100:.0f}% / {(tv <= -1.96).mean() * 100:.0f}%"])
        rep.table(["Factor", "Local rate ratio per SD: 10th pct", "Median", "90th pct", "Global model", "Areas significantly + / −"], rows_,
                  f"GWPR with {GW['nn']} nearest areas (adaptive bisquare, chosen by AICc): AICc {GW['aicc']:.1f} vs global {GW['global_aicc']:.1f}")
        rep.text((f"The local model fits clearly better than one global model (AICc lower by {GW['global_aicc'] - GW['aicc']:.1f}), so the "
                  "strength of at least one factor changes across the map." if better else
                  "The local model does not fit clearly better than one global model (AICc difference below 3): the factors act about the "
                  "same everywhere, and the global regression above is the one to use.") +
                 (f" Deviance explained: local {GW['pseudo_r2'] * 100:.0f}%, global {GW['pseudo_r2_global'] * 100:.0f}%." if GW.get("pseudo_r2") is not None else ""))
        if better:
            spread = [(np.ptp(GW["beta"][:, j + 1]), j) for j in range(len(GW["names"]))]
            for _, j in sorted(spread, reverse=True)[:2]:
                b = np.exp(GW["beta"][:, j + 1])
                qb = [0.8, 0.95, 1.05, 1.25]
                cls_ = {1: ("< 0.80", "#2c7bb6"), 2: ("0.80–0.95", "#abd9e9"), 3: ("0.95–1.05", "#f2efe6"), 4: ("1.05–1.25", "#fdae61"), 5: ("> 1.25", "#d7191c")}
                rep.map(MapSpec(amap(1 + np.digitize(b, qb)), g, f"Local effect of {lab.get(GW['names'][j], GW['names'][j]).lower()}",
                                classes=cls_, source="Your data; satellites"), "Local rate ratio per standard deviation (GWPR).")
    if R.get("trend_keys"):
        rep.heading("Trends")
        rep.line([(f"{dis} {perlab}", R["trend_total"], "#9e1b16")], [str(k) for k in R["trend_keys"]], "Overall rate per period.", perlab)
        if R.get("emerging") is not None:
            em = np.where(R["emerging"], 1, np.where(R["persistent"], 2, 3))
            rep.map(MapSpec(amap(em), g, f"{dis}: emerging hot spots", classes={1: ("Emerging (new)", "#d7301f"), 2: ("Persistent", "#7f0000"),
                                                                                     3: ("Not hot now", "#ece9df")}, source="Your data"),
                    "Hot in the latest period: new (not hot in the first half of the periods) or persistent.")
        KA = R.get("knox_areal")
        if KA:
            rep.text(f"<b>Knox test</b> (space-time interaction): pairs of cases in the same or neighbouring areas and in the same or next "
                     f"period: {KA['obs']:,.0f} observed vs {KA['exp']:,.0f} expected if timing and place were independent "
                     f"(ratio {KA['ratio']:.3f}, Monte Carlo p = {KA['p']:.3f}). " +
                     ("Cases bunch together in space <i>and</i> time more than chance: a sign of outbreaks or local, short-lived causes."
                      if KA["p"] <= 0.05 else "No sign that cases bunch in space and time together beyond what the overall pattern explains.") +
                     (" The excess is small (under 5%), so with this many cases even a slight shift is significant; the space-time scan "
                      "shows where and when." if (KA["p"] <= 0.05 and (KA["ratio"] or 1) < 1.05) else ""))
    EW, FCn = R.get("ew"), R.get("fc")
    if EW or FCn:
        rep.heading("Early warning and forecast")
        ks_ = R.get("trend_keys") or []
        if EW:
            na = int(EW["alert"].sum())
            rep.map(MapSpec(amap(np.where(EW["alert"], 1, 2)), g, f"{dis}: alert in the latest period ({ks_[-1] if ks_ else ''})",
                            classes={1: ("More cases than expected (alert)", "#d7301f"), 2: ("Within expectation", "#ece9df")},
                            source="Your data"), "Areas whose latest count is above the upper 99% prediction bound from earlier periods.")
            top = np.argsort(-(EW["observed"] - EW["upper"]))[:10]
            rep.text(f"A trend model fitted on the earlier periods predicts each area's count for the latest period. <b>{na}</b> of "
                     f"{len(ids)} areas are above the upper 99% bound (and have at least 3 cases): worth a field check. "
                     f"Common trend per period ×{EW['trend']:.2f}; over-dispersion {EW['phi']:.2f}.")
            rows_ = [[ids[i], (f"< {kmin}" if kmin > 0 and 0 < EW['observed'][i] < kmin else f"{EW['observed'][i]:.0f}"),
                      f"{EW['expected'][i]:.1f}", f"{EW['upper'][i]:.1f}", "ALERT" if EW["alert"][i] else "–"]
                     for i in top if EW["observed"][i] > EW["expected"][i]]
            if rows_:
                rep.table(["Area", "Latest observed", "Expected", "Upper 99% bound", "Flag"], rows_,
                          "Areas furthest above expectation in the latest period (full list in forecast_and_alerts.csv)")
        if FCn:
            tot_m, tot_lo, tot_hi = float(FCn["mean"].sum()), float(FCn["lo"].sum()), float(FCn["hi"].sum())
            lo_, hi_ = FCn["trend_ci"]
            rep.text(f"<b>Next period (forecast):</b> about {fmt(tot_m, 0)} cases in all areas (sum of area 95% intervals "
                     f"{fmt(tot_lo, 0)}–{fmt(tot_hi, 0)}), if populations stay as in the latest period and the trend continues "
                     f"(×{FCn['trend']:.2f} per period, 95% CI ×{lo_:.2f}–×{hi_:.2f}). A short series gives a rough forecast only.")
    if R.get("fac"):
        rep.heading("Access to care")
        E = R.get("env") or {}
        if "dist_fac" in E:
            far = np.argsort(-np.nan_to_num(E["dist_fac"][u]))[:10]
            rep.table(["Area", "Population-weighted km to facility", "Smoothed rate", "Gi*"],
                      [[ids[i], _n(E["dist_fac"][u[i]]), fmt(R["ebg"][i], 1), HOT[int(hcls[i])][0]] for i in far],
                      f"Farthest areas from care ({len(R['fac'])} facilities, {R['fac_src']}; distance {R.get('dist_how', 'straight line')})")
            if R.get("dist_how") == "along roads" and R.get("detour"):
                rep.text(f"Distances follow OpenStreetMap roads; on average they are {R['detour']:.2f}× the straight-line distance.")
            elif R.get("roads_err"):
                rep.text("Distances are straight lines: roads could not be read from OpenStreetMap this time. Road travel is usually 1.2–1.6× longer.")
    elif R.get("fac_err"):
        rep.text(f"Access was not measured: {R['fac_err']}")

    PT = R.get("pts")
    if PT:
        rep.heading("Case locations: density and risk surface")
        kg = PT["grid"]
        dens = PT["density"]
        if np.isfinite(dens).any():
            rep.map(MapSpec(dens, kg, f"{dis}: case density", ramp="heat", vmin=0, vmax=float(np.nanpercentile(dens, 99)) or 1.0,
                            unit="cases/km²", source="Your case points"), f"Kernel density of {PT['inside']:,} cases (bandwidth {PT['bw_m'] / 1000:.1f} km by likelihood cross-validation; Scott's rule gave {PT['bw_scott'] / 1000:.1f} km).")
        if PT.get("cls") is not None:
            KC = {1: ("High risk, significant (p ≤ 0.025, ≥ 1.25×)", "#b2182b"), 2: ("Higher than average (≥ 1.25×)", "#f4a582"),
                  3: ("About average", "#f2efe6"), 4: ("Lower than average (≤ 0.8×)", "#92c5de")}
            rep.map(MapSpec(PT["cls"], kg, f"{dis}: risk surface (cases vs population)", classes=KC, source="Your case points; Meta population"),
                    "Ratio of the case density to the population density, scaled so 1 = average; significance from 99 simulations in which "
                    "cases follow the population (Kelsall & Diggle 1995).")
            hi_km2 = float((PT["cls"] == 1).sum() * kg.px_km2)
            share = float((PT["cls"] == 1).sum() / max((PT["cls"] > 0).sum(), 1) * 100)
            msg = f"{hi_km2:,.1f} km² ({share:.0f}% of the area) carry significantly more cases than their population explains."
            if share > 25:
                msg += (" That is a very large share: the case points probably do not follow the Meta population map (for example cases "
                        "geocoded to village or ward centres, or a population very different from Meta's). Read the map as broad "
                        "contrasts, not as exact hot spots.")
            rep.text(msg)

        if PT.get("knox"):
            K = PT["knox"]
            rep.text(f"<b>Knox test on the case points</b> ({K['n']:,} dated cases): {K['obs']:,} pairs are within "
                     f"{float(R['params'].get('knox_km') or 2):g} km and {float(R['params'].get('knox_days') or 14):g} days of each other, "
                     f"against {K['exp']:,.1f} expected (ratio {K['ratio']:.2f}, Monte Carlo p = {K['p']:.3f}). " +
                     ("Close-in-space cases are also close in time: a sign of transmission or a short-lived local source."
                      if K["p"] <= 0.05 else "No space-time interaction beyond chance."))
    elif R.get("pts_err"):
        rep.text(f"The case points were not analysed: {R['pts_err']}")
    SF = R.get("sfca")
    if SF:
        acc = SF["acc"][u]
        unit = (f"{SF['capf']} per 10,000 people" if SF["used_cap"] else "facilities per 10,000 people")
        qa = np.nanpercentile(acc, [20, 40, 60, 80]) if np.isfinite(acc).any() else [0, 0, 0, 0]
        AC5 = ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]
        rep.heading("Access to care: two-step floating catchment (2SFCA)")
        rep.map(MapSpec(amap(1 + np.digitize(np.nan_to_num(acc), qa)), g, f"Access to care ({unit})",
                        classes={k + 1: (f"{'≤' if k < 4 else '>'} {qa[min(k, 3)]:.2f}", AC5[k]) for k in range(5)}, source=R.get("fac_src") or ""),
                f"2SFCA index within {SF['d0']:g} km (Gaussian distance decay): supply shared by everyone who can reach it.")
        low = np.argsort(np.nan_to_num(acc, nan=np.inf))[:10]
        rep.table(["Area", f"Access ({unit})", "Smoothed rate", "Gi*"],
                  [[ids[i], _n(acc[i], 2), fmt(R["ebg"][i], 1), HOT[int(hcls[i])][0]] for i in low], "Areas with the poorest access")
        if np.isfinite(acc).sum() >= 5:
            rho = ST.spearman(acc, R["ebg"])
            if rho is not None:
                rep.text(f"Access and the smoothed rate correlate at Spearman ρ = {rho:+.2f} across areas.")
    rep.validation(vrows)
    rep.limits([
        "Results depend on the boundaries chosen (modifiable areal unit problem) and describe areas, not individuals (ecological fallacy).",
        "Reported cases reflect testing and reporting effort as well as disease.",
        "Satellite factors describe the place, not the people's exposure; associations are not proof of cause.",
        "The scan statistic searches circles; long, thin clusters along rivers or roads are found only in parts.",
        "The forecast assumes the recent trend and populations continue; with few periods it is rough, and seasonal patterns are not "
        "modelled. In simulations the alert flagged 2.5% of areas without an outbreak and caught 68% of doublings and 95% of triplings.",
        "Distance to care and the 2SFCA catchments follow OpenStreetMap roads when they can be read (distance, not travel time; "
        "areas up to 5,000 km², up to 80 facilities for 2SFCA), otherwise straight lines.",
        "GWPR local coefficients are noisy with few areas; trust maps of local effects only when AICc clearly favours the local model.",
        "The Bayesian model's spatial share φ is often weakly determined with few areas; its credible interval shows how weakly.",
        "Without age groups, rates and SMR are crude: areas with older (or younger) populations can look worse (or better) only because of age.",
        "The space-time scan needs 3 or more periods; with fewer, only the purely spatial scan and per-period rates are given.",
        "The space-time permutation scan uses crude case counts; with age groups, differences in how populations aged or grew over "
        "time can show as weak (ratio < 1.2) space-time clusters.",
    ])
    rep.conclusions(pts + [f"Overall confidence: <b>{verdict['level']}</b>."])
    rep.references(["Farrington, C. P., Andrews, N. J., Beale, A. D., Catchpole, M. A. (1996). A statistical algorithm for the early detection of outbreaks of infectious disease. JRSS A 159, 547–563.",
                    
        "Kulldorff, M. (1997). A spatial scan statistic. Communications in Statistics – Theory and Methods 26, 1481–1496.",
        "Anselin, L. (1995). Local indicators of spatial association: LISA. Geographical Analysis 27, 93–115.",
        "Getis, A., Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. Geographical Analysis 24, 189–206.",
        "Marshall, R. J. (1991). Mapping disease and mortality rates using empirical Bayes estimators. Applied Statistics 40, 283–294.",
        "Breslow, N. E., Day, N. E. (1987). Statistical Methods in Cancer Research II (Byar's approximation).",
        "Wang, J. F. et al. (2010). Geographical detectors-based health risk assessment. IJGIS 24, 107–127.",
        "Riebler, A. et al. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. Stat. Methods Med. Res. 25, 1145–1165.",
        "Besag, J., York, J., Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. AISM 43, 1–59.",
        "Kulldorff, M. et al. (2005). A space-time permutation scan statistic for disease outbreak detection. PLoS Medicine 2(3), e59.",
        "Kulldorff, M. et al. (1998). Evaluating cluster alarms: a space-time scan statistic and brain cancer in Los Alamos. AJPH 88, 1377–1380.",
        "Fay, M. P., Feuer, E. J. (1997). Confidence intervals for directly standardized rates: a method based on the gamma distribution. Stat. Med. 16, 791–801.",
        "Ahmad, O. B. et al. (2001). Age standardization of rates: a new WHO standard. GPE Discussion Paper 31, WHO.",
        "Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B 57, 289–300.",
        "Nakaya, T. et al. (2005). Geographically weighted Poisson regression for disease association mapping. Stat. Med. 24, 2695–2717.",
        "Knox, E. G. (1964). The detection of space-time interactions. Applied Statistics 13, 25–30.",
        "Kelsall, J. E., Diggle, P. J. (1995). Kernel estimation of relative risk. Bernoulli 1, 3–16.",
        "Luo, W., Wang, F. (2003). Measures of spatial accessibility to health care in a GIS environment. Environ. Plann. B 30, 865–884.",
        "Meta / CIESIN High Resolution Settlement Layer (CC BY 4.0).", D.CREDIT_WC, D.CREDIT_DEM, D.CREDIT_OSM,
    ])
    rep.proof(kit.sources(), kit.reproduce_command())
    rep.files(files + ["report.html", "report.docx", "report.pdf"])
    docs = rep.write(out_dir, kit.seconds())
    kit.step(100, "Done")
    return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers,
                summary=f"{int((hcls <= 2).sum())} hot spot areas, {len(sig_scan)} scan clusters among {nu} areas.")
