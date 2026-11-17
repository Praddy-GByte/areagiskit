# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Report chapter for the AreaGIS Kit: Urban."""

import numpy as np

from ...core import datasets as D
from ...core.maps import MapSpec, render_multi_line
from ...core.report import Report, fmt, grade
from ..flood import extra as FX
from . import city as CT
from . import heat as HT


def _n(v, d=1):
    return "–" if v is None else fmt(v, d)


def _mk(mk, unit="°C/decade"):
    if not mk:
        return "not enough years"
    s = mk.get("sen", mk.get("slope", 0.0)) * 10
    return f"{s:+.2f} {unit} ({'significant' if mk['p'] < 0.05 else 'not significant'}, p = {mk['p']:.2f})"


from . import climate as CM
from . import surface as SF


def build(kit, R, aoi, out_dir, files, layers):
    from ..urbanstudy import EXPO, GROWTH, HVI, UTFVI, ZONES
    g, lst = R["g"], R["lst"]
    year, months = R["year"], R["months"]
    mlab = ",".join(map(str, months))

    # ------------------------------------------------------------ verdict
    checks, vrows = [], []
    nd = len(R["ldates"])
    checks.append((2 if nd >= 4 else 1 if nd >= 2 else 0, 2, f"Heat map from {nd} clear Landsat days in the hot season."))
    tl = R["lst_lc"]
    if 1 in tl and 7 in tl:
        ok = tl[1] < tl[7]
        checks.append((1 if ok else 0, 1, "Water is cooler than built-up land, as physics requires." if ok else
                       "Water is not cooler than built-up land: the heat map may hold cloud or mixed-pixel errors."))
        vrows.append(("Physics check", f"water {tl[1]:.1f} °C vs built-up {tl[7]:.1f} °C", "Open water must be cooler than roofs and roads by day."))
    if R.get("suhi") is not None and R.get("rural") is not None:
        ua, ra = float(R["urban"].sum()), float(R["rural"].sum())
        ok = ra >= 0.5 * ua
        checks.append((2 if ok else 0, 2, "The countryside ring is large enough to compare with the core." if ok else
                       f"The countryside ring is only {ra / max(ua, 1) * 100:.0f}% of the core's area: draw the study area wider around the city "
                       "for a reliable heat island."))
        vrows.append(("Countryside reference", f"{fmt(ra * g.px_km2)} km² vs core {fmt(ua * g.px_km2)} km²",
                      "Should be about the same area as the core (Peng et al. 2012)."))
    if R.get("canopy_check"):
        a, b, c = R["canopy_check"]
        vrows.append(("Tree canopy: Sentinel-2 vs ESA WorldCover trees", f"{a:.0f}% vs {b:.0f}% of the area; {c:.0f}% of WorldCover trees found",
                      "Independent 10 m tree map (2021)."))
        if R.get("canopy_block"):
            rb, mae = R["canopy_block"]
            vrows.append(("Tree canopy per 100 m block vs WorldCover", f"r = {rb:.2f}; mean difference {mae:.0f} percentage points",
                          "Fairer than pixel overlap for scattered trees; WorldCover is from 2021."))
            checks.append((1 if (rb >= 0.6 and abs(a - b) <= 8) else 0, 1,
                           f"Canopy per 100 m block agrees with ESA WorldCover at r = {rb:.2f}; total canopy {a:.0f}% vs {b:.0f}%."))
        else:
            checks.append((1 if c >= 60 else 0, 1, f"{c:.0f}% of ESA WorldCover tree cover is also canopy in our Sentinel-2 map."))
    if R.get("modis_cmp"):
        c = R["modis_cmp"]
        r = float(np.mean([x["r"] for x in c]))
        b = float(np.mean([x["bias"] for x in c]))
        u = float(np.mean([x["ubrmse"] for x in c]))
        checks.append((3 if r >= 0.8 and abs(b) <= 3 else 2 if r >= 0.65 else 1 if r >= 0.5 else 0, 3,
                       f"Landsat agrees with MODIS on the same mornings: r = {r:.2f}, bias {b:+.1f} °C ({len(c)} days, 1 km)."))
        vrows.append(("MODIS Terra, same day and hour (independent satellite)",
                      f"r = {r:.2f}; bias {b:+.1f} °C; spread {u:.1f} °C; {sum(x['n'] for x in c)} cells, {len(c)} days",
                      "Landsat averaged to 1 km and compared with MOD11A1 for the same morning; a few °C bias is normal between sensors."))
    else:
        checks.append((0, 3, "No same-day MODIS image to cross-check Landsat."))
    if R.get("suhi") is not None and (R.get("suhi_modis") or {}).get("Day") is not None:
        a, m = R["suhi"], R["suhi_modis"]["Day"]
        same = np.sign(a) == np.sign(m) or abs(a - m) < 1.0
        checks.append((2 if same and abs(a - m) <= 2 else 1 if same else 0, 2,
                       f"Heat island by Landsat {a:+.1f} °C vs MODIS daytime {m:+.1f} °C (season mean, 1 km)."))
        vrows.append(("Heat island, two satellites", f"Landsat {a:+.1f} °C · MODIS day {m:+.1f} °C",
                      "Same urban core and countryside ring; different sensors, times and resolution."))
    if R.get("built_agree"):
        ua, pa = R["built_agree"]
        checks.append((2 if ua >= 70 else 1 if ua >= 50 else 0, 2, f"{ua:.0f}% of built-up land (Esri) is also built-up in ESA WorldCover 2021."))
        vrows.append(("Built-up: Esri vs ESA WorldCover 2021", f"{ua:.0f}% of Esri built-up agrees; {pa:.0f}% of WorldCover built-up agrees",
                      "Two independent 10 m land cover maps."))
    if R.get("pop_check"):
        a, b, ep = R["pop_check"]
        d = abs(a - b) / max(b, 1) * 100
        checks.append((2 if d <= 20 else 1 if d <= 40 else 0, 2, f"Population: Meta {fmt(a, 0)} vs GHS-POP {ep} {fmt(b, 0)} ({d:.0f}% apart)."))
        vrows.append(("Population: Meta HRSL vs GHS-POP", f"{fmt(a, 0)} vs {fmt(b, 0)} ({d:.0f}% apart)",
                      "Two independent population maps (buildings vs census disaggregation)."))
    if R.get("no2_city") is not None:
        ratio = R["no2_city"] / max(R["no2_bg"], 1e-6)
        vrows.append(("NO₂ plausibility", f"city {R['no2_city']:.0f} vs countryside {R['no2_bg']:.0f} µmol/m² (×{ratio:.2f})",
                      "Traffic and industry should make the city higher than its surroundings."))
    for kind, a in (R.get("access") or {}).items():
        if "n" in a:
            vrows.append((f"Facilities ({kind})", f"{a['n']} from {a['src']}",
                          "OpenStreetMap can miss facilities in smaller towns; add an official list for planning." if a["src"] == "OpenStreetMap" else "Your list."))
    verdict = grade(checks)

    # ------------------------------------------------------------ summary
    rep = Report(kit.NAME, f"City study · heat season {mlab}/{year}", aoi.name, verdict, "#b2182b", aoi, g)
    lo, hi = np.nanpercentile(lst[np.isfinite(lst) & g.mask], [2, 98])
    lmap = MapSpec(lst, g, "Surface temperature", ramp="heat", vmin=float(lo), vmax=float(hi), unit="°C",
                   subtitle=f"{aoi.name} · median of {nd} Landsat days, months {mlab} {year}", source="Landsat 8/9")
    results = []
    if R.get("suhi") is not None:
        results.append(("Surface heat island (day, Landsat)", f"{R['suhi']:+.1f} °C"))
    sm = R.get("suhi_modis") or {}
    if sm.get("Night") is not None:
        results.append(("Heat island at night (MODIS)", f"{sm['Night']:+.1f} °C"))
    G = R.get("green") or {}
    if G.get("canopy_urban") is not None:
        results.append(("Tree canopy in the urban core", f"{G['canopy_urban']:.0f}%"))
    if G.get("green_per_person") is not None:
        results.append(("Green space per person (core)", f"{G['green_per_person']:.1f} m²"))
    if R.get("hvi_pop"):
        hp = R["hvi_pop"]
        results.append(("People in high/very high heat vulnerability", fmt(hp[4] + hp[5], 0)))
    h = (R.get("access") or {}).get("health") or {}
    if h.get("pop_far") is not None:
        results.append((f"People > {h['limit']:g} km from a hospital/clinic", f"{fmt(h['pop_far'], 0)} ({h['pop_far_pct']:.0f}%)"))
    if R.get("built_flood_km2") is not None:
        results.append(("Built-up in river flood zones / low ground", f"{fmt(R['built_flood_km2'])} km²"))
    if R.get("built_pocket_km2"):
        results.append(("Built-up in low-lying pockets (waterlogging)", f"{fmt(R['built_pocket_km2'])} km²"))
    if R.get("air"):
        results.append(("Hot-season days ≥ 40 °C air temperature", f"{R['air']['days_40']} of {R['air']['n_days']}"))
    if R.get("air") and R["air"].get("wbgt_hours_28") is not None:
        results.append(("Hours with outdoor heat stress (WBGT ≥ 28 °C)", f"{R['air']['wbgt_hours_28']}"))
    if R.get("greening"):
        results.append(("+10% tree canopy in the core", f"{R['greening']['dT']:+.1f} °C surface temperature"))
    if R.get("pm25"):
        results.append(("PM2.5 annual mean", f"{R['pm25']['mean']:.0f} µg/m³ ({R['pm25']['times_who']:.0f}× WHO)"))
    points = []
    if R.get("suhi") is not None:
        points.append(f"By day the urban core is <b>{R['suhi']:+.1f} °C</b> against a matched countryside ring "
                      f"({R['t_urban']:.1f} vs {R['t_rural']:.1f} °C at the surface)" + (f"; at night <b>{sm['Night']:+.1f} °C</b> (MODIS)." if sm.get("Night") is not None else "."))
    if R.get("suhi") is not None and R["suhi"] < 0:
        points.append("A negative daytime value is a <b>surface cool island</b>: the surrounding bare or dry fields heat up more than the "
                      "city in the hot season, as is common in dry regions (e.g. north-west India before the monsoon). Night heat is the health concern.")
    if R.get("slope_tree") is not None:
        points.append(f"Each extra 10% of tree canopy goes with <b>{R['slope_tree']:+.1f} °C</b> surface temperature (300 m blocks, r = {R['r_tree_lst']:.2f}).")
    cp = R.get("cool_park")
    if cp and cp[2] is not None:
        points.append(f"Parks of 1 ha or more cool their surroundings by <b>{cp[2]:.1f} °C</b> up to about <b>{cp[3]:.0f} m</b> away.")
    if R.get("mk_Night") or R.get("mk_Day"):
        points.append("Heat island trend since 2003 (MODIS, Aqua): day " + _mk(R.get("mk_Day")) + "; night " + _mk(R.get("mk_Night")) + ".")
    if G.get("pop_300m_green") is not None:
        points.append(f"<b>{G['pop_300m_green']:.0f}%</b> of people live within 300 m of green space of 0.5 ha or more "
                      f"(3-30-300 rule); <b>{G['pop_30pct_canopy']:.0f}%</b> live in neighbourhoods with 30% tree canopy.")
    if R.get("built_on_water_km2"):
        points.append(f"<b>{fmt(R['built_on_water_km2'])} km²</b> of built-up land stands where water was seen at least 10% of the time since 1984.")
    rep.summary(results, points, lmap)
    zz = np.where(R["urban"], 1, np.where(R["rural"] if R.get("rural") is not None else False, 2, np.where(g.mask, 3, 0))).astype(np.uint8) \
        if R.get("urban") is not None else np.where(g.mask, 3, 0).astype(np.uint8)
    facts = [("Heat season", f"months {mlab} of {year}"), ("Built-up", f"{fmt(R['built_km2'][1])} km² ({R['lc_year']})")]
    if R.get("ring_m"):
        facts.append(("Countryside ring", f"{R['ring_m'] / 1000:.2f} km wide, same area as the core, ±50 m elevation, no water"))
    if R.get("ur_note"):
        facts.append(("Note", R["ur_note"]))
    rep.study_area(MapSpec(zz, g, "Urban core and countryside reference", classes=ZONES, source="Esri land cover, Copernicus DEM"), facts)
    nd5 = len(R.get("no2_days") or [])
    rows = [
        ("Landsat 8/9 Collection 2 Level-2", "Surface temperature, heat island, cooling", "30 m (100 m thermal)", f"{nd} days {mlab}/{year}; 2014–{year} for trend", "USGS, public"),
        ("MODIS MOD11A1 / MOD11A2 / MYD11A2", "Same-day check, day and night heat island, trend", "1 km", f"2003–{year}", "NASA, public"),
        ("Esri / IO 10 m land cover", "Built-up, trees, growth", "10 m", f"{D.LULC_YEARS[0]} and {R['lc_year']}", "CC BY 4.0"),
        ("ESA WorldCover 2021", "Independent built-up check", "10 m", "2021", "CC BY 4.0"),
        ("Copernicus DEM / ASF HAND", "Elevation match, flood-prone ground", "30 m", "2021", "Copernicus / CC0"),
        ("JRC Global Surface Water", "Former water bodies", "30 m", "1984–2021", "CC BY 4.0"),
        (R.get("pop_src") or "Population (not reachable)", "People, heat vulnerability, access", "30–100 m", "~2020", "CC BY 4.0"),
    ]
    if nd5:
        rows.append(("Sentinel-5P TROPOMI NO₂", "Air pollution", "5.5 × 3.5 km", f"{nd5} days in the last year", "Copernicus data"))
    if R.get("air"):
        rows.append(("NASA POWER (MERRA-2)", "Air temperature, humidity, heat index, 1981 trend", "≈ 50 km", f"hourly {R['air']['start']} to {R['air']['end']}; daily 1981–", "NASA, public"))
    if R.get("pm25"):
        rows.append(("Copernicus CAMS via Open-Meteo", "PM2.5 fine particles", "≈ 40 km", f"daily {R['pm25']['year']} and earlier years", "CC BY 4.0"))
    if R.get("height") is not None:
        rows.append(("JRC GHS-BUILT-H", "Building height", "100 m", "2018", "CC BY 4.0"))
    if R.get("rwi") is not None:
        rows.append(("Meta Relative Wealth Index (HDX)", "Relative wealth in heat vulnerability", "2.4 km", "2021 release", "CC BY-NC 4.0 (non-commercial)"))
    if R.get("lcz") is not None:
        rows.append(("Global Local Climate Zones v3 (Demuzere et al. 2022)", "Urban form classes", "100 m", "2018 imagery", "CC BY 4.0"))
    if any("n" in a for a in (R.get("access") or {}).values()):
        rows.append(("OpenStreetMap / your layer", "Hospitals, clinics, schools", "points", "today", "ODbL"))
    rep.data(rows)
    rep.method([
        ("Surface temperature", "Landsat Collection 2 surface temperature (emissivity-corrected by USGS) on the clearest hot-season days; "
         "clouds, shadows and cirrus masked; per-pixel median."),
        ("Heat island", "Urban core = 500 m cells at least half built-up (largest clusters). Countryside = a ring around it of the same "
         "area, not built-up, not water and within ±50 m of the core's elevation (Peng et al. 2012). SUHI = mean core − mean ring."),
        ("Day and night", "MODIS 8-day land surface temperature (Terra 10:30/22:30, Aqua 13:30/01:30) over the same core and ring. Trend: "
         "Aqua hot-season SUHI per year since 2003, Mann–Kendall test with Sen's slope; Landsat SUHI per year since 2014 at 120 m."),
        ("Hot spots and comfort", "Getis-Ord Gi* on 300 m cells (95% and 99% confidence); Urban Thermal Field Variance Index (Zhang et al. 2006)."),
        ("Cooling", "Mean temperature in 30 m rings around parks (tree patches ≥ 1 ha) and water; cooling intensity = rise to the first "
         "turning point, cooling distance = where the rise flattens (< 0.1 °C per ring; Yu et al. 2017)."),
        ("Green space and 3-30-300", "Tree canopy = Sentinel-2 NDVI ≥ 0.40 in the hot (dry) season, when grass and crops are brown; "
         "green space = canopy or NDVI ≥ 0.5 at the greenest time of the year, excluding farm fields (10 m); share of people within 300 m of green space of "
         "0.5 ha or more, and in 300 m neighbourhoods with ≥ 30% canopy (Konijnendijk 2023). The '3 trees from home' part needs street "
         "photos and is not measured. Green space per person against the WHO-cited 9 m² minimum."),
        ("Heat vulnerability", "On 300 m cells: z-scores of surface temperature (hazard), log population (exposure) and lack of green "
         "space (sensitivity), plus low relative wealth (Meta Relative Wealth Index, 2.4 km tiles) where it exists, averaged and cut "
         "into quintiles. Age and housing are not in open global data; add them by ward."),
        ("Access", "Distance from every cell to the nearest hospital/clinic and school along OpenStreetMap roads (Dijkstra over road "
         "cells; walk to the nearest road, then follow roads), or straight line when roads cannot be read; people beyond the chosen limit."),
        ("Exposure", "Built-up on ground less than 5 m above the nearest stream (HAND), and on land where JRC saw water at least 10% of "
         "the time since 1984 (filled lakes, wetlands, river beds)."),
        ("Air", "Sentinel-5P tropospheric NO₂ (QA ≥ 0.75) on days spread over the last year, mapped on a 2 km grid by nearest pixel."),
        ("Air temperature and heat stress", "Hourly 2 m air temperature and humidity (NASA POWER, MERRA-2) at the area centre for the hot "
         "season; heat index by the NOAA/NWS Rothfusz equation with its low- and high-humidity adjustments; days with Tmax ≥ 40 °C and "
         "≥ 45 °C, warm nights (Tmin ≥ 25 °C) and hours in the NWS caution/danger bands. Trend of hot-season Tmax and Tmin since 1981: "
         "Mann–Kendall test with Sen's slope."),
        ("PM2.5", "Daily mean PM2.5 from CAMS at the centre (and four more points for areas wider than one CAMS cell); days above the "
         "WHO 2021 24-hour guideline (15 µg/m³), above 35 and 60 µg/m³, and the annual mean against the WHO annual guideline (5 µg/m³)."),
        ("Building height", "Average gross building height per 100 m cell (GHS-BUILT-H, 2018) on built-up land; its correlation with surface "
         "temperature on 300 m blocks."),
        ("Outdoor heat stress", "Simplified wet-bulb globe temperature of the Australian Bureau of Meteorology, WBGT = 0.567·T + 0.393·e + 3.94 "
         "(e = vapour pressure, hPa; average sun and light wind assumed), and Steadman's apparent temperature in the shade, "
         "AT = T + 0.33·e − 0.70·wind − 4.0, from hourly NASA POWER data. Hours with WBGT ≥ 28 °C (strenuous activity unsafe) and ≥ 31 °C."),
        ("UTCI", "Universal Thermal Climate Index from the 210-term polynomial of Bröde et al. (2012) (Ta, wind at 10 m as the "
         "index defines it, vapour pressure, Tmrt − Ta). Mean radiant temperature for a standing person in the open from NASA POWER "
         "hourly direct, diffuse and long-wave radiation (Di Napoli et al. 2020; absorptivity 0.7, emissivity 0.97, ground albedo "
         "0.2, ground at air temperature); 'shade' removes the short-wave terms. Classes: moderate 26, strong 32, very strong 38, "
         "extreme 46 °C."),
        ("Albedo and cool roofs", "Shortwave albedo from Landsat surface reflectance with Liang's (2001) coefficients "
         "(0.356·blue + 0.130·red + 0.373·NIR + 0.085·SWIR1 + 0.072·SWIR2 − 0.0018) on the same clear days as the temperature map. "
         "Dark built-up surfaces = albedo < 0.15; the albedo–temperature slope on built-up 300 m blocks gives °C per +0.1 albedo."),
        ("Local Climate Zones", "The global 100 m LCZ map (Demuzere et al. 2022; Stewart & Oke 2012 classes) is read for the area; "
         "mean surface temperature and people per zone."),
        ("Greening scenario", "The canopy–temperature slope fitted on 300 m blocks (with its standard error) is applied to +10 percentage "
         "points of canopy in built-up cells of the urban core; people living there are counted. It is a statistical association, not a model "
         "of airflow or shade."),
    ])

    # ------------------------------------------------------------ heat
    rep.heading("Surface temperature and heat island")
    rep.map(lmap, f"Median surface temperature of {nd} clear Landsat days ({', '.join(R['ldates'])}).")
    if tl:
        rep.bars([(D.LULC[k][0], v, D.LULC[k][1]) for k, v in sorted(tl.items(), key=lambda kv: -kv[1])],
                 "Mean surface temperature by land cover (°C).", "°C")
    if R.get("suhi") is not None:
        t = [["Urban core", fmt(R["urban"].sum() * g.px_km2), f"{R['t_urban']:.1f}"],
             ["Countryside ring", fmt(R["rural"].sum() * g.px_km2), f"{R['t_rural']:.1f}"],
             ["Surface heat island (day, Landsat)", "", f"{R['suhi']:+.1f}"]]
        for v in ("Day", "Night"):
            if sm.get(v) is not None:
                t.append([f"Surface heat island ({v.lower()}, MODIS 1 km)", "", f"{sm[v]:+.1f}"])
        rep.table(["", "Area km²", "°C"], t, "Heat island")
    if R.get("gi") is not None:
        rep.map(MapSpec(R["gi"], g, "Heat hot spots", classes=FX.GISTAR, source="Landsat 8/9"),
                "Getis-Ord Gi* on 300 m cells: statistically significant clusters of hot and cool surfaces.")
    rep.map(MapSpec(R["utfvi"], g, "Thermal comfort (UTFVI)", classes=UTFVI, source="Landsat 8/9"),
            "Urban Thermal Field Variance Index: 'worst' classes are where heat stress at the surface is highest.")
    uc = {c: float((R["utfvi"] == c).sum() * g.px_km2) for c in UTFVI}
    rep.table(["Class", "Area km²"], [[UTFVI[c][0], fmt(uc[c])] for c in UTFVI], "Thermal comfort classes", [UTFVI[c][1] for c in UTFVI])
    ms = R.get("modis_season") or {}
    for v, nm in (("Night", "night"), ("Day", "day")):
        if v in ms:
            a = ms[v]
            if np.isfinite(a).sum() > 3:
                rep.map(MapSpec(a, R["gk"], f"MODIS {nm} surface temperature", ramp="heat", vmin=float(np.nanpercentile(a, 2)),
                                vmax=float(np.nanpercentile(a, 98)), unit="°C", source="MODIS Terra + Aqua"),
                        f"Mean {nm} surface temperature, months {mlab} {year} (1 km). Night heat matters most for health.")
    tr, lt = R.get("modis_trend") or {}, R.get("landsat_trend") or {}
    if len(tr) >= 3 or len(lt) >= 3:
        yrs = sorted(set(tr) | set(lt))
        x = np.array(yrs, float)
        ser = []
        if tr:
            ser += [("MODIS day (Aqua 13:30)", [tr.get(y, {}).get("Day") for y in yrs], "#d73027"),
                    ("MODIS night (Aqua 01:30)", [tr.get(y, {}).get("Night") for y in yrs], "#4575b4")]
        if lt:
            ser.append(("Landsat (10:30)", [lt.get(y, {}).get("suhi") for y in yrs], "#fdae61"))
        rep.figure(render_multi_line(ser, x, "Year", "Heat island (°C)"),
                   "Surface heat island in the hot season, every year. Trend: MODIS day " + _mk(R.get("mk_Day")) + ", night "
                   + _mk(R.get("mk_Night")) + ", Landsat " + _mk(R.get("mk_landsat")) + ".")
    if R.get("lst_new_vs_old") and all(v is not None for v in R["lst_new_vs_old"]):
        a, b = R["lst_new_vs_old"]
        rep.text(f"Land built since {D.LULC_YEARS[0]} averages {a:.1f} °C at the surface, against {b:.1f} °C for older built-up land.")

    # ------------------------------------------------------------ cooling and green
    if G:
        rep.heading("Trees, green space and cooling")
        rep.map(MapSpec(np.where(g.mask, G["tree"], np.nan), g, "Tree canopy", ramp="ndvi", vmin=0, vmax=1, unit="fraction",
                        source="Esri 10 m land cover"), "Share of each 30 m cell covered by trees.")
        t = []
        if G.get("canopy_urban") is not None:
            t += [["Tree canopy, urban core", f"{G['canopy_urban']:.0f}%", "Many cities aim for 30% or more"],
                  ["Green space (trees + grass), urban core", f"{G['green_urban']:.0f}%", ""]]
        if G.get("green_per_person") is not None:
            ok = G["green_per_person"] >= 9
            t.append(["Green space per person, urban core", f"{G['green_per_person']:.1f} m²", "meets the 9 m² minimum" if ok else "below the 9 m² minimum"])
        if G.get("pop_300m_green") is not None:
            t += [["People within 300 m of green space ≥ 0.5 ha", f"{G['pop_300m_green']:.0f}%", "3-30-300 target: 100%"],
                  ["People in neighbourhoods with ≥ 30% canopy", f"{G['pop_30pct_canopy']:.0f}%", "3-30-300 target: 100%"]]
        if t:
            rep.table(["Indicator", "Value", "Benchmark"], t, "Green space indicators")
        series = []
        for nm, key, col in (("Parks (trees ≥ 1 ha)", "cool_park", "#1a9850"), ("Water bodies", "cool_water", "#2c7fb8")):
            c = R.get(key)
            if c and len(c[0]) >= 4:
                series.append((nm, c[0], c[1], col, c))
        if series:
            xs = np.array(max((s[1] for s in series), key=len), float)
            rep.figure(render_multi_line([(nm, ys, col) for nm, _, ys, col, _ in series], xs, "Distance from the edge (m)", "Surface temperature (°C)"),
                       "Temperature with distance from parks and water: the cooling effect fades with distance.")
            rep.table(["Feature", "Cooling intensity °C", "Cooling distance m"],
                      [[nm, _n(c[2]), _n(c[3], 0)] for nm, _, _, _, c in series], "Cooling by parks and water")
        if R.get("r_tree_lst") is not None:
            rep.text(f"Across 300 m blocks, surface temperature correlates with tree canopy at r = {R['r_tree_lst']:.2f} and with built-up "
                     f"share at r = {R['r_built_lst']:.2f}.")

    # ------------------------------------------------------------ vulnerability and access
    if R.get("hvi") is not None:
        rep.heading("Who is most at risk from heat")
        rep.map(MapSpec(R["hvi"], g, "Heat vulnerability index", classes=HVI, source="Landsat, Meta/GHS population, Esri land cover"),
                "Heat vulnerability: hot surface × many people × little green, on 300 m cells (quintiles).")
        hp = R["hvi_pop"]
        tot = sum(hp.values())
        rep.table(["Class", "People", "Share"], [[HVI[c][0], fmt(hp[c], 0), f"{hp[c] / max(tot, 1) * 100:.0f}%"] for c in HVI],
                  "People by heat vulnerability", [HVI[c][1] for c in HVI])
        if R.get("rwi") is not None:
            rep.text(f"Relative wealth is included as a fourth component ({R['rwi_n']} tiles of Meta's Relative Wealth Index, "
                     "Chi et al. 2022; CC BY-NC 4.0, non-commercial use only). Poorer neighbourhoods have fewer coolers, fans and "
                     "shaded homes, so they rank as more vulnerable.")
        elif R.get("rwi_note"):
            rep.text(f"Relative wealth was not added: {R['rwi_note']}.")
    acc = R.get("access") or {}
    if acc:
        rep.heading("Access to services")
        t = []
        for kind, a in acc.items():
            if "dist" in a:
                t.append([{"health": "Hospital / clinic", "school": "School"}[kind], str(a["n"]), f"{a['limit']:g} km",
                          fmt(a.get("pop_far"), 0) if a.get("pop_far") is not None else "–",
                          f"{a['pop_far_pct']:.0f}%" if a.get("pop_far_pct") is not None else "–"])
            else:
                t.append([kind, "–", "–", "–", f"skipped: {a.get('err', '')[:80]}"])
        hows = {x.get("how") for x in acc.values() if "dist" in x}
        how_txt = "along roads" if hows == {"along roads"} else "straight-line distance"
        rep.table(["Service", "Facilities", "Limit", "People beyond", "Share"], t, f"Access ({how_txt})")
        dets = [x.get("detour") for x in acc.values() if x.get("detour")]
        if dets:
            rep.text(f"Road distances are on average {dets[0]:.2f}× the straight-line distance ({R.get('roads_note', '')}).")
        elif R.get("roads_note") and "could not" in R["roads_note"]:
            rep.text("Distances are straight lines: OpenStreetMap roads could not be read this time. Real travel along roads is usually "
                     "1.2–1.6× longer.")
        a = acc.get("health") or {}
        if "dist" in a:
            rep.map(MapSpec(a["dist"], g, "Distance to the nearest hospital or clinic", ramp="risk", vmin=0, vmax=3 * a["limit"], unit="km",
                            source=a["src"], points=[(lon, lat) for lon, lat, *_ in a["fac"][:400]]), f"Distance to health care ({a.get('how', 'straight line')}).")

    # ------------------------------------------------------------ growth and exposure
    rep.heading("Growth and building in harm's way")
    b0, b1 = R["built_km2"]
    rep.map(MapSpec(R["growth"], g, f"Built-up growth {D.LULC_YEARS[0]}–{R['lc_year']}", classes=GROWTH, source="Esri 10 m land cover"),
            f"Built-up land grew from {fmt(b0)} to {fmt(b1)} km².")
    gf = {k: v for k, v in R["growth_from"].items() if v > 0}
    if gf:
        rep.bars([(D.LULC[k][0], v, D.LULC[k][1]) for k, v in sorted(gf.items(), key=lambda kv: -kv[1])], "What new built-up land replaced (km²).", "km²")
    rep.map(MapSpec(R["expo"], g, "Built-up land on flood-prone ground and former water", classes=EXPO, source="ASF HAND, JRC Global Surface Water"),
            "Low ground near streams floods first; filled lakes and river beds lose their storage.")
    t = []
    if R.get("built_flood_km2") is not None:
        src = "GloFAS 1-in-100-year river flood map" + ("" if R.get("hand_flat", True) else " or < 5 m above the nearest stream (HAND)")
        t.append([f"Built-up in a river flood zone ({src})", fmt(R["built_flood_km2"]), fmt(R.get("built_flood_new_km2")),
                  fmt(R.get("pop_flood"), 0) if R.get("pop_flood") is not None else "–"])
    if R.get("built_pocket_km2") is not None:
        t.append(["Built-up in low-lying pockets (≥ 2 m below the surrounding 1 km)", fmt(R["built_pocket_km2"]), "–", "–"])
    if R.get("built_on_water_km2") is not None:
        t.append(["Built-up where water was seen since 1984 (≥ 10% of the time)", fmt(R["built_on_water_km2"]), fmt(R.get("built_on_water_new_km2")), "–"])
    if t:
        rep.table(["", "km²", f"of which built since {D.LULC_YEARS[0]}", "People"], t, "Built-up land exposed to water")
    if R.get("hand_flat"):
        rep.text("The terrain is flat: almost all built-up land is less than 5 m above the nearest mapped stream, so HAND cannot tell "
                 "flood-prone ground apart here and is not used. Run the AreaGIS Kit: Flood for observed flood extents.")

    # ------------------------------------------------------------ air
    if R.get("no2") is not None:
        rep.heading("Air: nitrogen dioxide (NO₂)")
        v = R["no2"]
        rep.map(MapSpec(v, R["g5"], "Tropospheric NO₂", ramp="heat", vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)), unit="µmol/m²",
                        source="Sentinel-5P TROPOMI"), f"Mean NO₂ column on {len(R['no2_days'])} days in the last year (traffic, industry, power).")
        mo = R.get("no2_month") or {}
        if len(mo) >= 3:
            rep.line([("NO₂ (µmol/m²)", list(mo.values()), "#7f2704")], list(mo.keys()), "NO₂ by month (few days per month: indicative).", "µmol/m²")
        if R.get("no2_city") is not None:
            rep.text(f"Urban core {R['no2_city']:.0f} µmol/m² against {R['no2_bg']:.0f} in the surrounding ring (×{R['no2_city'] / max(R['no2_bg'], 1e-6):.2f}).")
    elif R.get("no2_err"):
        rep.text(f"NO₂ was not read: {R['no2_err']}")

    # ------------------------------------------------------------ air temperature, heat stress, PM2.5, height
    A = R.get("air")
    if A:
        rep.heading("Air temperature and heat stress")
        rep.table(["Measure", "Value"], [
            ["Mean daily maximum air temperature", f"{A['tmax_mean']:.1f} °C"], ["Mean daily minimum", f"{A['tmin_mean']:.1f} °C"],
            ["Hottest hour", f"{A['tmax_abs']:.1f} °C"], ["Highest heat index", f"{A['hi_max']:.1f} °C"],
            ["Days with Tmax ≥ 40 °C / ≥ 45 °C", f"{A['days_40']} / {A['days_45']} of {A['n_days']}"],
            ["Warm nights (Tmin ≥ 25 °C)", str(A["warm_nights"])],
            ["Hours: heat index 32–41 °C (extreme caution)", str(A["hours_extreme_caution"])],
            ["Hours: heat index 41–54 °C (danger)", str(A["hours_danger"])], ["Hours: heat index ≥ 54 °C (extreme danger)", str(A["hours_extreme_danger"])],
        ], f"Air temperature and heat index, {A['start']} to {A['end']} (NASA POWER, ≈ 50 km cell)")
        dd = A["daily"]
        if dd:
            rep.line([("Tmax", [x[1] for x in dd], "#b2182b"), ("Tmin", [x[2] for x in dd], "#2166ac"), ("Heat index max", [x[3] for x in dd], "#ef8a62")],
                     [f"{x[0][6:]}/{x[0][4:6]}" for x in dd], "Daily air temperature and heat index in the hot season.", "°C")
        tr = A.get("trend")
        at = R.get("air_trend") or {}
        if tr and at.get("tmax") and at["tmax"][0] is not None:
            (s1, p1), (s2, p2) = at["tmax"], at["tmin"]
            rep.line([("Hot-season mean Tmax", tr["tmax"], "#b2182b"), ("Hot-season mean Tmin", tr["tmin"], "#2166ac")], tr["years"],
                     "Hot-season air temperature since 1981 (NASA POWER).", "°C")
            rep.text(f"Since {tr['years'][0]} hot-season days warmed by <b>{s1 * 10:+.2f} °C per decade</b> (Mann–Kendall p = {p1:.3f}) and "
                     f"nights by <b>{s2 * 10:+.2f} °C per decade</b> (p = {p2:.3f}). Air temperature is for a ≈ 50 km cell: it shows the "
                     "regional climate, while the satellite maps above show which streets are hotter.")
    elif R.get("air_err"):
        rep.text(f"Air temperature was not read: {R['air_err']}")
    P = R.get("pm25")
    if P:
        rep.heading("Air: fine particles (PM2.5)")
        rep.table(["Measure", "Value"], [
            ["Annual mean PM2.5", f"{P['mean']:.0f} µg/m³ ({P['times_who']:.0f}× the WHO annual guideline of 5)"],
            ["Days above WHO 24-h guideline (15 µg/m³)", f"{P['days_who']} of {P['n_days']}"],
            ["Days above 35 / 60 µg/m³", f"{P['days_35']} / {P['days_60']}"], ["Worst day", f"{P['max']:.0f} µg/m³"],
        ] + [[f"Annual mean {y}", f"{v:.0f} µg/m³"] for y, v in sorted(P["per_year"].items())],
                  f"PM2.5 from Copernicus CAMS, {P['year']} ({P['cells']} CAMS cell{'s' if P['cells'] > 1 else ''})")
        if P["month"]:
            rep.bars([(f"{m:02d}", v, "#7f2704") for m, v in P["month"].items()], f"Mean PM2.5 by month, {P['year']} (µg/m³).", "µg/m³")
    elif R.get("pm25_err"):
        rep.text(f"PM2.5 was not read: {R['pm25_err']}")
    if R.get("height") is not None:
        hs = R["height_stats"]
        rep.heading("Building height")
        rep.map(MapSpec(R["height"], g, "Average building height (2018)", ramp="seq", vmin=0, vmax=float(max(np.nanpercentile(R["height"], 99), 3)),
                        unit="m", source="JRC GHS-BUILT-H R2023A"), "Average gross building height per 100 m cell.")
        txt = (f"On built-up land buildings average <b>{hs['mean']:.1f} m</b> (90th percentile {hs['p90']:.1f} m, tallest cell {hs['max']:.0f} m); "
               f"{hs['share_tall']:.0f}% of built-up cells average 15 m or more.")
        if R.get("r_height_lst") is not None:
            txt += (f" Between blocks, height and surface temperature correlate at r = {R['r_height_lst']:+.2f} "
                    "(taller blocks shade the ground and are often newer, with more paving or more green; see the canopy section).")
        rep.text(txt)

    # ------------------------------------------------------------ outdoor heat stress, albedo, LCZ, greening
    A2 = R.get("air")
    if A2 and A2.get("wbgt_max") is not None:
        rep.heading("Outdoor heat stress (WBGT and apparent temperature)")
        rep.table(["Measure", "Value"], [
            ["Highest WBGT (simplified)", f"{A2['wbgt_max']:.1f} °C"],
            ["Hours with WBGT ≥ 28 °C (strenuous work and sport unsafe)", f"{A2['wbgt_hours_28']} of {A2['n_hours']}"],
            ["Hours with WBGT ≥ 31 °C", str(A2["wbgt_hours_31"])],
            ["Days with at least one hour WBGT ≥ 28 °C", f"{A2['wbgt_days_28']} of {A2['n_days']}"],
            ["Highest apparent temperature (shade)", f"{A2['at_max']:.1f} °C"],
            ["Hours with apparent temperature ≥ 40 °C", str(A2["at_hours_40"])]],
            "Outdoor heat stress in the hot season (NASA POWER, ≈ 50 km cell)")
        UTc = A2.get("utci")
        if UTc:
            hs, hd = UTc["hours_sun"], UTc["hours_shade"]
            rep.table(["UTCI heat-stress class", "Hours in the sun", "Hours in full shade (sunlight removed; simple model)"], [
                ["Extreme (≥ 46 °C)", str(hs["extreme"]), str(hd["extreme"])],
                ["Very strong (38–46 °C)", str(hs["very_strong"]), str(hd["very_strong"])],
                ["Strong (32–38 °C)", str(hs["strong"]), str(hd["strong"])],
                ["Moderate (26–32 °C)", str(hs["moderate"]), str(hd["moderate"])],
                ["Highest UTCI", f"{UTc['max_sun']:.1f} °C", f"{UTc['max_shade']:.1f} °C"],
                ["Days with very strong stress or worse", f"{UTc['days_38_sun']} of {A2['n_days']}", f"{UTc['days_38_shade']} of {A2['n_days']}"]],
                "Universal Thermal Climate Index (UTCI) for a person outdoors, in the open sun and in full shade")
            dd = UTc["daily"]
            rep.line([("UTCI in the sun", [x[1] for x in dd], "#b2182b"), ("UTCI in the shade", [x[2] for x in dd], "#4393c3"),
                      ("Very strong stress (38 °C)", [38.0] * len(dd), "#999999")],
                     [f"{x[0][6:]}/{x[0][4:6]}" for x in dd], "Daily highest UTCI in the hot season: shade lowers heat stress by "
                     f"{UTc['max_sun'] - UTc['max_shade']:.0f} °C at the peak.", "°C")
        elif A2.get("utci_err"):
            rep.text(f"UTCI was not computed: {A2['utci_err']}")
    if R.get("albedo") is not None:
        rep.heading("Surface albedo and cool roofs")
        rep.map(MapSpec(R["albedo"], g, "Surface albedo (hot season)", ramp="seq", vmin=0.05, vmax=0.35, unit="albedo",
                        source="Landsat 8/9 surface reflectance; Liang (2001)"), "Share of sunlight reflected by the surface (dark = absorbs more heat).")
        al = R.get("albedo_lc") or {}
        if al:
            rep.bars([(D.LULC[k_][0], v_, D.LULC[k_][1]) for k_, v_ in sorted(al.items(), key=lambda kv: kv[1])], "Mean albedo by land cover.", "albedo")
        rf = R.get("roof")
        if rf:
            txt = (f"Built-up land reflects on average {rf['alb_built']:.2f} of sunlight; <b>{fmt(rf['dark_km2'])} km²</b> of built-up "
                   f"surfaces are dark (albedo < 0.15)")
            if rf.get("lst_dark") is not None and rf.get("lst_bright") is not None:
                txt += f", at {rf['lst_dark']:.1f} °C against {rf['lst_bright']:.1f} °C for bright ones (albedo ≥ 0.20)"
            txt += "."
            if rf.get("slope") is not None:
                txt += (f" Across built-up 300 m blocks, +0.1 albedo goes with {rf['slope']:+.1f} °C surface temperature "
                        f"(± {1.96 * rf['se']:.1f}, r = {rf['r']:+.2f}).")
                if rf["slope"] < 0:
                    txt += " Brighter surfaces are cooler here, so cool roofs and light paving on the dark surfaces are a direct lever."
                else:
                    txt += (" Here brighter built-up blocks are <b>hotter</b>, not cooler: in the hot season bright surfaces in these blocks are "
                            "mostly dry bare soil, dust and construction sites, which heat up despite reflecting more. This map therefore "
                            "cannot show the effect of cool roofs; that needs roof-level data. The dark-surface area is still where cool "
                            "materials would apply.")
            rep.text(txt)
    elif R.get("albedo_err"):
        rep.text(f"Albedo was not computed: {R['albedo_err']}")
    if R.get("lcz") is not None:
        rep.heading("Local Climate Zones")
        rep.map(MapSpec(R["lcz"], g, "Local Climate Zones", classes={c_: v for c_, v in SF.LCZ.items() if (R["lcz"] == c_).any()},
                        source="Demuzere et al. 2022, global LCZ map v3"), "Urban form (Stewart & Oke 2012 classes) from the global 100 m map.")
        rows_l = [r_ for r_ in R["lcz_rows"]]
        rep.table(["Zone", "Area km²", "Mean surface temperature °C", "People"],
                  [[SF.LCZ[r_["code"]][0], fmt(r_["km2"]), _n(r_["lst"]), _n(r_["pop"], 0)] for r_ in rows_l],
                  "Surface temperature and people by Local Climate Zone", [SF.LCZ[r_["code"]][1] for r_ in rows_l])
    elif R.get("lcz_err"):
        rep.text(f"Local Climate Zones were not read: {R['lcz_err']}")
    G2 = R.get("greening")
    if G2:
        rep.heading("Greening scenario: +10% tree canopy")
        txt = (f"If tree canopy in the built-up core rose by 10 percentage points (from {G2['canopy_now']:.0f}% today), the fitted relation "
               f"suggests surface temperature <b>{G2['dT']:+.1f} °C</b> (95% range {G2['lo']:+.1f} to {G2['hi']:+.1f} °C) on {fmt(G2['km2'])} km²")
        if G2.get("people"):
            txt += f", where about {fmt(G2['people'], 0)} people live ({fmt(G2['people_hot'], 0)} of them in the hottest fifth of the city)"
        rep.text(txt + ". This is a statistical estimate from today's pattern, not a planting plan; air temperature changes less than surface temperature.")

    # ------------------------------------------------------------ zones
    if R.get("zones"):
        rep.heading("By ward / zone")
        rep.table(["Zone", "Area km²", "Built %", "LST °C", "People", "Canopy %", "Green m²/person", "High HVI %", "Far from health %"],
                  [[z["name"], fmt(z["area"]), f"{z['built']:.0f}", _n(z["lst"]), _n(z["pop"], 0), _n(z["canopy"], 0), _n(z["green_pp"]),
                    _n(z["hvi_high"], 0), _n(z.get("far_health"), 0)] for z in sorted(R["zones"], key=lambda z: -(z["lst"] or 0))[:80]],
                  "Statistics by zone, hottest first (full list in zone_statistics.csv)")

    if R.get("lcz_check"):
        p1, p2 = R["lcz_check"]
        vrows.append(("Built LCZ vs Esri built-up", f"{p1:.0f}% of built LCZ area is built-up in Esri; {p2:.0f}% of Esri built-up is a built LCZ",
                      "Two independent maps (2018 vs today); differences include new construction and mixed 100 m cells."))
    if R.get("roof") and R["roof"].get("r") is not None:
        vrows.append(("Albedo vs surface temperature (built-up)", f"r = {R['roof']['r']:+.2f} on {R['roof']['n']} blocks",
                      "Expected negative (brighter = cooler). " + ("As expected." if R["roof"]["r"] < 0 else
                      "Positive here: bright built-up blocks are mostly bare soil and construction; read the albedo results with care.")))
    if R.get("rwi_check"):
        rr_, nn_ = R["rwi_check"]
        vrows.append(("Relative wealth vs built-up (2.4 km blocks)", f"r = {rr_:+.2f} over {nn_} blocks",
                      "City cores are usually wealthier than their rural fringe; a negative value would make the wealth layer suspect here."))
    if R.get("air_vs_modis_night"):
        ta, tl_ = R["air_vs_modis_night"]
        vrows.append(("Air vs surface at night", f"air Tmin {ta:.1f} °C vs MODIS night surface {tl_:.1f} °C (countryside)",
                      "Independent sources; at night air and ground are usually within a few degrees."))
    if R.get("pm25") and len(R["pm25"]["per_year"]) >= 2:
        vv = list(R["pm25"]["per_year"].values())
        vrows.append(("PM2.5 stability", f"annual means {min(vv):.0f}–{max(vv):.0f} µg/m³ across years",
                      "CAMS is a model with satellite data assimilation; compare with a ground monitor if one is near."))
    rep.validation(vrows + [("Your own check", "60 random built-up points in validation_points.geojson",
                             "Compare hot and cool points with imagery or a handheld thermometer on a clear afternoon.")])
    rep.limits([
        "Satellites measure the temperature of surfaces (roofs, roads, leaves), not the air people breathe; air temperature differences are smaller.",
        "Landsat passes at about 10:30; the afternoon peak and the night are covered only at 1 km by MODIS.",
        "The heat island depends on what counts as countryside; here it is a matched ring, which can include dry fields that are hot by day.",
        "Heat vulnerability uses temperature, people and green only. Age, health, income and housing are not in open global data.",
        "Access follows OpenStreetMap roads when they can be read (distance only: no speeds, one-way streets or traffic), otherwise straight lines; OpenStreetMap facilities can be incomplete.",
        "Former water bodies come from 1984 onwards; older filled tanks and lakes are not seen.",
        "Sentinel-5P pixels are several kilometres wide and only a sample of days is read: NO₂ shows broad patterns, not streets.",
        "Air temperature (NASA POWER) and PM2.5 (CAMS) are model cells of ≈ 40–50 km: one value for the whole city, not street-level. "
        "CAMS often underestimates PM2.5 in South Asian cities in winter; ground monitors are more accurate where they exist.",
        "Building height is a 2018 estimate per 100 m cell (average of all buildings there), not the height of single towers.",
        "WBGT here is the simplified Bureau of Meteorology form (average sun, light wind): it can be several degrees off in full sun or calm air.",
        "UTCI uses one ≈ 50 km weather cell and a simple open-ground radiation model: real streets differ with shade, walls and wind; "
        "the sun/shade pair brackets what a person outdoors experiences.",
        "Albedo is from Landsat at 30 m on clear hot-season days; the LCZ map is from 2018 imagery at 100 m.",
        "Night-time lights are not included: no open source without an account was found (NASA Black Marble and EOG VIIRS need a login).",
        "The greening scenario extrapolates today's canopy–temperature association; shade, wind and irrigation are not modelled. "
        "Its range uses every third 300 m block to reduce the effect of neighbouring blocks being alike, so it is approximate.",
    ])
    rep.conclusions(points + [f"Overall confidence: <b>{verdict['level']}</b>."])
    rep.references([
        D.CREDIT_LS, HT.CREDIT_MODIS_LST, D.CREDIT_LULC, D.CREDIT_WC, D.CREDIT_DEM, D.CREDIT_HAND, D.CREDIT_JRC, CT.CREDIT_HRSL,
        CT.CREDIT_S5P, D.CREDIT_OSM, CM.CREDIT_POWER, CM.CREDIT_CAMS, CM.CREDIT_GHSH,
        "Rothfusz, L. P. (1990). The heat index equation. NWS Technical Attachment SR 90-23.",
        SF.CREDIT_LCZ,
        "Stewart, I. D., Oke, T. R. (2012). Local Climate Zones for urban temperature studies. BAMS 93, 1879–1900.",
        "Liang, S. (2001). Narrowband to broadband conversions of land surface albedo I: Algorithms. Remote Sensing of Environment 76, 213–238.",
        "Australian Bureau of Meteorology. About the approximation to the WBGT used by the Bureau of Meteorology.",
        "Chi, G., Fang, H., Chatterjee, S., Blumenstock, J. E. (2022). Microestimates of wealth for all low- and middle-income countries. PNAS 119(3), e2113658119.",
        "Bröde, P. et al. (2012). Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494.",
        "Di Napoli, C., Hogan, R. J., Pappenberger, F. (2020). Mean radiant temperature from global-scale numerical weather prediction models. Int. J. Biometeorol. 64, 1233–1245.",
        "Steadman, R. G. (1994). Norms of apparent temperature in Australia. Aust. Met. Mag. 43, 1–16.",
        "WHO (2021). WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.",
        "Peng, S. et al. (2012). Surface urban heat island across 419 global big cities. Environ. Sci. Technol. 46, 696–703.",
        "Zhang, J., Wang, Y., Li, Y. (2006). A C++ program for retrieving land surface temperature from the data of Landsat TM/ETM+ band 6. Computers & Geosciences 32, 1796–1805.",
        "Getis, A., Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. Geographical Analysis 24, 189–206.",
        "Yu, Z. et al. (2017). Quantifying the cooling efficiency of urban green spaces. Urban Forestry & Urban Greening 21, 127–138.",
        "Konijnendijk, C. C. (2023). Evidence-based guidelines for greener, healthier, more resilient neighbourhoods: introducing the 3–30–300 rule. J. For. Res. 34, 821–830.",
        "Chakraborty, T., Lee, X. (2019). A simplified urban-extent algorithm to characterize surface urban heat islands on a global scale. Int. J. Appl. Earth Obs. Geoinf. 74, 269–280.",
        "Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. JASA 63, 1379–1389.",
    ])
    rep.proof(kit.sources(), kit.reproduce_command())
    rep.files(files + ["report.html", "report.docx", "report.pdf"])
    docs = rep.write(out_dir, kit.seconds())
    kit.step(100, "Done")
    s = f"Heat island {R['suhi']:+.1f} °C by day" if R.get("suhi") is not None else "Heat study done"
    return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers, summary=s + ".")
