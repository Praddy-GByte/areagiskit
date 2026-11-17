# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Report chapter for the AreaGIS Kit: Crop."""

import datetime as dt

import numpy as np

from ...core import datasets as D
from ...core.maps import MapSpec, render_grouped_bars, render_line
from ...core.report import Report, fmt, grade
from ..flood import hydro as HY
from . import modis as MO
from . import types as TY
from . import yieldfield as YF
from . import water as WA


def _n(v, d=0):
    return "–" if v is None else fmt(v, d)


def build(kit, R, aoi, out_dir, files, layers):
    from ..cropstudy import INTENSITY
    g, P, C = R["g"], R["P"], R["centers"]
    crop = R["crop"]
    ckm2 = float(crop.sum() * g.px_km2)
    inten = R["intensity"]
    V = R.get("val") or {}

    # ------------------------------------------------------------ verdict
    checks, vrows = [], []
    used = sum(1 for u in R["used"] if u)
    checks.append((2 if used >= 25 else 1 if used >= 15 else 0, 2, f"{used} of 36 ten-day periods had a usable Sentinel-2 image."))
    vrows.append(("Sentinel-2 coverage", f"{used} of 36 ten-day periods", "Gaps are filled by interpolation; long gaps blur sowing and harvest dates."))
    if V.get("double_cdl") is not None:
        d = V
        f1 = 2 * d["double_found"] * d["double_correct"] / max(d["double_found"] + d["double_correct"], 1e-9) / 100
        if d["double_cdl"] >= 2:
            checks.append((2 if f1 >= 0.75 else 1 if f1 >= 0.55 else 0, 2,
                           f"Double cropping: {d['double_ours']:.0f}% here vs {d['double_cdl']:.0f}% in USDA CDL {d['cdl_year']} "
                           f"(found {d['double_found']:.0f}%, correct {d['double_correct']:.0f}%, F1 {f1:.2f})."))
        else:
            checks.append((2 if d["double_ours"] <= 3 else 1 if d["double_ours"] <= 6 else 0, 2,
                           f"USDA CDL {d['cdl_year']} shows almost no double cropping ({d['double_cdl']:.1f}%); we map {d['double_ours']:.0f}%."))
        vrows.append((f"USDA Cropland Data Layer {d['cdl_year']} (double crops)",
                      f"ours {d['double_ours']:.0f}% vs CDL {d['double_cdl']:.0f}%; found {d['double_found']:.0f}%, correct {d['double_correct']:.0f}%",
                      "Independent government crop map (field surveys + satellites)." + ("" if d["cdl_same_year"] else " Different year: compare with care.")))
        if d.get("rice_cdl_km2") is not None and d["rice_cdl_km2"] > 1:
            checks.append((2 if d["rice_found"] >= 70 and d["rice_correct"] >= 70 else 1 if d["rice_found"] >= 50 else 0, 2,
                           f"Rice: found {d['rice_found']:.0f}% of CDL rice, {d['rice_correct']:.0f}% of our rice is CDL rice."))
            vrows.append(("USDA CDL (rice)", f"CDL {d['rice_cdl_km2']:.1f} km², ours {d['rice_ours_km2']:.1f} km²; found {d['rice_found']:.0f}%, correct {d['rice_correct']:.0f}%",
                          "Paddy rice from Sentinel-1 against the government crop map."))
    if R.get("sup"):
        s = R["sup"]
        checks.append((2 if s["oa"] >= 0.85 else 1 if s["oa"] >= 0.7 else 0, 2, f"Crop map from your points: {s['oa'] * 100:.0f}% correct on {s['n_test']} held-out points."))
        vrows.append(("Your field points (30% held out)", f"{s['oa'] * 100:.0f}% correct", "Points never used for training."))
    if V.get("field_purity") is not None:
        checks.append((1 if V["field_purity"] >= 85 else 0, 1,
                       f"Detected fields are {V['field_purity']:.0f}% one crop in USDA CDL (area-weighted; {V['field_pure90']:.0f}% of fields ≥ 90% one crop)."))
        vrows.append(("Detected field boundaries vs USDA CDL", f"{V['field_purity']:.0f}% one crop per field (area-weighted); "
                      f"{V['field_pure90']:.0f}% of {V['field_n']:,} fields are ≥ 90% one crop",
                      "A good field boundary encloses a single crop. CDL is 30 m, so narrow fields and edges lower purity."))
    if R.get("gpp_ndvi_r") is not None:
        r_ = R["gpp_ndvi_r"]
        checks.append((1 if r_ >= 0.6 else 0, 1, f"Year-to-year, MODIS GPP and MODIS NDVI season totals agree at r = {r_:.2f}."))
        vrows.append(("Productivity (yield proxy) anomaly: GPP vs NDVI (two MODIS products)", f"r = {r_:.2f} across years",
                      "Two different products from different algorithms; agreement means the good and bad years are robust."))
    if R.get("cal"):
        c = R["cal"]
        checks.append((2 if (c["r2"] or 0) >= 0.6 else 1 if (c["r2"] or 0) >= 0.3 else 0, 2,
                       f"Yield calibration on {c['n']} zones: R² {c['r2']:.2f}, leave-one-out error {c['loo_rmse']:.2f} {c['unit']}."))
        if c["b"] <= 0:
            checks.append((0, 1, "Reported yield does not rise with the satellite index (negative slope): the t/ha map is not reliable."))
        vrows.append(("Your reported yields (leave-one-out)", f"R² {c['r2']:.2f}; RMSE {c['rmse']:.2f}; LOO RMSE {c['loo_rmse']:.2f} {c['unit']} "
                      f"({c['loo_mape']:.0f}%)", "Each zone predicted from a model fitted without it."))
    VH = R.get("vhi")
    if VH and R.get("gpp"):
        cy = [y for y in VH["rows"] if y in R["gpp"]]
        if len(cy) >= 8:
            r_ = float(np.corrcoef([VH["rows"][y]["vhi"] for y in cy], [R["gpp"][y]["anomaly"] for y in cy])[0, 1])
            R["vhi_gpp_r"] = r_
            checks.append((1 if r_ >= 0.5 else 0, 1, f"Year-to-year, the Vegetation Health Index and the MODIS GPP productivity (yield proxy) anomaly agree at r = {r_:.2f}."))
            vrows.append(("Drought index vs productivity (VHI vs GPP)", f"r = {r_:.2f} over {len(cy)} seasons",
                          "VHI uses NDVI and surface temperature; GPP is a separate MODIS product. Agreement means the drought years are real."))
    SM = R.get("sm")
    if SM and SM.get("season"):
        from . import soil as SL
        szs = {y: v["z"] for y, v in SM["season"].items()}
        if R.get("gpp"):
            r_, n_ = SL.corr(szs, {y: v["gpp"] for y, v in R["gpp"].items()})
            if r_ is not None:
                SM["r_gpp"] = (r_, n_)
                checks.append((1 if r_ >= 0.3 else 0, 1, f"Year-to-year, crop-season root-zone soil moisture and MODIS GPP agree at r = {r_:.2f}."))
                vrows.append(("Soil moisture vs productivity (root zone vs GPP)", f"r = {r_:.2f} over {n_} seasons",
                              "Independent sources (land-surface model vs satellite). Weak where crops are irrigated."))
        vh_ = (R.get("vhi") or {}).get("rows") or {}
        if vh_:
            r_, n_ = SL.corr(szs, {y: v["vhi"] for y, v in vh_.items()})
            if r_ is not None:
                SM["r_vhi"] = (r_, n_)
                vrows.append(("Soil moisture vs Vegetation Health Index", f"r = {r_:.2f} over {n_} seasons", "Dry soil should go with low VHI."))
    if R.get("spi") and R.get("gpp"):
        a_, b_ = R["season_window"]
        mid_m = ((a_.month + ((b_ - a_).days % 365) // 60) - 1) % 12 + 1
        s3 = R["spi"].get(3, {})
        cy = [y for y in R["gpp"] if (y, mid_m) in s3]
        if len(cy) >= 8:
            r_ = float(np.corrcoef([s3[(y, mid_m)] for y in cy], [R["gpp"][y]["anomaly"] for y in cy])[0, 1])
            R["spi_gpp_r"] = (r_, mid_m)
            vrows.append((f"Rain vs productivity (SPI-3 in {TY.MONTHS[mid_m - 1]} vs GPP)", f"r = {r_:.2f} over {len(cy)} seasons",
                          "Weak where crops are irrigated or rain is rarely limiting; strong in rain-fed, drought-prone areas."))
    if V.get("lanid_found") is not None:
        lc_ = V.get("lanid_correct")
        checks.append((1 if (V["lanid_found"] >= 60 and (lc_ or 0) >= 60) else 0, 1,
                       f"Irrigated cropland vs LANID-US 2017: found {V['lanid_found']:.0f}%, correct " + ("–" if lc_ is None else f"{lc_:.0f}%") + "."))
        vrows.append(("Irrigated vs rainfed: LANID-US 2017 (Landsat irrigation map)",
                      f"found {V['lanid_found']:.0f}% of LANID irrigated; " + ("–" if lc_ is None else f"{lc_:.0f}% of ours is LANID irrigated")
                      + f" (on the {V['lanid_det']:.0f}% of cropland where a dry spell allowed a decision)",
                      "Independent irrigation map (2017; this crop year may differ). Land idle this year (CDL) is left out."))
    elif V.get("lanid_ref") is not None:
        vrows.append(("Irrigated vs rainfed: LANID-US 2017", f"LANID marks {V['lanid_ref']:.0f}% of this cropland as irrigated",
                      "No dry spell this crop year, so greenness cannot separate irrigated and rain-fed crops here (only paddy rice is marked)."))
    vrows.append(("Cropland actively growing", f"{V.get('active_share', 0):.0f}% of mapped cropland had a crop season",
                  "Land-cover cropland with no season this year is fallow, abandoned or misclassified."))
    verdict = grade(checks)

    # ------------------------------------------------------------ summary
    s0, s1 = R["start"], R["start"] + dt.timedelta(days=359)
    rep = Report(kit.NAME, f"Crop year {s0:%b %Y} – {s1:%b %Y}", aoi.name, verdict, "#1a9641", aoi, g)
    imap = MapSpec(inten, g, "Cropping intensity", classes=INTENSITY, subtitle=f"{aoi.name} · {s0:%b %Y}–{s1:%b %Y}", source="Sentinel-2")
    shares = {c: float((inten == c).sum() * g.px_km2) for c in INTENSITY}
    results = [("Cropland", f"{fmt(ckm2)} km²"), ("Cropping intensity", f"{R['ci']:.0f}%"),
               ("One / two / three crops", f"{fmt(shares[2])} / {fmt(shares[3])} / {fmt(shares[4])} km²"),
               ("Fallow this year", f"{fmt(shares[1])} km²")]
    if R.get("rice") is not None:
        results.append(("Paddy rice (Sentinel-1)", f"{fmt(R['rice'].sum() * g.px_km2)} km²"))
    if R.get("modis_year"):
        yrs = R["modis_year"]
        cy = [y for y in yrs if y >= s0.year]
        if cy:
            y = cy[0]
            vals = sorted(v["integral"] for v in yrs.values())
            rank = 1 + sum(1 for v in yrs.values() if v["integral"] > yrs[y]["integral"])
            results.append((f"Crop greenness {y} vs {min(yrs)}–{max(yrs)}", f"rank {rank} of {len(yrs)}"))
    if R.get("gpp"):
        ys_ = sorted(R["gpp"])
        cy = [y for y in ys_ if y == s0.year]
        if cy:
            results.append((f"Productivity anomaly {cy[0]} (MODIS GPP, a yield proxy, not yield)", f"{R['gpp'][cy[0]]['anomaly']:+.0f}% vs median year"))
    if R.get("vhi"):
        vr = R["vhi"]["rows"]
        cy = [y for y in vr if y == s0.year] or ([max(vr)] if vr else [])
        if cy:
            y = cy[0]
            results.append((f"Vegetation Health Index {y}", f"{vr[y]['vhi']:.0f} ({WA.vhi_class(vr[y]['vhi'])})"))
    if R.get("irr") is not None:
        ir = R["irr"]
        if not (R.get("irr_info") or {}).get("n_dry"):
            results.append(("Irrigated cropland", "not determinable (no dry spell this year)"))
        else:
            results.append(("Irrigated cropland (indicative)", f"{fmt((ir == 1).sum() * g.px_km2)} km² ({(ir == 1)[crop].mean() * 100:.0f}%)"))
    if R.get("cal"):
        c = R["cal"]
        results.append(("Estimated yield (calibrated)", f"{np.nanmean(c['map']):.2f} {c['unit']} mean"))
    if R.get("field_auto") and R.get("fields"):
        results.append(("Fields detected automatically", f"{len(R['fields'][1]):,}"))
    if R.get("burn_year"):
        last = max(y for y in R["burn_year"] if y <= s1.year)
        results.append((f"Burned cropland {last}", f"{fmt(R['burn_year'][last])} km²"))
    points = [f"Of {fmt(ckm2)} km² of cropland, <b>{fmt(shares[3] + shares[4])} km²</b> grew two or more crops and "
              f"<b>{fmt(shares[1])} km²</b> stayed fallow; cropping intensity <b>{R['ci']:.0f}%</b>."]
    if R.get("season_month"):
        top = sorted(R["season_month"].items(), key=lambda kv: -kv[1])[:2]
        points.append("Crops peak mostly in " + " and ".join(f"<b>{TY.MONTHS[m - 1]}</b> ({fmt(a)} km²)" for m, a in top) + ".")
    if R.get("modis_trend"):
        t = R["modis_trend"]
        points.append(f"Since 2001 cropland greenness {'rose' if t['slope'] > 0 else 'fell'} ({'significant' if t['p'] < 0.05 else 'not significant'}, p = {t['p']:.2f}).")
    rep.summary(results, points, imap)
    rep.study_area(MapSpec(np.where(crop, 1, np.where(g.mask, 2, 0)).astype(np.uint8), g, "Cropland",
                           classes={1: ("Cropland", "#e49635"), 2: ("Other land", "#ece9df")}, source="ESA WorldCover, Esri land cover"),
                   [("Crop year", f"{s0} to {s1}"), ("Cropland", f"{fmt(ckm2)} km² ({ckm2 / max(aoi.area_km2, 1e-9) * 100:.0f}% of the area)")])
    rep.data([
        ("Sentinel-2 L2A", "NDVI every 10 days; crop calendar, intensity, groups", "20 m", f"{used} of 36 periods, {s0}–{s1}", "Copernicus data"),
        ("Sentinel-1 RTC (VH)", "Paddy rice (fields flooded at sowing or transplanting)", "20 m", f"{len(R.get('vh_dates') or [])} passes", "Copernicus data"),
        ("MODIS MOD13Q1/MYD13Q1", "Crop condition 2001–today, anomaly, VCI", "250 m", f"{len(R.get('modis') or [])} composites", "NASA, public"),
        ("MODIS MCD64A1", "Crop residue burning", "500 m", "2001–today", "NASA, public"),
        ("MODIS MOD17A2HGF", "Yield anomaly (gross primary production)", "500 m", f"{len(R.get('gpp') or {})} seasons", "NASA, public"),
        ("ESA WorldCover / Esri land cover", "Cropland mask", "10 m", "2021 / " + str(min(max(s0.year, D.LULC_YEARS[0]), D.LULC_YEARS[-1])), "CC BY 4.0"),
        ("NASA POWER", "Rainfall against normal, SPI, dry spells", "~50 km", "1981–today", "Public"),
        ("NASA POWER (MERRA-2 land model)", "Root-zone and surface soil moisture", "~50 km", "1981–today, monthly", "Public"),
        ("MODIS MOD11A2", "Temperature Condition Index (VHI)", "1 km", f"{len((R.get('vhi') or {}).get('years') or [])} seasons", "NASA, public"),
        ("MODIS MOD16A3GF", "Evapotranspiration, water productivity", "500 m", f"{len(R.get('et') or {})} years", "NASA, public"),
    ] + ([("LANID-US 2017 (Xie & Lark 2021)", "Independent irrigation map (USA)", "30 m", "2017", "CC BY 4.0")] if V.get("lanid_ref") is not None else []) + ([("USDA Cropland Data Layer", "Independent validation (USA)", "30 m", str(V["cdl_year"]), "Public domain")] if V.get("cdl_year") else []))
    rep.method([
        ("Crop calendar", "For each 10-day period the least cloudy Sentinel-2 image is used; clouds and shadows are masked (scene "
         "classification), gaps are filled along time and the curve is smoothed (Savitzky–Golay, with cloud dips lifted)."),
        ("Seasons", "A crop season is a peak at least 0.20 NDVI above the lowest value on both sides and reaching 0.50; peaks closer than "
         "two months are one season. A second or third season must also reach 85% of the main peak (or NDVI 0.70) and last at least 90 "
         "days, and between the two crops NDVI must fall at least 0.30 below the lower peak (a harvest and nearly bare soil), so winter "
         "weeds, cover crops and regrowth on an unharvested field are not counted as crops. Sowing and harvest are where the curve "
         "crosses 20% of the season's rise and fall. Tested against USDA CDL 2021 (test areas): double crops found 92% (Arkansas) and "
         "93% (Kentucky); of what is mapped as double, 68% and 86% is double in CDL."),
        ("Crop groups", "Without field data, cropland pixels are grouped by the shape of their NDVI calendar (k-means). Groups are named by "
         "their seasons (e.g. 'Double season, peaks Sep + Mar'), not by crop names, because the same calendar can belong to different crops."),
        ("Paddy rice", "Sentinel-1 VH every 6–12 days: the field is flooded (VH ≤ −22 dB) between 30 days before and 60 days after the "
         "Sentinel-2 season start, and backscatter rises by at least 3 dB within 60 days as the rice grows. Tying the flood to the "
         "season start removes winter-wet bare soil. Tested against USDA CDL 2021: California water-seeded rice found 89%, correct 82%."),
        ("Crop condition", "MODIS NDVI over cropland for every 16-day period since 2001 gives the normal curve, this year's anomaly, the "
         "Vegetation Condition Index (Kogan 1995) and a Mann–Kendall trend."),
        ("Burning", "MODIS burned area (MCD64A1) on cropland, per month since 2001."),
        ("Yield", "Three levels. (1) Relative yield index: the area under each pixel's Sentinel-2 NDVI curve of its main season "
         "(sowing to harvest, above the bare-soil base) divided by the median of the area's cropland × 100. Season-integrated NDVI "
         "tracks absorbed light and biomass, the standard satellite yield proxy. (2) Yield anomaly per year since 2001: MODIS gross "
         "primary production summed over the same season window, as % of the median year. (3) Tonnes per hectare only if you give "
         "reported yields for zones: a linear fit of zone-mean index to reported yield, tested leave-one-out."),
        ("Drought (VHI)", "Vegetation Health Index (Kogan 1995): VCI from the crop-season mean MODIS NDVI and TCI from the crop-season "
         "mean MODIS daytime land surface temperature, each scaled between its lowest and highest value over all seasons since 2001 "
         "(hot = stressed); VHI = average of the two. Below 40 = drought (30 moderate, 20 severe, 10 extreme)."),
        ("Soil moisture", "NASA POWER monthly root-zone (GWETROOT, top ~1 m) and surface (GWETTOP, top 5 cm) soil wetness, 0 = dry, "
         "1 = saturated, from the MERRA-2 land model; anomaly = standardised per calendar month over 1981–today (z ≤ −1 = dry)."),
        ("Drought (SPI)", "Standardised Precipitation Index (McKee et al. 1993) for 1, 3 and 6 months from NASA POWER rainfall since 1981: "
         "a gamma distribution fitted for each calendar month (Thom 1958), dry months handled with the mixed distribution, transformed to "
         "a standard normal value. SPI ≤ −1 moderate, ≤ −1.5 severe, ≤ −2 extreme drought."),
        ("Water use", "MODIS yearly evapotranspiration (MOD16A3GF) over cropland; water-productivity proxy = crop-season GPP ÷ annual ET "
         "(gC/m² per mm), a relative 'crop per drop' index for comparing years and places, not grain per m³."),
        ("Irrigated vs rainfed (indicative)", "Dry spells are 10-day periods with less than 60 mm of rain in the previous 60 days and less "
         "than 25 mm in the previous 30 days (NASA POWER, area mean). Cropland with smoothed Sentinel-2 NDVI ≥ 0.60 during a dry spell, "
         "or flooded paddy rice, is marked likely irrigated; other cropland likely rainfed. Without any dry spell in the crop year the map "
         "says 'not determinable'. In the USA it is checked against LANID-US 2017."),
        ("Field boundaries", "If you give no field boundaries, fields are found from the NDVI time series: edges are where NDVI changes "
         "sharply in space on many dates (different crops, sowing dates, margins, roads); the mean gradient is split by Otsu's threshold "
         "inside cropland and the enclosed regions of at least 0.5 ha become fields. Within each field, low/medium/high zones are the "
         "terciles of peak NDVI."),
    ])

    # ------------------------------------------------------------ calendar
    rep.heading("Crop calendar and seasons")
    lab = [c.strftime("%b") if c.day <= 10 else "" for c in C]
    series = [("This year, cropland (Sentinel-2)", R["crop_curve"], "#1a9641")]
    if R.get("modis_clim"):
        cl = R["modis_clim"]
        norm = [cl.get((c.timetuple().tm_yday - 1) // 16, (None,))[0] for c in C]
        series.append(("Normal 2001–today (MODIS)", norm, "#636363"))
    rep.line(series, lab, "Mean cropland NDVI through the crop year, against the long-term normal.", "NDVI", 0, 1)
    rep.map(imap, "Number of crops grown this year on each pixel.")
    rep.table(["Crops this year", "Area km²", "Share of cropland"],
              [[INTENSITY[c][0], fmt(shares[c]), f"{shares[c] / max(ckm2, 1e-9) * 100:.0f}%"] for c in INTENSITY],
              "Cropping intensity", [INTENSITY[c][1] for c in INTENSITY])
    if R.get("season_month"):
        sm = R["season_month"]
        rep.figure(render_grouped_bars(TY.MONTHS, [("Crop area peaking in the month (km²)", [sm.get(m, 0) for m in range(1, 13)], "#1a9641")], "km²"),
                   "When crops peak: area whose crop reaches maximum greenness in each month.")
    for i in range(2):
        a = P["sos"][i]
        if np.isfinite(a).sum() > 100:
            doy = np.array([c.timetuple().tm_yday for c in C], np.float32)[np.clip(np.nan_to_num(a).astype(int), 0, len(C) - 1)]
            doy = np.where(np.isfinite(a), doy, np.nan)
            med = C[int(np.nanmedian(a))]
            e = P["eos"][i]
            med_e = C[int(np.nanmedian(e))] if np.isfinite(e).any() else None
            rep.map(MapSpec(doy, g, f"Season {i + 1}: sowing / green-up date", ramp="seq", vmin=float(np.nanpercentile(doy, 2)),
                            vmax=float(np.nanpercentile(doy, 98)), unit="day of year", source="Sentinel-2"),
                    f"Season {i + 1}: typical green-up {med:%d %b}" + (f", typical harvest {med_e:%d %b}." if med_e else "."))
    # ------------------------------------------------------------ groups
    gm, prof, labels = R["groups"]
    if gm is not None:
        rep.heading("Crop groups by calendar")
        cls = {i + 1: (lab_[:60], TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i, lab_ in enumerate(labels)}
        rep.map(MapSpec(gm, g, "Crop calendar groups", classes=cls, source="Sentinel-2"), "Cropland grouped by the shape of its NDVI calendar.")
        rep.line([(f"Group {i + 1}", [float(v) for v in prof[i]], TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i in range(len(labels))],
                 lab, "Average NDVI calendar of each group.", "NDVI", 0, 1)
        rep.table(["Group", "Calendar", "Area km²"], [[i + 1, labels[i], fmt((gm == i + 1).sum() * g.px_km2)] for i in range(len(labels))],
                  "Crop groups", [TY.GROUP_COLS[i % len(TY.GROUP_COLS)] for i in range(len(labels))])
    if R.get("rice") is not None:
        rep.heading("Paddy rice from radar")
        rk = float(R["rice"].sum() * g.px_km2)
        ck = float(crop.sum() * g.px_km2) or 1.0
        note = ("" if rk / ck >= 0.02 else
                " This is under 2% of the cropland: in places where rice is not grown it is usually wet or waterlogged fields after heavy "
                "rain, not paddy. Check a few of these fields before using the number.")
        rep.text(f"<b>{fmt(rk)} km²</b> shows the paddy signature (flooded before transplanting, then a steep rise in radar backscatter)." + note)
        rep.map(MapSpec(np.where(R["rice"], 1, np.where(crop, 2, 0)).astype(np.uint8), g, "Paddy rice", classes={1: ("Paddy rice", "#2c7fb8"), 2: ("Other cropland", "#ece9df")},
                        source="Sentinel-1"), "Paddy rice from Sentinel-1 VH.")
        if R.get("vh_curve_rice") and any(v is not None for v in R["vh_curve_rice"]):
            rep.line([("Rice fields", R["vh_curve_rice"], "#2c7fb8"), ("Other cropland", R["vh_curve_other"], "#e49635")],
                     [d[5:] for d in R["vh_dates"]], "Radar backscatter (VH) through the year.", "dB")
    if R.get("sup"):
        s = R["sup"]
        rep.heading("Crop map from your field points")
        cls = {i + 1: (n, TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i, n in enumerate(s["names"])}
        rep.map(MapSpec(s["map"], g, "Crop map", classes=cls, source="Sentinel-2/1 + your points"), "Random Forest crop map.")
        rep.table(["Crop", "Area km²"], [[n, fmt(s["area"][n])] for n in s["names"]], f"Crop areas ({s['n']} field points)")
        rep.table(["Map \\ field"] + s["names"], [[n] + [str(x) for x in s["cm"][i]] for i, n in enumerate(s["names"])],
                  f"Confusion matrix on {s['n_test']} held-out points (overall {s['oa'] * 100:.0f}% correct)")

    # ------------------------------------------------------------ condition
    if R.get("modis_year"):
        rep.heading("Crop condition 2001–today (MODIS)")
        ys = sorted(R["modis_year"])
        rep.figure(render_grouped_bars([str(y) for y in ys], [("Season greenness (sum of NDVI)", [R["modis_year"][y]["integral"] for y in ys], "#1a9641")], ""),
                   "Total cropland greenness each year (a proxy for crop production).")
        worst = sorted(ys[:-1], key=lambda y: R["modis_year"][y]["integral"])[:3]
        rep.text("Weakest crop years on record: " + ", ".join(str(y) for y in worst) + ".")
        if R.get("modis_trend"):
            t = R["modis_trend"]
            rep.text(f"Trend: Sen's slope {t['slope']:+.3f} per year (Mann–Kendall p = {t['p']:.3f}).")
    if R.get("vci"):
        v, d, g250, cm = R["vci"]
        rep.map(MapSpec(np.where(cm, v, np.nan), g250, f"Vegetation Condition Index, {d:%d %b %Y}", ramp="ndvi", vmin=0, vmax=100, unit="VCI",
                        source="MODIS"), "VCI: 0 = worst on record for this time of year, 100 = best. Below 35 indicates agricultural drought.")
        drought = float((v[cm] < 35).mean() * 100)
        rep.text(f"{drought:.0f}% of cropland was below VCI 35 (drought conditions) on {d:%d %b %Y}.")
    VH = R.get("vhi")
    if VH or R.get("spi_rows"):
        rep.heading("Drought: Vegetation Health Index and SPI")
    if VH:
        ys = sorted(VH["rows"])
        rep.figure(render_grouped_bars([str(y) for y in ys], [("VHI", [VH["rows"][y]["vhi"] for y in ys], "#1a9641"),
                                                                ("VCI (greenness)", [VH["rows"][y]["vci"] for y in ys], "#a6d96a"),
                                                                ("TCI (temperature)", [VH["rows"][y]["tci"] for y in ys], "#fdae61")], "index"),
                   "Vegetation Health Index of the crop season each year (cropland mean). Below 40 = drought.")
        yl = s0.year if s0.year in VH["maps"] else max(VH["maps"])
        rep.map(MapSpec(np.where(R["cm250"], VH["maps"][yl], np.nan), R["g250"], f"Vegetation Health Index, season {yl}", ramp="ndvi",
                        vmin=0, vmax=100, unit="VHI", source="MODIS NDVI + land surface temperature"),
                "VHI: under 40 drought (red), above 60 healthy (green).")
        worst = sorted(ys, key=lambda y: VH["rows"][y]["vhi"])[:3]
        rep.table(["Season", "VHI", "VCI", "TCI", "Cropland in drought (VHI < 40)", "Class"],
                  [[str(y), f"{VH['rows'][y]['vhi']:.0f}", f"{VH['rows'][y]['vci']:.0f}", f"{VH['rows'][y]['tci']:.0f}",
                    f"{VH['rows'][y]['drought_pct']:.0f}%", WA.vhi_class(VH["rows"][y]["vhi"])] for y in worst + ([yl] if yl not in worst else [])],
                  "Worst seasons on record and this season")
    elif R.get("vhi_err"):
        rep.text(f"The Vegetation Health Index could not be computed: {R['vhi_err']}")
    if R.get("spi_rows"):
        sr = R["spi_rows"]
        rep.figure(render_grouped_bars([r["month"][:3] for r in sr], [("SPI-1", [r["spi1"] or 0 for r in sr], "#6baed6"),
                                                                       ("SPI-3", [r["spi3"] or 0 for r in sr], "#2171b5"),
                                                                       ("SPI-6", [r["spi6"] or 0 for r in sr], "#08306b")], "SPI"),
                   "Standardised Precipitation Index in the crop year (below −1 = drought, above +1 = wet).")
        dry = [r["month"] for r in sr if (r["spi3"] is not None and r["spi3"] <= -1)]
        txt = ("Months in meteorological drought (SPI-3 ≤ −1): " + ", ".join(dry) + ". ") if dry else "No month of the crop year was in meteorological drought on the 3-month scale. "
        if R.get("spi_dry_years"):
            txt += "Years since 1981 with the most drought months (SPI-3 ≤ −1): " + ", ".join(f"{y} ({n} months)" for y, n in R["spi_dry_years"]) + "."
        rep.text(txt)
    if R.get("yield_index") is not None and np.isfinite(R["yield_index"]).any():
        rep.heading("Yield")
        yi = R["yield_index"]
        rep.map(MapSpec(yi, g, "Relative yield index (main season)", ramp="ndvi", vmin=60, vmax=140, unit="100 = median",
                        source="Sentinel-2"), "Relative yield index: season-integrated NDVI against the area's median cropland. "
                "This is a ranking, <b>not tonnes</b>, unless reported yields are given.")
        lowp = float((yi[crop] < 80).mean() * 100) if crop.any() else 0
        highp = float((yi[crop] > 120).mean() * 100) if crop.any() else 0
        rep.text(f"{lowp:.0f}% of cropland is more than 20% below the area's median productivity, {highp:.0f}% more than 20% above.")
        if R.get("cal"):
            c = R["cal"]
            rep.map(MapSpec(c["map"], g, f"Estimated yield ({c['unit']})", ramp="ndvi", vmin=float(np.nanpercentile(c["map"], 2)),
                            vmax=float(np.nanpercentile(c["map"], 98)), unit=c["unit"], source="Sentinel-2 + your yield statistics"),
                    f"Yield calibrated on {c['n']} zones: {c['unit']} = {c['a']:.2f} + {c['b']:.4f} × season NDVI integral.")
            rep.table(["Zone", "Season NDVI integral", f"Reported ({c['unit']})", f"Predicted ({c['unit']})"],
                      [[r["name"], f"{r['index']:.2f}", f"{r['reported']:.2f}", f"{r['pred']:.2f}"] for r in c["rows"][:60]],
                      f"Calibration: R² {c['r2']:.2f}, RMSE {c['rmse']:.2f}, leave-one-out RMSE {c['loo_rmse']:.2f} {c['unit']}")
        elif R.get("cal_err"):
            rep.text(f"Yield could not be calibrated: {R['cal_err']}")
        if R.get("gpp"):
            ys = sorted(R["gpp"])
            rep.figure(render_grouped_bars([str(y) for y in ys], [("Productivity anomaly (% vs median year)", [R["gpp"][y]["anomaly"] for y in ys], "#7a8f1f")], "%"),
                       "Crop-season gross primary production (MODIS) each year against the median year: a proxy for the yield anomaly. GPP is plant carbon uptake, not harvested grain: it tracks good and bad years but usually understates the size of losses (Kentucky 2012: −19% vs about −58% in reported corn yield).")
            bad = [y for y in ys if R["gpp"][y]["anomaly"] <= -15]
            rep.text(("Poor seasons (≥ 15% below the median): " + ", ".join(str(y) for y in bad) + ". " if bad else "No season was 15% or more below the median. ")
                     + f"Season window used: {R['season_window'][0]:%d %b} – {R['season_window'][1]:%d %b} each year."
                     + (f" Left out because the archive data are faulty (GPP near zero): {', '.join(str(y) for y in R['gpp_dropped'])}."
                        if R.get("gpp_dropped") else ""))
    if R.get("rain_months"):
        rm = R["rain_months"]
        rep.figure(render_grouped_bars([r["month"][:3] for r in rm], [("Rain this year (mm)", [r["rain"] for r in rm], "#2f6fb0"),
                                                                       ("Normal (mm)", [r["normal"] for r in rm], "#bdbdbd")], "mm"),
                   "Monthly rainfall in the crop year against the long-term normal (NASA POWER).")
        dry = [r["month"] for r in rm if r["z"] <= -1]
        wet = [r["month"] for r in rm if r["z"] >= 1]
        rep.text(("Unusually dry months: " + ", ".join(dry) + ". " if dry else "") + ("Unusually wet months: " + ", ".join(wet) + "." if wet else "")
                 + (f" Year-to-year, rainfall and crop greenness correlate at r = {R['rain_ndvi_r']:.2f}" +
                    (" (weak: irrigation probably buffers rain)." if abs(R["rain_ndvi_r"]) < 0.3 else ".") if R.get("rain_ndvi_r") is not None else ""))
    SMr = R.get("sm")
    if SMr and SMr.get("rows"):
        from . import soil as SL
        rep.heading("Soil moisture")
        sr = SMr["rows"]
        rep.figure(render_grouped_bars([r["month"][:3] for r in sr],
                                       [("Root zone (z)", [r["z_root"] or 0 for r in sr], "#8c510a"),
                                        ("Surface (z)", [r["z_top"] or 0 for r in sr], "#d8b365")], "z"),
                   "Soil moisture anomaly in the crop year (standard deviations from the 1981–today normal for each month).")
        rep.table(["Month", "Root-zone wetness", "Surface wetness", "Root-zone anomaly", "Condition"],
                  [[r["month"], "–" if r["root"] is None else f"{r['root']:.2f}", "–" if r["top"] is None else f"{r['top']:.2f}",
                    "–" if r["z_root"] is None else f"{r['z_root']:+.2f}", "–" if r["z_root"] is None else SL.sm_class(r["z_root"])] for r in sr],
                  "Soil wetness (0 = dry, 1 = saturated) and anomaly, month by month")
        ys_ = sorted(SMr.get("season") or {})
        if len(ys_) >= 8:
            rep.line([("Crop-season root-zone wetness", [SMr["season"][y]["wet"] for y in ys_], "#8c510a")], [str(y) for y in ys_],
                     "Crop-season mean root-zone soil wetness, every year since 1981.", "wetness (0–1)")
        dry = [r["month"] for r in sr if r["z_root"] is not None and r["z_root"] <= -1]
        txt = ("Months with dry root-zone soil (anomaly ≤ −1): " + ", ".join(dry) + ". ") if dry else "No month of the crop year had unusually dry root-zone soil. "
        if SMr.get("dry_years"):
            txt += "Crop seasons since 1981 with the most dry-soil months: " + ", ".join(f"{y} ({n})" for y, n in SMr["dry_years"]) + ". "
        if SMr.get("r_gpp"):
            txt += f"Year to year, crop-season soil moisture and the MODIS GPP productivity (yield proxy) anomaly agree at r = {SMr['r_gpp'][0]:.2f}"
            txt += (f"; soil moisture and VHI at r = {SMr['r_vhi'][0]:.2f}." if SMr.get("r_vhi") else ".")
        rep.text(txt)
    elif R.get("sm_err"):
        rep.text(f"Soil moisture could not be read: {R['sm_err']}")
    if R.get("et"):
        rep.heading("Water use: evapotranspiration and water productivity")
        ys = sorted(R["et"])
        series = [("Evapotranspiration (mm/year)", [R["et"][y] for y in ys], "#2c7fb8")]
        rep.figure(render_grouped_bars([str(y) for y in ys], series, "mm"), "Yearly evapotranspiration over cropland (MODIS MOD16A3GF).")
        if R.get("wp"):
            yw = sorted(R["wp"])
            rep.figure(render_grouped_bars([str(y) for y in yw], [("Season GPP per mm of ET", [R["wp"][y] for y in yw], "#7a8f1f")], "gC/m² per mm"),
                       "Water-productivity proxy: crop-season gross primary production divided by annual evapotranspiration.")
        if R.get("wp_map"):
            y, a_ = R["wp_map"]
            if np.isfinite(a_).any():
                rep.map(MapSpec(a_, R["g500y"], f"Water productivity proxy, {y}", ramp="ndvi", vmin=float(np.nanpercentile(a_, 2)),
                                vmax=float(np.nanpercentile(a_, 98)), unit="gC/m² per mm", source="MODIS GPP / MODIS ET"),
                        "More crop growth per mm of water used = higher (green).")
        txt = f"Mean cropland ET {np.mean(list(R['et'].values())):.0f} mm/year ({min(ys)}–{max(ys)})."
        if R.get("et_trend"):
            t_ = R["et_trend"]
            txt += f" Trend {t_['slope']:+.1f} mm/year (Mann–Kendall p = {t_['p']:.2f})."
        rep.text(txt)
    elif R.get("et_err"):
        rep.text(f"Evapotranspiration could not be read: {R['et_err']}")
    if R.get("irr") is not None:
        ir, inf = R["irr"], R.get("irr_info") or {}
        rep.heading("Irrigated vs rainfed cropland (indicative)")
        rep.map(MapSpec(np.where(crop, ir, 0).astype(np.uint8), g, "Irrigated vs rainfed cropland", classes=WA.IRR_CLASSES,
                        source="Sentinel-2, Sentinel-1, NASA POWER"),
                "Likely irrigated: green through a dry spell, or flooded paddy rice.")
        if inf.get("n_dry"):
            ds = inf["dry_periods"]
            rep.text(f"{inf['n_dry']} ten-day periods of the crop year were dry spells (< 60 mm of rain in 60 days), from "
                     f"{ds[0][:10] if ds else '–'} onwards. {fmt((ir == 1).sum() * g.px_km2)} km² ({(ir == 1)[crop].mean() * 100:.0f}% of cropland) "
                     f"stayed green through them or was paddy rice: likely irrigated.")
        else:
            rep.text("There was no dry spell in this crop year (rain was never below 60 mm in 60 days), so crop greenness cannot tell "
                     "irrigated from rain-fed fields here; only paddy rice is marked as irrigated. Try a drier year or season.")
    elif R.get("irr_err"):
        rep.text(f"The irrigation map could not be made: {R['irr_err']}")
    if R.get("burn_year"):
        rep.heading("Crop residue burning")
        ys = sorted(R["burn_year"])
        rep.figure(render_grouped_bars([str(y) for y in ys], [("Burned cropland (km²)", [R["burn_year"][y] for y in ys], "#d7301f")], "km²"),
                   "Cropland burned each year (MODIS burned area, 500 m; small fires are missed).")
        mon = {}
        for (y, m), v in R["burn"].items():
            mon[m] = mon.get(m, 0) + v
        rep.figure(render_grouped_bars(TY.MONTHS, [("Burned cropland, all years (km²)", [mon.get(m, 0) for m in range(1, 13)], "#d7301f")], "km²"),
                   "Burning season.")
        if R.get("burn_trend"):
            t = R["burn_trend"]
            rep.text(f"Trend in burned cropland: {t['slope']:+.2f} km²/year (p = {t['p']:.2f}).")
    if R.get("fields"):
        z, rows = R["fields"]
        rep.heading("Fields detected automatically" if R.get("field_auto") else "Your fields")
        if R.get("field_auto"):
            ids = R["field_ids"]
            sizes = np.array([r["area_ha"] for r in rows])
            rep.map(MapSpec(np.where(ids > 0, (ids % 7) + 1, np.where(crop, 8, 0)).astype(np.uint8), g, "Fields found from the NDVI time series",
                            classes={**{i + 1: (f"Field (colour {i + 1})", TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i in range(7)},
                                     8: ("Field edge / unassigned cropland", "#d9d9d9")}, source="Sentinel-2"),
                    f"{len(rows):,} fields (median {np.median(sizes):.1f} ha, 10–90%: {np.percentile(sizes, 10):.1f}–{np.percentile(sizes, 90):.1f} ha), "
                    f"covering {((ids > 0) & crop).sum() / max(crop.sum(), 1) * 100:.0f}% of cropland (the rest is field edges, lanes and very small plots). "
                    "Polygons with statistics are in fields_detected.gpkg.")
        rep.map(MapSpec(z, g, "Management zones (within-field)", classes={1: ("Low", "#d7191c"), 2: ("Medium", "#fdae61"), 3: ("High", "#1a9641")},
                        source="Sentinel-2"), "Low, medium and high productivity zones inside each field (terciles of peak NDVI).")
        rows_s = sorted(rows, key=lambda r: -r["area_ha"])
        rep.table(["Field", "Area ha", "Peak NDVI", "Variability (CV %)", "Low zone %", "Seasons", "Yield index", "Green-up"],
                  [[r["id"], fmt(r["area_ha"], 1), f"{r['peak']:.2f}", f"{r['cv']:.0f}", f"{r['low']:.0f}", f"{r['seasons']:.1f}",
                    "–" if r["yidx"] is None else f"{r['yidx']:.0f}", r["sowing"]] for r in rows_s[:25]],
                  "Field statistics (full list in field_statistics.csv)")
    if R.get("zones"):
        rep.heading("By zone")
        rep.table(["Zone", "Area km²", "Cropland km²", "Cropping intensity %", "Rice km²"],
                  [[z["name"], fmt(z["area"]), fmt(z["crop"]), _n(z["ci"], 0), _n(z["rice"])] for z in R["zones"][:60]], "Crop statistics by zone")
    if V.get("groups_cdl"):
        rep.heading("Validation with the USDA Cropland Data Layer")
        rep.table(["Group", "Calendar", "Area km²", "Main CDL crops"], [[r["group"], r["label"], fmt(r["km2"]), r["cdl"]] for r in V["groups_cdl"]],
                  f"What the government crop map says each calendar group is (CDL {V['cdl_year']})")

    rep.validation(vrows + [("Your own check", "60 random points in validation_points.geojson", "Look at them on imagery and fill the 'correct' column.")])
    rep.limits([
        "Persistent cloud (monsoon) leaves gaps; sowing and harvest dates are then less precise (±10–20 days).",
        "Calendar groups are not crop names: the same calendar can be maize or soybean. Add field points for a named crop map.",
        "Rice detection needs fields flooded at sowing or transplanting (most Asian paddy, California). Drill-seeded rice that is "
        "flooded only weeks later (e.g. Arkansas, USA) and upland rice are mostly missed: in Arkansas tests only 18–30% was found.",
        "Double cropping: on USDA CDL 2021 test areas about 92% of CDL double crops are found, and 68% (Arkansas, where almost no "
        "field is double-cropped) to 86% (Kentucky) of what we call double is double in CDL. The rest are mostly cover crops or "
        "volunteer growth after a real harvest; CDL itself also under-records double cropping. The harvest-gap rule removes about 7% of "
        "clear rice–wheat double crops in Punjab tests (bare period hidden by cloud or residue).",
        "MODIS (250–500 m) mixes fields with other land; small fires and short burns are missed.",
        "Yield: without your reported yields the kit gives a relative index (100 = median of the area) and a yearly anomaly, not tonnes. "
        "Season-integrated NDVI saturates in very dense canopies and does not see grain filling problems (heat at flowering, pests, lodging), "
        "so a green field can still yield poorly. Calibration with a handful of zones is only as good as those statistics.",
        "MODIS GPP (500 m) mixes crops in small-field landscapes; in a 2012-drought test (western Kentucky) it ranked the year correctly but "
        "magnitudes are smaller than the yield loss (see README validation).",
        "Soil moisture comes from a land-surface model on a ≈ 50 km grid (NASA POWER/MERRA-2), not from field probes or satellite "
        "radiometers; it shows regional dryness, not field-to-field differences, and ignores irrigation.",
        "Crop names without your field points: the ESA WorldCereal 2021 global crop-type maps are open (CC BY 4.0) but published as "
        "0.7–20 GB compressed archives per agro-ecological zone (or through openEO with an account), which cannot be read for a small "
        "area on the fly; they are therefore not used.",
        "VHI and SPI are regional: VHI uses 250 m–1 km MODIS (mixed pixels in small-field landscapes) and SPI one ≈ 50 km rainfall cell; "
        "irrigated fields can look healthy in a meteorological drought.",
        "Evapotranspiration (MOD16) is a model driven by satellite leaf area and weather; it underestimates ET of irrigated crops, so the "
        "water-productivity proxy is for comparing years and places, not an absolute 'crop per drop'.",
        "Irrigated vs rainfed is indicative: it needs a dry spell in the crop year (none in humid climates, then only paddy rice is marked), "
        "and wet soil after rain, shallow groundwater or wetlands can keep rain-fed crops green. In a test in California's Sacramento "
        "Valley (2021) it found 90% of LANID irrigated cropland and 86% of its irrigated pixels were irrigated in LANID, on land "
        "cropped that year.",
        "Automatic field boundaries come from 20 m pixels: fields narrower than ~60 m, fields with the same crop and date on both sides of a "
        "boundary, and fields split by in-field variability are merged or split. Edge pixels are left unassigned.",
    ])
    rep.conclusions(points + [f"Overall confidence: <b>{verdict['level']}</b>."])
    rep.references([
        "Sentinel-2 L2A and Sentinel-1 RTC: Copernicus data via Microsoft Planetary Computer.", MO.CREDIT_MODIS, D.CREDIT_WC, D.CREDIT_LULC, HY.CREDIT_POWER,
        "USDA NASS Cropland Data Layer (public domain).",
        "Kogan, F. N. (1995). Application of vegetation index and brightness temperature for drought detection. Adv. Space Res. 15, 91–100.",
        "Jönsson, P., Eklundh, L. (2004). TIMESAT—a program for analyzing time-series of satellite sensor data. Computers & Geosciences 30, 833–845.",
        "Savitzky, A., Golay, M. J. E. (1964). Smoothing and differentiation of data. Analytical Chemistry 36, 1627–1639.",
        "Nguyen, D. B., Gruber, A., Wagner, W. (2016). Mapping rice extent and cropping scheme in the Mekong Delta using Sentinel-1A. Remote Sensing Letters 7, 1209–1218.",
        YF.CREDIT_GPP,
        "Running, S. W. et al. (2004). A continuous satellite-derived measure of global terrestrial primary production. BioScience 54, 547–560.",
        "Lobell, D. B. et al. (2015). A scalable satellite-based crop yield mapper. Remote Sensing of Environment 164, 324–333.",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. Systems, Man, and Cybernetics 9, 62–66.",
        "Giglio, L. et al. (2018). The Collection 6 MODIS burned area mapping algorithm and product. Remote Sensing of Environment 217, 72–85.",
        WA.CREDIT_LST, WA.CREDIT_ET,
        "Kogan, F. N. (1997). Global drought watch from space. Bulletin of the American Meteorological Society 78, 621–636.",
        "McKee, T. B., Doesken, N. J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conf. Applied Climatology, 179–184.",
        "Thom, H. C. S. (1958). A note on the gamma distribution. Monthly Weather Review 86, 117–122.",
        "Gelaro, R. et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). J. Climate 30, 5419–5454.",
        "Reichle, R. H. et al. (2017). Assessment of MERRA-2 land surface hydrology estimates. J. Climate 30, 2937–2960.",
        "Running, S. W., Mu, Q., Zhao, M., Moreno, A. (2017). MODIS Global Terrestrial Evapotranspiration (ET) Product (MOD16A2/A3) User's Guide, v6.",
        "Pervez, M. S., Brown, J. F. (2010). Mapping irrigated lands at 250-m scale by merging MODIS data and national agricultural statistics. Remote Sensing 2, 2388–2412.",
        WA.CREDIT_LANID,
    ])
    rep.proof(kit.sources(), kit.reproduce_command())
    rep.files(files + ["report.html", "report.docx", "report.pdf"])
    docs = rep.write(out_dir, kit.seconds())
    kit.step(100, "Done")
    return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers,
                summary=f"Cropland {fmt(ckm2)} km², cropping intensity {R['ci']:.0f}%.")
