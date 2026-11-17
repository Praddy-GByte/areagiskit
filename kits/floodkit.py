# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""FloodKit: flood extent from Sentinel-1 radar (sees through cloud) + flood-prone zones."""

import datetime as dt
import json
import os
import warnings
from collections import defaultdict

import numpy as np
from osgeo import gdal, ogr

from ..core import datasets as D
from ..core.aoi import Grid
from ..core.data import KitError, item_date, search, warp_items, warp_s2
from ..core.raster import (neighbours, otsu, preview_classes, preview_float, write_classes,
                           write_float, write_geojson)
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

FLOOD = {1: ("Flood water", "#1f6fd1"), 2: ("Normal water", "#0b2f5c"), 3: ("Dry land", "#e9e6dc")}
AG = {1: ("Both satellites see water", "#1f6fd1"), 2: ("Only optical sees water", "#e8b02e"),
      3: ("Only radar sees water", "#c82828"), 4: ("Both see dry land", "#ece9df")}
PRONE = {1: ("Very high (HAND 0–2 m)", "#0b2f5c"), 2: ("High (2–5 m)", "#2f6fb0"),
         3: ("Moderate (5–10 m)", "#8db8dc"), 4: ("Low (10–20 m)", "#dfe9f2")}


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


class FloodKit(Kit):
    NAME = "FloodKit"
    KEY = "flood"
    TAGLINE = "Flood extent from radar + flood-prone zones"
    DESCRIPTION = ("Give a flood date: Sentinel-1 radar (works through cloud) before and after the event is "
                   "compared to map new water, with permanent water removed and terrain checks. Also maps "
                   "flood-prone land from height above the nearest drainage, and what was flooded.")
    MAX_KM2 = 3000
    PARAMS = [
        dict(name="event_date", type="date", label="Flood date (approx. peak)", default="2022-09-03"),
        dict(name="search_days", type="int", label="Look for a radar pass up to N days after", default=6),
        dict(name="reference", type="enum", label="Dry reference radar", default="last_year",
             options=["before", "last_year"], help="before = passes in the weeks before the event; "
             "last_year = same season one year earlier (better when rain started long before the peak)"),
        dict(name="reference_days", type="int", label="Reference window: days before the event", default=60),
    ]

    # -------------------------------------------------------------- data
    def _by_orbit_date(self, items):
        g = defaultdict(list)
        for it in items:
            p = it["properties"]
            g[(p.get("sat:relative_orbit"), item_date(it))].append(it)
        return g

    @staticmethod
    def _coverage(items, aoi):
        u = None
        for it in items:
            g = ogr.CreateGeometryFromJson(json.dumps(it["geometry"]))
            u = g if u is None else u.Union(g)
        if u is None:
            return 0.0
        inter = u.Intersection(aoi.geom)
        return inter.GetArea() / max(aoi.geom.GetArea(), 1e-12) if inter else 0.0

    def _optical_check(self, aoi, grid, date, radar_water, valid):
        d0 = _d(date)
        items = search("sentinel-2-l2a", aoi.bbox, str(d0 - dt.timedelta(days=2)), str(d0 + dt.timedelta(days=2)),
                       query={"eo:cloud_cover": {"lt": 15}}, max_items=50)
        if not items:
            return None
        by = {}
        for it in items:
            key = (it["properties"].get("s2:mgrs_tile"), item_date(it))
            if key not in by or it["id"] > by[key]["id"]:
                by[key] = it
        dates = {}
        for (t, d), it in by.items():
            dates.setdefault(d, []).append(it)
        d, its = min(dates.items(), key=lambda kv: (abs((_d(kv[0]) - d0).days),
                                                     np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]])))
        g = warp_s2(its, "B03", grid)
        sw = warp_s2(its, "B11", grid)
        scl = warp_items(its, "SCL", grid, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
        mndwi = (g - sw) / (g + sw)
        ok = valid & np.isfinite(mndwi) & ~np.isin(scl, [0, 1, 3, 8, 9, 10])
        if ok.sum() < 0.5 * valid.sum():
            return None
        opt = (mndwi > 0) & ok
        rad = radar_water & ok
        both = (opt & rad).sum()
        po = (opt == rad)[ok].mean()
        pa, pb = opt[ok].mean(), rad[ok].mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        cmp = np.zeros(grid.shape, np.uint8)
        cmp[ok] = 4
        cmp[opt & rad] = 1
        cmp[opt & ~rad] = 2
        cmp[rad & ~opt] = 3
        return dict(date=d, scene=its[0]["id"], confirmed=both / max(rad.sum(), 1) * 100,
                    found=both / max(opt.sum(), 1) * 100, agree=po * 100, kappa=(po - pe) / (1 - pe) if pe < 1 else 0,
                    opt_km2=opt.sum() * grid.px_km2, map=cmp, cover=ok.sum() / max(valid.sum(), 1) * 100)

    def _read_db(self, items, grid, pol="vv"):
        a = warp_items(items, pol, grid, resample="average", dtype=gdal.GDT_Float32,
                       src_nodata=-32768, dst_nodata=np.nan)
        a = np.where(a > 0, a, np.nan)
        db = 10 * np.log10(a)
        db[~grid.mask] = np.nan
        return db

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        try:
            ev = _d(params["event_date"])
        except Exception:
            raise KitError("Give the flood date as YYYY-MM-DD.")
        sd = int(params.get("search_days") or 6)
        rd = int(params.get("reference_days") or 60)
        grid = Grid(aoi, 20, max_pixels=12_000_000)

        self.step(4, "Searching Sentinel-1 radar passes around the flood date")
        post_items = search("sentinel-1-rtc", aoi.bbox, str(ev - dt.timedelta(days=1)), str(ev + dt.timedelta(days=sd)))
        if not post_items:
            raise KitError(f"No Sentinel-1 pass over this area between {ev - dt.timedelta(days=1)} and "
                           f"{ev + dt.timedelta(days=sd)}. Try a later date or a larger search window.")
        post_g = self._by_orbit_date(post_items)
        cov = {k: self._coverage(v, aoi) for k, v in post_g.items()}
        good = [k for k in post_g if cov[k] >= 0.6] or [max(cov, key=cov.get)]
        # best coverage first, then closest to the flood date
        orbit, post_date = sorted(good, key=lambda k: (-round(cov[k], 1), abs((_d(k[1]) - ev).days)))[0]

        if params.get("reference", "before") == "last_year":
            ry = ev.replace(year=ev.year - 1) if not (ev.month == 2 and ev.day == 29) else ev.replace(year=ev.year - 1, day=28)
            r0, r1 = ry - dt.timedelta(days=30), ry + dt.timedelta(days=30)
            ref_label = "same season, previous year"
        else:
            r0, r1 = ev - dt.timedelta(days=rd), ev - dt.timedelta(days=10)
            ref_label = f"{rd} days before the event"
        ref_items = [i for i in search("sentinel-1-rtc", aoi.bbox, str(r0), str(r1))
                     if i["properties"].get("sat:relative_orbit") == orbit]
        ref_g = self._by_orbit_date(ref_items)
        ref_keys = sorted([k for k in ref_g if self._coverage(ref_g[k], aoi) >= 0.6 * cov[(orbit, post_date)]],
                          key=lambda k: k[1])[-4:]
        if not ref_keys:
            raise KitError("No reference radar pass on the same orbit covering this area. "
                           "Increase the reference period or switch the reference to last year.")
        ref_dates = [k[1] for k in ref_keys]

        self.step(12, f"Reading radar after the flood ({post_date}, orbit {orbit}, {cov[(orbit, post_date)] * 100:.0f}% coverage)")
        post = self._read_db(post_g[(orbit, post_date)], grid)
        post_vh = self._read_db(post_g[(orbit, post_date)], grid, "vh")
        refs = []
        for i, k in enumerate(ref_keys):
            self.step(20 + i * 8, f"Reading reference radar {k[1]}")
            refs.append(self._read_db(ref_g[k], grid))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(np.stack(refs), axis=0) if len(refs) > 1 else refs[0]

        self.step(55, "Reading terrain (HAND, slope) and permanent water")
        hand = D.hand(grid)
        elev = D.dem(grid)
        slope, _ = D.slope_aspect(elev, grid)
        occ = D.water_occurrence(grid)
        lc_year = min(max(ev.year - 1, D.LULC_YEARS[0]), D.LULC_YEARS[-1])
        self.step(68, f"Reading land cover {lc_year} for impact")
        lc, _ = D.lulc(grid, lc_year)

        # ------------------------------------------------------------ detection
        self.step(76, "Detecting flood water")
        thr = otsu(post[(post > -30) & (post < 5)])
        thr = -18.0 if thr is None else float(np.clip(thr, -23, -15))
        valid = np.isfinite(post) & np.isfinite(ref) & grid.mask
        terrain_ok = (np.nan_to_num(hand, nan=999) <= 20) & (np.nan_to_num(slope, nan=90) <= 8)

        def detect(after, before):
            new = (after < thr) & ((after - before) < -3) & valid & terrain_ok
            keep = neighbours(new, 1) >= 4
            return new & keep

        water_after = (post < thr) & valid
        normal = valid & ((ref < thr) | (np.nan_to_num(occ, nan=0) >= 80))
        flood = detect(post, ref) & ~normal
        cls = np.zeros(grid.shape, np.uint8)
        cls[valid] = 3
        cls[normal & water_after] = 2
        cls[normal & ~water_after & (np.nan_to_num(occ, nan=0) >= 80)] = 2
        cls[flood] = 1

        prone = np.zeros(grid.shape, np.uint8)
        h = np.nan_to_num(hand, nan=999)
        m = grid.mask & np.isfinite(hand)
        prone[m & (h <= 2)] = 1
        prone[m & (h > 2) & (h <= 5)] = 2
        prone[m & (h > 5) & (h <= 10)] = 3
        prone[m & (h > 10) & (h <= 20)] = 4

        px = grid.px_km2
        flood_km2 = flood.sum() * px
        normal_km2 = (cls == 2).sum() * px
        cover = valid.sum() / max(grid.mask.sum(), 1)

        # impact by land cover
        impact = {}
        for k in D.LULC:
            n = (flood & (lc == k)).sum()
            if n:
                impact[k] = n * px

        # ------------------------------------------------------------ validation
        self.step(84, "Validating")
        checks = []
        fa = None
        if len(refs) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base = np.nanmedian(np.stack(refs[:-1]), axis=0) if len(refs) > 2 else refs[0]
            false = detect(refs[-1], base) & ~normal
            fa = false.sum() / max(valid.sum(), 1) * 100
            ratio = fa / max(flood.sum() / max(valid.sum(), 1) * 100, 1e-6)
            checks.append((2 if fa < 0.5 else 1 if fa < 2 else 0, 2,
                           f"Running the same method on two dry dates flags {fa:.2f}% of the area as 'flood' "
                           f"(false alarms), versus {flood.sum() / max(valid.sum(), 1) * 100:.2f}% after the event."))
        in_low = (flood & (h <= 10)).sum() / max(flood.sum(), 1) * 100 if flood.any() else None
        if in_low is not None:
            checks.append((2 if in_low >= 80 else 1 if in_low >= 60 else 0, 2,
                           f"{in_low:.0f}% of mapped flood water lies on low ground (HAND ≤ 10 m), where floods are physically expected."))
        vh_agree = None
        if flood.any():
            thr_vh = otsu(post_vh[valid & (post_vh > -35)]) or -24
            vh_water = (post_vh < thr_vh) & valid
            vh_agree = (flood & vh_water).sum() / flood.sum() * 100
            checks.append((1 if vh_agree >= 70 else 0, 1,
                           f"The second radar polarisation (VH) independently confirms {vh_agree:.0f}% of the flood pixels."))
        checks.append((1 if cover >= 0.95 else 0, 1, f"Radar covers {cover * 100:.0f}% of the area on all dates used."))

        # independent check with a different sensor: Sentinel-2 optical water (MNDWI) within 2 days
        self.step(86, "Cross-checking with an optical satellite image")
        optical = self._optical_check(aoi, grid, post_date, (cls == 1) | (cls == 2), valid)
        if optical:
            o = optical
            checks.append((3 if o["confirmed"] >= 85 and o["kappa"] >= 0.6 else 2 if o["confirmed"] >= 70 else 0, 3,
                           f"A different satellite (Sentinel-2 optical, {o['date']}) independently sees water at "
                           f"{o['confirmed']:.0f}% of the radar water pixels; overall agreement {o['agree']:.0f}%, kappa {o['kappa']:.2f}."))
        else:
            checks.append((0, 0, "No cloud-free optical image within 2 days of the radar pass, so the optical cross-check was not possible."))
        verdict = grade(checks)

        rng = np.random.default_rng(7)
        pts = []
        for lab, sel in (("Flood water", flood), ("Dry land", cls == 3)):
            idx = np.flatnonzero(sel.ravel())
            for p in (rng.choice(idx, size=min(15, idx.size), replace=False) if idx.size else []):
                r, c = divmod(int(p), grid.cols)
                pts.append((grid.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "expected": lab, "date": post_date, "correct": None}))

        # ------------------------------------------------------------ outputs
        self.step(90, "Writing maps")
        f_flood = write_classes(os.path.join(out_dir, f"flood_extent_{post_date}.tif"), cls, grid, FLOOD)
        f_prone = write_classes(os.path.join(out_dir, "flood_prone_zones.tif"), prone, grid, PRONE)
        f_post, _ = write_float(os.path.join(out_dir, f"radar_vv_{post_date}_db.tif"), post, grid, "gray", -25, 0, "dB")
        f_ref, _ = write_float(os.path.join(out_dir, "radar_vv_dry_reference_db.tif"), ref, grid, "gray", -25, 0, "dB")
        f_hand, _ = write_float(os.path.join(out_dir, "hand_m.tif"), np.where(grid.mask, hand, np.nan), grid, "depth", 0, 30, "m")
        f_pts = write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts)
        f_csv = os.path.join(out_dir, "flood_impact_by_landcover.csv")
        with open(f_csv, "w", encoding="utf-8") as fh:
            fh.write("land_cover,flooded_km2\n")
            for k, v in sorted(impact.items(), key=lambda kv: -kv[1]):
                fh.write(f"{D.LULC[k][0]},{v:.4f}\n")

        self.step(95, "Writing the report")
        src = "Copernicus Sentinel-1 RTC (Microsoft Planetary Computer); JRC Global Surface Water; ASF GLO-30 HAND"
        share_vh = (prone == 1).sum() / max(grid.mask.sum(), 1)
        prone_km2 = ((prone == 1) | (prone == 2)).sum() * px
        rep = Report(self.NAME, f"Flood of {ev.day} {ev:%B %Y}", aoi.name, verdict, "#1f5fa8", aoi, grid)
        flood_map = MapSpec(cls, grid, f"Flood extent, {_d(post_date).day} {_d(post_date):%B %Y}", classes=FLOOD,
                            subtitle=f"{aoi.name} · Sentinel-1 radar", source=src)
        results = [("Flooded area", f"{fmt(flood_km2)} km²"),
                   ("Share of the area", f"{flood_km2 / max(aoi.area_km2, 1e-9) * 100:.1f}%"),
                   ("Radar image date", post_date)]
        if impact.get(5, 0) > 0.01:
            results.append(("Cropland flooded", f"{fmt(impact[5])} km²"))
        if impact.get(7, 0) > 0.01:
            results.append(("Built-up land flooded", f"{fmt(impact[7])} km²"))
        if optical:
            results.append(("Confirmed by optical satellite", f"{optical['confirmed']:.0f}%"))
        points = [f"<b>{fmt(flood_km2)} km²</b> was under flood water on <b>{post_date}</b>, in addition to "
                  f"{fmt(normal_km2)} km² of normal rivers and lakes."]
        for k in (5, 7):
            if impact.get(k, 0) > 0.01:
                points.append(f"<b>{fmt(impact[k])} km²</b> of {D.LULC[k][0].lower()} was flooded "
                              f"({impact[k] / max((lc == k).sum() * px, 1e-9) * 100:.0f}% of all {D.LULC[k][0].lower()} in the area).")
        if optical:
            points.append(f"A second satellite (Sentinel-2 optical, {optical['date']}) independently confirms "
                          f"<b>{optical['confirmed']:.0f}%</b> of the mapped water.")
        rep.summary(results, points, flood_map)
        rep.study_area(MapSpec(np.where(grid.mask, elev, np.nan), grid, "Study area: elevation", ramp="elev",
                               vmin=float(np.nanpercentile(elev, 2)), vmax=float(np.nanpercentile(elev, 98)), unit="m",
                               subtitle=aoi.name, source="Copernicus DEM GLO-30"),
                       [("Flood date given", f"{ev.day} {ev:%B %Y}")])
        rep.data([
            ("Sentinel-1 RTC (VV, VH)", "Water after the flood and in the dry reference", "10 m (analysed at 20 m)",
             f"{post_date}; reference {', '.join(ref_dates)}", "Copernicus data; RTC by Microsoft"),
            ("Sentinel-2 L2A", "Independent optical check", "10–20 m", optical["date"] if optical else "none found", "Copernicus data"),
            ("ASF GLO-30 HAND", "Flood-prone zones, terrain mask", "30 m", "2021 release", "CC0"),
            ("Copernicus DEM GLO-30", "Slope mask, study area map", "30 m", "2011–2015", "Copernicus DEM licence"),
            ("JRC Global Surface Water", "Permanent water mask", "30 m", "1984–2021", "Free with attribution"),
            ("IO/Esri annual land cover", "What was flooded", "10 m", str(lc_year), "CC BY 4.0"),
        ])
        rep.method([
            ("Find the right radar images", f"The pass closest to the flood date that covers the area was used ({post_date}, orbit {orbit}). "
             f"The reference is the median of {len(ref_dates)} passes on the same orbit ({ref_label}), so the viewing angle is identical."),
            ("Why radar", "Sentinel-1 sends its own microwave signal and sees through cloud. Calm water reflects the signal away like a mirror and looks dark; land scatters it back and looks bright."),
            ("Detect new water", f"A pixel is flood water if it is darker than {thr:.1f} dB (threshold found automatically with Otsu's method) "
             "and at least 3 dB darker than in the reference."),
            ("Remove false water", "Permanent water (JRC occurrence ≥ 80%), slopes above 8° and land more than 20 m above drainage are excluded, "
             "because radar shadows there look like water. Isolated pixels are removed."),
            ("Measure impact", f"Flood pixels are overlaid on the {lc_year} land cover map to measure flooded cropland, built-up land and trees."),
            ("Validate", "The same rules are run on two dry dates to measure false alarms, compared with the second radar polarisation, "
             "checked against terrain, and compared with an optical image from another satellite when a clear one exists."),
        ])
        rep.heading("Flood extent")
        rep.text(f"Figure below shows flood water on {post_date}. Dark blue is water that is normally present (rivers, lakes); "
                 "bright blue is new water from the flood.")
        rep.map(flood_map, f"Flood extent on {post_date} from Sentinel-1 radar.")
        rep.map(MapSpec(post, grid, f"Radar backscatter (VV), {post_date}", ramp="gray", vmin=-25, vmax=0, unit="dB",
                        subtitle="Dark = smooth water, bright = land and buildings", source="Copernicus Sentinel-1 RTC"),
                "Radar image after the flood, the raw evidence behind the flood map.")
        if impact:
            items = sorted(impact.items(), key=lambda kv: -kv[1])
            rep.heading("What was flooded")
            rep.table(["Land cover", "Flooded km²", "Share of flood"],
                      [[D.LULC[k][0], fmt(v), f"{v / max(flood_km2, 1e-9) * 100:.0f}%"] for k, v in items],
                      "Flooded area by land cover", [D.LULC[k][1] for k, _ in items])
            rep.bars([(D.LULC[k][0], v, D.LULC[k][1]) for k, v in items], "Flooded area by land cover (km²).", "km²")
        rep.heading("Flood-prone zones")
        rep.text("HAND is the height of each place above the nearest stream or river level. Land a few metres above drainage floods "
                 f"first. <b>{fmt(prone_km2)} km²</b> lies within 5 m of drainage level."
                 + (f" Here {share_vh * 100:.0f}% of the area is within 2 m of drainage: a flat floodplain where embankments "
                    "and drains, not terrain, decide where water goes." if share_vh > 0.6 else ""))
        rep.map(MapSpec(prone, grid, "Flood-prone zones (height above nearest drainage)", classes=PRONE, subtitle=aoi.name,
                        source="ASF GLO-30 HAND"), "Flood-prone zones from terrain.")
        if optical:
            rep.heading("Independent check with an optical satellite")
            rep.text(f"Sentinel-2 photographed the same area on {optical['date']}. Water was detected in it with the MNDWI index, "
                     "which uses sunlight, not radar. Blue = both satellites agree there is water; yellow = only the optical image "
                     "sees water (usually shallow water among crops, which radar under-detects); red = only radar sees water.")
            rep.map(MapSpec(optical["map"], grid, f"Radar ({post_date}) vs optical ({optical['date']})", classes=AG,
                            subtitle=f"Optical scene {optical['scene']}", source="Copernicus Sentinel-1 and Sentinel-2"),
                    "Pixel-by-pixel comparison of the two satellites.")
        rows = []
        if optical:
            o = optical
            rows.append(("Second satellite (optical)", f"{o['confirmed']:.0f}% of radar water confirmed; agreement {o['agree']:.1f}%, kappa {o['kappa']:.2f}",
                         "Different physics, same answer: strong evidence the water is real. Kappa above 0.6 is substantial agreement."))
        if fa is not None:
            rows.append(("False alarms on dry dates", f"{fa:.2f}% of the area",
                         "The method run between two dry dates; anything flagged there is noise, an honest error floor."))
        if in_low is not None:
            rows.append(("Water on low ground", f"{in_low:.0f}% of flood pixels at HAND ≤ 10 m",
                         "Floods should sit on low ground; a low share would signal errors."))
        if vh_agree is not None:
            rows.append(("Second polarisation (VH)", f"{vh_agree:.0f}% of flood pixels also dark",
                         "A second, independent radar channel agrees."))
        rows.append(("Radar coverage", f"{cover * 100:.0f}% of the area", "Share of the area seen on all dates."))
        rows.append(("Your own check", f"{len(pts)} random points in validation_points.geojson",
                     "Compare with photos or field reports and fill the 'correct' column for a field accuracy."))
        rep.validation(rows)
        rep.limits([
            "Flooding inside dense towns is often missed: buildings and water together bounce radar back brightly.",
            "Water under tree canopy or tall crops can be hidden from radar.",
            "Strong wind roughens water and can make it look like land.",
            f"The map shows water on {post_date} only; the flood may have been larger before the satellite passed.",
            "Flood-prone zones use terrain only; embankments, drains and pumps are not included.",
        ])
        rep.conclusions(points + [f"Overall confidence in the flood map: <b>{verdict['level']}</b>."])
        rep.references([D.CREDIT_S1, D.CREDIT_S2, D.CREDIT_HAND, D.CREDIT_DEM, D.CREDIT_JRC, D.CREDIT_LULC,
                        "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Trans. SMC 9(1), 62–66.",
                        "Nobre, A. D. et al. (2011). Height Above the Nearest Drainage. Journal of Hydrology 404, 13–29."])
        files = [f_flood, f_prone, f_post, f_ref, f_hand, f_csv, f_pts]
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, summary=f"{fmt(flood_km2)} km² flooded on {post_date}.", layers=[
            dict(path=f_pts, name="Validation points", kind="vector"),
            dict(path=f_flood, name=f"Flood extent {post_date}", kind="raster"),
            dict(path=f_prone, name="Flood-prone zones (HAND)", kind="raster", hidden=True),
            dict(path=f_post, name=f"Radar VV {post_date}", kind="raster", hidden=True),
            dict(path=f_ref, name="Radar VV dry reference", kind="raster", hidden=True),
            dict(path=f_hand, name="HAND (m)", kind="raster", hidden=True)])
