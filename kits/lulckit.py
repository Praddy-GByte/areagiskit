# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: LULC: a complete land use / land cover study in one run.

Landsat 1984-today → own classification for each chosen year → accuracy →
change analysis → landscape metrics → urban expansion → spectral indices and LST →
terrain → drivers → ecosystem service value → Markov-CA projection with hindcast →
zonal statistics → report chapter (HTML, Word, PDF) and GIS layers.
"""

import csv
import json
import os

import numpy as np

from ..core import datasets as D
from ..core.aoi import Grid, read_vector
from ..core.data import KitError
from ..core.maps import (MapSpec, render_grouped_bars, render_multi_line, render_polar)
from ..core.raster import write_classes, write_float, write_geojson
from ..core.report import Report, fmt, grade
from .base import Kit
from .lulc import accuracy as AC
from .lulc import change as CH
from .lulc import classify as C
from .lulc import landscape as LS
from .lulc import models as M
from .lulc import extra as EX
from .lulc import annual as AN
from .lulc import compare as CP
from .lulc import userdata as UD
from .lulc.composite import INDEX_INFO, composite, features
from ..core.net import skipped

CL = C.CLASSES
LEI_CLASSES = {1: ("Infilling", "#7a1414"), 2: ("Edge-expansion", "#e0633a"), 3: ("Outlying (leapfrog)", "#f2c14e")}
FREQ = {1: ("Never changed", "#ece9df"), 2: ("Changed once", "#f2c14e"), 3: ("Changed twice", "#e0633a"),
        4: ("Changed 3+ times", "#7a1414")}
HOT = {1: ("Hot spot of change (99%)", "#9e1b16"), 2: ("Hot spot (95%)", "#d6402b"), 3: ("Hot spot (90%)", "#f08a6a"),
       4: ("Not significant", "#ece9df"), 5: ("Cold spot (90%+)", "#4a8cc4")}
PERSIST = {1: ("Persistence", "#d9d6cc"), 2: ("Gain of built-up", "#c4281b"), 3: ("Other change", "#f2c14e")}


def _years(s):
    try:
        ys = sorted({int(x) for x in str(s).replace(";", ",").split(",") if x.strip()})
    except ValueError:
        raise KitError("Years must be numbers separated by commas, e.g. 1995,2005,2015,2025.")
    if len(ys) < 2:
        raise KitError("Give at least two years.")
    if ys[0] < 1985 or ys[-1] > 2026:
        raise KitError("Years must be between 1985 and 2026 (Landsat 5 onwards).")
    return ys


def _months(s):
    s = str(s or "all").strip().lower()
    if s in ("all", ""):
        return list(range(1, 13))
    try:
        m = sorted({int(x) for x in s.split(",") if x.strip()})
        if not m or not all(1 <= v <= 12 for v in m):
            raise ValueError("months must be numbers from 1 to 12")
        return m
    except Exception:
        raise KitError("Months must be 'all' or numbers 1–12 separated by commas.")


class LULCStudyKit(Kit):
    NAME = "AreaGIS Kit: LULC"
    KEY = "lulc"
    TAGLINE = "Complete land use / land cover study, 1985–today"
    DESCRIPTION = ("A full LULC study in one run: Landsat classification for each chosen year (auto-trained Random Forest), "
                   "accuracy assessment, transition and intensity analysis, gains/losses, landscape metrics, urban "
                   "expansion (rate, direction, gradient, expansion type), spectral indices and land surface temperature, "
                   "terrain zones, drivers of urban growth, ecosystem service value, Markov-CA projection with hindcast "
                   "validation, optional village/ward statistics, and a full report chapter.")
    MAX_KM2 = 1500
    PARAMS = [
        dict(name="years", type="text", label="Years to classify (comma list)", default="1995,2005,2015,2025",
             help="Equal intervals give the best projection, e.g. 1995,2005,2015,2025."),
        dict(name="months", type="text", label="Months to use ('all' or e.g. 1,2,3,10,11,12)", default="all"),
        dict(name="max_cloud", type="int", label="Skip scenes with more cloud than (%)", default=60),
        dict(name="project", type="enum", label="Project future land cover", default="yes", options=["yes", "no"]),
        dict(name="zones", type="polygons", label="Zones for tables (villages, wards…) (optional)", optional=True),
        dict(name="zone_field", type="text", label="Zone name field", default="name"),
        dict(name="reference_points", type="points", label="Your labelled reference points (optional)", optional=True,
             help="Points with fields 'year' and 'reference_class' (1–8, see legend). Gives area-adjusted accuracy."),
        dict(name="training_points", type="points", label="Your own training points and classes (optional)", optional=True,
             help="Points with a class name field (e.g. 'Fallow land', 'Plantation', 'Mining') and optionally 'year' and "
                  "'level1' (1–8). Replaces automatic training; your own classes are mapped as Level-II."),
        dict(name="class_field", type="text", label="Class name field in training points", default="class"),
        dict(name="annual_trend", type="enum", label="Every-year trend analysis (adds ~5 min)", default="yes", options=["yes", "no"]),
        dict(name="texture", type="enum", label="Extra texture features (GLCM-like contrast/homogeneity)", default="no", options=["yes", "no"]),
        dict(name="storm_mm", type="float", label="Design storm rainfall for runoff (mm)", default=100.0),
        dict(name="plan", type="polygons", label="Master plan / zoning layer (optional)", optional=True),
        dict(name="plan_field", type="text", label="Planned land use field", default="landuse"),
        dict(name="census", type="polygons", label="Census / statistics units (optional)", optional=True),
        dict(name="census_id", type="text", label="Unit name field", default="name"),
        dict(name="census_fields", type="text", label="Numeric fields to relate (comma list; empty = all numeric)", default=""),
    ]

    # ================================================================== run
    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        years = _years(params.get("years", "1995,2005,2015,2025"))
        months = _months(params.get("months", "all"))
        maxc = int(params.get("max_cloud") or 60)
        grid = Grid(aoi, 30, max_pixels=2_500_000)
        px = grid.px_km2
        valid_km2 = grid.mask.sum() * px
        R = dict(years=years, months=months, grid=grid)

        self.step(2, "Reading terrain")
        elev = D.dem(grid)
        slope, _ = D.slope_aspect(elev, grid)
        user = None
        if params.get("training_points"):
            self.step(4, "Reading your training points")
            user = UD.read_training(params["training_points"], aoi, grid, params.get("class_field") or "class")
            classes_l2, upts = user
            lab, ref_agree, io30 = None, None, None
            all_years = list(years)
        else:
            self.step(4, "Building training labels from two independent 2021 maps")
            lab, ref_agree, io30 = C.reference_labels(grid)
            if (lab > 0).sum() < 500:
                raise KitError("Too few reliable training pixels in this area. Try a larger area.")
            all_years = sorted(set(years) | {C.REF_YEAR})

        # ---------------------------------------------------- composites
        comps, feats = {}, {}
        for i, y in enumerate(all_years):
            base = 6 + 54 * i / len(all_years)
            span = 54 / len(all_years)
            comps[y] = composite(aoi, grid, y, months,
                                 progress=lambda f, m, b=base, s=span: self.step(b + s * f, m), max_cloud=maxc)
            f, names, ix = features(comps[y], elev, slope, extended_texture=str(params.get("texture", "no")).lower() == "yes")
            feats[y] = (f, ix)
        fnames = names

        # ---------------------------------------------------- classification
        res, ghsl = {}, {}
        latest = years[-1]
        jrc = D.water_occurrence(grid)
        if user:
            def near(y0_):
                return min(years, key=lambda yy: abs(yy - y0_))
            upts = [(r_, c_, k_, near(y_) if y_ is not None else None) for r_, c_, k_, y_ in upts]
            pyears = UD.point_years(upts, latest)
            built_l2 = [k for k, v in classes_l2.items() if v[2] == 2]
        for i, y in enumerate(all_years):
            self.step(61 + 9 * i / len(all_years), f"Classifying {y}")
            f, ix = feats[y]
            gh, ep = C.ghsl_fraction(grid, y) if y <= 2020 else (None, None)
            ghsl[y] = (gh, ep)
            if user:
                src = min(pyears, key=lambda yy: abs(yy - y))
                ly = UD.label_raster(upts, grid.shape, src, latest)
                where = np.ones(grid.shape, bool) if src == y else C.relaxed_mask(ix, feats[src][1], 1.0)
                r = C.classify_year(f, ly, where, grid, seed=y, min_class=5, min_total=30, keep_samples=(y == latest))
                r["notes"] = [] if src == y else [(0, f"training points from {src} moved to {y} where land looks unchanged")]
                r["map"] = C.majority(r["map"], grid.mask)
                r["ghsl_fixed"] = C.ghsl_constraint(r, gh, built_l2, ix["NDVI_p90"]) if (gh is not None and y < latest) else 0
                r["stable_share"] = float(where[ly > 0].mean()) if (ly > 0).any() else 0.0
                r["user_points"] = int((ly > 0).sum())
            else:
                stable = np.ones(grid.shape, bool) if y == C.REF_YEAR else C.stable_mask(ix, feats[C.REF_YEAR][1])
                ly, where = C.year_labels(lab, stable, gh, y)
                notes = [] if y == C.REF_YEAR else C.fill_missing(
                    ly, where, np.where((lab == 2) & ~(np.nan_to_num(gh) >= 0.2), 0, lab) if gh is not None else lab,
                    C.relaxed_mask(ix, feats[C.REF_YEAR][1]), ix, jrc_occ=jrc)
                r = C.classify_year(f, ly, where, grid, seed=y, keep_samples=(y == latest))
                r["notes"] = notes
                r["map"] = C.majority(r["map"], grid.mask)
                r["ghsl_fixed"] = C.ghsl_constraint(r, gh, (2,), ix["NDVI_p90"]) if (gh is not None and y < C.REF_YEAR) else 0
                r["stable_share"] = float(stable[lab > 0].mean())
            r["water_fixed"] = C.water_constraint(r, jrc, ix["MNDWI"], y)
            res[y] = r
        if user:
            r1 = {}
            for y in all_years:
                par, pr = UD.l1_proba(res[y], classes_l2)
                r1[y] = dict(map=UD.to_level1(res[y]["map"], classes_l2), proba=pr, classes=par)
            reversal = C.no_urban_reversal(r1, all_years)
            for y in all_years:
                UD.resync_l2(res[y], r1[y]["map"], classes_l2)
            maps_l2 = {y: res[y]["map"].copy() for y in years}
            maps = {y: UD.to_level1(res[y]["map"], classes_l2) for y in years}
        else:
            reversal = C.no_urban_reversal(res, all_years)
            maps_l2 = None
            maps = {y: res[y]["map"] for y in years}
        CLX = dict(CL)
        if user:
            CLX.update({k: (v[0], v[1]) for k, v in classes_l2.items()})
        samples = res[latest].pop("samples", None)
        for y in all_years:
            feats[y] = (None, feats[y][1])  # free feature memory, keep indices
        present = [c for c in CL if any((maps[y] == c).any() for y in years)]

        # ---------------------------------------------------- accuracy
        self.step(71, "Assessing accuracy")
        acc = {}
        for y in all_years:
            yt, yp = res[y]["test"]
            m = AC.confusion(yt, yp, res[y]["classes"])
            acc[y] = dict(classes=res[y]["classes"], matrix=m, **AC.metrics(m), oob=res[y]["oob"], counts=res[y]["counts"])
        wc = None
        try:
            from ..core.data import search, warp_items
            from osgeo import gdal
            items = search("esa-worldcover", aoi.bbox, "2021-01-01", "2021-12-31")
            raw = warp_items(items, "map", grid, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
            wc = C._lut(C.WC_TO)[raw]
            wc[~grid.mask] = 0
        except Exception as exc:
            skipped(exc)
        map_checks = []
        ref_y = C.REF_YEAR if C.REF_YEAR in res else min(years, key=lambda yy: abs(yy - C.REF_YEAR))
        m21 = maps[ref_y] if ref_y in maps else res[ref_y]["map"]
        if wc is not None:
            a = AC.agreement(m21, wc, present)
            if a:
                map_checks.append(("ESA WorldCover 2021", "2021", a))
        if io30 is not None:
            a = AC.agreement(m21, io30, present)
            if a:
                map_checks.append(("IO/Esri land cover 2021", "2021", a))
        cci_checks = []
        for y in years:
            if y <= 2020 or y < 1992:
                cci, cy = C.cci_map(grid, y)
                if cci is None:
                    continue
                blk = 10
                rr, cc = grid.rows // blk, grid.cols // blk
                ours = maps[y][:rr * blk, :cc * blk].reshape(rr, blk, cc, blk).transpose(0, 2, 1, 3).reshape(rr, cc, -1)
                mode = np.array(list(CL), np.uint8)[np.stack([(ours == k).sum(-1) for k in CL], -1).argmax(-1)]
                ref = cci[blk // 2:rr * blk:blk, blk // 2:cc * blk:blk][:rr, :cc]
                v = (ref > 0) & (mode > 0)
                if v.sum() > 20:
                    cci_checks.append((y, cy, float((ref[v] == mode[v]).mean())))
        ghsl_checks = []
        for y in years:
            gh, ep = ghsl[y] if y in ghsl else (None, None)
            if gh is None:
                continue
            dense = np.nan_to_num(gh) >= 0.2
            b = maps[y] == 2
            ghsl_checks.append((y, ep, b.sum() * px, dense.sum() * px,
                                (b & dense).sum() / max(dense.sum(), 1) * 100))
        ref_acc = None
        if params.get("reference_points"):
            ref_acc = self._reference_accuracy(params["reference_points"], aoi, grid, maps, px)
        design = AC.sample_design(maps[latest], grid.mask, present)
        f_design = write_geojson(os.path.join(out_dir, f"accuracy_sample_{latest}.geojson"),
                                 [(grid.pixel_to_lonlat(*divmod(i, grid.cols)),
                                   {"id": n + 1, "year": latest, "map_class": int(c), "map_name": CL[c][0],
                                    "reference_class": None}) for n, (i, c) in enumerate(design)])

        # ---------------------------------------------------- areas and change
        self.step(74, "Analysing change")
        areas = {y: {c: float((maps[y] == c).sum() * px) for c in present} for y in years}
        Ts = [CH.transition(maps[a], maps[b], present, px) for a, b in zip(years[:-1], years[1:])]
        T_all = CH.transition(maps[years[0]], maps[years[-1]], present, px)
        comp = [CH.components(T) for T in Ts]
        comp_all = CH.components(T_all)
        inten = CH.intensity(Ts, years) if len(Ts) >= 1 else None
        bi = present.index(2) if 2 in present else None
        trans_int = [CH.transition_intensity(T, d, bi) for T, d in zip(Ts, inten["durations"])] if bi is not None else []
        freq = CH.frequency([maps[y] for y in years])
        anyv = np.all([maps[y] > 0 for y in years], axis=0)
        cell = max(3, int(round(1000 / grid.res)))
        share = CH.grid_change(maps[years[0]] != maps[years[-1]], anyv, cell)
        z = CH.gi_star_grid(share)
        hot_grid = np.full(share.shape, 0, np.uint8)
        hot_grid[np.isfinite(z)] = 4
        hot_grid[z >= 1.645] = 3
        hot_grid[z >= 1.96] = 2
        hot_grid[z >= 2.576] = 1
        hot_grid[z <= -1.645] = 5
        hot = np.zeros(grid.shape, np.uint8)
        hot[:hot_grid.shape[0] * cell, :hot_grid.shape[1] * cell] = np.kron(hot_grid, np.ones((cell, cell), np.uint8))
        hot[~grid.mask] = 0
        persist = np.zeros(grid.shape, np.uint8)
        persist[anyv] = 1
        persist[anyv & (maps[years[0]] != maps[years[-1]])] = 3
        persist[anyv & (maps[years[0]] != 2) & (maps[years[-1]] == 2)] = 2

        # ---------------------------------------------------- landscape metrics
        self.step(78, "Computing landscape metrics")
        lm = {y: LS.metrics(maps[y], grid, present) for y in years}

        # ---------------------------------------------------- urban expansion
        self.step(81, "Analysing urban expansion")
        urb = []
        ctr = LS.centre(maps[years[0]] == 2, grid)
        sect, lei_maps = [], []
        for a, b in zip(years[:-1], years[1:]):
            b1, b2 = maps[a] == 2, maps[b] == 2
            urb.append(dict(period=f"{a}–{b}", **LS.urban_metrics(b1, b2, a, b, valid_km2, px)))
            new = b2 & ~b1
            sect.append(LS.sectors(new, ctr, px))
            tmap, tarea = LS.lei(b1, new, grid)
            lei_maps.append((f"{a}–{b}", tmap, tarea))
        ring_km = max(0.5, round(np.sqrt(valid_km2) / 12, 1))
        grad = LS.rings([maps[y] == 2 for y in years], ctr, grid, ring_km)

        # ---------------------------------------------------- indices and LST
        self.step(84, "Summarising spectral indices and land surface temperature")
        idx_tab, lst_tab, lst_corr = {}, {}, {}
        for y in years:
            ix = feats[y][1]
            idx_tab[y] = {k: {c: float(np.nanmean(ix[k][maps[y] == c])) if (maps[y] == c).any() else None for c in present}
                          for k in INDEX_INFO}
            lst = comps[y]["lst"]
            lst_tab[y] = {c: float(np.nanmean(lst[maps[y] == c])) if np.isfinite(lst[maps[y] == c]).any() else None
                          for c in present}
            lst_corr[y] = (M.pearson(lst, ix["NDVI"], grid.mask), M.pearson(lst, ix["NDBI"], grid.mask))
        uhi = {}
        for y in years:
            t = lst_tab[y]
            green = [t[c] for c in (3, 4, 5) if c in t and t[c] is not None]
            if t.get(2) is not None and green:
                uhi[y] = t[2] - float(np.mean(green))

        # ---------------------------------------------------- terrain
        ez, elabs = LS.elevation_zones(elev, grid.mask)
        sz, slabs = LS.slope_zones(slope, grid.mask)
        terr_e = LS.by_zones(maps[latest], ez, elabs, present, px)
        terr_s = LS.by_zones(maps[latest], sz, slabs, present, px)
        gain_built = (maps[years[0]] != 2) & (maps[latest] == 2)
        gain_by_slope = [(lab_, float((gain_built & (sz == z_)).sum() * px)) for z_, lab_ in slabs]

        # ---------------------------------------------------- drivers
        self.step(87, "Modelling drivers of urban growth")

        def driver_layers(t1):
            built1 = maps[t1] == 2 if t1 in maps else res[t1]["map"] == 2
            water1 = (maps[t1] if t1 in maps else res[t1]["map"]) == 1
            rr, cc = np.indices(grid.shape)
            return {
                "Distance to existing built-up (m)": grid.distance_to(built1),
                "Distance to water (m)": grid.distance_to(water1) if water1.any() else np.zeros(grid.shape),
                "Distance to centre (m)": np.hypot((rr - ctr[0]) * grid.res, (cc - ctr[1]) * grid.res),
                "Built-up density within 450 m": M.density(built1, 7),
                "Elevation (m)": np.nan_to_num(elev, nan=np.nanmean(elev)),
                "Slope (°)": np.nan_to_num(slope),
            }
        drv = M.drivers(maps[years[-2]] == 2, maps[years[-1]] == 2, grid.mask & (maps[years[-1]] > 0),
                        driver_layers(years[-2]))

        # ---------------------------------------------------- projection
        proj, skill = {}, None
        if params.get("project", "yes") == "yes" and len(years) >= 2:
            self.step(90, "Projecting future land cover (Markov + cellular automata)")
            L = years[-1] - years[-2]
            P = M.markov(Ts[-1])
            suit = drv["suitability"] if drv else None
            if len(years) >= 3:
                Pp = M.markov(Ts[-2])
                d_prev = M.drivers(maps[years[-3]] == 2, maps[years[-2]] == 2, grid.mask, driver_layers(years[-3]))
                steps = max(1, round((years[-1] - years[-2]) / (years[-2] - years[-3])))
                sim = M.ca_allocate(maps[years[-2]], Pp, present, d_prev["suitability"] if d_prev else None, steps)
                skill = M.simulation_skill(maps[years[-2]], maps[years[-1]], sim, present)
                skill["years"] = (years[-3], years[-2], years[-1])
            R["proj_stopped"] = None
            if skill and (skill["fom"] < 0.1 or skill["oa"] <= skill["null_oa"]):
                R["proj_stopped"] = (f"The projection method failed its own hindcast ({skill['years'][1]}→{skill['years'][2]}: figure of merit "
                                     f"{skill['fom']:.2f}, agreement {skill['oa'] * 100:.0f}% vs {skill['null_oa'] * 100:.0f}% for a no-change "
                                     f"model), so no future map is produced for this area.")
            for s in (() if R["proj_stopped"] else (1, 2)):
                fy = years[-1] + s * L
                pm = M.ca_allocate(maps[years[-1]], P, present, suit, s, seed=fy)
                pm[~grid.mask] = 0
                proj[fy] = pm
            R["P"] = P

        # ---------------------------------------------------- ecosystem services
        lat = aoi.center[1]
        esv = {y: M.esv(areas[y], lat) for y in years}
        for fy, pm in proj.items():
            esv[fy] = M.esv({c: float((pm == c).sum() * px) for c in present}, lat)
        esv_sens = M.esv_sensitivity(areas[latest], lat)

        # ---------------------------------------------------- zones
        zonal = None
        if params.get("zones"):
            zonal = self._zones(params["zones"], params.get("zone_field") or "name", aoi, grid, maps, years, px)

        # ---------------------------------------------------- extra analyses
        X = _extras(self, dict(locals()), params, out_dir)

        # ---------------------------------------------------- files
        self.step(97, "Writing maps and tables")
        files, layers = [], []
        for y in years:
            p = write_classes(os.path.join(out_dir, f"lulc_{y}.tif"), maps[y], grid, {c: CL[c] for c in present})
            files.append(p)
            layers.append(dict(path=p, name=f"LULC {y}", kind="raster", hidden=y != latest))
            pc, _ = write_float(os.path.join(out_dir, f"confidence_{y}.tif"), res[y]["conf"], grid, "ndvi", 0.3, 1.0)
            files.append(pc)
            pl, _ = write_float(os.path.join(out_dir, f"lst_{y}.tif"), comps[y]["lst"], grid, "heat",
                                float(np.nanpercentile(comps[y]["lst"], 2)), float(np.nanpercentile(comps[y]["lst"], 98)), "°C")
            files.append(pl)
            for k in ("NDVI", "NDBI", "MNDWI"):
                pi, _ = write_float(os.path.join(out_dir, f"{k.lower()}_{y}.tif"), feats[y][1][k], grid,
                                    "ndvi" if k == "NDVI" else ("heat" if k == "NDBI" else "depth"), -0.5, 0.8)
                files.append(pi)
        fz = np.zeros(grid.shape, np.uint8)
        fz[anyv] = 1 + np.minimum(freq[anyv], 3)
        for name, arr, cls in (("change_frequency", fz, FREQ), ("change_hotspots", hot, HOT), ("persistence_gain", persist, PERSIST)):
            p = write_classes(os.path.join(out_dir, f"{name}.tif"), arr, grid, cls)
            files.append(p)
            layers.append(dict(path=p, name=name.replace("_", " ").title(), kind="raster", hidden=True))
        for per, tmap, _ in lei_maps:
            p = write_classes(os.path.join(out_dir, f"urban_expansion_type_{per.replace('–', '_')}.tif"), tmap, grid, LEI_CLASSES)
            files.append(p)
        for fy, pm in proj.items():
            p = write_classes(os.path.join(out_dir, f"projected_lulc_{fy}.tif"), pm, grid, {c: CL[c] for c in present})
            files.append(p)
            layers.append(dict(path=p, name=f"Projected LULC {fy}", kind="raster", hidden=True))
        if drv:
            p, _ = write_float(os.path.join(out_dir, "urban_growth_suitability.tif"), drv["suitability"], grid, "risk", 0, 1)
            files.append(p)
        files += self._tables(out_dir, years, present, areas, Ts, T_all, comp, comp_all, inten, lm, urb, lei_maps, sect,
                              idx_tab, lst_tab, esv, acc, drv, zonal, proj, px)
        files.append(f_design)
        files += X["files"]
        layers += X["layers"]
        layers.insert(0, dict(path=f_design, name=f"Accuracy sample {latest} (label me)", kind="vector", hidden=True))

        # ---------------------------------------------------- validation grade
        checks = []
        mean_oa = float(np.mean([acc[y]["oa"] for y in years]))
        if user:
            checks.append((3 if mean_oa >= 0.85 else 2 if mean_oa >= 0.75 else 0, 3,
                           f"Accuracy on your own held-out training points averages {mean_oa * 100:.0f}% across the years."))
        else:
            checks.append((1 if mean_oa >= 0.85 else 0, 1,
                           f"Held-out accuracy on automatic labels averages {mean_oa * 100:.0f}% across the {len(years)} years "
                           f"(optimistic: test labels come from the same two maps as the training labels)."))
        if map_checks:
            o = map_checks[0][2]["oa"]
            checks.append((1 if o >= 0.7 else 0, 1,
                           f"The 2021 classification agrees {o * 100:.0f}% with ESA WorldCover (not independent: WorldCover is one of "
                           f"the two maps the training labels come from)."))
        if ghsl_checks:
            rec = float(np.mean([g[4] for g in ghsl_checks]))
            checks.append((1 if rec >= 85 else 0, 1,
                           f"Built-up maps contain {rec:.0f}% of the dense built-up land in GHSL on average (partly dependent: GHSL also screens built-up before 2021)."))
        if skill:
            checks.append((1 if skill["fom"] >= 0.1 else 0, 1,
                           f"Projection hindcast {skill['years'][1]}→{skill['years'][2]}: figure of merit {skill['fom']:.2f}, "
                           f"agreement {skill['oa'] * 100:.0f}% (no-change model {skill['null_oa'] * 100:.0f}%)."))
        cov = float(np.mean([(maps[y] > 0)[grid.mask].mean() for y in years]) * 100)
        checks.append((1 if cov >= 95 else 0, 1, f"{cov:.1f}% of the area was classified in every year."))
        if ref_acc:
            checks.append((3 if ref_acc["oa"] >= 0.85 else 2 if ref_acc["oa"] >= 0.75 else 0, 3,
                           f"Against your {ref_acc['n']} reference points, area-adjusted overall accuracy is {ref_acc['oa'] * 100:.0f}%."))
        verdict = grade(checks)

        # ---------------------------------------------------- report
        self.step(98, "Writing the report")
        R.update(locals())
        docs = self._report(R, out_dir, files)
        self.step(100, "Done")
        b0, b1 = areas[years[0]].get(2, 0), areas[latest].get(2, 0)
        return dict(report=docs["html"], documents=docs, verdict=verdict, layers=layers,
                    summary=f"Built-up {fmt(b0)} → {fmt(b1)} km² ({years[0]}–{latest}); hold-out accuracy {mean_oa * 100:.0f}% (optimistic; see the report).")

    # ================================================================== helpers
    def _reference_accuracy(self, path, aoi, grid, maps, px):
        feats = read_vector(path, aoi)
        by_year = {}
        for gl, attrs, gw in feats:
            try:
                y = int(attrs.get("year"))
                ref = int(attrs.get("reference_class"))
            except (TypeError, ValueError):
                continue
            p = gw.Centroid()
            r, c = grid.lonlat_to_pixel(p.GetX(), p.GetY())
            if y in maps and 0 <= r < grid.rows and 0 <= c < grid.cols and maps[y][r, c] > 0:
                by_year.setdefault(y, []).append((int(maps[y][r, c]), ref))
        if not by_year:
            return None
        y = max(by_year, key=lambda k: len(by_year[k]))
        pairs = by_year[y]
        classes = sorted(set(CL) & ({p for p, _ in pairs} | {r for _, r in pairs}))
        m = AC.confusion([r for _, r in pairs], [p for p, _ in pairs], classes)
        area_px = np.array([(maps[y] == c).sum() for c in classes], float)
        o = AC.olofsson(m, area_px, px)
        o.update(dict(year=y, n=len(pairs), classes=classes, matrix=m, kappa=AC.metrics(m)["kappa"]))
        return o

    def _zones(self, path, field, aoi, grid, maps, years, px):
        feats = read_vector(path, aoi)
        if not feats:
            return None
        names = [str(a.get(field, i + 1)) for i, (_, a, _) in enumerate(feats)]
        zid = grid.rasterize([g for g, _, _ in feats], values=list(range(1, len(feats) + 1)))
        rows = []
        for i, n in enumerate(names, start=1):
            sel = zid == i
            if not sel.any():
                continue
            r = {"zone": n, "area_km2": float(sel.sum() * px)}
            for y in years:
                r[f"built_{y}"] = float(((maps[y] == 2) & sel).sum() * px)
            r["built_change_km2"] = r[f"built_{years[-1]}"] - r[f"built_{years[0]}"]
            r["built_share_latest_pct"] = r[f"built_{years[-1]}"] / max(r["area_km2"], 1e-9) * 100
            rows.append(r)
        val = np.zeros(grid.shape, np.uint8)
        q = np.percentile([r["built_change_km2"] / max(r["area_km2"], 1e-9) for r in rows], [20, 40, 60, 80]) if rows else []
        for i, n in enumerate(names, start=1):
            rr = [r for r in rows if r["zone"] == n]
            if rr:
                v = rr[0]["built_change_km2"] / max(rr[0]["area_km2"], 1e-9)
                val[zid == i] = 1 + int(np.digitize(v, q))
        return dict(rows=rows, map=val)

    def _tables(self, out, years, present, areas, Ts, T_all, comp, comp_all, inten, lm, urb, lei_maps, sect, idx_tab,
                lst_tab, esv, acc, drv, zonal, proj, px):
        files = []

        def w(name, header, rows):
            p = os.path.join(out, name)
            with open(p, "w", newline="", encoding="utf-8") as fh:
                c = csv.writer(fh)
                c.writerow(header)
                c.writerows(rows)
            files.append(p)

        nm = [CL[c][0] for c in present]
        w("class_areas_km2.csv", ["class"] + [str(y) for y in years],
          [[CL[c][0]] + [round(areas[y][c], 4) for y in years] for c in present])
        for (a, b), T in zip(zip(years[:-1], years[1:]), Ts):
            w(f"transition_{a}_{b}_km2.csv", [f"from \\ to"] + nm, [[nm[i]] + list(np.round(T[i], 4)) for i in range(len(nm))])
        w(f"transition_{years[0]}_{years[-1]}_km2.csv", ["from \\ to"] + nm, [[nm[i]] + list(np.round(T_all[i], 4)) for i in range(len(nm))])
        w("gain_loss_net_swap_km2.csv", ["period", "class", "gain", "loss", "net", "swap", "total_change"],
          [[f"{a}–{b}", nm[i], *(round(float(cp[k][i]), 4) for k in ("gain", "loss", "net", "swap", "total"))]
           for (a, b), cp in zip(zip(years[:-1], years[1:]), comp) for i in range(len(nm))])
        w("intensity_analysis.csv", ["period", "level", "class", "gain_intensity_pct_per_yr", "loss_intensity_pct_per_yr", "uniform"],
          [[f"{years[i]}–{years[i + 1]}", "category", nm[j], round(float(c["gain"][j]), 4), round(float(c["loss"][j]), 4),
            round(c["uniform"], 4)] for i, c in enumerate(inten["category"]) for j in range(len(nm))]
          + [[f"{years[i]}–{years[i + 1]}", "interval", "all", round(s, 4), "", round(inten["uniform"], 4)] for i, s in enumerate(inten["interval"])])
        keys = ["PLAND", "NP", "PD", "LPI", "MPS", "ED", "MSI", "AI"]
        w("landscape_metrics_class.csv", ["year", "class"] + keys,
          [[y, CL[c][0]] + [round(lm[y][0][c][k], 4) for k in keys] for y in years for c in present if c in lm[y][0]])
        lk = ["NP", "PD", "LPI", "ED", "SHDI", "SHEI", "CONTAG", "AI"]
        w("landscape_metrics_landscape.csv", ["year"] + lk, [[y] + [round(lm[y][1][k], 4) for k in lk] for y in years])
        w("urban_expansion.csv", ["period", "built_start_km2", "built_end_km2", "growth_km2", "annual_growth_rate_pct", "UEII_pct",
                                  "infilling_km2", "edge_expansion_km2", "outlying_km2"] + [f"sector_{s}_km2" for s in LS.SECTORS],
          [[u["period"], round(u["u1"], 4), round(u["u2"], 4), round(u["growth"], 4),
            "" if u["agr"] is None else round(u["agr"], 3), round(u["ueii"], 4),
            round(l[2][1], 4), round(l[2][2], 4), round(l[2][3], 4)] + [round(float(v), 4) for v in s]
           for u, l, s in zip(urb, lei_maps, sect)])
        w("spectral_indices_by_class.csv", ["year", "index", "class", "mean"],
          [[y, k, CL[c][0], "" if idx_tab[y][k][c] is None else round(idx_tab[y][k][c], 4)]
           for y in years for k in INDEX_INFO for c in present])
        w("lst_by_class_c.csv", ["year", "class", "mean_lst_c"],
          [[y, CL[c][0], "" if lst_tab[y][c] is None else round(lst_tab[y][c], 3)] for y in years for c in present])
        w("ecosystem_service_value_usd.csv", ["year", "total_usd_per_yr"] + [CL[c][0] for c in present],
          [[y, round(esv[y][0])] + [round(esv[y][1].get(c, 0)) for c in present] for y in esv])
        w("accuracy_holdout.csv", ["year", "overall_accuracy", "kappa", "oob_accuracy", "n_test"],
          [[y, round(acc[y]["oa"], 4), round(acc[y]["kappa"], 4), "" if acc[y]["oob"] is None else round(acc[y]["oob"], 4), acc[y]["n"]]
           for y in acc])
        if drv:
            w("urban_growth_drivers.csv", ["factor", "coefficient_per_sd", "odds_ratio_per_sd", "z", "p"],
              [[r["name"], round(r["coef"], 4), round(r["odds"], 4), round(r["z"], 3), f"{r['p']:.2e}"] for r in drv["rows"]])
        if zonal and zonal["rows"]:
            hdr = list(zonal["rows"][0].keys())
            w("zones_built_up.csv", hdr, [[r[h] if not isinstance(r[h], float) else round(r[h], 4) for h in hdr] for r in zonal["rows"]])
        if proj:
            w("projected_areas_km2.csv", ["class"] + [str(y) for y in proj],
              [[CL[c][0]] + [round(float((proj[y] == c).sum() * px), 4) for y in proj] for c in present])
        return files

    # ================================================================== report
    def _report(self, R, out_dir, files):
        g = R["grid"]
        aoi = g.aoi
        years, present, areas, latest = R["years"], R["present"], R["areas"], R["latest"]
        y0 = years[0]
        nm = {c: CL[c][0] for c in present}
        col = {c: CL[c][1] for c in present}
        src = "Landsat Collection 2 (USGS) via Microsoft Planetary Computer; own classification"
        rep = Report(self.NAME, f"Land use / land cover change {y0}–{latest}", aoi.name, R["verdict"], "#2f5d50", aoi, g)
        classes_all = {c: CL[c] for c in present}
        missing = {(y, k) for y in years for k, how in R["res"][y].get("notes", []) if how.startswith("not")}
        def A(y, c):
            return "n/m" if (y, c) in missing else fmt(areas[y][c])

        # ------------------------------------------------ summary
        d = {c: areas[latest][c] - areas[y0][c] for c in present}
        tot = sum(areas[y0].values()) or 1
        key = [("Period", f"{y0}–{latest} ({len(years)} dates)")]
        for c in (2, 3, 4, 1):
            if c in present and (y0, c) not in missing and (latest, c) not in missing:
                key.append((nm[c], f"{fmt(areas[y0][c])} → {fmt(areas[latest][c])} km² ({d[c]:+,.1f})"))
        key.append(("Mean classification accuracy", f"{R['mean_oa'] * 100:.0f}% (hold-out on " + ("your points)" if R.get("user") else
                    "automatic labels; optimistic, not an independent test)")))
        if R["proj"]:
            fy = max(R["proj"])
            key.append((f"Projected built-up {fy}", f"{fmt(float((R['proj'][fy] == 2).sum() * g.px_km2))} km²"))
        key.append(("Ecosystem service value", f"US$ {R['esv'][y0][0] / 1e6:,.1f} M → {R['esv'][latest][0] / 1e6:,.1f} M per year"))
        changed = R["comp_all"]["total"].sum() / 2
        pts = [f"<b>{fmt(changed)} km²</b> ({changed / tot * 100:.0f}% of the area) changed class between {y0} and {latest}."]
        big = sorted(present, key=lambda c: -abs(d[c]))[:3]
        for c in big:
            if abs(d[c]) > 0.01 and (y0, c) not in missing and (latest, c) not in missing:
                r = CH.annual_rate(areas[y0][c], areas[latest][c], latest - y0)
                pts.append(f"<b>{nm[c]}</b> {'grew' if d[c] > 0 else 'shrank'} by <b>{fmt(abs(d[c]))} km²</b>"
                           + (f" ({r:+.2f}% per year)." if r is not None else "."))
        T = R["T_all"]
        off = [(T[i, j], i, j) for i in range(len(present)) for j in range(len(present)) if i != j]
        off.sort(reverse=True)
        if off:
            v, i, j = off[0]
            pts.append(f"Largest conversion: <b>{nm[present[i]]} → {nm[present[j]]}</b>, {fmt(v)} km².")
        if R["uhi"]:
            yy = max(R["uhi"])
            pts.append(f"Surface heat contrast, built-up minus vegetated land, {yy}: <b>{R['uhi'][yy]:+.1f} °C</b>.")
        rep.summary(key, pts, MapSpec(R["maps"][latest], g, f"Land use / land cover {latest}", classes=classes_all,
                                      subtitle=aoi.name, source=src))

        rep.study_area(MapSpec(np.where(g.mask, R["elev"], np.nan), g, "Study area: elevation", ramp="elev",
                               vmin=float(np.nanpercentile(R["elev"], 2)), vmax=float(np.nanpercentile(R["elev"], 98)),
                               unit="m", subtitle=aoi.name, source="Copernicus DEM GLO-30"),
                       [("Years analysed", ", ".join(map(str, years))),
                        ("Months used", "all" if len(R["months"]) == 12 else ",".join(map(str, R["months"]))),
                        ("Classes", ", ".join(nm[c] for c in present))])

        # ------------------------------------------------ data
        rows = []
        for y in R["all_years"]:
            c = R["comps"][y]
            rows.append((f"Landsat {', '.join(p.replace('landsat-', '') for p in c['platforms'])} C2 L2", f"Composite {y}"
                         + (" (training reference)" if y == C.REF_YEAR and y not in years else ""), "30 m",
                         f"{len(c['dates'])} dates, {c['dates'][0]} to {c['dates'][-1]}", "Public domain (USGS)"))
        if R["user"]:
            rows += [("Your training points", f"Training for {len(R['classes_l2'])} classes", "points", "as supplied", "yours")]
        else:
            rows += [("IO/Esri annual land cover 2021", "Training labels (where it agrees with WorldCover)", "10 m", "2021", "CC BY 4.0")]
        rows += [("ESA WorldCover 2021", ("Independent check" if R["user"] else "Training labels (so not an independent check)"), "10 m", "2021", "CC BY 4.0"),
                 ("JRC GHSL built-up surface", "Historic built-up evidence and check", "100 m", ", ".join(
                     str(R["ghsl"][y][1]) for y in years if R["ghsl"].get(y, (None, None))[1]), "CC BY 4.0"),
                 ("ESA CCI land cover", "Independent check of older years", "300 m", "1992–2020", "ESA CCI licence"),
                 ("Copernicus DEM GLO-30", "Elevation, slope features and terrain zones", "30 m", "2011–2015", "Copernicus DEM licence")]
        rep.data(rows + _extra_data_rows(R))

        # ------------------------------------------------ method
        rep.method([
            ("Yearly composites", "For each year, up to 16 Landsat dates (the clearest of each month first) are read for the area only. "
             "Clouds, shadows and snow are removed with the QA band; Landsat 5/7 reflectance is harmonised to Landsat 8/9 "
             "(Roy et al. 2016); the per-pixel median is taken. The 10th and 90th percentile NDVI capture the seasonal cycle."),
            ("Features", "Six reflectance bands, NDVI, MNDWI, NDBI, BSI, NDMI, SAVI, seasonal NDVI percentiles, seasonal water "
             "(MNDWI 90th percentile), NIR texture, elevation and slope."),
            ("Training labels (your points)", "Only your training points are used. Points without a year belong to the latest "
             "year; other years use points whose spectral behaviour is unchanged.") if R["user"] else
            ("Training labels without fieldwork", "Labels come from 30 m cells where two independent 2021 maps (IO/Esri and ESA "
             "WorldCover) agree on at least 7 of 9 sub-pixels. For other years only pixels whose spectral behaviour is unchanged "
             "from 2021 keep their label. Built-up labels for years before 2021 are kept only where the JRC GHSL shows buildings in "
             "that epoch, because bare dry land looks like built-up in Landsat."),
            ("Classification", "A Random Forest (40 trees) is trained per year; 30% of samples are held out for testing. A 3×3 "
             "majority filter removes speckle. In years up to 2020, built-up is removed where GHSL shows under 5% building cover "
             "and the evidence is weak (built-up probability below 0.7, or a green season with NDVI above 0.3). Water is removed "
             "where JRC Global Surface Water never recorded water in 1984–2021 and no water signal is present. A gap in built-up "
             "between two built-up dates is filled; built-up that never appears again later is removed. These rules were tuned "
             "against USGS LCMAP in Las Vegas and tested on other areas."),
            ("Accuracy", "Hold-out confusion matrices; agreement with ESA WorldCover and IO/Esri for 2021, with ESA CCI (300 m) for "
             "older years, and with GHSL for built-up. A stratified random sample is produced for your own reference labelling "
             "(area-adjusted accuracy following Olofsson et al. 2014)."),
            ("Change analysis", "Transition matrices; gain, loss, net change and swap (Pontius et al. 2004); annual rate of change "
             "(Puyravaud 2003); intensity analysis at interval, category and transition levels (Aldwaik & Pontius 2012); change "
             "frequency; hot spots of change with Getis-Ord Gi* on a 1 km grid."),
            ("Landscape metrics", "Class and landscape metrics after FRAGSTATS definitions: PLAND, NP, PD, LPI, MPS, ED, MSI, AI, "
             "SHDI, SHEI and CONTAG (8-neighbour patches)."),
            ("Urban expansion", "Annual growth rate and urban expansion intensity index (UEII), expansion by compass sector from the "
             f"{y0} centre, built-up density in {R['ring_km']} km rings, and expansion type from the landscape expansion index "
             "(infilling, edge-expansion, outlying; Liu et al. 2010)."),
            ("Indices and temperature", "Mean spectral indices by class; land surface temperature from the Landsat thermal band "
             "(median of the year), its relation to NDVI and NDBI, and the built-up minus vegetation contrast."),
            ("Drivers", f"Logistic regression of where new built-up land appeared in {years[-2]}–{years[-1]} against distance to "
             "existing built-up, water and centre, neighbourhood density, elevation and slope; tested by AUC on held-out pixels."),
            ("Projection", "Markov transition probabilities from the last interval; cellular-automata allocation using the driver "
             "suitability (built-up) and neighbourhood (other classes). Validated by hindcasting the last known date from the previous "
             "interval: figure of merit, kappa, quantity and allocation disagreement (Pontius & Millones 2011)."),
            ("Ecosystem service value", "Area × value coefficient (Costanza et al. 1997, US$/ha/yr), with the coefficient of "
             "sensitivity for ±50% change in each coefficient."),
        ])

        # ------------------------------------------------ results: maps and areas
        rep.heading("4.1 Land use / land cover maps")
        for y in years:
            rep.map(MapSpec(R["maps"][y], g, f"Land use / land cover {y}", classes=classes_all, subtitle=aoi.name, source=src,
                            show_classes=present), f"Land use / land cover classification, {y}.")
        rep.heading("4.2 Area statistics")
        rep.table(["Class"] + [f"{y} (km²)" for y in years] + ["Change (km²)", "Change (%)", "Rate (%/yr)"],
                  [[nm[c]] + [A(y, c) for y in years] + [f"{d[c]:+,.2f}",
                   f"{d[c] / areas[y0][c] * 100:+.1f}%" if areas[y0][c] > 0.01 else "new",
                   (lambda r: "–" if r is None else f"{r:+.2f}")(CH.annual_rate(areas[y0][c], areas[latest][c], latest - y0))]
                   for c in present], "Area of each class by year", [col[c] for c in present])
        rep.table(["Class"] + [f"{y} (%)" for y in years],
                  [[nm[c]] + ["n/m" if (y, c) in missing else f"{areas[y][c] / tot * 100:.1f}" for y in years] for c in present],
                  "Share of the study area by class (n/m = class could not be mapped that year)", [col[c] for c in present])
        rep.figure(render_multi_line([(nm[c], [areas[y][c] for y in years], col[c]) for c in present], np.array(years),
                                     "Year", "Area (km²)"), "Area of each class over time.")

        # ------------------------------------------------ change
        rep.heading("4.3 Transitions and gains/losses")
        for (a, b), Tm, cp in zip(zip(years[:-1], years[1:]), R["Ts"], R["comp"]):
            rep.table([f"{a} ↓  {b} →"] + [nm[c] for c in present],
                      [[nm[present[i]]] + [fmt(Tm[i, j]) for j in range(len(present))] for i in range(len(present))],
                      f"Transition matrix {a}–{b} (km²); diagonal = unchanged", [col[c] for c in present])
            rep.figure(render_grouped_bars([nm[c] for c in present], [("Gain", list(cp["gain"]), "#2f7a4d"),
                                                                    ("Loss", list(-cp["loss"]), "#b8322a"),
                                                                    ("Net", list(cp["net"]), "#1c2621")], "km²"),
                       f"Gains, losses and net change by class, {a}–{b}.")
        ca = R["comp_all"]
        rep.table(["Class", "Gain", "Loss", "Net change", "Swap", "Total change"],
                  [[nm[c], fmt(ca["gain"][i]), fmt(ca["loss"][i]), f"{ca['net'][i]:+,.2f}", fmt(ca["swap"][i]), fmt(ca["total"][i])]
                   for i, c in enumerate(present)], f"Change components {y0}–{latest} (km², Pontius et al. 2004)", [col[c] for c in present])
        top = [(T[i, j], present[i], present[j]) for i in range(len(present)) for j in range(len(present)) if i != j and T[i, j] > 0]
        top.sort(reverse=True)
        rep.table(["From", "To", "Area (km²)", "Share of area"], [[nm[a], nm[b], fmt(v), f"{v / tot * 100:.2f}%"] for v, a, b in top[:10]],
                  f"The ten largest conversions {y0}–{latest}", [col[a] for _, a, _ in top[:10]])
        rep.map(MapSpec(R["persist"], g, f"Persistence and change {y0}–{latest}", classes=PERSIST, subtitle=aoi.name, source=src),
                "Unchanged land, new built-up land and other change.")

        rep.heading("4.4 Intensity analysis")
        it = R["inten"]
        rep.text(f"Interval level: the uniform (average) annual change over the whole period is <b>{it['uniform']:.2f}% of the area per year</b>. "
                 "An interval is fast if its annual change is above this line and slow if below. At category level, a class whose gain "
                 "(or loss) intensity is above the interval's annual change is an active gainer (or loser); below it, dormant.")
        rep.figure(render_grouped_bars([f"{years[i]}–{years[i + 1]}" for i in range(len(R["Ts"]))],
                                       [("Annual change", it["interval"], "#2f5d50")], "% of area per year",
                                       ref_lines=[(it["uniform"], f"uniform {it['uniform']:.2f}%")]),
                   "Intensity analysis, interval level (Aldwaik & Pontius 2012).")
        for i, c_ in enumerate(it["category"]):
            per = f"{years[i]}–{years[i + 1]}"
            rep.table(["Class", "Gain intensity (%/yr)", "Gain", "Loss intensity (%/yr)", "Loss"],
                      [[nm[c], f"{c_['gain'][j]:.2f}", "active" if c_["gain"][j] > c_["uniform"] else "dormant",
                        f"{c_['loss'][j]:.2f}", "active" if c_["loss"][j] > c_["uniform"] else "dormant"] for j, c in enumerate(present)],
                      f"Intensity analysis, category level, {per} (uniform line {c_['uniform']:.2f}%/yr)", [col[c] for c in present])
        if R["trans_int"]:
            rows = []
            for i, (Rv, W) in enumerate(R["trans_int"]):
                for j, c in enumerate(present):
                    if c == 2 or not np.isfinite(Rv[j]):
                        continue
                    rows.append([f"{years[i]}–{years[i + 1]}", nm[c], f"{Rv[j]:.3f}", f"{W:.3f}", "targeted" if Rv[j] > W else "avoided"])
            rep.table(["Period", "Gain of built-up from", "Intensity (%/yr)", "Uniform (%/yr)", "Built-up growth…"], rows,
                      "Intensity analysis, transition level: which classes built-up growth targets")

        rep.heading("4.5 Where change concentrates")
        fz = np.zeros(g.shape, np.uint8)
        fz[R["anyv"]] = 1 + np.minimum(R["freq"][R["anyv"]], 3)
        rep.map(MapSpec(fz, g, "Change frequency", classes=FREQ, subtitle=f"Number of class changes across {len(years) - 1} intervals",
                        source=src), "How many times each pixel changed class.")
        rep.map(MapSpec(R["hot"], g, f"Hot spots of change {y0}–{latest}", classes=HOT, subtitle="Getis-Ord Gi* on a 1 km grid",
                        source=src), "Statistically significant clusters of change.")

        # ------------------------------------------------ landscape
        rep.heading("4.6 Landscape metrics")
        lk = ["NP", "PD", "LPI", "ED", "SHDI", "SHEI", "CONTAG", "AI"]
        rep.table(["Year"] + lk, [[str(y)] + [fmt(R["lm"][y][1][k], 2) for k in lk] for y in years], "Landscape-level metrics")
        ck = ["PLAND", "NP", "PD", "LPI", "MPS", "ED", "MSI", "AI"]
        for c in present:
            rep.table(["Year"] + ck, [[str(y)] + [fmt(R["lm"][y][0][c][k], 2) if c in R["lm"][y][0] else "–" for k in ck] for y in years],
                      f"Class-level metrics: {nm[c]}")
        rep.figure(render_multi_line([(nm[c], [R["lm"][y][0][c]["PD"] if c in R["lm"][y][0] else None for y in years], col[c]) for c in present],
                                     np.array(years), "Year", "Patch density (patches per 100 ha)"),
                   "Fragmentation over time: patch density by class.")
        rep.text("PLAND = share of landscape (%); NP = number of patches; PD = patches per 100 ha; LPI = largest patch (% of area); "
                 "MPS = mean patch size (ha); ED = edge density (m/ha); MSI = mean shape index; AI = aggregation index (%); "
                 "SHDI/SHEI = Shannon diversity/evenness; CONTAG = contagion (%).")

        # ------------------------------------------------ urban
        if 2 in present:
            rep.heading("4.7 Urban expansion")
            rep.table(["Period", "Start (km²)", "End (km²)", "Growth (km²)", "Annual growth (%)", "UEII (%)",
                       "Infilling", "Edge-expansion", "Outlying"],
                      [[u["period"], fmt(u["u1"]), fmt(u["u2"]), fmt(u["growth"]), "–" if u["agr"] is None else f"{u['agr']:.2f}",
                        f"{u['ueii']:.3f}", fmt(l[2][1]), fmt(l[2][2]), fmt(l[2][3])] for u, l in zip(R["urb"], R["lei_maps"])],
                      "Urban growth rate, expansion intensity (UEII) and expansion type (km²)")
            rep.figure(render_polar([(per, list(s), c_) for (per, _, _), s, c_ in
                                     zip(R["lei_maps"], R["sect"], ["#f2c14e", "#e0633a", "#9e1b16", "#5b2333"])], LS.SECTORS,
                                    "km² of new built-up"), f"Direction of urban expansion from the {y0} centre.")
            rep.figure(render_multi_line([(str(y), gr, c_) for y, gr, c_ in zip(years, R["grad"],
                                          ["#f2c14e", "#e0633a", "#9e1b16", "#5b2333", "#2b1020"])],
                                         np.arange(1, max(len(x) for x in R["grad"]) + 1) * R["ring_km"],
                                         "Distance from centre (km)", "Built-up density (%)"),
                       "Built-up density gradient from the centre.")
            per, tmap, _ = R["lei_maps"][-1]
            rep.map(MapSpec(tmap, g, f"Urban expansion type {per}", classes=LEI_CLASSES, subtitle="Landscape expansion index",
                            source=src), f"Infilling, edge-expansion and outlying growth, {per}.")

        # ------------------------------------------------ indices and LST
        rep.heading("4.8 Spectral indices and land surface temperature")
        for k in INDEX_INFO:
            rep.table(["Class"] + [str(y) for y in years],
                      [[nm[c]] + ["–" if R["idx_tab"][y][k][c] is None else f"{R['idx_tab'][y][k][c]:.3f}" for y in years] for c in present],
                      f"Mean {k} by class ({INDEX_INFO[k]})", [col[c] for c in present])
        ix0, ixl = R["feats"][y0][1], R["feats"][latest][1]
        for k, ramp in (("NDVI", "ndvi"), ("NDBI", "heat")):
            rep.map(MapSpec(ixl[k], g, f"{k} {latest}", ramp=ramp, vmin=-0.4, vmax=0.8, unit=k, subtitle=INDEX_INFO[k], source=src),
                    f"{k}, {latest} (median of the year).")
        rep.table(["Class"] + [f"{y} (°C)" for y in years],
                  [[nm[c]] + ["–" if R["lst_tab"][y][c] is None else f"{R['lst_tab'][y][c]:.1f}" for y in years] for c in present],
                  "Mean land surface temperature by class (annual median)", [col[c] for c in present])
        rep.table(["Year", "r (LST, NDVI)", "r (LST, NDBI)", "Built-up minus vegetation (°C)"],
                  [[str(y), "–" if R["lst_corr"][y][0] is None else f"{R['lst_corr'][y][0]:.2f}",
                    "–" if R["lst_corr"][y][1] is None else f"{R['lst_corr'][y][1]:.2f}",
                    "–" if y not in R["uhi"] else f"{R['uhi'][y]:+.2f}"] for y in years],
                  "Relation of land surface temperature to vegetation and built-up indices")
        lst_l = R["comps"][latest]["lst"]
        rep.map(MapSpec(lst_l, g, f"Land surface temperature {latest}", ramp="heat", vmin=float(np.nanpercentile(lst_l, 2)),
                        vmax=float(np.nanpercentile(lst_l, 98)), unit="°C", subtitle="Median of the year", source="Landsat C2 L2 (USGS)"),
                f"Land surface temperature, {latest}.")

        # ------------------------------------------------ terrain
        rep.heading("4.9 Terrain")
        rep.table(["Elevation zone"] + [nm[c] for c in present],
                  [[z] + [fmt(v[c]) for c in present] for z, v in R["terr_e"]], f"Class area (km²) by elevation zone, {latest}")
        rep.table(["Slope zone"] + [nm[c] for c in present] + ["New built-up since " + str(y0)],
                  [[z] + [fmt(v[c]) for c in present] + [fmt(dict(R["gain_by_slope"])[z])] for z, v in R["terr_s"]],
                  f"Class area (km²) by slope zone, {latest}")

        # ------------------------------------------------ drivers
        drv = R["drv"]
        if drv:
            rep.heading("4.10 Drivers of urban growth")
            rep.text(f"Logistic regression of new built-up land {years[-2]}–{years[-1]} ({drv['n']:,} sample pixels). Odds ratios are per "
                     f"one standard deviation of each factor; above 1 increases the chance of urbanisation. Held-out AUC: <b>{drv['auc']:.2f}</b>.")
            rep.table(["Factor", "Coefficient", "Odds ratio", "z", "p"],
                      [[r["name"], f"{r['coef']:+.3f}", f"{r['odds']:.2f}", f"{r['z']:.1f}", f"{r['p']:.1e}"] for r in drv["rows"]],
                      "Drivers of urban growth")
            rep.map(MapSpec(drv["suitability"], g, "Suitability for urban growth", ramp="risk", vmin=0, vmax=1, unit="probability",
                            subtitle=f"From {years[-2]}–{years[-1]} growth", source=src), "Modelled suitability for new built-up land.")

        # ------------------------------------------------ projection
        if not R["proj"] and R.get("proj_stopped"):
            rep.heading("4.11 Future land use / land cover")
            rep.text(R["proj_stopped"] + " A projection that cannot reproduce the recent past should not be used to plan for the future; "
                     "scenario maps are skipped for the same reason.")
        if R["proj"]:
            rep.heading("4.11 Future land use / land cover")
            P = R["P"]
            rep.table([f"From ↓ to →"] + [nm[c] for c in present],
                      [[nm[present[i]]] + [f"{P[i, j]:.3f}" for j in range(len(present))] for i in range(len(present))],
                      f"Markov transition probabilities ({years[-2]}–{years[-1]})", [col[c] for c in present])
            sk = R["skill"]
            if sk:
                rep.table(["Measure", "Value", "Meaning"],
                          [["Overall agreement", f"{sk['oa'] * 100:.1f}%", f"No-change model scores {sk['null_oa'] * 100:.1f}%"],
                           ["Kappa", f"{sk['kappa']:.3f}", "Agreement beyond chance"],
                           ["Figure of merit", f"{sk['fom']:.3f}", "Overlap of observed and simulated change (0–1); 0.1–0.3 is typical"],
                           ["Quantity disagreement", f"{sk['Q'] * 100:.1f}%", "Wrong amounts of each class"],
                           ["Allocation disagreement", f"{sk['A'] * 100:.1f}%", "Right amounts, wrong places"]],
                          f"Hindcast: {sk['years'][1]} simulated from {sk['years'][0]}–{sk['years'][1]} and compared with the real {sk['years'][2]} map")
            fys = sorted(R["proj"])
            rep.table(["Class", f"{latest} (km²)"] + [f"{fy} (km²)" for fy in fys],
                      [[nm[c], fmt(areas[latest][c])] + [fmt(float((R["proj"][fy] == c).sum() * g.px_km2)) for fy in fys] for c in present],
                      "Projected area by class", [col[c] for c in present])
            for fy in fys:
                rep.map(MapSpec(R["proj"][fy], g, f"Projected land use / land cover {fy}", classes=classes_all, subtitle="Markov-CA",
                                source=src, show_classes=present), f"Projected land use / land cover, {fy} (business as usual).")

        # ------------------------------------------------ ESV
        rep.heading("4.12 Ecosystem service value")
        esv = R["esv"]
        ey = sorted(esv)
        rep.table(["Class", "Coefficient (US$/ha/yr)"] + [str(y) for y in ey],
                  [[nm[c], f"{esv[latest][2][c]:,}"] + [f"{esv[y][1].get(c, 0) / 1e6:,.2f}" for y in ey] for c in present]
                  + [["Total", ""] + [f"{esv[y][0] / 1e6:,.2f}" for y in ey]],
                  "Ecosystem service value (million US$ per year, 1994 values; projected years included)", [col[c] for c in present] + [None])
        rep.table(["Class", "Coefficient of sensitivity"],
                  [[nm[c], f"{v:.3f}"] for c, v in R["esv_sens"].items()],
                  "Sensitivity of total value to ±50% change in each coefficient (CS < 1 means the result is robust)")

        # ------------------------------------------------ zones
        if R["zonal"] and R["zonal"]["rows"]:
            rep.heading("4.13 Results by zone")
            zr = sorted(R["zonal"]["rows"], key=lambda r: -r["built_change_km2"])[:20]
            rep.table(["Zone", "Area (km²)"] + [f"Built-up {y}" for y in years] + ["Change", "Built-up share now"],
                      [[r["zone"], fmt(r["area_km2"])] + [fmt(r[f"built_{y}"]) for y in years] + [f"{r['built_change_km2']:+,.2f}",
                       f"{r['built_share_latest_pct']:.0f}%"] for r in zr],
                      "Zones with the largest built-up growth (km²); full table in zones_built_up.csv")

        rep.method_steps += _extra_methods(R)
        _report_extras(rep, R)

        # ------------------------------------------------ validation
        acc = R["acc"]
        vrows = []
        for y in R["all_years"]:
            a = acc[y]
            vrows.append((f"Hold-out test {y}" + (" (reference year)" if y == C.REF_YEAR and y not in years else ""),
                          f"OA {a['oa'] * 100:.1f}%, kappa {a['kappa']:.2f}, n = {a['n']}",
                          ("30% of your training points held out: an independent test if your points are accurate." if R["user"] else
                           "30% of auto-generated samples held out; optimistic because samples are from stable, pure pixels.")))
        for name, yy, a in R["map_checks"]:
            vrows.append((f"Agreement with {name}", f"{a['oa'] * 100:.1f}%, kappa {a['kappa']:.2f}",
                          ("Not independent: this map is one of the two sources of the training labels. " if any(t in name for t in ("WorldCover", "Esri", "IO")) else "")
                          + "Differences include class definitions, not only errors."))
        for y, cy, v in R["cci_checks"]:
            vrows.append((f"Agreement with ESA CCI {cy} (300 m) for {y}", f"{v * 100:.0f}%",
                          "Coarse independent map; checks broad patterns in older years."))
        for y, ep, b, dn, rc in R["ghsl_checks"]:
            vrows.append((f"Built-up vs GHSL {ep} for {y}", f"ours {fmt(b)} km²; GHSL ≥20% built {fmt(dn)} km²; recall {rc:.0f}%",
                          "Not fully independent: GHSL is also used to screen built-up in years before 2021. Land cover 'built-up' includes roads and yards, so it is normally larger than GHSL's building share."))
        for y in R["all_years"]:
            for k, how in R["res"][y].get("notes", []):
                if k == 0:
                    vrows.append((f"Training for {y}", how, "Fewer points than the year the points were collected; check this year's map."))
                    continue
                vrows.append((f"{CL[k][0]} in {y}", how,
                              "Too few unchanged training pixels; " + ("treat this class as missing in this year, not as zero."
                                                                         if how.startswith("not") else "labels from a fallback rule.")))
        vrows.append(("Corrections applied", f"{sum(R['res'][y]['ghsl_fixed'] for y in R['all_years']):,} pixels by GHSL rule; "
                      f"{sum(R['res'][y].get('water_fixed', 0) for y in R['all_years']):,} by water rule; "
                      f"{R['reversal']:,} by built-up consistency rule",
                      "Rules that remove impossible built-up and water; tested against USGS LCMAP (see validation notes)."))
        if R["ref_acc"]:
            o = R["ref_acc"]
            vrows.append((f"Your reference points ({o['year']})", f"area-adjusted OA {o['oa'] * 100:.1f}%, kappa {o['kappa']:.2f}, n = {o['n']}",
                          "Olofsson et al. (2014) estimator; the strongest evidence in this report."))
        else:
            vrows.append(("Your reference points", f"not supplied; {len(R['design'])} sample points ready",
                          f"Label accuracy_sample_{latest}.geojson (field 'reference_class') and rerun for area-adjusted accuracy."))
        rep.validation(vrows)
        a = acc[latest]
        rep.table(["Class", "Producer's accuracy", "User's accuracy", "F1", "Training samples"],
                  [[R["CLX"][c][0], f"{a['pa'][i] * 100:.1f}%", f"{a['ua'][i] * 100:.1f}%", f"{a['f1'][i]:.2f}", f"{a['counts'].get(c, 0):,}"]
                   for i, c in enumerate(a["classes"])], f"Per-class hold-out accuracy, {latest}")
        if R["ref_acc"]:
            o = R["ref_acc"]
            rep.table(["Class", "Mapped area (km²)", "Adjusted area (km²)", "95% CI (±km²)", "User's", "Producer's"],
                      [[CL[c][0], fmt(o["area_map"][i]), fmt(o["area"][i]), fmt(o["ci"][i]), f"{o['ua'][i] * 100:.0f}%", f"{o['pa'][i] * 100:.0f}%"]
                       for i, c in enumerate(o["classes"])], f"Area-adjusted estimates from your reference points ({o['year']})")

        rep.limits([
            "Training labels come from 2021 maps and are moved to other years only where the land looks unchanged; classes that "
            "were common in the past but rare today (for example old wetlands) may be under-represented.",
            "Built-up land in years before 2021 is constrained by GHSL (100 m), so its edges follow 100 m blocks.",
            "Hold-out accuracy is optimistic; use the reference sample for an unbiased, area-adjusted accuracy.",
            "Landsat 7 images after May 2003 have stripes; medians of several dates fill most gaps.",
            "The projection assumes past trends continue (business as usual); it is not a forecast of policy outcomes.",
            "Ecosystem values use global coefficients (Costanza et al. 1997) and indicate direction and scale, not local market values.",
            "Land surface temperature is an annual median of satellite skin temperature, not air temperature.",
        ])
        rep.conclusions(pts + [f"Overall confidence: <b>{R['verdict']['level']}</b>."])
        rep.references([
            D.CREDIT_LS, D.CREDIT_LULC, D.CREDIT_WC, C.CREDIT_GHSL, D.CREDIT_DEM,
            "ESA Land Cover CCI Project Team; Defourny, P. (2019). ESA Land Cover Climate Change Initiative: Land Cover Maps v2.0.7.",
            "Roy, D. P. et al. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and NDVI continuity. RSE 185, 57–70.",
            "Breiman, L. (2001). Random forests. Machine Learning 45, 5–32.",
            "Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. RSE 148, 42–57.",
            "Pontius, R. G., Shusas, E., McEachern, M. (2004). Detecting important categorical land changes. Agric. Ecosyst. Environ. 101, 251–268.",
            "Puyravaud, J.-P. (2003). Standardizing the calculation of the annual rate of deforestation. For. Ecol. Manage. 177, 593–596.",
            "Aldwaik, S. Z., Pontius, R. G. (2012). Intensity analysis to unify measurements of size and stationarity of land changes. "
            "Landscape and Urban Planning 106, 103–114.",
            "McGarigal, K., Marks, B. J. (1995). FRAGSTATS: spatial pattern analysis program. USDA Forest Service PNW-GTR-351.",
            "Liu, X. et al. (2010). A new landscape index for quantifying urban expansion. Landscape Ecology 25, 671–682.",
            "Pontius, R. G., Millones, M. (2011). Death to Kappa. Int. J. Remote Sensing 32, 4407–4429.",
            "Costanza, R. et al. (1997). The value of the world's ecosystem services and natural capital. Nature 387, 253–260.",
            "Getis, A., Ord, J. K. (1992). The analysis of spatial association. Geographical Analysis 24, 189–206.",
        ] + EXTRA_REFS)
        rep.proof(self.sources(), self.reproduce_command())
        rep.files([os.path.basename(f) for f in files] + ["report.html", "report.docx", "report.pdf"])
        return rep.write(out_dir, self.seconds())


# ======================================================================= extra analyses

RINGS_CLASSES = None


def _extras(kit, L, params, out_dir):
    """Population/SDG, sprawl, RSEI, UTFVI, carbon, habitat, risk, gradient, annual trends, runoff/erosion,
    scenarios, classifier comparison, Geodetector/GWR, Level-II, master plan and census."""
    import os as _os
    from osgeo import ogr
    from ..core import osm
    g, years, maps, comps, feats = L["grid"], L["years"], L["maps"], L["comps"], L["feats"]
    present, px, latest, y0 = L["present"], L["px"], L["latest"], L["years"][0]
    aoi, ctr, ring_km = g.aoi, L["ctr"], L["ring_km"]
    X = {"files": [], "layers": []}

    def wcls(name, arr, cls, label=None, hidden=True, grid_=None):
        p = write_classes(_os.path.join(out_dir, name), arr, grid_ or g, cls)
        X["files"].append(p)
        if label:
            X["layers"].append(dict(path=p, name=label, kind="raster", hidden=hidden))
        return p

    def wflt(name, arr, ramp, lo, hi, unit="", label=None, grid_=None):
        p, _ = write_float(_os.path.join(out_dir, name), arr, grid_ or g, ramp, lo, hi, unit)
        X["files"].append(p)
        if label:
            X["layers"].append(dict(path=p, name=label, kind="raster", hidden=True))
        return p

    # --- population and SDG 11.3.1
    kit.step(88.0, "Population and SDG 11.3.1")
    pops, pop_tot, pop_ep = {}, {}, {}
    for y in years:
        p_, e = EX.population(g, y)
        if p_ is not None:
            pops[y], pop_tot[y], pop_ep[y] = p_, float(np.nansum(p_)), e
    X["pop"] = dict(total=pop_tot, epoch=pop_ep)
    built_km2 = {y: float((maps[y] == 2).sum() * px) for y in years}
    X["sdg"] = EX.sdg_1131(built_km2, pop_tot, years) if len(pop_tot) == len(years) else None

    # --- sprawl
    kit.step(88.5, "Sprawl indices")
    rid = EX.ring_ids(g, ctr, ring_km)
    rr, cc = np.indices(g.shape)
    ang = (np.degrees(np.arctan2(cc - ctr[1], -(rr - ctr[0]))) + 360 + 22.5) % 360
    sec = (ang // 45).astype(int)
    spr = []
    for y in years:
        b = maps[y] == 2
        hs, hsr = EX.shannon(np.bincount(sec[b], minlength=8))
        hr, hrr = EX.shannon(np.bincount(rid[b & (rid > 0)], minlength=rid.max() + 1)[1:])
        pl = [(a, p) for c_, a, p in LS.patches(np.where(b, 2, 0).astype(np.uint8), g) if c_ == 2]
        comp_, fd = EX.compactness(pl, g.res)
        spr.append(dict(year=y, H_sector=hs, H_sector_rel=hsr, H_ring=hr, H_ring_rel=hrr, compact=comp_, fractal=fd, patches=len(pl)))
    X["sprawl"] = spr

    # --- RSEI and UTFVI
    kit.step(89.0, "Ecological index (RSEI) and urban heat (UTFVI)")
    rs, ut = {}, {}
    for y in years:
        c = comps[y]
        r_ = EX.rsei(c["bands"], c["lst"], feats[y][1]["NDVI"], g.mask)
        if r_:
            r_["classes"] = EX.classify_01(r_["rsei"], g.mask)
            rs[y] = r_
        u, uc = EX.utfvi(c["lst"], g.mask)
        ut[y] = uc
    X["rsei"], X["utfvi"] = rs, ut
    if rs:
        wflt(f"rsei_{latest}.tif", rs[max(rs)]["rsei"], "ndvi", 0, 1, "", f"RSEI {max(rs)}")
    wcls(f"utfvi_{latest}.tif", ut[latest], EX.UTFVI_CLASSES, f"UTFVI {latest}")

    # --- carbon and habitat quality
    kit.step(89.5, "Carbon storage and habitat quality")
    lat = aoi.center[1]
    carb, hq = {}, {}
    for y in years:
        tot, by, dens, table = EX.carbon(maps[y], px, lat)
        carb[y] = (tot, by)
        q, _ = EX.habitat_quality(maps[y], g.mask, g.res)
        hq[y] = q
    for fy, pm in L["proj"].items():
        carb[fy] = EX.carbon(pm, px, lat)[:2]
    X["carbon"], X["carbon_table"], X["hq"] = carb, table, hq
    wflt(f"habitat_quality_{latest}.tif", hq[latest], "ndvi", 0, 1, "", f"Habitat quality {latest}")
    d0 = EX.carbon(maps[y0], px, lat)[2]
    d1 = EX.carbon(maps[latest], px, lat)[2]
    cdiff = np.where(g.mask, d1 - d0, np.nan)
    X["carbon_diff"] = cdiff
    wflt(f"carbon_change_{y0}_{latest}.tif", cdiff, "heat", -150, 150, "Mg C/ha")

    # --- landscape ecological risk
    kit.step(90.0, "Landscape ecological risk")
    eri = {}
    for y in years:
        pl = LS.patches(maps[y], g)
        e, cpx, Ri = EX.ecological_risk(maps[y], pl, g)
        eri[y] = e
    allv = np.concatenate([e[np.isfinite(e)] for e in eri.values()])
    br = np.quantile(allv, [0.2, 0.4, 0.6, 0.8]) if allv.size else [0, 0, 0, 0]
    eri_cls = {}
    for y, e in eri.items():
        cls_ = np.zeros(e.shape, np.uint8)
        v = np.isfinite(e)
        cls_[v] = 1 + np.digitize(e[v], br)
        eri_cls[y] = EX.expand_cells(cls_, cpx, g.shape, np.uint8, 0)
        eri_cls[y][~g.mask] = 0
    X["eri"], X["eri_cls"], X["eri_breaks"] = eri, eri_cls, br
    wcls(f"ecological_risk_{latest}.tif", eri_cls[latest], EX.ERI_CLASSES, f"Ecological risk {latest}")

    # --- gradient
    kit.step(90.5, "Urban–rural gradient")
    X["gradient"] = {y: EX.gradient_metrics(maps[y], rid, g, present, LS.metrics) for y in (y0, latest)}

    # --- annual series and trends
    X["annual"] = None
    if params.get("annual_trend", "yes") == "yes":
        g60 = Grid(aoi, 60, max_pixels=1_500_000)
        try:
            yrs, nd, nbr, lsta = AN.series(aoi, g60, y0, latest,
                                          progress=lambda f, m: kit.step(90.5 + 2.5 * f, m))
            Zn, Pn, Sn = AN.mk_sen(nd, yrs)
            Zt, Pt, St = AN.mk_sen(lsta, yrs)
            when, mag = AN.disturbance(nd, yrs)
            tn, tt = AN.trend_classes(Zn, Pn, g60.mask), AN.trend_classes(Zt, Pt, g60.mask)
            X["annual"] = dict(grid=g60, years=yrs, ndvi_mean=[float(np.nanmean(a)) for a in nd],
                               lst_mean=[float(np.nanmean(a)) for a in lsta], tn=tn, tt=tt, Sn=Sn, St=St, when=when, mag=mag,
                               mk_mean_ndvi=AN.mk_sen(np.array([np.nanmean(a) for a in nd])[:, None, None], yrs),
                               mk_mean_lst=AN.mk_sen(np.array([np.nanmean(a) for a in lsta])[:, None, None], yrs))
            wcls("ndvi_trend_classes.tif", tn, AN.TREND_CLASSES, "NDVI trend (Mann-Kendall)", grid_=g60)
            wcls("lst_trend_classes.tif", tt, AN.TREND_CLASSES, "LST trend (Mann-Kendall)", grid_=g60)
            wflt("ndvi_sen_slope_per_year.tif", Sn, "ndvi", -0.02, 0.02, "NDVI/yr", grid_=g60)
            wflt("lst_sen_slope_c_per_year.tif", St, "heat", -0.3, 0.3, "°C/yr", grid_=g60)
            wflt("vegetation_loss_year.tif", np.where(when > 0, when, np.nan).astype(np.float32), "seq",
                 float(yrs[0]), float(yrs[-1]), "year", "Year of vegetation loss", grid_=g60)
            # tree-cover loss alerts: loss where the first years looked like tree cover (NDVI >= 0.55)
            with np.errstate(all="ignore"):
                early = np.nanmean(nd[:min(3, len(nd))], 0)
            treeish = (np.nan_to_num(early) >= 0.55) & g60.mask
            # also require the first classified map to call it forest / tree cover (class 3), so green crops are not counted
            if 3 in present:
                f30 = (maps[y0] == 3).astype(np.float32)
                rr_ = np.minimum(((np.arange(g60.rows) + 0.5) * g60.res / g.res).astype(int), g.rows - 1)
                cc_ = np.minimum(((np.arange(g60.cols) + 0.5) * g60.res / g.res).astype(int), g.cols - 1)
                treeish &= f30[np.ix_(rr_, cc_)] > 0
            else:
                treeish &= False
            yrs_arr = np.asarray(yrs)
            tl = (when > 0) & treeish & (when >= (yrs_arr[min(3, len(yrs_arr) - 1)] if len(yrs_arr) else 0))
            X["annual"]["tree_loss"] = np.where(tl, when, 0).astype(np.int16)
            X["annual"]["tree_px"] = float(treeish.sum() * g60.px_km2)
            if tl.any():
                wflt("tree_cover_loss_year.tif", np.where(tl, when, np.nan).astype(np.float32), "seq",
                     float(yrs[0]), float(yrs[-1]), "year", "Year of tree-cover loss", grid_=g60)
        except Exception as exc:  # the annual series is optional
            X["annual_error"] = str(exc)

    # --- runoff and erosion cover factor
    kit.step(93.0, "Runoff (SCS-CN) and erosion cover factor")
    hsg = CP.soil_groups(g)
    X["hydro"] = None
    if hsg is not None:
        P_mm = float(params.get("storm_mm") or 100)
        hy = {}
        for y in years:
            Q, cn = CP.runoff(maps[y], hsg, P_mm)
            Cf = CP.c_factor(feats[y][1]["NDVI"])
            Cf[~g.mask] = np.nan
            hy[y] = dict(Q=float(np.nanmean(Q)), vol=float(np.nansum(Q) / 1000 * g.res * g.res), CN=float(np.nanmean(cn[cn > 0])),
                         C=float(np.nanmean(Cf)), Qmap=Q, Cmap=Cf,
                         C_by={c: float(np.nanmean(Cf[maps[y] == c])) if (maps[y] == c).any() else None for c in present})
        X["hydro"] = dict(P=P_mm, years=hy, hsg={k: float((hsg == k).sum() * px) for k in (1, 2, 3, 4)})
        wflt(f"runoff_mm_{latest}.tif", hy[latest]["Qmap"], "depth", 0, P_mm, "mm", f"Runoff {latest} ({P_mm:g} mm storm)")
        wflt(f"rusle_c_{latest}.tif", hy[latest]["Cmap"], "risk", 0, 1, "")

    # --- scenarios
    X["scen"] = None
    if L["proj"] and "P" in L:
        kit.step(94.0, "Land-use scenarios")
        P = L["P"]
        fy = min(L["proj"])
        steps = 1
        suit = L["drv"]["suitability"] if L["drv"] else None
        sc = {}
        for name in CP.SCENARIOS:
            Q_ = CP.scenario_matrix(P, present, name)
            m_ = L["proj"][fy] if name == "Business as usual" else M.ca_allocate(maps[latest], Q_, present, suit, steps, seed=fy + 7)
            m_[~g.mask] = 0
            ar = {c: float((m_ == c).sum() * px) for c in present}
            sc[name] = dict(map=m_, areas=ar, esv=M.esv(ar, lat)[0], carbon=EX.carbon(m_, px, lat)[0])
            if name != "Business as usual":
                wcls(f"scenario_{name.lower().replace(' ', '_')}_{fy}.tif", m_, {c: CL[c] for c in present}, f"Scenario {fy}: {name}")
        X["scen"] = dict(year=fy, runs=sc)

    # --- classifier comparison
    X["clf"] = None
    smp = L.get("samples")
    if smp is not None:
        kit.step(94.5, "Comparing classifiers")
        Xtr, ytr, Xte, yte, prf = smp
        cls_ = sorted(set(ytr.tolist()))
        rows = []
        import time as _t
        for nm_, fn in (("Random Forest (used)", None), ("SVM, RBF kernel (LS approx.)", CP.ls_svm_rbf),
                        ("Maximum Likelihood", CP.max_likelihood), ("Minimum Distance", CP.min_distance)):
            t0 = _t.time()
            pr = prf if fn is None else fn(Xtr, ytr, Xte, cls_)
            m_ = AC.metrics(AC.confusion(yte, pr, cls_))
            chi, pv = (None, None) if fn is None else CP.mcnemar(yte, prf, pr)
            rows.append(dict(name=nm_, oa=m_["oa"], kappa=m_["kappa"], f1=float(np.nanmean(m_["f1"])), chi=chi, p=pv,
                             sec=_t.time() - t0))
        X["clf"] = dict(rows=rows, n_train=len(ytr), n_test=len(yte), year=latest)

    # --- spatial drivers: Geodetector and GWR on 1 km cells
    kit.step(95.0, "Geodetector and GWR")
    X["geo"] = None
    road_d = None
    try:
        lines = osm.roads(aoi.bbox)
        geoms = []
        for ln in lines:
            if len(ln) < 2:
                continue
            gl = ogr.Geometry(ogr.wkbLineString)
            for lo, la in ln:
                x_, y_, _ = aoi.to_local.TransformPoint(lo, la)
                gl.AddPoint_2D(x_, y_)
            geoms.append(gl)
        if geoms:
            road_d = g.distance_to(g.rasterize(geoms, all_touched=True))
            X["roads_n"] = len(geoms)
    except Exception as exc:
        X["roads_error"] = str(exc)
    cpx = max(3, int(round(1000 / g.res)))
    newb = ((maps[latest] == 2) & (maps[y0] != 2)).astype(float)
    newb[~g.mask] = np.nan
    yv = CP.cells(newb, cpx)
    layers_ = {"Distance to built-up": g.distance_to(maps[y0] == 2),
               "Distance to centre": np.hypot((rr - ctr[0]) * g.res, (cc - ctr[1]) * g.res),
               "Elevation": np.nan_to_num(L["elev"], nan=np.nanmean(L["elev"])), "Slope": np.nan_to_num(L["slope"])}
    if (maps[y0] == 1).any():
        layers_["Distance to water"] = g.distance_to(maps[y0] == 1)
    if y0 in pops:
        layers_["Population density"] = np.nan_to_num(pops[y0]) / (g.res * g.res / 1e6)
    if road_d is not None:
        layers_["Distance to major road"] = road_d
    names_ = list(layers_)
    Xc = np.column_stack([CP.cells(layers_[n_].astype(float), cpx).ravel() for n_ in names_])
    ok = np.isfinite(yv.ravel()) & np.isfinite(Xc).all(1)
    if ok.sum() >= 30 and np.nanstd(yv) > 0:
        fac, inter = CP.geodetector(yv.ravel()[ok], Xc[ok], names_)
        r_, c_ = np.indices(yv.shape)
        coords = np.column_stack([r_.ravel()[ok], c_.ravel()[ok]]).astype(float)
        G = CP.gwr(yv.ravel()[ok], Xc[ok], coords) if ok.sum() <= 3000 else None
        top = max(fac, key=lambda d: d["q"])["name"]
        cmap_ = None
        if G:
            k = names_.index(top) + 1
            arr = np.full(yv.shape, np.nan)
            arr.ravel()[np.flatnonzero(ok)] = G["coef"][:, k]
            cmap_ = EX.expand_cells(arr, cpx, g.shape)
            cmap_[~g.mask] = np.nan
        X["geo"] = dict(fac=fac, inter=inter[:8], gwr=G, top=top, coef_map=cmap_, n=int(ok.sum()), names=names_)

    # --- Level-II (own classes)
    if L.get("user"):
        cl2 = L["classes_l2"]
        X["l2"] = dict(classes=cl2, areas={y: {k: float((L["maps_l2"][y] == k).sum() * px) for k in cl2} for y in years})
        for y in years:
            wcls(f"lulc_level2_{y}.tif", L["maps_l2"][y], {k: (v[0], v[1]) for k, v in cl2.items()}, f"Level-II {y}", y != latest)

    # --- master plan
    X["plan"] = None
    if params.get("plan"):
        kit.step(96.0, "Checking land use against the master plan")
        X["plan"] = UD.plan_conformity(params["plan"], params.get("plan_field") or "landuse", aoi, g, maps, years, px)
        if X["plan"]:
            wcls("master_plan_conformity.tif", X["plan"]["map"], UD.PLAN_CLASSES, "Master plan conformity", False)

    # --- census
    X["census"] = None
    if params.get("census"):
        kit.step(96.5, "Relating land use to census data")
        lay = {f"Built-up share {latest} (%)": (maps[latest] == 2).astype(float) * 100,
               f"Built-up change {y0}–{latest} (pp)": ((maps[latest] == 2).astype(float) - (maps[y0] == 2)) * 100,
               f"Tree cover {latest} (%)": (maps[latest] == 3).astype(float) * 100,
               f"NDVI {latest}": feats[latest][1]["NDVI"], f"LST {latest} (°C)": comps[latest]["lst"]}
        if latest in rs:
            lay[f"RSEI {latest}"] = rs[latest]["rsei"]
        for k_, v_ in lay.items():
            v_ = np.asarray(v_, float)
            v_[~g.mask] = np.nan
            lay[k_] = v_
        X["census"] = UD.census(params["census"], params.get("census_id") or "name", params.get("census_fields") or "",
                                aoi, g, lay)

    # --- tables
    import csv as _csv

    def w(name, header, rows):
        p = _os.path.join(out_dir, name)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            c_ = _csv.writer(fh)
            c_.writerow(header)
            c_.writerows(rows)
        X["files"].append(p)

    if X["sdg"]:
        w("sdg_11_3_1.csv", ["period", "land_consumption_rate", "population_growth_rate", "LCRPGR", "m2_built_per_person_start",
                             "m2_built_per_person_end"],
          [[r["period"]] + [None if r[k] is None else round(r[k], 5) for k in ("lcr", "pgr", "ratio", "m2_per_person_start",
                                                                              "m2_per_person_end")] for r in X["sdg"]])
    w("sprawl_indices.csv", list(spr[0].keys()), [[None if v is None else (round(v, 4) if isinstance(v, float) else v)
                                                   for v in r.values()] for r in spr])
    if rs:
        w("rsei.csv", ["year", "mean_rsei", "pc1_share_pct"] + list(EX.RSEI_CLASSES.keys()),
          [[y, round(r["mean"], 4), round(r["pc1_share"], 2)] + [round(float((r["classes"] == k).sum() * px), 4) for k in EX.RSEI_CLASSES]
           for y, r in rs.items()])
    w("carbon_storage_mg.csv", ["year", "total_MgC"] + [CL[c][0] for c in present],
      [[y, round(v[0])] + [round(v[1].get(c, 0)) for c in present] for y, v in carb.items()])
    w("habitat_quality.csv", ["year", "mean"], [[y, round(float(np.nanmean(q)), 4)] for y, q in hq.items()])
    w("ecological_risk.csv", ["year", "mean_eri"] + [v[0] for v in EX.ERI_CLASSES.values()],
      [[y, round(float(np.nanmean(eri[y])), 5)] + [round(float((eri_cls[y] == k).sum() * px), 4) for k in EX.ERI_CLASSES] for y in years])
    if X["hydro"]:
        w("runoff_erosion.csv", ["year", "mean_runoff_mm", "runoff_volume_m3", "mean_CN", "mean_RUSLE_C"],
          [[y, round(v["Q"], 2), round(v["vol"]), round(v["CN"], 2), round(v["C"], 4)] for y, v in X["hydro"]["years"].items()])
    if X["annual"]:
        A = X["annual"]
        w("annual_series.csv", ["year", "mean_ndvi", "mean_lst_c"],
          [[int(y), round(a, 4), round(b, 3)] for y, a, b in zip(A["years"], A["ndvi_mean"], A["lst_mean"])])
    if X["scen"]:
        w("scenarios.csv", ["scenario"] + [CL[c][0] for c in present] + ["ESV_usd", "carbon_MgC"],
          [[n_] + [round(r["areas"][c], 4) for c in present] + [round(r["esv"]), round(r["carbon"])] for n_, r in X["scen"]["runs"].items()])
    if X["clf"]:
        w("classifier_comparison.csv", ["classifier", "overall_accuracy", "kappa", "mean_F1", "mcnemar_chi2", "p", "seconds"],
          [[r["name"], round(r["oa"], 4), round(r["kappa"], 4), round(r["f1"], 4), r["chi"], r["p"], round(r["sec"], 2)] for r in X["clf"]["rows"]])
    if X["geo"]:
        w("geodetector.csv", ["factor", "q", "p"], [[f_["name"], round(f_["q"], 4), round(f_["p"], 4)] for f_ in X["geo"]["fac"]])
    if X.get("l2"):
        cl2 = X["l2"]["classes"]
        w("level2_areas_km2.csv", ["class", "level1"] + [str(y) for y in years],
          [[v[0], CL[v[2]][0]] + [round(X["l2"]["areas"][y][k], 4) for y in years] for k, v in cl2.items()])
    if X["plan"]:
        w("master_plan_conformity.csv", ["category", "area_km2", "conformity_pct", "built_in_nonbuilt_km2", "built_growth_km2"],
          [[r["category"], round(r["area"], 4), "" if r["conformity"] is None else round(r["conformity"], 2),
            round(r["built_in_nonbuilt"], 4), round(r["built_growth"], 4)] for r in X["plan"]["rows"]])
    if X["census"]:
        w("census_correlations.csv", ["field", "lulc_metric", "pearson_r", "spearman_rho", "p_permutation", "n"],
          [[r["field"], r["metric"], round(r["r"], 4), round(r["rho"], 4), round(r["p"], 4), r["n"]] for r in X["census"]["rows"]])
    # --- 10 m change from the annual Esri/IO maps (merged LandKit)
    X["esri10"] = None
    try:
        if g.aoi.area_km2 <= 5000:
            from .landkit import LandKit
            ya, yb = D.LULC_YEARS[0], D.LULC_YEARS[-1]
            kit.step(96.0, f"10 m land cover change {ya}–{yb} (annual Sentinel-2 maps)")
            sub = _os.path.join(out_dir, "change_10m")
            from ..core.data import TRACE
            saved = list(TRACE)
            try:
                r10 = LandKit(progress=None, cancelled=kit._cancelled).run(aoi, dict(year_from=ya, year_to=yb), sub)
            finally:
                new = [s for s in TRACE if s not in saved]
                TRACE[:] = saved + new
            rows = []
            with open(_os.path.join(sub, "class_areas.csv"), encoding="utf-8") as fh:
                next(fh)
                for line in fh:
                    nm_, a_, b_, d_ = line.strip().split(",")
                    if float(a_) > 0.001 or float(b_) > 0.001:
                        rows.append([nm_, fmt(float(a_)), fmt(float(b_)), f"{float(d_):+,.2f}"])
            X["esri10"] = dict(ya=ya, yb=yb, summary=r10["summary"], verdict=r10["verdict"], rows=rows)
            for lay in r10["layers"]:
                lay["hidden"] = True
                lay["name"] = "10 m " + lay["name"]
                X["layers"].append(lay)
            X["files"].append(_os.path.join(sub, "report.html"))
    except Exception as exc:  # never lose the main study for this extra
        X["esri10_error"] = str(exc)

    # --- Sentinel-2 10 m map of the latest year (beyond the last annual IO map)
    X["s2"] = None
    try:
        if latest >= 2017:
            from .lulc import s2map as S2
            kit.step(96.5, f"Sentinel-2 10 m land cover {latest}")
            s2 = S2.s2_lulc(aoi, latest, progress=lambda f, m: kit.step(96.5 + 0.4 * f, m))
            if s2:
                p_ = write_classes(_os.path.join(out_dir, f"lulc_sentinel2_{latest}.tif"), s2["map"], s2["grid"],
                                   {k: v for k, v in CL.items() if k in s2["areas"]})
                X["files"].append(p_)
                X["layers"].append(dict(path=p_, name=f"LULC {latest} (Sentinel-2 {s2['res']:.0f} m)", kind="raster", hidden=True))
                s2["path"] = p_
                X["s2"] = s2
    except Exception as exc:
        X["s2_error"] = str(exc)
    return X


EXTRA_REFS = [
    EX.CREDIT_POP, CP.CREDIT_SOIL, "OpenStreetMap contributors (roads), ODbL 1.0.",
    "UN-Habitat (2018). SDG Indicator 11.3.1 metadata: ratio of land consumption rate to population growth rate.",
    "Yeh, A. G.-O., Li, X. (2001). Measurement and monitoring of urban sprawl using Shannon's entropy. Photogramm. Eng. Remote Sens. 67, 83–90.",
    "Xu, H. (2013). A remote sensing urban ecological index and its application. Acta Ecologica Sinica 33, 7853–7862.",
    "Baig, M. H. A. et al. (2014). Derivation of a tasselled cap transformation based on Landsat 8 at-satellite reflectance. RS Letters 5, 423–431.",
    "Zhang, Y. et al. (2006). Quantitative analysis of urban thermal field variance (UTFVI). Ecological evaluation of urban heat islands.",
    "Sharp, R. et al. (2020). InVEST User's Guide (habitat quality, carbon). Natural Capital Project, Stanford University.",
    "IPCC (2006). Guidelines for National Greenhouse Gas Inventories, Vol. 4 AFOLU (Tier-1 carbon stocks).",
    "Xie, H. et al. (2013). Landscape ecological risk assessment. Ecological Indicators.",
    "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica 13, 245–259; Kendall, M. G. (1975). Rank Correlation Methods.",
    "Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. JASA 63, 1379–1389.",
    "Kennedy, R. E. et al. (2010). Detecting trends in forest disturbance and recovery (LandTrendr). RSE 114, 2897–2910.",
    "USDA NRCS (1986). Urban Hydrology for Small Watersheds, TR-55 (SCS curve numbers).",
    "van der Knijff, J. M. et al. (2000). Soil erosion risk assessment in Europe (NDVI-based C factor). EUR 19044 EN.",
    "Rahimi, A., Recht, B. (2007). Random features for large-scale kernel machines. NIPS 20.",
    "Foody, G. M. (2004). Thematic map comparison: evaluating the statistical significance of differences (McNemar). PE&RS 70, 627–633.",
    "Wang, J.-F. et al. (2010). Geographical detectors-based health risk assessment. IJGIS 24, 107–127.",
    "Fotheringham, A. S., Brunsdon, C., Charlton, M. (2002). Geographically Weighted Regression. Wiley.",
]


def _extra_data_rows(R):
    X = R["X"]
    rows = []
    if X["pop"]["total"]:
        rows.append(("JRC GHSL population (GHS-POP)", "Population, SDG 11.3.1, drivers", "100 m",
                     ", ".join(str(e) for e in sorted(set(X["pop"]["epoch"].values()))), "CC BY 4.0"))
    if X["hydro"]:
        rows.append(("SoilGrids 2.0 clay and sand (0–5 cm)", "Hydrologic soil groups for runoff", "250 m", "2020 release", "CC BY 4.0"))
    if X.get("roads_n"):
        rows.append(("OpenStreetMap major roads", "Driver: distance to road", "vector", "current", "ODbL 1.0"))
    if X["annual"]:
        rows.append(("Landsat every year (light composites)", "Trends and disturbance year", "60 m",
                     f"{int(X['annual']['years'][0])}–{int(X['annual']['years'][-1])}, 4 dates per year", "Public domain (USGS)"))
    if X.get("esri10"):
        rows.append(("IO/Microsoft/Esri annual land cover v02", "10 m change (second source)", "10 m",
                     f"{X['esri10']['ya']}, {X['esri10']['yb']}", "CC BY 4.0"))
    if X.get("s2"):
        rows.append(("Sentinel-2 L2A", "10 m land cover of the latest year", f"{X['s2']['res']:.0f} m",
                     f"{X['s2']['year']}, {len(X['s2']['scenes'])} scenes", "Copernicus open data"))
    return rows


def _extra_methods(R):
    X = R["X"]
    out = [
        ("Population and SDG 11.3.1", "Population from GHS-POP is summed over the area for each year. Land consumption rate "
         "LCR = ln(Urb₂/Urb₁)/y, population growth rate PGR = ln(Pop₂/Pop₁)/y, and their ratio LCRPGR (UN-Habitat 2018)."),
        ("Sprawl", "Shannon entropy of built-up area across 8 compass sectors and distance rings (Yeh & Li 2001), relative to its "
         "maximum ln(n); area-weighted compactness 2√(πA)/P and fractal dimension 2 ln(P/4)/ln(A) of built-up patches."),
        ("Ecological quality (RSEI)", "Greenness (NDVI), wetness (tasselled-cap, Baig et al. 2014), heat (LST) and dryness (NDBSI) "
         "are normalised and combined by principal components; PC1 is rescaled to 0–1 (Xu 2013). Water is excluded."),
        ("Urban thermal field (UTFVI)", "UTFVI = (Tₛ − Tmean)/Tₛ in kelvin, in six classes from 'none' to 'strongest' "
         "(Zhang et al. 2006)."),
        ("Carbon and habitat", "Carbon stock = area × indicative IPCC Tier-1 densities (above-ground, below-ground, soil, dead "
         "matter). Habitat quality follows the InVEST model: suitability reduced by distance-decayed threats from built-up land "
         "(8 km, exponential) and cropland (4 km, linear)."),
        ("Landscape ecological risk", "For each 1 km cell, area-weighted sum of class risk = disturbance (fragmentation, "
         "isolation, dominance; weights 0.5/0.3/0.2) × vulnerability rank; five classes from pooled quantiles."),
        ("Urban–rural gradient", "Landscape metrics computed in concentric rings from the urban centre."),
    ]
    if str((R.get("params") or {}).get("texture", "no")).lower() == "yes":
        out.append(("Texture features", "GLCM-like texture on NIR quantised to 8 grey levels: contrast (i−j)² and homogeneity 1/(1+(i−j)²) for "
                    "horizontal and vertical neighbours, averaged over 5×5 cells, plus 5×5 NDVI standard deviation (Haralick et al. 1973). "
                    "On the Gurugram test they changed held-out accuracy by −0.4 to +0.7 points (mean +0.3), so they are optional."))
    if X["annual"]:
        out.append(("Every-year trends", "For each year up to four clear Landsat dates are combined at 60 m. NDVI and LST "
                    "trends use the Mann-Kendall test (p ≤ 0.05) and Sen's slope; the year of vegetation loss is the year with the "
                    "largest sustained (2-year) NDVI drop above 0.15, a simplified LandTrendr-style rule. Tree-cover loss alerts are such "
                    "losses, after the first three years, on land that the first classified map calls forest / tree cover and whose NDVI averaged "
                    "at least 0.55 in the first three years."))
    if X["hydro"]:
        out.append(("Runoff and erosion", f"Hydrologic soil groups from SoilGrids clay and sand; SCS curve numbers (TR-55) by class "
                    f"and soil group; event runoff for a {X['hydro']['P']:g} mm storm. The RUSLE cover factor C = exp(−2·NDVI/(1−NDVI)) "
                    "(van der Knijff et al. 2000) shows how land cover changes erosion risk, other factors held constant."))
    if X["scen"]:
        out.append(("Scenarios", "Markov probabilities are adjusted for ecological protection and cropland protection and "
                    "allocated with the same cellular automata as business as usual."))
    if X["clf"]:
        out.append(("Classifier comparison", "On the same training and test samples: Random Forest, an RBF-kernel least-squares SVM "
                    "(random Fourier features), Maximum Likelihood and Minimum Distance; differences tested with McNemar's test (Foody 2004)."))
    if X["geo"]:
        out.append(("Geodetector and GWR", "New built-up share in 1 km cells is related to drivers. Geodetector q measures how much "
                    "of its variance each factor explains (quantile strata, permutation p) and how pairs interact (Wang et al. 2010). "
                    "Geographically weighted regression (adaptive bisquare kernel, leave-one-out bandwidth) shows where each "
                    "driver matters."))
    if X.get("s2"):
        out.append(("Sentinel-2 10 m map", "Clearest Sentinel-2 scene of each month (cloud and shadow masked with the scene "
                    "classification layer) gives median bands B2–B12, NDVI median/10th/90th percentile, MNDWI, NDBI and BSI, plus "
                    "elevation and slope. Training pixels are where IO/Esri 2021 and ESA WorldCover 2021 agree and IO/Esri 2023 "
                    "shows the same class (stable). Random forest (40 trees), 70/30 split, 3×3 majority filter."))
    if R["user"]:
        out.append(("Your own classes", "Your training points are the only training data. Points without a year are used for "
                    "the latest year; for other years only points whose spectral behaviour is unchanged are used. Your classes "
                    "(Level-II) are grouped into the eight main classes (Level-I) for all other analyses."))
    if X["plan"]:
        out.append(("Master plan", "Each planned land-use category is matched to the main classes it allows (e.g. residential → "
                    "built-up; green belt → forest, grass); the share of actual land cover that fits is the conformity."))
    if X["census"]:
        out.append(("Census relation", "Land-use statistics are averaged per unit and correlated with your fields (Pearson r, "
                    "Spearman ρ, permutation p with 999 shuffles)."))
    return out


def _report_extras(rep, R):
    X = R["X"]
    g = R["grid"]
    aoi = g.aoi
    years, latest, y0, present = R["years"], R["latest"], R["years"][0], R["present"]
    nm = {c: CL[c][0] for c in present}
    col = {c: CL[c][1] for c in present}
    src = "Landsat Collection 2 (USGS); own analysis"
    n = [14 if R["zonal"] and R["zonal"]["rows"] else 13]

    def H(t):
        rep.heading(f"4.{n[0]} {t}")
        n[0] += 1

    # population / SDG
    if X["sdg"]:
        H("Population and SDG 11.3.1")
        pt = X["pop"]["total"]
        rep.table(["Year", "Population (GHS-POP)", "Built-up (km²)", "Built-up per person (m²)"],
                  [[str(y), f"{pt[y]:,.0f}", fmt(R["areas"][y].get(2, 0)), f"{R['areas'][y].get(2, 0) * 1e6 / pt[y]:.1f}" if pt[y] else "–"]
                   for y in years], "Population and built-up land")
        rep.table(["Period", "Land consumption rate", "Population growth rate", "LCRPGR", "Reading"],
                  [[r["period"], "–" if r["lcr"] is None else f"{r['lcr']:.4f}", "–" if r["pgr"] is None else f"{r['pgr']:.4f}",
                    "–" if r["ratio"] is None else f"{r['ratio']:.2f}",
                    "–" if r["ratio"] is None else ("land consumed faster than population grew" if r["ratio"] > 1 else
                                                    "population grew faster than land consumption (densification)")]
                   for r in X["sdg"]], "SDG 11.3.1: ratio of land consumption rate to population growth rate")

    # sprawl
    H("Urban sprawl indices")
    sp = X["sprawl"]
    rep.table(["Year", "Entropy, sectors", "Relative", "Entropy, rings", "Relative", "Compactness", "Fractal dimension", "Built-up patches"],
              [[str(r["year"]), f"{r['H_sector']:.3f}", f"{r['H_sector_rel']:.3f}", f"{r['H_ring']:.3f}", f"{r['H_ring_rel']:.3f}",
                "–" if r["compact"] is None else f"{r['compact']:.3f}", "–" if r["fractal"] is None else f"{r['fractal']:.3f}",
                f"{r['patches']:,}"] for r in sp], "Shannon entropy, compactness and fractal dimension of built-up land")
    rep.text("Relative entropy close to 1 means built-up land is spread evenly across directions or distances (dispersed, sprawling "
             "growth); close to 0 means it is concentrated. Lower compactness and higher fractal dimension mean more irregular, "
             "fragmented urban form.")

    # RSEI
    if X["rsei"]:
        H("Ecological quality (RSEI)")
        rs = X["rsei"]
        rep.table(["Year", "Mean RSEI", "PC1 explains"] + [v[0] for v in EX.RSEI_CLASSES.values()],
                  [[str(y), f"{r['mean']:.3f}", f"{r['pc1_share']:.0f}%"] + [fmt(float((r["classes"] == k).sum() * g.px_km2))
                                                                             for k in EX.RSEI_CLASSES] for y, r in rs.items()],
                  "Remote Sensing Ecological Index: mean and area by class (km²)")
        ly = max(rs)
        rep.table(["Indicator", "PC1 loading"], [[k, f"{v:+.3f}"] for k, v in rs[ly]["loadings"].items()],
                  f"PC1 loadings, {ly} (greenness and wetness should be positive, heat and dryness negative)")
        rep.map(MapSpec(rs[ly]["classes"], g, f"Ecological quality (RSEI) {ly}", classes=EX.RSEI_CLASSES, subtitle=aoi.name, source=src),
                f"RSEI classes, {ly}.")

    # UTFVI
    H("Urban thermal field (UTFVI)")
    ut = X["utfvi"]
    rep.table(["Year"] + [f"{v[0]} · {EX.UTFVI_EEI[k]}" for k, v in EX.UTFVI_CLASSES.items()],
              [[str(y)] + [fmt(float((ut[y] == k).sum() * g.px_km2)) for k in EX.UTFVI_CLASSES] for y in years],
              "Area by UTFVI class and ecological evaluation (km²)")
    rep.map(MapSpec(ut[latest], g, f"Urban thermal field variance index {latest}", classes=EX.UTFVI_CLASSES, subtitle=aoi.name,
                    source=src), f"UTFVI, {latest}.")

    # carbon and habitat
    H("Carbon storage and habitat quality")
    cb = X["carbon"]
    rep.table(["Year", "Carbon stock (Mg C)", "Change from first year"],
              [[str(y), f"{v[0]:,.0f}", "–" if y == y0 else f"{v[0] - cb[y0][0]:+,.0f}"] for y, v in cb.items()],
              "Carbon stock (indicative IPCC Tier-1 densities; projected years included)")
    rep.table(["Class", "Above", "Below", "Soil", "Dead", "Total (Mg C/ha)"],
              [[CL[c][0]] + [str(x) for x in X["carbon_table"][c]] + [str(sum(X["carbon_table"][c]))] for c in present],
              "Carbon densities used (replace with local values for a carbon inventory)")
    rep.map(MapSpec(X["carbon_diff"], g, f"Change in carbon density {y0}–{latest}", ramp="heat", vmin=-150, vmax=150, unit="Mg C/ha",
                    subtitle="Negative = carbon lost", source=src), "Where carbon was lost or gained.")
    hq = X["hq"]
    rep.table(["Year", "Mean habitat quality (0–1)"], [[str(y), f"{float(np.nanmean(q)):.3f}"] for y, q in hq.items()],
              "Habitat quality (InVEST-style)")
    rep.map(MapSpec(hq[latest], g, f"Habitat quality {latest}", ramp="ndvi", vmin=0, vmax=1, unit="index", subtitle=aoi.name,
                    source=src), f"Habitat quality, {latest}.")

    # ecological risk
    H("Landscape ecological risk")
    rep.table(["Year", "Mean ERI"] + [v[0] for v in EX.ERI_CLASSES.values()],
              [[str(y), f"{float(np.nanmean(X['eri'][y])):.4f}"] + [fmt(float((X["eri_cls"][y] == k).sum() * g.px_km2)) for k in EX.ERI_CLASSES]
               for y in years], "Ecological risk index and area by risk class (km²; class breaks pooled over all years)")
    rep.map(MapSpec(X["eri_cls"][latest], g, f"Landscape ecological risk {latest}", classes=EX.ERI_CLASSES, subtitle="1 km cells",
                    source=src), f"Landscape ecological risk, {latest}.")

    # gradient
    gr = X["gradient"]
    if gr.get(latest):
        H("Urban–rural gradient")
        rows = []
        for y in (y0, latest):
            for r in gr[y]:
                rows.append([str(y), f"{(r['ring'] - 1) * R['ring_km']:.1f}–{r['ring'] * R['ring_km']:.1f}", f"{r['built']:.1f}",
                             f"{r['PD']:.2f}", f"{r['ED']:.1f}", f"{r['SHDI']:.3f}", f"{r['LPI']:.1f}"])
        rep.table(["Year", "Ring (km)", "Built-up (%)", "PD", "ED", "SHDI", "LPI"], rows, "Landscape metrics along the urban–rural gradient")

    # annual trends
    A = X["annual"]
    if A:
        H("Every-year trends and disturbance")
        yrs = [int(v) for v in A["years"]]
        mn, ml = A["mk_mean_ndvi"], A["mk_mean_lst"]
        rep.text(f"Mean NDVI trend: Sen's slope <b>{float(mn[2].ravel()[0]):+.4f} per year</b> (Mann-Kendall p = {float(mn[1].ravel()[0]):.3f}); "
                 f"mean LST trend: <b>{float(ml[2].ravel()[0]):+.3f} °C per year</b> (p = {float(ml[1].ravel()[0]):.3f}).")
        from ..core.maps import render_multi_line
        rep.figure(render_multi_line([("Mean NDVI", A["ndvi_mean"], "#2f7a3a")], np.array(yrs), "Year", "NDVI"),
                   "Mean NDVI every year.")
        rep.figure(render_multi_line([("Mean LST (°C)", A["lst_mean"], "#c4281b")], np.array(yrs), "Year", "°C"),
                   "Mean land surface temperature every year (median of up to four clear dates).")
        g60 = A["grid"]
        rep.table(["Trend class", "NDVI area (km²)", "LST area (km²)"],
                  [[v[0], fmt(float((A["tn"] == k).sum() * g60.px_km2)), fmt(float((A["tt"] == k).sum() * g60.px_km2))]
                   for k, v in AN.TREND_CLASSES.items()], "Pixel trends (Mann-Kendall, p ≤ 0.05 = significant)")
        rep.map(MapSpec(A["tn"], g60, f"NDVI trend {yrs[0]}–{yrs[-1]}", classes=AN.TREND_CLASSES, subtitle="Mann-Kendall",
                        source=src), "Where vegetation greened or browned significantly.")
        w = A["when"]
        cnt = [(y, float((w == y).sum() * g60.px_km2)) for y in yrs if (w == y).any()]
        if cnt:
            from ..core.maps import render_grouped_bars
            rep.figure(render_grouped_bars([str(y) for y, _ in cnt], [("Vegetation loss", [v for _, v in cnt], "#b2182b")], "km²"),
                       "Area of vegetation loss by the year it happened.")
        tl = A.get("tree_loss")
        if tl is not None:
            tcnt = {y: float((tl == y).sum() * g60.px_km2) for y in yrs}
            tot_t = sum(tcnt.values())
            if tot_t > 0:
                rep.map(MapSpec(np.where(tl > 0, tl, np.nan).astype(np.float32), g60, "Tree-cover loss by year", ramp="seq",
                                vmin=float(yrs[0]), vmax=float(yrs[-1]), unit="year", subtitle="Landsat every year, 60 m", source=src),
                        "Where tree cover of the first years (forest class and NDVI ≥ 0.55) lost its greenness, coloured by year.")
                recent = sum(v for y, v in tcnt.items() if y >= yrs[-1] - 1)
                rep.text(f"<b>{fmt(tot_t)} km²</b> of the {fmt(A['tree_px'])} km² of tree cover in {yrs[0]}–{yrs[min(2, len(yrs) - 1)]} "
                         f"lost its greenness for at least two years; <b>{fmt(recent)} km²</b> of it in the last two years "
                         "(recent alerts to check on the ground). Cutting, fire, drought and construction all show as loss.")
                rep.table(["Year", "Vegetation loss (km²)", "of which tree cover (km²)"],
                          [[str(y), fmt(float((w == y).sum() * g60.px_km2)), fmt(tcnt[y])] for y in yrs if (w == y).any() or tcnt[y] > 0],
                          "Vegetation and tree-cover loss by year")
            else:
                rep.text(f"No tree-cover loss was detected: none of the {fmt(A['tree_px'])} km² of tree cover in the first years lost its greenness "
                         "for two years or more.")

    # hydrology
    Hd = X["hydro"]
    if Hd:
        H("Runoff and erosion risk")
        rep.table(["Year", f"Mean runoff, {Hd['P']:g} mm storm (mm)", "Runoff volume (m³)", "Mean curve number", "Mean RUSLE C"],
                  [[str(y), f"{v['Q']:.1f}", f"{v['vol']:,.0f}", f"{v['CN']:.1f}", f"{v['C']:.3f}"] for y, v in Hd["years"].items()],
                  "Event runoff (SCS-CN) and erosion cover factor")
        rep.table(["Hydrologic soil group", "Area (km²)"], [[CP.HSG_NAMES[k], fmt(v)] for k, v in Hd["hsg"].items() if v > 0],
                  "Hydrologic soil groups from SoilGrids texture")
        rep.map(MapSpec(Hd["years"][latest]["Qmap"], g, f"Runoff depth {latest}, {Hd['P']:g} mm storm", ramp="depth", vmin=0,
                        vmax=Hd["P"], unit="mm", subtitle="SCS curve number method", source="Land cover + SoilGrids"),
                f"Event runoff depth, {latest}.")

    # scenarios
    S = X["scen"]
    if S:
        H(f"Land-use scenarios for {S['year']}")
        rep.table(["Class"] + list(S["runs"]), [[nm[c]] + [fmt(r["areas"][c]) for r in S["runs"].values()] for c in present]
                  + [["Ecosystem value (M US$/yr)"] + [f"{r['esv'] / 1e6:,.2f}" for r in S["runs"].values()],
                     ["Carbon stock (Mg C)"] + [f"{r['carbon']:,.0f}" for r in S["runs"].values()]],
                  f"Area by class (km²) under three scenarios, {S['year']}", [col[c] for c in present] + [None, None])
        rep.text(" ".join(f"<b>{k}:</b> {v}" for k, v in CP.SCENARIOS.items()))
        eco = S["runs"].get("Ecological protection")
        if eco:
            rep.map(MapSpec(eco["map"], g, f"Scenario {S['year']}: ecological protection", classes={c: CL[c] for c in present},
                            subtitle="Markov-CA", source=src, show_classes=present), "Ecological protection scenario.")

    # classifier comparison
    K = X["clf"]
    if K:
        H("Classifier comparison")
        rep.table(["Classifier", "Overall accuracy", "Kappa", "Mean F1", "McNemar χ² vs RF", "p", "Time (s)"],
                  [[r["name"], f"{r['oa'] * 100:.1f}%", f"{r['kappa']:.3f}", f"{r['f1']:.3f}", "–" if r["chi"] is None else f"{r['chi']:.1f}",
                    "–" if r["p"] is None else f"{r['p']:.3g}", f"{r['sec']:.1f}"] for r in K["rows"]],
                  f"Four classifiers on the same {K['n_train']:,} training and {K['n_test']:,} test samples ({K['year']}); test samples come from "
                  f"the same automatic labels, so this ranks the classifiers but is not an independent accuracy")
        rep.text("p < 0.05 means the accuracy difference from Random Forest is statistically significant.")

    # geodetector and GWR
    Gd = X["geo"]
    if Gd:
        H("Spatial drivers: Geodetector and GWR")
        rep.table(["Factor", "q (explanatory power)", "p"], [[f_["name"], f"{f_['q']:.3f}", f"{f_['p']:.3f}"] for f_ in
                                                            sorted(Gd["fac"], key=lambda d: -d["q"])],
                  f"Geodetector factor detector: new built-up share {y0}–{latest} in {Gd['n']} cells of 1 km")
        rep.table(["Factor A", "Factor B", "q(A ∩ B)", "Interaction"], [[i["a"], i["b"], f"{i['q']:.3f}", i["kind"]] for i in Gd["inter"]],
                  "Geodetector interaction detector (strongest pairs)")
        if Gd["gwr"]:
            rep.text(f"GWR (adaptive bandwidth {Gd['gwr']['k']} neighbours) explains <b>{Gd['gwr']['r2'] * 100:.0f}%</b> of the variance, "
                     f"against {Gd['gwr']['r2_global'] * 100:.0f}% for a single global regression: the drivers act differently in "
                     "different parts of the area.")
            if Gd["coef_map"] is not None:
                v = np.nanpercentile(np.abs(Gd["coef_map"]), 98) or 1
                rep.map(MapSpec(Gd["coef_map"], g, f"GWR local coefficient: {Gd['top']}", ramp="heat", vmin=-v, vmax=v,
                                unit="coef.", subtitle="Effect of the strongest factor varies by place", source=src),
                        f"Local effect of {Gd['top'].lower()} on new built-up land.")
        if X.get("roads_error"):
            rep.text("Distance to roads was not included: OpenStreetMap could not be reached during this run.")
        rep.text("Night-time lights were not included: the free global series (NASA Black Marble) requires an account.")

    # Level-II
    L2 = X.get("l2")
    if L2:
        H("Your own classes (Level-II)")
        cl2 = L2["classes"]
        rep.table(["Class", "Main class"] + [str(y) for y in years],
                  [[v[0], CL[v[2]][0]] + [fmt(L2["areas"][y][k]) for y in years] for k, v in cl2.items()],
                  "Area of your classes by year (km²)", [v[1] for v in cl2.values()])
        rep.map(MapSpec(R["maps_l2"][latest], g, f"Level-II land use {latest}", classes={k: (v[0], v[1]) for k, v in cl2.items()},
                        subtitle="Your classes", source=src), f"Your classes, {latest}.")

    # master plan
    Pl = X["plan"]
    if Pl:
        H("Land use against the master plan")
        rep.table(["Planned use", "Area (km²)", "Conforms", "Built-up in non-built zone (km²)", f"New built-up since {y0} (km²)"],
                  [[r["category"], fmt(r["area"]), "not recognised" if r["conformity"] is None else f"{r['conformity']:.0f}%",
                    fmt(r["built_in_nonbuilt"]), fmt(r["built_growth"])] for r in Pl["rows"]],
                  f"Actual land cover {latest} compared with planned use")
        rep.map(MapSpec(Pl["map"], g, "Conformity with the master plan", classes=UD.PLAN_CLASSES, subtitle=f"Land cover {latest}",
                        source="Your plan + own analysis"), "Where actual land use matches the plan.")

    # census
    Cs = X["census"]
    if Cs and Cs["rows"]:
        H("Relation to census and zone statistics")
        rep.table(["Your field", "Land-use measure", "Pearson r", "Spearman ρ", "p", "Units"],
                  [[r["field"], r["metric"], f"{r['r']:+.2f}", f"{r['rho']:+.2f}", f"{r['p']:.3f}", str(r["n"])] for r in Cs["rows"][:15]],
                  "Strongest correlations between your statistics and land use (p from 999 permutations)")

    # 10 m change (annual Sentinel-2 maps)
    E = X.get("esri10")
    if E:
        H(f"10 m land cover change {E['ya']}–{E['yb']} (second, independent source)")
        rep.text(f"A second view of recent change from the Impact Observatory/Microsoft/Esri annual 10 m maps "
                 f"(deep learning on Sentinel-2). {E['summary']} Its own checks give confidence <b>{E['verdict']['level']}</b>. "
                 f"The full chapter, maps and transition table are in <i>change_10m/report.html</i>. "
                 f"Where this and the Landsat classification agree on a trend, the trend is robust; class definitions differ "
                 f"(e.g. rangeland), so compare directions rather than exact km².")
        rep.table(["Class", f"{E['ya']} (km²)", f"{E['yb']} (km²)", "Change (km²)"], E["rows"],
                  f"Class areas from the 10 m annual maps, {E['ya']} and {E['yb']}")

    # Sentinel-2 10 m map
    S = X.get("s2")
    if S:
        H(f"Sentinel-2 {S['res']:.0f} m land cover {S['year']}")
        k_ = f", kappa {S['kappa']:.2f}" if S["kappa"] is not None else ""
        chk = []
        if S["agree_wc"] is not None:
            chk.append(f"{S['agree_wc'] * 100:.0f}% agreement with ESA WorldCover 2021")
        if S["agree_io"] is not None:
            chk.append(f"{S['agree_io'] * 100:.0f}% with the IO/Esri map of the same year")
        rep.text(f"A finer map of {S['year']} from {len(S['scenes'])} Sentinel-2 scenes, classified with a random forest trained "
                 f"on {S['n_train']:,} pixels where two independent 10 m maps agree and did not change 2021–2023. Held-out accuracy "
                 f"{S['oa'] * 100:.0f}%{k_}" + ("; " + "; ".join(chk) if chk else "") + ". Use it for detail inside the Landsat "
                 "study; for change over time keep to the Landsat maps, which are consistent since 1985.")
        tot = sum(S["areas"].values()) or 1
        rep.table(["Class", "Area (km²)", "Share", "F1 (held-out)"],
                  [[CL[k][0], fmt(a), f"{a / tot * 100:.1f}%", f"{S['f1'][k]:.2f}" if k in S["f1"] else "–"] for k, a in S["areas"].items()],
                  f"Sentinel-2 land cover {S['year']}", [CL[k][1] for k in S["areas"]])
        rep.map(MapSpec(S["map"], S["grid"], f"Land cover {S['year']} (Sentinel-2 {S['res']:.0f} m)",
                        classes={k: v for k, v in CL.items() if k in S["areas"]}, subtitle=aoi.name,
                        source="Copernicus Sentinel-2 L2A; own classification"), f"Sentinel-2 land cover, {S['year']}.")
