# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: Landslide: a complete landslide study for an area in one run.

Automatic landslide inventory from Sentinel-2 (new bare scars on slopes, event and year by year), nineteen
conditioning factors (terrain hydrology computed here: TWI, SPI, curvature, streams), ten susceptibility models
with spatial cross-validation, physically based slope stability, rainfall triggers, activity trend, exposure and
risk, and a thesis-style report with a proof appendix.
"""

import datetime as dt
import os
import warnings

import numpy as np

from ..core import datasets as D
from ..core import osm
from ..core.aoi import Grid, read_vector
from ..core.data import KitError
from ..core.raster import neighbours, write_classes, write_float, write_geojson
from .base import Kit
from .flood import extra as FX
from .flood import hydro as HY
from .flood import suscept as S
from .landslide import context as CX
from .landslide import detect as DT
from .landslide import models as LM
from .landslide import stability as ST
from .landslide import terrain as TR
from .landslide import dating as DA
from .landslide import runout as RO
from .lulc.classify import ghsl_fraction
from .lulc.compare import SOILGRIDS
from .lulc.extra import population
from .lulc.models import auc

SUSC = {1: ("Very low", "#2f7a4d"), 2: ("Low", "#9cc96b"), 3: ("Moderate", "#f2d36b"), 4: ("High", "#ec8b3a"), 5: ("Very high", "#b8322a")}
MODELS = [("fr", "Frequency Ratio", "#8c6bb1"), ("iv", "Information Value", "#6a51a3"), ("woe", "Weights of Evidence", "#9e9ac8"),
          ("ioe", "Index of Entropy", "#bcbddc"), ("ahp", "AHP (expert weights)", "#e08214"), ("lr", "Logistic Regression", "#2171b5"),
          ("rf", "Random Forest", "#238b45"), ("svm", "Support Vector Machine", "#d6604d"),
          ("gbm", "Gradient Boosting", "#41ab5d"), ("ens", "Ensemble", "#1c2621")]


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


def _soil(grid, var):
    from osgeo import gdal
    from ..core.data import trace, warp
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".vrt,.tif,.TIF,.tiff,.zip")
    u = SOILGRIDS.format(v=var)
    trace(url=u, label="soilgrids")
    try:
        a = warp(["/vsicurl/" + u], grid, resample="bilinear", dtype=gdal.GDT_Float32, src_nodata=-32768, dst_nodata=np.nan) / 10.0
        a[~grid.mask] = np.nan
        return a if np.isfinite(a).any() else None
    except Exception:
        return None


class LandslideStudyKit(Kit):
    NAME = "AreaGIS Kit: Landslide"
    KEY = "landslide"
    TAGLINE = "Complete landslide study: automatic inventory, 10 models, stability, runout, hazard, risk"
    DESCRIPTION = ("A complete landslide study in one run. Finds landslides automatically from Sentinel-2 (new bare scars on "
                   "slopes, for an event and year by year), derives up to 20 conditioning factors (lithology from Macrostrat), builds ten susceptibility models "
                   "(Frequency Ratio, Information Value, Weights of Evidence, Index of Entropy, AHP, Logistic Regression, Random "
                   "Forest, Support Vector Machine, Gradient Boosting, ensemble) with spatial cross-validation, maps physical slope stability, analyses rainfall triggers and "
                   "landslide activity, and measures people, buildings and roads at risk. Report in Word, PDF and HTML.")
    MAX_KM2 = 1500
    PARAMS = [
        dict(name="event_date", type="text", label="Date of a landslide-triggering event (YYYY-MM-DD; blank = none)", default=""),
        dict(name="inventory", type="polygons", label="Your landslide inventory (points or polygons; optional)", optional=True),
        dict(name="date_field", type="text", label="Date field in your inventory (optional, for rainfall thresholds)", default=""),
        dict(name="activity_years", type="int", label="Years of landslide activity to scan with Sentinel-2", default=6),
        dict(name="runout_angle", type="float", label="Runout travel angle in degrees (lower = debris travels further)", default=11.0),
        dict(name="soil_depth", type="float", label="Soil depth for slope stability (m)", default=1.5),
        dict(name="soil_depth_varies", type="bool", label="Let soil depth vary with slope shape (curvature)", default=True),
        dict(name="seismic", type="bool", label="Earthquake shaking factor (GEM hazard map is non-commercial; USGS ShakeMap)", default=True),
        dict(name="zones", type="polygons", label="Zones for statistics (villages, wards, districts)", optional=True),
        dict(name="zone_field", type="field", label="Zone name field", parent="zones", optional=True),
    ]

    # ================================================================== run
    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        R = dict(params=params)
        res = 20.0 if aoi.area_km2 <= 2000 else 30.0
        g = Grid(aoi, res, max_pixels=6_000_000)
        R["g"] = g
        ev = None
        if (params.get("event_date") or "").strip():
            try:
                ev = _d(params["event_date"].strip())
            except Exception:
                raise KitError("Give the event date as YYYY-MM-DD, or leave it blank.")
        R["ev"] = ev

        # ------------------------------------------------------------ factors
        self.step(3, "Reading elevation and computing terrain hydrology (fill, flow, TWI, curvature, streams)")
        dem = D.dem(g)
        slope, aspect = D.slope_aspect(dem, g)
        ti, streams = TR.indices(dem, g.mask, g.res)
        try:
            hand = D.hand(g)
        except KitError:
            hand = None
        self.step(15, "Reading land cover, vegetation, soil, faults, roads")
        lc_year = min(max((ev.year - 1) if ev else D.LULC_YEARS[-1], D.LULC_YEARS[0]), D.LULC_YEARS[-1])
        try:
            lc, _ = D.lulc(g, lc_year)
        except KitError:
            lc = D.worldcover(g)
        R["lc_year"] = lc_year
        today = dt.date.today()
        nd_end = (ev - dt.timedelta(days=10)) if ev else today
        ndvi, _, nd_dates = DT.ndvi_composite(aoi, g, nd_end - dt.timedelta(days=365), nd_end, max_scenes=8)
        R["ndvi_dates"] = nd_dates
        clay, sand = _soil(g, "clay"), _soil(g, "sand")
        try:
            dist_fault, fault_names = CX.fault_distance(g)
        except Exception as e:
            dist_fault, fault_names = None, []
            R["fault_err"] = str(e)
        R["faults"] = fault_names
        roads = None
        try:
            roads = osm.roads(aoi.bbox)
        except Exception as e:
            R["osm_err"] = str(e)
        dist_road = CX.road_distance(g, roads) if roads else None
        R["roads"] = roads
        geol = None
        try:
            from .landslide import geology as GE
            geol, R["geol_info"] = GE.lithology(g)
            if geol is not None and len([k for k in np.unique(geol[g.mask]) if k > 0]) < 2:
                R["geol_info"]["single"] = True
                geol = None
        except Exception as e:
            R["geol_err"] = str(e)
        pga = pga_ev = None
        if str(params.get("seismic", True)).lower() not in ("false", "0", "no", "off", ""):
            from .landslide import seismic as SM
            try:
                pga, R["pga_info"] = SM.hazard_pga(g)
            except Exception as e:
                R["pga_err"] = str(e)
            if ev:
                try:
                    sm_ev = SM.find_shakemap(ev, aoi.bbox)
                    if sm_ev:
                        R["shakemap"] = sm_ev
                        pga_ev = SM.event_pga(g, sm_ev)
                        if pga_ev is not None:
                            R["shakemap"]["min"], R["shakemap"]["max"] = float(np.nanmin(pga_ev)), float(np.nanmax(pga_ev))
                except Exception as e:
                    R["shakemap_err"] = str(e)
        relief = D.relief(dem, max(1, int(round(150 / g.res))))
        from .flood.depth import _box_mean
        tpi = dem - _box_mean(dem, max(1, int(round(250 / g.res))))
        dist_stream = g.distance_to(streams) if streams.any() else None
        dd = _box_mean(streams.astype(np.float32), max(1, int(round(500 / g.res)))) / g.res * 1000.0
        F = dict(slope=slope, relief=relief, lc=lc.astype(np.float32) if lc is not None else None, ndvi=ndvi,
                 dist_stream=dist_stream, twi=ti["twi"], plan=ti["plan"], profile=ti["profile"],
                 aspect=LM.aspect_classes(aspect, slope), elev=dem, spi=ti["spi"], tri=ti["tri"], tpi=tpi, dd=dd,
                 clay=clay, sand=sand, dist_fault=dist_fault, dist_road=dist_road, hand=hand,
                 geol=geol.astype(np.float32) if geol is not None else None, pga=pga, pga_event=pga_ev)
        F = {k: np.where(g.mask, v, np.nan).astype(np.float32) for k, v in F.items()
             if v is not None and np.isfinite(v[g.mask]).any() and np.nanstd(v[g.mask]) > 0}
        R["F"], R["streams"], R["acc"] = F, streams, ti["acc_km2"]

        # ------------------------------------------------------------ inventory
        self.step(30, "Finding landslides automatically with Sentinel-2")
        inv = np.zeros(g.shape, bool)
        R["inv_parts"] = {}
        if ev:
            try:
                e = DT.event_inventory(aoi, g, ev, slope, lc, hand=hand)
                if e:
                    R["event"] = e
                    inv |= e["map"]
                    R["inv_parts"]["event"] = float(e["map"].sum() * g.px_km2)
            except Exception as ex:
                R["event_err"] = str(ex)
        years = int(params.get("activity_years") or 6)
        act = {}
        first = np.zeros(g.shape, np.int16)
        comp = {}

        stacks, dated, scene_dates = {}, {}, set()

        def yearly(y):   # greenest-pixel composite of the year (one clear scene per month); scenes kept for dating
            if y not in comp:
                r_ = DT.ndvi_composite(aoi, g, dt.date(y, 1, 1), dt.date(y, 12, 31), max_scenes=12, stat="max", keep=True)
                comp[y], stacks[y] = r_[0], r_[3]
                scene_dates.update(r_[2])
            return comp[y]
        recov = {}

        def recovered(yy, nxt):
            """Real landslide scars stay bare for more than a year; pruned tea, coffee or harvested plantations green up again.
            Drop scar cells whose greenest NDVI one year later is back to at least 90% of the pre-scar value."""
            if yy not in act or nxt is None or pre_of.get(yy) is None:
                return
            back = act[yy] & np.isfinite(nxt) & (nxt >= 0.9 * pre_of[yy])
            if back.any():
                act[yy] &= ~back
                recov[yy] = float(back.sum() * g.px_km2)

        def date_year(yy):   # date each scar between the last green and first bare view (month level)
            if yy not in act:
                return
            m = act[yy]
            first[m & (first == 0)] = yy
            try:
                cnt, per = DA.date_scars(m, pre_of[yy], stacks.get(yy - 1), stacks.get(yy), pixels=True)
                for k_, n_ in cnt.items():
                    dated[k_] = dated.get(k_, 0) + n_
                rr_, cc_ = np.nonzero(m)
                for r0, c0, w_ in zip(rr_, cc_, per):
                    if w_ is not None and first[r0, c0] == yy:
                        mid = DA._d(w_[0]) + (DA._d(w_[1]) - DA._d(w_[0])) / 2
                        first[r0, c0] = mid.year
            except Exception as ex:
                R["dating_err"] = str(ex)

        pre_of = {}
        ev_next = None
        yrs_scan = list(range(today.year - years, today.year))
        for i, y in enumerate(yrs_scan):
            self.step(32 + 18 * i / max(years, 1), f"Scanning {y} against {y - 1} for new landslide scars")
            try:
                a, b = yearly(y), yearly(y - 1)
                if a is None or b is None:
                    continue
                recovered(y - 1, a)
                date_year(y - 1)
                comp.pop(y - 2, None)
                stacks.pop(y - 2, None)
                if ev and y == ev.year + 1:
                    ev_next = a
                m, _ = DT.scars(b, a, None, slope, lc, g.mask, hand=hand)
                m &= np.isfinite(a) & np.isfinite(b)
                act[y] = m
                pre_of[y] = b
            except Exception:
                continue
        if yrs_scan and yrs_scan[-1] in act:   # the last year: test recovery with this year's images so far (if any)
            try:
                nxt = yearly(today.year) if today.month >= 4 else None
            except Exception:
                nxt = None
            recovered(yrs_scan[-1], nxt)
            date_year(yrs_scan[-1])
        R["recovered"] = recov
        if ev and R.get("event") and ev_next is None and ev.year + 1 == today.year and today.month >= 4:
            try:
                ev_next = yearly(today.year)
            except Exception:
                ev_next = None
        if ev and R.get("event") and ev_next is not None:
            e_ = R["event"]
            back = e_["map"] & np.isfinite(ev_next) & (ev_next >= 0.9 * e_["ndvi_pre"])
            if back.any():
                e_["map"] = e_["map"] & ~back
                e_["recovered_km2"] = float(back.sum() * g.px_km2)
                inv &= ~back
                inv |= e_["map"]
                R["inv_parts"]["event"] = float(e_["map"].sum() * g.px_km2)
        R["recovery_next"] = {y: (y + 1) for y in act}
        stacks.clear()
        R["activity"], R["first_year"] = act, first
        R["dated"] = dict(windows=DA.windows(dated, g.res ** 2), scene_dates=sorted(scene_dates),
                          n_px=int(sum(dated.values())), n_scar_px=int(sum(int(v.sum()) for v in act.values())))
        if act:
            u = np.zeros(g.shape, bool)
            for m in act.values():
                u |= m
            inv |= u
            R["inv_parts"]["years"] = float(u.sum() * g.px_km2)
        user = None
        if params.get("inventory"):
            try:
                feats = read_vector(params["inventory"], aoi)
                geoms = [f[0] for f in feats]
                user = g.rasterize(geoms, all_touched=True) > 0
                if user.sum() < 10:  # points: mark the 3 x 3 cells around each
                    user = neighbours(user, 1) > 0
                R["user_n"] = len(feats)
                R["user_dates"] = [str(f[1].get(params.get("date_field"))) for f in feats] if params.get("date_field") else []
            except Exception as ex:
                R["user_err"] = str(ex)
        R["user"] = user
        R["inv"] = inv

        # ------------------------------------------------------------ susceptibility
        self.step(52, "Building ten susceptibility models with spatial cross-validation")
        train_inv = inv if inv.sum() >= 30 else (user if user is not None and user.sum() >= 30 else None)
        R["train_src"] = ("automatic (Sentinel-2)" if train_inv is inv else "your inventory") if train_inv is not None else None
        try:
            self._models(R, F, g, train_inv, user)
        except Exception as ex:
            R["model_err"] = str(ex)
        if R.get("sus") is None:
            idx, w, cr, lam, keys = LM.ahp(F, g.mask)
            R["sus"] = dict(maps={"ahp": S.norm01(idx, g.mask), "ens": S.norm01(idx, g.mask)}, weights=w, cr=cr, lam=lam,
                            ahp_keys=keys, only_ahp=True)
            R["sus"]["cls"], _ = S.classify5(R["sus"]["maps"]["ens"], g.mask)

        # ------------------------------------------------------------ physical stability
        self.step(68, "Physically based slope stability (infinite slope, dry to saturated)")
        z = float(params.get("soil_depth") or 1.5)
        vary = str(params.get("soil_depth_varies", True)).lower() not in ("false", "0", "no", "off", "")
        zmap = ST.soil_depth_map(ti["plan"], z) if vary else z
        R["soil_depth"] = dict(base=z, varies=vary)
        fs = {m: ST.factor_of_safety(slope, clay, sand, lc, depth=zmap, m=m) for m in (0.0, 0.5, 1.0)}
        R["fs"], R["fs_cls"] = fs, ST.classes(fs[0.0], fs[0.5], fs[1.0])
        R["fs_cls"][~g.mask] = 0
        if train_inv is not None:
            okm = g.mask & np.isfinite(fs[1.0])
            sm = S.sample(train_inv & okm, ~train_inv & okm & (np.nan_to_num(slope) >= 5), n=3000, seed=4)
            if sm is not None:
                R["fs_auc"] = auc(sm[1], -fs[1.0].ravel()[sm[0]])

        # ------------------------------------------------------------ rainfall
        self.step(74, "Rainfall: event, return periods, trend, landslide season")
        try:
            self._rain(R, aoi, ev, act)
        except Exception as ex:
            R["rain_err"] = str(ex)

        # ------------------------------------------------------------ exposure and risk
        self.step(80, "People, buildings, roads and settlements at risk")
        try:
            self._exposure(R, aoi, g, lc)
        except Exception as ex:
            R["exp_err"] = str(ex)

        self.step(84, "Where debris travels: runout (energy line, multiple flow direction)")
        try:
            self._runout(R, g, dem, slope, user)
        except Exception as ex:
            R["runout_err"] = str(ex)

        if ev and R.get("event") and R.get("activity"):
            self.step(86, "Hindcast: would a map made before the event have warned?")
            try:
                self._hindcast(R, F, g, dem, slope, ev)
            except Exception as ex:
                R["hindcast"] = dict(err=str(ex))

        self.step(87, "Landslide hazard over time (Poisson) and guidance")
        from .landslide import hazard as HZ
        try:
            if R.get("sus") and R["sus"].get("cls") is not None and R.get("activity"):
                R["hazard"] = HZ.temporal(R["activity"], R["sus"]["cls"], g.mask, g)
        except Exception as ex:
            R["hazard_err"] = str(ex)
        try:
            R["guidance"] = HZ.guidance(R)
        except Exception as ex:
            R["guidance_err"] = str(ex)

        if params.get("zones"):
            try:
                self._zones(R, params, aoi, g)
            except Exception as ex:
                R["zones_err"] = str(ex)

        self.step(88, "Writing maps and tables")
        files, layers = self._write(R, out_dir)
        self.step(93, "Writing the report")
        from .landslide.report_landslide import build
        return build(self, R, aoi, out_dir, files, layers)

    # ================================================================== models
    def _models(self, R, F, g, inv, user):
        mask = g.mask
        if inv is None:
            R["sus"] = None
            return
        slope = F["slope"]
        pos = inv & mask
        # absence: stable-looking ground away from known landslides (>100 m), on slopes above 5° (flat land is trivially safe)
        far = ~(neighbours(inv, max(1, int(round(100 / g.res)))) > 0)
        neg = mask & far & (np.nan_to_num(slope) >= 5)
        smp = S.sample(pos, neg, n=5000, seed=1)
        if smp is None:
            R["sus"] = None
            return
        idx, y = smp
        rng = np.random.default_rng(5)
        te = rng.random(idx.size) < 0.3
        itr, ytr, ite, yte = idx[~te], y[~te], idx[te], y[te]
        T, C = LM.tables(F, itr[ytr == 1], mask)
        biv, stats = LM.bivariate(T, C, mask)
        enc = {k: biv_lut for k, biv_lut in ((k, _encode_fr(T[k], C[k])) for k in LM.CATEGORICAL if k in T)}
        keys = [k for k, *_ in LM.FACTORS if k in F and k not in LM.CONTEXT_ONLY]
        ahp_idx, w, cr, lam, akeys = LM.ahp(F, mask)
        Xtr, Xte = LM.design(F, keys, itr, enc), LM.design(F, keys, ite, enc)
        aidx = np.flatnonzero(mask.ravel())
        Xall = LM.design(F, keys, aidx, enc)
        p_lr, b, se, pv = S.logistic_model(Xtr, ytr, Xall)
        p_rf, rfm = S.rf_model(Xtr, ytr, Xall, keys)
        from .landslide import ml as ML
        p_svm = ML.svm_model(Xtr, ytr, Xall, seed=1)
        p_gbm, _ = ML.gbm_model(Xtr, ytr, Xall, seed=1)
        imp = S.permutation_importance(rfm, Xte, yte, keys)

        def to_map(v):
            m = np.full(g.shape, np.nan, np.float32)
            m.ravel()[aidx] = v
            return m
        maps = {k: S.norm01(biv[k], mask) for k in ("fr", "iv", "woe", "ioe")}
        maps["ahp"] = S.norm01(ahp_idx, mask)
        maps["lr"], maps["rf"] = to_map(p_lr), to_map(p_rf)
        maps["svm"], maps["gbm"] = S.norm01(to_map(p_svm), mask), to_map(p_gbm)
        maps["ens"] = S.norm01(np.nanmean(np.stack([S.norm01(maps[k], mask)
                                                    for k in ("fr", "iv", "woe", "ioe", "ahp", "lr", "rf", "svm", "gbm")]), 0), mask)
        res = {}
        for k, *_ in MODELS:
            s = maps[k].ravel()[ite]
            ok = np.isfinite(s)
            res[k] = dict(auc_holdout=auc(yte[ok], s[ok]), roc=S.roc(yte[ok], s[ok]))
            sx, sy, sa = LM.success_rate(maps[k], pos, mask)
            res[k]["success"] = (sx, sy, sa)
        # independent check against the user's own inventory (never used for training)
        if user is not None and inv is not user and user.sum() >= 10:
            sm = S.sample(user & mask, neg & ~user, n=3000, seed=8)
            if sm is not None:
                for k in res:
                    s = maps[k].ravel()[sm[0]]
                    ok = np.isfinite(s)
                    res[k]["auc_user"] = auc(sm[1][ok], s[ok])
        # temporal check: train on earlier years' scars, predict later years' scars
        act = R.get("activity") or {}
        yrs = sorted(act)
        if len(yrs) >= 4:
            mid = yrs[len(yrs) // 2]
            early = np.zeros(g.shape, bool)
            late = np.zeros(g.shape, bool)
            for yy in yrs:
                (early if yy < mid else late)[...] |= act[yy]
            late &= ~early
            s1 = S.sample(early & mask, neg, n=3000, seed=11)
            s2 = S.sample(late & mask, neg & ~early, n=2000, seed=12)
            if s1 is not None and s2 is not None:
                T2, C2 = LM.tables(F, s1[0][s1[1] == 1], mask)
                b2, _ = LM.bivariate(T2, C2, mask)
                e2 = {k: _encode_fr(T2[k], C2[k]) for k in LM.CATEGORICAL if k in T2}
                X1 = LM.design(F, keys, s1[0], e2)
                X2 = LM.design(F, keys, s2[0], e2)
                pr = {"fr": b2["fr"].ravel()[s2[0]], "lr": S.logistic_model(X1, s1[1], X2)[0],
                      "rf": S.rf_model(X1, s1[1], X2, keys, seed=2)[0], "gbm": ML.gbm_model(X1, s1[1], X2, seed=2)[0]}
                R["temporal"] = dict(split=mid, auc={k: auc(s2[1], np.nan_to_num(v)) for k, v in pr.items()})
        block = max(4, int(round(1000 / g.res)))
        cv = LM.spatial_cv(F, keys, enc, pos, neg, mask, block)
        cls5, _ = S.classify5(maps["ens"], mask)
        dens = S.class_density(cls5, pos, mask)
        R["sus"] = dict(maps=maps, res=res, tables=T, stats=stats, weights=w, cr=cr, lam=lam, ahp_keys=akeys, keys=keys,
                        lr=dict(b=b, se=se, p=pv), rf_imp=imp, vif=S.vif(Xtr, keys), cv=cv, cv_block_m=block * g.res,
                        cls=cls5, density=dens, n_pos=int(pos.sum()))

    def _rain(self, R, aoi, ev, act):
        out = {}
        end = dt.date.today()   # whole record: the event window and every satellite-dated landslide window
        days, rain = HY.area_rainfall(aoi, dt.date(1981, 1, 1), end)
        idx = {d: i for i, d in enumerate(days)}
        yrs = {}
        for d, v in zip(days, rain):
            yrs.setdefault(d.year, []).append(v)
        full = {y: v for y, v in yrs.items() if len(v) >= 330}
        out["annual"] = {y: float(np.nansum(v)) for y, v in full.items()}
        out["annual_max"] = {y: float(np.nanmax(v)) for y, v in full.items()}
        out["trend_total"] = HY.mann_kendall([out["annual"][y] for y in sorted(out["annual"])])
        out["trend_max"] = HY.mann_kendall([out["annual_max"][y] for y in sorted(out["annual_max"])])
        mon = {}
        for d, v in zip(days, rain):
            mon.setdefault(d.month, []).append(v)
        out["month"] = {m: float(np.nanmean(v) * 30.4) for m, v in mon.items()}
        if ev and ev in idx:
            i = idx[ev]
            out["window"] = [(days[j], float(rain[j])) for j in range(max(0, i - 30), min(len(days), i + 8))]
            tot = {k: float(np.nansum(rain[max(0, i - k + 1):i + 1])) for k in (1, 3, 5, 7, 15, 30)}
            out["tot"] = tot
            rp = {}
            for k in (1, 3, 5, 7):
                am = HY.annual_max(days, rain, k)
                fit = HY.gumbel(am.values())
                if fit:
                    mx = max(float(np.nansum(rain[max(0, j - k + 1):j + 1])) for j in range(max(0, i - 5), i + 1))
                    D_, crit = HY.gof_ks(am.values(), fit)
                    rp[k] = dict(value=mx, rp=HY.return_period(mx, fit), q={T: HY.quantile(T, fit) for T in (2, 10, 25, 100)}, ks=D_, crit=crit)
            out["rp"] = rp
        # empirical event-duration threshold from dated landslides (user inventory dates)
        dates = []
        for s in R.get("user_dates") or []:
            try:
                dates.append(_d(s))
            except Exception:
                pass
        dates = sorted({d for d in dates if d in idx})
        if len(dates) >= 5:
            rows = []
            for k in (1, 3, 7, 15, 30):
                cum = np.array([np.nansum(rain[max(0, idx[d] - k + 1):idx[d] + 1]) for d in dates])
                allc = np.convolve(np.nan_to_num(rain), np.ones(k), mode="full")[:len(rain)]
                thr5 = float(np.percentile(cum, 5))
                rows.append(dict(days=k, median=float(np.median(cum)), thr5=thr5,
                                 exceed=float((allc >= thr5).mean() * 100)))
            out["threshold"] = dict(n=len(dates), rows=rows)
        wins = (R.get("dated") or {}).get("windows") or []
        if len(wins) >= 3:
            try:
                out["sat_threshold"] = DA.rain_thresholds(days, rain, wins, R["dated"]["scene_dates"])
            except Exception as ex:
                out["sat_threshold_err"] = str(ex)
        R["rain"] = out

    def _exposure(self, R, aoi, g, lc):
        X = {}
        built, ge = ghsl_fraction(g, 2020)
        pop, pe = population(g, dt.date.today().year)
        if built is None and lc is not None:
            built = np.where(g.mask, (lc == 7).astype(np.float32), np.nan)
            X["built_fallback"] = True
        X["pop_epoch"], X["ghsl_epoch"] = pe, ge
        cls = R["sus"]["cls"]
        rows = []
        for c in range(1, 6):
            m = cls == c
            rows.append(dict(cls=c, km2=float(m.sum() * g.px_km2),
                             pop=float(np.nansum(np.where(m, pop, 0))) if pop is not None else None,
                             built=float(np.nansum(np.where(m, built, 0)) * g.res ** 2 / 1e6) if built is not None else None))
        X["by_class"] = rows
        high = cls >= 4
        if built is not None:
            lab, st = FX.settlements(built, pop, high, g)
            X["settlements"] = sorted([r for r in st if r["flooded_pct"] > 0], key=lambda r: (-r["people_flooded"], -r["area_km2"] * r["flooded_pct"]))
            X["n_settlements"] = len(st)
        if R.get("roads"):
            rl = IM_road(R["roads"], g, high)
            if rl:
                X["roads_km"] = rl
        try:
            fac = osm.facilities(aoi.bbox, "all", buffer_deg=0)
            out = []
            for lo, la, n, t in fac:
                r, c = g.lonlat_to_pixel(lo, la)
                k = int(cls[r, c]) if (0 <= r < g.rows and 0 <= c < g.cols) else 0
                out.append((lo, la, n, t, k))
            X["facilities"] = out
        except Exception as ex:
            X["fac_err"] = str(ex)
        hz = R["sus"]["maps"]["ens"]
        if pop is not None or built is not None:
            X["risk_idx"], X["risk"] = FX.risk(hz, pop, built, g.mask)
            X["risk_rows"] = [dict(cls=c, km2=float((X["risk"] == c).sum() * g.px_km2),
                                   pop=float(np.nansum(np.where(X["risk"] == c, pop, 0))) if pop is not None else None) for c in range(1, 6)]
        dens = R["inv"].astype(np.float32) * 100
        cell = max(2, int(round(500 / g.res)))
        z, _ = FX.gistar(dens, g.mask, cell)
        if z is not None and R["inv"].sum() > 0:
            X["gi"] = FX.gi_classes(FX.expand(z, cell, g.shape))
            X["gi"][~g.mask] = 0
            X["gi_cell"] = cell * g.res
        R["exp"], R["pop"], R["built"] = X, pop, built

    def _hindcast(self, R, F, g, dem, slope, ev):
        """Rebuild susceptibility and runout from landslides seen BEFORE the event year only, then test them on the event's
        own scars. Answers the practical question: would a map made beforehand have flagged the slopes that failed and
        the valley the debris ran down?"""
        from .landslide import ml as ML
        mask = g.mask
        pre = np.zeros(g.shape, bool)
        yrs = sorted(y for y in R["activity"] if y < ev.year)
        for y in yrs:
            pre |= R["activity"][y]
        pre &= mask
        evt = R["event"]["map"] & mask & ~pre
        H = dict(years=(yrs[0], yrs[-1]) if yrs else None, pre_km2=float(pre.sum() * g.px_km2), evt_km2=float(evt.sum() * g.px_km2))
        R["hindcast"] = H
        if pre.sum() < 30 or evt.sum() < 20:
            H["err"] = ("Too few landslides were seen before the event to train a pre-event map." if pre.sum() < 30
                        else "Too few new event scars to test against.")
            return
        far = ~(neighbours(pre, max(1, int(round(100 / g.res)))) > 0)
        neg = mask & far & (np.nan_to_num(slope) >= 5)
        smp = S.sample(pre, neg & ~evt, n=4000, seed=21)
        if smp is None:
            H["err"] = "Could not sample pre-event landslides."
            return
        idx, y = smp
        T, C = LM.tables(F, idx[y == 1], mask)
        biv, _ = LM.bivariate(T, C, mask)
        enc = {k: _encode_fr(T[k], C[k]) for k in LM.CATEGORICAL if k in T}
        keys = [k for k, *_ in LM.FACTORS if k in F and k not in LM.CONTEXT_ONLY and k != "pga_event"]
        aidx = np.flatnonzero(mask.ravel())
        X, Xall = LM.design(F, keys, idx, enc), LM.design(F, keys, aidx, enc)

        def to_map(v):
            m = np.full(g.shape, np.nan, np.float32)
            m.ravel()[aidx] = v
            return m
        maps = {"fr": S.norm01(biv["fr"], mask), "lr": to_map(S.logistic_model(X, y, Xall)[0]),
                "rf": to_map(S.rf_model(X, y, Xall, keys, seed=3)[0]), "gbm": to_map(ML.gbm_model(X, y, Xall, seed=3)[0])}
        maps["ens"] = S.norm01(np.nanmean(np.stack([S.norm01(maps[k], mask) for k in ("fr", "lr", "rf", "gbm")]), 0), mask)
        cls, _ = S.classify5(maps["ens"], mask)
        test = S.sample(evt, neg & ~evt & ~pre, n=4000, seed=22)
        H["auc"] = {k: auc(test[1], np.nan_to_num(v).ravel()[test[0]]) for k, v in maps.items()} if test is not None else {}
        hi = mask & (cls >= 4)
        H["evt_in_high"] = float((evt & hi).sum() / evt.sum() * 100)
        H["high_share"] = float(hi.sum() / mask.sum() * 100)
        H["evt_in_vhigh"] = float((evt & (cls == 5)).sum() / evt.sum() * 100)
        H["vhigh_share"] = float((mask & (cls == 5)).sum() / mask.sum() * 100)
        # source of the event: the highest-lying event scar cells (top 5% by elevation)
        ze = dem[evt]
        top = evt & (dem >= np.nanpercentile(ze, 95))
        H["crown_cls"] = int(np.max(cls[neighbours(top, 2) > 0]))
        rr, cc = np.nonzero(top)
        k0 = int(np.argmax(dem[rr, cc]))
        H["crown"] = (*g.pixel_to_lonlat(rr[k0], cc[k0]), float(dem[rr[k0], cc[k0]]))
        # runout from what was known before: old scars plus very-high pre-event susceptibility
        src = RO.sources(pre, cls, slope, mask)
        angle = float(R["params"].get("runout_angle") or 11.0)
        ro = RO.runout(dem, src, mask, g.res, angle_deg=angle, vmax=15.0, exponent=4.0)
        tgt = evt & ~src
        ok = mask & ~src
        if tgt.sum() >= 20:
            H["debris_in_runout"] = float((tgt & ro["reach"]).sum() / tgt.sum() * 100)
            H["runout_share"] = float((ok & ro["reach"]).sum() / ok.sum() * 100)
            gentle = mask & (np.nan_to_num(slope) < 15)
            tg = tgt & gentle
            if tg.sum() >= 10:
                H["g_debris_in_runout"] = float((tg & ro["reach"]).sum() / tg.sum() * 100)
                H["g_runout_share"] = float((gentle & ro["reach"]).sum() / max(gentle.sum(), 1) * 100)
        # distance of debris travel actually observed vs predicted reach from the crown
        dist_c = g.distance_to(top)
        H["evt_length_m"] = float(np.nanmax(np.where(evt, dist_c, np.nan)))
        H["vmax_track"] = float(np.nanmax(np.where(evt & ro["reach"], ro["v"], np.nan))) if (evt & ro["reach"]).any() else None
        built, pop = R.get("built"), R.get("pop")
        if built is not None:
            bt = np.nan_to_num(built)
            hit_b = evt | (neighbours(evt, 1) > 0)
            H["built_hit_km2"] = float((bt * hit_b).sum() * g.res ** 2 / 1e6)
            H["built_hit_in_runout"] = float((bt * (hit_b & ro["reach"])).sum() / max((bt * hit_b).sum(), 1e-9) * 100)
            H["built_hit_in_high"] = float((bt * (hit_b & (hi | ro["reach"]))).sum() / max((bt * hit_b).sum(), 1e-9) * 100)
        if pop is not None:
            H["pop_runout"] = float(np.nansum(np.where(ro["reach"] & ~src, pop, 0)))
        H["cls"], H["reach"], H["src"] = cls, ro["reach"], src

    def _runout(self, R, g, dem, slope, user):
        src = RO.sources(R["inv"], R["sus"]["cls"], slope, g.mask)
        if src.sum() < 3:
            R["runout"] = None
            return
        angle = float(R["params"].get("runout_angle") or 11.0)
        ro = RO.runout(dem, src, g.mask, g.res, angle_deg=angle, vmax=15.0, exponent=4.0,
                       progress=lambda f: self.step(84 + 3 * f, "Where debris travels: runout"))
        down = ro["reach"] & ~src
        ro["km2"], ro["src_km2"] = float(down.sum() * g.px_km2), float(src.sum() * g.px_km2)
        dist = g.distance_to(src)
        ro["max_reach_m"] = float(np.nanmax(np.where(down, dist, np.nan))) if down.any() else 0.0
        pop, built = R.get("pop"), R.get("built")
        ro["pop"] = float(np.nansum(np.where(down, pop, 0))) if pop is not None else None
        ro["built_km2"] = float(np.nansum(np.where(down, built, 0)) * g.res ** 2 / 1e6) if built is not None else None
        ro["fast_km2"] = float((down & (np.nan_to_num(ro["v"]) >= 5)).sum() * g.px_km2)
        if built is not None:
            _, st = FX.settlements(built, pop, down, g)
            ro["settlements"] = sorted([r for r in st if r["flooded_pct"] > 0], key=lambda r: -r["people_flooded"])
        if R.get("roads"):
            rl = IM_road(R["roads"], g, down)
            if rl:
                ro["roads_km"] = rl
        fac = (R.get("exp") or {}).get("facilities") or []
        hit = []
        for lo, la, n, t_, _ in fac:
            r, c = g.lonlat_to_pixel(lo, la)
            if 0 <= r < g.rows and 0 <= c < g.cols and down[r, c]:
                hit.append((lo, la, n, t_, float(ro["v"][r, c]) if np.isfinite(ro["v"][r, c]) else None))
        ro["facilities"] = hit
        # validation: does mapped debris (inventory outside the release cells) fall where runout says it can?
        if user is not None and (user & ~src & g.mask).sum() >= 20:
            tgt = user & ~src & g.mask
            ok = g.mask & ~src
            ro["val"] = dict(hit=float((tgt & ro["reach"]).sum() / tgt.sum() * 100),
                             area=float((ok & ro["reach"]).sum() / ok.sum() * 100))
            gentle = g.mask & (np.nan_to_num(slope) < 15)
            tg = tgt & gentle
            if tg.sum() >= 10:
                ro["val"]["g_hit"] = float((tg & ro["reach"]).sum() / tg.sum() * 100)
                ro["val"]["g_area"] = float((gentle & ro["reach"]).sum() / max(gentle.sum(), 1) * 100)
            sm = S.sample(tgt, ok & ~user, n=4000, seed=5)
            if sm is not None:
                ro["val"]["auc"] = auc(sm[1], np.nan_to_num(ro["p"]).ravel()[sm[0]])
                ro["val"]["auc_sus"] = auc(sm[1], np.nan_to_num(R["sus"]["maps"]["ens"]).ravel()[sm[0]])
        R["runout"] = ro

    def _zones(self, R, params, aoi, g):
        feats = read_vector(params["zones"], aoi)
        fld = params.get("zone_field")
        z = g.rasterize([x for x, _, _ in feats], values=list(range(1, len(feats) + 1)))
        rows = []
        for i, (_, a, _) in enumerate(feats):
            m = (z == i + 1) & g.mask
            if not m.any():
                continue
            rows.append(dict(name=str(a.get(fld, i + 1)) if fld else str(i + 1), area=float(m.sum() * g.px_km2),
                             high=float((m & (R["sus"]["cls"] >= 4)).sum() / m.sum() * 100),
                             ls=float((m & R["inv"]).sum() * g.px_km2),
                             pop_high=float(np.nansum(np.where(m & (R["sus"]["cls"] >= 4), R["pop"], 0))) if R.get("pop") is not None else None))
        R["zones"] = sorted(rows, key=lambda r: -r["high"])

    # ================================================================== files
    def _write(self, R, out_dir):
        g = R["g"]
        files, layers = [], []

        def add(path, name, hidden=True, kind="raster"):
            files.append(path)
            layers.append(dict(path=path, name=name, kind=kind, hidden=hidden))
        sus = R["sus"]
        add(write_classes(os.path.join(out_dir, "landslide_susceptibility_classes.tif"), sus["cls"], g, SUSC),
            "Landslide susceptibility (ensemble)", hidden=False)
        for k, nm, _ in MODELS:
            if k in sus["maps"]:
                files.append(write_float(os.path.join(out_dir, f"susceptibility_{k}.tif"), sus["maps"][k], g, "risk", 0, 1, "")[0])
        inv = np.where(g.mask, np.where(R["inv"], 1, 0), 0).astype(np.uint8)
        add(write_classes(os.path.join(out_dir, "landslides_detected.tif"), inv, g, {1: ("Landslide scar (Sentinel-2)", "#7f2704")}),
            "Landslides found by Sentinel-2", hidden=False)
        if R["first_year"].any():
            add(write_float(os.path.join(out_dir, "landslide_year_first_seen.tif"), np.where(R["first_year"] > 0, R["first_year"], np.nan), g,
                            "seq", float(min(R["activity"])), float(max(R["activity"])), "year")[0], "Year each scar appeared")
        add(write_classes(os.path.join(out_dir, "slope_stability_classes.tif"), R["fs_cls"], g, ST.CLASSES), "Slope stability (infinite slope)")
        files.append(write_float(os.path.join(out_dir, "factor_of_safety_saturated.tif"), R["fs"][1.0], g, "risk", 0, 3, "FS")[0])
        for k, a in R["F"].items():
            files.append(write_float(os.path.join(out_dir, f"factor_{k}.tif"), a, g, "seq", float(np.nanpercentile(a, 2)),
                                     float(np.nanpercentile(a, 98)), "")[0])
        files.append(write_float(os.path.join(out_dir, "flow_accumulation_km2.tif"), R["acc"], g, "depth", 0, 5, "km²")[0])
        X = R.get("exp") or {}
        if X.get("risk") is not None:
            add(write_classes(os.path.join(out_dir, "landslide_risk_classes.tif"), X["risk"], g, FX.RISK), "Landslide risk")
        if X.get("gi") is not None:
            add(write_classes(os.path.join(out_dir, "landslide_hotspots_gistar.tif"), X["gi"], g, FX.GISTAR), "Landslide hot spots (Gi*)")
        ro = R.get("runout")
        if ro:
            add(write_classes(os.path.join(out_dir, "runout_zones.tif"), ro["cls"], g, RO.CLASSES), "Landslide runout (where debris travels)")
            files.append(write_float(os.path.join(out_dir, "runout_frequency.tif"), ro["p"], g, "risk", 0, 1, "")[0])
            add(write_float(os.path.join(out_dir, "runout_max_velocity.tif"), ro["v"], g, "risk", 0, 15, "m/s")[0], "Runout velocity (m/s)")
            if ro.get("settlements"):
                p = os.path.join(out_dir, "settlements_in_runout.csv")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write("rank,lat,lon,area_km2,people,runout_pct,people_in_runout\n")
                    for i, r in enumerate(ro["settlements"]):
                        fh.write(f"{i + 1},{r['lat']:.5f},{r['lon']:.5f},{r['area_km2']:.4f},{r['people']:.0f},{r['flooded_pct']:.1f},{r['people_flooded']:.0f}\n")
                files.append(p)
        dw = (R.get("dated") or {}).get("windows")
        if dw:
            p = os.path.join(out_dir, "landslide_dates_from_satellite.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("last_green_view,first_bare_view,window_days,area_km2\n")
                for w_ in dw:
                    fh.write(f"{w_['start']},{w_['end']},{w_['days']},{w_['km2']:.4f}\n")
            files.append(p)
        if X.get("settlements"):
            p = os.path.join(out_dir, "settlements_at_risk.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("rank,lat,lon,area_km2,people,high_susceptibility_pct,people_in_high\n")
                for i, r in enumerate(X["settlements"]):
                    fh.write(f"{i + 1},{r['lat']:.5f},{r['lon']:.5f},{r['area_km2']:.4f},{r['people']:.0f},{r['flooded_pct']:.1f},{r['people_flooded']:.0f}\n")
            files.append(p)
        if sus.get("tables"):
            p = os.path.join(out_dir, "bivariate_tables.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("factor,class,range,pixels,landslide_pixels,FR,IV,W_plus,W_minus,contrast\n")
                for k, rows in sus["tables"].items():
                    for r in rows:
                        fh.write(f"{k},{r['cls']},\"{r['label']}\",{r['npx']:.0f},{r['nls']:.0f},{r.get('fr', 0):.4f},{r.get('iv', 0):.4f},"
                                 f"{r.get('w_plus', 0):.4f},{r.get('w_minus', 0):.4f},{r.get('contrast', 0):.4f}\n")
            files.append(p)
        if R["activity"]:
            p = os.path.join(out_dir, "landslide_activity_by_year.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("year,new_scar_area_km2\n")
                for y in sorted(R["activity"]):
                    fh.write(f"{y},{R['activity'][y].sum() * g.px_km2:.4f}\n")
            files.append(p)
        # validation points
        rng = np.random.default_rng(7)
        pts = []
        for lab, sel in (("Landslide scar", R["inv"]), ("Very high susceptibility, no scar", (sus["cls"] == 5) & ~R["inv"])):
            idx = np.flatnonzero(sel.ravel())
            for q in (rng.choice(idx, size=min(25, idx.size), replace=False) if idx.size else []):
                r, c = divmod(int(q), g.cols)
                pts.append((g.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "map_says": lab, "correct": None}))
        add(write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts), "Validation points", kind="vector")
        hz = R.get("hazard")
        if hz is not None and hz.get("map") is not None:
            from .landslide.hazard import HAZ
            add(write_classes(os.path.join(out_dir, "landslide_hazard_50yr.tif"), hz["map"], g, HAZ), "Landslide hazard (chance in 50 years)")
            fn = os.path.join(out_dir, "landslide_hazard_by_class.csv")
            with open(fn, "w", encoding="utf-8", newline="") as fh:
                fh.write("susceptibility_class,area_km2,landslides,per_km2_per_year,p_1yr,p_10yr,p_50yr\n")
                for r_ in hz["rows"]:
                    fh.write(f"{r_['cls']},{r_['area']:.4f},{r_['n']},{r_['rate']:.5f},{r_['p'][1]:.4f},{r_['p'][10]:.4f},{r_['p'][50]:.4f}\n")
            files.append(fn)
        if "geol" in R["F"]:
            from .landslide.geology import NAMES as GN
            gc = np.nan_to_num(R["F"]["geol"]).astype(np.uint8)
            cols_ = ["#f6e8c3", "#c7a252", "#e7d4e8", "#dfc27d", "#80cdc1", "#f4a582", "#b2182b", "#8073ac", "#bdbdbd"]
            add(write_classes(os.path.join(out_dir, "factor_lithology.tif"), gc, g,
                              {k: (GN[k], cols_[k - 1]) for k in GN if (gc == k).any()}), "Lithology (Macrostrat)")
        return files, layers


def _encode_fr(rows, c):
    n = int(c.max()) + 1
    arr = np.zeros(n, np.float32)
    for r in rows:
        arr[r["cls"]] = r.get("fr", 1.0)
    return arr[c]


def IM_road(roads, g, high):
    from .flood.impact import road_length
    r = road_length(roads, g, high)
    return r
