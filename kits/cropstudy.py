# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: Crop: a complete crop study for an area and a crop year in one run.

Cropland, Sentinel-2 crop calendar (10-day NDVI, phenology per pixel), cropping intensity and seasons, sowing and
harvest timing, crop groups without field data, paddy rice from Sentinel-1 radar, supervised crop map with your field
points, 25-year crop condition from MODIS (anomaly, VCI, drought years, trend), rainfall, residue burning, field
statistics and management zones, validation (USDA CDL in the USA, your points elsewhere), report and proof.
"""

import datetime as dt
import os

import numpy as np

from ..core import datasets as D
from ..core.aoi import Grid, read_vector
from ..core.data import KitError, search, warp_items
from ..core.raster import write_classes, write_float, write_geojson
from .base import Kit
from .crop import modis as MO
from .crop import series as SE
from .crop import types as TY
from .crop import yieldfield as YF
from .crop import water as WA
from .flood import hydro as HY
from .lulc.models import auc

INTENSITY = {1: ("Fallow / no crop this year", "#d9c9a3"), 2: ("One crop", "#a6d96a"), 3: ("Two crops", "#1a9641"),
             4: ("Three crops", "#00441b")}


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


class CropStudyKit(Kit):
    NAME = "AreaGIS Kit: Crop"
    KEY = "cropstudy"
    TAGLINE = "Complete crop study: calendar, intensity, rice, condition, drought, burning"
    DESCRIPTION = ("A complete crop study for one crop year in one run: cropland, a Sentinel-2 crop calendar for every pixel "
                   "(sowing, peak, harvest), cropping intensity, crop groups without field data, paddy rice from Sentinel-1 radar, "
                   "a crop map if you add field points, 25 years of crop condition from MODIS (anomaly, VCI, drought years, trend), "
                   "rainfall, crop residue burning, field statistics and management zones. Report in Word, PDF and HTML.")
    MAX_KM2 = 1500
    PARAMS = [
        dict(name="start", type="text", label="Crop year starts (YYYY-MM-DD; blank = the last 12 months)", default=""),
        dict(name="groups", type="int", label="Number of crop groups to find", default=6),
        dict(name="training", type="points", label="Field points with crop names (optional, for a crop map)", optional=True),
        dict(name="crop_field", type="field", label="Crop name field", parent="training", optional=True),
        dict(name="fields", type="polygons", label="Field boundaries (optional, for field statistics and zones)", optional=True),
        dict(name="zones", type="polygons", label="Zones for statistics (villages, blocks, districts)", optional=True),
        dict(name="zone_field", type="field", label="Zone name field", parent="zones", optional=True),
        dict(name="yield_zones", type="polygons", label="Yield statistics by zone (optional; districts/blocks with reported yield, for t/ha)", optional=True),
        dict(name="yield_field", type="field", label="Reported yield field (e.g. t/ha, same crop year)", parent="yield_zones", optional=True),
        dict(name="yield_unit", type="text", label="Unit of the reported yield", default="t/ha"),
    ]

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        R = dict(params=params)
        today = dt.date.today()
        st = (params.get("start") or "").strip()
        if st:
            try:
                start = _d(st)
            except Exception:
                raise KitError("Give the crop-year start as YYYY-MM-DD, or leave it blank.")
        else:
            m0 = dt.date(today.year, today.month, 1)
            start = (m0 - dt.timedelta(days=360)).replace(day=1)
        if start + dt.timedelta(days=360) > today + dt.timedelta(days=5):
            raise KitError("The crop year has not finished yet; choose a start at least 12 months ago.")
        R["start"] = start
        g = Grid(aoi, 20, max_pixels=4_500_000)
        R["g"] = g

        # ------------------------------------------------------------ cropland
        self.step(3, "Cropland mask (ESA WorldCover, Esri land cover)")
        wc = D.worldcover(g)
        try:
            lc, _ = D.lulc(g, min(max(start.year, D.LULC_YEARS[0]), D.LULC_YEARS[-1]))
        except KitError:
            lc = None
        crop0 = np.zeros(g.shape, bool)
        if wc is not None:
            crop0 |= wc == 5
        if lc is not None:
            crop0 |= lc == 5
        crop0 &= g.mask
        if crop0.sum() < 100:
            raise KitError("Almost no cropland in this area according to WorldCover and Esri land cover.")
        R["wc"], R["lc"] = wc, lc

        # ------------------------------------------------------------ Sentinel-2 calendar
        def prog(i, n):
            self.step(5 + 30 * i / max(n, 1), f"Reading Sentinel-2 crop calendar ({i}/{n} ten-day periods)")
        Y, centers, used = SE.ndvi_series(aoi, g, start, progress=prog)
        R["centers"], R["used"] = centers, used
        self.step(36, "Filling cloud gaps and smoothing the NDVI curves")
        S = SE.fill_smooth(Y, crop0)
        del Y
        self.step(40, "Finding crop seasons for every pixel (sowing, peak, harvest)")
        P = SE.phenology(S, centers)
        R["S"], R["P"] = S, P
        amp = np.nanmax(S, 0) - np.nanmin(S, 0)
        crop = crop0 & np.isfinite(amp)
        R["crop"] = crop
        n = P["n"]
        inten = np.zeros(g.shape, np.uint8)
        inten[crop] = 1 + np.clip(n[crop], 0, 3)
        R["intensity"] = inten
        R["ci"] = float(np.mean(n[crop])) * 100   # cropping intensity %
        # season areas by peak month
        pk = P["pos"]
        smon = {}
        for i in range(3):
            v = pk[i][crop & np.isfinite(pk[i])].astype(int)
            for t, c in zip(*np.unique(v, return_counts=True)):
                mth = centers[t].month
                smon[mth] = smon.get(mth, 0) + c * g.px_km2
        R["season_month"] = smon
        R["crop_curve"] = [float(np.nanmean(S[t][crop])) for t in range(S.shape[0])]

        # ------------------------------------------------------------ yield index (Sentinel-2)
        best, tot = YF.main_integral(P)
        R["yield_integral"] = best
        R["yield_index"], R["yield_median"] = YF.relative_index(best, crop)
        R["season_window"] = YF.season_window(P, crop, centers)

        # ------------------------------------------------------------ crop groups
        self.step(46, "Grouping fields by crop calendar")
        k = max(3, min(int(params.get("groups") or 6), 10))
        gm, prof, labels = TY.calendar_groups(S, crop, centers, k=k)
        R["groups"] = (gm, prof, labels)

        # ------------------------------------------------------------ rice from radar
        self.step(50, "Looking for paddy rice with Sentinel-1 radar")
        try:
            vh, vdates = TY.s1_series(aoi, g, start)
            if vh is not None:
                R["rice"] = TY.rice(vh, crop, vdates, P)
                # isolated pixels are mostly wet soil or speckle: keep paddy pixels with >= 4 paddy neighbours
                # (cached CDL tests: California correct 82% -> 84%, false paddy on other crops -11%; Arkansas -32%)
                from ..core.raster import neighbours as _nb
                R["rice"] = R["rice"] & (_nb(R["rice"], 1) >= 4)
                R["vh_dates"] = vdates
                R["vh_curve_rice"] = [float(np.nanmean(vh[t][R["rice"]])) if R["rice"].any() else None for t in range(vh.shape[0])]
                R["vh_curve_other"] = [float(np.nanmean(vh[t][crop & ~R["rice"]])) for t in range(vh.shape[0])]
                R["vh"] = vh
        except Exception as e:
            R["rice_err"] = str(e)

        # ------------------------------------------------------------ supervised crop map
        if params.get("training"):
            self.step(55, "Training a crop map from your field points")
            try:
                self._supervised(R, params, aoi, g)
            except Exception as e:
                R["sup_err"] = str(e)

        # ------------------------------------------------------------ MODIS long record
        self.step(58, "25 years of crop condition from MODIS")
        try:
            gm250 = Grid(aoi, 250, max_pixels=200_000)
            cm = D.worldcover(gm250)
            cm250 = (cm == 5) & gm250.mask if cm is not None else gm250.mask

            def mp(i, n):
                self.step(58 + 12 * i / max(n, 1), f"Reading MODIS NDVI record ({i}/{n})")
            rec, arrs = MO.ndvi_record(aoi, gm250, cm250, progress=mp)
            R["modis"] = rec
            R["modis_clim"] = MO.climatology(rec)
            months = sorted({c.month for c in centers})
            R["modis_year"] = MO.yearly(rec)
            yi = [(y, v["integral"]) for y, v in sorted(R["modis_year"].items()) if y < today.year]
            R["modis_trend"] = HY.mann_kendall([v for _, v in yi]) if len(yi) >= 8 else None
            # VCI for the most recent composite of the crop year, against all years for that 16-day bin
            if arrs:
                last_k = max(k_ for k_ in arrs if k_ <= start + dt.timedelta(days=360))
                b = (last_k.timetuple().tm_yday - 1) // 16
                same = [arrs[k_].astype(np.float32) for k_ in arrs if (k_.timetuple().tm_yday - 1) // 16 == b]
                if len(same) >= 8:
                    st_ = np.stack(same)
                    R["vci"] = (MO.vci(arrs[last_k].astype(np.float32), np.nanmin(st_, 0), np.nanmax(st_, 0)), last_k, gm250, cm250)
            R["g250"] = gm250
            if arrs:
                R["ndvi_season_y"] = WA.season_ndvi(arrs, R["season_window"])
                R["cm250"] = cm250
            del arrs
        except Exception as e:
            R["modis_err"] = str(e)

        # ------------------------------------------------------------ vegetation health (VCI + TCI)
        if R.get("ndvi_season_y"):
            self.step(69, "Vegetation Health Index: MODIS land surface temperature every season since 2001")
            try:
                gm250 = R["g250"]

                def lp(i, n):
                    self.step(69 + i / max(n, 1), f"Reading MODIS land surface temperature ({i}/{n})")
                lst_y = WA.season_lst(aoi, gm250, R["season_window"], sorted(R["ndvi_season_y"]), progress=lp)
                R["vhi"] = WA.vhi(R["ndvi_season_y"], lst_y, R["cm250"])
            except Exception as e:
                R["vhi_err"] = str(e)
            R.pop("ndvi_season_y", None)

        # ------------------------------------------------------------ yield anomaly (MODIS GPP)
        self.step(70, "Yield anomaly every year since 2001 (MODIS gross primary production)")
        try:
            g500y = Grid(aoi, 500, max_pixels=100_000)
            cmy = D.worldcover(g500y)
            cmy = (cmy == 5) & g500y.mask if cmy is not None else g500y.mask

            def gp(i, n):
                self.step(70 + 2 * i / max(n, 1), f"Reading MODIS GPP record ({i}/{n})")
            rec, maps, med = YF.gpp_record(aoi, g500y, cmy, R["season_window"], progress=gp)
            R["gpp"], R["gpp_median"], R["g500y"], R["gpp_crop"] = rec, med, g500y, cmy
            R["gpp_maps"] = maps
            R["gpp_dropped"] = list(getattr(YF.gpp_record, "dropped", []))
            if maps:
                ly = max(maps)
                R["gpp_last"] = (ly, np.where(cmy, (maps[ly] / med - 1) * 100, np.nan) if med else None)
            if R.get("modis"):
                a_, b_ = R["season_window"]
                mons = sorted({(a_.replace(day=1) + dt.timedelta(days=31 * i)).month for i in range(12)
                               if (a_.replace(day=1) + dt.timedelta(days=31 * i)) <= b_ + dt.timedelta(days=(365 if b_ < a_ else 0))})
                ny = MO.yearly(R["modis"], months=mons)
                common = [y for y in rec if y in ny]
                if len(common) >= 8:
                    R["gpp_ndvi_r"] = float(np.corrcoef([rec[y]["gpp"] for y in common], [ny[y]["integral"] for y in common])[0, 1])
        except Exception as e:
            R["gpp_err"] = str(e)

        # ------------------------------------------------------------ evapotranspiration and water productivity
        if R.get("g500y") is not None:
            self.step(72, "Evapotranspiration and water productivity (MODIS ET)")
            try:
                et, etm = WA.et_record(aoi, R["g500y"], R["gpp_crop"])
                R["et"] = et
                gp = R.get("gpp") or {}
                R["wp"] = {y: gp[y]["gpp"] / et[y] for y in et if y in gp and et[y] > 0}
                both = [y for y in etm if y in (R.get("gpp_maps") or {})]
                if both:
                    y = max(both)
                    with np.errstate(all="ignore"):
                        R["wp_map"] = (y, np.where(R["gpp_crop"] & (etm[y] > 0), R["gpp_maps"][y] / etm[y], np.nan).astype(np.float32))
                    R["et_map"] = (y, np.where(R["gpp_crop"], etm[y], np.nan).astype(np.float32))
                ys_ = [y for y in sorted(et) if y < today.year]
                R["et_trend"] = HY.mann_kendall([et[y] for y in ys_]) if len(ys_) >= 8 else None
            except Exception as e:
                R["et_err"] = str(e)
            R.pop("gpp_maps", None)

        # ------------------------------------------------------------ rainfall
        self.step(73, "Rainfall in the crop year against normal")
        try:
            days, rain = HY.area_rainfall(aoi, dt.date(1981, 1, 1), min(today, start + dt.timedelta(days=365)))
            mon = {}
            for d, v in zip(days, rain):
                mon.setdefault((d.year, d.month), []).append(v)
            tot = {k_: float(np.nansum(v)) for k_, v in mon.items() if len(v) >= 25}
            rows = []
            for i in range(12):
                d = (start.replace(day=1) + dt.timedelta(days=31 * i)).replace(day=1)
                hist = [v for (y, m), v in tot.items() if m == d.month and y < d.year]
                cur = tot.get((d.year, d.month))
                if hist and cur is not None:
                    mu, sd = float(np.mean(hist)), float(np.std(hist))
                    rows.append(dict(month=f"{TY.MONTHS[d.month - 1]} {d.year}", rain=cur, normal=mu, z=(cur - mu) / max(sd, 1e-6)))
            R["rain_months"] = rows
            yr = {}
            for (y, m), v in tot.items():
                yr.setdefault(y, []).append(v)
            R["rain_year"] = {y: float(np.sum(v)) for y, v in yr.items() if len(v) == 12}
            # standardised precipitation index (1, 3, 6 months) over the whole record
            try:
                sp = WA.spi(tot)
                R["spi"] = sp
                R["spi_rows"] = []
                for i in range(12):
                    d = (start.replace(day=1) + dt.timedelta(days=31 * i)).replace(day=1)
                    k_ = (d.year, d.month)
                    if any(k_ in sp.get(s_, {}) for s_ in (1, 3, 6)):
                        R["spi_rows"].append(dict(month=f"{TY.MONTHS[d.month - 1]} {d.year}", **{f"spi{s_}": sp.get(s_, {}).get(k_) for s_ in (1, 3, 6)}))
                s3 = sp.get(3, {})
                if s3:
                    yrs3 = {}
                    for (y, m), v in s3.items():
                        if v <= -1:
                            yrs3[y] = yrs3.get(y, 0) + 1
                    R["spi_dry_years"] = sorted(yrs3.items(), key=lambda kv: -kv[1])[:5]
            except Exception as e:
                R["spi_err"] = str(e)
            # soil moisture (NASA POWER root-zone and surface soil wetness, monthly since 1981)
            try:
                from .crop import soil as SL
                sm = SL.monthly(aoi.bbox, aoi.center)
                a_, b_ = R["season_window"]
                smon = sorted({(a_.replace(day=1) + dt.timedelta(days=31 * i)).month for i in range(12)
                               if (a_.replace(day=1) + dt.timedelta(days=31 * i)) <= b_ + dt.timedelta(days=(365 if b_ < a_ else 0))})
                za, zt = SL.anomalies(sm, "root"), SL.anomalies(sm, "top")
                rows_sm = []
                for i in range(12):
                    d = (start.replace(day=1) + dt.timedelta(days=31 * i)).replace(day=1)
                    k_ = (d.year, d.month)
                    if k_ in sm:
                        rows_sm.append(dict(month=f"{TY.MONTHS[d.month - 1]} {d.year}", root=sm[k_].get("root"), top=sm[k_].get("top"),
                                            z_root=za.get(k_), z_top=zt.get(k_)))
                dry_y = {}
                full_y = {y for y in {k[0] for k in za} if all((y, m) in za for m in smon)}   # complete seasons only
                for (y, m), z in za.items():
                    if z <= -1 and m in smon and y in full_y:
                        dry_y[y] = dry_y.get(y, 0) + 1
                ser = SL.season_series(sm, smon, "root")
                R["sm"] = dict(monthly=sm, months=smon, rows=rows_sm, season=ser,
                               dry_years=sorted(dry_y.items(), key=lambda kv: (-kv[1], -kv[0]))[:5])
            except Exception as e:
                R["sm_err"] = str(e)
            # irrigated vs rainfed (indicative): green crops through real dry spells, plus paddy rice
            try:
                R["irr"], R["irr_info"] = WA.irrigated(R["S"], R["crop"], R["centers"], days, rain, rice=R.get("rice"))
            except Exception as e:
                R["irr_err"] = str(e)
            if R.get("modis_year"):
                common = [y for y in R["rain_year"] if y in R["modis_year"]]
                if len(common) >= 8:
                    a = np.array([R["rain_year"][y] for y in common])
                    b = np.array([R["modis_year"][y]["integral"] for y in common])
                    R["rain_ndvi_r"] = float(np.corrcoef(a, b)[0, 1])
        except Exception as e:
            R["rain_err"] = str(e)

        # ------------------------------------------------------------ burning
        self.step(78, "Crop residue burning from MODIS burned area (2001–today)")
        try:
            g500 = Grid(aoi, 500, max_pixels=100_000)
            cm = D.worldcover(g500)
            cm500 = (cm == 5) & g500.mask if cm is not None else g500.mask
            b, cnt = MO.burned(aoi, g500, cm500)
            R["burn"], R["burn_count"], R["g500"] = b, cnt, g500
            yrs = {}
            for (y, m), v in b.items():
                yrs[y] = yrs.get(y, 0) + v
            R["burn_year"] = yrs
            full = [yrs[y] for y in sorted(yrs) if y < today.year]
            R["burn_trend"] = HY.mann_kendall(full) if len(full) >= 8 else None
        except Exception as e:
            R["burn_err"] = str(e)

        # ------------------------------------------------------------ fields, zones, validation
        self.step(80, "Fields: " + ("your boundaries" if params.get("fields") else "finding field boundaries automatically"))
        try:
            self._fields(R, params, aoi, g)
        except Exception as e:
            R["fields_err"] = str(e)
        if params.get("yield_zones"):
            try:
                self._calibrate(R, params, aoi, g)
            except Exception as e:
                R["cal_err"] = str(e)
        if params.get("zones"):
            try:
                self._zones(R, params, aoi, g)
            except Exception as e:
                R["zones_err"] = str(e)
        self.step(84, "Validation (USDA CDL in the USA, land cover agreement)")
        try:
            self._validate(R, aoi, g)
        except Exception as e:
            R["val_err"] = str(e)

        self.step(88, "Writing maps and tables")
        files, layers = self._write(R, out_dir)
        self.step(93, "Writing the report")
        from .crop.report_crop import build
        return build(self, R, aoi, out_dir, files, layers)

    # ------------------------------------------------------------------ pieces
    def _features(self, R):
        S = R["S"]
        P = R["P"]
        F = [S[t] for t in range(0, S.shape[0], 2)]
        F += [P["peak"][0], P["pos"][0], P["sos"][0], P["integral"][0], P["n"].astype(np.float32)]
        if R.get("vh") is not None:
            vh = R["vh"].astype(np.float32)
            F += [np.nanmin(vh, 0), np.nanmax(vh, 0), np.nanmean(vh, 0)]
        return np.stack([np.nan_to_num(f.astype(np.float32), nan=-1) for f in F], -1)

    def _supervised(self, R, params, aoi, g):
        from .lulc.rf import RandomForest
        feats = read_vector(params["training"], aoi)
        fld = params.get("crop_field")
        if not fld:
            raise KitError("Choose the crop name field of your field points.")
        names = sorted({str(a.get(fld)) for _, a, _ in feats if a.get(fld) not in (None, "")})
        code = {n: i + 1 for i, n in enumerate(names)}
        pts = []
        for gl, a, gw in feats:
            c = gw.Centroid()
            r, cc = g.lonlat_to_pixel(c.GetX(), c.GetY())
            if 0 <= r < g.rows and 0 <= cc < g.cols and g.mask[r, cc] and a.get(fld) not in (None, ""):
                pts.append((r, cc, code[str(a.get(fld))]))
        if len(pts) < 20:
            raise KitError("At least 20 field points inside the area are needed for a crop map.")
        X = self._features(R)
        idx = np.array([r * g.cols + c for r, c, _ in pts])
        y = np.array([k for *_, k in pts])
        Xf = X.reshape(-1, X.shape[-1])
        rng = np.random.default_rng(3)
        te = rng.random(len(y)) < 0.3
        m = RandomForest(n_trees=80, max_depth=14, min_leaf=2, seed=1).fit(Xf[idx[~te]], y[~te], list(range(1, len(names) + 1)))
        pred_te, _ = m.predict(Xf[idx[te]])
        cm = np.zeros((len(names), len(names)), int)
        for t, p in zip(y[te], pred_te):
            cm[p - 1, t - 1] += 1
        oa = float(np.trace(cm) / max(cm.sum(), 1))
        m = RandomForest(n_trees=80, max_depth=14, min_leaf=2, seed=1).fit(Xf[idx], y, list(range(1, len(names) + 1)))
        cl = np.zeros(g.shape[0] * g.shape[1], np.uint8)
        ci = np.flatnonzero(R["crop"].ravel())
        for s in range(0, ci.size, 300_000):
            cl[ci[s:s + 300_000]] = m.predict(Xf[ci[s:s + 300_000]])[0]
        R["sup"] = dict(map=cl.reshape(g.shape), names=names, oa=oa, cm=cm, n=len(pts), n_test=int(te.sum()),
                        area={n: float((cl == code[n]).sum() * g.px_km2) for n in names})

    def _fields(self, R, params, aoi, g):
        """Field statistics and within-field zones: your boundaries if given, else boundaries found from the NDVI time series."""
        crop = R["crop"]
        if params.get("fields"):
            feats = read_vector(params["fields"], aoi)
            ids = g.rasterize([x for x, _, _ in feats], values=list(range(1, len(feats) + 1))).astype(np.int32)
            n, geoms, auto = len(feats), [x for x, _, _ in feats], False
        else:
            ids, n, thr, geoms = YF.segment_fields(R["S"], crop, g)
            auto = True
            R["field_thr"] = thr
            if n == 0:
                raise KitError("No field boundaries could be found.")
        ids = np.where(g.mask, ids, 0)
        P = R["P"]
        sos = P["sos"][0]
        st = YF.field_stats(ids, n, dict(peak=P["peak"][0], seasons=P["n"].astype(np.float32), yidx=R["yield_index"], sos=sos,
                                         crop=crop.astype(np.float32)))
        z, low = YF.zones_within_fields(P["peak"][0], ids, n)
        rows = []
        for i in range(1, n + 1):
            if st["count"][i] < 10 or not np.isfinite(st["peak"][i]):
                continue
            s = st["sos"][i]
            rows.append(dict(id=i, area_ha=float(st["count"][i] * g.px_km2 * 100), peak=float(st["peak"][i]),
                             cv=float(st["peak_sd"][i] / max(st["peak"][i], 1e-6) * 100), low=float(low[i]),
                             seasons=float(st["seasons"][i]), yidx=float(st["yidx"][i]) if np.isfinite(st["yidx"][i]) else None,
                             sowing=str(R["centers"][int(round(s))]) if np.isfinite(s) else "–"))
        R["fields"] = (z, rows)
        R["field_ids"], R["field_n"], R["field_geoms"], R["field_auto"], R["field_stats"] = ids, n, geoms, auto, st

    def _zones(self, R, params, aoi, g):
        feats = read_vector(params["zones"], aoi)
        fld = params.get("zone_field")
        z = g.rasterize([x for x, _, _ in feats], values=list(range(1, len(feats) + 1)))
        rows = []
        for i, (_, a, _) in enumerate(feats):
            m = (z == i + 1) & g.mask
            if not m.any():
                continue
            c = m & R["crop"]
            rows.append(dict(name=str(a.get(fld, i + 1)) if fld else str(i + 1), area=float(m.sum() * g.px_km2), crop=float(c.sum() * g.px_km2),
                             ci=float(np.mean(R["P"]["n"][c]) * 100) if c.any() else None,
                             rice=float((c & R["rice"]).sum() * g.px_km2) if R.get("rice") is not None else None))
        R["zones"] = rows

    def _calibrate(self, R, params, aoi, g):
        """Calibrate the yield index to reported yields by zone (linear, leave-one-out)."""
        fld = params.get("yield_field")
        if not fld:
            raise KitError("Choose the reported yield field of your yield zones.")
        feats = read_vector(params["yield_zones"], aoi)
        z = g.rasterize([x for x, _, _ in feats], values=list(range(1, len(feats) + 1)))
        rows, xs, ys = [], [], []
        for i, (_, a, _) in enumerate(feats):
            m = (z == i + 1) & R["crop"] & np.isfinite(R["yield_integral"])
            try:
                yv = float(a.get(fld))
            except (TypeError, ValueError):
                continue
            if m.sum() < 25:
                continue
            x = float(np.mean(R["yield_integral"][m]))
            rows.append(dict(name=str(a.get(params.get("zone_field") or "name", i + 1)), index=x, reported=yv))
            xs.append(x)
            ys.append(yv)
        cal = YF.calibrate(xs, ys)
        if cal is None:
            raise KitError("At least 3 yield zones with cropland are needed to calibrate yield.")
        for r in rows:
            r["pred"] = cal["a"] + cal["b"] * r["index"]
        ymap = np.where(R["crop"], cal["a"] + cal["b"] * R["yield_integral"], np.nan)
        ymap = np.clip(ymap, 0, 1.5 * max(ys)).astype(np.float32)        # no extrapolation far beyond the reported range
        R["cal"] = dict(cal, rows=rows, map=ymap, unit=params.get("yield_unit") or "t/ha")

    def _validate(self, R, aoi, g):
        V = {}
        # 1) USDA Cropland Data Layer (USA only): crop-year match up to 2021
        yr = min(R["start"].year + (1 if R["start"].month >= 9 else 0), 2021)
        items = [i for i in search("usda-cdl", aoi.bbox, f"{yr}-01-01", f"{yr}-12-31", max_items=50) if i["id"].startswith("cropland_")]
        if items:
            from osgeo import gdal
            cdl = warp_items(items, "cropland", g, resample="mode", dtype=gdal.GDT_Byte, src_nodata=0, dst_nodata=0)
            V["cdl_year"] = yr
            V["cdl_same_year"] = yr == R["start"].year or (yr == R["start"].year + 1 and R["start"].month >= 9)
            crop = R["crop"]
            # CDL double-crop classes: 26, 225-241, 254
            dbl = np.isin(cdl, [26] + list(range(225, 242)) + [254])
            V["double_cdl"] = float(dbl[crop].mean() * 100)
            V["double_ours"] = float((R["P"]["n"][crop] >= 2).mean() * 100)
            n = R["P"]["n"]
            tp = (crop & dbl & (n >= 2)).sum()
            V["double_found"] = float(tp / max((crop & dbl).sum(), 1) * 100)
            V["double_correct"] = float(tp / max((crop & (n >= 2)).sum(), 1) * 100)
            if R.get("rice") is not None:
                rc = cdl == 3
                V["rice_cdl_km2"] = float((rc & crop).sum() * g.px_km2)
                V["rice_ours_km2"] = float(R["rice"].sum() * g.px_km2)
                tp = (rc & R["rice"]).sum()
                V["rice_found"] = float(tp / max((rc & crop).sum(), 1) * 100)
                V["rice_correct"] = float(tp / max(R["rice"].sum(), 1) * 100)
            gm, prof, labels = R["groups"]
            if gm is not None:
                rows = []
                names = {1: "Corn", 5: "Soybeans", 24: "Winter wheat", 2: "Cotton", 3: "Rice", 4: "Sorghum", 36: "Alfalfa",
                         26: "Dbl wheat/soy", 37: "Other hay", 61: "Fallow/idle", 176: "Grass/pasture", 6: "Sunflower", 54: "Tomatoes",
                         75: "Almonds", 76: "Walnuts", 69: "Grapes", 141: "Deciduous forest", 121: "Developed (open)", 111: "Open water",
                         190: "Woody wetlands", 195: "Herbaceous wetlands", 152: "Shrubland"}
                for i, lab in enumerate(labels):
                    m = (gm == i + 1)
                    if m.sum() < 50:
                        continue
                    v, c = np.unique(cdl[m], return_counts=True)
                    top = sorted(zip(c, v), reverse=True)[:3]
                    rows.append(dict(group=i + 1, label=lab, km2=float(m.sum() * g.px_km2),
                                     cdl=", ".join(f"{names.get(int(cls), 'class ' + str(int(cls)))} {cnt / m.sum() * 100:.0f}%" for cnt, cls in top)))
                V["groups_cdl"] = rows
            R["cdl"] = cdl
            if R.get("field_ids") is not None and R.get("field_auto"):
                pur, _ = YF.cdl_purity(R["field_ids"], R["field_n"], cdl)
                ok = np.isfinite(pur[1:])
                if ok.any():
                    area = R["field_stats"]["count"][1:][ok]
                    V["field_purity"] = float(np.average(pur[1:][ok], weights=area) * 100)
                    V["field_pure90"] = float((pur[1:][ok] >= 0.9).mean() * 100)
                    V["field_n"] = int(ok.sum())
        # 1b) irrigated vs rainfed against LANID-US 2017 (USA only, where CDL exists)
        if items and R.get("irr") is not None:
            try:
                from osgeo import gdal
                from ..core.data import trace, warp
                gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".zip,.tif,.TIF,.tiff")
                u = "https://zenodo.org/records/5548555/files/lanid2011-2017.zip"
                L = warp(["/vsizip//vsicurl/" + u + "/lanid2017.tif"], g, resample="near", dtype=gdal.GDT_Int16, src_nodata=256, dst_nodata=-1)
                trace(url=u, label="lanid-us-2017")
                crop = R["crop"]
                cls = R["irr"]
                ok = crop & (L >= 0) & (cls != 3)
                if not (R.get("irr_info") or {}).get("n_dry"):
                    ok &= False                              # no dry spell: greenness could not decide anywhere
                if R.get("cdl") is not None:
                    ok &= R["cdl"] != 61                     # land left idle this year cannot show irrigation
                ref, ours = ok & (L == 1), ok & (cls == 1)
                V["lanid_ref"] = float(((L == 1) & crop & (L >= 0)).sum() / max((crop & (L >= 0)).sum(), 1) * 100)
                V["lanid_det"] = float(ok.sum() / max((crop & (L >= 0)).sum(), 1) * 100)
                if ok.sum() >= 100:
                    tp = (ref & ours).sum()
                    V["lanid_found"] = float(tp / max(ref.sum(), 1) * 100)
                    V["lanid_correct"] = float(tp / max(ours.sum(), 1) * 100) if ours.any() else None
            except Exception as e:
                V["lanid_err"] = str(e)

        # 2) cropland agreement: active crop growth on land the maps call cropland
        V["active_share"] = float((R["P"]["n"][R["crop"]] >= 1).mean() * 100)
        R["val"] = V

    def _write(self, R, out_dir):
        g = R["g"]
        files, layers = [], []

        def add(path, name, hidden=True, kind="raster"):
            files.append(path)
            layers.append(dict(path=path, name=name, kind=kind, hidden=hidden))
        add(write_classes(os.path.join(out_dir, "cropping_intensity.tif"), R["intensity"], g, INTENSITY), "Cropping intensity", hidden=False)
        P, C = R["P"], R["centers"]
        for i in range(2):
            for k, nm in (("sos", "sowing (start of season)"), ("pos", "peak"), ("eos", "harvest (end of season)")):
                a = P[k][i]
                if np.isfinite(a).any():
                    doy = np.where(np.isfinite(a), np.array([c.timetuple().tm_yday for c in C], np.float32)[np.clip(np.nan_to_num(a).astype(int), 0, len(C) - 1)], np.nan)
                    files.append(write_float(os.path.join(out_dir, f"season{i + 1}_{k}_day_of_year.tif"), doy, g, "seq", 1, 366, "day of year")[0])
            files.append(write_float(os.path.join(out_dir, f"season{i + 1}_peak_ndvi.tif"), P["peak"][i], g, "ndvi", 0.2, 0.9, "NDVI")[0])
        gm, prof, labels = R["groups"]
        if gm is not None:
            cls = {i + 1: (lab[:60], TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i, lab in enumerate(labels)}
            add(write_classes(os.path.join(out_dir, "crop_calendar_groups.tif"), gm, g, cls), "Crop calendar groups")
        if R.get("rice") is not None:
            add(write_classes(os.path.join(out_dir, "paddy_rice.tif"), np.where(R["rice"], 1, 0).astype(np.uint8), g,
                              {1: ("Paddy rice (Sentinel-1)", "#2c7fb8")}), "Paddy rice")
        if R.get("sup"):
            s = R["sup"]
            cls = {i + 1: (n, TY.GROUP_COLS[i % len(TY.GROUP_COLS)]) for i, n in enumerate(s["names"])}
            add(write_classes(os.path.join(out_dir, "crop_map_from_your_points.tif"), s["map"], g, cls), "Crop map (your points)", hidden=False)
        if R.get("vci"):
            v, d, g250, cm = R["vci"]
            files.append(write_float(os.path.join(out_dir, f"vci_{d}.tif"), np.where(cm, v, np.nan), g250, "ndvi", 0, 100, "VCI")[0])
        if R.get("vhi"):
            vh_ = R["vhi"]
            yl = R["start"].year if R["start"].year in vh_["maps"] else max(vh_["maps"])
            files.append(write_float(os.path.join(out_dir, f"vhi_{yl}.tif"), np.where(R["cm250"], vh_["maps"][yl], np.nan).astype(np.float32),
                                     R["g250"], "ndvi", 0, 100, "VHI")[0])
            p = os.path.join(out_dir, "vegetation_health_by_year.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("season_year,vci,tci,vhi,cropland_in_drought_pct,mean_lst_c,mean_ndvi\n")
                for y in sorted(vh_["rows"]):
                    r = vh_["rows"][y]
                    fh.write(f"{y},{r['vci']:.1f},{r['tci']:.1f},{r['vhi']:.1f},{r['drought_pct']:.1f},{r['lst']:.2f},{r['ndvi']:.4f}\n")
            files.append(p)
        if R.get("sm"):
            p = os.path.join(out_dir, "soil_moisture_monthly.csv")
            from .crop import soil as SL
            za_, zt_ = SL.anomalies(R["sm"]["monthly"], "root"), SL.anomalies(R["sm"]["monthly"], "top")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("year,month,root_zone_wetness,surface_wetness,root_zone_z,surface_z\n")
                for (y, m), d in sorted(R["sm"]["monthly"].items()):
                    def f_(v):
                        return "" if v is None else f"{v:.3f}"
                    fh.write(f"{y},{m},{f_(d.get('root'))},{f_(d.get('top'))},{f_(za_.get((y, m)))},{f_(zt_.get((y, m)))}\n")
            files.append(p)
        if R.get("spi"):
            p = os.path.join(out_dir, "spi_monthly.csv")
            ks = sorted(set().union(*[set(v) for v in R["spi"].values()]))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("year,month,spi1,spi3,spi6\n")
                for k_ in ks:
                    vals = [R["spi"].get(s_, {}).get(k_) for s_ in (1, 3, 6)]
                    fh.write(f"{k_[0]},{k_[1]}," + ",".join("" if v is None else f"{v:.3f}" for v in vals) + "\n")
            files.append(p)
        if R.get("et"):
            p = os.path.join(out_dir, "evapotranspiration_water_productivity.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("year,et_mm,season_gpp_gc_m2,water_productivity_gc_per_m2_per_mm\n")
                for y in sorted(R["et"]):
                    gpv = (R.get("gpp") or {}).get(y, {}).get("gpp")
                    wpv = (R.get("wp") or {}).get(y)
                    fh.write(f"{y},{R['et'][y]:.1f}," + ("" if gpv is None else f"{gpv:.1f}") + "," + ("" if wpv is None else f"{wpv:.4f}") + "\n")
            files.append(p)
        if R.get("et_map"):
            y, a_ = R["et_map"]
            files.append(write_float(os.path.join(out_dir, f"evapotranspiration_{y}_mm.tif"), a_, R["g500y"], "depth",
                                     float(np.nanpercentile(a_, 2)) if np.isfinite(a_).any() else 0,
                                     float(np.nanpercentile(a_, 98)) if np.isfinite(a_).any() else 1, "mm")[0])
        if R.get("wp_map"):
            y, a_ = R["wp_map"]
            if np.isfinite(a_).any():
                files.append(write_float(os.path.join(out_dir, f"water_productivity_{y}.tif"), a_, R["g500y"], "ndvi",
                                         float(np.nanpercentile(a_, 2)), float(np.nanpercentile(a_, 98)), "gC/m² per mm")[0])
        if R.get("irr") is not None:
            add(write_classes(os.path.join(out_dir, "irrigated_vs_rainfed.tif"), R["irr"], g, WA.IRR_CLASSES), "Irrigated vs rainfed (indicative)")
        if R.get("burn_count") is not None:
            files.append(write_float(os.path.join(out_dir, "burned_cropland_count_2001_now.tif"),
                                     np.where(R["burn_count"] > 0, R["burn_count"], np.nan).astype(np.float32), R["g500"], "heat", 0, 15, "months")[0])
        p = os.path.join(out_dir, "crop_calendar_curve.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("date,mean_cropland_ndvi\n")
            for c, v in zip(C, R["crop_curve"]):
                fh.write(f"{c},{v:.4f}\n")
        files.append(p)
        if R.get("modis"):
            p = os.path.join(out_dir, "modis_cropland_ndvi_2001_now.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("date,mean_cropland_ndvi\n")
                for d, v in R["modis"]:
                    fh.write(f"{d},{v:.4f}\n")
            files.append(p)
        if R.get("fields"):
            z, rows = R["fields"]
            add(write_classes(os.path.join(out_dir, "management_zones.tif"), z, g,
                              {1: ("Low", "#d7191c"), 2: ("Medium", "#fdae61"), 3: ("High", "#1a9641")}), "Management zones")
            p = os.path.join(out_dir, "field_statistics.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("field,area_ha,mean_peak_ndvi,variability_cv_pct,low_zone_pct,seasons,relative_yield_index,sowing\n")
                for r in rows:
                    yi = "" if r["yidx"] is None else f"{r['yidx']:.1f}"
                    fh.write(f"{r['id']},{r['area_ha']:.2f},{r['peak']:.3f},{r['cv']:.1f},{r['low']:.1f},{r['seasons']:.2f},{yi},{r['sowing']}\n")
            files.append(p)
            if R.get("field_auto") and R.get("field_geoms"):
                st = R["field_stats"]
                st2 = dict(peak_ndvi=st["peak"], seasons=st["seasons"], yield_index=st["yidx"])
                add(YF.write_fields_gpkg(os.path.join(out_dir, "fields_detected.gpkg"), R["field_geoms"], g, st2, list(st2)),
                    "Fields (detected automatically)", kind="vector")
        if R.get("yield_index") is not None and np.isfinite(R["yield_index"]).any():
            add(write_float(os.path.join(out_dir, "relative_yield_index.tif"), R["yield_index"], g, "ndvi", 60, 140,
                            "index (100 = median)")[0], "Relative yield index", hidden=False)
        if R.get("cal"):
            c = R["cal"]
            lo, hi = np.nanpercentile(c["map"], [2, 98])
            add(write_float(os.path.join(out_dir, f"yield_estimate_{c['unit'].replace('/', '_per_')}.tif"), c["map"], g, "ndvi",
                            float(lo), float(hi), c["unit"])[0], f"Yield estimate ({c['unit']})", hidden=False)
            p = os.path.join(out_dir, "yield_calibration.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"zone,season_ndvi_integral,reported_{c['unit']},predicted_{c['unit']}\n")
                for r in c["rows"]:
                    fh.write(f"{r['name']},{r['index']:.3f},{r['reported']:.3f},{r['pred']:.3f}\n")
            files.append(p)
        if R.get("gpp"):
            p = os.path.join(out_dir, "yield_anomaly_modis_gpp.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("season_start_year,season_gpp_gC_m2,anomaly_pct_vs_median\n")
                for y, v in sorted(R["gpp"].items()):
                    fh.write(f"{y},{v['gpp']:.1f},{v['anomaly']:.1f}\n")
            files.append(p)
            if R.get("gpp_last") and R["gpp_last"][1] is not None:
                ly, arr = R["gpp_last"]
                files.append(write_float(os.path.join(out_dir, f"yield_anomaly_{ly}_pct.tif"), arr.astype(np.float32), R["g500y"], "ndvi",
                                         -40, 40, "% vs median")[0])
        rng = np.random.default_rng(7)
        pts = []
        for lab, sel in (("Two crops", R["intensity"] == 3), ("One crop", R["intensity"] == 2), ("Fallow", R["intensity"] == 1)):
            idx = np.flatnonzero(sel.ravel())
            for q in (rng.choice(idx, size=min(20, idx.size), replace=False) if idx.size else []):
                r, c = divmod(int(q), g.cols)
                pts.append((g.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "map_says": lab, "correct": None}))
        add(write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts), "Validation points", kind="vector")
        return files, layers
