# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: Flood: a complete flood study for an area in one run.

Builds on the official Copernicus Global Flood Monitoring archive (every Sentinel-1 pass since 2015)
and adds what it does not provide: the event sequence, duration, depth and volume, exposure and damage,
ten-year flood history and trend, flood susceptibility with four models and an automatic flood inventory,
an independent re-detection, comparison with expert Copernicus EMS maps and the GloFAS hazard model,
and a thesis-style report with a proof appendix.
"""

import datetime as dt
import os
import warnings

import numpy as np

from ..core import datasets as D
from ..core import osm
from ..core.aoi import Grid, read_vector
from ..core.data import KitError, item_date, search, warp_items
from ..core.maps import MapSpec, render_bars, render_grouped_bars, render_line, render_multi_line
from ..core.raster import neighbours, otsu, write_classes, write_float, write_geojson
from ..core.report import Report, fmt, grade
from .base import Kit
from .flood import depth as DP
from .flood import fuse as FL
from .flood import cnn as CNN
from .flood import hydro as HY
from .flood import extra as FX
from .flood import ems as EMS
from .flood import gfm as G
from .flood import impact as IM
from .flood import suscept as S
from .lulc.annual import mk_sen
from .lulc.classify import CREDIT_GHSL, ghsl_fraction
from .lulc.compare import CREDIT_SOIL, SOILGRIDS
from .lulc.extra import CREDIT_POP, population
from .lulc.models import auc

FV = {1: ("Possible flood under trees / crops", "#1b7837"), 2: ("Possible flood in built-up area", "#762a83")}
SAFE = {1: ("Safe high ground (never flooded, ≥ 1 m above nearby flood)", "#1a9850"), 2: ("Flood water (event maximum)", "#1f6fd1"),
        3: ("Other land", "#ece9df")}

EVENT = {1: ("Flood water", "#1f6fd1"), 2: ("Permanent water", "#0b2f5c"), 3: ("Dry land", "#ece9df"),
         4: ("Not observable (radar)", "#c9c9c9")}
FREQ = {1: ("Flooded once", "#c6dbef"), 2: ("2–3 times", "#6baed6"), 3: ("4–10 times", "#2171b5"),
        4: ("More than 10 times", "#08306b")}
SUSC = {1: ("Very low", "#2f7a4d"), 2: ("Low", "#9cc96b"), 3: ("Moderate", "#f2d36b"),
        4: ("High", "#ec8b3a"), 5: ("Very high", "#b8322a")}
CMP = {1: ("Both maps: flood", "#1f6fd1"), 2: ("Only this study", "#e8b02e"), 3: ("Only the reference", "#c82828"),
       4: ("Both: dry", "#ece9df")}
DCLS = [(0, 0.5, "< 0.5 m"), (0.5, 1, "0.5–1 m"), (1, 2, "1–2 m"), (2, 99, "> 2 m")]
MODELS = [("fr", "Frequency Ratio", "#8c6bb1"), ("ahp", "AHP (expert weights)", "#e08214"),
          ("lr", "Logistic Regression", "#2171b5"), ("rf", "Random Forest", "#238b45"), ("ens", "Ensemble", "#1c2621")]


def _d(s):
    return dt.date.fromisoformat(str(s)[:10])


def _months(txt):
    txt = (txt or "").strip()
    if not txt:
        return set(range(1, 13))
    out = set()
    for part in txt.replace(";", ",").split(","):
        part = part.strip()
        if "-" in part:
            a, b = [int(x) for x in part.split("-")]
            out |= set(range(a, b + 1)) if a <= b else set(range(a, 13)) | set(range(1, b + 1))
        elif part:
            out.add(int(part))
    return out or set(range(1, 13))


def _clay(grid):
    from osgeo import gdal
    from ..core.data import trace, warp
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".vrt,.tif,.TIF,.tiff,.zip")
    u = SOILGRIDS.format(v="clay")
    trace(url=u, label="soilgrids")
    try:
        a = warp(["/vsicurl/" + u], grid, resample="bilinear", dtype=gdal.GDT_Float32, src_nodata=-32768, dst_nodata=np.nan) / 10.0
        a[~grid.mask] = np.nan
        return a if np.isfinite(a).any() else None
    except Exception:
        return None


class FloodStudyKit(Kit):
    NAME = "AreaGIS Kit: Flood"
    KEY = "floodstudy"
    TAGLINE = "Complete flood study: event, depth, impact, 10-year history, susceptibility"
    DESCRIPTION = ("A complete flood study in one run. Uses the official Copernicus Global Flood Monitoring archive "
                   "(every Sentinel-1 radar pass since 2015) and adds the event sequence, flood duration, depth and "
                   "volume, people, buildings, crops and roads affected, damage, the flood history and trend, flood "
                   "susceptibility with four models (automatic flood inventory), an independent re-detection, and "
                   "comparison with expert Copernicus EMS maps and the GloFAS hazard model. Report in Word, PDF and HTML.")
    MAX_KM2 = 3000
    PARAMS = [
        dict(name="event_date", type="text", label="Flood date to study (YYYY-MM-DD; blank = largest flood on record)", default=""),
        dict(name="start_year", type="int", label="History from year", default=2017),
        dict(name="months", type="text", label="Months for the history (e.g. 6-10; blank = all year)", default=""),
        dict(name="own_detection", type="enum", label="Re-detect the event independently from raw radar", default="yes",
             options=["yes", "no"]),
        dict(name="zones", type="polygons", label="Zones for statistics (villages, wards, districts)", optional=True),
        dict(name="zone_field", type="field", label="Zone name field", parent="zones", optional=True),
        dict(name="reference_flood", type="polygons", label="Your observed flood extent for validation", optional=True),
        dict(name="value_built", type="float", label="Replacement value per m² of building footprint (0 = skip money)", default=0.0),
        dict(name="value_crop", type="float", label="Crop value per m² of cropland (0 = skip money)", default=0.0),
        dict(name="currency", type="text", label="Currency label", default="INR"),
        dict(name="wealth_file", type="file", label="Relative Wealth Index CSV (optional; else downloaded from HDX)", optional=True),
    ]

    # ================================================================== run
    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        R = dict(params=params)
        today = dt.date.today()
        y0 = int(params.get("start_year") or 2017)
        y0 = min(max(y0, 2015), today.year)
        months = _months(params.get("months"))
        hres = max(30.0, float(np.sqrt(aoi.area_km2 * 1e6 / 1_500_000)))
        hres = float(np.ceil(hres / 10) * 10)
        HG = Grid(aoi, hres, max_pixels=2_000_000)
        R["HG"] = HG

        # ------------------------------------------------------------ GFM archive
        self.step(2, f"Searching the Copernicus flood archive {y0}–{today.year}")
        items = G.search(aoi, f"{y0}-01-01", str(today))
        groups = {k: v for k, v in G.passes(items).items() if int(k[5:7]) in months}
        if not groups:
            raise KitError("The Copernicus Global Flood Monitoring archive has no radar passes over this area for these dates.")
        R["n_items"], R["n_passes_found"] = len(items), len(groups)

        def prog(i, n):
            self.step(4 + 26 * i / max(n, 1), f"Reading flood maps for each radar pass ({i}/{n})")
        H = G.history(aoi, HG, groups, progress=prog, cancelled=self._cancelled)
        R["H"] = H
        px_h = HG.px_km2
        self.step(31, "Summarising the flood history")
        self._history_stats(R, H, HG, px_h)

        # ------------------------------------------------------------ choose the event
        ev_txt = (params.get("event_date") or "").strip()
        ser = H["series"]
        if ev_txt:
            try:
                ev = _d(ev_txt)
            except Exception:
                raise KitError("Give the flood date as YYYY-MM-DD, or leave it blank.")
            near = [s for s in ser if abs((_d(s["date"]) - ev).days) <= 4]
            if not near:
                raise KitError(f"No Copernicus flood observation within 4 days of {ev}. Nearest passes: "
                               + ", ".join(sorted({s['date'] for s in ser}, key=lambda d: abs((_d(d) - ev).days))[:3]))
            peak = max(near, key=lambda s: (s["flood_px"] / max(s["valid_px"], 1), s["cover"]))
            R["event_auto"] = False
        else:
            peak = max(ser, key=lambda s: s["flood_px"])
            ev = _d(peak["date"])
            R["event_auto"] = True
        R["ev"], R["peak"] = ev, peak
        pdate = _d(peak["date"])

        # ------------------------------------------------------------ event at full detail
        eres = max(20.0, float(np.ceil(np.sqrt(aoi.area_km2 * 1e6 / 7_000_000) / 10) * 10))
        EG = Grid(aoi, eres, max_pixels=8_000_000)
        R["EG"] = EG
        px = EG.px_km2
        wkeys = [k for k in groups if -15 <= (_d(k[:10]) - pdate).days <= 45]
        self.step(33, f"Reading {len(wkeys)} radar passes around the flood at {eres:.0f} m")
        stack = G.event_stack(groups, wkeys, EG)
        stack = [(k, a) for k, a in stack if (a != 255).sum() >= 0.3 * EG.mask.sum()]
        if not stack:
            raise KitError("The flood maps around the event could not be read.")
        R["groups"] = groups
        self._event(R, stack, groups, EG, pdate, px)

        # ------------------------------------------------------------ terrain and context layers
        self.step(42, "Reading terrain, permanent water, land cover, buildings and population")
        lc_year = min(max(ev.year - 1, D.LULC_YEARS[0]), D.LULC_YEARS[-1])
        R["lc_year"] = lc_year
        L = {}
        for name, g in (("E", EG), ("H", HG)):
            dem = D.dem(g)
            slope, _ = D.slope_aspect(dem, g)
            try:
                hand = D.hand(g)
            except KitError:
                hand = None
            occ = D.water_occurrence(g)
            try:
                lc, _ = D.lulc(g, lc_year)
            except KitError:
                lc = D.worldcover(g)
            built, ge = ghsl_fraction(g, min(ev.year, 2020))
            pop, pe = population(g, ev.year)
            if built is None and lc is not None:
                # JRC server unreachable: fall back to the built-up class of the land cover map (share 0 or 1 per cell)
                built = np.where(g.mask, (lc == 7).astype(np.float32), np.nan)
                R["built_fallback"] = True
            L[name] = dict(dem=dem, slope=slope, hand=hand, occ=occ, lc=lc, built=built, pop=pop)
            R["ghsl_epoch"], R["pop_epoch"] = ge, pe
        R["L"] = L
        E = L["E"]
        perm_e = (np.nan_to_num(E["occ"], nan=0) >= 75) if E["occ"] is not None else np.zeros(EG.shape, bool)
        if R.get("refwater") is not None:
            perm_e |= R["refwater"]
        R["perm_e"] = perm_e

        # ------------------------------------------------------------ second detector and combined map
        R["peak_ext_gfm"] = R["peak_ext"].copy()
        if params.get("own_detection", "yes") == "yes":
            self.step(46, "Running a second flood detector on the raw Sentinel-1 radar")
            try:
                self._own(R, aoi, EG, pdate)
            except Exception as e:  # never fail the study because of the second detector
                R["own_err"] = str(e)
            o = R.get("own")
            cn = R.get("cnn")
            if cn is not None and abs((_d(cn["key"][:10]) - pdate).days) <= 1:
                # CNN where its radar pass sees the ground; GFM from other passes of the same days elsewhere
                comb = cn["map"] | (R["peak_ext"] & ~cn["valid"])
                R["combined"] = "cnn"
                R["peak_ext"] = comb & R["peak_obs"] & ~R["perm_e"]
                R["max_ext"] = R["max_ext"] | R["peak_ext"]
            elif o and o.get("own_map") is not None and abs((_d(o["date"]) - pdate).days) <= 1:
                # Benchmark on 20 expert-mapped floods: GFM ensemble OR second detector, then cleaned,
                # raised the critical success index from 0.45 to 0.51 on held-out sites (recall 60% -> 69%, precision ~72%).
                comb = FL.clean((R["peak_ext"] | o["own_map"]) & R["peak_obs"] & ~R["perm_e"])
                R["combined"] = True
                R["peak_ext"] = comb
                R["max_ext"] = R["max_ext"] | comb

        # ------------------------------------------------------------ depth, volume, exposure, damage
        self.step(50, "Estimating flood depth and water volume")
        fmax = R["max_ext"]
        d, wse, nedge = DP.depth(fmax, R["obs_any"], E["dem"], permanent=perm_e)
        R["depth"], R["n_edge"] = d, nedge
        # Is the flood visibly lower than the dry land around it in the elevation model? If not, depth from edges is unreliable.
        dem_e = E["dem"]
        near = (DP._box_mean(fmax.astype(np.float32), 25) > 0) & ~fmax & R["obs_any"] & np.isfinite(dem_e)
        if fmax.any() and near.any():
            R["dem_drop"] = float(np.nanmedian(dem_e[near]) - np.nanmedian(dem_e[fmax & np.isfinite(dem_e)]))
            R["depth_reliable"] = R["dem_drop"] >= 0.5
        else:
            R["dem_drop"], R["depth_reliable"] = None, False
        self.step(54, "Measuring people, buildings, crops and roads affected")
        self._impact(R, EG, px, params)

        # ------------------------------------------------------------ GloFAS hazard model
        self.step(66, "Reading the GloFAS river flood hazard model (10, 100 and 500-year floods)")
        R["glofas"] = {rp: IM.glofas(HG, rp) for rp in (10, 100, 500)}

        # ------------------------------------------------------------ susceptibility
        self.step(70, "Building flood susceptibility models (FR, AHP, LR, RF)")
        try:
            self._susceptibility(R, HG)
        except Exception as e:
            R["sus_err"] = str(e)

        # ------------------------------------------------------------ expert reference maps
        self.step(78, "Rainfall and river discharge: return periods, trends, flood-discharge relation")
        try:
            self._hydro(R, aoi, HG)
        except Exception as e:
            R["hydro_err"] = str(e)
        try:
            self._design(R, HG)
        except Exception as e:
            R["design_err"] = str(e)
        self.step(81, "Settlements affected, cut-off places, hazard, risk, hot spots, crop damage")
        try:
            self._more(R, aoi, EG, HG)
        except Exception as e:
            R["more_err"] = str(e)

        self.step(84, "Looking for expert Copernicus EMS flood maps of this event")
        self._ems(R, aoi, EG, out_dir)
        if params.get("reference_flood"):
            try:
                feats = read_vector(params["reference_flood"], aoi)
                ref = EG.rasterize([g for _, _, g in feats]) > 0
                R["user_ref"] = {"peak": EMS.scores(R["peak_ext"], ref, R["peak_obs"] & EG.mask),
                                 "max": EMS.scores(R["max_ext"], ref, R["obs_any"] & EG.mask)}
            except Exception as e:
                R["user_ref_err"] = str(e)

        # ------------------------------------------------------------ zones
        if params.get("zones"):
            try:
                self._zones(R, params, aoi, EG, px)
            except Exception as e:
                R["zones_err"] = str(e)

        import shutil
        shutil.rmtree(os.path.join(out_dir, "_ems"), ignore_errors=True)
        self.step(88, "Writing maps and tables")
        files, layers = self._write(R, out_dir)
        self.step(93, "Writing the report")
        return self._report(R, aoi, out_dir, files, layers)

    # ================================================================== pieces
    def _history_stats(self, R, H, HG, px):
        ser = H["series"]
        for s in ser:
            s["km2"] = s["flood_px"] * px
            s["share_obs"] = s["flood_px"] / max(s["valid_px"], 1) * 100
        years = H["years"]
        R["annual"] = [(y, float(H["year_any"][y].sum() * px), int(sum(1 for s in ser if s["date"][:4] == str(y)))) for y in years]
        mon = {m: [] for m in range(1, 13)}
        for s in ser:
            mon[int(s["date"][5:7])].append(s["share_obs"])
        R["monthly"] = {m: (float(np.mean(v)) if v else None, len(v)) for m, v in mon.items()}
        # top events: largest flooded area, one per 20 days
        top = []
        for s in sorted(ser, key=lambda s: -s["km2"]):
            if s["km2"] <= 0:
                break
            if all(abs((_d(s["date"]) - _d(t["date"])).days) > 20 for t in top):
                top.append(s)
            if len(top) == 8:
                break
        R["top"] = top
        fn, on = H["flood_n"].astype(np.float32), H["obs_n"].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            R["freq_pct"] = np.where(HG.mask & (on > 0), fn / on * 100, np.nan).astype(np.float32)
        cnt = np.zeros(HG.shape, np.uint8)
        cnt[(fn >= 1)] = 1
        cnt[(fn >= 2)] = 2
        cnt[(fn >= 4)] = 3
        cnt[(fn > 10)] = 4
        cnt[~HG.mask] = 0
        R["freq_cls"] = cnt
        yrs = np.zeros(HG.shape, np.uint8)
        for y in years:
            yrs += H["year_any"][y]
        R["years_flooded"] = np.where(HG.mask, yrs, 0).astype(np.uint8)
        R["n_years"] = len(years)
        R["ever_km2"] = float((fn >= 1).sum() * px)
        R["recurrent_km2"] = float((yrs >= max(2, len(years) // 2)).sum() * px)
        # trend of annual flooded area (Mann-Kendall, Sen's slope) on complete years
        full = [(y, a) for y, a, n in R["annual"] if n >= 20 and y < dt.date.today().year]
        R["trend"] = None
        if len(full) >= 5:
            Y = np.array([a for _, a in full], np.float32).reshape(-1, 1, 1)
            t = np.array([y for y, _ in full], np.float32)
            Z, P, Sl = mk_sen(Y, t)
            R["trend"] = dict(z=float(Z[0, 0]), p=float(P[0, 0]), slope=float(Sl[0, 0]), n=len(full))

    def _event(self, R, stack, groups, EG, pdate, px):
        times = [k for k, _ in stack]
        arrs = {k: a for k, a in stack}
        obs_any = np.zeros(EG.shape, bool)
        max_ext = np.zeros(EG.shape, bool)
        first = np.full(EG.shape, np.iinfo(np.int32).max, np.int32)
        last = np.full(EG.shape, -1, np.int32)
        tl = []
        for k in times:
            a = arrs[k]
            v, f = a != 255, a == 1
            obs_any |= v
            max_ext |= f
            dd = (_d(k[:10]) - pdate).days
            first[f] = np.minimum(first[f], dd)
            last[f] = np.maximum(last[f], dd)
            tl.append(dict(time=k, date=k[:10], day=dd, km2=float(f.sum() * px), obs=float(v.sum() / max(EG.mask.sum(), 1) * 100),
                           share=float(f.sum() / max(v.sum(), 1) * 100)))
        # peak: union of passes within ±2 days of the peak date (other orbits fill gaps)
        peak_ext = np.zeros(EG.shape, bool)
        peak_obs = np.zeros(EG.shape, bool)
        for k in times:
            if abs((_d(k[:10]) - pdate).days) <= 2:
                peak_ext |= arrs[k] == 1
                peak_obs |= arrs[k] != 255
        dur = np.where(max_ext, last - first + 1, 0).astype(np.float32)
        dur[~EG.mask] = np.nan
        R.update(timeline=tl, obs_any=obs_any & EG.mask, max_ext=max_ext & EG.mask, peak_ext=peak_ext & EG.mask,
                 peak_obs=peak_obs & EG.mask, duration=dur, event_keys=times, event_arrs=arrs)
        # GFM's own reference water and exclusion layers for the peak pass
        pk = min(times, key=lambda k: abs((_d(k[:10]) - pdate).days))
        rw = G.aux(groups, pk, EG, "reference_water_mask")
        ex = G.aux(groups, pk, EG, "exclusion_mask")
        R["refwater"] = (rw == 1) & EG.mask if rw is not None else None
        R["excluded_e"] = (ex == 1) & EG.mask if ex is not None else None
        R["peak_key"] = pk

    def _impact(self, R, EG, px, params):
        E = R["L"]["E"]
        ext = {"peak": R["peak_ext"], "max": R["max_ext"]}
        lc = E["lc"]
        out = {}
        for nm, f in ext.items():
            o = {}
            o["km2"] = float(f.sum() * px)
            o["pop"] = float(np.nansum(np.where(f, E["pop"], 0))) if E["pop"] is not None else None
            o["built_m2"] = float(np.nansum(np.where(f, E["built"], 0)) * EG.res ** 2) if E["built"] is not None else None
            o["lc"] = {k: float((f & (lc == k)).sum() * px) for k in D.LULC} if lc is not None else {}
            out[nm] = o
        R["impact"] = out
        d = R["depth"]
        if d is not None:
            dm = R["max_ext"] & np.isfinite(d)
            R["vol_mcm"] = float(np.nansum(np.where(dm, d, 0)) * EG.res ** 2 / 1e6)
            R["depth_stats"] = dict(median=float(np.nanmedian(d[dm])) if dm.any() else None,
                                    p90=float(np.nanpercentile(d[dm], 90)) if dm.any() else None)
            rows = []
            for lo, hi, lab in DCLS:
                m = dm & (d >= lo) & (d < hi)
                rows.append(dict(label=lab, km2=float(m.sum() * px),
                                 pop=float(np.nansum(np.where(m, E["pop"], 0))) if E["pop"] is not None else None,
                                 built_m2=float(np.nansum(np.where(m, E["built"], 0)) * EG.res ** 2) if E["built"] is not None else None,
                                 crop=float((m & (lc == 5)).sum() * px) if lc is not None else None))
            R["depth_rows"] = rows
            # damage (Huizinga et al. 2017)
            cont = IM.continent(*EG.aoi.center)
            fr_res = IM.damage_factor(np.where(dm, d, 0), IM.RESIDENTIAL[cont])
            ag_curve, ag_src = IM.agri_curve(cont)
            fr_ag = IM.damage_factor(np.where(dm, d, 0), ag_curve)
            dmg_b = float(np.nansum(fr_res * np.nan_to_num(E["built"])) * EG.res ** 2) if E["built"] is not None else None
            crop = (lc == 5) if lc is not None else np.zeros(EG.shape, bool)
            dmg_c = float(np.nansum(fr_ag * crop) * EG.res ** 2)
            vb, vc = float(params.get("value_built") or 0), float(params.get("value_crop") or 0)
            R["damage"] = dict(continent=cont, ag_src=ag_src, built_eq_m2=dmg_b, crop_eq_m2=dmg_c,
                               money_built=dmg_b * vb if (dmg_b is not None and vb > 0) else None,
                               money_crop=dmg_c * vc if vc > 0 else None, currency=params.get("currency") or "")
        # roads and facilities from OpenStreetMap
        R["roads"] = R["fac"] = None
        try:
            R["roads"] = IM.road_length(osm.roads(EG.aoi.bbox), EG, R["max_ext"])
        except Exception as e:
            R["osm_err"] = str(e)
        try:
            fac = osm.facilities(EG.aoi.bbox, "all", buffer_deg=0)
            hit = []
            for lon, lat, name, typ in fac:
                r, c = EG.lonlat_to_pixel(lon, lat)
                if 0 <= r < EG.rows and 0 <= c < EG.cols and EG.mask[r, c]:
                    near = R["max_ext"][max(0, r - 1):r + 2, max(0, c - 1):c + 2].any()
                    hit.append((lon, lat, name, typ, bool(near)))
            R["fac"] = hit
        except Exception as e:
            R["osm_err"] = str(e)

    def _own(self, R, aoi, EG, pdate):
        from .floodkit import FloodKit
        fk = FloodKit()
        post = search("sentinel-1-rtc", aoi.bbox, str(pdate - dt.timedelta(days=1)), str(pdate + dt.timedelta(days=1)))
        if not post:
            raise KitError("no Sentinel-1 RTC pass")
        g = fk._by_orbit_date(post)
        cov = {k: fk._coverage(v, aoi) for k, v in g.items()}
        orbit, d = max(cov, key=cov.get)
        ry = pdate.replace(year=pdate.year - 1) if not (pdate.month == 2 and pdate.day == 29) else pdate.replace(year=pdate.year - 1, day=28)
        refs = [i for i in search("sentinel-1-rtc", aoi.bbox, str(ry - dt.timedelta(days=30)), str(ry + dt.timedelta(days=30)))
                if i["properties"].get("sat:relative_orbit") == orbit]
        rg = fk._by_orbit_date(refs)
        rk = sorted([k for k in rg if fk._coverage(rg[k], aoi) >= 0.6 * cov[(orbit, d)]], key=lambda k: k[1])[-3:]
        if not rk:
            raise KitError("no reference pass")
        after = fk._read_db(g[(orbit, d)], EG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            before = np.nanmedian(np.stack([fk._read_db(rg[k], EG) for k in rk]), axis=0)
        thr = otsu(after[(after > -30) & (after < 5)])
        thr = -18.0 if thr is None else float(np.clip(thr, -23, -15))
        valid = np.isfinite(after) & np.isfinite(before) & EG.mask
        E = R["L"]["E"]
        terrain = (np.nan_to_num(E["hand"], nan=999) <= 20) & (np.nan_to_num(E["slope"], nan=90) <= 8) if E["hand"] is not None \
            else (np.nan_to_num(E["slope"], nan=90) <= 8)
        new = (after < thr) & ((after - before) < -3) & valid & terrain & ~R["perm_e"]
        own = new & (neighbours(new, 1) >= 4)
        # compare with the GFM pass of the same day (or nearest)
        k = min(R["event_keys"], key=lambda k: abs((_d(k[:10]) - _d(d)).days))
        gfm = R["event_arrs"][k] == 1
        both_obs = valid & (R["event_arrs"][k] != 255)
        if R.get("excluded_e") is not None:
            both_obs &= ~R["excluded_e"]
        sc = EMS.scores(own, gfm, both_obs)
        cmpm = np.zeros(EG.shape, np.uint8)
        cmpm[both_obs] = 4
        cmpm[both_obs & own & gfm] = 1
        cmpm[both_obs & own & ~gfm] = 2
        cmpm[both_obs & ~own & gfm] = 3
        R["own_pair"] = (before.astype(np.float16), after.astype(np.float16))

        # ---- CNN flood classifier (trained on 52 expert-mapped floods; validation/floodbench)
        try:
            self.step(52, "Running the flood classifier trained on 52 expert-mapped floods")
            after_vh = fk._read_db(g[(orbit, d)], EG, "vh")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                before_vh = np.nanmedian(np.stack([fk._read_db(rg[x], EG, "vh") for x in rk]), axis=0)
            pre = [i for i in search("sentinel-1-rtc", aoi.bbox, str(pdate - dt.timedelta(days=60)), str(pdate - dt.timedelta(days=12)))
                   if i["properties"].get("sat:relative_orbit") == orbit]
            pg = fk._by_orbit_date(pre)
            pk_ = sorted([x for x in pg if fk._coverage(pg[x], aoi) >= 0.6 * cov[(orbit, d)]], key=lambda x: x[1])[-3:]
            vv_pre = vh_pre = None
            if pk_:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    vv_pre = np.nanmedian(np.stack([fk._read_db(pg[x], EG) for x in pk_]), axis=0)
                    vh_pre = np.nanmedian(np.stack([fk._read_db(pg[x], EG, "vh") for x in pk_]), axis=0)
            gr = R["groups"]
            lay = {a: G.aux(gr, k, EG, a) for a in ("dlr_flood_extent", "tuw_flood_extent", "list_flood_extent",
                                                    "ensemble_likelihood", "exclusion_mask")}
            lay = {a: (v if v is not None else np.full(EG.shape, 255, np.uint8)) for a, v in lay.items()}
            wc = D.worldcover(EG)
            ens_k = R["event_arrs"][k]
            cvalid = EG.mask & (ens_k != 255) & np.isfinite(after)
            X = CNN.features(after, after_vh, before, before_vh, vv_pre, vh_pre, E["hand"], E["slope"], ens_k,
                             lay["dlr_flood_extent"], lay["tuw_flood_extent"], lay["list_flood_extent"],
                             lay["ensemble_likelihood"], lay["exclusion_mask"], E["occ"],
                             wc if wc is not None else np.zeros(EG.shape, np.uint8), cvalid)
            prob = CNN.predict(X)
            del X
            R["cnn"] = dict(prob=prob, valid=cvalid, key=k,
                            map=FL.clean((prob >= 0.5) & cvalid & ~R["perm_e"]))
        except Exception as e:
            R["cnn_err"] = str(e)
        # ---- flood hidden under trees, crops and in built-up areas: double bounce raises VV while VH does not rise
        # (tested on the Gudbrandsdalen 2023 flood against the EMS expert map: VV brightening alone was no better than chance,
        #  VV +3 dB with VH not rising gave 1.4–1.6 times the flood share of comparable land nearby)
        try:
            wc_ = D.worldcover(EG)
            if wc_ is not None:
                try:
                    a_vh, b_vh = after_vh, before_vh
                except NameError:
                    a_vh = fk._read_db(g[(orbit, d)], EG, "vh")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        b_vh = np.nanmedian(np.stack([fk._read_db(rg[x], EG, "vh") for x in rk]), axis=0)
                hand_ok = (np.nan_to_num(E["hand"], nan=999) <= 10) if E["hand"] is not None else terrain
                open_w = own | R["peak_ext"]
                near = DP._box_mean(open_w.astype(np.float32), max(2, int(round(1000 / EG.res)))) > 0
                inc = ((after - before) >= 3.0) & ((a_vh - b_vh) <= 0.0) & valid & hand_ok & near & ~open_w & ~R["perm_e"]
                # D.worldcover returns the kit's land-cover codes: 2 trees, 4 flooded vegetation, 5 crops, 11 shrub/grass, 7 built
                veg = inc & np.isin(wc_, [2, 4, 5, 11])
                urb = inc & (wc_ == 7)
                veg &= neighbours(veg, 1) >= 3
                urb &= neighbours(urb, 1) >= 2
                fv = np.zeros(EG.shape, np.uint8)
                fv[veg] = 1
                fv[urb] = 2
                cmp_m = valid & hand_ok & near & ~open_w & ~R["perm_e"] & np.isin(wc_, [2, 4, 5, 7, 11])
                R["fv"] = dict(map=fv, veg_km2=float(veg.sum() * EG.px_km2), urb_km2=float(urb.sum() * EG.px_km2),
                               pop=float(np.nansum(np.where(fv > 0, E["pop"], 0))) if E.get("pop") is not None else None,
                               open_km2=float(open_w.sum() * EG.px_km2), compare=cmp_m)
        except Exception as e:
            R["fv_err"] = str(e)
        R["own"] = dict(date=d, gfm_date=k[:10], orbit=orbit, thr=thr, refs=[x[1] for x in rk], scores=sc,
                        own_km2=float((own & both_obs).sum() * EG.px_km2), gfm_km2=float((gfm & both_obs).sum() * EG.px_km2),
                        map=cmpm, own_map=own)
        try:
            R["optical"] = fk._optical_check(aoi, EG, d, own | (R["perm_e"] & valid), valid)
        except Exception:
            R["optical"] = None

    def _susceptibility(self, R, HG):
        H, Lh = R["H"], R["L"]["H"]
        mask = HG.mask
        perm = (np.nan_to_num(Lh["occ"], nan=0) >= 75) if Lh["occ"] is not None else np.zeros(HG.shape, bool)
        clay = _clay(HG)
        F = S.factors(HG, Lh["dem"], Lh["slope"], Lh["hand"], Lh["occ"], Lh["lc"], Lh["built"], clay)
        R["factors"] = F
        excluded = np.zeros(HG.shape, bool)
        pos, neg = S.inventory(H["flood_n"], H["obs_n"], perm, excluded, mask)
        R["inv"] = dict(pos_km2=float(pos.sum() * HG.px_km2), neg_km2=float(neg.sum() * HG.px_km2))
        smp = S.sample(pos, neg, n=4000)
        if smp is None:
            raise KitError("Too few flooded pixels in the archive to build a susceptibility model.")
        idx, y = smp
        rng = np.random.default_rng(5)
        te = rng.random(idx.size) < 0.3
        itr, ytr, ite, yte = idx[~te], y[~te], idx[te], y[te]
        tables, enc, fr_idx = S.frequency_ratio(F, itr, ytr, mask)
        keys = [k for k, *_ in S.FACTORS if k in F]
        ahp_idx, w, A, cr, lam, akeys = S.ahp(F, mask)
        Xall, aidx = S.full_matrix(F, enc, keys, mask)
        Xtr, Xte = S.design(F, enc, keys, itr), S.design(F, enc, keys, ite)
        p_lr, b, se, pv = S.logistic_model(Xtr, ytr, Xall)
        p_rf, rfm = S.rf_model(Xtr, ytr, Xall, keys)
        imp = S.permutation_importance(rfm, Xte, yte, keys)
        maps = {}

        def to_map(v):
            m = np.full(HG.shape, np.nan, np.float32)
            m.ravel()[aidx] = v
            return m
        maps["fr"] = S.norm01(fr_idx, mask)
        maps["ahp"] = S.norm01(ahp_idx, mask)
        maps["lr"] = to_map(p_lr)
        maps["rf"] = to_map(p_rf)
        maps["ens"] = S.norm01(np.nanmean(np.stack([S.norm01(maps[k], mask) for k in ("fr", "ahp", "lr", "rf")]), axis=0), mask)
        # validation 1: random hold-out
        res = {}
        for k, *_ in MODELS:
            s_te = maps[k].ravel()[ite]
            ok = np.isfinite(s_te)
            res[k] = dict(auc_holdout=auc(yte[ok], s_te[ok]), roc=S.roc(yte[ok], s_te[ok]))
        # validation 2: temporal. Train on early years, test on places first flooded later
        years = H["years"]
        if len(years) >= 4:
            split = years[int(len(years) * 0.6) - 1]
            early = np.zeros(HG.shape, bool)
            for yy in years:
                if yy <= split:
                    early |= H["year_any"][yy]
            late_new = np.zeros(HG.shape, bool)
            for yy in years:
                if yy > split:
                    late_new |= H["year_any"][yy]
            late_new &= ~early & mask & ~perm
            pos_e = early & mask & ~perm
            smp2 = S.sample(pos_e, neg, n=4000, seed=9)
            if smp2 is not None and late_new.sum() >= 30:
                i2, y2 = smp2
                t_fr, t_enc, t_fr_idx = S.frequency_ratio(F, i2, y2, mask)
                X2 = S.design(F, t_enc, keys, i2)
                Xa2, _ = S.full_matrix(F, t_enc, keys, mask)
                tl = {"fr": S.norm01(t_fr_idx, mask), "ahp": maps["ahp"]}
                tl["lr"] = to_map(S.logistic_model(X2, y2, Xa2)[0])
                tl["rf"] = to_map(S.rf_model(X2, y2, Xa2, keys, seed=4)[0])
                tl["ens"] = np.nanmean(np.stack([S.norm01(tl[k], mask) for k in ("fr", "ahp", "lr", "rf")]), axis=0)
                tp, tn = S.sample(late_new, neg, n=3000, seed=13) or (None, None)
                if tp is not None:
                    for k in res:
                        s = tl[k].ravel()[tp]
                        ok = np.isfinite(s)
                        res[k]["auc_temporal"] = auc(tn[ok], s[ok])
                    R["temporal"] = dict(split=split, late_km2=float(late_new.sum() * HG.px_km2))
        # validation 3: independent physical model (GloFAS 100-year flood zone)
        g100 = R["glofas"].get(100)
        if g100 is not None and np.isfinite(g100).sum() >= 30:
            zone = np.isfinite(g100) & mask & ~perm
            out_zone = ~np.isfinite(g100) & mask & ~perm
            sm = S.sample(zone, out_zone, n=3000, seed=21)
            if sm is not None:
                gi, gy = sm
                for k in res:
                    s = maps[k].ravel()[gi]
                    ok = np.isfinite(s)
                    res[k]["auc_glofas"] = auc(gy[ok], s[ok])
        R["sus"] = dict(maps=maps, res=res, tables=tables, weights=w, ahp_matrix=A, cr=cr, lam=lam, ahp_keys=akeys,
                        lr=dict(b=b, se=se, p=pv), rf_imp=imp, keys=keys, vif=S.vif(Xtr, keys))
        cls5, breaks = S.classify5(maps["ens"], mask)
        R["sus"]["cls"] = cls5
        ever = H["flood_n"] >= 1
        R["sus"]["density"] = S.class_density(cls5, ever & ~perm, mask & ~perm)
        pop, built, lc = Lh["pop"], Lh["built"], Lh["lc"]
        rows = []
        for c in range(1, 6):
            m = cls5 == c
            rows.append(dict(cls=c, km2=float(m.sum() * HG.px_km2),
                             pop=float(np.nansum(np.where(m, pop, 0))) if pop is not None else None,
                             built=float(np.nansum(np.where(m, built, 0)) * HG.res ** 2 / 1e6) if built is not None else None,
                             crop=float((m & (lc == 5)).sum() * HG.px_km2) if lc is not None else None))
        R["sus"]["exposure"] = rows
        # agreement of observed flooding with the GloFAS model zones
        if g100 is not None:
            obs = ever & mask & ~perm
            R["glofas_obs"] = {rp: (float((obs & np.isfinite(R["glofas"][rp])).sum() / max(obs.sum(), 1) * 100)
                                    if R["glofas"][rp] is not None else None) for rp in (10, 100, 500)}

    def _ems(self, R, aoi, EG, out_dir):
        R["ems"] = []
        try:
            acts = EMS.find(aoi, R["ev"], days=30)
        except Exception as e:
            R["ems_err"] = str(e)
            return
        R["ems_acts"] = [(a["code"], a["name"], a["eventTime"][:10]) for a in acts[:5]]
        for a in acts[:3]:
            try:
                prods = EMS.products(a["code"], aoi)
            except Exception:
                continue
            if prods:
                top = max(p["share"] for p in prods)
                prods = [p for p in prods if p["share"] >= max(0.02, 0.5 * top)]
                ref_t = _d(R["peak"]["date"])
                def gap(p):
                    return abs((_d(p["time"][:10]) - ref_t).days) if p.get("time") else 99
                # prefer images from another satellite than Sentinel-1 (fully independent of GFM), then the closest date
                prods.sort(key=lambda p: (gap(p) > 3, str(p.get("sensor") or "").startswith("Sentinel-1"), gap(p)))
            for p in prods[:2]:
                try:
                    ref, inside = EMS.load(p, EG, workdir=os.path.join(out_dir, "_ems"))
                except Exception as e:
                    R["ems_err"] = str(e)
                    continue
                if ref is None or ref.sum() < 20:
                    continue
                t = _d(p["time"][:10]) if p.get("time") else R["ev"]
                k = min(R["event_keys"], key=lambda k: abs((_d(k[:10]) - t).days))
                gap = abs((_d(k[:10]) - t).days)
                arr = R["event_arrs"][k]
                where = inside & (arr != 255)
                if R.get("excluded_e") is not None:
                    where &= ~R["excluded_e"]
                where &= ~R["perm_e"]
                rec = dict(code=p["code"], aoi=p["aoi"], image=str(p.get("time"))[:16], sensor=p.get("sensor"),
                           gfm_date=k[:10], gap=gap, share=p["share"], ref_km2=float((ref & where).sum() * EG.px_km2),
                           gfm=EMS.scores(arr == 1, ref, where))
                if R.get("combined") and abs((_d(R["peak"]["date"]) - t).days) <= 1:
                    rec["combined"] = EMS.scores(R["peak_ext"], ref, where & R["peak_obs"])
                if R.get("own") and R["own"].get("own_map") is not None and abs((_d(R["own"]["date"]) - t).days) <= 2:
                    rec["own"] = EMS.scores(R["own"]["own_map"], ref, where & np.isfinite(R["L"]["E"]["dem"]))
                cm = np.zeros(EG.shape, np.uint8)
                pf = arr == 1
                cm[where] = 4
                cm[where & pf & ref] = 1
                cm[where & pf & ~ref] = 2
                cm[where & ~pf & ref] = 3
                rec["map"] = cm
                fv = R.get("fv")
                if fv is not None and R.get("own") and abs((_d(R["own"]["date"]) - t).days) <= 2:
                    cand = (fv["map"] > 0) & inside
                    comp = fv["compare"] & inside
                    if cand.sum() >= 10 and comp.sum() >= 100:
                        rec["fv_check"] = dict(precision=float((cand & ref).sum() / cand.sum()),
                                               base=float((comp & ref).sum() / comp.sum()), km2=float(cand.sum() * EG.px_km2))
                R["ems"].append(rec)
            if R["ems"]:
                break

    def _hydro(self, R, aoi, HG):
        ev = _d(R["peak"]["date"])
        today = dt.date.today()
        H = {}
        try:
            days, rain = HY.area_rainfall(aoi, dt.date(1981, 1, 1), min(today, ev + dt.timedelta(days=60)))
            idx = {d: i for i, d in enumerate(days)}
            if ev in idx:
                i = idx[ev]
                H["rain_window"] = [(days[j], float(rain[j])) for j in range(max(0, i - 30), min(len(days), i + 11))]
                tot = {k: float(np.nansum(rain[max(0, i - k + 1):i + 1])) for k in (1, 3, 5, 7, 15, 30)}
                H["rain_tot"] = tot
                rp = {}
                for k in (1, 3, 5, 7):
                    am = HY.annual_max(days, rain, k)
                    fit = HY.gumbel(am.values())
                    if fit:
                        mx = max(float(np.nansum(rain[max(0, j - k + 1):j + 1])) for j in range(max(0, i - 10), i + 1))
                        D, crit = HY.gof_ks(am.values(), fit)
                        rp[k] = dict(value=mx, rp=HY.return_period(mx, fit), q={T: HY.quantile(T, fit) for T in (2, 5, 10, 25, 50, 100)},
                                     ks=D, ks_crit=crit, n=len(am))
                H["rain_rp"] = rp
                yrs = {}
                for d, v in zip(days, rain):
                    yrs.setdefault(d.year, []).append(v)
                full = {y: v for y, v in yrs.items() if len(v) >= 330}
                H["rain_annual"] = {y: float(np.nansum(v)) for y, v in full.items()}
                H["rain_annual_max"] = {y: float(np.nanmax(v)) for y, v in full.items()}
                H["rain_trend_total"] = HY.mann_kendall([H["rain_annual"][y] for y in sorted(H["rain_annual"])])
                H["rain_trend_max"] = HY.mann_kendall([H["rain_annual_max"][y] for y in sorted(H["rain_annual_max"])])
                mon = {}
                for d, v in zip(days, rain):
                    mon.setdefault(d.month, []).append(v)
                H["rain_month"] = {m: float(np.nanmean(v) * 30.4) for m, v in mon.items()}
        except Exception as e:
            H["rain_err"] = str(e)
        try:
            pt, (qd, qv, loc) = HY.main_river(aoi, dt.date(1984, 1, 1), min(today, ev + dt.timedelta(days=60)))
            qi = {d: i for i, d in enumerate(qd)}
            if ev in qi and np.isfinite(qv).any():
                i = qi[ev]
                win = [(qd[j], float(qv[j])) for j in range(max(0, i - 30), min(len(qd), i + 46))]
                pk = max(win[25:41], key=lambda x: x[1]) if len(win) > 41 else max(win, key=lambda x: x[1])
                am = HY.annual_max(qd, qv)
                fit = HY.gumbel(am.values())
                H["q"] = dict(loc=loc, window=win, peak=pk, mean=float(np.nanmean(qv)), am=am,
                              rp=HY.return_period(pk[1], fit) if fit else None,
                              rp_emp=HY.empirical_rp(list(am.values()), pk[1]),
                              quant={T: HY.quantile(T, fit) for T in (2, 5, 10, 25, 50, 100)} if fit else None,
                              ks=HY.gof_ks(am.values(), fit) if fit else None,
                              trend=HY.mann_kendall([am[y] for y in sorted(am)]))
                if fit:
                    H["q"]["days_above"] = {T: int(sum(1 for _, v in win if v >= HY.quantile(T, fit))) for T in (2, 5)}
                H["qrel"] = FX.extent_vs_discharge(R["H"]["series"], qd, qv, HG.px_km2)
                if H["qrel"] and H["qrel"].get("fit") and fit:
                    f = H["qrel"]["fit"]
                    H["qrel"]["pred"] = {T: max(0.0, f["slope"] * HY.quantile(T, fit) + f["intercept"]) for T in (2, 10, 25, 100)}
        except Exception as e:
            H["q_err"] = str(e)
        # river-flow forecast for the next months (GloFAS ensemble statistics)
        try:
            from .flood import design as DS
            pt_, (qd_, qv_, loc_) = pt, (qd, qv, loc)
            fit_all = HY.gumbel(HY.annual_max(qd_, qv_).values())
            url, F = DS.forecast(HY._get, pt_[0], pt_[1])
            from ..core.data import trace as _tr
            _tr(url=url.split("&daily")[0] + "&daily=river_discharge(ensemble)", label="glofas-forecast")
            if fit_all and np.isfinite(F["median"]).any():
                Q = {T: HY.quantile(T, fit_all) for T in (2, 5, 10, 25, 50, 100)}
                im, ix = int(np.nanargmax(F["median"])), int(np.nanargmax(F["max"]))
                fc = dict(issued=str(F["dates"][0]), dates=[str(x) for x in F["dates"]], median=F["median"].tolist(),
                          p25=F["p25"].tolist(), p75=F["p75"].tolist(), max=F["max"].tolist(), min=F["min"].tolist(), Q=Q,
                          peak_med=(str(F["dates"][im]), float(F["median"][im])), peak_max=(str(F["dates"][ix]), float(F["max"][ix])),
                          rp_med=HY.return_period(float(F["median"][im]), fit_all), rp_max=HY.return_period(float(F["max"][ix]), fit_all),
                          days_med_q2=int(np.nansum(F["median"] >= Q[2])), days_p75_q2=int(np.nansum(F["p75"] >= Q[2])),
                          days_max_q2=int(np.nansum(F["max"] >= Q[2])), days_max_q10=int(np.nansum(F["max"] >= Q[10])),
                          mean_flow=float(np.nanmean(qv_)), loc=loc_)
                f = (H.get("qrel") or {}).get("fit")
                if f:
                    fc["area_med"] = max(0.0, f["slope"] * fc["peak_med"][1] + f["intercept"])
                    fc["area_max"] = max(0.0, f["slope"] * fc["peak_max"][1] + f["intercept"])
                # hindcast skill is not available from the API; report the level only
                lvl = "high" if fc["peak_max"][1] >= Q[10] else "medium" if fc["peak_max"][1] >= Q[2] else "low"
                fc["level"] = lvl
                H["forecast"] = fc
        except Exception as e:
            H["forecast_err"] = str(e)
        # the event among all floods seen by radar (empirical return period of the yearly maximum flooded area)
        ann = [a for y, a, n in R["annual"] if n >= 20]
        if len(ann) >= 4:
            H["event_area_rp"] = HY.empirical_rp(ann, R["impact"]["peak"]["km2"])
            H["event_rank"] = 1 + sum(1 for s in R["top"] if s["km2"] > R["impact"]["peak"]["km2"] * 1.0001)
        R["hydro"] = H

    def _design(self, R, HG):
        """Design-flood depth from HAND inside the floodplain, calibrated on the observed flood (see kits/flood/design.py)."""
        from .flood import design as DS
        Lh = R["L"]["H"]
        hand = Lh.get("hand")
        if hand is None:
            return
        EG = R["EG"]
        perm = (np.nan_to_num(Lh["occ"], nan=0) >= 75) if Lh["occ"] is not None else np.zeros(HG.shape, bool)
        obs = DS.to_grid(R["max_ext"], EG, HG)
        obs_v = DS.to_grid(R["obs_any"], EG, HG)
        dperm = HG.distance_to(perm) if perm.any() else None
        cal = DS.floodplain_calibrate(hand, Lh.get("slope"), dperm, obs, obs_v, HG.mask, perm)
        if cal is None:
            return
        ok = cal.pop("ok")
        flat_dem = (not R.get("depth_reliable")) and cal["flood_slope"] < 1.0
        if cal["csi"] < 0.25 or flat_dem:
            # flat plains: HAND is near zero everywhere; elsewhere one stage per floodplain may still not fit
            cal["reason"] = "flat" if (flat_dem or (cal["stage"] < 0.5 and cal["flood_slope"] < 1.0)) else "poor"
            R["design"] = dict(cal=cal, rows=[], maps={}, unusable=True)
            return
        out = dict(cal=cal, rows=[], maps={})
        pred = ((R.get("hydro") or {}).get("qrel") or {}).get("pred") or {}
        targets = {T: a for T, a in pred.items() if a and a > 0}
        if not targets:
            out["no_targets"] = True
            R["design"] = out
            return
        for T, km2 in sorted(targets.items()):
            st = DS.stage_in(hand, ok, km2 * cal["k"], HG.px_km2)
            if st is None:
                continue
            d = DS.depth_in(hand, ok, st)
            wet = np.isfinite(d)
            row = dict(T=T, km2=float(wet.sum() * HG.px_km2), stage=st, d_med=float(np.nanmedian(d)) if wet.any() else 0.0,
                       d_p90=float(np.nanpercentile(d, 90)) if wet.any() else 0.0,
                       pop=float(np.nansum(np.where(wet, Lh["pop"], 0))) if Lh["pop"] is not None else None,
                       built=float(np.nansum(np.where(wet, Lh["built"], 0)) * HG.res ** 2 / 1e6) if Lh["built"] is not None else None)
            gl = (R.get("glofas") or {}).get(100 if T == 100 else 10 if T == 10 else None)
            if gl is not None:
                row["csi_glofas"] = DS.csi(wet, np.isfinite(gl), HG.mask)
            out["rows"].append(row)
            out["maps"][T] = d
        out["src"] = "flow–area relation"
        R["design"] = out

    def _more(self, R, aoi, EG, HG):
        E, Lh = R["L"]["E"], R["L"]["H"]
        X = {}
        fl = R["max_ext"]
        if E["built"] is not None:
            lab, rows = FX.settlements(E["built"], E["pop"], fl, EG)
            X["settlements"] = sorted([r for r in rows if r["flooded_pct"] > 0],
                                      key=lambda r: (-r["people_flooded"], -r["area_km2"] * r["flooded_pct"]))
            X["n_settlements"] = len(rows)
        iso, irows = FX.isolated(R["peak_ext"], R["peak_obs"], R["perm_e"], E["built"], EG)
        X["isolated"], X["isolated_rows"] = iso, irows
        if E["pop"] is not None:
            X["isolated_people"] = float(np.nansum(np.where(iso, E["pop"], 0)))
        # nearest safe high ground for evacuation planning (straight lines; no road network)
        try:
            from .flood import response as RS
            ever_e = RS.hg_to_eg(R["H"]["flood_n"] >= 1, HG, EG)
            cand = [dict(r, kind="settlement") for r in (X.get("settlements") or []) if r["people_flooded"] > 0]
            cand += [dict(lat=r["lat"], lon=r["lon"], area_km2=r["area_km2"], people=None, people_flooded=0.0, flooded_pct=100.0,
                          kind="cut-off place", built_share=r["built_share"])
                     for r in (irows or []) if r.get("built_share", 0) >= 5]
            if cand and E.get("dem") is not None:
                rd_m = None
                try:
                    from .urban import network as NW
                    rd_m = NW.road_mask(EG, NW.fetch_roads(aoi.bbox))
                except Exception as e_:
                    X["roads_err"] = str(e_)
                X["evac"] = RS.evacuation(EG, cand, None, fl, R["perm_e"], ever_e, E["dem"], roads=rd_m)
        except Exception as e:
            X["evac_err"] = str(e)
        if R["depth"] is not None:
            X["hazard"] = FX.hazard_classes(R["depth"], R["duration"], fl)
        # risk = susceptibility (or flood frequency) x exposure
        if R.get("sus"):
            haz = R["sus"]["maps"]["ens"]
            X["risk_basis"] = "flood susceptibility (ensemble)"
        else:
            haz = np.clip(np.nan_to_num(R["freq_pct"]) / 30, 0, 1)
            X["risk_basis"] = "flood frequency"
        if Lh["pop"] is None and Lh["built"] is None:
            R["more"] = X
            X["risk"] = None
        else:
            X["risk_idx"], X["risk"] = FX.risk(haz, Lh["pop"], Lh["built"], HG.mask)
            # proxy vulnerability (low-ground buildings, distance to safe ground, distance to health care) -> risk with vulnerability
            try:
                from .flood import response as RS
                from .flood import design as DS
                ds_h = None
                ev_ = X.get("evac")
                if ev_ and ev_.get("dist_coarse") is not None:
                    ds_e = RS.expand(ev_["dist_coarse"].astype(np.float32), ev_["k"], EG.shape, fill=np.nan)
                    ds_h = DS.to_grid(ds_e, EG, HG)
                dh = None
                if R.get("fac"):
                    seed = np.zeros(HG.shape, bool)
                    for lon, lat, _n, typ, _f in R["fac"]:
                        if typ and typ.lower() not in ("school", "kindergarten", "college", "university"):
                            r_, c_ = HG.lonlat_to_pixel(lon, lat)
                            if 0 <= r_ < HG.rows and 0 <= c_ < HG.cols:
                                seed[r_, c_] = True
                    if seed.any():
                        dh = HG.distance_to(seed)
                rw_h = None
                try:
                    from .urban import wealth as WL
                    pts_, note_ = WL.fetch(HG.aoi.bbox, R["params"].get("wealth_file") or None)
                    X["rwi_note"] = note_
                    if pts_:
                        rw_h = WL.to_grid(pts_, HG)
                except Exception as e_:
                    X["rwi_note"] = f"not read ({e_})"
                V, comps = RS.vulnerability(HG, Lh["built"], Lh.get("hand"), ds_h, dh, wealth=rw_h)
                if V is not None:
                    X["vuln"], X["vuln_comps"] = V, comps
                    X["risk_v_idx"], X["risk_v"] = FX.risk(haz, Lh["pop"], Lh["built"], HG.mask, vulnerability=V)
                    X["risk_v_rows"] = [dict(cls=c, km2=float((X["risk_v"] == c).sum() * HG.px_km2),
                                             pop=float(np.nansum(np.where(X["risk_v"] == c, Lh["pop"], 0))) if Lh["pop"] is not None else None)
                                        for c in range(1, 6)]
            except Exception as e:
                X["vuln_err"] = str(e)
        rows = []
        for c in range(1, 6):
            if X["risk"] is None:
                break
            m = X["risk"] == c
            rows.append(dict(cls=c, km2=float(m.sum() * HG.px_km2),
                             pop=float(np.nansum(np.where(m, Lh["pop"], 0))) if Lh["pop"] is not None else None))
        X["risk_rows"] = rows
        cell = max(2, int(round(500 / HG.res)))
        z, _ = FX.gistar(R["freq_pct"], HG.mask, cell)
        if z is not None:
            X["gi"] = FX.gi_classes(FX.expand(z, cell, HG.shape))
            X["gi"][~HG.mask] = 0
            X["gi_cell_m"] = cell * HG.res
        X["recession"] = FX.recession(R["timeline"])
        # encroachment: buildings and people moving into flood zones (GHSL epochs)
        zones = {"Flooded at least once (radar)": (R["H"]["flood_n"] >= 1) & HG.mask}
        if R["glofas"].get(100) is not None:
            zones["GloFAS 100-year zone"] = np.isfinite(R["glofas"][100]) & HG.mask
        if R.get("sus"):
            zones["High + very high susceptibility"] = (R["sus"]["cls"] >= 4) & HG.mask
        enc = []
        for yr in (1990, 2000, 2010, 2020):
            try:
                b, _ = ghsl_fraction(HG, yr)
                pp, _ = population(HG, yr if yr < 2020 else 2025)
            except Exception:
                continue
            if b is None and pp is None:
                continue
            row = dict(year=yr if yr < 2020 else "2020/2025")
            for zn, zm in zones.items():
                row[zn] = (float(np.nansum(np.where(zm, b, 0)) * HG.res ** 2 / 1e6) if b is not None else None,
                           float(np.nansum(np.where(zm, pp, 0))) if pp is not None else None)
            enc.append(row)
        X["encroach"], X["encroach_zones"] = enc, list(zones)
        # exposure in the modelled return-period zones
        rp_rows = []
        for rp in (10, 100, 500):
            g = R["glofas"].get(rp)
            if g is None:
                continue
            m = np.isfinite(g) & HG.mask
            rp_rows.append(dict(rp=rp, km2=float(m.sum() * HG.px_km2), pop=float(np.nansum(np.where(m, Lh["pop"], 0))) if Lh["pop"] is not None else None,
                                built=float(np.nansum(np.where(m, Lh["built"], 0)) * HG.res ** 2 / 1e6) if Lh["built"] is not None else None,
                                deep=float((m & (g >= 1)).sum() * HG.px_km2)))
        X["rp_exposure"] = rp_rows
        try:
            if E["lc"] is not None:
                X["crop"] = FX.crop_ndvi_change(aoi, EG, _d(R["peak"]["date"]), E["lc"] == 5, fl, search, warp_items, item_date)
        except Exception as e:
            X["crop_err"] = str(e)
        R["more"] = X

    def _zones(self, R, params, aoi, EG, px):
        feats = read_vector(params["zones"], aoi)
        fld = params.get("zone_field")
        E = R["L"]["E"]
        rows = []
        z = EG.rasterize([g for _, _, g in feats], values=list(range(1, len(feats) + 1)))
        for i, (_, a, _) in enumerate(feats):
            m = (z == i + 1) & EG.mask
            if not m.any():
                continue
            f = m & R["max_ext"]
            rows.append(dict(name=str(a.get(fld, i + 1)) if fld else str(i + 1), area=float(m.sum() * px),
                             flood=float(f.sum() * px), share=float(f.sum() / m.sum() * 100),
                             pop=float(np.nansum(np.where(f, E["pop"], 0))) if E["pop"] is not None else None,
                             depth=float(np.nanmedian(R["depth"][f])) if (R["depth"] is not None and f.any()) else None))
        R["zones"] = sorted(rows, key=lambda r: -r["flood"])

    # ================================================================== files
    def _write(self, R, out_dir):
        EG, HG = R["EG"], R["HG"]
        files, layers = [], []

        def add(path, name, hidden=True, kind="raster"):
            files.append(path)
            layers.append(dict(path=path, name=name, kind=kind, hidden=hidden))
        pk = R["peak"]["date"]
        ev = np.zeros(EG.shape, np.uint8)
        ev[EG.mask] = 4
        ev[R["peak_obs"]] = 3
        ev[R["peak_obs"] & R["perm_e"]] = 2
        ev[R["peak_ext"]] = 1
        R["ev_map"] = ev
        add(write_classes(os.path.join(out_dir, f"flood_extent_{pk}.tif"), ev, EG, EVENT), f"Flood extent {pk}", hidden=False)
        mx = np.where(EG.mask, np.where(R["max_ext"], 1, np.where(R["obs_any"], 3, 4)), 0).astype(np.uint8)
        add(write_classes(os.path.join(out_dir, "flood_maximum_extent.tif"), mx, EG, EVENT), "Maximum flood extent (event)")
        add(write_float(os.path.join(out_dir, "flood_duration_days.tif"), np.where(R["max_ext"], R["duration"], np.nan), EG,
                        "depth", 0, 60, "days")[0], "Flood duration (days)")
        if R["depth"] is not None:
            add(write_float(os.path.join(out_dir, "flood_depth_m.tif"), R["depth"], EG, "depth", 0, 3, "m")[0], "Flood depth (m)")
        add(write_float(os.path.join(out_dir, "flood_frequency_pct.tif"), np.where(R["H"]["flood_n"] > 0, R["freq_pct"], np.nan),
                        HG, "depth", 0, 30, "%")[0], "Flood frequency (% of radar looks)")
        add(write_classes(os.path.join(out_dir, "flood_count_classes.tif"), R["freq_cls"], HG, FREQ), "Times flooded (classes)")
        X_ = R.get("more") or {}
        ev_ = X_.get("evac")
        if ev_:
            from .flood import response as RS
            sm = RS.expand(ev_["safe"].astype(np.uint8), ev_["k"], EG.shape)
            emap = np.where(EG.mask, np.where(R["max_ext"], 2, np.where(sm == 1, 1, 3)), 0).astype(np.uint8)
            R["evac_map"] = emap
            add(write_classes(os.path.join(out_dir, "safe_high_ground.tif"), emap, EG, SAFE), "Safe high ground (evacuation planning)")
            feats = []
            for r in ev_["rows"]:
                props = {"kind": r["kind"], "people_in_water": round(r.get("people_flooded") or 0), "safe_km": None if r["safe_km"] is None else round(r["safe_km"], 2),
                         "direction": r["bearing"], "dry_road_km": None if r.get("road_km") is None else round(r["road_km"], 2)}
                if r["safe_lon"] is not None:
                    feats.append(({"type": "LineString", "coordinates": [[round(r["lon"], 6), round(r["lat"], 6)],
                                                                        [round(r["safe_lon"], 6), round(r["safe_lat"], 6)]]}, props))
                else:
                    feats.append(((r["lon"], r["lat"]), props))
            add(write_geojson(os.path.join(out_dir, "evacuation_to_safe_ground.geojson"), feats), "Straight line to nearest safe ground", kind="vector")
            fn = os.path.join(out_dir, "evacuation_to_safe_ground.csv")
            with open(fn, "w", encoding="utf-8", newline="") as fh:
                fh.write("kind,lat,lon,people_in_flood_water,safe_ground_km,dry_road_km,direction,safe_lat,safe_lon\n")
                for r in ev_["rows"]:
                    sk = "" if r["safe_km"] is None else f"{r['safe_km']:.2f}"
                    sla = "" if r["safe_lat"] is None else f"{r['safe_lat']:.5f}"
                    slo = "" if r["safe_lon"] is None else f"{r['safe_lon']:.5f}"
                    rk = "" if r.get("road_km") is None else f"{r['road_km']:.2f}"
                    fh.write(f"{r['kind']},{r['lat']:.5f},{r['lon']:.5f},{(r.get('people_flooded') or 0):.0f},{sk},{rk},{r['bearing'] or ''},{sla},{slo}\n")
            files.append(fn)
        if X_.get("vuln") is not None:
            add(write_float(os.path.join(out_dir, "flood_vulnerability_proxy.tif"), X_["vuln"], HG, "risk", 0, 1, "index")[0],
                "Flood vulnerability (proxy index)")
            add(write_classes(os.path.join(out_dir, "flood_risk_with_vulnerability.tif"), X_["risk_v"], HG, FX.RISK), "Flood risk incl. vulnerability")
        if R.get("fv") and (R["fv"]["veg_km2"] + R["fv"]["urb_km2"]) > 0:
            add(write_classes(os.path.join(out_dir, "flood_under_vegetation_and_urban.tif"), R["fv"]["map"], EG, FV),
                "Possible flood under trees/crops and in built-up areas")
        for T, d in ((R.get("design") or {}).get("maps") or {}).items():
            add(write_float(os.path.join(out_dir, f"design_flood_{T}yr_depth_m.tif"), d, HG, "depth", 0, 3, "m")[0], f"{T}-year flood depth (HAND model, m)")
        fc = (R.get("hydro") or {}).get("forecast")
        if fc:
            fn = os.path.join(out_dir, "river_flow_forecast.csv")
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write("date,median_m3s,p25,p75,min,max\n")
                for i, d_ in enumerate(fc["dates"]):
                    fh.write(f"{d_},{fc['median'][i]:.1f},{fc['p25'][i]:.1f},{fc['p75'][i]:.1f},{fc['min'][i]:.1f},{fc['max'][i]:.1f}\n")
            files.append(fn)
        add(write_float(os.path.join(out_dir, "years_flooded.tif"), np.where(R["years_flooded"] > 0, R["years_flooded"], np.nan),
                        HG, "depth", 0, R["n_years"], "years")[0], "Years with flooding")
        if R.get("sus"):
            add(write_classes(os.path.join(out_dir, "flood_susceptibility_classes.tif"), R["sus"]["cls"], HG, SUSC),
                "Flood susceptibility (ensemble)", hidden=False)
            for k, nm, _ in MODELS:
                add(write_float(os.path.join(out_dir, f"susceptibility_{k}.tif"), R["sus"]["maps"][k], HG, "risk", 0, 1, "")[0],
                    f"Susceptibility index: {nm}")
            for k, a in R["factors"].items():
                files.append(write_float(os.path.join(out_dir, f"factor_{k}.tif"), a, HG, "seq",
                                         float(np.nanpercentile(a, 2)), float(np.nanpercentile(a, 98)), "")[0])
        if R["glofas"].get(100) is not None:
            add(write_float(os.path.join(out_dir, "glofas_rp100_depth_m.tif"), R["glofas"][100], HG, "depth", 0, 5, "m")[0],
                "GloFAS 100-year flood depth (model)")
        if R.get("own"):
            add(write_classes(os.path.join(out_dir, "independent_redetection_vs_gfm.tif"), R["own"]["map"], EG, CMP),
                "Independent re-detection vs GFM")
        for e in R.get("ems", []):
            tag = e["image"][:10]
            add(write_classes(os.path.join(out_dir, f"ems_{e['code']}_{tag}_vs_gfm.tif"), e["map"], EG, CMP), f"EMS {e['code']} {tag} vs GFM")
        X = R.get("more") or {}
        if X.get("hazard") is not None:
            add(write_classes(os.path.join(out_dir, "flood_hazard_depth_duration.tif"), X["hazard"], EG, FX.HAZARD), "Flood hazard (depth × duration)")
        if X.get("risk") is not None:
            add(write_classes(os.path.join(out_dir, "flood_risk_classes.tif"), X["risk"], HG, FX.RISK), "Flood risk (hazard × exposure)")
            files.append(write_float(os.path.join(out_dir, "flood_risk_index.tif"), X["risk_idx"], HG, "risk", 0, 1, "")[0])
        if X.get("gi") is not None:
            add(write_classes(os.path.join(out_dir, "flood_hotspots_gistar.tif"), X["gi"], HG, FX.GISTAR), "Flood hot spots (Gi*)")
        if X.get("isolated") is not None and X["isolated"].any():
            iso = np.where(EG.mask, np.where(X["isolated"], 1, 0), 0).astype(np.uint8)
            add(write_classes(os.path.join(out_dir, "cut_off_dry_land.tif"), iso, EG, {1: ("Cut off by water", "#d7301f")}), "Cut-off dry land")
            p = os.path.join(out_dir, "cut_off_places.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("lat,lon,area_km2,built_share_pct\n")
                for r in X["isolated_rows"]:
                    fh.write(f"{r['lat']:.5f},{r['lon']:.5f},{r['area_km2']:.4f},{r['built_share']:.1f}\n")
            files.append(p)
        if X.get("settlements"):
            p = os.path.join(out_dir, "settlements_affected.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("rank,lat,lon,area_km2,people,flooded_pct,people_in_flood\n")
                for i, r in enumerate(X["settlements"]):
                    fh.write(f"{i + 1},{r['lat']:.5f},{r['lon']:.5f},{r['area_km2']:.4f},{r['people']:.0f},{r['flooded_pct']:.1f},{r['people_flooded']:.0f}\n")
            files.append(p)
            add(write_geojson(os.path.join(out_dir, "settlements_affected.geojson"),
                              [((r["lon"], r["lat"]), {"rank": i + 1, "people": round(r["people"]), "flooded_pct": round(r["flooded_pct"], 1),
                                                       "people_in_flood": round(r["people_flooded"])}) for i, r in enumerate(X["settlements"])]),
                "Settlements affected", kind="vector")
        if X.get("crop"):
            add(write_float(os.path.join(out_dir, "crop_ndvi_change.tif"), X["crop"]["change"], EG, "ndvi", -0.5, 0.5, "ΔNDVI")[0],
                "Crop NDVI change after flood")
        Hy = R.get("hydro") or {}
        if Hy.get("rain_window") or Hy.get("q"):
            p = os.path.join(out_dir, "rainfall_discharge_event.csv")
            rain = dict(Hy.get("rain_window") or [])
            q = dict((Hy.get("q") or {}).get("window") or [])
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("date,rain_mm,discharge_m3s\n")
                for d in sorted(set(rain) | set(q)):
                    fh.write(f"{d},{'' if d not in rain else round(rain[d], 2)},{'' if d not in q else round(q[d], 1)}\n")
            files.append(p)
        # tables
        p = os.path.join(out_dir, "flood_timeline.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("date,days_from_peak,flooded_km2,observed_pct,flooded_pct_of_observed\n")
            for t in R["timeline"]:
                fh.write(f"{t['date']},{t['day']},{t['km2']:.3f},{t['obs']:.1f},{t['share']:.2f}\n")
        files.append(p)
        p = os.path.join(out_dir, "flood_history_passes.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("date,observed_pct,flooded_km2,flooded_pct_of_observed,gfm_items\n")
            for s in R["H"]["series"]:
                fh.write(f"{s['date']},{s['cover'] * 100:.1f},{s['km2']:.3f},{s['share_obs']:.3f},{' '.join(s['items'])}\n")
        files.append(p)
        p = os.path.join(out_dir, "flood_annual.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("year,area_flooded_at_least_once_km2,radar_passes\n")
            for y, a, n in R["annual"]:
                fh.write(f"{y},{a:.3f},{n}\n")
        files.append(p)
        if R.get("sus"):
            p = os.path.join(out_dir, "frequency_ratio_tables.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("factor,class,range,area_pct,flood_pct,frequency_ratio\n")
                for k, rows in R["sus"]["tables"].items():
                    for r in rows:
                        fh.write(f"{k},{r['cls']},\"{r['label']}\",{r['area']:.2f},{r['flood']:.2f},{r['fr']:.3f}\n")
            files.append(p)
        if R.get("zones"):
            p = os.path.join(out_dir, "zones_flood_statistics.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("zone,area_km2,flooded_km2,flooded_pct,people_in_flood,median_depth_m\n")
                for z in R["zones"]:
                    fh.write(f"\"{z['name']}\",{z['area']:.3f},{z['flood']:.3f},{z['share']:.2f},"
                             f"{'' if z['pop'] is None else round(z['pop'])},{'' if z['depth'] is None else round(z['depth'], 2)}\n")
            files.append(p)
        # facilities and validation points
        if R.get("fac"):
            p = write_geojson(os.path.join(out_dir, "facilities_flood_check.geojson"),
                              [((lo, la), {"name": n, "type": t, "flooded": f}) for lo, la, n, t, f in R["fac"]])
            add(p, "Schools and health facilities", kind="vector")
        rng = np.random.default_rng(7)
        pts = []
        for lab, sel in (("Flood water", R["peak_ext"]), ("Dry land", R["ev_map"] == 3)):
            idx = np.flatnonzero(sel.ravel())
            for q in (rng.choice(idx, size=min(25, idx.size), replace=False) if idx.size else []):
                r, c = divmod(int(q), EG.cols)
                pts.append((EG.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "map_says": lab, "date": R["peak"]["date"], "correct": None}))
        add(write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts), "Validation points", kind="vector")
        return files, layers

    # ================================================================== report
    def _report(self, R, aoi, out_dir, files, layers):
        from .flood.report_flood import build
        return build(self, R, aoi, out_dir, files, layers)
