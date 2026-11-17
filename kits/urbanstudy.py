# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: Urban: a complete city diagnosis in one run.

Surface temperature (Landsat), surface urban heat island against a matched countryside ring (day and night, MODIS),
20-year heat island trend, hot spots (Gi*), thermal comfort (UTFVI), cooling by parks and water bodies, tree canopy and
green space per person, the 3-30-300 rule, people at risk (heat vulnerability index), access to health care, schools and
green space, built-up growth, building on flood-prone ground and on former water bodies, air quality (Sentinel-5P NO2),
statistics for your wards, independent checks (MODIS against Landsat, WorldCover against Esri, two population maps),
report and proof.
"""

import datetime as dt
import os
import warnings

import numpy as np

from ..core import datasets as D
from ..core import osm
from ..core.aoi import Grid, read_vector
from ..core.data import KitError
from ..core.raster import write_classes, write_float, write_geojson
from .base import Kit
from .flood import extra as FX
from .flood import hydro as HY
from .lulc import extra as LX
from .urban import city as CT
from .urban import heat as HT

HVI = {1: ("Very low", "#1a9850"), 2: ("Low", "#91cf60"), 3: ("Medium", "#fee08b"), 4: ("High", "#fc8d59"), 5: ("Very high", "#d73027")}
UTFVI = {1: ("Excellent (none)", "#1a9850"), 2: ("Good (weak)", "#91cf60"), 3: ("Normal (middle)", "#fee08b"),
         4: ("Bad (strong)", "#fdae61"), 5: ("Worse (stronger)", "#f46d43"), 6: ("Worst (strongest)", "#a50026")}
ZONES = {1: ("Urban core", "#b2182b"), 2: ("Countryside reference", "#4d9221"), 3: ("Other", "#ece9df")}
GROWTH = {1: ("Built before", "#8c3b2e"), 2: ("Newly built", "#f0663a"), 3: ("Not built", "#ece9df")}
EXPO = {1: ("Built in a river flood zone or on low ground by streams", "#2c7fb8"), 2: ("Built where water used to be", "#08306b"),
        3: ("Built in a low-lying pocket (waterlogging)", "#6baed6"), 4: ("Other built-up", "#d9d9d9")}


class UrbanStudyKit(Kit):
    NAME = "AreaGIS Kit: Urban"
    KEY = "urbanstudy"
    TAGLINE = "Complete city study: heat island, green space, people at risk, access, air, growth"
    DESCRIPTION = ("A complete city study in one run: surface temperature, heat island by day and night and its 20-year trend, "
                   "hot spots, cooling by parks and lakes, tree canopy, green space per person and the 3-30-300 rule, a heat "
                   "vulnerability map, access to hospitals and schools, growth, building on flood-prone ground and former lakes, "
                   "NO₂ air pollution, ward statistics and independent checks. Report in Word, PDF and HTML.")
    MAX_KM2 = 1500
    PARAMS = [
        dict(name="year", type="int", label="Year for the heat study (blank = last full year)", default=0),
        dict(name="hot_months", type="text", label="Hot-season months (comma list)", default="4,5,6",
             help="North India: 4,5,6. Europe/USA: 6,7,8. Southern hemisphere: 12,1,2."),
        dict(name="access_km", type="float", label="Acceptable distance to a hospital/clinic (km)", default=2.0),
        dict(name="school_km", type="float", label="Acceptable distance to a school (km)", default=1.0),
        dict(name="facilities", type="points", label="Your own health facilities (optional; else OpenStreetMap)", optional=True),
        dict(name="wealth_file", type="file", label="Relative Wealth Index CSV (optional; else downloaded from HDX)", optional=True),
        dict(name="air_days", type="int", label="Days of Sentinel-5P NO₂ to read (0 = skip; each ~60 MB)", default=24),
        dict(name="trend", type="bool", label="Heat island trend since 2003 (MODIS) and 2014 (Landsat)", default=True),
        dict(name="zones", type="polygons", label="Wards / zones for statistics (optional)", optional=True),
        dict(name="zone_field", type="field", label="Zone name field", parent="zones", optional=True),
    ]

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        R = dict(params=params)
        today = dt.date.today()
        try:
            months = [int(x) for x in str(params.get("hot_months") or "4,5,6").split(",") if x.strip()]
            assert months and all(1 <= m <= 12 for m in months)
        except Exception:
            raise KitError("Hot-season months must be numbers 1–12 separated by commas, e.g. 4,5,6.")
        year = int(params.get("year") or 0)
        if not year:
            year = today.year if dt.date(today.year, max(months), 28) < today - dt.timedelta(days=20) else today.year - 1
        R["year"], R["months"] = year, months
        g = Grid(aoi, 30, max_pixels=4_000_000)
        R["g"] = g

        # ------------------------------------------------------------ land, terrain, water
        self.step(3, "Land cover, elevation, water history")
        ly = min(max(year, D.LULC_YEARS[0]), D.LULC_YEARS[-1])
        lc, _ = D.lulc(g, ly)
        lc0, _ = D.lulc(g, D.LULC_YEARS[0])
        wc = D.worldcover(g)
        built = lc == 7
        R.update(lc=lc, lc0=lc0, wc=wc, built=built, lc_year=ly)
        g10 = Grid(aoi, 10, max_pixels=25_000_000)
        try:
            lc10, _ = D.lulc(g10, ly)
        except Exception:
            lc10, g10 = None, None
        try:
            elev = D.dem(g)
        except Exception:
            elev = None
        try:
            hand = D.hand(g)
        except Exception:
            hand = None
        try:
            occ = D.water_occurrence(g)
        except Exception:
            occ = None
        R.update(elev=elev, hand=hand, occ=occ)
        water = (lc == 1) | ((np.nan_to_num(occ) >= 50) if occ is not None else False)
        R["water"] = water

        # ------------------------------------------------------------ urban core and countryside ring
        self.step(10, "Finding the urban core and a matched countryside ring")
        urban, rural, width, note = HT.urban_rural(built.astype(np.float32), g, elev, water)
        R.update(urban=urban, rural=rural, ring_m=width, ur_note=note)

        # ------------------------------------------------------------ Landsat surface temperature
        self.step(14, f"Landsat surface temperature, {year} months {','.join(map(str, months))}")
        lst, ldates, per = HT.season_lst(g, year, months, n=8)
        R.update(lst=lst, ldates=ldates)
        if lst is None:
            raise KitError(f"No clear Landsat 8/9 thermal image in months {months} of {year}. Try other months or another year.")
        fin = np.isfinite(lst) & g.mask
        R["lst_lc"] = {k: float(lst[(lc == k) & fin].mean()) for k in D.LULC if ((lc == k) & fin).sum() >= 50}
        if urban is not None and rural is not None:
            R["suhi"] = float(np.nanmean(lst[urban & fin]) - np.nanmean(lst[rural & fin]))
            R["t_urban"], R["t_rural"] = float(np.nanmean(lst[urban & fin])), float(np.nanmean(lst[rural & fin]))
        # thermal comfort and hot spots
        u, ucls = LX.utfvi(lst, fin)
        R["utfvi"] = ucls
        k = max(1, int(round(300 / g.res)))
        z, _ = FX.gistar(lst, fin, k)
        R["gi"] = FX.gi_classes(FX.expand(z, k, g.shape)) if z is not None else None
        if R["gi"] is not None:
            R["gi"][~g.mask] = 0

        # ------------------------------------------------------------ independent check: MODIS on the same days
        self.step(26, "Checking Landsat against MODIS on the same days")
        gk = Grid(aoi, 990)
        R["gk"] = gk
        cmp = []
        for d, a in sorted(per, key=lambda t: -np.isfinite(t[1][g.mask]).mean())[:3]:
            m = HT.modis_day(gk, d)
            if m is None:
                continue
            cover = HT.to_grid(np.where(np.isfinite(a), 1.0, 0.0), g, gk)
            la = HT.to_grid(a, g, gk)
            la[cover < 0.8] = np.nan
            c = HT.compare(la, m, gk.mask)
            if c:
                c["date"] = d
                cmp.append(c)
        R["modis_cmp"] = cmp

        # ------------------------------------------------------------ MODIS day / night
        self.step(30, "MODIS day and night temperature (heat island by night)")
        urban_k = rural_k = None
        if urban is not None and rural is not None:
            urban_k = HT.to_grid(urban.astype(np.float32), g, gk) >= 0.5
            rural_k = HT.to_grid(rural.astype(np.float32), g, gk) >= 0.5
            R.update(urban_k=urban_k, rural_k=rural_k)
        try:
            ms = HT.modis_season(gk, year, months)
            R["modis_season"] = ms
            if urban_k is not None and urban_k.sum() >= 1 and rural_k.sum() >= 1:
                R["suhi_modis"] = {v: float(np.nanmean(a[urban_k]) - np.nanmean(a[rural_k])) for v, a in ms.items()
                                   if np.isfinite(a[urban_k]).any() and np.isfinite(a[rural_k]).any()}
        except Exception as ex:
            R["modis_err"] = str(ex)

        # ------------------------------------------------------------ trends
        want_trend = str(params.get("trend", True)).strip().lower() not in ("false", "0", "no", "off", "")
        if want_trend and urban_k is not None and urban_k.sum() >= 1 and rural_k.sum() >= 1:
            yrs = list(range(2003, year + 1))
            tr = {}
            for i, y in enumerate(yrs):
                self.step(34 + 12 * i / len(yrs), f"Heat island trend: MODIS {y}")
                try:
                    ms = HT.modis_season(gk, y, months, platforms=("aqua",))
                except Exception:
                    continue
                row = {}
                for v, a in ms.items():
                    if np.isfinite(a[urban_k]).any() and np.isfinite(a[rural_k]).any():
                        row[v] = float(np.nanmean(a[urban_k]) - np.nanmean(a[rural_k]))
                        row[v + "_urban"] = float(np.nanmean(a[urban_k]))
                if row:
                    tr[y] = row
            R["modis_trend"] = tr
            for v in ("Day", "Night"):
                vals = [tr[y].get(v) for y in sorted(tr)]
                if sum(x is not None for x in vals) >= 8:
                    mk = HY.mann_kendall([np.nan if x is None else x for x in vals])
                    R[f"mk_{v}"] = mk
            g120 = Grid(aoi, 120)
            uc = HT.to_grid(urban.astype(np.float32), g, g120) >= 0.5
            rc = HT.to_grid(rural.astype(np.float32), g, g120) >= 0.5

            def prog(i, n, y):
                self.step(46 + 8 * i / max(n, 1), f"Heat island trend: Landsat {y}")
            R["landsat_trend"] = HT.yearly_suhi(aoi, g120, uc, rc, months, list(range(2014, year + 1)), progress=prog)
            vals = [R["landsat_trend"][y]["suhi"] for y in sorted(R["landsat_trend"])]
            if len(vals) >= 6:
                R["mk_landsat"] = HY.mann_kendall(vals)

        # ------------------------------------------------------------ people
        self.step(55, "People: Meta 30 m population and GHS-POP")
        pop = CT.population_hrsl(g)
        R["pop_src"] = "Meta HRSL (30 m)" if pop is not None and np.nansum(pop) > 0 else None
        try:
            ghs, ep = LX.population(g, min(year, 2030))
        except Exception:
            ghs, ep = None, None
        if R["pop_src"] is None and ghs is not None:
            pop, R["pop_src"] = ghs, f"GHS-POP {ep}"
        R.update(pop=pop, ghs=ghs, ghs_epoch=ep)

        # ------------------------------------------------------------ green
        self.step(60, "Tree canopy, green space per person, 3-30-300")
        tree10 = green10 = None
        if g10 is not None:
            from .landslide import detect as DT
            try:
                d0 = HT.months_between(year, months)
                hot, _, hd = DT.ndvi_composite(aoi, g10, d0[0][0], d0[-1][1], max_scenes=4)
                mx, _, md = DT.ndvi_composite(aoi, g10, dt.date(year, 1, 1), dt.date(year, 12, 31), max_scenes=8, stat="max")
                if hot is not None and mx is not None:
                    # trees stay green in the hot dry season; grass and crops dry out (canopy: NDVI >= 0.40 then; balanced against ESA WorldCover trees in Gurugram and Bengaluru)
                    w10 = (lc10 == 1) if lc10 is not None else np.zeros(g10.shape, bool)
                    tree10 = (np.nan_to_num(hot) >= 0.40) & ~w10
                    crop10 = (lc10 == 5) if lc10 is not None else np.zeros(g10.shape, bool)
                    green10 = tree10 | ((np.nan_to_num(mx) >= 0.5) & ~w10 & ~crop10)   # farm fields are not green space
                    R["s2_dates"] = (hd, md)
            except Exception as ex_:
                R["s2_err"] = str(ex_)
            if tree10 is None and lc10 is not None:
                tree10, green10 = lc10 == 2, np.isin(lc10, [2, 11])
                R["canopy_src"] = "Esri land cover (Sentinel-2 not available)"
            else:
                R["canopy_src"] = "Sentinel-2 dry-season NDVI ≥ 0.40"
            try:
                wc10 = D.worldcover(g10)
                if wc10 is not None and tree10 is not None:
                    wt = wc10 == 2
                    R["canopy_check"] = (float(tree10[g10.mask].mean() * 100), float(wt[g10.mask].mean() * 100),
                                         float((tree10 & wt).sum() / max(wt.sum(), 1) * 100))
                    kb = 10   # 100 m blocks: pixel-exact overlap is harsh for scattered trees and 2021 vs today
                    rr, cc = (g10.shape[0] // kb) * kb, (g10.shape[1] // kb) * kb
                    bl = lambda x: x[:rr, :cc].reshape(rr // kb, kb, cc // kb, kb).mean((1, 3))
                    m10 = bl(g10.mask.astype(np.float32)) > 0.9
                    A_, B_ = bl(tree10.astype(np.float32))[m10], bl(wt.astype(np.float32))[m10]
                    if A_.size >= 30 and A_.std() > 0 and B_.std() > 0:
                        R["canopy_block"] = (float(np.corrcoef(A_, B_)[0, 1]), float(np.mean(np.abs(A_ - B_)) * 100))
            except Exception:
                pass
        if tree10 is not None:
            G = CT.green_metrics(tree10, green10, g10, g, pop, urban, built)
            R["green"] = G
            # cooling by parks and water bodies
            near = (g.distance_to(urban) <= 1000) & g.mask if urban is not None else g.mask
            parks = G["green_patches"] & (np.nan_to_num(G["tree"]) >= 0.5)
            lab, n = FX.label(parks, g)
            if n:
                sz = np.bincount(lab.ravel())
                keep = np.flatnonzero(sz >= int(np.ceil(10000 / g.res ** 2)))
                parks = np.isin(lab, keep[keep > 0])
            R["cool_park"] = HT.distance_profile(lst, parks & near, g, near & ~parks & ~water & built)
            wb = water & near
            R["cool_water"] = HT.distance_profile(lst, wb, g, near & ~water & built) if wb.sum() * g.px_km2 >= 0.01 else None
            # canopy vs temperature at 300 m
            tc, _ = CT.block(np.where(fin, G["tree"], np.nan), k)
            tl, _ = CT.block(np.where(fin, lst, np.nan), k)
            bf, _ = CT.block(np.where(g.mask, built.astype(np.float32), np.nan), k)
            ok = np.isfinite(tc) & np.isfinite(tl) & np.isfinite(bf)
            if ok.sum() > 20:
                R["r_tree_lst"] = float(np.corrcoef(tc[ok], tl[ok])[0, 1])
                R["r_built_lst"] = float(np.corrcoef(bf[ok], tl[ok])[0, 1])
                b = np.polyfit(tc[ok] * 100, tl[ok], 1)
                R["slope_tree"] = float(b[0] * 10)   # °C per +10% canopy
                from .urban import surface as SF
                # neighbouring 300 m blocks are not independent: take the standard error from every 3rd block (~900 m apart)
                th = np.zeros_like(ok)
                th[::3, ::3] = True
                okt = ok & th
                if okt.sum() >= 15:
                    b1, se1 = SF.slope_se(tc[okt] * 100, tl[okt])
                else:
                    b1, se1 = SF.slope_se(tc[ok] * 100, tl[ok])
                R["slope_tree_se"] = float(se1 * 10)

        # ------------------------------------------------------------ heat vulnerability
        self.step(64, "Heat vulnerability: temperature × people × lack of green")
        if pop is not None and R.get("green"):
            pk, _ = CT.block(np.where(g.mask, pop, np.nan), k, "sum")
            tl, _ = CT.block(np.where(fin, lst, np.nan), k)
            gr, _ = CT.block(np.where(g.mask, R["green"]["green"], np.nan), k)
            m = np.isfinite(pk) & np.isfinite(tl) & np.isfinite(gr) & (pk > 1)
            hz = CT.zscore(tl, m)
            ex = CT.zscore(np.log1p(np.nan_to_num(pk)), m)
            se = CT.zscore(1 - gr, m)
            comps_ = [hz, ex, se]
            # relative wealth (Meta RWI, 2.4 km tiles; HDX, CC BY-NC): poorer neighbourhoods cope worse with heat
            try:
                from .urban import wealth as WL
                pts_, R["rwi_note"] = WL.fetch(aoi.bbox, params.get("wealth_file") or None)
                if pts_:
                    rw = WL.to_grid(pts_, g)
                    rb, _ = CT.block(rw, k)
                    if np.isfinite(rb[m]).mean() > 0.5:
                        comps_.append(CT.zscore(np.where(np.isfinite(rb), -rb, np.nanmean(-rb[m])), m))
                        R["rwi"] = rw
                        R["rwi_n"] = len(pts_)
                        # plausibility: richer tiles should be more built-up in a city region (Chi et al. 2022)
                        bfr, _ = CT.block(np.where(g.mask, built.astype(np.float32), np.nan), max(1, int(round(2400 / g.res))))
                        rwb, _ = CT.block(rw, max(1, int(round(2400 / g.res))))
                        okb = np.isfinite(bfr) & np.isfinite(rwb)
                        if okb.sum() >= 10 and bfr[okb].std() > 0 and rwb[okb].std() > 0:
                            R["rwi_check"] = (float(np.corrcoef(bfr[okb], rwb[okb])[0, 1]), int(okb.sum()))
            except Exception as ex_:
                R["rwi_note"] = f"not read ({ex_})"
            hvi = np.where(m, sum(comps_) / len(comps_), np.nan)
            R["hvi_parts"] = len(comps_)
            q = CT.quintiles(hvi, m)
            R["hvi"] = np.nan_to_num(CT.expand(q.astype(np.float32), k, g.shape)).astype(np.uint8)
            R["hvi"][~g.mask] = 0
            popc = CT.expand(np.where(m, pk, np.nan), k, g.shape)
            R["hvi_pop"] = {c: float(np.nansum(pk[q == c])) for c in HVI}

        # ------------------------------------------------------------ access
        self.step(68, "Access to hospitals, schools and green space")
        lim = float(params.get("access_km") or 2.0)
        slim = float(params.get("school_km") or 1.0)
        acc = {}
        roads_m = None
        try:
            from .urban import network as NW
            roads_m = NW.road_mask(g, NW.fetch_roads(aoi.bbox))
            R["roads_note"] = f"{int(roads_m.sum() * g.res / 1000):,} km of OpenStreetMap roads"
        except Exception as ex_:
            R["roads_note"] = "OpenStreetMap roads could not be read; distances are straight lines"
            R["roads_err"] = str(ex_)
        for kind, limit in (("health", lim), ("school", slim)):
            fac, src = [], None
            try:
                if kind == "health" and params.get("facilities"):
                    for gl, attrs, gw in read_vector(params["facilities"], aoi):
                        p = gw.Centroid()
                        fac.append((p.GetX(), p.GetY(), str(attrs.get("name") or ""), "your layer"))
                    src = "your layer"
                else:
                    fac = osm.facilities(aoi.bbox, kind)
                    src = "OpenStreetMap"
            except Exception as ex:
                acc[kind] = dict(err=str(ex))
                continue
            if not fac:
                acc[kind] = dict(err="no facilities found")
                continue
            seed = np.zeros(g.shape, bool)
            for lon, lat, *_ in fac:
                r, c = g.lonlat_to_pixel(lon, lat)
                if -300 <= r < g.rows + 300 and -300 <= c < g.cols + 300:
                    seed[min(max(r, 0), g.rows - 1), min(max(c, 0), g.cols - 1)] = True
            a = CT.access(g, seed, pop, limit, roads=roads_m)
            a.update(n=len(fac), src=src, limit=limit, fac=fac)
            acc[kind] = a
        R["access"] = acc

        # ------------------------------------------------------------ growth and exposure
        self.step(72, "Growth, building on flood-prone ground and on former water")
        b0 = lc0 == 7
        gr = np.zeros(g.shape, np.uint8)
        gr[g.mask & (lc0 > 0) & (lc > 0)] = 3
        gr[b0 & built] = 1
        new = built & ~b0 & (lc0 > 0)
        gr[new] = 2
        R["growth"] = gr
        R["growth_from"] = {kk: float((new & (lc0 == kk)).sum() * g.px_km2) for kk in D.LULC if kk != 7}
        R["built_km2"] = (float(b0.sum() * g.px_km2), float(built.sum() * g.px_km2))
        ex = np.zeros(g.shape, np.uint8)
        ex[built] = 4
        # waterlogging pockets: ground at least 2 m below the mean of the surrounding 1 km
        if elev is not None:
            pocket = built & (elev - CT.box_mean(elev, int(round(500 / g.res))) <= -2.0)
            ex[pocket] = 3
            R["built_pocket_km2"] = float(pocket.sum() * g.px_km2)
        # river flood zone: GloFAS 1-in-100-year map; HAND < 5 m only where the terrain is not flat
        from .flood import impact as IM
        fz = np.zeros(g.shape, bool)
        gl = IM.glofas(g, 100)
        if gl is not None:
            fz |= np.isfinite(gl)
            R["glofas"] = True
        if hand is not None:
            low = built & (np.nan_to_num(hand, nan=99) < 5)
            R["hand_flat"] = bool(low.sum() > 0.5 * max(built.sum(), 1))
            if not R["hand_flat"]:
                fz |= np.nan_to_num(hand, nan=99) < 5
        fzb = built & fz
        if gl is not None or (hand is not None and not R["hand_flat"]):
            ex[fzb] = 1
            R["built_flood_km2"] = float(fzb.sum() * g.px_km2)
            R["built_flood_new_km2"] = float((fzb & new).sum() * g.px_km2)
            if pop is not None:
                R["pop_flood"] = float(np.nansum(pop[fzb]))
        if occ is not None:
            former = built & (np.nan_to_num(occ) >= 10)
            ex[former] = 2
            R["built_on_water_km2"] = float(former.sum() * g.px_km2)
            R["built_on_water_new_km2"] = float((former & new).sum() * g.px_km2)
        R["expo"] = ex
        if new.any():
            R["lst_new_vs_old"] = (float(np.nanmean(lst[new & fin])) if (new & fin).any() else None,
                                   float(np.nanmean(lst[b0 & built & fin])) if (b0 & built & fin).any() else None)

        # ------------------------------------------------------------ air
        nd = int(params.get("air_days") if params.get("air_days") is not None else 24)
        if nd > 0:
            g5 = Grid(aoi, 2000)
            R["g5"] = g5

            def prog(i, n):
                self.step(76 + 8 * i / max(n, 1), f"Air quality: Sentinel-5P NO₂ ({i + 1}/{n} days)")
            try:
                mean, mon, used = CT.no2(aoi, g5, days=nd, progress=prog)
                R["no2"], R["no2_month"], R["no2_days"] = mean, mon, used
                if mean is not None and urban is not None:
                    u5 = HT.to_grid(urban.astype(np.float32), g, g5) >= 0.3
                    r5 = (HT.to_grid(rural.astype(np.float32), g, g5) >= 0.3) if rural is not None else ~u5
                    if np.isfinite(mean[u5]).any() and np.isfinite(mean[r5 & ~u5]).any():
                        R["no2_city"], R["no2_bg"] = float(np.nanmean(mean[u5])), float(np.nanmean(mean[r5 & ~u5]))
            except Exception as ex:
                R["no2_err"] = str(ex)

        # ------------------------------------------------------------ air temperature, heat stress, PM2.5, building height
        from .urban import climate as CM
        lon_c, lat_c = (aoi.bbox[0] + aoi.bbox[2]) / 2, (aoi.bbox[1] + aoi.bbox[3]) / 2
        self.step(84, "Air temperature and heat stress (NASA POWER)")
        try:
            R["air"] = CM.air_heat(lat_c, lon_c, year, months)
            tr = R["air"].get("trend")
            if tr and len(tr["years"]) >= 10:
                R["air_trend"] = dict(tmax=CM.mann_kendall_slope(tr["tmax"]), tmin=CM.mann_kendall_slope(tr["tmin"]))
            # cross-check: hot-season night air temperature vs MODIS night LST in the countryside ring
            ms = R.get("modis_season") or {}
            if ms.get("Night") is not None and R.get("rural_k") is not None:
                R["air_vs_modis_night"] = (R["air"]["tmin_mean"], float(np.nanmean(ms["Night"][R["rural_k"]])))
        except Exception as ex:
            R["air_err"] = str(ex)
        self.step(85, "Fine particles PM2.5 (Copernicus CAMS)")
        try:
            R["pm25"] = CM.pm25(aoi.bbox, year)
        except Exception as ex:
            R["pm25_err"] = str(ex)
        self.step(86, "Building height (JRC GHS-BUILT-H)")
        try:
            hgt = CM.building_height(g)
            if hgt is not None and np.isfinite(hgt[built]).any():
                R["height"] = hgt
                hb = hgt[built & np.isfinite(hgt)]
                R["height_stats"] = dict(mean=float(hb.mean()), p90=float(np.percentile(hb, 90)), max=float(hb.max()),
                                         share_tall=float((hb >= 15).mean() * 100))
                if R.get("urban") is not None and R["urban"].any():
                    hu = hgt[R["urban"] & np.isfinite(hgt)]
                    R["height_stats"]["core"] = float(hu.mean()) if hu.size else None
                k = max(1, int(round(300 / g.res)))
                hl, _ = CT.block(np.where(built, hgt, np.nan), k)
                tl3, _ = CT.block(lst, k)
                ok = np.isfinite(hl) & np.isfinite(tl3)
                if ok.sum() >= 30:
                    R["r_height_lst"] = float(np.corrcoef(hl[ok], tl3[ok])[0, 1])
        except Exception as ex:
            R["height_err"] = str(ex)

        # ------------------------------------------------------------ albedo, cool roofs, local climate zones, greening scenario
        from .urban import surface as SF
        self.step(86.5, "Surface albedo and cool-roof potential (Landsat)")
        try:
            alb = SF.season_albedo(g, year, months, R.get("ldates") or [])
            if alb is not None and np.isfinite(alb[g.mask]).mean() > 0.3:
                R["albedo"] = alb
                fa = np.isfinite(alb) & fin
                R["albedo_lc"] = {kk: float(np.nanmean(alb[(lc == kk) & fa])) for kk in D.LULC if ((lc == kk) & fa).sum() >= 50}
                bb = built & fa
                if bb.sum() >= 200:
                    dark = bb & (alb < 0.15)
                    R["roof"] = dict(dark_km2=float(dark.sum() * g.px_km2), built_km2=float(bb.sum() * g.px_km2),
                                     lst_dark=float(np.nanmean(lst[dark])) if dark.any() else None,
                                     lst_bright=float(np.nanmean(lst[bb & (alb >= 0.20)])) if (bb & (alb >= 0.20)).any() else None,
                                     alb_built=float(np.nanmean(alb[bb])))
                    ab, _ = CT.block(np.where(bb, alb, np.nan), k)
                    tb_, _ = CT.block(np.where(bb, lst, np.nan), k)
                    okb = np.isfinite(ab) & np.isfinite(tb_)
                    if okb.sum() >= 30:
                        thb = np.zeros_like(okb)
                        thb[::3, ::3] = True
                        s_, se_ = SF.slope_se(ab[okb & thb], tb_[okb & thb]) if (okb & thb).sum() >= 15 else SF.slope_se(ab[okb], tb_[okb])
                        R["roof"].update(slope=s_ / 10, se=se_ / 10, r=float(np.corrcoef(ab[okb], tb_[okb])[0, 1]), n=int(okb.sum()))
        except Exception as ex:
            R["albedo_err"] = str(ex)
        self.step(87, "Local Climate Zones (global 100 m map)")
        try:
            lz = SF.lcz_map(g)
            if (lz > 0).any():
                R["lcz"] = lz
                R["lcz_rows"] = [dict(code=c_, km2=float((lz == c_).sum() * g.px_km2),
                                      lst=float(np.nanmean(lst[(lz == c_) & fin])) if ((lz == c_) & fin).sum() >= 30 else None,
                                      pop=float(np.nansum(np.where(lz == c_, pop, 0))) if pop is not None else None)
                                 for c_ in SF.LCZ if (lz == c_).sum() * g.px_km2 >= 0.05]
                if lc is not None:
                    bl = (lz >= 1) & (lz <= 10)
                    R["lcz_check"] = float((bl & built).sum() / max(bl.sum(), 1) * 100), float((bl & built).sum() / max(built.sum(), 1) * 100)
        except Exception as ex:
            R["lcz_err"] = str(ex)
        if R.get("slope_tree") is not None and R.get("green"):
            s10, se10 = R["slope_tree"], R.get("slope_tree_se") or 0.0
            u = (urban if urban is not None else built) & np.isfinite(R["green"]["tree"])
            room = u & (np.nan_to_num(R["green"]["tree"]) <= 0.9)
            ppl = float(np.nansum(np.where(room, pop, 0))) if pop is not None else None
            hot = room & (lst >= np.nanpercentile(lst[fin], 80)) if fin.any() else room
            R["greening"] = dict(dT=s10, lo=s10 - 1.96 * se10, hi=s10 + 1.96 * se10, km2=float(room.sum() * g.px_km2),
                                 people=ppl, people_hot=float(np.nansum(np.where(hot, pop, 0))) if pop is not None else None,
                                 canopy_now=float(np.nanmean(R["green"]["tree"][u]) * 100) if u.any() else None)

        # ------------------------------------------------------------ zones
        if params.get("zones"):
            try:
                self._zones(R, params, aoi, g)
            except Exception as ex:
                R["zones_err"] = str(ex)

        # ------------------------------------------------------------ validation extras
        if wc is not None:
            bw = wc == 7
            R["built_agree"] = (float((bw & built).sum() / max(built.sum(), 1) * 100), float((bw & built).sum() / max(bw.sum(), 1) * 100))
        if ghs is not None and pop is not None and R["pop_src"].startswith("Meta"):
            R["pop_check"] = (float(np.nansum(pop)), float(np.nansum(ghs)), ep)

        self.step(88, "Writing maps and tables")
        files, layers = self._write(R, out_dir)
        self.step(93, "Writing the report")
        from .urban.report_urban import build
        return build(self, R, aoi, out_dir, files, layers)

    # ================================================================== zones
    def _zones(self, R, params, aoi, g):
        feats = read_vector(params["zones"], aoi)
        fld = params.get("zone_field")
        z = g.rasterize([x for x, _, _ in feats], values=list(range(1, len(feats) + 1)))
        lst, pop, G = R["lst"], R.get("pop"), R.get("green") or {}
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, (_, a, _) in enumerate(feats):
                m = (z == i + 1) & g.mask
                if not m.any():
                    continue
                r = dict(name=str(a.get(fld, i + 1)) if fld else str(i + 1), area=float(m.sum() * g.px_km2),
                         built=float(R["built"][m].mean() * 100), lst=float(np.nanmean(lst[m])) if np.isfinite(lst[m]).any() else None,
                         pop=float(np.nansum(pop[m])) if pop is not None else None,
                         canopy=float(np.nanmean(G["tree"][m]) * 100) if "tree" in G else None,
                         green_pp=float(np.nansum(G["green"][m]) * g.res ** 2 / max(np.nansum(pop[m]), 1)) if "green" in G and pop is not None else None,
                         hvi_high=float(np.isin(R["hvi"][m], [4, 5]).mean() * 100) if R.get("hvi") is not None else None)
                h = (R.get("access") or {}).get("health") or {}
                if "dist" in h and pop is not None:
                    pm = m & (np.nan_to_num(pop) > 0)
                    r["far_health"] = float(np.nansum(pop[pm & (h["dist"] > h["limit"])]) / max(np.nansum(pop[pm]), 1) * 100)
                rows.append(r)
        R["zones"] = rows

    # ================================================================== outputs
    def _write(self, R, out_dir):
        g = R["g"]
        files, layers = [], []

        def add(path, name, hidden=True, kind="raster"):
            files.append(path)
            layers.append(dict(path=path, name=name, kind=kind, hidden=hidden))
        lst = R["lst"]
        lo, hi = np.nanpercentile(lst[g.mask & np.isfinite(lst)], [2, 98])
        add(write_float(os.path.join(out_dir, f"surface_temperature_{R['year']}.tif"), lst, g, "heat", float(lo), float(hi), "°C")[0],
            f"Surface temperature {R['year']} (°C)", hidden=False)
        add(write_classes(os.path.join(out_dir, "utfvi_thermal_comfort.tif"), R["utfvi"], g, UTFVI), "Thermal comfort (UTFVI)")
        if R.get("gi") is not None:
            add(write_classes(os.path.join(out_dir, "heat_hotspots_gistar.tif"), R["gi"], g, FX.GISTAR), "Heat hot spots (Gi*)")
        if R.get("urban") is not None:
            zz = np.where(R["urban"], 1, np.where(R["rural"] if R.get("rural") is not None else False, 2, np.where(g.mask, 3, 0))).astype(np.uint8)
            add(write_classes(os.path.join(out_dir, "urban_core_and_countryside.tif"), zz, g, ZONES), "Urban core and countryside ring")
        if R.get("hvi") is not None:
            add(write_classes(os.path.join(out_dir, "heat_vulnerability_index.tif"), R["hvi"], g, HVI), "Heat vulnerability", hidden=False)
        if R.get("albedo") is not None:
            add(write_float(os.path.join(out_dir, "surface_albedo.tif"), R["albedo"], g, "seq", 0.05, 0.35, "albedo")[0], "Surface albedo")
        if R.get("lcz") is not None:
            from .urban import surface as SF
            add(write_classes(os.path.join(out_dir, "local_climate_zones.tif"), R["lcz"], g,
                              {c_: v for c_, v in SF.LCZ.items() if (R["lcz"] == c_).any()}), "Local Climate Zones")
        if R.get("height") is not None:
            add(write_float(os.path.join(out_dir, "building_height_m.tif"), R["height"], g, "seq", 0,
                            float(max(np.nanpercentile(R["height"], 99), 3)), "m")[0], "Building height 2018 (m)")
        for key, name in (("air", "air_temperature_daily.csv"), ("pm25", "pm25_daily.csv")):
            dd = (R.get(key) or {}).get("daily")
            if dd:
                fn = os.path.join(out_dir, name)
                with open(fn, "w", encoding="utf-8") as fh:
                    if key == "air":
                        fh.write("date,tmax_c,tmin_c,heat_index_max_c\n")
                        fh.writelines(f"{d[:4]}-{d[4:6]}-{d[6:]},{a:.2f},{b:.2f},{c:.2f}\n" for d, a, b, c in dd)
                    else:
                        fh.write("date,pm25_ugm3\n")
                        fh.writelines(f"{d},{v:.2f}\n" for d, v in dd)
                files.append(fn)
        G = R.get("green")
        if G:
            add(write_float(os.path.join(out_dir, "tree_canopy_fraction.tif"), np.where(g.mask, G["tree"], np.nan), g, "ndvi", 0, 1, "fraction")[0], "Tree canopy")
            add(write_float(os.path.join(out_dir, "distance_to_green_space_m.tif"), np.where(g.mask, np.minimum(G["dist_green"], 3000), np.nan), g,
                            "risk", 0, 1000, "m")[0], "Distance to green space ≥ 0.5 ha")
        if R.get("pop") is not None:
            add(write_float(os.path.join(out_dir, "population_per_cell.tif"), R["pop"], g, "seq", 0, float(np.nanpercentile(R["pop"][R["pop"] > 0], 98)) if (R["pop"] > 0).any() else 1, "people")[0], "Population")
        for kind, a in (R.get("access") or {}).items():
            if "dist" in a:
                add(write_float(os.path.join(out_dir, f"distance_to_{kind}_km.tif"), a["dist"], g, "risk", 0, 3 * a["limit"], "km")[0], f"Distance to {kind}")
                p = write_geojson(os.path.join(out_dir, f"facilities_{kind}.geojson"),
                                  [((lon, lat), {"name": n, "type": t, "source": a["src"]}) for lon, lat, n, t in a["fac"]])
                add(p, f"Facilities ({kind})", kind="vector")
        add(write_classes(os.path.join(out_dir, f"built_up_growth_{D.LULC_YEARS[0]}_{R['lc_year']}.tif"), R["growth"], g, GROWTH), "Built-up growth")
        add(write_classes(os.path.join(out_dir, "built_up_exposure.tif"), R["expo"], g, EXPO), "Built-up on flood-prone ground / former water")
        if R.get("no2") is not None:
            v = R["no2"]
            add(write_float(os.path.join(out_dir, "no2_tropospheric_umol_m2.tif"), v, R["g5"], "heat", float(np.nanmin(v)), float(np.nanmax(v)), "µmol/m²")[0],
                "NO₂ (Sentinel-5P)")
        # tables
        p = os.path.join(out_dir, "heat_island_trend.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("year,modis_day_suhi_C,modis_night_suhi_C,landsat_suhi_C\n")
            tr, lt = R.get("modis_trend") or {}, R.get("landsat_trend") or {}
            for y in sorted(set(tr) | set(lt)):
                f = lambda v: "" if v is None else f"{v:.2f}"
                fh.write(f"{y},{f(tr.get(y, {}).get('Day'))},{f(tr.get(y, {}).get('Night'))},{f(lt.get(y, {}).get('suhi'))}\n")
        files.append(p)
        p = os.path.join(out_dir, "temperature_by_land_cover.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("land_cover,mean_lst_C\n")
            for kk, v in R["lst_lc"].items():
                fh.write(f"{D.LULC[kk][0]},{v:.2f}\n")
        files.append(p)
        for nm in ("cool_park", "cool_water"):
            c = R.get(nm)
            if c:
                p = os.path.join(out_dir, f"{nm.replace('cool_', 'cooling_by_')}.csv")
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write("distance_m,mean_lst_C\n")
                    for x, y in zip(c[0], c[1]):
                        fh.write(f"{x},{y:.2f}\n")
                files.append(p)
        if R.get("no2_month"):
            p = os.path.join(out_dir, "no2_monthly.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("month,mean_no2_umol_m2\n")
                for kk, v in R["no2_month"].items():
                    fh.write(f"{kk},{v:.1f}\n")
            files.append(p)
        if R.get("zones"):
            p = os.path.join(out_dir, "zone_statistics.csv")
            keys = list(R["zones"][0].keys())
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(",".join(keys) + "\n")
                for r in R["zones"]:
                    fh.write(",".join("" if r.get(kk) is None else (f"{r[kk]:.3f}" if isinstance(r[kk], float) else str(r[kk])) for kk in keys) + "\n")
            files.append(p)
        # random check points on built-up land (for checking hot spots on imagery)
        rng = np.random.default_rng(3)
        idx = np.flatnonzero((R["built"] & g.mask & np.isfinite(lst)).ravel())
        pts = []
        for i in rng.choice(idx, min(60, idx.size), replace=False) if idx.size else []:
            r, c = divmod(int(i), g.cols)
            lon, lat = g.pixel_to_lonlat(r, c)
            pts.append(((lon, lat), {"lst_C": round(float(lst[r, c]), 1), "hvi": int(R["hvi"][r, c]) if R.get("hvi") is not None else None,
                                     "correct": ""}))
        if pts:
            add(write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts), "Check points", kind="vector")
        return files, layers
