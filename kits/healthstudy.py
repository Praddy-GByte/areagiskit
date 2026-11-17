# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: Health: a complete small-area disease study in one run.

From your areas and case counts: rates with 95% intervals, expected cases and SMR, probability of excess risk, global and
spatial empirical Bayes smoothing, global Moran's I, local Moran (LISA) clusters and outliers, Getis-Ord Gi* hot spots with
FDR control, Kulldorff's spatial scan (most likely and secondary clusters), environment of every area read automatically
from satellites (population density, built-up, elevation, greenness, surface temperature, surface water, distance to care),
correlations, Geodetector and Poisson regression of the drivers, time trends and emerging hot spots if you give periods,
access to care, a planted-cluster test on your own map, report and proof.
"""

import csv
import datetime as dt
import json
import math
import os
import warnings

import numpy as np
from osgeo import gdal, ogr, osr

from ..core import datasets as D
from ..core import osm
from ..core.aoi import AOI, Grid, read_vector
from ..core.data import KitError, search, warp_items
from .base import Kit
from .flood import hydro as HY
from .health import stats as ST
from ..core.net import skipped

HOT = {1: ("Hot spot, 99%", "#9e1b16"), 2: ("Hot spot, 95%", "#d6402b"), 3: ("Hot spot, 90%", "#f08a6a"),
       4: ("Not significant", "#ece9df"), 5: ("Cold spot, 90%", "#9cc2e0"), 6: ("Cold spot, 95%", "#4a8cc4"),
       7: ("Cold spot, 99%", "#1d5a96")}
LISA = {1: ("High-High cluster", "#d7191c"), 2: ("Low-Low cluster", "#2c7bb6"), 3: ("Low-High outlier", "#abd9e9"),
        4: ("High-Low outlier", "#fdae61"), 5: ("Not significant", "#ece9df")}
RATE5 = ["#fdf0d5", "#f6c58f", "#ea8a50", "#c9502f", "#8b2318"]
ENV = [("pop_density", "Population density (people/km²)"), ("built", "Built-up share (%)"), ("elev", "Elevation (m)"),
       ("ndvi", "Greenness (MODIS NDVI)"), ("lst", "Day surface temperature (°C)"), ("water", "Surface water share (%)"),
       ("dist_fac", "Distance to nearest health facility (km)")]


def _num(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _key(v):
    """Area ID as text, so 37009, 37009.0 and ' 37009 ' all match between the layer and a CSV."""
    s = str(v).strip()
    try:
        f = float(s)
        if f.is_integer() and ("." in s or "e" in s.lower()):
            return str(int(f))
    except ValueError:
        pass
    return s


def _hav(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class HealthStudyKit(Kit):
    NAME = "AreaGIS Kit: Health"
    KEY = "healthstudy"
    TAGLINE = "Complete disease study: rates, clusters, scan statistic, environmental drivers, trends, access"
    DESCRIPTION = ("A complete small-area disease study from your areas and case counts: rates with confidence intervals, SMR and "
                   "excess-risk probability, smoothing, Moran's I, LISA, Gi* hot spots, Kulldorff's scan statistic, the environment "
                   "of every area from satellites (people, built-up, height, greenness, heat, water), drivers by Poisson regression "
                   "and Geodetector, time trends and emerging hot spots, access to care, and a power test on your own map. "
                   "Report in Word, PDF and HTML.")
    NEEDS_AOI = False
    MAX_KM2 = 200000
    PARAMS = [
        dict(name="areas", type="polygons", label="Areas (wards, villages, blocks, districts)"),
        dict(name="id_field", type="field", label="Area ID field", parent="areas"),
        dict(name="cases_csv", type="file", label="Cases CSV (optional, else use layer fields)", optional=True),
        dict(name="cases_field", type="text", label="Cases column (several, comma-separated = one per period, e.g. SID74,SID79)", default="cases",
             help="One column, or one column per period (wide table). With age groups, leave as is: totals come from the age columns."),
        dict(name="population_field", type="text", label="Population column (blank = estimate from Meta 30 m population)", default="population",
             help="One column, or one per period matching the cases columns (e.g. BIR74,BIR79)."),
        dict(name="age_groups", type="text", label="Age groups for age standardisation (optional, e.g. 0-4,5-14,15-44,45-64,65+)", default="",
             help="Then the table needs a cases and a population column for every group, named cases_<group> and pop_<group>, "
                  "e.g. cases_0-4 and pop_0-4, cases_65+ and pop_65+ (cases_0_4 / cases_65plus also work). Direct rates use the "
                  "WHO World Standard Population 2000–2025; SMR, smoothing, clusters and the scan then use age-adjusted expected cases."),
        dict(name="period_field", type="text", label="Period column in the CSV for trends (e.g. year; optional)", default=""),
        dict(name="disease", type="text", label="Name of disease / indicator", default="Cases"),
        dict(name="per", type="int", label="Rates per", default=100000),
        dict(name="facilities", type="points", label="Health facilities (optional; else OpenStreetMap)", optional=True),
        dict(name="access_km", type="float", label="Acceptable distance to a facility (km)", default=5.0),
        dict(name="environment", type="bool", label="Read the environment of each area from satellites", default=True),
        dict(name="capacity_field", type="text", label="Facility capacity field for the 2SFCA access index (beds, doctors; optional)", default=""),
        dict(name="case_points", type="points", label="Case locations as points (optional: density map, risk surface, Knox test)", optional=True),
        dict(name="case_date_field", type="text", label="Date field of the case points (optional, YYYY-MM-DD; for the Knox test)", default=""),
        dict(name="knox_km", type="float", label="Knox test: cases closer than (km)", default=2.0),
        dict(name="knox_days", type="int", label="Knox test: and closer in time than (days)", default=14),
        dict(name="suppress_below", type="int", label="Privacy: hide counts and crude rates of areas with fewer cases than (0 = off)", default=5),
    ]

    # ================================================================== input
    def _read(self, params, aoi):
        path = params.get("areas")
        if not path:
            raise KitError("Choose the polygon layer of areas.")
        probe = ogr.Open(path)
        if probe is None:
            raise KitError(f"Could not open {path}.")
        lyr = probe.GetLayer(0)
        x0, x1, y0, y1 = lyr.GetExtent()
        srs = lyr.GetSpatialRef()
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if srs is not None and not srs.IsSame(wgs):
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ct = osr.CoordinateTransformation(srs, wgs)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
                ring.AddPoint_2D(x, y)
            pg = ogr.Geometry(ogr.wkbPolygon)
            pg.AddGeometry(ring)
            pg.Transform(ct)
            x0, x1, y0, y1 = pg.GetEnvelope()
        aoi = aoi or AOI(bbox=[x0, y0, x1, y1], name=os.path.splitext(os.path.basename(path))[0])
        feats = read_vector(path, aoi)
        if len(feats) < 8:
            raise KitError("The AreaGIS Kit: Health needs at least 8 areas.")
        idf = params.get("id_field") or ""
        if idf and idf not in feats[0][1]:
            raise KitError(f"Field '{idf}' not found in the layer.")
        cf = params.get("cases_field") or "cases"
        pf = (params.get("population_field") or "").strip()
        tf = (params.get("period_field") or "").strip()
        wide_c = [x.strip() for x in cf.split(",") if x.strip()] if "," in cf else None
        wide_p = [x.strip() for x in pf.split(",") if x.strip()] if (pf and "," in pf) else None
        if wide_p and (not wide_c or len(wide_p) != len(wide_c)):
            raise KitError("Give as many population columns as cases columns (one per period).")
        groups = []
        if (params.get("age_groups") or "").strip():
            from .health.advanced import parse_groups
            try:
                groups = parse_groups(params["age_groups"])
            except ValueError as ex:
                raise KitError(str(ex))
            if wide_c:
                raise KitError("Use either several period columns or age groups, not both; for both, use a CSV with a period column.")
        rows = []
        if params.get("cases_csv"):
            with open(params["cases_csv"], newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                raise KitError("The CSV is empty.")
        key = (idf if rows and idf in rows[0] else (list(rows[0].keys())[0] if rows else None))
        age_cols = None
        if groups:
            from .health.advanced import column_names
            have = set(rows[0].keys()) if rows else set(feats[0][1].keys())
            low = {h.lower(): h for h in have}
            age_cols = []
            for lab, _, _ in groups:
                cc = next((low[x.lower()] for x in column_names(lab, "cases") if x.lower() in low), None)
                pp = next((low[x.lower()] for x in column_names(lab, "pop") if x.lower() in low), None)
                if cc is None or pp is None:
                    raise KitError(f"Age group '{lab}': columns cases_{lab} and pop_{lab} not found "
                                   f"(found: {', '.join(sorted(have))[:300]}).")
                age_cols.append((cc, pp))
        by = {}
        periods = {}
        for r in rows:
            k = _key(r.get(key, ""))
            if tf and tf in r:
                periods.setdefault(str(r[tf]).strip(), {})[k] = r
            by.setdefault(k, []).append(r)
        ids, gl, gw, C, P = [], [], [], [], []
        AC, AP = [], []
        unmatched = 0
        if wide_c:
            periods = {k: {} for k in wide_c}
        for i, (g_l, attrs, g_w) in enumerate(feats):
            fid = _key(attrs.get(idf, i) if idf else i)
            if rows:
                rr = by.get(fid)
                if rr is None:
                    unmatched += 1
                    c = p = None
                else:   # several periods: total cases, mean population
                    cs = [_num(r.get(cf)) for r in rr]
                    ps = [_num(r.get(pf)) for r in rr] if pf else []
                    c = sum(x for x in cs if x is not None) if any(x is not None for x in cs) else None
                    pv = [x for x in ps if x is not None]
                    p = (np.mean(pv) * (len(rr) if tf else 1)) if pv else None
                    if p is None and pf in attrs:
                        p = _num(attrs.get(pf))
                        if p is not None and tf:
                            p *= len(rr)
            elif wide_c:
                cs = [_num(attrs.get(k)) for k in wide_c]
                ps = [_num(attrs.get(k)) for k in wide_p] if wide_p else [None] * len(wide_c)
                for k, cv, pv in zip(wide_c, cs, ps):
                    periods[k][fid] = {"__c": cv, "__p": pv}
                c = sum(x for x in cs if x is not None) if any(x is not None for x in cs) else None
                p = sum(x for x in ps if x is not None) if wide_p and all(x is not None for x in ps) else None
            else:
                c, p = _num(attrs.get(cf)), (_num(attrs.get(pf)) if pf else None)
            if age_cols:
                src = (by.get(fid) or []) if rows else [attrs]
                cg, pg = [], []
                for cc, pp in age_cols:
                    cv = [_num(r.get(cc)) for r in src]
                    pv = [_num(r.get(pp)) for r in src]
                    cg.append(sum(x for x in cv if x is not None) if cv and all(x is not None for x in cv) else None)
                    pg.append((sum(x for x in pv if x is not None)) if pv and all(x is not None for x in pv) else None)
                ok = all(x is not None for x in cg + pg)
                AC.append(cg if ok else None)
                AP.append(pg if ok else None)
                if ok:
                    c, p = float(sum(cg)), float(sum(pg))
            ids.append(fid)
            gl.append(g_l)
            gw.append(g_w)
            C.append(c)
            P.append(p)
        if all(c is None for c in C):
            raise KitError(f"No values found in the cases column '{cf}'.")
        if wide_c:
            cf, pf = "__c", ("__p" if wide_p else "")
        return aoi, dict(ids=ids, gl=gl, gw=gw, C=C, P=P, unmatched=unmatched, periods=periods, key=key, cf=cf, pf=pf,
                         groups=groups, AC=AC, AP=AP, age_cols=age_cols)

    # ================================================================== run
    def run(self, aoi, params, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        from ..core.data import trace_reset
        trace_reset()
        self._aoi, self._params = None, params
        per = float(params.get("per") or 1e5)
        self.step(4, "Reading your areas and case counts")
        aoi, inp = self._read(params, aoi)
        self._aoi = aoi
        R = dict(params=params, per=per, aoi=aoi, I=inp, disease=params.get("disease") or "Cases")
        n = len(inp["ids"])
        span = (aoi.bbox[2] - aoi.bbox[0]) * (aoi.bbox[3] - aoi.bbox[1]) * 111 * 111 * math.cos(math.radians((aoi.bbox[1] + aoi.bbox[3]) / 2))
        res = max(100.0, math.ceil(math.sqrt(span * 1e6 / 2.5e6) / 10) * 10)
        g = Grid(aoi, res, max_pixels=3_000_000)
        R["g"] = g
        zone = g.rasterize(inp["gl"], values=list(range(1, n + 1)), all_touched=False)
        # tiny areas that fall between pixel centres: mark their centroid pixel
        for i, geom in enumerate(inp["gl"]):
            if not (zone == i + 1).any():
                c = geom.Centroid()
                r_, c_ = int((g.gt[3] - c.GetY()) / g.res), int((c.GetX() - g.gt[0]) / g.res)
                if 0 <= r_ < g.rows and 0 <= c_ < g.cols:
                    zone[r_, c_] = i + 1
        R["zone"] = zone

        # ------------------------------------------------------------ population (yours or Meta)
        need_pop = all(p is None for p in inp["P"])
        pop_grid = None
        if need_pop or params.get("environment", True) not in (False, "false", "0", 0):
            self.step(8, "Reading Meta 30 m population")
            from .urban import city as CT
            try:
                pop_grid = CT.population_hrsl(g)
            except Exception:
                pop_grid = None
        R["pop_grid"] = pop_grid
        est = None
        if pop_grid is not None:
            est = np.bincount(zone.ravel(), weights=np.nan_to_num(pop_grid).ravel(), minlength=n + 1)[1:]
            R["pop_est"] = est
        if need_pop:
            if est is None:
                raise KitError("No population column and Meta population could not be read; add a population column.")
            inp["P"] = [float(x) for x in est]
            R["pop_from_meta"] = True
        if inp["groups"]:   # age standardisation needs every group for an area; areas missing any are left out
            R["age_missing"] = sum(1 for a in inp["AC"] if a is None)
            inp["C"] = [c if a is not None else None for c, a in zip(inp["C"], inp["AC"])]
        use = np.array([c is not None and p is not None and p > 0 for c, p in zip(inp["C"], inp["P"])])
        if use.sum() < 8:
            raise KitError("Fewer than 8 areas have both cases and population.")
        Cn = np.array([c if c is not None else 0 for c in inp["C"]], float)
        Pn = np.array([p if p is not None and p > 0 else np.nan for p in inp["P"]], float)
        R.update(use=use, C=Cn, P=Pn)
        u = np.flatnonzero(use)
        c, p = Cn[u], Pn[u]

        # ------------------------------------------------------------ rates
        self.step(14, "Rates, 95% intervals, expected cases, SMR, excess-risk probability")
        raw = c / p * per
        lo, hi = ST.byar_ci(c, p, per)
        base0 = c.sum() / p.sum()
        pa = p                      # population used for expected counts (age-adjusted below if age groups are given)
        if inp["groups"]:
            self.step(16, "Age standardisation: direct (WHO standard) and indirect")
            from .health import advanced as AD
            cg = np.array([inp["AC"][k] for k in u], float)
            pg = np.array([inp["AP"][k] for k in u], float)
            w, covered = AD.who_weights(inp["groups"])
            asr, alo, ahi = AD.direct_asr(cg, pg, w, per)
            Eind, rs = AD.indirect_expected(cg, pg)
            pa = Eind / base0        # 'age-adjusted population': expected = overall rate x pa
            sr, slo_, shi_ = AD.direct_asr(cg.sum(0, keepdims=True), pg.sum(0, keepdims=True), w, per)
            smr_i = c / np.maximum(Eind, 1e-12)
            slo, shi = ST.byar_ci(c, Eind, 1.0)
            R["age"] = dict(groups=inp["groups"], w=w, covered=covered, asr=asr, lo=alo, hi=ahi, E=Eind, rs=rs, smr=smr_i,
                            smr_lo=slo, smr_hi=shi, overall=(float(sr[0]), float(slo_[0]), float(shi_[0])),
                            crude_by_group=cg.sum(0), pop_by_group=pg.sum(0),
                            r_crude_vs_asr=float(np.corrcoef(raw, asr)[0, 1]) if len(u) > 2 else None)
        e, smr, pex = ST.smr(c, pa)
        ebg = ST.eb_global(c, pa) * per
        R.update(raw=raw, lo=lo, hi=hi, exp=e, smr=smr, pex=pex, ebg=ebg, base=base0 * per, pa=pa)

        # ------------------------------------------------------------ neighbours
        self.step(20, "Finding neighbouring areas")
        gl = [inp["gl"][k] for k in u]
        env_ = [x.GetEnvelope() for x in gl]
        nb = [set() for _ in u]
        tol = max(1.0, g.res * 0.01)
        for i in range(len(u)):
            gi = gl[i].Buffer(tol)
            a = env_[i]
            for j in range(i + 1, len(u)):
                b = env_[j]
                if a[0] > b[1] + tol or b[0] > a[1] + tol or a[2] > b[3] + tol or b[2] > a[3] + tol:
                    continue
                if gi.Intersects(gl[j]):
                    nb[i].add(j)
                    nb[j].add(i)
        xy = np.array([(x.Centroid().GetX(), x.Centroid().GetY()) for x in gl])
        islands = [i for i in range(len(u)) if not nb[i]]
        for i in islands:   # islands get their two nearest areas so every area can be tested
            d = ((xy - xy[i]) ** 2).sum(1)
            d[i] = np.inf
            for j in np.argsort(d)[:2]:
                nb[i].add(int(j))
                nb[int(j)].add(i)
        nb = [sorted(v) for v in nb]
        R.update(nb=nb, xy=xy, islands=len(islands))
        ebl = ST.eb_local(c, pa, nb) * per
        R["ebl"] = ebl

        # ------------------------------------------------------------ clustering
        self.step(26, "Moran's I, local Moran (LISA) and Gi* hot spots")
        mi, mp = ST.morans_i(ebg, nb)
        Ii, lp, q = ST.lisa(ebg, nb)
        lsig = ST.fdr(lp, 0.05) | (lp <= 0.05)
        lcls = np.where(lp <= 0.05, q, 5)
        z = ST.gi_star(ebg, nb)
        from statistics import NormalDist
        gp = np.array([2 * (1 - NormalDist().cdf(abs(v))) for v in z])
        hcls = np.full(len(u), 4, np.uint8)
        for i in range(len(u)):
            if gp[i] <= 0.01:
                hcls[i] = 1 if z[i] > 0 else 7
            elif gp[i] <= 0.05:
                hcls[i] = 2 if z[i] > 0 else 6
            elif gp[i] <= 0.10:
                hcls[i] = 3 if z[i] > 0 else 5
        gfdr = ST.fdr(gp, 0.05)
        zr = ST.gi_star(raw, nb)
        hot_eb = hcls <= 2
        hot_raw = (np.array([2 * (1 - NormalDist().cdf(abs(v))) for v in zr]) <= 0.05) & (zr > 0)
        R.update(moran=(mi, mp), lisa=(Ii, lp, lcls), gi=(z, gp, hcls, gfdr),
                 robust=(float((hot_eb & hot_raw).sum() / max(hot_eb.sum(), 1) * 100) if hot_eb.any() else None))

        # ------------------------------------------------------------ scan statistic
        self.step(34, "Kulldorff spatial scan statistic (Monte Carlo)")
        sims = 999 if len(u) <= 600 else 199
        R["scan"] = ST.kulldorff(xy, c, pa, sims=sims)
        R["scan_sims"] = sims
        R["scan_member"] = np.zeros(len(u), int)
        for k, cl in enumerate(R["scan"]):
            if cl["p"] <= 0.05:
                R["scan_member"][cl["members"]] = k + 1

        # ------------------------------------------------------------ planted-cluster test on this map
        self.step(44, "Testing on your own map: can a real cluster be found here?")
        rng = np.random.default_rng(11)
        base = c.sum() / p.sum()
        p = pa
        T, hits, fa = 20, 0, 0
        for t in range(T):
            ctr = rng.integers(len(u))
            d = ((xy - xy[ctr]) ** 2).sum(1)
            mem = np.argsort(d)[:max(3, len(u) // 20)]
            lam = p * base * np.where(np.isin(np.arange(len(u)), mem), 2.0, 1.0)
            sim = rng.poisson(lam).astype(float)
            r1 = ST.kulldorff(xy, sim, p, sims=99, seed=t, max_clusters=1)
            if r1 and r1[0]["p"] <= 0.05 and len(set(mem) & set(r1[0]["members"])) >= 0.5 * len(mem):
                hits += 1
            r0 = ST.kulldorff(xy, rng.poisson(p * base).astype(float), p, sims=99, seed=100 + t, max_clusters=1)
            fa += bool(r0 and r0[0]["p"] <= 0.05)
        R["power"] = (hits, fa, T, max(3, len(u) // 20))
        p = R["P"][u]

        # ------------------------------------------------------------ Bayesian disease mapping
        self.step(48, "Bayesian disease mapping (BYM2, MCMC)")
        try:
            from .health import advanced as AD
            bm = AD.bym2(c, pa * base, nb)
            bm["rate"] = bm["rr"] * base * per
            R["bym"] = bm
        except Exception as ex:
            R["bym_err"] = str(ex)

        # ------------------------------------------------------------ environment
        if str(params.get("environment", True)).strip().lower() not in ("false", "0", "no", "off"):
            try:
                self._environment(R, aoi, g, zone, n, params)
            except Exception as ex:
                R["env_err"] = str(ex)

        # ------------------------------------------------------------ case points: density, risk surface, Knox test
        if params.get("case_points"):
            self.step(75, "Case points: density, risk surface and Knox test")
            try:
                self._points(R, aoi, g, params)
            except Exception as ex:
                R["pts_err"] = str(ex)

        # ------------------------------------------------------------ Bayesian model with the environment as covariates
        if R.get("drv_X") is not None and R.get("bym") is not None and R["drv_X"].shape[1] >= 1:
            self.step(74, "Bayesian model with environmental covariates (BYM2)")
            try:
                bc = AD.bym2(c, pa * base, nb, X=R["drv_X"])
                bc["names"] = R["drv_names"]
                bc["sigma_drop"] = 1 - bc["sigma"] / max(R["bym"]["sigma"], 1e-9)
                R["bym_cov"] = bc
            except Exception as ex:
                R["bym_cov_err"] = str(ex)

        # ------------------------------------------------------------ periods
        if inp["periods"]:
            self.step(78, "Time trends and emerging hot spots")
            try:
                self._trends(R, params)
            except Exception as ex:
                R["trend_err"] = str(ex)

        self.step(88, "Writing results")
        files, layers = self._write(R, out_dir)
        self.step(93, "Writing the report")
        from .health.report_health import build
        return build(self, R, aoi, out_dir, files, layers)

    # ================================================================== environment
    def _environment(self, R, aoi, g, zone, n, params):
        inp, u = R["I"], np.flatnonzero(R["use"])
        Z = zone.ravel()
        cnt = np.bincount(Z, minlength=n + 1)[1:].astype(float)

        def zmean(a):
            a = a.ravel()
            ok = np.isfinite(a)
            s = np.bincount(Z[ok], weights=a[ok], minlength=n + 1)[1:]
            k = np.bincount(Z[ok], minlength=n + 1)[1:]
            with np.errstate(all="ignore"):
                return np.where(k > 0, s / k, np.nan)
        E = {}
        self.step(52, "Environment of each area: people and built-up")
        area_km2 = np.array([x.GetArea() / 1e6 for x in inp["gl"]])
        if R.get("pop_est") is not None:
            E["pop_density"] = R["pop_est"] / np.maximum(area_km2, 1e-6)
        try:
            wc = D.worldcover(g)
            if wc is not None:
                E["built"] = zmean(np.where(g.mask, (wc == 7).astype(np.float32) * 100, np.nan))
                E["water"] = zmean(np.where(g.mask, (wc == 1).astype(np.float32) * 100, np.nan))
        except Exception as exc:
            skipped(exc)
        self.step(56, "Environment: elevation")
        try:
            E["elev"] = zmean(D.dem(g))
        except Exception as exc:
            skipped(exc)
        last = dt.date.today().year - 1
        self.step(60, "Environment: greenness (MODIS NDVI)")
        try:
            g250 = Grid(aoi, max(250.0, g.res), max_pixels=3_000_000)
            zone250 = g250.rasterize(inp["gl"], values=list(range(1, n + 1)))
            its = search("modis-13Q1-061", aoi.bbox, f"{last}-01-01", f"{last}-12-31", max_items=200)
            by = {}
            for it in its:
                by.setdefault((it["properties"].get("start_datetime") or it["properties"]["datetime"])[:10], []).append(it)
            keys = sorted(by)[::max(1, len(by) // 6)][:6]
            arrs = []
            for k in keys:
                v = warp_items(by[k], "250m_16_days_NDVI", g250, resample="average", dtype=gdal.GDT_Float32, src_nodata=-3000, dst_nodata=np.nan) * 1e-4
                arrs.append(v)
            if arrs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    nd = np.nanmean(np.stack(arrs), 0)
                Z2 = zone250.ravel()
                ok = np.isfinite(nd.ravel())
                s = np.bincount(Z2[ok], weights=nd.ravel()[ok], minlength=n + 1)[1:]
                k = np.bincount(Z2[ok], minlength=n + 1)[1:]
                with np.errstate(all="ignore"):
                    E["ndvi"] = np.where(k > 0, s / k, np.nan)
        except Exception as exc:
            skipped(exc)
        self.step(66, "Environment: day surface temperature (MODIS)")
        try:
            from .urban import heat as HT
            gk = Grid(aoi, max(1000.0, g.res), max_pixels=3_000_000)
            zk = gk.rasterize(inp["gl"], values=list(range(1, n + 1)))
            ms = HT.modis_season(gk, last, [1, 4, 7, 10], platforms=("terra",))
            if "Day" in ms:
                a = ms["Day"].ravel()
                Zk = zk.ravel()
                ok = np.isfinite(a)
                s = np.bincount(Zk[ok], weights=a[ok], minlength=n + 1)[1:]
                k = np.bincount(Zk[ok], minlength=n + 1)[1:]
                with np.errstate(all="ignore"):
                    lst = np.where(k > 0, s / k, np.nan)
                # small areas missing a 1 km pixel: nearest-centroid fill
                if np.isnan(lst).any() and np.isfinite(lst).any():
                    cx = np.array([(x.Centroid().GetX(), x.Centroid().GetY()) for x in inp["gl"]])
                    have = np.flatnonzero(np.isfinite(lst))
                    for i in np.flatnonzero(np.isnan(lst)):
                        j = have[np.argmin(((cx[have] - cx[i]) ** 2).sum(1))]
                        lst[i] = lst[j]
                E["lst"] = lst
        except Exception as exc:
            skipped(exc)
        # access
        self.step(70, "Access to health facilities")
        fac, src = [], None
        fac_caps = []
        try:
            if params.get("facilities"):
                capf = (params.get("capacity_field") or "").strip()
                for _, attrs, gw in read_vector(params["facilities"], aoi):
                    cc = gw.Centroid()
                    fac.append((cc.GetX(), cc.GetY(), str(attrs.get("name") or "")))
                    fac_caps.append(_num(attrs.get(capf)) if capf else None)
                src = "your layer"
            else:
                fac = [(lo, la, nm) for lo, la, nm, _ in osm.facilities(aoi.bbox, "health")]
                src = "OpenStreetMap"
        except Exception as ex:
            R["fac_err"] = str(ex)
        if fac:
            seed = np.zeros(g.shape, bool)
            for lon, lat, _ in fac:
                r_, c_ = g.lonlat_to_pixel(lon, lat)
                if 0 <= r_ < g.rows and 0 <= c_ < g.cols:
                    seed[r_, c_] = True
            if seed.any():
                d = g.distance_to(seed) / 1000.0
                R["dist_how"] = "straight line"
                if aoi.area_km2 <= 5000:      # roads for larger areas would be a very large OpenStreetMap download
                    try:
                        from .urban import network as NW
                        R["roads_m"] = NW.road_mask(g, NW.fetch_roads(aoi.bbox))
                        rr_, cc_ = np.nonzero(seed)
                        dn = NW.network_distance_km(g, R["roads_m"], list(zip(rr_.tolist(), cc_.tolist())))
                        if dn is not None:
                            R["detour"] = NW.detour_ratio(dn, d, g.mask)
                            d = np.where(np.isfinite(dn), dn, d)
                            R["dist_how"] = "along roads"
                    except Exception as ex_:
                        R["roads_err"] = str(ex_)
                if R.get("pop_grid") is not None:
                    w = np.nan_to_num(R["pop_grid"]).ravel()
                    s = np.bincount(Z, weights=(d.ravel() * w), minlength=n + 1)[1:]
                    k = np.bincount(Z, weights=w, minlength=n + 1)[1:]
                    with np.errstate(all="ignore"):
                        E["dist_fac"] = np.where(k > 0, s / k, zmean(d))
                    lim = float(params.get("access_km") or 5)
                    far = (d > lim).ravel()
                    R["pop_far"] = float(w[far & (Z > 0)].sum())
                    R["pop_far_pct"] = R["pop_far"] / max(w[Z > 0].sum(), 1) * 100
                else:
                    E["dist_fac"] = zmean(d)
            R["fac"], R["fac_src"] = fac, src
            try:
                self._two_sfca(R, g, zone, n, params, fac_caps)
            except Exception as ex:
                R["sfca_err"] = str(ex)
        R["env"] = E
        # drivers: Spearman, Geodetector, Poisson regression (standardised, collinear factors dropped)
        y = R["ebg"]
        names, cols = [], []
        res = {}
        for k, lab in ENV:
            if k not in E:
                continue
            x = E[k][u]
            if np.isfinite(x).sum() < 0.8 * len(u) or np.nanstd(x) == 0:
                continue
            res[k] = dict(rho=ST.spearman(x, y), q=ST.geodetector_q(y, x))
            names.append(k)
            cols.append(x)
        R["drivers"] = res
        if len(names) >= 1:
            X = np.column_stack(cols)
            X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
            if "pop_density" in names:
                X[:, names.index("pop_density")] = np.log1p(np.maximum(X[:, names.index("pop_density")], 0))
            Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
            keep = []
            for j in range(Xs.shape[1]):
                if all(abs(np.corrcoef(Xs[:, j], Xs[:, k2])[0, 1]) < 0.8 for k2 in keep):
                    keep.append(j)
            R["dropped"] = [names[j] for j in range(len(names)) if j not in keep]
            R["drv_X"], R["drv_names"] = Xs[:, keep], [names[j] for j in keep]
            fit = ST.poisson_glm(R["C"][u], Xs[:, keep], np.log(R["pa"]), [names[j] for j in keep])
            R["glm"] = fit
            resid = (R["C"][u] - fit["fitted"]) / np.sqrt(np.maximum(fit["fitted"], 1e-9))
            R["resid_moran"] = ST.morans_i(resid, R["nb"], perms=499)
            if len(u) >= 30:
                self.step(72, "Geographically weighted Poisson regression (local drivers)")
                try:
                    from .health import extra as HX
                    gw = HX.gwpr(R["C"][u], Xs[:, keep], np.log(R["pa"]), R["xy"])
                    if gw:
                        gw["names"] = [names[j] for j in keep]
                        R["gwpr"] = gw
                except Exception as ex:
                    R["gwpr_err"] = str(ex)

    # ================================================================== 2SFCA
    def _two_sfca(self, R, g, zone, n, params, caps):
        """Two-step floating catchment area index on population cells, averaged per area (population-weighted)."""
        from .health import extra as HX
        fac = R["fac"]
        cap = np.array([c if (c is not None and c > 0) else 1.0 for c in (caps or [None] * len(fac))], float)
        used_cap = bool(caps) and any(c is not None and c > 0 for c in caps)
        sx, sy = [], []
        for lon, lat, _ in fac:
            r_, c_ = g.lonlat_to_pixel(lon, lat)
            sx.append(g.gt[0] + (c_ + 0.5) * g.res)
            sy.append(g.gt[3] - (r_ + 0.5) * g.res)
        sup = np.column_stack([sx, sy])
        pop = R.get("pop_grid")
        Z = zone
        if pop is not None:
            k = max(1, int(math.ceil(math.sqrt(g.rows * g.cols / 40000.0))))   # demand cells: at most ~40k
            rr, cc = (g.rows // k) * k, (g.cols // k) * k
            P = np.nan_to_num(pop)[:rr, :cc].reshape(rr // k, k, cc // k, k).sum((1, 3))
            iy, ix = np.nonzero(P > 0)
            dem = np.column_stack([g.gt[0] + (ix * k + k / 2) * g.res, g.gt[3] - (iy * k + k / 2) * g.res])
            Dm = None
            if R.get("roads_m") is not None and len(fac) <= 80:
                from .urban import network as NW
                cols_ = []
                for lon, lat, _ in fac:
                    r_, c_ = g.lonlat_to_pixel(lon, lat)
                    dn = NW.network_distance_km(g, R["roads_m"], [(r_, c_)])
                    cols_.append(np.full(iy.size, np.inf) if dn is None else dn[iy * k + k // 2, ix * k + k // 2] * 1000.0)
                Dm = np.column_stack(cols_)
                R["sfca_how"] = "along roads"
            A, Rj = HX.two_step_fca(dem, P[iy, ix], sup, cap, float(params.get("access_km") or 5) * 1000.0, D=Dm)
            zc = Z[:rr, :cc][iy * k + k // 2, ix * k + k // 2]
            w = P[iy, ix]
            s = np.bincount(zc, weights=A * w, minlength=n + 1)[1:]
            q = np.bincount(zc, weights=w, minlength=n + 1)[1:]
            with np.errstate(all="ignore"):
                acc = np.where(q > 0, s / q, np.nan)
        else:
            inp = R["I"]
            dem = np.array([(x.Centroid().GetX(), x.Centroid().GetY()) for x in inp["gl"]])
            Pa = np.nan_to_num(R["P"])
            acc, Rj = HX.two_step_fca(dem, Pa, sup, cap, float(params.get("access_km") or 5) * 1000.0)
        R["sfca"] = dict(acc=acc * 10000.0, Rj=Rj, used_cap=used_cap, capf=(params.get("capacity_field") or "").strip(),
                         d0=float(params.get("access_km") or 5))

    # ================================================================== case points
    def _points(self, R, aoi, g, params):
        from .health import extra as HX
        pts = read_vector(params["case_points"], aoi)
        if len(pts) < 10:
            raise KitError("Fewer than 10 case points.")
        xy = np.array([(p.Centroid().GetX(), p.Centroid().GetY()) for p, _, _ in pts])
        df = (params.get("case_date_field") or "").strip()
        dates = []
        if df:
            for _, a, _ in pts:
                v = a.get(df)
                try:   # GDAL gives dates as YYYY/MM/DD; text fields may use YYYY-MM-DD
                    dates.append(dt.date.fromisoformat(str(v)[:10].replace("/", "-")).toordinal())
                except Exception:
                    dates.append(None)
        # density grid: the study grid, coarsened to <= 250k cells
        k = max(1, int(math.ceil(math.sqrt(g.rows * g.cols / 250000.0))))
        rr, cc = (g.rows // k) * k, (g.cols // k) * k
        res = g.res * k
        mask = g.mask[:rr, :cc].reshape(rr // k, k, cc // k, k).mean((1, 3)) > 0.5
        RC = np.zeros(mask.shape)
        for x, y in xy:
            c_ = int((x - g.gt[0]) / res)
            r_ = int((g.gt[3] - y) / res)
            if 0 <= r_ < RC.shape[0] and 0 <= c_ < RC.shape[1]:
                RC[r_, c_] += 1
        pop = R.get("pop_grid")
        P = (np.nan_to_num(pop)[:rr, :cc].reshape(rr // k, k, cc // k, k).sum((1, 3)) if pop is not None else None)
        bw_scott = max(HX.bandwidth_scott(xy), 2 * res)
        bw, bw_curve = HX.bandwidth_cv(RC, mask, res, bw_scott)
        out = dict(n=len(xy), inside=int(RC[mask].sum()), bw_m=bw, bw_scott=bw_scott, res=res, k=k)
        if P is not None and P[mask].sum() > 0:
            kr = HX.kde_risk(RC, P, mask, res, bw, sims=99)
            out.update(kr)
        else:
            s = bw / res
            out["density"] = np.where(mask, HX._gauss_blur(np.where(mask, RC, 0.0), s) / (res * res / 1e6), np.nan)
        import types
        kg = types.SimpleNamespace(aoi=g.aoi, res=res, rows=mask.shape[0], cols=mask.shape[1], shape=mask.shape, mask=mask,
                                   gt=(g.gt[0], res, 0, g.gt[3], 0, -res), wkt=g.wkt,
                                   bounds=(g.gt[0], g.gt[3] - mask.shape[0] * res, g.gt[0] + mask.shape[1] * res, g.gt[3]),
                                   px_km2=res * res / 1e6)
        out["grid"] = kg
        good = [i for i, d in enumerate(dates) if d is not None]
        if df and len(good) >= 20:
            t = np.array([dates[i] for i in good], float)
            out["knox"] = HX.knox_points(xy[good] / 1000.0, t, float(params.get("knox_km") or 2.0),
                                         float(params.get("knox_days") or 14), sims=999)
            out["dated"] = len(good)
        R["pts"] = out

    # ================================================================== trends
    def _trends(self, R, params):
        inp = R["I"]
        pf, cf = inp["pf"], inp["cf"]
        per = R["per"]
        keys = list(inp["periods"])
        if all(_num(k) is not None for k in keys):
            keys.sort(key=_num)          # 1, 2, 10 in numeric order
        elif not (inp["cf"] == "__c"):     # wide columns keep the order given; others alphabetic
            keys.sort()
        idx = {k: i for i, k in enumerate(inp["ids"])}
        tot = []
        mat = np.full((len(keys), len(inp["ids"])), np.nan)
        cmat = np.zeros((len(inp["ids"]), len(keys)))
        pmat = np.zeros((len(inp["ids"]), len(keys)))
        G = len(inp["age_cols"] or [])
        acg = np.zeros((len(inp["ids"]), len(keys), G))
        apg = np.zeros((len(inp["ids"]), len(keys), G))
        for t, k in enumerate(keys):
            cs, ps = 0.0, 0.0
            for aid, r in inp["periods"][k].items():
                if aid not in idx:
                    continue
                cv = _num(r.get(cf))
                pv = _num(r.get(pf)) if pf else None
                if inp.get("age_cols"):
                    a_c = [_num(r.get(cc)) for cc, _ in inp["age_cols"]]
                    a_p = [_num(r.get(pp)) for _, pp in inp["age_cols"]]
                    if all(x is not None for x in a_c):
                        cv = sum(a_c)
                    if all(x is not None for x in a_p):
                        pv = sum(a_p)
                    if all(x is not None for x in a_c + a_p):
                        acg[idx[aid], t] = a_c
                        apg[idx[aid], t] = a_p
                if pv is None and R.get("pop_from_meta"):
                    pv = R["P"][idx[aid]] / max(len(keys), 1)
                if cv is None or not pv:
                    continue
                mat[t, idx[aid]] = cv / pv * per
                cmat[idx[aid], t] = cv
                pmat[idx[aid], t] = pv
                cs += cv
                ps += pv
            tot.append(cs / ps * per if ps else None)
        R["trend_keys"], R["trend_total"], R["trend_mat"] = keys, tot, mat
        u0 = np.flatnonzero(R["use"])
        if len(keys) >= 3:
            self.step(82, "Space-time scan statistic (Kulldorff, Monte Carlo)")
            from .health import advanced as AD
            cu, pu = cmat[u0], pmat[u0]
            if G:   # age-adjusted 'population' per cell: expected from that period's age-specific rates
                cg_, pg_ = acg[u0], apg[u0]
                rs_t = cg_.sum(0) / np.maximum(pg_.sum(0), 1e-12)          # (T, G)
                Et = (pg_ * rs_t[None]).sum(2)                                # (n, T)
                rt = cu.sum(0) / np.maximum(pu.sum(0), 1e-12)
                pu = np.where(rt[None] > 0, Et / np.maximum(rt[None], 1e-12), pu)
                R["st_age_adjusted"] = True
            okp = pu.sum(1) > 0
            sims = 999 if len(u0) <= 300 else 199 if len(u0) <= 1500 else 99
            st = AD.st_scan(R["xy"], cu, pu, sims=sims, max_k=None if len(u0) <= 800 else 300)
            R["st_scan"], R["st_sims"] = st, sims
            m = np.zeros(len(u0), int)
            for k2, cl in enumerate(st):
                if cl["p"] <= 0.05:
                    m[cl["members"]] = k2 + 1
            R["st_member"] = m
            R["st_note"] = None if okp.all() else f"{int((~okp).sum())} areas have no period data and count as empty."
            # case-only check: space-time permutation scan (no population needed, robust to wrong denominators)
            try:
                self.step(85, "Space-time permutation scan (cases only)")
                stp = AD.st_scan(R["xy"], cu, None, sims=sims, max_k=None if len(u0) <= 800 else 300, model="permutation")
                R["stp_scan"] = stp
            except Exception as ex:
                R["stp_err"] = str(ex)
            try:
                self.step(85.5, "Early warning and forecast for the next period")
                from .health import forecast as FC
                R["ew"] = FC.early_warning(cu, pu)
                R["fc"] = FC.forecast_next(cu, pu)
            except Exception as ex:
                R["fc_err"] = str(ex)
            try:
                self.step(86, "Knox test: do cases cluster in space and time together?")
                from .health import extra as HX
                R["knox_areal"] = HX.knox_areal(cu, R["nb"], lag=1, sims=499 if len(u0) <= 1000 else 199)
            except Exception as ex:
                R["knox_areal_err"] = str(ex)
        R["trend_mk"] = HY.mann_kendall([np.nan if v is None else v for v in tot]) if len(keys) >= 6 else None
        u = np.flatnonzero(R["use"])
        up = 0
        rising = np.zeros(len(inp["ids"]), bool)
        if len(keys) >= 6:
            for j in u:
                mk = HY.mann_kendall(mat[:, j])
                if mk and mk["p"] < 0.05 and mk["slope"] > 0:
                    rising[j] = True
                    up += 1
        R["rising"], R["n_rising"] = rising, up
        # emerging hot spots: Gi* hot (95%) in the last period but not in the first half of the periods
        if len(keys) >= 3:
            from statistics import NormalDist

            def hot_at(rows):
                v = np.nanmean(mat[rows][:, u], 0)
                v = np.where(np.isfinite(v), v, np.nanmean(v))
                zz = ST.gi_star(v, R["nb"])
                return zz >= NormalDist().inv_cdf(0.975)
            last = hot_at([len(keys) - 1])
            early = hot_at(list(range(len(keys) // 2)))
            R["emerging"] = last & ~early
            R["persistent"] = last & early

    # ================================================================== outputs
    def _write(self, R, out_dir):
        inp, u = R["I"], np.flatnonzero(R["use"])
        pos = {k: i for i, k in enumerate(u)}
        z, gp, hcls, gfdr = R["gi"]
        Ii, lp, lcls = R["lisa"]
        fc = {"type": "FeatureCollection", "features": []}
        E = R.get("env") or {}
        kmin = int(R["params"].get("suppress_below") or 0)
        RAW = ("cases", "rate", "rate_lo95", "rate_hi95", "smr", "p_excess", "asr", "asr_lo95", "asr_hi95", "smr_age_adj", "smr_lo95",
               "smr_hi95", "latest_observed")
        for k in range(len(inp["ids"])):
            i = pos.get(k)
            pr = {"area_id": inp["ids"][k], "cases": inp["C"][k], "population": None if np.isnan(R["P"][k]) else round(float(R["P"][k]))}
            if i is not None:
                pr.update(rate=round(float(R["raw"][i]), 3), rate_lo95=round(float(R["lo"][i]), 3), rate_hi95=round(float(R["hi"][i]), 3),
                          expected=round(float(R["exp"][i]), 2), smr=round(float(R["smr"][i]), 3), p_excess=round(float(R["pex"][i]), 4),
                          eb_global=round(float(R["ebg"][i]), 3), eb_local=round(float(R["ebl"][i]), 3),
                          gi_z=round(float(z[i]), 3), gi_p=round(float(gp[i]), 4), hotspot=HOT[int(hcls[i])][0], gi_fdr=bool(gfdr[i]),
                          lisa=LISA[int(lcls[i])][0], lisa_p=round(float(lp[i]), 4), scan_cluster=int(R["scan_member"][i]))
                A = R.get("age")
                if A:
                    pr.update(asr=round(float(A["asr"][i]), 3), asr_lo95=round(float(A["lo"][i]), 3), asr_hi95=round(float(A["hi"][i]), 3),
                              expected_age_adj=round(float(A["E"][i]), 3), smr_age_adj=round(float(A["smr"][i]), 3),
                              smr_lo95=round(float(A["smr_lo"][i]), 3), smr_hi95=round(float(A["smr_hi"][i]), 3))
                B = R.get("bym")
                if B:
                    pr.update(bym_rr=round(float(B["rr"][i]), 4), bym_rr_lo95=round(float(B["lo"][i]), 4),
                              bym_rr_hi95=round(float(B["hi"][i]), 4), bym_p_rr_gt1=round(float(B["p_gt1"][i]), 4),
                              bym_rate=round(float(B["rate"][i]), 3))
                if R.get("st_member") is not None:
                    pr["spacetime_cluster"] = int(R["st_member"][i])
                if R.get("fc"):
                    pr.update(forecast_next=round(float(R["fc"]["mean"][i]), 1), forecast_lo95=round(float(R["fc"]["lo"][i]), 1),
                              forecast_hi95=round(float(R["fc"]["hi"][i]), 1))
                if R.get("ew"):
                    pr.update(latest_expected=round(float(R["ew"]["expected"][i]), 1), latest_alert=int(R["ew"]["alert"][i]))
                GW = R.get("gwpr")
                if GW:
                    for j, nm in enumerate(GW["names"]):
                        pr[f"gwpr_rr_{nm}"] = round(float(np.exp(GW["beta"][i, j + 1])), 4)
                        pr[f"gwpr_t_{nm}"] = round(float(GW["beta"][i, j + 1] / max(GW["se"][i, j + 1], 1e-12)), 3)
            for key, _ in ENV:
                if key in E and np.isfinite(E[key][k]):
                    pr[key] = round(float(E[key][k]), 3)
            if R.get("sfca") is not None and np.isfinite(R["sfca"]["acc"][k]):
                pr["access_2sfca_per10k"] = round(float(R["sfca"]["acc"][k]), 4)
            if R.get("emerging") is not None and i is not None:
                pr["emerging_hotspot"] = bool(R["emerging"][i])
                pr["rising_trend"] = bool(R["rising"][k])
            cval = inp["C"][k]
            if kmin > 0 and cval is not None and np.isfinite(cval) and 0 < cval < kmin:
                for kk in RAW:
                    if kk in pr:
                        pr[kk] = None
                pr["suppressed"] = f"fewer than {kmin} cases"
            fc["features"].append({"type": "Feature", "geometry": json.loads(inp["gw"][k].ExportToJson()), "properties": pr})
        out = os.path.join(out_dir, "health_study.geojson")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(fc, fh)
        cats = "".join(f'<category render="true" value="{nm}" symbol="{k - 1}" label="{nm}"/>' for k, (nm, _) in HOT.items())
        syms = "".join(f'<symbol type="fill" name="{k - 1}" alpha="1"><layer class="SimpleFill"><Option type="Map">'
                       f'<Option type="QString" name="color" value="{int(col[1:3], 16)},{int(col[3:5], 16)},{int(col[5:7], 16)},255"/>'
                       f'<Option type="QString" name="outline_color" value="80,80,80,255"/><Option type="QString" name="outline_width" value="0.2"/>'
                       f'</Option></layer></symbol>' for k, (_, col) in HOT.items())
        with open(os.path.splitext(out)[0] + ".qml", "w", encoding="utf-8") as fh:
            fh.write(f"<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'><qgis version=\"3.22.0\" styleCategories=\"Symbology\">"
                     f'<renderer-v2 type="categorizedSymbol" attr="hotspot" enableorderby="0" forceraster="0" symbollevels="0">'
                     f"<categories>{cats}</categories><symbols>{syms}</symbols></renderer-v2></qgis>")
        csvp = os.path.join(out_dir, "health_study_table.csv")
        keys = []
        for f in fc["features"]:
            for kk in f["properties"]:
                if kk not in keys:
                    keys.append(kk)
        with open(csvp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            for f in fc["features"]:
                w.writerow(f["properties"])
        files = [out, csvp]
        if R.get("scan"):
            p = os.path.join(out_dir, "scan_clusters.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("cluster,areas,observed,expected,relative_risk,llr,p_value,population,area_ids\n")
                for k, cl in enumerate(R["scan"]):
                    fh.write(f"{k + 1},{len(cl['members'])},{cl['obs']:.0f},{cl['exp']:.2f},{cl['rr']:.3f},{cl['llr']:.3f},{cl['p']:.4f},"
                             f"{cl['pop']:.0f},\"{' '.join(inp['ids'][u[m]] for m in cl['members'])}\"\n")
            files.append(p)
        if R.get("fc") or R.get("ew"):
            p = os.path.join(out_dir, "forecast_and_alerts.csv")
            with open(p, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["area_id", "latest_observed", "latest_expected", "latest_upper99", "alert", "next_forecast", "next_lo95", "next_hi95"])
                ew_, fc_ = R.get("ew") or {}, R.get("fc") or {}
                for j, k in enumerate(u):
                    def g_(d, key):
                        return "" if not d else round(float(d[key][j]), 2)
                    ob = g_(ew_, "observed")
                    if kmin > 0 and ob != "" and 0 < ob < kmin:
                        ob = f"<{kmin}"
                    w.writerow([inp["ids"][k], ob, g_(ew_, "expected"), g_(ew_, "upper"),
                                "" if not ew_ else int(ew_["alert"][j]), g_(fc_, "mean"), g_(fc_, "lo"), g_(fc_, "hi")])
            files.append(p)
        if R.get("st_scan"):
            p = os.path.join(out_dir, "spacetime_clusters.csv")
            keys_ = R["trend_keys"]
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("cluster,areas,from_period,to_period,observed,expected,relative_risk,llr,p_value,area_ids\n")
                for k, cl in enumerate(R["st_scan"]):
                    fh.write(f"{k + 1},{len(cl['members'])},{keys_[cl['t0']]},{keys_[cl['t1']]},{cl['obs']:.0f},{cl['exp']:.2f},"
                             f"{cl['rr']:.3f},{cl['llr']:.3f},{cl['p']:.4f},\"{' '.join(inp['ids'][u[m]] for m in cl['members'])}\"\n")
            files.append(p)
        if R.get("stp_scan"):
            p = os.path.join(out_dir, "spacetime_permutation_clusters.csv")
            keys_ = R["trend_keys"]
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("cluster,areas,from_period,to_period,observed,expected,ratio,llr,p_value,area_ids\n")
                for k, cl in enumerate(R["stp_scan"]):
                    fh.write(f"{k + 1},{len(cl['members'])},{keys_[cl['t0']]},{keys_[cl['t1']]},{cl['obs']:.0f},{cl['exp']:.2f},"
                             f"{cl['rr']:.3f},{cl['llr']:.3f},{cl['p']:.4f},\"{' '.join(inp['ids'][u[m]] for m in cl['members'])}\"\n")
            files.append(p)
        if R.get("bym_cov") and R["bym_cov"].get("beta"):
            bc, bt = R["bym_cov"], R["bym_cov"]["beta"]
            p = os.path.join(out_dir, "bym2_covariates.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("factor,rate_ratio_per_sd,ci_low,ci_high,p_positive,rhat\n")
                for j, nm in enumerate(bc["names"]):
                    fh.write(f"{nm},{bt['irr'][j]:.4f},{bt['irr_lo'][j]:.4f},{bt['irr_hi'][j]:.4f},{bt['p_pos'][j]:.3f},{bt['rhat'][j]:.3f}\n")
            files.append(p)
        if R.get("age"):
            A = R["age"]
            p = os.path.join(out_dir, "age_groups.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"age_group,cases,population,rate_per_{int(R['per'])},who_standard_weight\n")
                for (lab, _, _), cc, pp, ww in zip(A["groups"], A["crude_by_group"], A["pop_by_group"], A["w"]):
                    fh.write(f"{lab},{cc:.0f},{pp:.0f},{cc / pp * R['per'] if pp else float('nan'):.4f},{ww:.6f}\n")
            files.append(p)
        if R.get("glm"):
            p = os.path.join(out_dir, "drivers_poisson_regression.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("factor,rate_ratio_per_sd,ci_low,ci_high,p_value\n")
                for nm, rr, a, b, pv in R["glm"]["rows"]:
                    fh.write(f"{nm},{rr:.4f},{a:.4f},{b:.4f},{pv:.4g}\n")
            files.append(p)
        if R.get("knox_areal") or (R.get("pts") or {}).get("knox"):
            p = os.path.join(out_dir, "knox_test.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("test,observed_pairs,expected_pairs,ratio,p_value\n")
                for nm, K in (("areas_x_periods", R.get("knox_areal")), ("case_points", (R.get("pts") or {}).get("knox"))):
                    if K:
                        fh.write(f"{nm},{K['obs']:.0f},{K['exp']:.2f},{(K['ratio'] or float('nan')):.4f},{K['p']:.4f}\n")
            files.append(p)
        extra_layers = []
        PT = R.get("pts")
        if PT:
            from ..core.raster import write_classes, write_float
            kg = PT["grid"]
            d_ = PT["density"]
            f1, _ = write_float(os.path.join(out_dir, "case_density_per_km2.tif"), d_, kg, "heat", 0,
                                float(np.nanpercentile(d_, 99)) if np.isfinite(d_).any() else 1.0, "cases/km²")
            files.append(f1)
            extra_layers.append(dict(path=f1, name="Case density", kind="raster", hidden=True))
            if PT.get("ratio") is not None:
                f2, _ = write_float(os.path.join(out_dir, "case_relative_risk_surface.tif"), PT["ratio"], kg, "risk", 0.5, 2.0, "ratio")
                f3 = write_classes(os.path.join(out_dir, "case_risk_classes.tif"), PT["cls"], kg,
                                   {1: ("High risk, significant", "#b2182b"), 2: ("Higher than average", "#f4a582"),
                                    3: ("About average", "#f2efe6"), 4: ("Lower than average", "#92c5de")})
                files += [f2, f3]
                extra_layers.append(dict(path=f3, name="Case risk surface", kind="raster"))
        layers = [dict(path=out, name=f"{R['disease']} study", kind="vector")]
        layers += extra_layers
        if R.get("fac"):
            from ..core.raster import write_geojson
            f = write_geojson(os.path.join(out_dir, "health_facilities.geojson"),
                              [((lo, la), {"name": nm, "source": R["fac_src"]}) for lo, la, nm in R["fac"]])
            files.append(f)
            layers.insert(0, dict(path=f, name="Health facilities", kind="vector"))
        return files, layers
