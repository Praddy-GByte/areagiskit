# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""HealthKit: disease rates, small-number smoothing, hot spots and access to care.

Input is the user's own data: a polygon layer of areas (wards, villages, blocks) and the
case counts and population for each, either as fields on the layer or in a CSV.
"""

import csv
import math
import os
import time

import numpy as np
from osgeo import ogr

from ..core import datasets as D
from ..core import osm
from ..core.aoi import AOI, Grid, read_vector
from ..core.data import KitError
from ..core.raster import preview_classes
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

HOT = {1: ("Hot spot, 99%", "#9e1b16"), 2: ("Hot spot, 95%", "#d6402b"), 3: ("Hot spot, 90%", "#f08a6a"),
       4: ("Not significant", "#ece9df"), 5: ("Cold spot, 90%", "#9cc2e0"), 6: ("Cold spot, 95%", "#4a8cc4"),
       7: ("Cold spot, 99%", "#1d5a96")}
RATE5 = ["#fdf0d5", "#f6c58f", "#ea8a50", "#c9502f", "#8b2318"]


def _num(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _p_two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def _hav(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def eb_smooth(c, p):
    """Global empirical Bayes rates (Marshall 1991)."""
    b = c.sum() / p.sum()
    r = c / p
    s2 = (p * (r - b) ** 2).sum() / p.sum()
    a = max(s2 - b / p.mean(), 0.0)
    w = a / (a + b / p) if a > 0 else np.zeros_like(p)
    return b + w * (r - b), b


def gi_star(x, nbrs):
    n = len(x)
    xb, s = x.mean(), x.std()
    z = np.zeros(n)
    for i in range(n):
        J = list(nbrs[i]) + [i]
        w = len(J)
        num = x[J].sum() - xb * w
        den = s * math.sqrt((n * w - w * w) / (n - 1)) if n > 1 else 0
        z[i] = num / den if den > 0 else 0
    return z


def morans_i(x, nbrs, perms=499, seed=1):
    n = len(x)
    z = x - x.mean()
    W = sum(len(v) for v in nbrs)
    if W == 0:
        return None, None

    def morans_i(zz):
        return (n / W) * sum(zz[i] * zz[list(nbrs[i])].sum() for i in range(n) if nbrs[i]) / (zz ** 2).sum()

    obs = morans_i(z)
    rng = np.random.default_rng(seed)
    sims = np.array([morans_i(rng.permutation(z)) for _ in range(perms)])
    p = (np.sum(sims >= obs) + 1) / (perms + 1)
    return float(obs), float(p)


class HealthKit(Kit):
    NAME = "HealthKit"
    KEY = "health"
    TAGLINE = "Disease rates, hot spots and access to care"
    DESCRIPTION = ("From your area polygons and case counts: rates per 100,000, smoothing for small "
                   "populations, statistically tested hot spots (Getis-Ord Gi*), clustering test, and "
                   "distance to the nearest health facility.")
    NEEDS_AOI = False
    MAX_KM2 = 50000
    PARAMS = [
        dict(name="areas", type="polygons", label="Areas (wards, villages, blocks)"),
        dict(name="id_field", type="field", label="Area ID field", parent="areas"),
        dict(name="cases_csv", type="file", label="Cases CSV (optional, else use layer fields)", optional=True),
        dict(name="cases_field", type="text", label="Cases column", default="cases"),
        dict(name="population_field", type="text", label="Population column", default="population"),
        dict(name="disease", type="text", label="Name of disease / indicator", default="Cases"),
        dict(name="facilities", type="points", label="Health facilities (optional; else OpenStreetMap)", optional=True),
        dict(name="access_km", type="float", label="Acceptable distance to a facility (km)", default=5.0),
    ]

    def run(self, aoi, params, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        from ..core.data import trace_reset
        trace_reset()
        self._aoi, self._params = None, params
        path = params.get("areas")
        if not path:
            raise KitError("Choose the polygon layer of areas.")
        idf = params.get("id_field") or ""
        cf, pf = params.get("cases_field") or "cases", params.get("population_field") or "population"
        disease = params.get("disease") or "Cases"
        lim = float(params.get("access_km") or 5)

        self.step(5, "Reading your areas")
        probe = ogr.Open(path)
        if probe is None:
            raise KitError(f"Could not open {path}.")
        x0, x1, y0, y1 = probe.GetLayer(0).GetExtent()
        lyr = probe.GetLayer(0)
        srs = lyr.GetSpatialRef()
        from osgeo import osr
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
            raise KitError("HealthKit needs at least 8 areas to test for hot spots.")
        if idf and idf not in feats[0][1]:
            raise KitError(f"Field '{idf}' not found in the layer.")

        recs = {}
        if params.get("cases_csv"):
            with open(params["cases_csv"], newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                raise KitError("The CSV is empty.")
            key = idf if idf in rows[0] else list(rows[0].keys())[0]
            for r in rows:
                recs[str(r.get(key, "")).strip()] = r

        ids, geoms_l, geoms_w, C, P = [], [], [], [], []
        unmatched = 0
        for i, (gl, attrs, gw) in enumerate(feats):
            fid = str(attrs.get(idf, i) if idf else i).strip()
            src = recs.get(fid) if recs else attrs
            if src is None:
                unmatched += 1
                src = {}
            ids.append(fid)
            geoms_l.append(gl)
            geoms_w.append(gw)
            C.append(_num(src.get(cf)))
            P.append(_num(src.get(pf)))
        if all(c is None for c in C):
            raise KitError(f"No values found in the cases column '{cf}'.")
        if all(p is None for p in P):
            raise KitError(f"No values found in the population column '{pf}'.")
        use = np.array([c is not None and p is not None and p > 0 for c, p in zip(C, P)])
        Cn = np.array([c if c is not None else 0 for c in C], float)
        Pn = np.array([p if p is not None and p > 0 else np.nan for p in P], float)
        n_used = int(use.sum())
        if n_used < 8:
            raise KitError("Fewer than 8 areas have both cases and population.")

        self.step(25, "Calculating rates and smoothing small populations")
        raw = np.full(len(ids), np.nan)
        raw[use] = Cn[use] / Pn[use] * 1e5
        ebv, base = eb_smooth(Cn[use], Pn[use])
        eb = np.full(len(ids), np.nan)
        eb[use] = ebv * 1e5
        base *= 1e5

        self.step(40, "Finding neighbouring areas")
        env = [g.GetEnvelope() for g in geoms_l]
        nbrs_all = [set() for _ in ids]
        tol = 1.0  # metres: shared borders digitised slightly apart still count
        for i in range(len(ids)):
            gi = geoms_l[i].Buffer(tol)
            a = env[i]
            for j in range(i + 1, len(ids)):
                b = env[j]
                if a[0] > b[1] + tol or b[0] > a[1] + tol or a[2] > b[3] + tol or b[2] > a[3] + tol:
                    continue
                if gi.Intersects(geoms_l[j]):
                    nbrs_all[i].add(j)
                    nbrs_all[j].add(i)
        idx_used = np.flatnonzero(use)
        pos = {k: n for n, k in enumerate(idx_used)}
        nbrs = [[pos[j] for j in nbrs_all[k] if j in pos] for k in idx_used]
        islands = sum(1 for v in nbrs if not v)

        self.step(55, "Testing for hot spots (Getis-Ord Gi*)")
        x = eb[use]
        z = gi_star(x, nbrs)
        pv = np.array([_p_two(v) for v in z])
        cls_u = np.full(n_used, 4, np.uint8)
        for i in range(n_used):
            if pv[i] <= 0.01:
                cls_u[i] = 1 if z[i] > 0 else 7
            elif pv[i] <= 0.05:
                cls_u[i] = 2 if z[i] > 0 else 6
            elif pv[i] <= 0.10:
                cls_u[i] = 3 if z[i] > 0 else 5
        # false discovery rate (Benjamini-Hochberg) at 5%
        order = np.argsort(pv)
        thresh = 0.05 * (np.arange(1, n_used + 1) / n_used)
        passed = pv[order] <= thresh
        kmax = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
        fdr_sig = np.zeros(n_used, bool)
        fdr_sig[order[:kmax]] = True
        z_raw = gi_star(raw[use], nbrs)
        hot_eb = cls_u <= 2
        hot_raw = (np.array([_p_two(v) for v in z_raw]) <= 0.05) & (z_raw > 0)
        robust_hot = (hot_eb & hot_raw).sum() / max(hot_eb.sum(), 1) * 100 if hot_eb.any() else None

        self.step(68, "Testing whether cases cluster (Moran's I)")
        mi, mp = morans_i(x, nbrs)

        self.step(76, "Measuring distance to health facilities")
        fac, fac_src, fac_err = [], "", None
        if params.get("facilities"):
            for gl, attrs, gw in read_vector(params["facilities"], aoi):
                c = gw.Centroid()
                fac.append((c.GetX(), c.GetY(), str(attrs.get("name") or "")))
            fac_src = "your facilities layer"
        else:
            try:
                fac = [(lo, la, nm) for lo, la, nm, _ in osm.facilities(aoi.bbox, "health")]
                fac_src = "OpenStreetMap"
            except KitError as e:
                fac_err = str(e)
        dist = np.full(len(ids), np.nan)
        if fac:
            F = np.array([(f[0], f[1]) for f in fac])
            for i, g in enumerate(geoms_w):
                c = g.Centroid()
                lon, lat = c.GetX(), c.GetY()
                near = np.argsort((F[:, 0] - lon) ** 2 + (F[:, 1] - lat) ** 2)[:5]
                dist[i] = min(_hav(lon, lat, F[j, 0], F[j, 1]) for j in near)

        # ------------------------------------------------- validation
        self.step(84, "Validating")
        small = float((Pn[use] < 1000).mean() * 100)
        checks = [
            (2 if unmatched == 0 and n_used == len(ids) else 1 if n_used / len(ids) >= 0.9 else 0, 2,
             f"{n_used} of {len(ids)} areas have usable cases and population" + (f"; {unmatched} had no matching CSV row." if unmatched else ".")),
        ]
        if mi is not None:
            checks.append((2 if mp <= 0.05 else 0, 2, f"Moran's I = {mi:.2f} (p = {mp:.3f}): cases "
                           + ("cluster in space more than chance would produce." if mp <= 0.05 else "do not cluster beyond chance, so hot spots should be read with caution.")))
        if robust_hot is not None:
            checks.append((1 if robust_hot >= 70 else 0, 1,
                           f"{robust_hot:.0f}% of hot spots are also hot spots without smoothing, so they do not depend on the method."))
        checks.append((1 if islands == 0 else 0, 1, f"{islands} areas have no neighbours (islands) and cannot be tested." if islands else "Every area has at least one neighbour."))
        verdict = grade(checks)

        # ------------------------------------------------- outputs
        self.step(90, "Writing results")
        cls = np.full(len(ids), 0, np.uint8)
        cls[use] = cls_u
        zz = np.full(len(ids), np.nan)
        zz[use] = z
        pp = np.full(len(ids), np.nan)
        pp[use] = pv
        fdr = np.zeros(len(ids), bool)
        fdr[use] = fdr_sig
        out_path = os.path.join(out_dir, "health_hotspots.geojson")
        fc = {"type": "FeatureCollection", "features": []}
        import json
        for i in range(len(ids)):
            fc["features"].append({"type": "Feature", "geometry": json.loads(geoms_w[i].ExportToJson()), "properties": {
                "area_id": ids[i], "cases": C[i], "population": P[i],
                "rate_per_100k": None if np.isnan(raw[i]) else round(float(raw[i]), 2),
                "smoothed_rate": None if np.isnan(eb[i]) else round(float(eb[i]), 2),
                "gi_z": None if np.isnan(zz[i]) else round(float(zz[i]), 3),
                "gi_p": None if np.isnan(pp[i]) else round(float(pp[i]), 4),
                "hotspot": HOT[int(cls[i])][0] if cls[i] else "No data",
                "significant_after_fdr": bool(fdr[i]),
                "km_to_facility": None if np.isnan(dist[i]) else round(float(dist[i]), 2)}})
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(fc, fh)
        cats = "".join(f'<category render="true" value="{n}" symbol="{k - 1}" label="{n}"/>' for k, (n, _) in HOT.items())
        syms = "".join(f'<symbol type="fill" name="{k - 1}" alpha="1"><layer class="SimpleFill"><Option type="Map">'
                       f'<Option type="QString" name="color" value="{int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},255"/>'
                       f'<Option type="QString" name="outline_color" value="80,80,80,255"/>'
                       f'<Option type="QString" name="outline_width" value="0.2"/></Option></layer></symbol>'
                       for k, (_, c) in HOT.items())
        with open(os.path.splitext(out_path)[0] + ".qml", "w", encoding="utf-8") as fh:
            fh.write(f"<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'><qgis version=\"3.22.0\" styleCategories=\"Symbology\">"
                     f'<renderer-v2 type="categorizedSymbol" attr="hotspot" enableorderby="0" forceraster="0" symbollevels="0">'
                     f"<categories>{cats}</categories><symbols>{syms}</symbols></renderer-v2></qgis>")
        csv_path = os.path.join(out_dir, "health_table.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["area_id", "cases", "population", "rate_per_100k", "smoothed_rate", "gi_z", "gi_p", "hotspot", "km_to_facility"])
            for f in fc["features"]:
                p = f["properties"]
                w.writerow([p["area_id"], p["cases"], p["population"], p["rate_per_100k"], p["smoothed_rate"],
                            p["gi_z"], p["gi_p"], p["hotspot"], p["km_to_facility"]])
        files = [out_path, csv_path]
        layers = [dict(path=out_path, name=f"{disease} hot spots", kind="vector")]
        if fac:
            from ..core.raster import write_geojson
            f_fac = write_geojson(os.path.join(out_dir, "health_facilities.geojson"),
                                  [((lo, la), {"name": nm, "source": fac_src}) for lo, la, nm in fac])
            files.append(f_fac)
            layers.insert(0, dict(path=f_fac, name="Health facilities", kind="vector"))

        # maps for the report: rasterise the polygons
        self.step(94, "Writing the report")
        grid = Grid(aoi, 50, max_pixels=1_500_000)
        hot_r = grid.rasterize(geoms_l, values=[int(c) for c in cls])
        q = np.nanpercentile(eb[use], [20, 40, 60, 80])
        rc = np.zeros(len(ids), int)
        rc[use] = 1 + np.digitize(eb[use], q)
        rate_cls = {k + 1: (f"Quintile {k + 1}", RATE5[k]) for k in range(5)}
        rate_r = grid.rasterize(geoms_l, values=rc.tolist())
        nh = int((cls_u <= 3).sum())
        top = np.argsort(-np.where(use, eb, -np.inf))[:10]
        far = int(np.nansum(dist > lim)) if fac else None
        find = [f"Overall rate: <b>{fmt(base, 1)} per 100,000</b> across {n_used} areas ({int(Cn[use].sum()):,} cases, "
                f"{int(np.nansum(Pn[use])):,} people)."]
        find.append(f"<b>{nh}</b> areas are statistically significant hot spots (90% or higher), "
                    f"{int(fdr_sig.sum())} of which remain significant after correcting for testing many areas at once.")
        if mi is not None:
            find.append(f"Cases {'<b>cluster</b>' if mp <= 0.05 else 'do <b>not</b> clearly cluster'} geographically (Moran's I {mi:.2f}, p {mp:.3f}).")
        if far is not None:
            hot_far = int(((cls_u <= 3) & (dist[use] > lim)).sum())
            find.append(f"<b>{far}</b> areas are more than {lim:g} km from a health facility; <b>{hot_far}</b> of them are hot spots: the first places to send outreach.")
        rep = Report(self.NAME, f"{disease}: rates and hot spots", aoi.name, verdict, "#9e1b16", aoi, None)
        hot_map = MapSpec(hot_r.astype(np.uint8), grid, f"{disease} hot and cold spots (Getis-Ord Gi*)", classes=HOT,
                          subtitle=f"{aoi.name} · {n_used} areas", source="User case data and boundaries")
        results = [("Overall rate", f"{fmt(base, 1)} per 100,000"), ("Areas analysed", f"{n_used}"),
                   ("Hot spot areas (≥ 90%)", f"{nh} ({int(fdr_sig.sum())} after FDR)")]
        if mi is not None:
            results.append(("Clustering (Moran's I)", f"{mi:.2f}, p = {mp:.3f}"))
        if far is not None:
            results.append((f"Areas beyond {lim:g} km of a facility", f"{far}"))
        rep.summary(results, find, hot_map)
        rep.study_area(MapSpec(rate_r.astype(np.uint8), grid, f"{disease}: smoothed rate by area (quintiles)", classes=rate_cls,
                               subtitle=aoi.name, source="User case data and boundaries"),
                       [("Areas in the layer", f"{len(ids)}"), ("Areas with usable data", f"{n_used}"),
                        ("Total cases", f"{int(Cn[use].sum()):,}"), ("Total population", f"{int(np.nansum(Pn[use])):,}")])
        drows = [("Your area boundaries", "Units of analysis and neighbours", "polygons", "as supplied", "yours"),
                 ("Your case and population counts", "Rates", "per area", "as supplied", "yours")]
        if fac:
            drows.append(("Health facilities", "Distance to care", "points",
                          time.strftime("%Y-%m-%d") if fac_src == "OpenStreetMap" else "as supplied",
                          "ODbL (OpenStreetMap)" if fac_src == "OpenStreetMap" else "yours"))
        rep.data(drows)
        rep.method([
            ("Rates, not counts", "Cases ÷ population × 100,000 for each area, so large and small areas can be compared fairly."),
            ("Smoothing small numbers", "Global empirical Bayes (Marshall 1991) pulls each area's rate toward the overall rate in proportion to how "
             "unreliable it is: strongly for tiny populations, hardly at all for large ones."),
            ("Neighbours", "Areas that share a border or corner are neighbours (queen contiguity)."),
            ("Hot spot test", "Getis-Ord Gi* compares each area plus its neighbours with the whole region. A significant positive z-score marks "
             "a cluster of high rates, not just one high area. Significance is reported at 90, 95 and 99%, and the false discovery rate is controlled."),
            ("Clustering test", "Moran's I with 499 random shuffles tests whether similar rates sit next to each other more than chance would give."),
            ("Access", f"Straight-line distance from each area's centre to the nearest health facility ({fac_src or 'none found'})."),
        ])
        rep.heading("Hot and cold spots")
        rep.map(hot_map, f"{disease} hot and cold spots from smoothed rates.")
        rep.heading("Highest rates")
        rep.table(["Area", "Cases", "Population", "Rate /100k", "Smoothed /100k", "Hot spot", "km to facility"],
                  [[ids[i], fmt(C[i], 0), fmt(P[i], 0), fmt(raw[i], 1), fmt(eb[i], 1), HOT[int(cls[i])][0] if cls[i] else "–",
                    "–" if np.isnan(dist[i]) else fmt(dist[i], 1)] for i in top if use[i]],
                  "The ten areas with the highest smoothed rates")
        vrows = []
        if mi is not None:
            vrows.append(("Clustering beyond chance", f"Moran's I {mi:.2f}, p = {mp:.3f}",
                          "p ≤ 0.05 means cases really cluster; otherwise hot spots may be chance."))
        vrows.append(("Many tests at once", f"{int(fdr_sig.sum())} areas significant after FDR correction",
                      "Testing many areas at 95% flags about 5% by chance; FDR keeps only robust ones."))
        if robust_hot is not None:
            vrows.append(("Independent of smoothing", f"{robust_hot:.0f}% of hot spots also found with raw rates",
                          "Hot spots that appear either way do not depend on the method."))
        vrows.append(("Small populations", f"{small:.0f}% of areas under 1,000 people",
                      "Small areas give unstable rates; smoothing handles this."))
        vrows.append(("Data completeness", f"{n_used} of {len(ids)} areas usable" + (f"; {unmatched} without a CSV match" if unmatched else ""),
                      "Missing areas are left out, not filled in."))
        vrows.append(("Islands", f"{islands} areas without neighbours", "Islands cannot be tested for hot spots."))
        rep.validation(vrows)
        rep.limits([
            "Results depend on the area boundaries chosen (the modifiable areal unit problem).",
            "Reported cases reflect testing and reporting effort as well as disease.",
            "Distance is straight-line from area centres, not travel time.",
            "Area-level patterns do not describe individuals (ecological fallacy).",
        ])
        rep.conclusions(find + [f"Overall confidence: <b>{verdict['level']}</b>."])
        refs = ["Getis, A. & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. Geographical Analysis 24, 189–206.",
                "Marshall, R. J. (1991). Mapping disease and mortality rates using empirical Bayes estimators. Applied Statistics 40, 283–294.",
                "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika 37, 17–23.",
                "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B 57, 289–300."]
        if fac_src == "OpenStreetMap":
            refs.append(D.CREDIT_OSM)
        rep.references(refs)
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, layers=layers, summary=f"{nh} hot spot areas among {n_used}.")
