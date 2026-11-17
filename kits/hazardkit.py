# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""HazardKit: landslide susceptibility from terrain and land cover, with robustness and
(optional) inventory-based validation."""

import os

import numpy as np

from ..core import datasets as D
from ..core.aoi import Grid, read_vector
from ..core.data import KitError
from ..core.raster import preview_classes, preview_float, write_classes, write_float, write_geojson
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

CLASSES = {1: ("Very low", "#2f7a4d"), 2: ("Low", "#9cc96b"), 3: ("Moderate", "#f2d36b"),
           4: ("High", "#ec8b3a"), 5: ("Very high", "#b8322a")}
BREAKS = [0.2, 0.4, 0.6, 0.8]

# Land cover susceptibility score (0..1), from the general landslide literature:
# bare and sparsely vegetated slopes fail most, dense forest least.
LC_SCORE = {1: 0.0, 2: 0.3, 4: 0.2, 5: 0.6, 7: 0.7, 8: 1.0, 9: 0.0, 10: 0.5, 11: 0.8}

WEIGHTS = {"slope": 0.45, "relief": 0.15, "curvature": 0.15, "landcover": 0.15, "valley": 0.10}
FACTOR_LABEL = {"slope": "Slope", "relief": "Local relief", "curvature": "Concave hollows",
                "landcover": "Land cover", "valley": "Valley-side position"}


def _slope_score(s):
    # 0 below 10°, rising to 1 at 35°, easing slightly on near-vertical rock (>55°)
    sc = np.clip((s - 10) / 25, 0, 1)
    return np.where(s > 55, sc * 0.8, sc)


def classify(idx, mask):
    c = np.zeros(idx.shape, np.uint8)
    v = mask & np.isfinite(idx)
    c[v] = 1 + np.digitize(idx[v], BREAKS)
    return c


def auc(scores_pos, scores_all):
    """ROC AUC of the index at landslide pixels against all pixels (Mann-Whitney)."""
    a = np.sort(scores_all)
    ranks = np.searchsorted(a, scores_pos, side="left") + 0.5 * (
        np.searchsorted(a, scores_pos, side="right") - np.searchsorted(a, scores_pos, side="left"))
    return float(ranks.mean() / a.size)


class HazardKit(Kit):
    NAME = "HazardKit"
    KEY = "hazard"
    TAGLINE = "Landslide susceptibility"
    DESCRIPTION = ("Where slopes are most likely to fail: a susceptibility map from slope, relief, hollows, "
                   "valley position and land cover, with a robustness test and, if you add past landslide "
                   "points, a measured accuracy (AUC).")
    MAX_KM2 = 2000
    PARAMS = [
        dict(name="inventory", type="points", label="Past landslides (optional point layer)", optional=True,
             help="Any point file of known landslides. If given, the kit measures how well the map predicts them."),
    ]

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        grid = Grid(aoi, 30, max_pixels=6_000_000)

        self.step(8, "Reading elevation (Copernicus DEM 30 m)")
        elev = D.dem(grid)
        self.step(25, "Reading height above drainage (HAND)")
        try:
            hand = D.hand(grid)
        except KitError:
            hand = None
        self.step(40, "Reading land cover")
        lc, _ = D.lulc(grid, D.LULC_YEARS[-1])

        self.step(55, "Computing terrain factors")
        slope, _ = D.slope_aspect(elev, grid)
        rel = D.relief(elev, max(1, int(round(150 / grid.res))))
        z = np.where(np.isfinite(elev), elev, np.nanmean(elev))
        lap = (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z) / grid.res ** 2
        # positive Laplacian = concave hollow, where water and soil converge
        F = {
            "slope": _slope_score(np.nan_to_num(slope)),
            "relief": np.clip(np.nan_to_num(rel) / 300, 0, 1),
            "curvature": np.clip(lap / 0.004 * 0.5 + 0.5, 0, 1),
            "landcover": np.vectorize(lambda v: LC_SCORE.get(int(v), 0.5))(lc).astype(np.float32),
        }
        if hand is not None:
            h = np.nan_to_num(hand, nan=50)
            # lower valley sides (5-60 m above drainage) undercut by streams; floors and ridgetops less so
            F["valley"] = np.where((h > 2) & (h < 60), 1 - np.abs(h - 25) / 35, 0).clip(0, 1)
        W = {k: WEIGHTS[k] for k in F}
        tw = sum(W.values())
        W = {k: v / tw for k, v in W.items()}
        mask = grid.mask & np.isfinite(elev)
        excl = np.isin(lc, [1, 9]) | (np.nan_to_num(slope) < 5)  # water, snow, flat ground
        idx = sum(W[k] * F[k] for k in F).astype(np.float32)
        idx = np.where(excl, np.minimum(idx, 0.15), idx)
        idx[~mask] = np.nan
        cls = classify(idx, mask)

        steep = float((np.nan_to_num(slope) >= 15)[mask].mean() * 100)
        px = grid.px_km2
        areas = {k: (cls == k).sum() * px for k in CLASSES}

        # -------------------------------------------------------- validation
        self.step(70, "Testing robustness to the factor weights")
        rng = np.random.default_rng(11)
        stable = np.zeros(cls.shape, np.int16)
        runs = 20
        sub = (slice(None, None, 2), slice(None, None, 2))
        for _ in range(runs):
            w = {k: v * rng.uniform(0.8, 1.2) for k, v in W.items()}
            s = sum(w.values())
            i2 = sum((w[k] / s) * F[k][sub] for k in F)
            i2 = np.where(excl[sub], np.minimum(i2, 0.15), i2)
            stable[sub] += (classify(i2, mask[sub]) == cls[sub])
        robust = float((stable[sub][mask[sub]] == runs).mean() * 100)
        hi_robust = float((stable[sub][mask[sub] & (cls[sub] >= 4)] == runs).mean() * 100) if (cls[sub] >= 4).any() else None

        checks = [(2 if robust >= 80 else 1 if robust >= 60 else 0, 2,
                   f"{robust:.0f}% of pixels keep their class when every factor weight is varied by ±20% (20 runs).")]
        checks.append((1 if steep >= 10 else 0, 1,
                       f"{steep:.0f}% of the area has slopes of 15° or more" + (
                           "." if steep >= 10 else "; landslide analysis is of limited use on mostly flat ground.")))
        inv = None
        inv_path = params.get("inventory")
        if inv_path:
            self.step(78, "Checking the map against your past landslides")
            feats = read_vector(inv_path, aoi)
            rc = []
            for gl, _, _ in feats:
                p = gl.Centroid()
                r = int((grid.gt[3] - p.GetY()) / grid.res)
                c = int((p.GetX() - grid.gt[0]) / grid.res)
                if 0 <= r < grid.rows and 0 <= c < grid.cols and mask[r, c]:
                    rc.append((r, c))
            if len(rc) >= 5:
                pos = np.array([idx[r, c] for r, c in rc])
                allv = idx[mask]
                a = auc(pos, allv)
                in_high = float(np.mean([cls[r, c] >= 4 for r, c in rc]) * 100)
                hi_area = (areas[4] + areas[5]) / max(sum(areas.values()), 1e-9) * 100
                inv = dict(n=len(rc), auc=a, in_high=in_high, hi_area=hi_area)
                checks.append((3 if a >= 0.8 else 2 if a >= 0.7 else 1 if a >= 0.6 else 0, 3,
                               f"Against {len(rc)} known landslides the map scores AUC {a:.2f}; "
                               f"{in_high:.0f}% of them fall in the High/Very high classes, which cover {hi_area:.0f}% of the area."))
            else:
                checks.append((0, 1, f"Only {len(rc)} of your landslide points fall inside the area; at least 5 are needed to measure accuracy."))
        else:
            checks.append((0, 1, "No past-landslide points were given, so accuracy could not be measured. "
                                 "Add an inventory to get an AUC score."))
        verdict = grade(checks)

        # -------------------------------------------------------- outputs
        self.step(86, "Writing maps")
        f_cls = write_classes(os.path.join(out_dir, "landslide_susceptibility.tif"), cls, grid, CLASSES)
        f_idx, _ = write_float(os.path.join(out_dir, "susceptibility_index.tif"), idx, grid, "risk", 0, 1)
        f_slope, _ = write_float(os.path.join(out_dir, "slope_deg.tif"), np.where(mask, slope, np.nan), grid, "risk", 0, 45, "°")
        pts = []
        rng2 = np.random.default_rng(3)
        for k in (5, 4, 1):
            idxs = np.flatnonzero((cls == k).ravel())
            for p in (rng2.choice(idxs, size=min(10, idxs.size), replace=False) if idxs.size else []):
                r, c = divmod(int(p), grid.cols)
                pts.append((grid.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "class": CLASSES[k][0], "landslide_signs": None}))
        f_pts = write_geojson(os.path.join(out_dir, "field_check_points.geojson"), pts)
        f_csv = os.path.join(out_dir, "susceptibility_areas.csv")
        with open(f_csv, "w", encoding="utf-8") as fh:
            fh.write("class,area_km2,share_pct\n")
            tot = sum(areas.values()) or 1
            for k, (n, _) in CLASSES.items():
                fh.write(f"{n},{areas[k]:.4f},{areas[k] / tot * 100:.2f}\n")

        self.step(94, "Writing the report")
        tot = sum(areas.values()) or 1
        hi = areas[4] + areas[5]
        built_hi = ((cls >= 4) & (lc == 7)).sum() * px
        find = [f"<b>{fmt(hi)} km²</b> ({hi / tot * 100:.0f}% of the area) is in the High or Very high landslide susceptibility classes."]
        if built_hi > 0.005:
            find.append(f"<b>{fmt(built_hi)} km²</b> of built-up land lies in those classes: settlements and roads on unstable slopes.")
        find.append(f"Slope is the strongest factor: {steep:.0f}% of the area is steeper than 15°.")
        if inv:
            find.append(f"Tested on your {inv['n']} known landslides: AUC <b>{inv['auc']:.2f}</b> "
                        f"(0.5 = no better than chance, 1 = perfect).")

        src = "Copernicus DEM GLO-30; ASF GLO-30 HAND; IO/Esri land cover"
        rep = Report(self.NAME, "Landslide susceptibility", aoi.name, verdict, "#9a4a1c", aoi, grid)
        sus_map = MapSpec(cls, grid, "Landslide susceptibility", classes=CLASSES, subtitle=aoi.name, source=src,
                          show_classes=list(CLASSES))
        results = [("High + Very high", f"{fmt(hi)} km² ({hi / tot * 100:.0f}%)"),
                   ("Area steeper than 15°", f"{steep:.0f}%"),
                   ("Stable under ±20% weights", f"{robust:.0f}% of pixels")]
        if built_hi > 0.005:
            results.append(("Built-up land in High/Very high", f"{fmt(built_hi)} km²"))
        if inv:
            results.append(("Accuracy on known landslides", f"AUC {inv['auc']:.2f}"))
        rep.summary(results, find, sus_map)
        rep.study_area(MapSpec(np.where(mask, elev, np.nan), grid, "Study area: elevation", ramp="elev",
                               vmin=float(np.nanpercentile(elev, 2)), vmax=float(np.nanpercentile(elev, 98)), unit="m",
                               subtitle=aoi.name, source="Copernicus DEM GLO-30"),
                       [("Landslide inventory", f"{inv['n']} points supplied" if inv else "none supplied")])
        rep.data([
            ("Copernicus DEM GLO-30", "Slope, relief, curvature", "30 m", "2011–2015", "Copernicus DEM licence"),
            ("ASF GLO-30 HAND", "Valley-side position", "30 m", "2021 release", "CC0"),
            ("IO/Esri annual land cover", "Land cover factor", "10 m (resampled to 30 m)", str(D.LULC_YEARS[-1]), "CC BY 4.0"),
        ] + ([("Your landslide inventory", "Accuracy test (AUC)", "points", "as supplied", "yours")] if inv else []))
        rep.method([
            ("Slope", "Gravity acts more strongly on steeper slopes. The score is 0 below 10°, rises to 1 at 35°, and eases on near-vertical rock."),
            ("Relief and hollows", "High local relief (range of elevation within 150 m) and concave hollows, where water and soil converge, raise the score."),
            ("Valley sides", "Streams undercut lower valley sides; height above nearest drainage (HAND) between 2 and 60 m marks them."),
            ("Land cover", "Tree roots hold soil; bare ground, grass and cut slopes for roads and houses do not. Each class has a fixed score."),
            ("Weighted overlay", "Factors are scaled 0–1 and combined with the weights in Table 3. The index is cut into five classes at fixed values "
             "(0.2, 0.4, 0.6, 0.8), so maps of different areas are comparable."),
            ("Validation", "Weights are an expert choice, so they are varied ±20% twenty times to test stability. With a landslide inventory, "
             "the ROC AUC measures how well the index ranks real landslide locations."),
        ])
        rep.heading("Susceptibility map")
        rep.map(sus_map, "Landslide susceptibility in five classes.")
        rep.map(MapSpec(np.where(mask, slope, np.nan), grid, "Slope", ramp="risk", vmin=0, vmax=45, unit="°", subtitle=aoi.name,
                        source="Copernicus DEM GLO-30"), "Slope, the strongest factor.")
        rows = [[CLASSES[k][0], fmt(areas[k]), f"{areas[k] / tot * 100:.1f}%"] for k in CLASSES]
        rep.heading("Area by class")
        rep.table(["Class", "Area (km²)", "Share"], rows, "Area in each susceptibility class", [c for _, c in CLASSES.values()])
        rep.heading("Factor weights")
        rep.table(["Factor", "Weight"], [[FACTOR_LABEL[k], f"{W[k] * 100:.0f}%"] for k in W], "Weights used in the overlay")
        vrows = [("Robustness to weights", f"{robust:.0f}% of pixels keep their class"
                  + (f"; {hi_robust:.0f}% of High/Very high pixels" if hi_robust is not None else ""),
                  "Weights varied ±20% in 20 random runs. A high share means the map does not hinge on the exact weights.")]
        if inv:
            vrows.append(("Known landslides", f"AUC {inv['auc']:.2f}; {inv['in_high']:.0f}% of {inv['n']} in High/Very high",
                          f"High/Very high cover {inv['hi_area']:.0f}% of the area. AUC 0.5 = chance, 0.7 acceptable, 0.8+ good."))
        else:
            vrows.append(("Known landslides", "not tested",
                          "Add past landslide points (state disaster authority, Geological Survey, NASA Global Landslide Catalog, field survey) to measure accuracy."))
        vrows.append(("Terrain suitability", f"{steep:.0f}% of the area steeper than 15°",
                      "Below about 10%, landslide mapping adds little."))
        vrows.append(("Your own check", f"{len(pts)} points in field_check_points.geojson",
                      "Visit or view and note signs of instability (cracks, scars, tilted trees)."))
        rep.validation(vrows)
        rep.limits([
            "Rainfall, soil depth and geology are not included; no open global layer is precise enough at this scale.",
            "This shows where slopes are prone to fail, not when. Use it with rainfall alerts for early warning.",
            "30 m terrain cannot see small road cuts or retaining walls.",
            "Without a landslide inventory the map is an informed estimate, not a validated model.",
        ])
        rep.conclusions(find + [f"Overall confidence: <b>{verdict['level']}</b>."])
        rep.references([D.CREDIT_DEM, D.CREDIT_HAND, D.CREDIT_LULC,
                        "Reichenbach, P. et al. (2018). A review of statistically-based landslide susceptibility models. Earth-Science Reviews 180, 60–91.",
                        "Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters 27, 861–874."])
        files = [f_cls, f_idx, f_slope, f_csv, f_pts]
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, summary=f"{fmt(hi)} km² high or very high landslide susceptibility.", layers=[
            dict(path=f_pts, name="Field check points", kind="vector"),
            dict(path=f_cls, name="Landslide susceptibility", kind="raster"),
            dict(path=f_idx, name="Susceptibility index", kind="raster", hidden=True),
            dict(path=f_slope, name="Slope (°)", kind="raster", hidden=True)])
