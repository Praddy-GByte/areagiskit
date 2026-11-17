# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""LandKit: land use / land cover change between two years, validated."""

import os

import numpy as np

from ..core import datasets as D
from ..core.aoi import Grid
from ..core.data import KitError
from ..core.raster import preview_classes, write_classes, write_geojson
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

NO_CHANGE = 200
CHANGE_CLASSES = dict(D.LULC)
CHANGE_CLASSES[NO_CHANGE] = ("No change", "#d9d9d6")


class LandKit(Kit):
    NAME = "LandKit"
    KEY = "land"
    TAGLINE = "Land use / land cover change"
    DESCRIPTION = ("What changed on the ground between two years: 10 m land cover maps, change map, "
                   "area and transition tables, checked against an independent map.")
    MAX_KM2 = 5000
    PARAMS = [
        dict(name="year_from", type="year", label="From year", default=2017, options=D.LULC_YEARS),
        dict(name="year_to", type="year", label="To year", default=2023, options=D.LULC_YEARS),
    ]

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        ya, yb = sorted((int(params["year_from"]), int(params["year_to"])))
        if ya == yb:
            raise KitError("Choose two different years.")
        grid = Grid(aoi, 10, max_pixels=30_000_000)

        self.step(8, f"Reading {ya} land cover")
        a, tiles = D.lulc(grid, ya)
        self.step(28, f"Reading {yb} land cover")
        b, tb = D.lulc(grid, yb)
        check = yb + 1 if yb + 1 in D.LULC_YEARS else (yb - 1 if yb - 1 > ya else None)
        c = None
        if check:
            self.step(46, f"Reading {check} to test whether changes persist")
            c, _ = D.lulc(grid, check)
        self.step(62, "Reading ESA WorldCover for independent validation")
        wc = D.worldcover(grid)
        io21 = None
        if wc is not None:
            io21 = {ya: a, yb: b, check: c}.get(2021)
            if io21 is None:
                io21, _ = D.lulc(grid, 2021)

        self.step(78, "Calculating areas and transitions")
        px = grid.px_km2
        cnt_a = np.bincount(a.ravel(), minlength=256)
        cnt_b = np.bincount(b.ravel(), minlength=256)
        A = {k: cnt_a[k] * px for k in D.LULC}
        B = {k: cnt_b[k] * px for k in D.LULC}
        valid = (a > 0) & (b > 0)
        code = a[valid].astype(np.int32) * 256 + b[valid]
        tc = np.bincount(code, minlength=65536)
        trans = {(f, t): tc[f * 256 + t] * px for f in D.LULC for t in D.LULC if tc[f * 256 + t]}
        top = sorted(((k, v) for k, v in trans.items() if k[0] != k[1]), key=lambda kv: -kv[1])
        chg = np.zeros_like(a)
        chg[valid & (a == b)] = NO_CHANGE
        diff = valid & (a != b)
        chg[diff] = b[diff]

        # ---- validation
        checks = []
        agree = None
        if wc is not None and io21 is not None:
            v = (io21 > 0) & (wc > 0) & (io21 != 10)
            if v.any():
                agree = float((io21[v] == wc[v]).mean())
                per = {}
                for k in D.LULC:
                    o, t = io21[v] == k, wc[v] == k
                    if o.sum() + t.sum():
                        both = (o & t).sum()
                        per[k] = (100 * o.mean(), both / o.sum() if o.sum() else None,
                                  both / t.sum() if t.sum() else None)
                o = round(agree * 100)
                checks.append((2 if o >= 75 else 1 if o >= 60 else 0, 2,
                               f"{o}% of pixels agree with an independent map (ESA WorldCover 2021)."))
        persist = None
        if c is not None:
            ch = diff & (b != 10)
            if ch.any():
                persist = float((c[ch] == b[ch]).mean())
                p = round(persist * 100)
                checks.append((2 if p >= 70 else 1 if p >= 50 else 0, 2,
                               f"{p}% of detected changes are confirmed by the {check} map."))
        miss = float((((a == 0) | (b == 0) | (a == 10) | (b == 10))[grid.mask]).mean() * 100)
        checks.append((1 if miss < 5 else 0, 1, f"{miss:.1f}% of the area is missing or cloudy in one of the years."))
        verdict = grade(checks)

        # random points for a human check
        rng = np.random.default_rng(42)
        pts = []
        strata = [k for k, _ in top[:5]] + ["same"]
        for s in strata:
            if s == "same":
                idx = np.flatnonzero((valid & (a == b)).ravel())
                lab = "No change"
            else:
                idx = np.flatnonzero(((a == s[0]) & (b == s[1])).ravel())
                lab = f"{D.LULC[s[0]][0]} -> {D.LULC[s[1]][0]}"
            for p in rng.choice(idx, size=min(8, idx.size), replace=False) if idx.size else []:
                r, cc = divmod(int(p), grid.cols)
                pts.append(((grid.pixel_to_lonlat(r, cc)), {"id": len(pts) + 1, "expected": lab, "correct": None}))

        # ---- files
        self.step(88, "Writing maps and tables")
        f_a = write_classes(os.path.join(out_dir, f"landcover_{ya}.tif"), a, grid, D.LULC)
        f_b = write_classes(os.path.join(out_dir, f"landcover_{yb}.tif"), b, grid, D.LULC)
        f_c = write_classes(os.path.join(out_dir, f"change_{ya}_{yb}.tif"), chg, grid, CHANGE_CLASSES)
        f_p = write_geojson(os.path.join(out_dir, "validation_points.geojson"), pts)
        f_t = os.path.join(out_dir, "transitions.csv")
        with open(f_t, "w", encoding="utf-8") as fh:
            fh.write(f"from_{ya},to_{yb},area_km2\n")
            for (f, t), v in sorted(trans.items(), key=lambda kv: -kv[1]):
                fh.write(f"{D.LULC[f][0]},{D.LULC[t][0]},{v:.4f}\n")
        f_s = os.path.join(out_dir, "class_areas.csv")
        with open(f_s, "w", encoding="utf-8") as fh:
            fh.write(f"class,{ya}_km2,{yb}_km2,change_km2\n")
            for k, (n, _) in D.LULC.items():
                fh.write(f"{n},{A[k]:.4f},{B[k]:.4f},{B[k] - A[k]:.4f}\n")

        # ---- report
        self.step(94, "Writing the report")
        total = sum(v for k, v in A.items() if k != 10) or 1
        changed = sum(v for (f, t), v in trans.items() if f != t)
        find = []
        for k, d in sorted(((k, B[k] - A[k]) for k in D.LULC if k != 10), key=lambda kv: -abs(kv[1]))[:3]:
            if abs(d) >= 0.01:
                pct = f" ({d / A[k] * 100:+.1f}%)" if A[k] > 0.01 else ""
                find.append(f"<b>{D.LULC[k][0]}</b> {'grew' if d > 0 else 'shrank'} by <b>{fmt(abs(d))} km²</b>{pct}.")
        if top:
            (f, t), v = top[0]
            find.append(f"Largest conversion: <b>{D.LULC[f][0]} → {D.LULC[t][0]}</b>, {fmt(v)} km².")
        find.append(f"In total <b>{fmt(changed)} km²</b> ({changed / total * 100:.1f}% of the area) changed class.")

        src = "Impact Observatory/Microsoft/Esri 10 m annual land cover (Microsoft Planetary Computer)"
        rep = Report(self.NAME, f"Land cover change {ya}–{yb}", aoi.name, verdict, "#2f6b4f", aoi, grid)
        chg_map = MapSpec(chg, grid, f"Land cover change {ya}–{yb}", classes=CHANGE_CLASSES,
                          subtitle=f"{aoi.name} · colour shows the new class; grey = no change", source=src)
        results = [("Area that changed", f"{fmt(changed)} km² ({changed / total * 100:.1f}%)")]
        if top:
            (f, t), v = top[0]
            results.append(("Largest conversion", f"{D.LULC[f][0]} → {D.LULC[t][0]}: {fmt(v)} km²"))
        if A[7] > 0.001 or B[7] > 0.001:
            results.append(("Built area", f"{fmt(A[7])} → {fmt(B[7])} km²"))
        if A[2] > 0.001 or B[2] > 0.001:
            results.append(("Trees", f"{fmt(A[2])} → {fmt(B[2])} km²"))
        if agree is not None:
            results.append(("Agreement with ESA WorldCover", f"{round(agree * 100)}%"))
        rep.summary(results, find, chg_map)
        rep.study_area(MapSpec(a, grid, f"Study area: land cover {ya}", classes=D.LULC, subtitle=aoi.name, source=src),
                       [("Years compared", f"{ya} and {yb}")])
        rep.data([
            ("IO/Microsoft/Esri annual land cover v02", "Land cover in each year", "10 m", f"{ya}, {yb}" + (f", {check}" if check else ""), "CC BY 4.0"),
            ("ESA WorldCover", "Independent check", "10 m", "2021", "CC BY 4.0"),
        ])
        rep.method([
            ("Equal-area grid", f"The area was projected to an equal-area grid centred on it, so every {grid.res:.0f} m pixel has the same true area and km² figures are exact."),
            ("Annual land cover maps", "Each year's map was made by a deep-learning model from a full year of Sentinel-2 images. Only pixels inside the area were read."),
            ("Pixel-by-pixel comparison", f"Each pixel's class in {ya} is paired with its class in {yb}. Counting the pairs gives the transition table and the change map."),
            ("Areas", "Pixel count × pixel area gives km² for each class and each transition."),
            ("Validation", f"The {2021} map is compared with ESA WorldCover (different team, different method); detected change is tested against the {check or 'neighbouring'} map; random points are drawn for a human check."),
        ])
        rep.heading(f"Land cover in {ya} and {yb}")
        rep.map(MapSpec(b, grid, f"Land cover {yb}", classes=D.LULC, subtitle=aoi.name, source=src), f"Land cover in {yb}.")
        rep.heading("Where change happened")
        rep.map(chg_map, f"Pixels that changed class between {ya} and {yb}; colour shows the new class.")
        rows, sw = [], []
        for k, (n, col) in D.LULC.items():
            if A[k] < 0.001 and B[k] < 0.001:
                continue
            d = B[k] - A[k]
            rows.append([n, fmt(A[k]), fmt(B[k]), f"{d:+,.2f}", f"{d / A[k] * 100:+.1f}%" if A[k] > 0.001 else "new"])
            sw.append(col)
        rep.heading("Area by class")
        rep.table(["Class", f"{ya} (km²)", f"{yb} (km²)", "Change (km²)", "Change (%)"], rows, f"Area of each class in {ya} and {yb}", sw)
        rep.bars([(D.LULC[k][0], B[k] - A[k], D.LULC[k][1]) for k in D.LULC if k != 10 and (A[k] > 0.001 or B[k] > 0.001)],
                 f"Net change in area by class, {ya}–{yb} (km²; left = lost, right = gained).", "km²", signed=True)
        rep.heading("Largest conversions")
        rep.table(["From", "To", "Area (km²)", "Share of area"],
                  [[D.LULC[f][0], D.LULC[t][0], fmt(v), f"{v / total * 100:.1f}%"] for (f, t), v in top[:10]],
                  "The ten largest land cover conversions", [D.LULC[f][1] for (f, t), _ in top[:10]])
        vrows = []
        if agree is not None:
            pc = lambda x: "–" if x is None else f"{x * 100:.0f}%"
            vrows.append(("Independent map (ESA WorldCover 2021)", f"{round(agree * 100)}% of pixels agree",
                          "Different team, different method. Built area, water, trees and crops agree best; rangeland and shrubs are defined differently."))
            for k, (sh, x, y) in per.items():
                if sh >= 1:
                    vrows.append((f"  {D.LULC[k][0]}", f"ours confirmed {pc(x)}, theirs found {pc(y)}", f"{sh:.0f}% of the area"))
        if persist is not None:
            vrows.append(("Is the change real?", f"{round(persist * 100)}% of changed pixels confirmed by {check}",
                          "Real change also shows in the neighbouring year; pixels that flip back are noise."))
        vrows.append(("Missing or cloudy pixels", f"{miss:.1f}%", "Pixels without a valid class in one of the years."))
        vrows.append(("Your own check", f"{len(pts)} points in validation_points.geojson",
                      "Open each in satellite view and fill 'correct'; the share marked correct is your field accuracy."))
        rep.validation(vrows)
        rep.limits(["Changes of a few pixels are near the limit of 10 m data; treat very small areas as indicative.",
                    "Agreement compares two maps; it is a consistency check, not a field accuracy. Use the points for that.",
                    "Annual maps summarise a whole year, so seasonal change such as floods or harvests is not shown.",
                    "'Rangeland' covers grass, shrubs and open land and is often confused with crops in dry seasons."])
        rep.conclusions(find + [f"Overall confidence: <b>{verdict['level']}</b>."])
        rep.references([D.CREDIT_LULC, D.CREDIT_WC,
                        "Karra, K. et al. (2021). Global land use/land cover with Sentinel-2 and deep learning. IGARSS 2021."])
        files = [f_a, f_b, f_c, f_t, f_s, f_p]
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, layers=[
            dict(path=f_p, name="Validation points", kind="vector"),
            dict(path=f_c, name=f"Change {ya}→{yb}", kind="raster"),
            dict(path=f_b, name=f"Land cover {yb}", kind="raster", hidden=True),
            dict(path=f_a, name=f"Land cover {ya}", kind="raster", hidden=True)],
            summary=f"{fmt(changed)} km² changed class between {ya} and {yb}.")
