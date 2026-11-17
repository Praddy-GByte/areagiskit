# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""CropKit: a season of crop growth from Sentinel-2: cropping intensity, peak timing, vigour."""

import calendar
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ..core import datasets as D
from ..core.aoi import Grid
from ..core.data import KitError, item_date, search, trace, vsi, warp, warp_s2
from ..core.raster import preview_classes, preview_float, write_classes, write_float, write_geojson
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

SEASONS = {1: ("No crop cycle / fallow", "#c9c2b0"), 2: ("Single crop", "#e4b04a"),
           3: ("Double crop", "#5c9a3b"), 4: ("Triple crop", "#1f5a2c")}
VIGOUR = {1: ("Weak (peak NDVI < 0.5)", "#d9a24a"), 2: ("Fair (0.5–0.7)", "#a9c95f"), 3: ("Strong (> 0.7)", "#2f7a3a")}
SCL_OK = {4, 5, 6, 7}  # vegetation, bare soil, water, unclassified


def _month_add(d, k):
    m = d.month - 1 + k
    y = d.year + m // 12
    return dt.date(y, m % 12 + 1, 1)


class CropKit(Kit):
    NAME = "CropKit"
    KEY = "crop"
    TAGLINE = "Crop seasons, cropping intensity and vigour"
    DESCRIPTION = ("One year of Sentinel-2 images over the cropland in your area: how many crops are grown "
                   "per year, when each field peaks, how vigorous the crop is, and the average growth curve.")
    MAX_KM2 = 600
    PARAMS = [
        dict(name="start_month", type="date", label="First month of the crop year", default="2022-06-01",
             help="For India, June starts the kharif season and the year runs through rabi."),
        dict(name="months", type="int", label="Number of months", default=12),
        dict(name="max_cloud", type="int", label="Skip images with more cloud than (%)", default=60),
    ]

    def _ndvi_scene(self, items, grid):
        """NDVI for one date (all tiles of that date), with the scene classification cloud mask."""
        def rd(asset, res="average", typ=gdal.GDT_Float32, nod=0):
            trace(items)
            return warp([vsi(i["assets"][asset]["href"]) for i in items], grid, resample=res,
                        dtype=typ, src_nodata=nod, dst_nodata=np.nan if typ == gdal.GDT_Float32 else 0)
        trace(items)
        red, nir = warp_s2(items, "B04", grid), warp_s2(items, "B08", grid)
        scl = rd("SCL", "mode", gdal.GDT_Byte)
        ok = np.isin(scl, list(SCL_OK)) & (red > 0) & (nir > 0)
        nd = (nir - red) / (nir + red)
        return np.where(ok, nd, np.nan).astype(np.float32)

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        try:
            start = dt.date.fromisoformat(str(params.get("start_month") or "2022-06-01")[:10]).replace(day=1)
        except ValueError:
            raise KitError("Give the first month as YYYY-MM-DD.")
        n = max(4, min(int(params.get("months") or 12), 24))
        maxc = int(params.get("max_cloud") or 60)
        grid = Grid(aoi, 30, max_pixels=3_000_000)

        yr = min(max(start.year, D.LULC_YEARS[0]), D.LULC_YEARS[-1])
        self.step(4, f"Reading cropland from the {yr} land cover map")
        lc, _ = D.lulc(grid, yr)
        crop = (lc == 5) & grid.mask
        if crop.sum() * grid.px_km2 < 0.5:
            raise KitError("Less than 0.5 km² of cropland in this area; CropKit needs farmland.")

        months = [_month_add(start, k) for k in range(n)]
        stack = np.full((n,) + grid.shape, np.nan, np.float32)
        counts = []
        jobs = []
        for k, m in enumerate(months):
            last = calendar.monthrange(m.year, m.month)[1]
            items = search("sentinel-2-l2a", aoi.bbox, str(m), str(m.replace(day=last)),
                           query={"eo:cloud_cover": {"lt": maxc}}, max_items=200)
            by = {}
            for it in items:  # keep the newest processing of each tile/date
                key = (it["properties"].get("s2:mgrs_tile"), item_date(it))
                if key not in by or it["id"] > by[key]["id"]:
                    by[key] = it
            dates = {}
            for (tile, d), it in by.items():
                dates.setdefault(d, []).append(it)
            best = sorted(dates.items(), key=lambda kv: np.mean([i["properties"]["eo:cloud_cover"] for i in kv[1]]))[:3]
            counts.append(len(best))
            for d, its in best:
                jobs.append((k, d, its))
        if not jobs:
            raise KitError("No usable Sentinel-2 images in this period. Try a higher cloud limit.")

        done = [0]

        def work(job):
            k, d, its = job
            nd = self._ndvi_scene(its, grid)
            done[0] += 1
            self.step(8 + 60 * done[0] / len(jobs), f"Reading Sentinel-2 images ({done[0]}/{len(jobs)})")
            return k, nd

        with ThreadPoolExecutor(max_workers=6) as ex:
            for k, nd in ex.map(work, jobs):
                stack[k] = np.fmax(stack[k], nd)  # monthly maximum NDVI (clouds only lower NDVI)

        self.step(72, "Detecting crop cycles")
        # fill gaps by linear interpolation along time (vectorised), then smooth
        filled = stack.copy()
        flat = filled.reshape(n, -1)
        ok = np.isfinite(flat)
        T = np.arange(n)[:, None]
        prev = np.maximum.accumulate(np.where(ok, T, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(ok, T, n)[::-1], axis=0)[::-1]
        has_p, has_n = prev >= 0, nxt < n
        pv = np.take_along_axis(flat, np.clip(prev, 0, n - 1), 0)
        nv = np.take_along_axis(flat, np.clip(nxt, 0, n - 1), 0)
        w = np.where(has_p & has_n & (nxt > prev), (T - prev) / np.maximum(nxt - prev, 1), 0.0)
        interp = np.where(has_p & has_n, pv + (nv - pv) * w, np.where(has_p, pv, nv))
        enough = ok.sum(axis=0) >= max(3, n // 2)
        flat[:] = np.where(enough[None, :], interp, np.nan)
        sm = filled.copy()
        sm[1:-1] = (filled[:-2] + 2 * filled[1:-1] + filled[2:]) / 4

        peaks = np.zeros(grid.shape, np.uint8)
        for k in range(1, n - 1):
            c = sm[k]
            is_peak = (c >= sm[k - 1]) & (c > sm[k + 1]) & (c >= 0.45)
            # prominence: rise of at least 0.15 above the lowest point on each side
            left = np.nanmin(sm[:k], axis=0)
            right = np.nanmin(sm[k + 1:], axis=0)
            prom = c - np.fmax(left, right)
            peaks += (is_peak & (prom >= 0.15)).astype(np.uint8)
        seasons = np.zeros(grid.shape, np.uint8)
        have = crop & np.isfinite(sm).all(axis=0)
        seasons[have] = 1 + np.minimum(peaks[have], 3)
        peak = np.nanmax(sm, axis=0)
        peak_month = np.nanargmax(np.where(np.isfinite(sm), sm, -1), axis=0)
        vig = np.zeros(grid.shape, np.uint8)
        vig[have & (peak < 0.5)] = 1
        vig[have & (peak >= 0.5) & (peak <= 0.7)] = 2
        vig[have & (peak > 0.7)] = 3

        px = grid.px_km2
        crop_km2 = crop.sum() * px
        s_area = {k: (seasons == k).sum() * px for k in SEASONS}
        curve = [float(np.nanmean(stack[k][crop])) if np.isfinite(stack[k][crop]).any() else None for k in range(n)]
        curve_sm = [float(np.nanmean(sm[k][have])) if have.any() else None for k in range(n)]
        intensity = sum(s_area[k] * (k - 1) for k in SEASONS) / max(sum(s_area.values()), 1e-9)

        # -------------------------------------------------------- validation
        self.step(80, "Validating")
        obs = np.isfinite(stack[:, crop]).mean(axis=1) * 100
        completeness = float(obs.mean())
        months_ok = int((obs >= 50).sum())
        amp = (np.nanmax(sm, axis=0) - np.nanmin(sm, axis=0))
        consistent = float((amp[have] >= 0.25).mean() * 100) if have.any() else 0
        checks = [
            (2 if months_ok >= n - 1 else 1 if months_ok >= n * 0.7 else 0, 2,
             f"{months_ok} of {n} months have a cloud-free view of at least half the cropland (average {completeness:.0f}%)."),
            (2 if consistent >= 80 else 1 if consistent >= 60 else 0, 2,
             f"{consistent:.0f}% of mapped cropland shows a clear seasonal green-up (NDVI rise ≥ 0.25), confirming it is actively cropped."),
            (1 if have.sum() / max(crop.sum(), 1) >= 0.9 else 0, 1,
             f"{have.sum() / max(crop.sum(), 1) * 100:.0f}% of cropland pixels had enough images to detect cycles."),
        ]
        verdict = grade(checks)

        # -------------------------------------------------------- outputs
        self.step(88, "Writing maps")
        f_s = write_classes(os.path.join(out_dir, "cropping_intensity.tif"), seasons, grid, SEASONS)
        f_v = write_classes(os.path.join(out_dir, "crop_vigour.tif"), vig, grid, VIGOUR)
        f_p, _ = write_float(os.path.join(out_dir, "peak_ndvi.tif"), np.where(have, peak, np.nan), grid, "ndvi", 0.1, 0.9, "NDVI")
        stack_path = os.path.join(out_dir, "monthly_ndvi.tif")
        ds = gdal.GetDriverByName("GTiff").Create(stack_path, grid.cols, grid.rows, n, gdal.GDT_Float32,
                                                  options=["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=3"])
        ds.SetGeoTransform(grid.gt)
        ds.SetProjection(grid.wkt)
        for k in range(n):
            b = ds.GetRasterBand(k + 1)
            b.SetNoDataValue(-9999)
            b.SetDescription(months[k].strftime("%Y-%m"))
            b.WriteArray(np.where(np.isfinite(stack[k]) & grid.mask, stack[k], -9999))
        ds = None
        f_csv = os.path.join(out_dir, "cropland_ndvi_curve.csv")
        with open(f_csv, "w", encoding="utf-8") as fh:
            fh.write("month,mean_ndvi,smoothed_ndvi,cloud_free_share_pct\n")
            for k in range(n):
                fh.write(f"{months[k]:%Y-%m},{'' if curve[k] is None else f'{curve[k]:.4f}'},"
                         f"{'' if curve_sm[k] is None else f'{curve_sm[k]:.4f}'},{obs[k]:.1f}\n")
        rng = np.random.default_rng(5)
        pts = []
        for k in (2, 3, 1):
            idx = np.flatnonzero((seasons == k).ravel())
            for p in (rng.choice(idx, size=min(10, idx.size), replace=False) if idx.size else []):
                r, c = divmod(int(p), grid.cols)
                pts.append((grid.pixel_to_lonlat(r, c), {"id": len(pts) + 1, "expected": SEASONS[k][0],
                                                         "peak_month": months[int(peak_month[r, c])].strftime("%b %Y"), "correct": None}))
        f_pts = write_geojson(os.path.join(out_dir, "field_check_points.geojson"), pts)

        self.step(94, "Writing the report")
        tot = max(sum(s_area.values()), 1e-9)
        lab = [m.strftime("%b %y") for m in months]
        find = [f"<b>{fmt(crop_km2)} km²</b> of cropland analysed over {n} months from {months[0]:%b %Y}."]
        for k in (3, 2, 4):
            if s_area[k] / tot > 0.05:
                find.append(f"<b>{s_area[k] / tot * 100:.0f}%</b> of cropland grows a <b>{SEASONS[k][0].lower()}</b> per year ({fmt(s_area[k])} km²).")
        find.append(f"Cropping intensity: <b>{intensity * 100:.0f}%</b> "
                    f"({intensity:.2f} crops per field per year on average; 100% = one crop, 200% = two).")
        if any(v is not None for v in curve_sm):
            k = int(np.nanargmax([v if v is not None else -1 for v in curve_sm]))
            find.append(f"Crops are greenest in <b>{months[k]:%B %Y}</b> (average NDVI {curve_sm[k]:.2f}).")

        src = "Copernicus Sentinel-2 L2A (Microsoft Planetary Computer); IO/Esri land cover"
        rep = Report(self.NAME, f"Crop year {months[0]:%b %Y} – {months[-1]:%b %Y}", aoi.name, verdict, "#4c7a22", aoi, grid)
        int_map = MapSpec(seasons, grid, "Cropping intensity: crops per year", classes=SEASONS, subtitle=aoi.name, source=src)
        dates_used = sorted(d for _, d, _ in jobs)
        results = [("Cropland analysed", f"{fmt(crop_km2)} km²"),
                   ("Cropping intensity", f"{intensity * 100:.0f}%"),
                   ("Double or triple cropped", f"{(s_area[3] + s_area[4]) / tot * 100:.0f}% of cropland"),
                   ("Clear months", f"{months_ok} of {n}")]
        rep.summary(results, find, int_map)
        rep.study_area(MapSpec(lc, grid, f"Study area: land cover {yr}", classes=D.LULC, subtitle=aoi.name,
                               source="IO/Esri annual land cover"),
                       [("Cropland in the area", f"{fmt(crop_km2)} km²"), ("Crop year", f"{months[0]:%B %Y} to {months[-1]:%B %Y}")])
        rep.data([
            ("Sentinel-2 L2A (red, near-infrared, scene classes)", "Monthly greenness (NDVI)", "10 m (analysed at 30 m)",
             f"{len(dates_used)} dates, {dates_used[0]} to {dates_used[-1]}", "Copernicus data"),
            ("IO/Esri annual land cover", "Cropland mask", "10 m", str(yr), "CC BY 4.0"),
        ])
        rep.method([
            ("Cropland mask", f"Only pixels classed as crops in the {yr} land cover map are analysed."),
            ("Greenness", "NDVI = (near-infrared − red) / (near-infrared + red). Healthy leaves reflect near-infrared and absorb red, "
             "so NDVI rises as a crop grows and falls at harvest."),
            ("Cloud removal", "Each Sentinel-2 image has a scene classification layer; cloud, shadow, cirrus and snow pixels are removed."),
            ("Monthly composite", f"Up to three of the clearest dates per month are read; the highest NDVI per pixel is kept, because haze only lowers NDVI."),
            ("Counting crops", "Each pixel's monthly curve is gap-filled and smoothed. Each peak above 0.45 that rises at least 0.15 above "
             "the lowest point on either side counts as one crop."),
            ("Vigour", "Peak NDVI above 0.7 indicates a dense, healthy canopy; below 0.5 a thin, stressed or late crop."),
        ])
        rep.heading("Crops per year")
        rep.map(int_map, "Number of crop cycles per year on each cropland pixel.")
        rep.table(["Crops per year", "Area (km²)", "Share of cropland"],
                  [[SEASONS[k][0], fmt(s_area[k]), f"{s_area[k] / tot * 100:.1f}%"] for k in SEASONS],
                  "Cropping intensity classes", [c for _, c in SEASONS.values()])
        rep.heading("Growth through the year")
        rep.text("Each rise and fall of the curve is one crop: sowing, growth, harvest. Two humps in a year mean two crops, "
                 "such as kharif rice followed by rabi wheat.")
        rep.line([("Monthly mean NDVI", curve, "#9cba7f"), ("Smoothed", curve_sm, "#2f6b2a")], lab,
                 "Average greenness (NDVI) of the cropland, month by month.", "NDVI", 0, 0.9)
        rep.heading("Crop vigour")
        rep.map(MapSpec(vig, grid, "Crop vigour at peak", classes=VIGOUR, subtitle=aoi.name, source=src),
                "Crop vigour from the highest NDVI reached in the year.")
        rep.map(MapSpec(np.where(have, peak, np.nan), grid, "Peak greenness (NDVI)", ramp="ndvi", vmin=0.1, vmax=0.9, unit="NDVI",
                        subtitle=aoi.name, source=src), "Highest NDVI reached in the year on each cropland pixel.")
        vrows = [("Enough clear images", f"{months_ok} of {n} months with a clear view of at least half the cropland",
                  "Average clear share " + f"{completeness:.0f}%. Months: " + ", ".join(f"{lab[k]} {obs[k]:.0f}%" for k in range(n))),
                 ("Is the cropland really cropped?", f"{consistent:.0f}% show a seasonal green-up (NDVI rise ≥ 0.25)",
                  "Land cover maps sometimes call fallow land cropland; a real crop rises and falls through the year."),
                 ("Pixels with enough data", f"{have.sum() / max(crop.sum(), 1) * 100:.0f}% of cropland",
                  "Pixels with too few clear months are left out rather than guessed."),
                 ("Your own check", f"{len(pts)} points in field_check_points.geojson with the expected peak month",
                  "Ask farmers or visit to confirm the number of crops.")]
        rep.validation(vrows)
        rep.limits([
            "Crop type is not identified (rice vs wheat vs sugarcane); only the number and timing of cycles.",
            "Long cloudy monsoon months reduce accuracy; see the clear-month check.",
            "Fields smaller than about 30 × 30 m blend with neighbours.",
            "Perennial crops (sugarcane, orchards) show one long cycle and may be counted as a single crop.",
        ])
        rep.conclusions(find + [f"Overall confidence: <b>{verdict['level']}</b>."])
        rep.references([D.CREDIT_S2, D.CREDIT_LULC,
                        "Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. Remote Sensing of Environment 8, 127–150."])
        files = [f_s, f_v, f_p, stack_path, f_csv, f_pts]
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, summary=f"Cropping intensity {intensity * 100:.0f}% over {fmt(crop_km2)} km².", layers=[
            dict(path=f_pts, name="Field check points", kind="vector"),
            dict(path=f_s, name="Crops per year", kind="raster"),
            dict(path=f_v, name="Crop vigour", kind="raster", hidden=True),
            dict(path=f_p, name="Peak NDVI", kind="raster", hidden=True),
            dict(path=stack_path, name="Monthly NDVI (bands = months)", kind="raster", hidden=True)])
