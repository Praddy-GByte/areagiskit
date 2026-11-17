# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""UrbanKit: how a town grew, where it is hottest, and who is far from services."""

import datetime as dt
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal

from ..core import datasets as D
from ..core import osm
from ..core.aoi import Grid, read_vector
from ..core.data import KitError, item_date, search, trace, vsi, warp
from ..core.raster import (preview_classes, preview_float, write_classes, write_float, write_geojson)
from ..core.maps import MapSpec
from ..core.report import Report, fmt, grade
from .base import Kit

GROWTH = {1: ("Built before", "#8c3b2e"), 2: ("Newly built", "#f0663a"), 3: ("Not built", "#ece9df")}
ACCESS = {1: ("Within reach", "#2f7a4d"), 2: ("Far", "#e0a43a"), 3: ("Very far (2× limit)", "#b8322a")}


class UrbanKit(Kit):
    NAME = "UrbanKit"
    KEY = "urban"
    TAGLINE = "Urban growth, heat islands and access to services"
    DESCRIPTION = ("For a town or city: where it grew and what it replaced, the surface heat island from "
                   "Landsat thermal images, and which built-up areas are far from hospitals or schools.")
    MAX_KM2 = 1500
    PARAMS = [
        dict(name="year_from", type="year", label="Growth from year", default=2017, options=D.LULC_YEARS),
        dict(name="year_to", type="year", label="Growth to year", default=2023, options=D.LULC_YEARS),
        dict(name="heat_year", type="int", label="Year for the heat map", default=2023),
        dict(name="hot_months", type="text", label="Hot-season months (comma list)", default="4,5,6",
             help="India: April–June. Use 12,1,2 for southern-hemisphere summers."),
        dict(name="service", type="enum", label="Service to check access to", default="health",
             options=["health", "school", "all"]),
        dict(name="access_km", type="float", label="Acceptable distance (km)", default=2.0),
        dict(name="facilities", type="points", label="Your own facilities layer (optional)", optional=True,
             help="If empty, facilities are taken from OpenStreetMap."),
    ]

    # ------------------------------------------------------------- heat
    def _lst(self, grid, year, months):
        items = []
        for m in months:
            y = year
            last = (dt.date(y + (m == 12), (m % 12) + 1, 1) - dt.timedelta(days=1)).day
            items += search("landsat-c2-l2", grid.aoi.bbox, f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last}",
                            query={"eo:cloud_cover": {"lt": 30}, "platform": {"in": ["landsat-8", "landsat-9"]}},
                            max_items=100)
        by = {}
        for it in items:
            by.setdefault(item_date(it), []).append(it)
        dates = sorted(by, key=lambda d: np.mean([i["properties"]["eo:cloud_cover"] for i in by[d]]))[:6]
        if not dates:
            return None, []

        def one(d):
            its = by[d]
            trace(its)
            st = warp([vsi(i["assets"]["lwir11"]["href"]) for i in its], grid, resample="bilinear",
                      dtype=gdal.GDT_Float32, src_nodata=0, dst_nodata=np.nan)
            qa = warp([vsi(i["assets"]["qa_pixel"]["href"]) for i in its], grid, resample="near",
                      dtype=gdal.GDT_UInt16, src_nodata=1, dst_nodata=1)
            bad = (qa & 0b11111) > 0  # fill, dilated cloud, cirrus, cloud, shadow
            c = st * 0.00341802 + 149.0 - 273.15
            return np.where(bad | ~np.isfinite(c) | (c < -40) | (c > 80), np.nan, c).astype(np.float32)

        with ThreadPoolExecutor(max_workers=4) as ex:
            arrs = list(ex.map(one, dates))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lst = np.nanmedian(np.stack(arrs), axis=0)
        lst[~grid.mask] = np.nan
        return lst, dates

    def run(self, aoi, params, out_dir):
        self.prepare(aoi, out_dir, params)
        ya, yb = sorted((int(params.get("year_from") or 2017), int(params.get("year_to") or 2023)))
        if ya == yb:
            raise KitError("Choose two different years for growth.")
        hy = int(params.get("heat_year") or yb)
        try:
            months = [int(x) for x in str(params.get("hot_months") or "4,5,6").split(",") if x.strip()]
            if not all(1 <= m <= 12 for m in months):
                raise ValueError("months must be numbers from 1 to 12")
        except Exception:
            raise KitError("Hot-season months must be numbers 1–12 separated by commas, e.g. 4,5,6.")
        service = params.get("service") or "health"
        lim_km = float(params.get("access_km") or 2.0)
        grid = Grid(aoi, 30, max_pixels=4_000_000)
        px = grid.px_km2

        self.step(6, f"Reading land cover {ya} and {yb}")
        a, _ = D.lulc(grid, ya)
        b, _ = D.lulc(grid, yb)
        built_a, built_b = a == 7, b == 7
        growth = np.zeros(grid.shape, np.uint8)
        growth[grid.mask & (a > 0) & (b > 0)] = 3
        growth[built_a & built_b] = 1
        new = built_b & ~built_a & (a > 0)
        growth[new] = 2
        km_a, km_b, km_new = built_a.sum() * px, built_b.sum() * px, new.sum() * px
        replaced = {k: (new & (a == k)).sum() * px for k in D.LULC if k != 7}
        rate = ((km_b / km_a) ** (1 / (yb - ya)) - 1) * 100 if km_a > 0 else None

        self.step(22, f"Reading Landsat thermal images, {hy} months {','.join(map(str, months))}")
        lst, ldates = self._lst(grid, hy, months)
        lc_h, _ = D.lulc(grid, min(max(hy, D.LULC_YEARS[0]), D.LULC_YEARS[-1])) if lst is not None else (None, None)
        uhi = None
        if lst is not None:
            def m(sel):
                v = lst[sel & np.isfinite(lst)]
                return float(v.mean()) if v.size > 50 else None
            t_built, t_green = m(lc_h == 7), m(np.isin(lc_h, [2, 5, 11]))
            t_trees, t_water = m(lc_h == 2), m(lc_h == 1)
            uhi = (t_built - t_green) if (t_built is not None and t_green is not None) else None
            vb = lst[(lc_h == 7) & np.isfinite(lst)]
            hot_thr = float(vb.mean() + vb.std()) if vb.size else None
            hotspot = (lc_h == 7) & (np.nan_to_num(lst, nan=-99) > hot_thr) if hot_thr else np.zeros(grid.shape, bool)

        self.step(55, "Finding facilities")
        fac, fac_src, fac_err = [], "", None
        if params.get("facilities"):
            for gl, attrs, gw in read_vector(params["facilities"], aoi):
                p = gw.Centroid()
                fac.append((p.GetX(), p.GetY(), str(attrs.get("name") or ""), "your layer"))
            fac_src = "your facilities layer"
        else:
            try:
                fac = osm.facilities(aoi.bbox, service)
                fac_src = "OpenStreetMap"
            except KitError as e:
                fac_err = str(e)
        dist = None
        if fac:
            seed = np.zeros(grid.shape, bool)
            for lon, lat, *_ in fac:
                r, c = grid.lonlat_to_pixel(lon, lat)
                # facilities just outside the area still serve it: clamp to the edge
                if -200 <= r < grid.rows + 200 and -200 <= c < grid.cols + 200:
                    seed[min(max(r, 0), grid.rows - 1), min(max(c, 0), grid.cols - 1)] = True
            dist = grid.distance_to(seed) / 1000
            dist[~grid.mask] = np.nan
            acc = np.zeros(grid.shape, np.uint8)
            acc[built_b & (dist <= lim_km)] = 1
            acc[built_b & (dist > lim_km) & (dist <= 2 * lim_km)] = 2
            acc[built_b & (dist > 2 * lim_km)] = 3
            far_km2 = (acc >= 2).sum() * px
            far_pct = far_km2 / max(km_b, 1e-9) * 100

        # ----------------------------------------------------- validation
        self.step(72, "Validating")
        checks = []
        wc = D.worldcover(grid)
        built_agree = None
        if wc is not None:
            both = ((wc == 7) & built_b).sum()
            built_agree = both / max(built_b.sum(), 1) * 100
            checks.append((2 if built_agree >= 70 else 1 if built_agree >= 50 else 0, 2,
                           f"{built_agree:.0f}% of mapped built-up land is also built-up in ESA WorldCover 2021."))
        if lst is not None:
            phys = t_water is None or t_built is None or t_water < t_built
            checks.append((2 if len(ldates) >= 3 and phys else 1 if phys else 0, 2,
                           f"Heat map from {len(ldates)} clear Landsat scenes; "
                           + ("water is cooler than built-up land, as physics requires." if phys and t_water is not None
                              else "physical check passed." if phys else "water is not cooler than built-up land; treat the heat map with care.")))
        else:
            checks.append((0, 2, "No clear Landsat thermal scene in the chosen months; the heat map was skipped."))
        if fac:
            checks.append((1 if len(fac) >= 5 else 0, 1, f"{len(fac)} facilities from {fac_src}."
                           + (" OpenStreetMap can be incomplete in smaller towns; add your own layer for official lists." if fac_src == "OpenStreetMap" else "")))
        else:
            checks.append((0, 1, f"Access analysis skipped: {fac_err or 'no facilities found'}"))
        verdict = grade(checks)

        # ----------------------------------------------------- outputs
        self.step(82, "Writing maps")
        files, layers = [], []
        f_g = write_classes(os.path.join(out_dir, f"urban_growth_{ya}_{yb}.tif"), growth, grid, GROWTH)
        files.append(f_g)
        layers.append(dict(path=f_g, name=f"Urban growth {ya}→{yb}", kind="raster"))
        if lst is not None:
            lo, hi = np.nanpercentile(lst, [2, 98])
            f_l, _ = write_float(os.path.join(out_dir, f"surface_temperature_{hy}.tif"), lst, grid, "heat", float(lo), float(hi), "°C")
            files.append(f_l)
            layers.append(dict(path=f_l, name=f"Surface temperature {hy} (°C)", kind="raster", hidden=True))
        if dist is not None:
            f_d, _ = write_float(os.path.join(out_dir, f"distance_to_{service}_km.tif"), dist, grid, "risk", 0, 3 * lim_km, "km")
            f_a = write_classes(os.path.join(out_dir, f"access_{service}.tif"), acc, grid, ACCESS)
            f_f = write_geojson(os.path.join(out_dir, f"facilities_{service}.geojson"),
                                [((lon, lat), {"name": n, "type": t, "source": fac_src}) for lon, lat, n, t in fac])
            files += [f_a, f_d, f_f]
            layers = [dict(path=f_f, name=f"Facilities ({service})", kind="vector"),
                      dict(path=f_a, name=f"Access to {service}", kind="raster", hidden=True)] + layers
        f_csv = os.path.join(out_dir, "urban_summary.csv")
        with open(f_csv, "w", encoding="utf-8") as fh:
            fh.write("indicator,value\n")
            fh.write(f"built_{ya}_km2,{km_a:.3f}\nbuilt_{yb}_km2,{km_b:.3f}\nnew_built_km2,{km_new:.3f}\n")
            if rate is not None:
                fh.write(f"annual_growth_pct,{rate:.2f}\n")
            if uhi is not None:
                fh.write(f"surface_uhi_intensity_c,{uhi:.2f}\n")
            if dist is not None:
                fh.write(f"built_far_from_{service}_pct,{far_pct:.1f}\n")
        files.append(f_csv)

        self.step(92, "Writing the report")
        find = [f"Built-up land grew from <b>{fmt(km_a)}</b> to <b>{fmt(km_b)} km²</b> between {ya} and {yb}"
                + (f", about <b>{rate:.1f}% a year</b>." if rate is not None else ".")]
        top_rep = sorted(replaced.items(), key=lambda kv: -kv[1])
        if top_rep and top_rep[0][1] > 0.01:
            k, v = top_rep[0]
            find.append(f"New construction mostly replaced <b>{D.LULC[k][0].lower()}</b> ({fmt(v)} km², "
                        f"{v / max(km_new, 1e-9) * 100:.0f}% of new built-up land).")
        if uhi is not None:
            if uhi >= 0.5:
                find.append(f"In the hot season, built-up surfaces are <b>{uhi:.1f} °C hotter</b> than the surrounding "
                            f"trees, crops and grassland (surface heat island, Landsat {hy}).")
            else:
                find.append(f"No daytime surface heat island in this season: the surrounding land is <b>{abs(uhi):.1f} °C "
                            f"{'hotter' if uhi < 0 else 'cooler'}</b> or about as hot as built-up areas. This is common in dry "
                            "regions where fields are bare after harvest and heat up more than roofs and roads.")
            if t_trees is not None and t_built is not None and t_built - t_trees > 0.5:
                find.append(f"Tree cover is <b>{t_built - t_trees:.1f} °C cooler</b> than built-up land: the clearest "
                            "measure of how much shade and greenery cool the town.")
        if dist is not None:
            find.append(f"<b>{far_pct:.0f}%</b> of built-up land is more than {lim_km:g} km from the nearest {service} facility "
                        f"({len(fac)} facilities, {fac_src}).")

        src_lc = "IO/Esri annual land cover"
        rep = Report(self.NAME, f"Urban growth, heat and access", aoi.name, verdict, "#b2462a", aoi, grid)
        g_map = MapSpec(growth, grid, f"Urban growth {ya}–{yb}", classes=GROWTH, subtitle=aoi.name, source=src_lc)
        results = [("Built-up land", f"{fmt(km_a)} → {fmt(km_b)} km²")]
        if rate is not None:
            results.append(("Growth rate", f"{rate:.1f}% per year"))
        if uhi is not None:
            results.append(("Surface heat island (hot season)", "0.0 °C" if abs(uhi) < 0.05 else f"{uhi:+.1f} °C"))
            if t_trees is not None and t_built is not None:
                results.append(("Trees cooler than built-up by", f"{t_built - t_trees:.1f} °C"))
        if dist is not None:
            results.append((f"Built-up beyond {lim_km:g} km of {service}", f"{far_pct:.0f}%"))
        rep.summary(results, find, g_map)
        rep.study_area(MapSpec(b, grid, f"Study area: land cover {yb}", classes=D.LULC, subtitle=aoi.name, source=src_lc))
        drows = [("IO/Esri annual land cover", "Built-up land in each year", "10 m (analysed at 30 m)", f"{ya}, {yb}", "CC BY 4.0"),
                 ("ESA WorldCover", "Independent check of built-up land", "10 m", "2021", "CC BY 4.0")]
        if lst is not None:
            drows.append(("Landsat 8/9 Collection 2 surface temperature", "Hot-season heat map", "100 m (delivered at 30 m)",
                          ", ".join(ldates), "Public domain (USGS)"))
        if fac:
            drows.append((f"{service.title()} facilities", "Access analysis", "points", time.strftime("%Y-%m-%d") if fac_src == "OpenStreetMap" else "as supplied",
                          "ODbL (OpenStreetMap)" if fac_src == "OpenStreetMap" else "yours"))
        rep.data(drows)
        rep.method([
            ("Growth", f"Built-up pixels in the {ya} and {yb} land cover maps are compared; pixels built-up only in {yb} are new construction."),
            ("Heat from space", f"Landsat measures the heat surfaces radiate. Up to six of the clearest scenes in months {','.join(map(str, months))} "
             f"of {hy} are combined (median per pixel) after removing cloud and shadow with the Landsat quality band."),
            ("Surface heat island", "Mean surface temperature of built-up land minus that of trees, crops and grassland in the same area and season."),
            ("Access", f"Straight-line distance from every built-up cell to the nearest {service} facility ({fac_src or 'none found'}); "
             f"built-up cells beyond {lim_km:g} km are flagged."),
        ])
        rep.heading("Where the town grew")
        rep.map(g_map, f"Built-up land in {ya} and new built-up land by {yb}.")
        rows = [(D.LULC[k][0], v, D.LULC[k][1]) for k, v in top_rep if v > 0.005]
        if rows:
            rep.bars(rows, "What the new built-up land used to be (km²).", "km²")
        if lst is not None:
            rep.heading("Hot-season surface temperature")
            rep.map(MapSpec(lst, grid, f"Surface temperature, hot season {hy}", ramp="heat", vmin=float(lo), vmax=float(hi), unit="°C",
                            subtitle=aoi.name, source="Landsat 8/9 Collection 2 (USGS)"), "Median daytime surface temperature in the hot season.")
            trs = [(n, t) for n, t in (("Built-up", t_built), ("Trees, crops and grassland", t_green),
                                       ("Trees only", t_trees), ("Water", t_water)) if t is not None]
            rep.table(["Land cover", "Mean surface temperature (°C)"], [[n, fmt(t, 1)] for n, t in trs],
                      "Hot-season surface temperature by land cover")
            rep.text("Surface temperature is how hot roofs, roads and ground are, which is higher than air temperature.")
        if dist is not None:
            rep.heading(f"Access to {service} facilities")
            rep.map(MapSpec(acc, grid, f"Built-up land by distance to {service} facilities", classes=ACCESS, subtitle=aoi.name,
                            source=fac_src, points=[((lo_, la_), {"color": "#1c2621", "size": 2.5, "marker": "+"}) for lo_, la_, *_ in fac]),
                    f"Distance from built-up land to the nearest {service} facility (+ = facility).")
        vrows = []
        if built_agree is not None:
            vrows.append(("Built-up confirmed by a second map", f"{built_agree:.0f}% also built-up in ESA WorldCover",
                          "Different team and method agreeing on where buildings are."))
        if lst is not None:
            vrows.append(("Heat map inputs", f"{len(ldates)} clear Landsat scenes",
                          "Water should be cooler than built-up land; " + ("it is." if (t_water is None or t_built is None or t_water < t_built) else "it is not, so treat the heat map with care.")))
        if fac:
            vrows.append(("Facility list", f"{len(fac)} facilities from {fac_src}",
                          "OpenStreetMap can be incomplete in small towns; add an official list for planning use."))
        else:
            vrows.append(("Facility list", "not available", fac_err or "No facilities found."))
        rep.validation(vrows)
        rep.limits([
            "Land cover maps capture large construction; single new houses may be missed.",
            "Surface temperature is not air temperature; it is usually several degrees higher by day.",
            "Access uses straight-line distance and built-up area as a stand-in for where people live.",
            "OpenStreetMap coverage of facilities varies by town.",
        ])
        rep.conclusions(find + [f"Overall confidence: <b>{verdict['level']}</b>."])
        refs = [D.CREDIT_LULC, D.CREDIT_WC, D.CREDIT_LS]
        if fac_src == "OpenStreetMap":
            refs.append(D.CREDIT_OSM)
        rep.references(refs)
        rep.proof(self.sources(), self.reproduce_command())
        rep.files(files + ["report.html", "report.docx", "report.pdf"])
        docs = rep.write(out_dir, self.seconds())
        report = docs["html"]
        self.step(100, "Done")
        return dict(report=report, documents=docs, verdict=verdict, layers=layers,
                    summary=f"Built-up grew {fmt(km_new)} km²" + (f"; surface heat island {0.0 if abs(uhi) < 0.05 else uhi:+.1f} °C" if uhi is not None else "") + ".")
