# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Build one separate QGIS plugin per study (and one ArcGIS Pro toolbox zip) from the shared engine (the areagiskit folder next to this script).

Each plugin is self-contained (its own copy of the engine, its own folder/package name, its own Processing
provider id), so a user can install one, several or all of them side by side without conflicts.
"""
import os
import re
import shutil
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("KITS_SRC") or (HERE if os.path.basename(HERE) == "areagiskit" else os.path.join(HERE, "areagiskit"))
DIST = os.environ.get("KITS_DIST") or os.path.join(os.path.dirname(SRC), "dist")
OUT = os.environ.get("KITS_OUT") or os.path.join(os.path.dirname(SRC), "plugins")
VERSION = "0.14.0"
AUTHOR = "Pradeepika Kaushik"
GH = "https://github.com/Praddy-GByte"

COMMON_ABOUT = ("AreaGIS Kit is an open-source research workflow for QGIS (an ArcGIS Pro toolbox is also available, beta): pick an area, set one or two options and run. "
                "The plugin fetches open data for your area only (no account, no manual download, no extra installs: it uses "
                "only GDAL and numpy, which ship with QGIS), runs the standard analyses for its field, checks its results "
                "against independent data where it can, grades its own confidence (High/Medium/Low, computed from the checks), "
                "and writes a reproducible report in HTML, Word and PDF with print-ready maps, a validation table, a list of "
                "what the results cannot tell you, and a proof appendix that links every input scene. ")

PLUGINS = [
    dict(key="floodstudy", folder="areagiskit_flood", name="AreaGIS Kit: Flood", colour="#1f6fb2", glyph="F",
         description="Complete flood study in one run: Sentinel-1 flood extent, duration, depth, impact, flood history, "
                     "flood-prone zones, rainfall and river flow, checked against Copernicus EMS expert maps.",
         about="Flood extent from Sentinel-1 radar (through cloud) with a classifier trained on 52 expert-mapped floods and the "
               "Copernicus GFM archive, duration, depth and volume, people, buildings, roads and crops affected, damage (JRC curves, "
               "money only if you give asset values), flood history and frequency since 2017, susceptibility models, rainfall return "
               "periods, river flow, a river-flow outlook (GloFAS), water hidden under trees, crops and buildings (candidates), "
               "10/100-year screening depths from a HAND model, nearest safe high ground per settlement, and automatic comparison "
               "with Copernicus EMS expert maps where they exist.",
         tags="flood,sentinel-1,SAR,flood mapping,flood depth,disaster,copernicus,GFM,EMS,validation,report",
         validation=[
             "Benchmark: 52 real floods in 20 countries against Copernicus EMS expert maps made from other satellites "
             "(leave-event-out): 69% of expert-mapped flood found, 61% of mapped flood correct; mean CSI 0.46 vs 0.39 for GFM alone.",
             "Valley flood (Gudbrandsdalen, Norway 2023): CSI 0.58-0.63 against the expert map.",
             "Limits: about a quarter of expert-mapped flood is invisible to radar (water under crops or trees, wet soil); depth "
             "has not been checked against measured depths; the river-flow outlook has no local skill score; safe high ground is a "
             "straight-line distance, not a checked route."],
         example='floodstudy event_date=2022-09-03 --bbox 67.6 26.6 67.9 26.85 --name "Dadu, Sindh"'),
    dict(key="lulc", folder="areagiskit_lulc", name="AreaGIS Kit: LULC", colour="#2f6b4f", glyph="L",
         description="Complete land use / land cover study in one run: Landsat classification since 1985, change, "
                     "landscape metrics, urban growth, environment indicators and projection, with an accuracy sample.",
         about="Landsat classification for any years since 1985 (Random Forest; labels from Esri and ESA WorldCover or your own "
               "points), change and transitions, intensity analysis, landscape metrics, urban expansion and sprawl, surface "
               "temperature, RSEI, carbon, habitat quality and ecosystem value (transferred coefficients, stated as estimates), "
               "a Sentinel-2 10 m map of the latest year, tree-cover loss alerts, zone, master-plan and census tables, and "
               "Markov + cellular automata projection that is produced only if it passes its own hindcast.",
         tags="LULC,land cover,land use,landsat,change detection,classification,random forest,urban growth,projection,report",
         validation=[
             "Independent check: 77-82% agreement with the ESA CCI land cover map for older years (Gurugram).",
             "Accuracy on the automatic labels is reported as optimistic, because those labels come from the same maps used "
             "for training. For a true accuracy, label the stratified sample (Olofsson) that the plugin writes for you.",
             "Projection: hindcast figure of merit and disagreement are reported; no future map is drawn if the hindcast fails."],
         example='lulc years=2005,2015,2025 --bbox 76.98 28.40 77.08 28.48 --name Gurugram'),
    dict(key="landslide", folder="areagiskit_landslide", name="AreaGIS Kit: Landslide", colour="#8a5a2b", glyph="S",
         description="Complete landslide study in one run: automatic Sentinel-2 inventory, ten susceptibility models, "
                     "relative slope stability, debris runout, risk, and a hindcast of a dated event.",
         about="Automatic landslide inventory from Sentinel-2 with a recovery test (scars that are green again a year later, such as "
               "pruned plantations, are dropped), activity by year and new scars after an event, 15+ conditioning factors, ten "
               "susceptibility models (FR, IV, WoE, IoE, AHP, logistic regression, random forest, SVM, gradient boosting, ensemble) "
               "with spatial block cross-validation, relative slope stability, debris runout, risk to people and settlements, "
               "observed frequency per class, and a hindcast: maps rebuilt from scars seen before an event and tested on that event.",
         tags="landslide,susceptibility,sentinel-2,inventory,slope stability,runout,hindcast,hazard,risk,report",
         validation=[
             "Wayanad, Kerala, 30 July 2024 (hindcast): a map built only from 2019-2023 data flagged the published source "
             "area and 100% of the first 2.5 km of the debris path; about 30% of the next 1.5 km.",
             "The automatic event scar sits about 100 m from the published source point (11.4654 N, 76.1358 E).",
             "Limits: the default runout angle (11°) is too short for long debris flows (Wayanad ran about 8 km); rainfall from "
             "~50 km cells does not measure local triggers; slope stability uses textbook soil strengths, not site tests."],
         example='landslide event_date=2024-07-30 activity_years=7 --bbox 76.08 11.43 76.22 11.56 --name "Wayanad (Meppadi)"'),
    dict(key="cropstudy", folder="areagiskit_crop", name="AreaGIS Kit: Crop", colour="#7a8f1f", glyph="C",
         description="Complete crop study in one run: Sentinel-2 crop calendar, cropping intensity, crop groups, "
                     "radar rice, drought indices, irrigation, productivity since 2001 and residue burning.",
         about="10-day Sentinel-2 NDVI calendar with gap filling, sowing, peak and harvest per season, cropping intensity, crop "
               "groups without field data, rice from Sentinel-1 radar, a named crop map from your field points, drought (VCI, "
               "TCI, VHI, SPI, soil moisture), evapotranspiration and water productivity, irrigated vs rainfed, a productivity "
               "anomaly since 2001 from MODIS GPP (a yield proxy, not yield; t/ha only when you give reported yields), residue "
               "burning, automatic field boundaries and management zones.",
         tags="crop,agriculture,NDVI,sentinel-2,phenology,cropping intensity,rice,drought,VHI,irrigation,report",
         validation=[
             "Kentucky: the MODIS productivity anomaly picks 2007 and 2012 as the worst years (r = 0.78 with reported corn yield) "
             "but understates the size of losses; the Vegetation Health Index agrees with that productivity record (r = 0.75 over 22 seasons).",
             "Limits: crop names need your field points; irrigated vs rainfed is indicative outside the regions it was checked in."],
         example='cropstudy start=2021-01-01 --bbox -88.75 36.62 -88.65 36.70 --name "Graves County, Kentucky"'),
    dict(key="urbanstudy", folder="areagiskit_urban", name="AreaGIS Kit: Urban", colour="#c0392b", glyph="U",
         description="Complete city study in one run: day and night heat island, heat stress, green space, heat "
                     "vulnerability, access, air quality, local climate zones and growth.",
         about="Landsat surface temperature, heat island against an equal-area countryside ring (day Landsat, day/night MODIS) with "
               "its trend since 2003, hot spots, UTFVI, park and water cooling, tree canopy, green space per person and 3-30-300, "
               "heat vulnerability including poverty, access to hospitals and schools, built-up growth and building on low land, "
               "NO2 column (Sentinel-5P), PM2.5 (CAMS), hourly air temperature, heat index, WBGT and UTCI (NASA POWER), Local "
               "Climate Zones, building height, albedo, a +10% canopy scenario and ward tables.",
         tags="urban,urban heat island,land surface temperature,landsat,MODIS,UTCI,green space,heat vulnerability,air quality,report",
         validation=[
             "Gurugram 2024: Landsat vs MODIS surface temperature on the same day, r = 0.81.",
             "Limits: air temperature, heat index, WBGT and UTCI come from ~50 km cells; UTCI in shade uses a simple model; "
             "NO2 is a column, not ground-level concentration; PM2.5 is a ~40 km model."],
         example='urbanstudy year=2024 hot_months=4,5,6 --bbox 76.98 28.40 77.10 28.50 --name Gurugram'),
    dict(key="healthstudy", folder="areagiskit_health", name="AreaGIS Kit: Health", colour="#7b3294", glyph="H",
         description="Complete small-area disease study in one run: rates, SMR, smoothing, hot spots, Kulldorff scan, "
                     "Bayesian mapping, environmental drivers, space-time clusters and access.",
         about="Your areas and case counts (population from Meta 30 m maps if you have none): rates with 95% intervals, SMR, "
               "excess-risk probability, empirical Bayes, Moran's I, LISA and Gi* with FDR, Kulldorff scan, BYM2 Bayesian relative "
               "risk with satellite covariates, Spearman, Geodetector, Poisson regression and GWPR, age-standardised rates, "
               "space-time scans, Knox test, emerging hot spots, access (2SFCA), early warning (tested on simulations only), and a "
               "planted-cluster power test on your own map. Areas with fewer than 5 cases have counts and crude rates hidden.",
         tags="health,epidemiology,disease mapping,hot spot,cluster,Kulldorff,Moran,LISA,empirical Bayes,BYM2,report",
         validation=[
             "North Carolina SIDS: Moran's I, LISA and empirical Bayes identical to PySAL; BYM2 relative risks correlate 0.9997 "
             "with R-INLA; the scan finds the same 42-county cluster as R SpatialEpi.",
             "Planted-cluster test: real double-risk cluster found 34/40 times, false alarms 3/40.",
             "Limits: links with environment are associations, not causes; the early warning has not been tested on real outbreaks."],
         example='healthstudy areas=wards.gpkg id_field=ward_id cases_field=cases population_field=pop --name "My district"',
         needs_aoi=False),
]


def icon(path, colour, glyph):
    s = 4 * 64
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 8, s - 8, s - 8], radius=48, fill=colour)
    f = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(fp):
            f = ImageFont.truetype(fp, 160)
    d.text((s / 2, s / 2 + 4), glyph, fill="white", font=f, anchor="mm")
    im.resize((64, 64), Image.LANCZOS).save(path)


def readme(P):
    v = "\n".join(f"- {x}" for x in P["validation"])
    return f"""# {P['name']} (QGIS plugin)

{P['description']}

{COMMON_ABOUT}

## What it does
{P['about']}

## Validation
{v}

The confidence grade in every report is computed from these kinds of checks, not set by hand.
Every report lists its limitations.

## Install
QGIS 3.22 or newer: **Plugins → Manage and Install Plugins → Install from ZIP** → `{P['folder']}.zip`
(or search "{P['name']}" once it is in the official repository).
Then click the toolbar button, or **Processing Toolbox → {P['name']}**. Works in batch mode, Model Builder and `qgis_process`.

## Command line (without QGIS)
```
python -m {P['folder']}.cli {P['example']} --out out
python -m {P['folder']}.cli list
```

## Report
Every run writes `report.html`, `report.docx` and `report.pdf`: one-page summary, then study area, data, method,
results, validation, limitations, conclusions, references, Appendix A (proof: every satellite scene with a link to its
public record, and the exact command to rerun) and Appendix B (files).

## Data and licences
Code: GPL-3.0-or-later with additional terms (see LICENSE and NOTICE). Data keeps its own licence; see DATA_LICENSES.md. Every report carries the credits.

## Family
Part of AreaGIS Kit, a set of one-click study plugins by {AUTHOR}: Flood, LULC, Landslide, Crop, Urban and Health.
Each installs on its own; install any combination.
"""


def _no_percent(text):
    """The QGIS plugin site reads metadata.txt with Python's configparser, where a bare % is an error."""
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    return text.replace("%", " percent")


def metadata(P):
    return _no_percent(_metadata(P))


def _metadata(P):
    repo = f"{GH}/areagiskit"   # one repository for all six kits and the ArcGIS Pro toolbox
    return f"""[general]
name={P['name']}
qgisMinimumVersion=3.22
description={P['description']}
version={VERSION}
author={AUTHOR}
email=pradeepika.kaushik@gmail.com
about={COMMON_ABOUT}{P['about']}
tracker={repo}/issues
repository={repo}
homepage={repo}
tags={P['tags']}
category=Analysis
icon=icon.png
experimental=True
hasProcessingProvider=yes
deprecated=False
license=GPL-3.0-or-later
changelog={VERSION}: AreaGIS Kit release. Landslide hindcast; Sentinel-2 duplicate-tile fix; recovery test for landslide scars; privacy rule for small case counts; projection stops if its hindcast fails; every report opens with what it cannot tell you.
"""


def build(P):
    dst = os.path.join(DIST, P["folder"])
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "validation_floodbench", "tests", "build_plugins.py", ".*",
                                                            "README.md", "metadata.txt", "icon.png", "arcgis", "arcgis_tools.py"))
    with open(os.path.join(dst, "plugin_info.py"), "w") as fh:
        fh.write("# AreaGIS Kit\n# Copyright (C) 2026 Pradeepika Kaushik\n# SPDX-License-Identifier: GPL-3.0-or-later\n"
                 "# Additional terms under GPL section 7 (attribution and name): see NOTICE.\n")
        fh.write(f'"""Identity of this plugin (written by build_plugins.py)."""\n'
                 f'PLUGIN_NAME = "{P["name"]}"\nPROVIDER_ID = "{P["folder"]}"\nKIT_KEYS = ["{P["key"]}"]\n')
    # package docstrings / cli usage
    for f in ("__init__.py", "cli.py"):
        p = os.path.join(dst, f)
        t = open(p).read()
        t = re.sub(r'^"""[^\n]*', f'"""{P["name"]}: one-click, validated study for QGIS (and the command line)."""', t, 1) \
            if f == "__init__.py" else t.replace("areagiskit", P["folder"])
        open(p, "w").write(t)
    open(os.path.join(dst, "README.md"), "w").write(readme(P))
    open(os.path.join(dst, "metadata.txt"), "w").write(metadata(P))
    icon(os.path.join(dst, "icon.png"), P["colour"], P["glyph"])
    os.makedirs(OUT, exist_ok=True)
    z = os.path.join(OUT, P["folder"] + ".zip")
    if os.path.exists(z):
        os.remove(z)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dst):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, DIST))
    return z


ALL = dict(key=None, folder="areagiskit", name="AreaGIS Kit", colour="#0f3d5e", glyph="A",
           description="Six complete, self-checked GIS studies in one plugin: Flood, LULC, Landslide, Crop, Urban and Health. "
                       "Pick an area, press Run, get validated maps and a Word/PDF/HTML report.")


def build_all():
    """One QGIS plugin with all six kits: one toolbar button with a menu of kits, six Processing tools."""
    P = dict(ALL)
    P["about"] = " ".join(f"{K['name'].split(': ')[1]}: {K['about']}" for K in PLUGINS)
    P["validation"] = [f"{K['name'].split(': ')[1]}: {v}" for K in PLUGINS for v in K["validation"]]
    P["tags"] = ",".join(dict.fromkeys(t.strip() for K in PLUGINS for t in K["tags"].split(",")))
    P["example"] = PLUGINS[0]["example"]
    dst = os.path.join(DIST, P["folder"])
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "validation_floodbench", "tests", "build_plugins.py", ".*",
                                                            "README.md", "metadata.txt", "icon.png", "arcgis", "arcgis_tools.py"))
    with open(os.path.join(dst, "plugin_info.py"), "w") as fh:
        fh.write("# AreaGIS Kit\n# Copyright (C) 2026 Pradeepika Kaushik\n# SPDX-License-Identifier: GPL-3.0-or-later\n"
                 "# Additional terms under GPL section 7 (attribution and name): see NOTICE.\n")
        fh.write('"""Identity of this plugin (written by build_plugins.py)."""\n'
                 'PLUGIN_NAME = "AreaGIS Kit"\nPROVIDER_ID = "areagiskit"\nKIT_KEYS = None\n')
    readme_text = readme(P).replace("## Family\nPart of AreaGIS Kit, a set of one-click study plugins by", "## Kits\nAll six kits by") \
                           .replace("Each installs on its own; install any combination.", "One toolbar button opens a menu of the six kits.")
    open(os.path.join(dst, "README.md"), "w").write(readme_text)
    open(os.path.join(dst, "metadata.txt"), "w").write(metadata(P))
    icon(os.path.join(dst, "icon.png"), P["colour"], P["glyph"])
    os.makedirs(os.path.join(dst, "icons"), exist_ok=True)
    for K in PLUGINS:
        icon(os.path.join(dst, "icons", f"{K['key']}.png"), K["colour"], K["glyph"])
    os.makedirs(OUT, exist_ok=True)
    z = os.path.join(OUT, "areagiskit.zip")
    if os.path.exists(z):
        os.remove(z)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dst):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, DIST))
    return z


ARCGIS_README = """AreaGIS Kit for ArcGIS Pro (beta) {version}

Complete, self-checked geospatial studies of any area: Flood, LULC, Landslide, Crop, Urban and Health.
Same engine and the same reports as the QGIS plugins.

INSTALL
1. Unzip this folder anywhere (keep 'AreaGIS Kit.pyt' next to the 'areagiskit' folder).
2. ArcGIS Pro: Catalog pane > Toolboxes > right-click > Add Toolbox > choose 'AreaGIS Kit.pyt'.
3. Open a tool, choose an area (polygon layer or extent), set the options, choose an output folder, Run.

GDAL
The kits read satellite data with GDAL's Python package. If a tool shows a GDAL warning:
Project > Package Manager > Environment Manager > Clone arcgispro-py3 > activate the clone >
Add Packages > search "gdal" > Install > restart ArcGIS Pro.

WHAT YOU GET
report.html, report.docx and report.pdf in the output folder, GeoTIFF and CSV results, and the result layers
added to the active map. (Colour styles are QGIS .qml files; in ArcGIS Pro set symbology as you like.)

STATUS
Beta. The toolbox has been tested with a stand-in for arcpy and a full Health run; it has not yet been run inside
ArcGIS Pro itself. Please report problems with the tool name, the message shown and your ArcGIS Pro version.

Author: {author}. Code: GPL-3.0-or-later with additional terms in areagiskit/NOTICE. Data licences: see areagiskit/DATA_LICENSES.md.
"""


def build_arcgis():
    """One zip for ArcGIS Pro: the Python toolbox next to a copy of the engine."""
    root = os.path.join(DIST, "AreaGIS_Kit_for_ArcGIS_Pro")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    shutil.copytree(SRC, os.path.join(root, "areagiskit"), ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "validation_floodbench", "tests", "build_plugins.py", ".*", "arcgis", "metadata.txt", "icon.png"))
    shutil.copy(os.path.join(SRC, "arcgis", "AreaGIS Kit.pyt"), root)
    open(os.path.join(root, "README_ArcGIS.txt"), "w").write(ARCGIS_README.format(version=VERSION, author=AUTHOR))
    os.makedirs(OUT, exist_ok=True)
    z = os.path.join(OUT, "AreaGIS_Kit_for_ArcGIS_Pro.zip")
    if os.path.exists(z):
        os.remove(z)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for r, _, files in os.walk(root):
            for f in files:
                full = os.path.join(r, f)
                zf.write(full, os.path.relpath(full, DIST))
    return z


if __name__ == "__main__":
    only = sys.argv[1:]
    for P in PLUGINS:
        if not only or P["key"] in only:
            z = build(P)
            print(f"{P['name']:22s} {os.path.getsize(z) / 1e6:5.2f} MB  {z}")
    if not only or "all" in only:
        z = build_all()
        print(f"{'AreaGIS Kit (all six)':22s} {os.path.getsize(z) / 1e6:5.2f} MB  {z}")
    if not only or "arcgis" in only:
        z = build_arcgis()
        print(f"{'ArcGIS Pro toolbox':22s} {os.path.getsize(z) / 1e6:5.2f} MB  {z}")
