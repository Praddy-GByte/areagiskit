# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Build the Python package (pip install areagiskit) from this repository.

The repository is laid out as a QGIS plugin, so this script copies the engine into a clean staging folder with a
pyproject.toml, leaves out the QGIS and ArcGIS front ends, and runs `python -m build`. The wheel and sdist land in
dist_pypi/.

    python build_pypi.py
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "_pypi_stage")
OUT = os.path.join(HERE, "dist_pypi")


def version():
    text = open(os.path.join(HERE, "build_plugins.py"), encoding="utf-8").read()
    return re.search(r'^VERSION = "([^"]+)"', text, re.M).group(1)


PYPROJECT = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "areagiskit"
version = "{version}"
description = "One-click geospatial studies of any area: Flood, LULC, Landslide, Crop, Urban and Health (236 analyses), with validation and Word/PDF/HTML reports."
readme = "README.md"
requires-python = ">=3.9"
license = "GPL-3.0-or-later"
license-files = ["LICENSE", "NOTICE", "DATA_LICENSES.md"]
authors = [{{name = "Pradeepika Kaushik"}}]
keywords = ["GIS", "remote sensing", "Sentinel-1", "Sentinel-2", "flood mapping", "land use land cover",
            "landslide susceptibility", "agriculture", "urban heat island", "spatial epidemiology", "QGIS", "ArcGIS Pro"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: GIS",
]
dependencies = ["numpy>=1.21", "matplotlib>=3.5"]

[project.optional-dependencies]
fast = ["scipy>=1.7"]

[project.urls]
Homepage = "https://github.com/Praddy-GByte/areagiskit"
Source = "https://github.com/Praddy-GByte/areagiskit"
Issues = "https://github.com/Praddy-GByte/areagiskit/issues"
Releases = "https://github.com/Praddy-GByte/areagiskit/releases"
DOI = "https://doi.org/10.5281/zenodo.24326878"

[project.scripts]
areagiskit = "areagiskit.cli:main"

[tool.setuptools]
packages = {{find = {{include = ["areagiskit*"]}}}}
include-package-data = true

[tool.setuptools.package-data]
"*" = ["*.geojson", "*.npz", "*.txt", "*.md", "*.qml", "NOTICE", "LICENSE"]
"""

README = """# AreaGIS Kit

One area. Six studies. 236 analyses.

AreaGIS Kit runs a complete geospatial study of any area in one run: **Flood, LULC, Landslide, Crop, Urban heat and
Health**. It fetches the open data it needs for the area (no account, no manual download), runs the standard analyses
for the field, compares its results with independent data where it can, grades its confidence (High / Medium / Low)
and writes a report in Word, PDF and HTML that also states what the results cannot tell you.

This package is the engine and command line. The same studies are available as a QGIS plugin and an ArcGIS Pro
toolbox: see https://github.com/Praddy-GByte/areagiskit/releases

**Status: beta.** Please report problems at https://github.com/Praddy-GByte/areagiskit/issues

## Install

AreaGIS Kit needs GDAL's Python bindings, which pip cannot always install on its own. The easiest way:

```
conda create -n areagis -c conda-forge python=3.12 gdal numpy matplotlib scipy
conda activate areagis
pip install areagiskit
```

If GDAL is already on your system (for example the Python that ships with QGIS, or `apt install python3-gdal`),
`pip install areagiskit` is enough.

## Use

```
areagiskit list
areagiskit floodstudy --bbox 67.6 26.6 67.9 26.85 --name "Dadu, Sindh" event_date=2022-09-03 --out out/
areagiskit landslide  --bbox 76.08 11.43 76.22 11.56 --name "Wayanad" event_date=2024-07-30 --out out/
areagiskit healthstudy areas=wards.gpkg id_field=ward_id cases_field=cases population_field=pop --out out/
```

Every run writes `report.html`, `report.docx` and `report.pdf`, plus GeoTIFF and CSV results, in the output folder.

From Python:

```python
from areagiskit.core.aoi import AOI
from areagiskit.kits import get

kit = get("floodstudy")()
result = kit.run(AOI(bbox=[67.6, 26.6, 67.9, 26.85], name="Dadu"), {{"event_date": "2022-09-03"}}, "out/")
print(result["summary"], result["verdict"]["level"])
```

## Cite

Kaushik, P. (2026). AreaGIS Kit: one-click geospatial studies for QGIS and ArcGIS Pro [Software]. Zenodo.
https://doi.org/10.5281/zenodo.24326878

## Licence

Code: GPL-3.0-or-later with additional terms in NOTICE (keep the author credit, mark changed versions, and do not use
the name "AreaGIS Kit" or its logo for a changed version). Data keeps its own licence; some sources are for
non-commercial use only (see DATA_LICENSES.md).

Author: Pradeepika Kaushik
"""

TOP_ONLY = {"tests", "validation_floodbench", "arcgis", "assets", "icons", "_pypi_stage", "dist_pypi", "README.md",
            "CITATION.cff", "metadata.txt", "icon.png", "arcgis_tools.py", "build_plugins.py", "build_pypi.py"}
EVERYWHERE = shutil.ignore_patterns("__pycache__", "*.pyc", ".*", "qgis_*.py")


def skip(folder, names):
    """Leave out the QGIS/ArcGIS front ends and repository files; keep data folders inside the engine (core/assets)."""
    out = set(EVERYWHERE(folder, names))
    if os.path.abspath(folder) == HERE:
        out |= {n for n in names if n in TOP_ONLY}
    return out


def main():
    v = version()
    shutil.rmtree(STAGE, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(STAGE)
    shutil.copytree(HERE, os.path.join(STAGE, "areagiskit"), ignore=skip)
    for f in ("LICENSE", "NOTICE", "DATA_LICENSES.md"):
        shutil.copy(os.path.join(HERE, f), STAGE)
    open(os.path.join(STAGE, "pyproject.toml"), "w", encoding="utf-8").write(PYPROJECT.format(version=v))
    open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8").write(README.format())
    subprocess.check_call([sys.executable, "-m", "build", "--outdir", OUT, STAGE])
    shutil.rmtree(STAGE, ignore_errors=True)
    print("built:", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
