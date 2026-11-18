# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Command line for every kit (works without QGIS).

  python -m areagiskit.cli list
  python -m areagiskit.cli land  --bbox 76.95 28.38 77.15 28.52 --name Gurugram year_from=2017 year_to=2023 --out out/
  python -m areagiskit.cli flood --bbox 67.6 26.6 67.9 26.85 event_date=2022-09-03 --out out/
"""

import argparse
import sys

GDAL_HELP = """AreaGIS Kit needs GDAL's Python bindings (the 'osgeo' module), which are not installed in this Python.

Easiest fix (conda):
    conda create -n areagis -c conda-forge python=3.12 gdal numpy matplotlib scipy
    conda activate areagis
    pip install areagiskit

Or use the Python that ships with QGIS, or install GDAL for your system (e.g. apt install python3-gdal).
More help: https://github.com/Praddy-GByte/areagiskit"""

try:
    from .core.aoi import AOI
    from .core.data import KitError
    from .kits import all_kits, get
except ModuleNotFoundError as exc:          # GDAL missing: explain instead of printing a traceback
    if exc.name != "osgeo":
        raise
    AOI = KitError = all_kits = get = None


def _utf8_console():
    """Windows consoles default to a code page that cannot print °, ² or ₂; never crash on output."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv=None):
    _utf8_console()
    if AOI is None:
        print(GDAL_HELP, file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="areagiskit")
    ap.add_argument("kit")
    ap.add_argument("params", nargs="*", help="name=value")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--wkt")
    ap.add_argument("--name", default="Area of interest")
    ap.add_argument("--out", default="areagiskit_out")
    a = ap.parse_intermixed_args(argv)
    if a.kit == "list":
        for k in all_kits():
            print(f"{k.KEY:8s} {k.NAME:10s} {k.TAGLINE}")
            for p in k.PARAMS:
                print(f"           {p['name']}={p.get('default')}  ({p['type']}) {p['label']}")
        return 0
    K = get(a.kit)
    params = {p["name"]: p.get("default") for p in K.PARAMS}
    for kv in a.params:
        k, v = kv.split("=", 1)
        params[k] = v
    try:
        aoi = AOI(wkt=a.wkt, bbox=a.bbox, name=a.name) if (a.wkt or a.bbox) else None
        r = K(progress=lambda p, m: print(f"[{p:3d}%] {m}", flush=True)).run(aoi, params, a.out)
    except KitError as e:
        print("Error:", e)
        return 1
    print("Confidence:", r["verdict"]["level"] if r.get("verdict") else "n/a")
    print("Summary:", r.get("summary", ""))
    print("Report:", r["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
