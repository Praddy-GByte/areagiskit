# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""ArcGIS Pro front end: one geoprocessing tool per kit, built from each kit's PARAMS.

The kits themselves need only numpy and GDAL. ArcGIS Pro's Python has numpy; GDAL is checked at run time and the tool
explains how to add it if it is missing. Reports are written as HTML, Word and PDF (the PDF uses matplotlib when Qt is
not available, which is the normal case inside ArcGIS Pro).
"""

import json
import os
import tempfile
import time

import arcpy

from .core.aoi import AOI
from .core.data import KitError
from .kits import all_kits

WGS84 = arcpy.SpatialReference(4326)
AREA, EXTENT, NAME, OPEN, OUT = "area", "extent", "area_name", "open_report", "output_folder"


def _gdal_ok():
    try:
        from osgeo import gdal  # noqa: F401
        return True
    except Exception:
        return False


GDAL_HELP = ("This tool needs the GDAL Python package, which is not in this ArcGIS Pro Python environment. "
             "In ArcGIS Pro: Project > Package Manager > Environment Manager > clone arcgispro-py3, activate the clone, "
             "then Add Packages > search 'gdal' > Install, and restart ArcGIS Pro.")


def _param(display, name, datatype, required=False, default=None, direction="Input", category=None):
    p = arcpy.Parameter(displayName=display, name=name, datatype=datatype,
                        parameterType="Required" if required else "Optional", direction=direction, category=category)
    if default is not None:
        p.value = default
    return p


def _layer_to_geojson(value, path):
    """Copy a feature layer (any format ArcGIS can read) to GeoJSON in WGS84, keeping attributes."""
    fields = [f.name for f in arcpy.ListFields(value) if f.type not in ("Geometry", "Blob", "Raster")]
    feats = []
    with arcpy.da.SearchCursor(value, ["SHAPE@"] + fields) as cur:
        for row in cur:
            g = row[0]
            if g is None:
                continue
            g = g.projectAs(WGS84)
            props = {}
            for n, v in zip(fields, row[1:]):
                props[n] = v if isinstance(v, (int, float, str)) or v is None else str(v)
            feats.append({"type": "Feature", "geometry": g.__geo_interface__, "properties": props})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return path


def _area_from_layer(value, name):
    geoms = []
    with arcpy.da.SearchCursor(value, ["SHAPE@"]) as cur:
        for (g,) in cur:
            if g is not None:
                geoms.append(g.projectAs(WGS84))
    if not geoms:
        raise KitError("The area layer has no polygons.")
    u = geoms[0]
    for g in geoms[1:]:
        u = u.union(g)
    return AOI(wkt=u.WKT, name=name)


def _area_from_extent(ext, name):
    e = ext.projectAs(WGS84) if getattr(ext, "spatialReference", None) else ext
    return AOI(bbox=[e.XMin, e.YMin, e.XMax, e.YMax], name=name)


def make_tool(kit_cls):
    """Return a Python-toolbox tool class for one kit."""

    class Tool(object):
        KIT = kit_cls

        def __init__(self):
            self.label = kit_cls.NAME
            self.description = f"{kit_cls.TAGLINE}. {kit_cls.DESCRIPTION}"
            self.canRunInBackground = False
            self.category = "Kits"

        def getParameterInfo(self):
            P = []
            if kit_cls.NEEDS_AOI:
                a = _param("Area: polygon layer (optional if an extent is given)", AREA, "GPFeatureLayer")
                a.filter.list = ["Polygon"]
                P += [a, _param("…or area: extent", EXTENT, "GPExtent"), _param("Name for the report", NAME, "GPString")]
            index = {}
            for p in kit_cls.PARAMS:
                t, n, lab, d = p["type"], p["name"], p["label"], p.get("default")
                req = False
                if t == "year" and p.get("options"):
                    q = _param(lab, n, "GPString", default=str(d))
                    q.filter.type = "ValueList"; q.filter.list = [str(o) for o in p["options"]]
                elif t in ("int", "year"):
                    q = _param(lab, n, "GPLong", default=d)
                elif t == "float":
                    q = _param(lab, n, "GPDouble", default=d)
                elif t == "bool":
                    q = _param(lab, n, "GPBoolean", default=bool(d))
                elif t == "enum":
                    q = _param(lab, n, "GPString", default=d)
                    q.filter.type = "ValueList"; q.filter.list = list(p["options"])
                elif t == "points":
                    q = _param(lab, n, "GPFeatureLayer")
                elif t == "polygons":
                    q = _param(lab, n, "GPFeatureLayer")
                    q.filter.list = ["Polygon"]
                    req = not p.get("optional", False) and not kit_cls.NEEDS_AOI and n == "areas"
                    if req:
                        q.parameterType = "Required"
                elif t == "field":
                    q = _param(lab, n, "Field")
                    if p.get("parent") in index:
                        q.parameterDependencies = [index[p["parent"]]]
                elif t == "file":
                    q = _param(lab, n, "DEFile")
                    q.filter.list = ["csv"]
                else:
                    q = _param(lab, n, "GPString", default=d if d not in (None, "") else None)
                index[n] = len(P)
                P.append(q)
            P.append(_param("Open the report when finished", OPEN, "GPBoolean", default=True))
            P.append(_param("Save results in", OUT, "DEFolder", required=True))
            return P

        def isLicensed(self):
            return True

        def updateParameters(self, parameters):
            return

        def updateMessages(self, parameters):
            if not _gdal_ok():
                parameters[-1].setWarningMessage(GDAL_HELP)
            return

        def execute(self, parameters, messages):
            if not _gdal_ok():
                raise arcpy.ExecuteError(GDAL_HELP)
            V = {p.name: p for p in parameters}
            tmp = tempfile.mkdtemp(prefix="areagiskit_")
            try:
                aoi = None
                if kit_cls.NEEDS_AOI:
                    name = (V[NAME].valueAsText or "").strip()
                    if V[AREA].value:
                        aoi = _area_from_layer(V[AREA].value, name or str(V[AREA].valueAsText))
                    elif V[EXTENT].value:
                        aoi = _area_from_extent(V[EXTENT].value, name or "Selected area")
                    else:
                        raise KitError("Choose an area: a polygon layer or an extent.")
                params = {}
                src_names = {}
                for p in kit_cls.PARAMS:
                    t, n = p["type"], p["name"]
                    q = V[n]
                    if t in ("points", "polygons"):
                        src_names[n] = os.path.splitext(os.path.basename(str(q.valueAsText or n)))[0]
                        params[n] = _layer_to_geojson(q.value, os.path.join(tmp, f"{n}.geojson")) if q.value else None
                    elif t in ("int", "year") and not p.get("options"):
                        params[n] = int(q.value) if q.value is not None else p.get("default")
                    elif t == "year":
                        params[n] = int(q.valueAsText) if q.valueAsText else p.get("default")
                    elif t == "float":
                        params[n] = float(q.value) if q.value is not None else p.get("default")
                    elif t == "bool":
                        params[n] = bool(q.value)
                    elif t == "file":
                        params[n] = q.valueAsText or None
                    else:
                        params[n] = q.valueAsText if q.valueAsText not in (None, "") else (p.get("default") if t != "field" else None)
                out_root = V[OUT].valueAsText
                label = aoi.name if aoi else src_names.get("areas", "data")
                safe = "".join(ch if ch.isalnum() else "_" for ch in label)[:40]
                out_dir = os.path.join(out_root, f"{kit_cls.KEY}_{safe}_{time.strftime('%Y%m%d_%H%M%S')}")
                arcpy.SetProgressor("step", kit_cls.NAME, 0, 100, 1)

                def prog(pct, msg):
                    arcpy.SetProgressorLabel(msg)
                    arcpy.SetProgressorPosition(int(pct))
                    messages.addMessage(f"[{int(pct):3d}%] {msg}")

                cancelled = (lambda: bool(getattr(arcpy.env, "isCancelled", False)))
                res = kit_cls(progress=prog, cancelled=cancelled).run(aoi, params, out_dir)
            except KitError as e:
                messages.addErrorMessage(str(e))
                raise arcpy.ExecuteError(str(e))
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            messages.addMessage(res.get("summary", ""))
            if res.get("verdict"):
                messages.addMessage("Confidence: " + res["verdict"]["level"])
            messages.addMessage("Report: " + res["report"])
            for k, v in (res.get("documents") or {}).items():
                if isinstance(v, str):
                    messages.addMessage(f"{k.upper()}: {v}")
            try:
                m = arcpy.mp.ArcGISProject("CURRENT").activeMap
                if m is not None:
                    for spec in reversed(res.get("layers", [])):
                        try:
                            m.addDataFromPath(spec["path"])
                        except Exception:
                            pass
            except Exception:
                messages.addMessage("Results were not added to a map (no active map). They are in: " + os.path.dirname(res["report"]))
            if V[OPEN].value and os.name == "nt":
                try:
                    os.startfile(res["report"])
                except Exception:
                    pass
            return

    Tool.__name__ = f"{kit_cls.__name__}Tool"
    return Tool


def tools(keys=None):
    return [make_tool(k) for k in all_kits() if keys is None or k.KEY in keys]
