# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""One QGIS Processing algorithm per kit, built from the kit's PARAMS list."""

import datetime as dt
import json
import os
import tempfile
import time

from qgis.core import (
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsGeometry, QgsProcessing,
    QgsProcessingAlgorithm, QgsProcessingException, QgsProcessingOutputFile, QgsProcessingOutputFolder,
    QgsProcessingParameterBoolean, QgsProcessingParameterEnum, QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterField, QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterNumber, QgsProcessingParameterString,
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon

from .core.aoi import AOI
from .core.data import KitError

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


def _qenum(old_owner, old_name, new_path):
    """QGIS 3.22-3.x use QgsProcessing.TypeVectorPolygon etc.; QGIS 3.36+/4 prefer Qgis.ProcessingSourceType.VectorPolygon.
    Return whichever exists."""
    v = getattr(old_owner, old_name, None)
    if v is not None:
        return v
    from qgis.core import Qgis
    obj = Qgis
    for part in new_path.split("."):
        obj = getattr(obj, part)
    return obj


T_POLY = _qenum(QgsProcessing, "TypeVectorPolygon", "ProcessingSourceType.VectorPolygon")
T_ANY = _qenum(QgsProcessing, "TypeVectorAnyGeometry", "ProcessingSourceType.VectorAnyGeometry")
N_INT = _qenum(QgsProcessingParameterNumber, "Integer", "ProcessingNumberParameterType.Integer")
N_DBL = _qenum(QgsProcessingParameterNumber, "Double", "ProcessingNumberParameterType.Double")
G_POLY = _qenum(QgsWkbTypes, "PolygonGeometry", "GeometryType.Polygon")
ICON = os.path.join(os.path.dirname(__file__), "icon.png")


def _tr(s):
    return QCoreApplication.translate("AreaGIS Kit", s)


class KitAlgorithm(QgsProcessingAlgorithm):
    AREA, EXTENT, NAME, OUT, OPEN = "AREA", "EXTENT", "AREA_NAME", "OUTPUT_FOLDER", "OPEN_REPORT"

    def __init__(self, kit_cls):
        super().__init__()
        self.kit = kit_cls
        self._result = None

    # -------------------------------------------------------------- metadata
    def createInstance(self):
        return KitAlgorithm(self.kit)

    def name(self):
        return self.kit.KEY

    def displayName(self):
        return f"{self.kit.NAME}: {self.kit.TAGLINE}"

    def group(self):
        return "Kits"

    def groupId(self):
        return "kits"

    def icon(self):
        return QIcon(ICON)

    def shortHelpString(self):
        p = "".join(f"<li><b>{x['label']}</b>{(': ' + x['help']) if x.get('help') else ''}</li>" for x in self.kit.PARAMS)
        area = ("<p><b>Area</b>: choose a polygon layer (only selected features are used if you tick that box) "
                "or draw/choose an extent. Polygon layers clip results to the exact boundary.</p>") if self.kit.NEEDS_AOI else ""
        return (f"<p>{self.kit.DESCRIPTION}</p>{area}<ul>{p}</ul>"
                "<p>No account or API key. Data is streamed for your area only. A report opens in your browser "
                "and the maps are added to the project.</p>")

    # -------------------------------------------------------------- parameters
    def initAlgorithm(self, config=None):
        if self.kit.NEEDS_AOI:
            self.addParameter(QgsProcessingParameterFeatureSource(
                self.AREA, _tr("Area: polygon layer (optional if an extent is given)"),
                [T_POLY], optional=True))
            self.addParameter(QgsProcessingParameterExtent(
                self.EXTENT, _tr("…or area: extent"), optional=True))
            self.addParameter(QgsProcessingParameterString(
                self.NAME, _tr("Name for the report"), defaultValue="", optional=True))
        for p in self.kit.PARAMS:
            t, n, lab, d = p["type"], p["name"], p["label"], p.get("default")
            opt = p.get("optional", False)
            if t == "year" and p.get("options"):
                opts = [str(o) for o in p["options"]]
                self.addParameter(QgsProcessingParameterEnum(n, lab, opts, defaultValue=opts.index(str(d)) if str(d) in opts else 0))
            elif t in ("int", "year"):
                self.addParameter(QgsProcessingParameterNumber(n, lab, N_INT, defaultValue=d, optional=opt))
            elif t == "float":
                self.addParameter(QgsProcessingParameterNumber(n, lab, N_DBL, defaultValue=d, optional=opt))
            elif t == "bool":
                self.addParameter(QgsProcessingParameterBoolean(n, lab, defaultValue=bool(d)))
            elif t == "enum":
                opts = list(p["options"])
                self.addParameter(QgsProcessingParameterEnum(n, lab, opts, defaultValue=opts.index(d) if d in opts else 0))
            elif t == "points":
                self.addParameter(QgsProcessingParameterFeatureSource(n, lab, [T_ANY], optional=opt))
            elif t == "polygons":
                self.addParameter(QgsProcessingParameterFeatureSource(n, lab, [T_POLY], optional=opt))
            elif t == "field":
                self.addParameter(QgsProcessingParameterField(n, lab, parentLayerParameterName=p["parent"], optional=opt))
            elif t == "file":
                self.addParameter(QgsProcessingParameterFile(n, lab, extension="csv", optional=opt))
            else:  # text, date
                self.addParameter(QgsProcessingParameterString(n, lab, defaultValue=d, optional=opt or d in ("", None)))
        self.addParameter(QgsProcessingParameterBoolean(self.OPEN, _tr("Open the report when finished"), defaultValue=True))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUT, _tr("Save results in")))
        self.addOutput(QgsProcessingOutputFile("REPORT", _tr("Report (HTML)")))
        self.addOutput(QgsProcessingOutputFile("REPORT_DOCX", _tr("Report (Word)")))
        self.addOutput(QgsProcessingOutputFile("REPORT_PDF", _tr("Report (PDF)")))
        self.addOutput(QgsProcessingOutputFolder("FOLDER", _tr("Results folder")))

    # -------------------------------------------------------------- helpers
    def _source_to_geojson(self, source, path):
        """Write a feature source to GeoJSON in EPSG:4326 (keeps attributes)."""
        ct = None
        if source.sourceCrs().isValid() and source.sourceCrs() != WGS84:
            ct = QgsCoordinateTransform(source.sourceCrs(), WGS84, QgsProject.instance())
        names = [f.name() for f in source.fields()]
        feats = []
        for f in source.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isEmpty():
                continue
            if ct:
                g.transform(ct)
            props = {}
            for n in names:
                v = f[n]
                props[n] = v if isinstance(v, (int, float, str)) or v is None else str(v)
            feats.append({"type": "Feature", "geometry": json.loads(g.asJson(7)), "properties": props})
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh)
        return path

    def _aoi(self, parameters, context):
        name = self.parameterAsString(parameters, self.NAME, context).strip()
        src = self.parameterAsSource(parameters, self.AREA, context)
        if src is not None and src.featureCount() > 0:
            geoms = [QgsGeometry(f.geometry()) for f in src.getFeatures() if f.hasGeometry()]
            g = QgsGeometry.unaryUnion(geoms)
            if g.type() != G_POLY:
                raise QgsProcessingException("The area layer has no polygons.")
            if src.sourceCrs() != WGS84:
                g.transform(QgsCoordinateTransform(src.sourceCrs(), WGS84, QgsProject.instance()))
            return AOI(wkt=g.asWkt(7), name=name or src.sourceName())
        ext = self.parameterAsExtent(parameters, self.EXTENT, context, WGS84)
        if ext is None or ext.isEmpty():
            raise QgsProcessingException("Choose an area: a polygon layer or an extent.")
        return AOI(bbox=[ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum()], name=name or "Selected area")

    # -------------------------------------------------------------- run
    def processAlgorithm(self, parameters, context, feedback):
        out_root = self.parameterAsString(parameters, self.OUT, context)
        tmp = tempfile.mkdtemp(prefix="areagiskit_")
        try:
            aoi = self._aoi(parameters, context) if self.kit.NEEDS_AOI else None
            params = {}
            for p in self.kit.PARAMS:
                t, n = p["type"], p["name"]
                if t == "year" and p.get("options"):
                    params[n] = p["options"][self.parameterAsEnum(parameters, n, context)]
                elif t == "enum":
                    params[n] = p["options"][self.parameterAsEnum(parameters, n, context)]
                elif t in ("int", "year"):
                    params[n] = self.parameterAsInt(parameters, n, context)
                elif t == "float":
                    params[n] = self.parameterAsDouble(parameters, n, context)
                elif t == "bool":
                    params[n] = self.parameterAsBool(parameters, n, context)
                elif t in ("points", "polygons"):
                    s = self.parameterAsSource(parameters, n, context)
                    params[n] = self._source_to_geojson(s, os.path.join(tmp, f"{n}.geojson")) if s is not None else None
                elif t == "field":
                    get = getattr(self, "parameterAsStrings", None) or self.parameterAsFields   # 3.32+ / older QGIS
                    fl = get(parameters, n, context)
                    params[n] = fl[0] if fl else None
                elif t == "file":
                    params[n] = self.parameterAsFile(parameters, n, context) or None
                else:
                    params[n] = self.parameterAsString(parameters, n, context)
            label = aoi.name if aoi else os.path.splitext(os.path.basename(params.get("areas") or "data"))[0]
            safe = "".join(ch if ch.isalnum() else "_" for ch in label)[:40]
            out_dir = os.path.join(out_root, f"{self.kit.KEY}_{safe}_{time.strftime('%Y%m%d_%H%M%S')}")

            def prog(p, m):
                feedback.setProgress(p)
                feedback.setProgressText(m)

            from .core import report as _rep
            _rep.DEFER_PDF = True
            res = self.kit(progress=prog, cancelled=feedback.isCanceled).run(aoi, params, out_dir)
        except KitError as e:
            raise QgsProcessingException(str(e))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)      # copies of the input layers; the run has finished with them
        feedback.pushInfo(res.get("summary", ""))
        if res.get("verdict"):
            feedback.pushInfo("Confidence: " + res["verdict"]["level"])
        feedback.pushInfo("Report: " + res["report"])
        self._result = res
        self._open = self.parameterAsBool(parameters, self.OPEN, context)
        docs = res.get("documents", {})
        for k in ("docx", "pdf"):
            if docs.get(k):
                feedback.pushInfo(f"{k.upper()}: {docs[k]}")
        return {"REPORT": res["report"], "REPORT_DOCX": docs.get("docx"), "REPORT_PDF": docs.get("pdf"),
                "FOLDER": os.path.dirname(res["report"])}

    def postProcessAlgorithm(self, context, feedback):
        res = self._result
        if not res:
            return {}
        from .core.report import write_deferred_pdf
        if res.get("documents"):
            write_deferred_pdf(res["documents"])
            if res["documents"].get("pdf"):
                feedback.pushInfo("PDF: " + res["documents"]["pdf"])
        prj = context.project() or QgsProject.instance()
        root = prj.layerTreeRoot()
        grp = root.insertGroup(0, f"{self.kit.NAME} · {os.path.basename(os.path.dirname(res['report']))}")
        for spec in res.get("layers", []):
            lyr = (QgsVectorLayer(spec["path"], spec["name"], "ogr") if spec["kind"] == "vector"
                   else QgsRasterLayer(spec["path"], spec["name"]))
            if not lyr.isValid():
                continue
            lyr.loadDefaultStyle()
            prj.addMapLayer(lyr, False)
            node = grp.addLayer(lyr)
            if spec.get("hidden"):
                node.setItemVisibilityChecked(False)
        if getattr(self, "_open", True):
            QDesktopServices.openUrl(QUrl.fromLocalFile(res["report"]))
        docs = res.get("documents", {})
        return {"REPORT": res["report"], "REPORT_DOCX": docs.get("docx"), "REPORT_PDF": docs.get("pdf"),
                "FOLDER": os.path.dirname(res["report"])}
