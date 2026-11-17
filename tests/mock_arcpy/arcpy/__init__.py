# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""A small stand-in for arcpy, enough to exercise the ArcGIS Pro toolbox without ArcGIS. Geometry work is done with OGR."""
import json
from osgeo import ogr, osr

from . import da, mp  # noqa: F401


class ExecuteError(Exception):
    pass


class _Filter:
    def __init__(self):
        self.type = None
        self.list = []


class Parameter:
    def __init__(self, displayName="", name="", datatype="GPString", parameterType="Optional", direction="Input", category=None):
        self.displayName, self.name, self.datatype = displayName, name, datatype
        self.parameterType, self.direction, self.category = parameterType, direction, category
        self.filter = _Filter()
        self.value = None
        self.parameterDependencies = []
        self.warning = None

    @property
    def valueAsText(self):
        return None if self.value is None else str(self.value)

    def setWarningMessage(self, m):
        self.warning = m


class SpatialReference:
    def __init__(self, code):
        self.factoryCode = code


class Extent:
    def __init__(self, xmin, ymin, xmax, ymax, sr=None):
        self.XMin, self.YMin, self.XMax, self.YMax, self.spatialReference = xmin, ymin, xmax, ymax, sr

    def projectAs(self, sr):
        return self


class Geometry:
    def __init__(self, ogr_geom):
        self.g = ogr_geom

    def projectAs(self, sr):
        g = self.g.Clone()
        src = g.GetSpatialReference()
        if src is not None:
            dst = osr.SpatialReference(); dst.ImportFromEPSG(sr.factoryCode)
            dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            g.Transform(osr.CoordinateTransformation(src, dst))
        return Geometry(g)

    def union(self, other):
        return Geometry(self.g.Union(other.g))

    @property
    def WKT(self):
        return self.g.ExportToWkt()

    @property
    def __geo_interface__(self):
        return json.loads(self.g.ExportToJson())


class _Field:
    def __init__(self, name, ftype):
        self.name, self.type = name, ftype


def ListFields(path):
    ds = ogr.Open(str(path)); lyr = ds.GetLayer(0); d = lyr.GetLayerDefn()
    return [_Field("Shape", "Geometry")] + [_Field(d.GetFieldDefn(i).GetName(), "String") for i in range(d.GetFieldCount())]


class env:
    isCancelled = False


LOG = []


def SetProgressor(*a, **k):
    LOG.append(("progressor", a))


def SetProgressorLabel(m):
    LOG.append(("label", m))


def SetProgressorPosition(p):
    LOG.append(("pos", p))
