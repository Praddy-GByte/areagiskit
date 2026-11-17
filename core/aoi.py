# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Area of interest and the common analysis grid.

Every kit works on one grid: a local Lambert azimuthal equal-area projection
centred on the area, so a pixel is the same size everywhere and km² figures
are true areas. Results are clipped to the polygon itself, not its bounding box.
"""

import math

import numpy as np
from osgeo import gdal, ogr, osr

from .data import KitError

gdal.UseExceptions()
ogr.UseExceptions()


def _srs_wgs84():
    s = osr.SpatialReference()
    s.ImportFromEPSG(4326)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


class AOI:
    def __init__(self, wkt=None, bbox=None, name="Area of interest"):
        if wkt is None and bbox is None:
            raise KitError("No area selected.")
        if wkt is None:
            x0, y0, x1, y1 = bbox
            wkt = f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"
        g = ogr.CreateGeometryFromWkt(wkt)
        if g is None or g.IsEmpty():
            raise KitError("The selected area is empty or invalid.")
        if not g.IsValid():
            g = g.Buffer(0)
        self.geom = g
        self.wkt = g.ExportToWkt()
        x0, x1, y0, y1 = g.GetEnvelope()
        if not (-180 <= x0 < x1 <= 180 and -90 <= y0 < y1 <= 90):
            raise KitError("The area must be given in longitude/latitude (EPSG:4326).")
        self.bbox = [x0, y0, x1, y1]
        self.name = name
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.center = (cx, cy)
        self.srs = osr.SpatialReference()
        self.srs.ImportFromProj4(
            f"+proj=laea +lat_0={cy:.6f} +lon_0={cx:.6f} +datum=WGS84 +units=m +no_defs")
        self.srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.wgs = _srs_wgs84()
        self.to_local = osr.CoordinateTransformation(self.wgs, self.srs)
        self.to_wgs = osr.CoordinateTransformation(self.srs, self.wgs)
        gl = g.Clone()
        gl.Transform(self.to_local)
        self.local_geom = gl
        self.area_km2 = gl.GetArea() / 1e6
        lx0, lx1, ly0, ly1 = gl.GetEnvelope()
        self.local_bounds = (lx0, ly0, lx1, ly1)

    def check_size(self, max_km2, min_km2=0.05, kit="This kit"):
        if self.area_km2 > max_km2:
            raise KitError(f"The area is {self.area_km2:,.0f} km². {kit} handles up to {max_km2:,} km² "
                           "in one run; choose a smaller area.")
        if self.area_km2 < min_km2:
            raise KitError(f"The area is too small (under {min_km2} km²).")


class Grid:
    def __init__(self, aoi, res, max_pixels=None):
        x0, y0, x1, y1 = aoi.local_bounds
        if max_pixels:
            box = (x1 - x0) * (y1 - y0)
            if box / (res * res) > max_pixels:
                res = float(math.ceil(math.sqrt(box / max_pixels)))
        self.res = float(res)
        x0 = math.floor(x0 / res) * res
        y0 = math.floor(y0 / res) * res
        x1 = math.ceil(x1 / res) * res
        y1 = math.ceil(y1 / res) * res
        self.bounds = (x0, y0, x1, y1)
        self.cols = int(round((x1 - x0) / res))
        self.rows = int(round((y1 - y0) / res))
        self.shape = (self.rows, self.cols)
        self.gt = (x0, res, 0, y1, 0, -res)
        self.wkt = aoi.srs.ExportToWkt()
        self.px_km2 = res * res / 1e6
        self.aoi = aoi
        self.mask = self._rasterize([aoi.local_geom])

    def _rasterize(self, geoms, values=None, all_touched=False):
        ds = gdal.GetDriverByName("MEM").Create("", self.cols, self.rows, 1, gdal.GDT_Int32)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.wkt)
        mem = ogr.GetDriverByName("Memory").CreateDataSource("m")
        lyr = mem.CreateLayer("m", srs=self.aoi.srs)
        lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
        for i, g in enumerate(geoms):
            f = ogr.Feature(lyr.GetLayerDefn())
            f.SetGeometry(g)
            f.SetField("v", int(values[i]) if values is not None else 1)
            lyr.CreateFeature(f)
        opts = ["ATTRIBUTE=v"] + (["ALL_TOUCHED=TRUE"] if all_touched else [])
        gdal.RasterizeLayer(ds, [1], lyr, options=opts)
        a = ds.GetRasterBand(1).ReadAsArray()
        return a if values is not None else a.astype(bool)

    def rasterize(self, local_geoms, values=None, all_touched=False):
        return self._rasterize(local_geoms, values, all_touched)

    def pixel_to_lonlat(self, r, c):
        x = self.gt[0] + (c + 0.5) * self.res
        y = self.gt[3] - (r + 0.5) * self.res
        lon, lat, _ = self.aoi.to_wgs.TransformPoint(x, y)
        return lon, lat

    def lonlat_to_pixel(self, lon, lat):
        x, y, _ = self.aoi.to_local.TransformPoint(lon, lat)
        c = int((x - self.gt[0]) / self.res)
        r = int((self.gt[3] - y) / self.res)
        return r, c

    def distance_to(self, seed):
        """Distance (m) from every cell to the nearest True cell of `seed` (GDAL proximity)."""
        drv = gdal.GetDriverByName("MEM")
        src = drv.Create("", self.cols, self.rows, 1, gdal.GDT_Byte)
        src.SetGeoTransform(self.gt)
        src.SetProjection(self.wkt)
        src.GetRasterBand(1).WriteArray(seed.astype(np.uint8))
        dst = drv.Create("", self.cols, self.rows, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(self.gt)
        dst.SetProjection(self.wkt)
        gdal.ComputeProximity(src.GetRasterBand(1), dst.GetRasterBand(1),
                              ["VALUES=1", "DISTUNITS=GEO"])
        out = dst.GetRasterBand(1).ReadAsArray()
        if not seed.any():
            out[:] = np.inf
        return out


def read_vector(path, aoi, where=None):
    """Read features from any OGR-readable file, reprojected to the AOI grid CRS.

    Returns a list of (local_geometry, attributes dict, lonlat_geometry)."""
    ds = ogr.Open(path)
    if ds is None:
        raise KitError(f"Could not open {path}.")
    lyr = ds.GetLayer(0)
    if where:
        lyr.SetAttributeFilter(where)
    src = lyr.GetSpatialRef()
    if src is None:
        src = _srs_wgs84()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_local = osr.CoordinateTransformation(src, aoi.srs)
    to_wgs = osr.CoordinateTransformation(src, _srs_wgs84())
    out = []
    defn = lyr.GetLayerDefn()
    names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        gl = g.Clone()
        gl.Transform(to_local)
        gw = g.Clone()
        gw.Transform(to_wgs)
        out.append((gl, {n: f.GetField(n) for n in names}, gw))
    return out
