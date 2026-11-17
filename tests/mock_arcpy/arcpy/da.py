# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
from osgeo import ogr


class SearchCursor:
    """Yields rows of (Geometry, field values...) read with OGR."""

    def __init__(self, path, fields):
        self.path, self.fields = str(path), fields

    def __enter__(self):
        from . import Geometry
        ds = ogr.Open(self.path)
        lyr = ds.GetLayer(0)
        rows = []
        for f in lyr:
            row = []
            for n in self.fields:
                if n == "SHAPE@":
                    g = f.GetGeometryRef()
                    g = g.Clone() if g is not None else None
                    if g is not None and g.GetSpatialReference() is None and lyr.GetSpatialRef() is not None:
                        g.AssignSpatialReference(lyr.GetSpatialRef())
                    row.append(Geometry(g) if g is not None else None)
                else:
                    row.append(f.GetField(n))
            rows.append(tuple(row))
        self.rows = rows
        return iter(rows)

    def __exit__(self, *a):
        return False
