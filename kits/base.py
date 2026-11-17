# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""What every kit looks like.

A kit declares its inputs once (PARAMS). The QGIS Processing provider, the
command line and any future front end (ArcGIS Pro toolbox, web app) build
their forms from that list, then call run().

PARAMS entry: dict(name, type, label, default=None, optional=False, options=None, help="")
types: 'year', 'date', 'int', 'float', 'enum', 'points', 'polygons', 'file', 'field', 'text'
"""

import os
import time

from ..core.data import TRACE, KitError, configure_gdal, trace_reset


class Kit:
    NAME = ""          # "FloodKit"
    KEY = ""           # "flood"
    TAGLINE = ""
    DESCRIPTION = ""
    NEEDS_AOI = True
    MAX_KM2 = 5000
    PARAMS = []
    CREDITS = ""

    def __init__(self, progress=None, cancelled=None):
        self._progress = progress
        self._cancelled = cancelled
        self.t0 = time.time()

    def step(self, pct, msg):
        if self._cancelled and self._cancelled():
            raise KitError("Cancelled.")
        if self._progress:
            self._progress(int(pct), msg)

    def seconds(self):
        return time.time() - self.t0

    def prepare(self, aoi, out_dir, params=None):
        trace_reset()
        self._aoi, self._params = aoi, params or {}
        if self.NEEDS_AOI:
            aoi.check_size(self.MAX_KM2, kit=self.NAME)
        os.makedirs(out_dir, exist_ok=True)
        configure_gdal()

    def reproduce_command(self):
        """The exact command line that reruns this analysis."""
        parts = ["python -m areagiskit.cli", self.KEY]
        a = getattr(self, "_aoi", None)
        if a is not None:
            parts.append(f'--wkt "{a.wkt}"' if len(a.wkt) < 600 else "--bbox " + " ".join(f"{v:.5f}" for v in a.bbox))
            parts.append(f'--name "{a.name}"')
        for k, v in (getattr(self, "_params", {}) or {}).items():
            if v not in (None, ""):
                parts.append(f'{k}="{v}"' if " " in str(v) else f"{k}={v}")
        return " ".join(parts)

    def sources(self):
        return list(TRACE)

    def run(self, aoi, params, out_dir):
        """Return dict(report=path, layers=[dict(path, name, kind)], verdict=..., summary=str)."""
        raise NotImplementedError
