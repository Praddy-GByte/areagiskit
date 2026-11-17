# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Offline tests for the earthquake shaking factor (no network)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from areagiskit.kits.landslide import models as LM  # noqa: E402
from areagiskit.kits.landslide import seismic as SM  # noqa: E402

GRID_XML = b"""<?xml version="1.0" encoding="US-ASCII" standalone="yes"?>
<shakemap_grid xmlns="http://earthquake.usgs.gov/eqcenter/shakemap" event_id="t1">
<grid_specification lon_min="88.0" lat_min="27.0" lon_max="88.2" lat_max="27.2" nominal_lon_spacing="0.1" nominal_lat_spacing="0.1" nlon="3" nlat="3" />
<grid_field index="1" name="LON" units="dd" />
<grid_field index="2" name="LAT" units="dd" />
<grid_field index="3" name="PGA" units="pctg" />
<grid_data>
88.0 27.2 10
88.1 27.2 20
88.2 27.2 30
88.0 27.1 11
88.1 27.1 21
88.2 27.1 31
88.0 27.0 12
88.1 27.0 22
88.2 27.0 32
</grid_data>
</shakemap_grid>"""


class TestSeismic(unittest.TestCase):
    def test_grid_xml_parsed_to_g_top_row_first(self):
        lon0, lat1, dlon, dlat, a = SM.parse_grid_xml(GRID_XML)
        self.assertEqual(a.shape, (3, 3))
        self.assertAlmostEqual(lon0, 88.0)
        self.assertAlmostEqual(lat1, 27.2)
        self.assertAlmostEqual(float(a[0, 0]), 0.10, places=6)   # %g -> g, top-left
        self.assertAlmostEqual(float(a[2, 2]), 0.32, places=6)   # bottom-right
        self.assertAlmostEqual(float(a[1, 2]), 0.31, places=6)

    def test_cells_across_small_area_is_below_threshold(self):
        nx, ny = SM._cells_across((132.70, 34.32, 132.78, 34.37))   # Hiroshima test box: ~1.6 cells
        self.assertLess(nx * ny, 4)
        nx, ny = SM._cells_across((88.17, 27.11, 88.47, 27.36))     # Sikkim: ~30 cells
        self.assertGreater(nx * ny, 4)

    def test_pga_kept_out_of_machine_learning_factors(self):
        keys = {k for k, *_ in LM.FACTORS}
        self.assertTrue({"pga", "pga_event"} <= keys)
        self.assertTrue({"pga", "pga_event"} <= LM.CONTEXT_ONLY)
        F = {k: np.zeros((4, 4), np.float32) for k in ("slope", "relief", "pga")}
        ml_keys = [k for k, *_ in LM.FACTORS if k in F and k not in LM.CONTEXT_ONLY]
        self.assertEqual(ml_keys, ["slope", "relief"])

    def test_pga_scores_higher_for_stronger_shaking_in_ahp(self):
        a = np.linspace(0.1, 0.5, 100, dtype=np.float32).reshape(10, 10)
        s = LM.ahp_score("pga", a, np.ones((10, 10), bool))
        self.assertLess(float(s[0, 0]), float(s[-1, -1]))


if __name__ == "__main__":
    unittest.main()
