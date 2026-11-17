# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""The ArcGIS Pro toolbox loads and builds a form for every kit (with a stand-in arcpy)."""
import os
import runpy
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYT = os.path.join(ROOT, "arcgis", "AreaGIS Kit.pyt")


class ArcGISToolbox(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(PYT), "toolbox file not in this build")
    def test_toolbox_and_forms(self):
        sys.path.insert(0, os.path.join(HERE, "mock_arcpy"))
        sys.path.insert(0, os.path.dirname(ROOT))
        try:
            ns = runpy.run_path(PYT)
            tb = ns["Toolbox"]()
            self.assertEqual(len(tb.tools), 6)
            for T in tb.tools:
                t = T()
                ps = t.getParameterInfo()
                names = [p.name for p in ps]
                self.assertEqual(names[-1], "output_folder")
                self.assertEqual(len(names), len(set(names)))
                for p in ps:
                    if p.datatype == "Field":
                        self.assertTrue(p.parameterDependencies, p.name)
                if T.KIT.NEEDS_AOI:
                    self.assertEqual(names[:3], ["area", "extent", "area_name"])
        finally:
            sys.path.remove(os.path.join(HERE, "mock_arcpy"))
            sys.path.remove(os.path.dirname(ROOT))
            for m in [m for m in sys.modules if m == "arcpy" or m.startswith("arcpy.")]:
                del sys.modules[m]


if __name__ == "__main__":
    unittest.main()
