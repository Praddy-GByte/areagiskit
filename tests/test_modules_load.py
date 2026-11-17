# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Every engine module imports and is not empty (catches a file lost or truncated by an edit)."""
import importlib
import os
import pkgutil
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.basename(ROOT)
SKIP = ("qgis_", "arcgis_tools")


class ModulesLoad(unittest.TestCase):
    def test_all_modules(self):
        sys.path.insert(0, os.path.dirname(ROOT))
        pkg = importlib.import_module(PKG)
        for info in pkgutil.walk_packages(pkg.__path__, PKG + "."):
            name = info.name
            if ".tests" in name or any(name.split(".")[-1].startswith(s) for s in SKIP):
                continue
            mod = importlib.import_module(name)
            path = getattr(mod, "__file__", "")
            if path and not path.endswith("__init__.py"):
                self.assertGreater(os.path.getsize(path), 200, f"{name} looks empty")


if __name__ == "__main__":
    unittest.main()
