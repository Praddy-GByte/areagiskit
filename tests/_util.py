# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Shared helpers: make the plugin package importable as `areagiskit` whatever folder it lives in."""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the plugin folder
PARENT = os.path.dirname(ROOT)
PKG = os.path.basename(ROOT)                                            # areagiskit, areagiskit_flood, ...
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def mod(name):
    """Import `<package>.<name>`, e.g. mod("kits.health.stats")."""
    return importlib.import_module(f"{PKG}.{name}")


def rook(nr, nc):
    """Rook neighbours on an nr x nc lattice (list of lists)."""
    nb = []
    for r in range(nr):
        for c in range(nc):
            v = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < nr and 0 <= cc < nc:
                    v.append(rr * nc + cc)
            nb.append(v)
    return nb
