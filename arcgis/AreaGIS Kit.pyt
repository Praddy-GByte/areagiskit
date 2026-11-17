# -*- coding: utf-8 -*-
# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit for ArcGIS Pro: complete geospatial studies of any area.

Keep this file next to the 'areagiskit' folder, then in ArcGIS Pro: Catalog pane > Toolboxes > Add Toolbox > choose this file.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from areagiskit.arcgis_tools import tools as _tools  # noqa: E402

_TOOLS = _tools()
for _t in _TOOLS:
    globals()[_t.__name__] = _t


class Toolbox(object):
    def __init__(self):
        self.label = "AreaGIS Kit"
        self.alias = "areagiskit"
        self.tools = list(_TOOLS)
