# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""AreaGIS Kit: one-click, validated GIS analyses for QGIS (and the command line)."""


def classFactory(iface):  # QGIS entry point
    from .qgis_plugin import AreaGISKitPlugin
    return AreaGISKitPlugin(iface)
