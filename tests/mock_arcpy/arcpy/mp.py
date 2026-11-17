# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
ADDED = []


class _Map:
    def addDataFromPath(self, p):
        ADDED.append(p)


class ArcGISProject:
    def __init__(self, which):
        self.activeMap = _Map()
