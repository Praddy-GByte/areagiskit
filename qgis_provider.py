# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .kits import all_kits
from .qgis_algorithm import KitAlgorithm
from .plugin_info import PLUGIN_NAME, PROVIDER_ID


class AreaGISKitProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        for k in all_kits():
            self.addAlgorithm(KitAlgorithm(k))

    def id(self):
        return PROVIDER_ID

    def name(self):
        return PLUGIN_NAME

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), "icon.png"))
