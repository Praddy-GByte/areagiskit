# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
import os

from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QMenu, QToolButton

try:                                    # Qt5 builds of QGIS
    from qgis.PyQt.QtWidgets import QAction
except ImportError:                     # Qt6 builds moved QAction to QtGui
    from qgis.PyQt.QtGui import QAction

INSTANT_POPUP = getattr(QToolButton, "InstantPopup", None) or QToolButton.ToolButtonPopupMode.InstantPopup

from .kits import all_kits
from .qgis_provider import AreaGISKitProvider
from .plugin_info import PLUGIN_NAME, PROVIDER_ID


class AreaGISKitPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.actions = []
        self.button = None

    def initProcessing(self):
        self.provider = AreaGISKitProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()
        icon = QIcon(os.path.join(os.path.dirname(__file__), "icon.png"))
        menu = QMenu(PLUGIN_NAME)
        for k in all_kits():
            kp = os.path.join(os.path.dirname(__file__), "icons", f"{k.KEY}.png")
            a = QAction(QIcon(kp) if os.path.exists(kp) else icon, f"{k.NAME}: {k.TAGLINE}", self.iface.mainWindow())
            a.setToolTip(k.DESCRIPTION)
            a.triggered.connect(lambda _=False, key=k.KEY: self.open(key))
            menu.addAction(a)
            self.iface.addPluginToMenu(PLUGIN_NAME, a)
            self.actions.append(a)
        self.menu = menu
        self.button = QToolButton()
        self.button.setIcon(icon)
        self.button.setToolTip(f"{PLUGIN_NAME}: one-click study")
        kits = all_kits()
        if len(kits) == 1:                       # single-study plugin: the button opens it directly
            self.button.clicked.connect(lambda _=False, key=kits[0].KEY: self.open(key))
        else:
            self.button.setMenu(menu)
            self.button.setPopupMode(INSTANT_POPUP)
        self.toolbar_action = self.iface.addToolBarWidget(self.button)

    def open(self, key):
        import processing
        processing.execAlgorithmDialog(f"{PROVIDER_ID}:{key}", {})

    def unload(self):
        for a in self.actions:
            self.iface.removePluginMenu(PLUGIN_NAME, a)
        if getattr(self, "toolbar_action", None):
            self.iface.removeToolBarIcon(self.toolbar_action)
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
