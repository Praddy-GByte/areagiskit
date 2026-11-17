# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Registry of kits."""


def all_kits():
    from .landkit import LandKit
    from .floodkit import FloodKit
    from .hazardkit import HazardKit
    from .cropkit import CropKit
    from .urbankit import UrbanKit
    from .healthkit import HealthKit
    from .lulckit import LULCStudyKit
    from .floodstudy import FloodStudyKit
    from .landslidestudy import LandslideStudyKit
    from .cropstudy import CropStudyKit
    from .urbanstudy import UrbanStudyKit
    from .healthstudy import HealthStudyKit
    from ..plugin_info import KIT_KEYS
    ks = [LULCStudyKit, FloodStudyKit, LandslideStudyKit, CropStudyKit, UrbanStudyKit, HealthStudyKit]
    return ks if not KIT_KEYS else [k for k in ks if k.KEY in KIT_KEYS]


def get(key):
    from .landkit import LandKit          # quick 10 m change: now part of the AreaGIS Kit: LULC, still on the command line
    for k in all_kits() + [LandKit]:
        if k.KEY == key:
            return k
    raise KeyError(key)
