# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Combining flood detectors, tested on 20 floods mapped by Copernicus EMS analysts (validation/floodbench)."""
from ...core.raster import neighbours


def clean(m, k_fill=6, k_keep=3):
    """Fill radar pinholes (a dry cell with >= k_fill of 8 flooded neighbours) and drop isolated cells
    (a flooded cell with fewer than k_keep flooded neighbours)."""
    n = neighbours(m, 1) - m
    m = m | (n >= k_fill)
    n = neighbours(m, 1) - m
    return m & (n >= k_keep)
