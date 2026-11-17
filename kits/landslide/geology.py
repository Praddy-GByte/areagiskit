# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Lithology from the Macrostrat geologic map API (https://macrostrat.org, CC BY 4.0; no account).

The API answers point queries with the map units of every compiled map at that point (scales large, medium, small, tiny).
A regular grid of points is queried, the finest available unit is kept, its lithology text is grouped into broad classes
that matter for slope stability, and each cell takes the class of its nearest grid point. Detail varies by country: large
and medium scale maps exist for the USA, Japan, parts of Europe and Australia; elsewhere only the 1:20 million world map
('tiny'), which is usually one class for a small area and then adds nothing."""

import concurrent.futures as cf
import json
import urllib.request

import numpy as np

API = "https://macrostrat.org/api/v2/geologic_units/map?lat={lat:.5f}&lng={lon:.5f}"
CREDIT = ("Lithology: Macrostrat geologic map compilation (Peters, S. E., Husson, J. M., Czaplewski, J. 2018. Macrostrat: a platform "
          "for geological data integration and deep-time Earth crust research. G-cubed 19, 1393–1409), CC BY 4.0.")
SCALE_RANK = {"large": 0, "medium": 1, "small": 2, "tiny": 3}

# broad classes, roughly from weak to strong for shallow landslides
GROUPS = [
    (1, "Unconsolidated sediments", ("alluvium", "colluv", "unconsolidated", "sand", "gravel", "clay", "silt", "till", "loess",
                                     "talus", "landslide", "debris", "fill", "terrace", "fan", "quaternary sediment", "deposit", "valley floor",
                                     "coastal plain", "floodplain")),
    (2, "Weathered / weak sedimentary", ("mudstone", "shale", "siltstone", "claystone", "marl", "flysch", "mélange", "melange", "slate")),
    (3, "Pyroclastic and volcaniclastic", ("tuff", "pyroclastic", "ignimbrite", "volcaniclastic", "ash", "lahar", "breccia")),
    (4, "Sedimentary rock", ("sandstone", "conglomerate", "sedimentary", "greywacke", "graywacke", "chert", "siliciclastic")),
    (5, "Carbonate rock", ("limestone", "dolomite", "carbonate", "dolostone", "chalk")),
    (6, "Granitic / plutonic", ("granite", "granodiorite", "diorite", "plutonic", "tonalite", "gabbro", "intrusive", "monzonite")),
    (7, "Volcanic lava rock", ("basalt", "andesite", "rhyolite", "dacite", "lava", "volcanic")),
    (8, "Metamorphic rock", ("schist", "gneiss", "phyllite", "metamorphic", "quartzite", "marble", "amphibolite", "migmatite")),
]
NAMES = {k: n for k, n, _ in GROUPS}
NAMES[9] = "Other / unknown"


def group_of(text):
    t = (text or "").lower()
    for k, _, words in GROUPS:
        if any(w in t for w in words):
            return k
    return 9


def _query(lat, lon, ua, timeout=20):
    req = urllib.request.Request(API.format(lat=lat, lon=lon), headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)["success"]["data"]
    if not data:
        return None
    best = min(data, key=lambda d: SCALE_RANK.get(d.get("scale"), 9))
    txt = " ".join(str(best.get(k) or "") for k in ("lith", "descrip", "name"))
    return group_of(txt), best.get("scale"), (best.get("lith") or best.get("name") or "")[:80]


def lithology(grid, n=16, ua="QGIS-Study-Kits (open-source QGIS plugin)", workers=8):
    """Return (class raster uint8, info dict) or (None, info). info: scales used, n points, share by class, examples."""
    rows, cols = grid.shape
    rr = np.linspace(0, rows - 1, n + 2)[1:-1].round().astype(int)
    cc = np.linspace(0, cols - 1, n + 2)[1:-1].round().astype(int)
    pts = [(r, c, *grid.pixel_to_lonlat(r, c)) for r in rr for c in cc]
    res = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_query, lat, lon, ua): (r, c) for r, c, lon, lat in pts}
        for f in cf.as_completed(futs):
            try:
                v = f.result()
            except Exception:
                v = None
            if v:
                res[futs[f]] = v
    info = dict(n=len(pts), answered=len(res), scales={}, examples={})
    if len(res) < 0.5 * len(pts):
        return None, info
    for (_, _), (k, sc, txt) in res.items():
        info["scales"][sc] = info["scales"].get(sc, 0) + 1
        info["examples"].setdefault(k, txt)
    if set(info["scales"]) <= {"tiny"}:
        info["coarse"] = True
    keys = list(res)
    pr = np.array([k[0] for k in keys], float)
    pc = np.array([k[1] for k in keys], float)
    val = np.array([res[k][0] for k in keys], np.uint8)
    # nearest grid point for every cell (cheap: separable because points lie on a regular grid)
    out = np.zeros(grid.shape, np.uint8)
    R_, C_ = np.indices(grid.shape)
    best = np.full(grid.shape, np.inf)
    for i in range(len(val)):
        d = (R_ - pr[i]) ** 2 + (C_ - pc[i]) ** 2
        m = d < best
        best[m] = d[m]
        out[m] = val[i]
    out[~grid.mask] = 0
    present = [k for k in np.unique(out) if k > 0]
    info["share"] = {int(k): float((out == k)[grid.mask].mean() * 100) for k in present}
    return out, info
