# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Analyses that use the user's own data: own classes and training points (Level-II),
master plan conformity, and correlation with census or other zone statistics."""

import colorsys
import math
import re

import numpy as np

from ...core.aoi import read_vector
from ...core.data import KitError
from .classify import CLASSES

KEYWORDS = {
    2: ["built", "urban", "settlement", "residential", "commercial", "industrial", "road", "building", "habitation",
        "town", "village", "abadi", "infrastructure", "airport", "rail"],
    1: ["water", "river", "lake", "pond", "reservoir", "canal", "tank", "stream", "sea"],
    3: ["forest", "tree", "plantation", "orchard", "mangrove", "woodland", "jungle", "grove", "garden"],
    4: ["crop", "agri", "farm", "fallow", "kharif", "rabi", "paddy", "rice", "wheat", "sugarcane", "cultivat"],
    5: ["grass", "scrub", "shrub", "pasture", "rangeland", "meadow", "grazing", "bush"],
    6: ["barren", "waste", "rock", "sand", "bare", "mining", "quarry", "desert", "dune", "gully", "ravine"],
    7: ["wetland", "marsh", "swamp", "mudflat", "bog", "salt pan"],
    8: ["snow", "ice", "glacier"],
}


def parent_of(name, explicit=None):
    if explicit not in (None, ""):
        try:
            v = int(explicit)
            if v in CLASSES:
                return v
        except (TypeError, ValueError):
            for k, (n, _) in CLASSES.items():
                if str(explicit).strip().lower() in n.lower():
                    return k
    low = str(name).lower()
    for k, words in KEYWORDS.items():
        if any(w in low for w in words):
            return k
    return None


def _shade(hex_col, i, n):
    r, g, b = (int(hex_col[j:j + 2], 16) / 255 for j in (1, 3, 5))
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    light = min(0.85, max(0.2, light + (i - (n - 1) / 2) * 0.12))
    r, g, b = colorsys.hls_to_rgb(h, light, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def read_training(path, aoi, grid, class_field="class", year_field="year", level1_field="level1"):
    """Read user training points. Returns (classes_l2, points) where
    classes_l2 = {code: (name, colour, parent)} and points = [(row, col, code, year or None)]."""
    feats = read_vector(path, aoi)
    if not feats:
        raise KitError("The training points layer is empty.")
    if class_field not in feats[0][1]:
        raise KitError(f"Field '{class_field}' not found in the training points.")
    names, parents = [], {}
    for _, a, _ in feats:
        n = str(a.get(class_field) or "").strip()
        if n and n not in parents:
            names.append(n)
            parents[n] = parent_of(n, a.get(level1_field))
    unknown = [n for n in names if parents[n] is None]
    if unknown:
        raise KitError("Could not tell which main class these belong to: " + ", ".join(unknown[:6])
                       + f". Add a field '{level1_field}' with 1–8 (1 water, 2 built-up, 3 forest, 4 cropland, 5 grass/scrub, "
                         "6 barren, 7 wetland, 8 snow) or rename them.")
    by_parent = {}
    for n in sorted(names):
        by_parent.setdefault(parents[n], []).append(n)
    classes = {}
    code = 21
    for p in sorted(by_parent):
        lst = by_parent[p]
        for i, n in enumerate(lst):
            classes[code] = (n, _shade(CLASSES[p][1], i, len(lst)), p)
            code += 1
    name_code = {v[0]: k for k, v in classes.items()}
    pts = []
    for gl, a, gw in feats:
        n = str(a.get(class_field) or "").strip()
        if n not in name_code:
            continue
        c = gw.Centroid()
        r, cc = grid.lonlat_to_pixel(c.GetX(), c.GetY())
        if not (0 <= r < grid.rows and 0 <= cc < grid.cols and grid.mask[r, cc]):
            continue
        y = a.get(year_field) if year_field in a else None
        try:
            y = int(y) if y not in (None, "") else None
        except (TypeError, ValueError):
            y = None
        pts.append((r, cc, name_code[n], y))
    if len(pts) < 30:
        raise KitError(f"Only {len(pts)} training points fall inside the area; at least 30 are needed.")
    return classes, pts


def label_raster(points, shape, year, default_year):
    lab = np.zeros(shape, np.uint8)
    for r, c, code, y in points:
        if (y if y is not None else default_year) == year:
            lab[r, c] = code
    return lab


def point_years(points, default_year):
    return sorted({(y if y is not None else default_year) for *_, y in points})


def to_level1(m2, classes_l2):
    lut = np.zeros(256, np.uint8)
    for k, (_, _, p) in classes_l2.items():
        lut[k] = p
    return lut[m2]


def l1_proba(res, classes_l2):
    """Aggregate Level-II class probabilities to Level-I parents."""
    parents = sorted({classes_l2[c][2] for c in res["classes"]})
    pr = np.zeros((len(parents),) + res["proba"].shape[1:], np.float32)
    for i, c in enumerate(res["classes"]):
        pr[parents.index(classes_l2[c][2])] += res["proba"][i].astype(np.float32)
    return parents, pr


def resync_l2(res, l1_new, classes_l2):
    """Where the Level-I class was changed by a rule, choose the most likely Level-II class within the new parent."""
    cls = np.asarray(res["classes"])
    cur_l1 = to_level1(res["map"], classes_l2)
    bad = (cur_l1 != l1_new) & (l1_new > 0)
    if not bad.any():
        return 0
    pr = res["proba"][:, bad].astype(np.float32)
    par = np.array([classes_l2[c][2] for c in cls])
    want = l1_new[bad]
    pr[par[:, None] != want[None, :]] = -1
    res["map"][bad] = cls[pr.argmax(0)]
    return int(bad.sum())


# ======================================================================= master plan

PLAN_KEYS = [
    ({2}, ["residential", "commercial", "industrial", "public", "semi-public", "semi public", "transport", "mixed", "urban",
           "built", "institutional", "utility", "housing", "development", "abadi"]),
    ({4, 5}, ["agri", "rural", "farm", "crop"]),
    ({3, 5}, ["green", "park", "forest", "recreation", "open space", "conservation", "eco", "garden", "buffer", "reserve"]),
    ({1, 7}, ["water", "river", "lake", "wetland", "drain"]),
]


def plan_expected(name):
    low = str(name).lower()
    for exp, words in PLAN_KEYS:
        if any(w in low for w in words):
            return exp
    return None


def plan_conformity(path, field, aoi, grid, maps, years, px):
    feats = read_vector(path, aoi)
    if not feats:
        return None
    if field not in feats[0][1]:
        raise KitError(f"Field '{field}' not found in the master plan layer.")
    cats = sorted({str(a.get(field)) for _, a, _ in feats})
    cid = {c: i + 1 for i, c in enumerate(cats)}
    z = grid.rasterize([g for g, _, _ in feats], values=[cid[str(a.get(field))] for _, a, _ in feats])
    latest, first = years[-1], years[0]
    m = maps[latest]
    conf = np.zeros(grid.shape, np.uint8)
    rows = []
    for c in cats:
        sel = (z == cid[c]) & (m > 0)
        if not sel.any():
            continue
        exp = plan_expected(c)
        comp = {k: float((sel & (m == k)).sum() * px) for k in CLASSES}
        area = float(sel.sum() * px)
        if exp is None:
            conf[sel] = 3
            ok_share = None
        else:
            good = sel & np.isin(m, list(exp))
            conf[good] = 1
            conf[sel & ~np.isin(m, list(exp))] = 2
            ok_share = float(good.sum() / sel.sum() * 100)
        illegal_built = float((sel & (m == 2)).sum() * px) if exp is not None and 2 not in exp else 0.0
        growth = float(((z == cid[c]) & (maps[latest] == 2) & (maps[first] != 2)).sum() * px)
        rows.append(dict(category=c, expected=exp, area=area, conformity=ok_share, built_in_nonbuilt=illegal_built,
                         built_growth=growth, composition=comp))
    return dict(rows=rows, map=conf)


PLAN_CLASSES = {1: ("Conforms to plan", "#2f7a4d"), 2: ("Does not conform", "#c4281b"), 3: ("Plan category not recognised", "#bdbdbd")}


# ======================================================================= census / zone statistics

def _rank(x):
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def corr_test(x, y, perms=999, seed=2):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6 or x.std() == 0 or y.std() == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(perms)])
    p = float((np.sum(null >= abs(r)) + 1) / (perms + 1))
    return dict(r=r, rho=rho, p=p, n=int(len(x)))


def census(path, id_field, fields, aoi, grid, layers):
    """layers: dict name -> raster (zone mean is taken). fields: user numeric fields. Returns table rows and per-zone values."""
    feats = read_vector(path, aoi)
    if not feats:
        return None
    fl = [f.strip() for f in re.split(r"[,;]", fields or "") if f.strip()]
    if not fl:
        fl = [k for k, v in feats[0][1].items() if isinstance(v, (int, float)) and k != id_field]
    z = grid.rasterize([g for g, _, _ in feats], values=list(range(1, len(feats) + 1)))
    n = len(feats)
    zv = {}
    flat = z.ravel()
    cnt = np.bincount(flat, minlength=n + 1).astype(float)
    for name, arr in layers.items():
        a = np.nan_to_num(arr.ravel().astype(float))
        ok = np.isfinite(arr.ravel())
        s = np.bincount(flat[ok], weights=a[ok], minlength=n + 1)
        c = np.bincount(flat[ok], minlength=n + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zv[name] = np.where(c > 0, s / c, np.nan)[1:]
    uv = {}
    for f in fl:
        vals = []
        for _, a, _ in feats:
            try:
                vals.append(float(a.get(f)))
            except (TypeError, ValueError):
                vals.append(np.nan)
        uv[f] = np.array(vals)
    valid = cnt[1:] > 0
    rows = []
    for f, yv in uv.items():
        for name, xv in zv.items():
            t = corr_test(xv[valid], yv[valid])
            if t:
                rows.append(dict(field=f, metric=name, **t))
    rows.sort(key=lambda d: -abs(d["r"]))
    ids = [str(a.get(id_field, i + 1)) for i, (_, a, _) in enumerate(feats)]
    return dict(rows=rows, zone_values=zv, user_values=uv, ids=ids, fields=fl)
