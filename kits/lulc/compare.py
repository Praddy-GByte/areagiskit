# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Hydrology (SCS-CN runoff, RUSLE cover factor), land-use scenarios, classifier comparison,
and spatial drivers (Geodetector, geographically weighted regression)."""

import math

import numpy as np
from osgeo import gdal

from ...core.data import trace, warp

# ======================================================================= hydrology

SOILGRIDS = "https://files.isric.org/soilgrids/latest/data/{v}/{v}_0-5cm_mean.vrt"
CREDIT_SOIL = "SoilGrids 2.0: Poggio, L. et al. (2021). SOIL 7, 217–240. ISRIC, CC BY 4.0."
HSG_NAMES = {1: "A (sandy, high infiltration)", 2: "B (loamy)", 3: "C (clay loam)", 4: "D (clayey, low infiltration)"}
# SCS curve numbers, antecedent moisture II (USDA TR-55), by class and hydrologic soil group A-D
CN = {1: (100, 100, 100, 100), 2: (77, 85, 90, 92), 3: (30, 55, 70, 77), 4: (67, 78, 85, 89), 5: (49, 69, 79, 84),
      6: (77, 86, 91, 94), 7: (90, 92, 94, 95), 8: (98, 98, 98, 98)}


def soil_groups(grid):
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".vrt,.tif,.TIF,.tiff,.zip")
    out = {}
    for v in ("clay", "sand"):
        u = SOILGRIDS.format(v=v)
        trace(url=u, label="soilgrids")
        try:
            out[v] = warp(["/vsicurl/" + u], grid, resample="bilinear", dtype=gdal.GDT_Float32,
                          src_nodata=-32768, dst_nodata=np.nan) / 10.0  # g/kg → %
        except Exception:
            return None
    clay, sand = out["clay"], out["sand"]
    hsg = np.full(grid.shape, 2, np.uint8)
    hsg[(clay < 10) & (sand > 70)] = 1
    hsg[(clay >= 20) & (clay < 40)] = 3
    hsg[clay >= 40] = 4
    hsg[~np.isfinite(clay)] = 2
    hsg[~grid.mask] = 0
    return hsg


def runoff(cmap, hsg, p_mm):
    """SCS-CN event runoff depth (mm) for a design storm of p_mm."""
    cn = np.zeros(cmap.shape, np.float32)
    for c, v in CN.items():
        for g in (1, 2, 3, 4):
            cn[(cmap == c) & (hsg == g)] = v[g - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = 25400.0 / cn - 254.0
        Ia = 0.2 * S
        Q = np.where(p_mm > Ia, (p_mm - Ia) ** 2 / (p_mm - Ia + S), 0)
    Q[cn == 0] = np.nan
    return Q.astype(np.float32), cn


def c_factor(ndvi, alpha=2.0, beta=1.0):
    """RUSLE cover-management factor from NDVI (van der Knijff et al. 2000)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c = np.exp(-alpha * ndvi / (beta - ndvi))
    return np.clip(c, 0, 1).astype(np.float32)


# ======================================================================= scenarios

SCENARIOS = {
    "Business as usual": "Recent transition probabilities continue unchanged.",
    "Ecological protection": "Conversion of forest, water and wetland to built-up halved; conversion of forest to other uses cut by 30%; "
                             "gains of forest from cropland and grass raised by 30%.",
    "Cropland protection": "Conversion of cropland to built-up halved and to other uses cut by 30%.",
}


def scenario_matrix(P, classes, name):
    Q = P.copy()
    ix = {c: i for i, c in enumerate(classes)}

    def scale(a, b, f):
        if a in ix and b in ix:
            Q[ix[a], ix[b]] *= f

    if name == "Ecological protection":
        for a in (3, 1, 7):
            scale(a, 2, 0.5)
        for b in classes:
            if b != 3:
                scale(3, b, 0.7)
        scale(4, 3, 1.3)
        scale(5, 3, 1.3)
    elif name == "Cropland protection":
        scale(4, 2, 0.5)
        for b in classes:
            if b not in (4, 2):
                scale(4, b, 0.7)
    for i in range(len(classes)):  # rebalance: what is not transferred stays
        off = Q[i].sum() - Q[i, i]
        if off > 1:
            Q[i] *= 1 / Q[i].sum()
        else:
            Q[i, i] = 1 - off
    return Q


# ======================================================================= classifier comparison

def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def min_distance(Xtr, ytr, Xte, classes):
    a, b = _std(Xtr, Xte)
    C = np.stack([a[ytr == c].mean(0) for c in classes])
    d = ((b[:, None, :] - C[None]) ** 2).sum(-1)
    return np.asarray(classes)[d.argmin(1)]


def max_likelihood(Xtr, ytr, Xte, classes):
    a, b = _std(Xtr, Xte)
    ll = []
    for c in classes:
        x = a[ytr == c]
        mu = x.mean(0)
        cov = np.cov(x.T) + 1e-3 * np.eye(x.shape[1])
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        d = b - mu
        ll.append(-0.5 * (logdet + np.einsum("ij,jk,ik->i", d, inv, d)))
    return np.asarray(classes)[np.stack(ll, 1).argmax(1)]


def ls_svm_rbf(Xtr, ytr, Xte, classes, D=600, lam=1e-2, seed=0):
    """RBF-kernel least-squares SVM, approximated with random Fourier features (Rahimi & Recht 2007)."""
    a, b = _std(Xtr, Xte)
    rng = np.random.default_rng(seed)
    gamma = 1.0 / a.shape[1]
    W = rng.normal(0, math.sqrt(2 * gamma), (a.shape[1], D))
    u = rng.uniform(0, 2 * math.pi, D)
    fa = math.sqrt(2 / D) * np.cos(a @ W + u)
    fb = math.sqrt(2 / D) * np.cos(b @ W + u)
    Y = np.stack([(ytr == c) * 2.0 - 1 for c in classes], 1)
    A = fa.T @ fa + lam * len(fa) * np.eye(D)
    beta = np.linalg.solve(A, fa.T @ Y)
    return np.asarray(classes)[(fb @ beta).argmax(1)]


def mcnemar(y, p1, p2):
    a = (p1 == y) & (p2 != y)
    b = (p1 != y) & (p2 == y)
    n = a.sum() + b.sum()
    if n == 0:
        return 0.0, 1.0
    chi = (abs(int(a.sum()) - int(b.sum())) - 1) ** 2 / n
    return float(chi), float(math.erfc(math.sqrt(chi / 2)))


# ======================================================================= Geodetector and GWR

def cells(arr, cpx, how="mean"):
    r, c = arr.shape[0] // cpx, arr.shape[1] // cpx
    b = arr[:r * cpx, :c * cpx].reshape(r, cpx, c, cpx).astype(np.float64)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3)) if how == "mean" else np.nansum(b, axis=(1, 3))


def _q(y, strata):
    N, var = len(y), y.var()
    if var == 0:
        return 0.0
    ssw = sum((strata == h).sum() * y[strata == h].var() for h in np.unique(strata))
    return float(1 - ssw / (N * var))


def geodetector(y, X, names, k=5, perms=199, seed=1):
    """Factor detector (q and permutation p) and interaction detector (Wang et al. 2010)."""
    rng = np.random.default_rng(seed)
    strata = {}
    for n, x in zip(names, X.T):
        qs = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
        strata[n] = np.digitize(x, qs)
    fac = []
    for n in names:
        q = _q(y, strata[n])
        null = [_q(rng.permutation(y), strata[n]) for _ in range(perms)]
        p = (np.sum(np.array(null) >= q) + 1) / (perms + 1)
        fac.append(dict(name=n, q=q, p=float(p)))
    inter = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            qi = _q(y, strata[a] * (k + 1) + strata[b])
            q1, q2 = fac[i]["q"], fac[j]["q"]
            if qi > q1 + q2:
                kind = "nonlinear enhancement"
            elif qi > max(q1, q2):
                kind = "bivariate enhancement"
            elif qi < min(q1, q2):
                kind = "nonlinear weakening"
            else:
                kind = "univariate weakening"
            inter.append(dict(a=a, b=b, q=qi, kind=kind))
    inter.sort(key=lambda d: -d["q"])
    return fac, inter


def gwr(y, X, coords, bandwidths=(30, 50, 80, 120, 200)):
    """Geographically weighted regression, adaptive bisquare kernel, bandwidth by leave-one-out CV."""
    n = len(y)
    Xd = np.column_stack([np.ones(n), (X - X.mean(0)) / (X.std(0) + 1e-9)])
    D = np.sqrt(((coords[:, None, :] - coords[None]) ** 2).sum(-1))
    order = np.argsort(D, 1)

    def fit(k, loo=False):
        k = min(k, n - 1)
        B = np.zeros((n, Xd.shape[1]))
        yhat = np.zeros(n)
        for i in range(n):
            h = D[i, order[i, k]] + 1e-9
            w = np.where(D[i] < h, (1 - (D[i] / h) ** 2) ** 2, 0)
            if loo:
                w[i] = 0
            XtW = Xd.T * w
            beta = np.linalg.lstsq(XtW @ Xd + 1e-6 * np.eye(Xd.shape[1]), XtW @ y, rcond=None)[0]
            B[i] = beta
            yhat[i] = Xd[i] @ beta
        return B, yhat

    best = None
    for k in bandwidths:
        if k >= n:
            continue
        _, yh = fit(k, loo=True)
        cv = float(((y - yh) ** 2).sum())
        if best is None or cv < best[1]:
            best = (k, cv)
    if best is None:
        return None
    B, yhat = fit(best[0])
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else 0
    bg = np.linalg.lstsq(Xd, y, rcond=None)[0]
    r2g = 1 - ((y - Xd @ bg) ** 2).sum() / ss if ss > 0 else 0
    return dict(k=best[0], coef=B, r2=float(r2), r2_global=float(r2g), global_coef=bg)
