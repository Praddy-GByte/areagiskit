# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""More small-area health analyses: geographically weighted Poisson regression (GWPR), the Knox space-time interaction
test (points and areas x periods), kernel density of case points with a risk-ratio surface and Monte Carlo tolerance
contours, and the two-step floating catchment area (2SFCA) accessibility index. numpy only."""

import math

import numpy as np
from ...core.net import skipped


# ------------------------------------------------------------------ GWPR
def _bisquare(d, h):
    w = np.zeros_like(d)
    m = d < h
    w[m] = (1 - (d[m] / h) ** 2) ** 2
    return w


def _local_poisson(y, X, off, w, b0=None, iters=30):
    """Weighted Poisson IRLS. Returns beta, cov (unscaled), final mu."""
    k = X.shape[1]
    b = np.zeros(k) if b0 is None else b0.copy()
    if b0 is None:
        b[0] = math.log(max((w * y).sum(), 1e-9) / max((w * np.exp(off)).sum(), 1e-12))
    for _ in range(iters):
        eta = X @ b + off
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta - off + (y - mu) / np.maximum(mu, 1e-12)
        A = w * mu
        XtA = X.T * A
        M = XtA @ X + 1e-8 * np.eye(k)
        try:
            nb = np.linalg.solve(M, XtA @ z)
        except np.linalg.LinAlgError:
            break
        done = np.max(np.abs(nb - b)) < 1e-7
        b = nb
        if done:
            break
    mu = np.exp(np.clip(X @ b + off, -30, 30))
    A = w * mu
    M = (X.T * A) @ X + 1e-8 * np.eye(k)
    return b, M, mu


def _deviance(y, mu):
    with np.errstate(all="ignore"):
        return float(2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu)))


def gwpr_fit(y, X, off, coords, nn):
    """GWPR with an adaptive bisquare kernel of `nn` nearest neighbours (Nakaya et al. 2005).
    X without intercept. Returns dict(beta (n,k+1), se, mu, tr_s, dev, aicc)."""
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    k = A.shape[1]
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    nn = int(min(max(nn, k + 2), n))
    beta = np.zeros((n, k))
    se = np.zeros((n, k))
    s_ii = np.zeros(n)
    mu_hat = np.zeros(n)
    bg, _, _ = _local_poisson(y, A, off, np.ones(n))
    for i in range(n):
        h = np.partition(D[i], nn - 1)[nn - 1] * 1.0000001 + 1e-9
        w = _bisquare(D[i], h)
        b, M, mu = _local_poisson(y, A, off, w, b0=bg)
        beta[i] = b
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)
        mu_hat[i] = math.exp(min(30.0, float(A[i] @ b + off[i])))
        # hat-matrix diagonal for the IRLS at convergence: x_i (X'W_iAX)^-1 x_i' w_ii mu_i
        s_ii[i] = float(A[i] @ Minv @ A[i]) * w[i] * mu_hat[i]
        # sandwich-free local SE (Nakaya 2005, approximate): sqrt(diag(M^-1 X'W^2 A X M^-1))
        C = Minv @ ((A.T * (w * w * mu)) @ A) @ Minv
        se[i] = np.sqrt(np.maximum(np.diag(C), 0))
    tr = float(s_ii.sum())
    dev = _deviance(y, mu_hat)
    aicc = dev + 2 * tr + (2 * tr * (tr + 1) / max(n - tr - 1, 1e-9))
    return dict(beta=beta, se=se, mu=mu_hat, tr_s=tr, dev=dev, aicc=aicc, nn=nn)


def gwpr(y, X, off, coords, candidates=None):
    """Choose the number of neighbours by AICc over a grid and fit. Also fits the global Poisson model for comparison."""
    n = len(y)
    k = X.shape[1] + 1
    if candidates is None:
        lo = max(k + 5, int(0.1 * n))
        candidates = sorted({int(v) for v in np.linspace(lo, n, 8)} | {n})
        if n > 600:      # keep the search cheap on large maps
            candidates = sorted({int(v) for v in np.linspace(lo, n, 5)})
    fits = {}
    for c in candidates:
        try:
            fits[c] = gwpr_fit(y, X, off, coords, c)
        except Exception as exc:
            skipped(exc)
            continue
    if not fits:
        return None
    best = min(fits.values(), key=lambda f: f["aicc"])
    A = np.column_stack([np.ones(n), X])
    bg, M, mu_g = _local_poisson(y, A, off, np.ones(n))
    dev_g = _deviance(y, mu_g)
    aicc_g = dev_g + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
    with np.errstate(all="ignore"):
        dev0 = _deviance(y, y.sum() / np.exp(off).sum() * np.exp(off))
    best.update(global_beta=bg, global_aicc=aicc_g, global_dev=dev_g, null_dev=dev0,
                aicc_curve=sorted((c, f["aicc"]) for c, f in fits.items()),
                pseudo_r2=1 - best["dev"] / dev0 if dev0 > 0 else None,
                pseudo_r2_global=1 - dev_g / dev0 if dev0 > 0 else None)
    return best


# ------------------------------------------------------------------ Knox test
def knox_points(xy_km, t_days, d_km, dt_days, sims=999, seed=3, max_n=4000):
    """Knox test for case points: pairs close in space (≤ d_km) and in time (≤ dt_days), Monte Carlo by permuting dates."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy_km, float)
    t = np.asarray(t_days, float)
    if len(t) > max_n:
        k = rng.choice(len(t), max_n, replace=False)
        xy, t = xy[k], t[k]
    n = len(t)
    iu = np.triu_indices(n, 1)
    ds = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))[iu] <= d_km
    obs = int((ds & (np.abs(t[:, None] - t[None, :])[iu] <= dt_days)).sum())
    sim = np.empty(sims)
    for s in range(sims):
        tp = rng.permutation(t)
        sim[s] = (ds & (np.abs(tp[:, None] - tp[None, :])[iu] <= dt_days)).sum()
    pt = (np.abs(t[:, None] - t[None, :])[iu] <= dt_days)
    exp_ = ds.sum() * pt.sum() / max(len(ds), 1)
    return dict(n=n, obs=obs, exp=float(exp_), ratio=obs / exp_ if exp_ > 0 else None,
                p=float((np.sum(sim >= obs) + 1) / (sims + 1)), close_space=int(ds.sum()), close_time=int(pt.sum()))


def _knox_stat(C, Wsp, Wt):
    S = float(np.sum(Wsp * (C @ Wt @ C.T)))
    return (S - C.sum()) / 2.0


def knox_areal(C, nbrs, lag=1, sims=999, seed=5):
    """Knox test for counts in areas x periods: pairs of cases in the same or neighbouring areas and within `lag` periods.
    Monte Carlo: period labels of the cases are shuffled, keeping each area's total and each period's total."""
    C = np.asarray(C, float)
    n, T = C.shape
    Wsp = np.eye(n)
    for i, v in enumerate(nbrs):
        for j in v:
            Wsp[i, j] = 1
    Wt = (np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]) <= lag).astype(float)
    obs = _knox_stat(C, Wsp, Wt)
    Ci = np.round(C).astype(int)
    areas = np.repeat(np.repeat(np.arange(n), T), Ci.ravel())
    pers = np.repeat(np.tile(np.arange(T), n), Ci.ravel())
    rng = np.random.default_rng(seed)
    sim = np.empty(sims)
    for s in range(sims):
        pp = rng.permutation(pers)
        M = np.zeros((n, T))
        np.add.at(M, (areas, pp), 1)
        sim[s] = _knox_stat(M, Wsp, Wt)
    e = float(sim.mean())
    return dict(obs=obs, exp=e, ratio=obs / e if e > 0 else None, p=float((np.sum(sim >= obs) + 1) / (sims + 1)),
                cases=int(Ci.sum()), periods=T, lag=lag)


# ------------------------------------------------------------------ kernel density and risk ratio
def _gauss_blur(a, sigma_px):
    """Separable Gaussian blur (truncated at 3 sigma) with nan-free input."""
    if sigma_px <= 0.3:
        return a.copy()
    r = int(math.ceil(3 * sigma_px))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)


def bandwidth_scott(xy_m):
    """Scott's rule for 2-D points (metres)."""
    xy = np.asarray(xy_m, float)
    n = max(len(xy), 2)
    s = float(np.sqrt(0.5 * (xy[:, 0].var() + xy[:, 1].var())))
    return max(s * n ** (-1 / 6), 1.0)


def bandwidth_cv(rc, mask, res_m, base_m, factors=(0.15, 0.25, 0.35, 0.5, 0.7, 1.0)):
    """Likelihood cross-validation of the kernel bandwidth on the gridded case counts: leave-one-out log density at each case,
    over multiples of Scott's bandwidth. Returns the chosen bandwidth (m) and the curve."""
    n = float(rc[mask].sum())
    if n < 10:
        return base_m, []
    curve = []
    for f in factors:
        bw = max(base_m * f, 1.5 * res_m)
        s = bw / res_m
        k = _gauss_blur(np.where(mask, rc, 0.0), s)
        # self-contribution of one case at the centre of its cell: centre weight of the 2-D kernel
        r = int(math.ceil(3 * s))
        x = np.arange(-r, r + 1)
        k1 = np.exp(-0.5 * (x / s) ** 2) if s > 0.3 else np.array([1.0])
        k1 = k1 / k1.sum()
        self_w = float(k1[len(k1) // 2] ** 2)
        loo = np.maximum(k - self_w, 1e-12) / max(n - 1, 1)
        ll = float((rc[mask] * np.log(loo[mask])).sum())
        curve.append((bw, ll))
    best = max(curve, key=lambda c: c[1])[0]
    return best, curve


def kde_risk(rc, pop, mask, res_m, bw_m, sims=99, seed=9):
    """Case points rasterised as counts `rc`, population grid `pop` (same shape).
    Returns density of cases (per km²), relative risk surface (1 = average) and Monte Carlo tolerance classes:
    1 = significantly high (> 97.5th pct of 'cases follow population' simulations and ratio ≥ 1.25), 2 = high, 3 = average, 4 = low.
    On null data (cases follow population) about 1-2% of the area is flagged."""
    rng = np.random.default_rng(seed)
    P = np.where(mask, np.nan_to_num(pop), 0.0)
    n = int(rc[mask].sum())
    s = bw_m / res_m
    kc = _gauss_blur(np.where(mask, rc, 0.0), s)
    kp = _gauss_blur(P, s)
    km = _gauss_blur(mask.astype(float), s)
    with np.errstate(all="ignore"):
        dens = np.where(mask, kc / np.maximum(km, 1e-9) / (res_m * res_m / 1e6), np.nan)
        ratio = np.where(mask & (kp > 1e-9), (kc / kp) * (P.sum() / max(n, 1)), np.nan)
    prob = P.ravel() / max(P.sum(), 1e-12)
    hi = np.zeros(rc.shape)
    pv_hi = np.zeros(rc.shape)
    flat_idx = np.flatnonzero(prob > 0)
    for _ in range(sims):
        draw = rng.choice(flat_idx, size=n, p=prob[flat_idx] / prob[flat_idx].sum())
        R = np.zeros(rc.size)
        np.add.at(R, draw, 1)
        ks = _gauss_blur(R.reshape(rc.shape), s)
        with np.errstate(all="ignore"):
            rs = (ks / np.maximum(kp, 1e-9)) * (P.sum() / max(n, 1))
        pv_hi += rs >= np.nan_to_num(ratio, nan=np.inf)
    pv_hi = (pv_hi + 1) / (sims + 1)
    cls = np.zeros(rc.shape, np.uint8)
    ok = mask & np.isfinite(ratio)
    cls[ok] = 3
    cls[ok & (ratio >= 1.25)] = 2
    cls[ok & (pv_hi <= 0.025) & (ratio >= 1.25)] = 1
    cls[ok & (ratio <= 0.8)] = 4
    return dict(density=dens, ratio=ratio, p_high=np.where(ok, pv_hi, np.nan), cls=cls, n=n, bw_m=bw_m)


# ------------------------------------------------------------------ 2SFCA
def two_step_fca(dem_xy, dem_pop, sup_xy, sup_cap, d0, decay="gaussian", D=None):
    """Two-step floating catchment area (Luo & Wang 2003; Gaussian distance decay, Dai 2010).
    dem_xy (m, metres), dem_pop (m), sup_xy (k), sup_cap (k), d0 catchment (same units).
    Returns accessibility per demand point (capacity per person) and supply-to-demand ratio per facility."""
    dem_xy, sup_xy = np.asarray(dem_xy, float), np.asarray(sup_xy, float)
    if D is None:      # straight line; pass D (demand × supply, same units as d0) for distances along roads
        D = np.sqrt(((dem_xy[:, None, :] - sup_xy[None, :, :]) ** 2).sum(-1))
    D = np.where(np.isfinite(D), D, np.inf)
    if decay == "gaussian":
        G = np.where(D <= d0, (np.exp(-0.5 * (D / d0) ** 2) - math.exp(-0.5)) / (1 - math.exp(-0.5)), 0.0)
    else:
        G = (D <= d0).astype(float)
    demand = (np.asarray(dem_pop, float)[:, None] * G).sum(0)
    Rj = np.where(demand > 0, np.asarray(sup_cap, float) / np.maximum(demand, 1e-12), 0.0)
    A = (G * Rj[None, :]).sum(1)
    return A, Rj
