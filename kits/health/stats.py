# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Small-area disease statistics: confidence intervals, SMR, excess-risk probability, spatial empirical Bayes, local Moran
(LISA), Kulldorff's spatial scan statistic (Poisson), Poisson regression with population offset, Geodetector q."""

import math

import numpy as np


# ------------------------------------------------------------------ rates
def _chi2_ppf_approx(p, k):
    """Wilson-Hilferty approximation of the chi-square quantile."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def byar_ci(c, p, per=1e5):
    """95% confidence interval of a rate (Byar's approximation to the exact Poisson interval)."""
    lo, hi = np.zeros(len(c)), np.zeros(len(c))
    for i, (x, n) in enumerate(zip(c, p)):
        if not np.isfinite(n) or n <= 0:
            lo[i] = hi[i] = np.nan
            continue
        low = 0.0 if x == 0 else x * (1 - 1 / (9 * x) - 1.96 / (3 * math.sqrt(x))) ** 3
        u = (x + 1) * (1 - 1 / (9 * (x + 1)) + 1.96 / (3 * math.sqrt(x + 1))) ** 3
        lo[i], hi[i] = low / n * per, u / n * per
    return lo, hi


def poisson_sf(k, mu):
    """P(X >= k) for X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    # sum of pmf for 0..k-1 in log space
    s, term = 0.0, math.exp(-mu)
    for i in range(int(k)):
        s += term
        term *= mu / (i + 1)
        if term == 0 and i > mu:
            break
    return max(0.0, 1.0 - s)


def smr(c, p):
    """Expected cases (overall rate x population), SMR and probability of seeing at least this many cases by chance."""
    base = c.sum() / p.sum()
    e = p * base
    s = np.where(e > 0, c / e, np.nan)
    pr = np.array([poisson_sf(int(x), m) for x, m in zip(c, e)])
    return e, s, pr


def eb_global(c, p):
    """Global empirical Bayes rates (Marshall 1991)."""
    b = c.sum() / p.sum()
    r = c / p
    s2 = (p * (r - b) ** 2).sum() / p.sum()
    a = max(s2 - b / p.mean(), 0.0)
    w = a / (a + b / p) if a > 0 else np.zeros_like(p)
    return b + w * (r - b)


def eb_local(c, p, nbrs):
    """Spatial (local) empirical Bayes: each area shrinks towards its neighbourhood rate (Marshall 1991)."""
    n = len(c)
    out = np.zeros(n)
    for i in range(n):
        J = list(nbrs[i]) + [i]
        cj, pj = c[J], p[J]
        b = cj.sum() / pj.sum()
        r = cj / pj
        s2 = (pj * (r - b) ** 2).sum() / pj.sum()
        a = max(s2 - b / pj.mean(), 0.0)
        w = a / (a + b / p[i]) if a > 0 else 0.0
        out[i] = b + w * (c[i] / p[i] - b)
    return out


# ------------------------------------------------------------------ spatial autocorrelation
def _row_std(nbrs):
    return [np.array(sorted(v), int) for v in nbrs]


def morans_i(x, nbrs, perms=999, seed=1):
    """Global Moran's I with row-standardised weights and a permutation p-value (one-sided, positive)."""
    n = len(x)
    z = x - x.mean()
    N = _row_std(nbrs)
    has = np.array([len(v) > 0 for v in N])
    S0 = has.sum()
    if S0 == 0:
        return None, None

    def morans_i(zz):
        lag = np.array([zz[v].mean() if len(v) else 0.0 for v in N])
        return (n / S0) * (zz * lag).sum() / (zz ** 2).sum()
    obs = morans_i(z)
    rng = np.random.default_rng(seed)
    sims = np.array([morans_i(rng.permutation(z)) for _ in range(perms)])
    p = (np.sum(sims >= obs) + 1) / (perms + 1)
    return float(obs), float(p)


def lisa(x, nbrs, perms=999, seed=2):
    """Local Moran's I (Anselin 1995), row-standardised, conditional permutation p-values.
    Returns (Ii, p, quadrant) with quadrant 1 HH, 2 LL, 3 LH, 4 HL."""
    n = len(x)
    z = (x - x.mean()) / (x.std() or 1)
    N = _row_std(nbrs)
    lag = np.array([z[v].mean() if len(v) else 0.0 for v in N])
    Ii = z * lag * (n - 1) / n          # same scaling as PySAL esda.Moran_Local
    rng = np.random.default_rng(seed)
    p = np.ones(n)
    idx = np.arange(n)
    for i in range(n):
        k = len(N[i])
        if k == 0:
            continue
        others = np.delete(idx, i)
        draws = np.array([rng.choice(others, k, replace=False) for _ in range(perms)])
        sim = z[i] * z[draws].mean(1)
        larger = (sim >= Ii[i]).sum() if Ii[i] >= 0 else (sim <= Ii[i]).sum()
        p[i] = (larger + 1) / (perms + 1)
    q = np.where((z > 0) & (lag > 0), 1, np.where((z < 0) & (lag < 0), 2, np.where((z < 0) & (lag > 0), 3, 4)))
    return Ii, p, q


def gi_star(x, nbrs):
    n = len(x)
    xb, s = x.mean(), x.std()
    z = np.zeros(n)
    for i in range(n):
        J = list(nbrs[i]) + [i]
        w = len(J)
        num = x[J].sum() - xb * w
        den = s * math.sqrt((n * w - w * w) / (n - 1)) if n > 1 else 0
        z[i] = num / den if den > 0 else 0
    return z


def fdr(p, q=0.05):
    """Benjamini–Hochberg: boolean array of discoveries."""
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    out = np.zeros(n, bool)
    out[o[:k]] = True
    return out


# ------------------------------------------------------------------ Kulldorff scan
def _llr(c, e, C):
    """Poisson log-likelihood ratio for circles with c observed, e expected (total C); only high-rate circles."""
    with np.errstate(all="ignore"):
        v = np.where((c > e) & (c > 0),
                     c * np.log(c / e) + np.where(C - c > 0, (C - c) * np.log((C - c) / (C - e)), 0.0), 0.0)
    return np.nan_to_num(v)


def kulldorff(xy, c, p, max_pop=0.5, sims=999, seed=7, max_clusters=5):
    """Kulldorff's circular spatial scan statistic, Poisson model (Kulldorff 1997). Circles are centred on each area and grow
    through its nearest neighbours up to `max_pop` of the population. Returns a list of clusters (most likely first,
    then secondary clusters that do not overlap) with members, observed, expected, relative risk, LLR and Monte Carlo p."""
    n = len(c)
    C = float(c.sum())
    P = float(p.sum())
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, 1)
    cp = np.cumsum(p[order], 1)
    valid = cp <= max_pop * P
    valid[:, 0] = True
    E = cp / P * C

    def best(cases):
        cc = np.cumsum(cases[order], 1)
        L = np.where(valid, _llr(cc, E, cases.sum()), 0.0)
        return L
    L = best(c)
    # Monte Carlo: redistribute all cases by population (multinomial)
    rng = np.random.default_rng(seed)
    pr = p / P
    maxs = np.empty(sims)
    for s in range(sims):
        sim = rng.multinomial(int(round(C)), pr).astype(float)
        maxs[s] = best(sim).max()
    out, used = [], np.zeros(n, bool)
    flat = np.argsort(L, axis=None)[::-1]
    for f in flat:
        i, k = divmod(int(f), n)
        if L[i, k] <= 0 or len(out) >= max_clusters:
            break
        mem = order[i, :k + 1]
        if used[mem].any():
            continue
        o, e = float(c[mem].sum()), float(E[i, k])
        pval = (np.sum(maxs >= L[i, k]) + 1) / (sims + 1)
        rr = (o / e) / ((C - o) / max(C - e, 1e-9)) if C - o > 0 else float("inf")
        out.append(dict(centre=int(i), members=mem.tolist(), radius=float(math.sqrt(d[i, mem[-1]])), obs=o, exp=e,
                        rr=float(rr), llr=float(L[i, k]), p=float(pval), pop=float(p[mem].sum())))
        used[mem] = True
    return out


# ------------------------------------------------------------------ regression
def poisson_glm(y, X, offset, names, iters=50):
    """Poisson regression with log link and log(population) offset (IRLS). Returns rows of (name, IRR, 95% CI, p),
    deviance, null deviance, pseudo-R² and fitted values. X must not include the intercept."""
    from statistics import NormalDist
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    b = np.zeros(A.shape[1])
    b[0] = math.log(max(y.sum(), 1e-9) / np.exp(offset).sum())
    for _ in range(iters):
        eta = A @ b + offset
        mu = np.exp(np.clip(eta, -30, 30))
        W = mu
        zt = eta - offset + (y - mu) / np.maximum(mu, 1e-12)
        AtW = A.T * W
        try:
            nb = np.linalg.solve(AtW @ A + 1e-9 * np.eye(A.shape[1]), AtW @ zt)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(nb - b)) < 1e-8:
            b = nb
            break
        b = nb
    mu = np.exp(np.clip(A @ b + offset, -30, 30))
    with np.errstate(all="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
        mu0 = y.sum() / np.exp(offset).sum() * np.exp(offset)
        dev0 = 2 * np.sum(np.where(y > 0, y * np.log(y / mu0), 0) - (y - mu0))
    df = max(n - A.shape[1], 1)
    disp = max(float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-12)) / df), 1.0)   # quasi-Poisson: inflate for overdispersion
    cov = np.linalg.pinv((A.T * mu) @ A) * disp
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    rows = []
    nd = NormalDist()
    for j, nm in enumerate(["(intercept)"] + list(names)):
        z = b[j] / se[j] if se[j] > 0 else 0.0
        p = 2 * (1 - nd.cdf(abs(z)))
        rows.append((nm, math.exp(b[j]), math.exp(b[j] - 1.96 * se[j]), math.exp(b[j] + 1.96 * se[j]), p))
    return dict(rows=rows, deviance=float(dev), null=float(dev0), r2=float(1 - dev / dev0) if dev0 > 0 else None,
                dispersion=disp, fitted=mu)


def geodetector_q(y, x, bins=5):
    """Geodetector factor detector q (Wang et al. 2010): share of the variance of y explained by strata of x."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if y.size < 10 or y.var() == 0:
        return None
    qs = np.unique(np.percentile(x, np.linspace(0, 100, bins + 1)))
    s = np.clip(np.digitize(x, qs[1:-1]), 0, bins - 1)
    within = sum(((y[s == k] - y[s == k].mean()) ** 2).sum() for k in np.unique(s))
    return float(1 - within / ((y - y.mean()) ** 2).sum())


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
