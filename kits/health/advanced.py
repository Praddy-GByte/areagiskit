# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Advanced small-area methods: age standardisation (direct with Fay–Feuer intervals, indirect SMR), Kulldorff's space-time
scan (Poisson), and BYM2 Bayesian disease mapping (Riebler et al. 2016) by MCMC. Pure numpy."""

import math
import re

import numpy as np

# ------------------------------------------------------------------ special functions
def _gammainc_lower_reg(a, x):
    """Regularised lower incomplete gamma P(a, x) (Numerical Recipes series / continued fraction)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return min(1.0, s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return max(0.0, 1.0 - math.exp(-x + a * math.log(x) - gln) * h)


def gamma_ppf(q, a):
    """Quantile of Gamma(shape a, scale 1) by bisection on P(a, x)."""
    if a <= 0:
        return 0.0
    lo, hi = 0.0, max(10.0, a * 10 + 50)
    while _gammainc_lower_reg(a, hi) < q:
        hi *= 2
    for _ in range(2000):
        mid = math.sqrt(lo * hi) if lo > 0 else (0.5 * hi if hi > 1e-300 else 0.0)   # geometric steps resolve tiny quantiles
        if mid <= 0:
            break
        if _gammainc_lower_reg(a, mid) < q:
            lo = mid
        else:
            hi = mid
        if lo > 0 and hi / lo - 1 < 1e-12:
            break
    return math.sqrt(lo * hi) if lo > 0 else hi


def chi2_ppf(q, k):
    return 2.0 * gamma_ppf(q, k / 2.0)


# ------------------------------------------------------------------ age standardisation
# WHO World Standard Population 2000–2025 (Ahmad et al. 2001), per 100,000, 5-year bands, last band 100+.
WHO_STD = [8860, 8690, 8600, 8470, 8220, 7930, 7610, 7150, 6590, 6040, 5370, 4550, 3720, 2960, 2210, 1520, 910, 440, 150, 40, 5]


def parse_groups(text):
    """'0-4,5-14,15-44,45-64,65+' -> list of (label, low_age, high_age_exclusive)."""
    out = []
    for raw in [t.strip() for t in str(text or "").split(",") if t.strip()]:
        s = raw.replace(" ", "")
        m = re.fullmatch(r"(\d+)\s*(?:\+|plus|andover|_plus)", s, re.I)
        if m:
            out.append((raw, int(m.group(1)), 200))
            continue
        m = re.fullmatch(r"(\d+)[-_–to]+(\d+)", s, re.I)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise ValueError(f"Age group '{raw}': upper age below lower age.")
            out.append((raw, a, b + 1))
            continue
        m = re.fullmatch(r"(?:<|under)(\d+)", s, re.I)
        if m:
            out.append((raw, 0, int(m.group(1))))
            continue
        m = re.fullmatch(r"(\d+)", s)
        if m:
            a = int(m.group(1))
            out.append((raw, a, a + 1))
            continue
        raise ValueError(f"Could not read age group '{raw}'. Use e.g. 0-4,5-14,15-44,45-64,65+.")
    out.sort(key=lambda g: g[1])
    for (l1, a1, b1), (l2, a2, b2) in zip(out[:-1], out[1:]):
        if a2 < b1:
            raise ValueError(f"Age groups '{l1}' and '{l2}' overlap.")
    return out


def who_weights(groups):
    """Share of the WHO standard population in each user group (arbitrary bounds, partial 5-year bands pro rata)."""
    w = []
    for _, a, b in groups:
        s = 0.0
        for k, v in enumerate(WHO_STD):
            lo, hi = 5 * k, (5 * k + 5) if k < len(WHO_STD) - 1 else 200
            ov = max(0, min(b, hi) - max(a, lo))
            if ov > 0:
                s += v * ov / (hi - lo)
        w.append(s)
    w = np.array(w, float)
    covered = w.sum() / sum(WHO_STD)
    return w / w.sum(), covered


def column_names(label, kind):
    """Accepted column names for an age group, e.g. cases_0-4, cases_0_4, cases0_4, cases_65+, cases_65plus."""
    s = label.strip()
    v = {s, s.replace("-", "_"), s.replace("+", "plus"), s.replace("+", "_plus"), s.replace("-", "_").replace("+", "plus"),
         s.replace("+", ""), s.replace("-", "to")}
    return [f"{kind}_{x}" for x in v] + [f"{kind}{x}" for x in v]


def direct_asr(cg, pg, w, per):
    """Direct age-standardised rates with Fay & Feuer (1997) gamma 95% intervals. cg, pg: (n, G)."""
    with np.errstate(all="ignore"):
        rg = np.where(pg > 0, cg / pg, 0.0)
        y = (rg * w).sum(1)
        v = (np.where(pg > 0, cg / pg ** 2, 0.0) * w ** 2).sum(1)
        wm = np.where(pg > 0, w / pg, 0.0).max(1)
    lo, hi = np.zeros(len(y)), np.zeros(len(y))
    for i in range(len(y)):
        if not np.isfinite(y[i]) or wm[i] <= 0:
            lo[i] = hi[i] = np.nan
            continue
        if y[i] > 0 and v[i] > 0:
            lo[i] = v[i] / (2 * y[i]) * chi2_ppf(0.025, 2 * y[i] ** 2 / v[i])
        else:
            lo[i] = 0.0
        yy, vv = y[i] + wm[i], v[i] + wm[i] ** 2
        hi[i] = vv / (2 * yy) * chi2_ppf(0.975, 2 * yy ** 2 / vv)
    return y * per, lo * per, hi * per


def indirect_expected(cg, pg):
    """Expected cases from the study's own age-specific rates (indirect standardisation)."""
    rs = cg.sum(0) / np.maximum(pg.sum(0), 1e-12)
    return (pg * rs).sum(1), rs


# ------------------------------------------------------------------ space-time scan
def _llr(c, e, C):
    with np.errstate(all="ignore"):
        v = np.where((c > e) & (c > 0),
                     c * np.log(c / e) + np.where(C - c > 0, (C - c) * np.log((C - c) / (C - e)), 0.0), 0.0)
    return np.nan_to_num(v)


def st_scan(xy, c, p, max_pop=0.5, max_time=0.5, sims=999, seed=7, max_clusters=5, max_k=None, model="poisson"):
    """Kulldorff space-time scan statistic, Poisson model (Kulldorff et al. 1998 / 2001), retrospective.
    c, p: (n areas, T periods). Expected counts are adjusted for the overall time trend (each period's own rate), so a
    cluster means more cases in a place *and* a time window than that period's rate predicts. Cylinders: circles through
    nearest neighbours up to `max_pop` of the population × every window of up to `max_time` of the periods.
    Monte Carlo: cases redistributed within each period by population."""
    n, T = c.shape
    C = float(c.sum())
    Ct = c.sum(0)
    if model == "permutation":
        # Kulldorff et al. (2005): case data only; expected = area total x period total / grand total. Circle size is limited
        # by the share of all cases (no population needed); Monte Carlo permutes the period of each case.
        Ci = c.sum(1)
        mu = np.outer(Ci, Ct) / max(C, 1e-12)
        p = np.repeat(Ci[:, None], T, 1) / max(T, 1)
    else:
        Pt = p.sum(0)
        mu = p * np.where(Pt > 0, Ct / np.maximum(Pt, 1e-12), 0.0)
    Pt = p.sum(0)
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, 1)
    K = n if max_k is None else min(n, max_k)
    order = order[:, :K]
    pop = p.sum(1)
    cp = np.cumsum(pop[order], 1)
    valid = cp <= max_pop * pop.sum()
    valid[:, 0] = True
    Lmax = max(1, int(math.floor(max_time * T)))
    wins = [(a, a + L) for L in range(1, Lmax + 1) for a in range(T - L + 1)]

    def scan(cc):
        best = np.zeros((len(wins), n, K))
        for w, (a, b) in enumerate(wins):
            cw = cc[:, a:b].sum(1)
            ew = mu[:, a:b].sum(1)
            best[w] = np.where(valid, _llr(np.cumsum(cw[order], 1), np.cumsum(ew[order], 1), C), 0.0)
        return best
    L = scan(c)
    rng = np.random.default_rng(seed)
    maxs = np.empty(sims)
    if model == "permutation":
        ci = np.repeat(np.repeat(np.arange(n), T), np.round(c).astype(int).ravel())   # area of each case
        ti = np.repeat(np.tile(np.arange(T), n), np.round(c).astype(int).ravel())     # period of each case
    for s in range(sims):
        if model == "permutation":
            sim = np.bincount(ci * T + rng.permutation(ti), minlength=n * T).reshape(n, T).astype(float)
        else:
            sim = np.zeros_like(c, dtype=float)
            for t in range(T):
                if Ct[t] > 0 and Pt[t] > 0:
                    sim[:, t] = rng.multinomial(int(round(Ct[t])), p[:, t] / Pt[t])
        maxs[s] = scan(sim).max()
    out, used = [], set()
    flat = np.argsort(L, axis=None)[::-1]
    for f in flat[:200000]:
        w, i, k = np.unravel_index(int(f), L.shape)
        if L[w, i, k] <= 0 or len(out) >= max_clusters:
            break
        mem = order[i, :k + 1]
        a, b = wins[w]
        cells = {(int(m), t) for m in mem for t in range(a, b)}
        if cells & used:
            continue
        o = float(c[mem, a:b].sum())
        e = float(mu[mem, a:b].sum())
        rr = (o / e) / ((C - o) / max(C - e, 1e-9)) if C - o > 0 else float("inf")
        out.append(dict(centre=int(i), members=mem.tolist(), t0=a, t1=b - 1, obs=o, exp=e, rr=float(rr), llr=float(L[w, i, k]), model=model,
                        p=float((np.sum(maxs >= L[w, i, k]) + 1) / (sims + 1)), pop=float(pop[mem].sum())))
        used |= cells
    return out


# ------------------------------------------------------------------ BYM2
def _components(nbrs):
    n = len(nbrs)
    comp = -np.ones(n, int)
    k = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = k
        while stack:
            i = stack.pop()
            for j in nbrs[i]:
                if comp[j] < 0:
                    comp[j] = k
                    stack.append(j)
        k += 1
    return comp, k


def icar_scale(nbrs):
    """Riebler et al. (2016) scaling: geometric mean of the marginal variances of the sum-to-zero ICAR, per component."""
    n = len(nbrs)
    comp, k = _components(nbrs)
    var = np.ones(n)
    for c in range(k):
        idx = np.flatnonzero(comp == c)
        if len(idx) == 1:
            continue
        pos = {v: i for i, v in enumerate(idx)}
        Q = np.zeros((len(idx), len(idx)))
        for a in idx:
            for b in nbrs[a]:
                if b in pos:
                    Q[pos[a], pos[b]] = -1
            Q[pos[a], pos[a]] = len(nbrs[a])
        A = Q + np.ones_like(Q) / len(idx)          # constrained generalised inverse
        G = np.linalg.inv(A) - np.ones_like(Q) / len(idx)
        var[idx] = np.diag(G)
    s = float(np.exp(np.mean(np.log(np.maximum(var[[len(nbrs[i]) > 0 for i in range(n)]], 1e-12)))))
    return s, comp, k


def _colour(nbrs):
    col = -np.ones(len(nbrs), int)
    for i in np.argsort([-len(v) for v in nbrs]):
        used = {col[j] for j in nbrs[i] if col[j] >= 0}
        c = 0
        while c in used:
            c += 1
        col[i] = c
    return col


def _rhat(ch):
    """Split R-hat (Gelman et al. 2013) for chains (m, n) of one scalar."""
    m, n = ch.shape
    h = n // 2
    s = np.concatenate([ch[:, :h], ch[:, h:2 * h]], 0)
    W = s.var(1, ddof=1).mean()
    B = h * s.mean(1).var(ddof=1)
    if W <= 0:
        return 1.0
    return float(math.sqrt(((h - 1) / h * W + B / h) / W))


def bym2(y, E, nbrs, iters=None, chains=2, seed=3, sigma_u=1.0, sigma_alpha=0.01, X=None, beta_sd=1.0):
    """BYM2 Poisson model: y ~ Poisson(E exp(b0 + b)), b = sigma (sqrt(1-phi) v + sqrt(phi) u), v iid N(0,1),
    u scaled ICAR (sum to zero per connected part). Priors: PC prior on sigma (P(sigma > sigma_u) = sigma_alpha),
    phi ~ Beta(1,1), b0 flat. Metropolis-within-Gibbs: v vectorised, u updated by independent colour classes,
    hyperparameters by random-walk Metropolis; step sizes tuned in burn-in.
    X (n, k), standardised covariates: eta = b0 + X beta + b, beta_j ~ N(0, beta_sd^2) (weakly informative on the log scale
    per standard deviation); each beta updated by random-walk Metropolis. exp(beta) = rate ratio per 1 SD of the covariate."""
    y = np.asarray(y, float)
    E = np.maximum(np.asarray(E, float), 1e-9)
    n = len(y)
    X = None if X is None or np.size(X) == 0 else np.asarray(X, float).reshape(n, -1)
    kx = 0 if X is None else X.shape[1]
    if iters is None:
        iters = 20000 if n <= 300 else 10000 if n <= 1500 else 6000
    burn = iters // 2
    s, comp, ncomp = icar_scale(nbrs)
    col = _colour(nbrs)
    classes = [np.flatnonzero(col == c) for c in range(col.max() + 1)]
    deg = np.array([len(v) for v in nbrs], float)
    has = deg > 0
    # neighbour sums via sparse lists
    rows = np.repeat(np.arange(n), [len(v) for v in nbrs])
    cols = np.concatenate([np.array(v, int) for v in nbrs]) if rows.size else np.zeros(0, int)
    lam = -math.log(sigma_alpha) / sigma_u

    def nsum(u):
        return np.bincount(rows, weights=u[cols], minlength=n)

    def loglik_vec(eta, idx=None):
        if idx is None:
            return y * eta - E * np.exp(eta)
        return y[idx] * eta - E[idx] * np.exp(eta)

    keep = max(1, (iters - burn) // 1000)
    out_b, out_h, out_beta = [], [], []
    rng = np.random.default_rng(seed)
    acc_tot = []
    for ch in range(chains):
        b0 = math.log(max(y.sum(), 0.5) / E.sum()) + rng.normal(0, 0.1)
        sig, phi = 0.5 * math.exp(rng.normal(0, 0.3)), min(0.9, max(0.1, 0.5 + rng.normal(0, 0.15)))
        v = rng.normal(0, 0.1, n)
        u = rng.normal(0, 0.1, n)
        u[has] -= 0
        sv, su, sh = np.full(n, 0.5), np.full(n, 0.5), np.array([0.05, 0.1, 0.2])   # step sizes: v_i, u_i, (b0, log sig, logit phi)
        av, au, ah = np.zeros(n), np.zeros(n), np.zeros(3)
        beta = np.zeros(kx)
        sbeta, abeta = np.full(kx, 0.05), np.zeros(kx)
        xb = np.zeros(n)
        cb, ch_h, cbeta = [], [], []

        def bfield(v_, u_, sig_, phi_):
            return sig_ * (math.sqrt(1 - phi_) * v_ + math.sqrt(phi_) * u_)
        eta = b0 + bfield(v, u, sig, phi)
        for it in range(iters):
            # v (independent given everything else)
            pv = v + sv * rng.normal(size=n)
            eta_p = eta + sig * math.sqrt(1 - phi) * (pv - v)
            lr = loglik_vec(eta_p) - loglik_vec(eta) - 0.5 * (pv ** 2 - v ** 2)
            ok = np.log(rng.random(n)) < lr
            v = np.where(ok, pv, v)
            eta = np.where(ok, eta_p, eta)
            av += ok
            # u by colour classes (conditionally independent given neighbours)
            for cls in classes:
                cls = cls[has[cls]]
                if not cls.size:
                    continue
                ns = nsum(u)[cls]
                pu = u[cls] + su[cls] * rng.normal(size=cls.size)
                d_eta = sig * math.sqrt(phi) * (pu - u[cls])
                # ICAR prior (precision s): -s/2 * sum_j (u_i - u_j)^2 = -s/2 * (k u_i^2 - 2 u_i nsum) + const
                lp_new = -0.5 * s * (deg[cls] * pu ** 2 - 2 * pu * ns)
                lp_old = -0.5 * s * (deg[cls] * u[cls] ** 2 - 2 * u[cls] * ns)
                lr = loglik_vec(eta[cls] + d_eta, cls) - loglik_vec(eta[cls], cls) + lp_new - lp_old
                ok = np.log(rng.random(cls.size)) < lr
                u[cls] = np.where(ok, pu, u[cls])
                eta[cls] = np.where(ok, eta[cls] + d_eta, eta[cls])
                au[cls] += ok
            # centre u within each connected part (sum-to-zero), absorbing the shift into b0 is not needed: eta recomputed
            for c in range(ncomp):
                m = comp == c
                if m.sum() > 1:
                    u[m] -= u[m].mean()
            eta = b0 + xb + bfield(v, u, sig, phi)
            # covariates
            for j in range(kx):
                nbj = beta[j] + sbeta[j] * rng.normal()
                d_ = (nbj - beta[j]) * X[:, j]
                lr = (loglik_vec(eta + d_).sum() - loglik_vec(eta).sum()
                      - 0.5 * (nbj ** 2 - beta[j] ** 2) / beta_sd ** 2)
                if math.log(rng.random()) < lr:
                    beta[j] = nbj
                    eta = eta + d_
                    xb = xb + d_
                    abeta[j] += 1
            # hyperparameters
            ll = loglik_vec(eta).sum()
            # b0 (flat prior)
            nb0 = b0 + sh[0] * rng.normal()
            ll_n = loglik_vec(eta + (nb0 - b0)).sum()
            if math.log(rng.random()) < ll_n - ll:
                eta += nb0 - b0
                b0, ll = nb0, ll_n
                ah[0] += 1
            # sigma: PC prior (exponential on sigma), random walk on log sigma (Jacobian sigma)
            ls = math.log(sig) + sh[1] * rng.normal()
            ns_ = math.exp(ls)
            eta_n = b0 + xb + bfield(v, u, ns_, phi)
            ll_n = loglik_vec(eta_n).sum()
            if math.log(rng.random()) < (ll_n - lam * ns_ + ls) - (ll - lam * sig + math.log(sig)):
                sig, eta, ll = ns_, eta_n, ll_n
                ah[1] += 1
            # phi: Beta(1,1), random walk on logit (Jacobian phi(1-phi))
            lg = math.log(phi / (1 - phi)) + sh[2] * rng.normal()
            nphi = min(1 - 1e-6, max(1e-6, 1 / (1 + math.exp(-lg))))
            eta_n = b0 + xb + bfield(v, u, sig, nphi)
            ll_n = loglik_vec(eta_n).sum()
            if math.log(rng.random()) < (ll_n + math.log(nphi * (1 - nphi))) - (ll + math.log(phi * (1 - phi))):
                phi, eta, ll = nphi, eta_n, ll_n
                ah[2] += 1
            # adapt during burn-in every 50 iterations towards ~0.44 acceptance
            if it < burn and (it + 1) % 50 == 0:
                sv *= np.exp(np.clip(av / 50 - 0.44, -0.5, 0.5))
                su *= np.exp(np.clip(au / 50 - 0.44, -0.5, 0.5))
                sh *= np.exp(np.clip(ah / 50 - 0.44, -0.5, 0.5))
                sbeta *= np.exp(np.clip(abeta / 50 - 0.44, -0.5, 0.5))
                av[:], au[:], ah[:], abeta[:] = 0, 0, 0, 0
            if it == burn:
                av[:], au[:], ah[:], abeta[:] = 0, 0, 0, 0
            if it >= burn and (it - burn) % keep == 0:
                cb.append(b0 + xb + bfield(v, u, sig, phi))
                ch_h.append((b0, sig, phi))
                cbeta.append(beta.copy())
        acc_tot.append((float(av.mean() / (iters - burn)), float(au[has].mean() / (iters - burn)) if has.any() else None))
        out_b.append(np.array(cb))
        out_h.append(np.array(ch_h))
        out_beta.append(np.array(cbeta).reshape(len(cbeta), kx))
    m = min(len(a) for a in out_b)
    B = np.stack([a[:m] for a in out_b])           # (chains, draws, n): log relative risk incl. intercept
    H = np.stack([a[:m] for a in out_h])           # (chains, draws, 3)
    rr = np.exp(B)
    flat = rr.reshape(-1, n)
    rh_rr = np.array([_rhat(np.log(rr[:, :, i])) for i in range(n)])
    rh_h = [_rhat(H[:, :, j]) for j in range(3)]
    beta_out = None
    if kx:
        Bt = np.stack([a[:m] for a in out_beta])        # (chains, draws, k)
        fb = Bt.reshape(-1, kx)
        rh_b = [_rhat(Bt[:, :, j]) for j in range(kx)]
        rh_h = rh_h + rh_b
        beta_out = dict(mean=fb.mean(0), lo=np.percentile(fb, 2.5, 0), hi=np.percentile(fb, 97.5, 0),
                        irr=np.exp(fb).mean(0), irr_lo=np.exp(np.percentile(fb, 2.5, 0)), irr_hi=np.exp(np.percentile(fb, 97.5, 0)),
                        p_pos=(fb > 0).mean(0), rhat=rh_b)
    return dict(beta=beta_out,rr=flat.mean(0), lo=np.percentile(flat, 2.5, 0), hi=np.percentile(flat, 97.5, 0), p_gt1=(flat > 1).mean(0),
                phi=float(H[:, :, 2].mean()), phi_ci=(float(np.percentile(H[:, :, 2], 2.5)), float(np.percentile(H[:, :, 2], 97.5))),
                sigma=float(H[:, :, 1].mean()), sigma_ci=(float(np.percentile(H[:, :, 1], 2.5)), float(np.percentile(H[:, :, 1], 97.5))),
                rhat_max=float(max(rh_rr.max(), *rh_h)), rhat_hyper=rh_h, rhat_rr_share_ok=float((rh_rr < 1.1).mean()),
                draws=int(m * len(out_b)), iters=iters, chains=chains, scale=s, accept=acc_tot)
