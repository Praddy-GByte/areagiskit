# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Forecast and early warning for small areas with several periods of counts.

Model (Farrington-like, simplified): log E[C_it] = log P_it + a_i + b·t, with an area effect a_i and one common trend b,
fitted by maximum likelihood (profile over b; a_i in closed form), and over-dispersion phi = Pearson chi-square / df
(quasi-Poisson, never below 1). Prediction variance combines the count noise (phi·mu) and the uncertainty of the fitted
mean (mu²·var(log mu)).

Alert: the model is fitted on all periods except the latest, the latest is predicted, and an area is flagged when its
observed count exceeds the upper 99% prediction bound (as in Farrington et al. 1996) and is at least 3 cases. Forecast: the model refitted on all
periods projects the next period with a 95% prediction interval (population of the latest period carried forward)."""

import math

import numpy as np

Z975 = 1.959964


def fit(C, P):
    """C, P: (n areas, T periods). Returns dict(a, b, var_b, phi, var_a) or None."""
    C = np.asarray(C, float)
    P = np.asarray(P, float)
    n, T = C.shape
    ok = (P > 0) & np.isfinite(C)
    if T < 2 or ok.sum() < 2 * n:
        return None
    t = np.arange(T, dtype=float) - (T - 1) / 2.0
    Cs = np.where(ok, C, 0.0)
    Ps = np.where(ok, P, 0.0)
    b = 0.0
    for _ in range(50):
        w = Ps * np.exp(b * t)                     # (n, T)
        sw = w.sum(1)
        a = np.log(np.maximum(Cs.sum(1), 1e-9) / np.maximum(sw, 1e-12))
        mu = np.exp(a)[:, None] * w
        g = (Cs * t - mu * t).sum()                 # score for b with a profiled
        tbar = (w * t).sum(1) / np.maximum(sw, 1e-12)
        h = (mu * (t - tbar[:, None]) ** 2).sum()   # profile information
        if h <= 0:
            break
        step = g / h
        b += step
        if abs(step) < 1e-10:
            break
    w = Ps * np.exp(b * t)
    sw = w.sum(1)
    a = np.log(np.maximum(Cs.sum(1), 0.5) / np.maximum(sw, 1e-12))
    mu = np.exp(a)[:, None] * w
    df = max(int(ok.sum()) - n - 1, 1)
    phi = max(float((np.where(ok, (Cs - mu) ** 2 / np.maximum(mu, 1e-12), 0)).sum() / df), 1.0)
    tbar = (w * t).sum(1) / np.maximum(sw, 1e-12)
    h = (mu * (t - tbar[:, None]) ** 2).sum()
    var_b = phi / h if h > 0 else np.inf
    var_a = phi / np.maximum(mu.sum(1), 1e-9)
    return dict(a=a, b=float(b), var_b=float(var_b), phi=phi, var_a=var_a, t=t, tbar=tbar, T=T)


def predict(m, P_new, t_new):
    """Mean and 95% prediction interval for one new period at time index t_new (in the fit's centred scale)."""
    mu = np.exp(m["a"] + m["b"] * t_new) * np.asarray(P_new, float)
    var_log = m["var_a"] + (t_new - m["tbar"]) ** 2 * m["var_b"]
    var = m["phi"] * mu + mu ** 2 * var_log
    sd = np.sqrt(np.maximum(var, 0))
    return mu, np.maximum(mu - Z975 * sd, 0), mu + Z975 * sd


Z99 = 2.326348


def early_warning(C, P, min_cases=3, z=Z99):
    """Fit on periods 1..T-1, test the latest period. Returns dict(expected, upper, observed, alert) or None."""
    C = np.asarray(C, float)
    P = np.asarray(P, float)
    if C.shape[1] < 3:
        return None
    m = fit(C[:, :-1], P[:, :-1])
    if m is None:
        return None
    t_last = m["t"][-1] + 1.0
    mu = np.exp(m["a"] + m["b"] * t_last) * np.asarray(P[:, -1], float)
    var_log = m["var_a"] + (t_last - m["tbar"]) ** 2 * m["var_b"]
    hi = mu + z * np.sqrt(np.maximum(m["phi"] * mu + mu ** 2 * var_log, 0))
    obs = C[:, -1]
    alert = (obs > hi) & (obs >= min_cases) & (P[:, -1] > 0)
    return dict(expected=mu, upper=hi, observed=obs, alert=alert, phi=m["phi"], trend=math.exp(m["b"]))


def forecast_next(C, P):
    """Refit on all periods and project the next one (population of the last period carried forward)."""
    C = np.asarray(C, float)
    P = np.asarray(P, float)
    m = fit(C, P)
    if m is None:
        return None
    mu, lo, hi = predict(m, P[:, -1], m["t"][-1] + 1.0)
    return dict(mean=mu, lo=lo, hi=hi, trend=math.exp(m["b"]), trend_ci=(math.exp(m["b"] - Z975 * math.sqrt(m["var_b"])),
                                                                          math.exp(m["b"] + Z975 * math.sqrt(m["var_b"]))),
                phi=m["phi"])
