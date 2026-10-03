"""Step 3: differential candidates = OPLS-DA VIP >= 1 AND limma (moderated t) FDR < 0.05."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .stats import bh_adjust


# ---------------------------------------------------------------- limma
def _trigamma_inverse(x: float) -> float:
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if -dif / y < 1e-8:
            break
    return float(y)


def fit_f_dist(s2: np.ndarray, df: float):
    """Empirical-Bayes prior (s0^2, d0) from per-feature residual variances (limma::fitFDist)."""
    x = np.maximum(s2, 0)
    x = np.maximum(x, 1e-5 * np.median(x))
    z = np.log(x)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, df / 2)
    if evar > 0:
        d0 = 2 * _trigamma_inverse(evar)
        s0 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s0 = np.inf, np.exp(emean)
    return s0, d0


def limma_two_group(Y: np.ndarray, group: np.ndarray, n_cov: int = 0):
    """Moderated t-test for Y (n x p) between group==1 and group==0. Returns dict of arrays."""
    g = group.astype(bool)
    n1, n0 = g.sum(), (~g).sum()
    n = n1 + n0
    m1, m0 = Y[g].mean(0), Y[~g].mean(0)
    beta = m1 - m0
    ssr = ((Y[g] - m1) ** 2).sum(0) + ((Y[~g] - m0) ** 2).sum(0)
    df = n - 2 - n_cov
    s2 = ssr / df
    sd_unscaled = np.sqrt(1 / n1 + 1 / n0)
    s0, d0 = fit_f_dist(s2, df)
    if np.isinf(d0):
        s2post, dtot = np.full_like(s2, s0), np.inf
    else:
        s2post, dtot = (d0 * s0 + df * s2) / (d0 + df), df + d0
    t = beta / (np.sqrt(s2post) * sd_unscaled)
    p = 2 * stats.t.sf(np.abs(t), dtot)
    return {"logFC": beta, "t": t, "P.Value": p, "adj.P.Val": bh_adjust(p), "s0_sq": s0, "d0": d0}


# ---------------------------------------------------------------- OPLS-DA
def opls_da(X: np.ndarray, y: np.ndarray, n_ortho: int = 1, tol: float = 1e-12):
    """Single-response OPLS (Trygg & Wold) with unit-variance scaling.
    Returns VIP of the predictive component (mean VIP^2 = 1) and diagnostics."""
    Xs = X - X.mean(0)
    sd = Xs.std(0, ddof=1)
    sd[sd < tol] = 1.0
    Xs = Xs / sd
    yc = (y - y.mean()).astype(float)
    yc = yc / yc.std(ddof=1)

    w = Xs.T @ yc / (yc @ yc)
    w /= np.linalg.norm(w)
    Xf = Xs.copy()
    for _ in range(n_ortho):
        t = Xf @ w
        p = Xf.T @ t / (t @ t)
        w_o = p - (w @ p) / (w @ w) * w
        nrm = np.linalg.norm(w_o)
        if nrm < tol:
            break
        w_o /= nrm
        t_o = Xf @ w_o
        p_o = Xf.T @ t_o / (t_o @ t_o)
        Xf = Xf - np.outer(t_o, p_o)
        w = Xf.T @ yc / (yc @ yc)
        w /= np.linalg.norm(w)
    t = Xf @ w
    q = (yc @ t) / (t @ t)
    r2y = float(1 - np.sum((yc - q * t) ** 2) / np.sum(yc ** 2))
    p_load = Xf.T @ t / (t @ t)
    ss = (q ** 2) * (t @ t)
    vip = np.sqrt(Xs.shape[1] * (ss * w ** 2) / ss)     # single predictive component
    return {"vip": vip, "R2Y": r2y, "w": w}


def vip_tier(vip: float, diff: bool) -> str:
    if not diff:
        return "none"
    return "VIP>=3" if vip >= 3 else "VIP2-3" if vip >= 2 else "VIP1-2"


def differential_candidates(L: pd.DataFrame, y: pd.Series, n_cov: int, vip_cutoff: float = 1.0,
                            fdr_cutoff: float = 0.05, n_ortho: int = 1) -> pd.DataFrame:
    Y = L.to_numpy()
    g = y.loc[L.index].to_numpy()
    lim = limma_two_group(Y, g, n_cov)
    op = opls_da(Y, g, n_ortho)
    out = pd.DataFrame({
        "logFC": lim["logFC"], "t": lim["t"], "P.Value": lim["P.Value"], "adj.P.Val": lim["adj.P.Val"],
        "VIP": op["vip"],
    }, index=L.columns)
    out["differential"] = (out["VIP"] >= vip_cutoff) & (out["adj.P.Val"] < fdr_cutoff)
    out["vipTier"] = [vip_tier(v, d) for v, d in zip(out["VIP"], out["differential"])]
    out.attrs["opls_R2Y"] = op["R2Y"]
    out.attrs["limma_d0"] = lim["d0"]
    return out
