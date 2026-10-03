"""Steps 8 / 8b: p-gain of metabolite ratios (Krumsiek et al. 2011) w.r.t. MetS."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import corr_to_p


def single_assoc(L: pd.DataFrame, y: pd.Series, n_cov: int = 0):
    """Per-metabolite point-biserial association with MetS: r, p."""
    X = L.to_numpy()
    yv = y.loc[L.index].to_numpy(dtype=float)
    Xc, yc = X - X.mean(0), yv - yv.mean()
    cov_y = Xc.T @ yc / (len(yv) - 1)
    r = cov_y / (Xc.std(0, ddof=1) * yc.std(ddof=1))
    return r, corr_to_p(r, len(yv) - 2 - n_cov), cov_y


def _tier(pgain, n_tests, alpha, suggestive):
    strong_cut = n_tests / (2 * alpha)       # B / 2alpha  (= 10*B for alpha = 0.05)
    return np.where(pgain >= strong_cut, 2, np.where(pgain >= suggestive, 1, 0)), strong_cut


def _ratio_stats(Xc, yc, var_x, cov_y, cov_xx_ij, i, j, n, n_cov):
    """log-ratio r_i - r_j vs y: correlation, p, direction (sign of slope) for index pairs (i, j)."""
    cov_ratio_y = cov_y[i] - cov_y[j]
    var_ratio = var_x[i] + var_x[j] - 2 * cov_xx_ij
    r = cov_ratio_y / np.sqrt(np.maximum(var_ratio, 1e-300) * yc.var(ddof=1))
    return r, corr_to_p(r, n - 2 - n_cov), np.sign(cov_ratio_y)


def pgain_pairs(L: pd.DataFrame, y: pd.Series, pairs: np.ndarray, columns, n_cov: int,
                alpha: float, suggestive: float, n_tests: int) -> pd.DataFrame:
    """p-gain for explicit index pairs (into `columns` of L) -> DataFrame."""
    X = L[columns].to_numpy()
    n = len(X)
    yv = y.loc[L.index].to_numpy(dtype=float)
    Xc, yc = X - X.mean(0), yv - yv.mean()
    var_x = Xc.var(0, ddof=1)
    cov_y = Xc.T @ yc / (n - 1)
    r_single = cov_y / np.sqrt(var_x * yc.var(ddof=1))
    p_single = corr_to_p(r_single, n - 2 - n_cov)
    i, j = pairs[:, 0], pairs[:, 1]
    cov_ij = np.einsum("ij,ij->j", Xc[:, i], Xc[:, j]) / (n - 1)
    r, p_ratio, dirn = _ratio_stats(Xc, yc, var_x, cov_y, cov_ij, i, j, n, n_cov)
    p_ratio = np.maximum(p_ratio, 1e-300)
    pg = np.minimum(p_single[i], p_single[j]) / p_ratio
    tier, strong_cut = _tier(pg, n_tests, alpha, suggestive)
    cols = np.asarray(columns)
    return pd.DataFrame({
        "metabolite_a": cols[i], "metabolite_b": cols[j],
        "p_a": p_single[i], "p_b": p_single[j], "p_ratio": p_ratio,
        "pgain": pg, "pgainDir": dirn.astype(int), "pgainTier": tier,
        "pgainPass": (tier == 2) & (p_ratio < alpha / n_tests),
    })


def pgain_all_pairs(L: pd.DataFrame, y: pd.Series, columns, n_cov: int, alpha: float,
                    suggestive: float, keep_all: bool = False, chunk: int = 2000):
    """Step 8b: the same test for every metabolite pair. Returns (kept pairs, plot sample, B)."""
    cols = list(columns)
    p = len(cols)
    n_tests = p * (p - 1) // 2
    X = L[cols].to_numpy()
    n = len(X)
    yv = y.loc[L.index].to_numpy(dtype=float)
    Xc, yc = X - X.mean(0), yv - yv.mean()
    var_x = Xc.var(0, ddof=1)
    cov_y = Xc.T @ yc / (n - 1)
    r_single = cov_y / np.sqrt(var_x * yc.var(ddof=1))
    p_single = corr_to_p(r_single, n - 2 - n_cov)
    C = Xc.T @ Xc / (n - 1)
    strong_cut = n_tests / (2 * alpha)
    kept, sample = [], []
    rng = np.random.default_rng(0)
    for start in range(0, p, chunk):
        rows = np.arange(start, min(start + chunk, p))
        I, J = np.meshgrid(rows, np.arange(p), indexing="ij")
        m = I < J
        i, j = I[m], J[m]
        if len(i) == 0:
            continue
        r, pr, dirn = _ratio_stats(Xc, yc, var_x, cov_y, C[i, j], i, j, n, n_cov)
        pr = np.maximum(pr, 1e-300)
        pg = np.minimum(p_single[i], p_single[j]) / pr
        tier = np.where(pg >= strong_cut, 2, np.where(pg >= suggestive, 1, 0))
        df = pd.DataFrame({"i": i, "j": j, "p_ratio": pr, "pgain": pg, "pgainDir": dirn.astype(int),
                           "pgainTier": tier})
        keep = df if keep_all else df[df["pgainTier"] >= 1]
        kept.append(keep)
        bg = df[df["pgainTier"] == 0]
        if len(bg):
            sample.append(bg.iloc[rng.choice(len(bg), size=min(len(bg), 20000), replace=False)])
    allk = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["i", "j"])
    smp = pd.concat(sample + [allk[allk["pgainTier"] >= 1]], ignore_index=True) if sample else allk
    cols_a = np.asarray(cols)
    for d in (allk, smp):
        d["p_a"] = p_single[d["i"].to_numpy(dtype=int)]
        d["p_b"] = p_single[d["j"].to_numpy(dtype=int)]
        d["metabolite_a"] = cols_a[d["i"].to_numpy(dtype=int)]
        d["metabolite_b"] = cols_a[d["j"].to_numpy(dtype=int)]
        d["pgainPass"] = (d["pgainTier"] == 2) & (d["p_ratio"] < alpha / n_tests)
    out_cols = ["metabolite_a", "metabolite_b", "p_a", "p_b", "p_ratio", "pgain", "pgainDir", "pgainTier", "pgainPass"]
    allk = allk[out_cols].sort_values("pgain", ascending=False).reset_index(drop=True)
    return allk, smp[out_cols], n_tests
