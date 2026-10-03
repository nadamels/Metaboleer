"""Steps 6-7: module selection and Gaussian graphical model (partial correlations)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .stats import corr_to_p
from .wgcna import WGCNAResult


def select_modules(res: WGCNAResult, n_samples: int, alpha: float = 0.05,
                   forced: Optional[List[str]] = None, log=print):
    """Pick top MetS-associated modules while keeping #features <= #samples (-2 so that
    partial correlations stay estimable). Returns (module list, feature list)."""
    limit = n_samples - 2
    mt = res.module_trait
    if forced:
        mods = list(forced)
        unknown = [m for m in mods if m not in set(res.modules)]
        if unknown:
            raise ValueError(f"unknown modules {unknown}; available: {sorted(set(res.modules))}")
    else:
        cand = mt[(mt["adj_p"] < alpha) & (mt.index != "grey")].sort_values("adj_p")
        if cand.empty:
            log(f"  warning: no module with adj. p < {alpha}; falling back to the top module")
            cand = mt[mt.index != "grey"].head(1)
        mods, used = [], 0
        for m, row in cand.iterrows():
            k = int(row["n_features"])
            if used + k <= limit:
                mods.append(m)
                used += k
            elif not mods:   # top module alone is too large: keep its most trait-associated features
                mods.append(m)
                used = limit
                break
    feats: List[str] = []
    for m in mods:
        feats += list(res.modules.index[res.modules == m])
    if len(feats) > limit:
        feats = list(res.trait_cor[feats].abs().sort_values(ascending=False).index[:limit])
        log(f"  top module has too many features; trimmed to the {limit} most MetS-correlated")
    return mods, feats


def partial_correlations(X: np.ndarray):
    """Partial correlation matrix from the inverse covariance, plus two-sided p-values."""
    n, p = X.shape
    S = np.cov(X.T)
    P = np.linalg.pinv(S) if np.linalg.cond(S) > 1e12 else np.linalg.inv(S)
    d = np.sqrt(np.abs(np.diag(P)))
    R = -P / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -1, 1)
    df = n - p            # n - 2 - (p - 2)
    Pv = corr_to_p(R, df)
    np.fill_diagonal(Pv, 0.0)
    return R, Pv


def ggm_edges(L: pd.DataFrame, features: List[str], alpha: float = 0.01) -> pd.DataFrame:
    X = L[features].to_numpy()
    R, Pv = partial_correlations(X)
    p = len(features)
    iu = np.triu_indices(p, 1)
    n_tests = len(iu[0])
    thr = alpha / n_tests                     # Bonferroni
    sel = Pv[iu] < thr
    edges = pd.DataFrame({
        "source": np.array(features)[iu[0][sel]], "target": np.array(features)[iu[1][sel]],
        "cor": R[iu][sel], "p": np.maximum(Pv[iu][sel], 1e-300),
    }).sort_values("p").reset_index(drop=True)
    edges.attrs.update(n_tests=n_tests, threshold=thr)
    return edges
