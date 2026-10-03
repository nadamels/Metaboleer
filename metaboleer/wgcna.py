"""Step 4: WGCNA-style co-abundance modules (soft threshold, TOM, dynamic cut, eigengene merge)
and module-trait association against MetS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .colors import GREY, module_color_name
from .stats import bh_adjust, corr_to_p

POWERS = list(range(1, 11)) + list(range(12, 21, 2))


@dataclass
class WGCNAResult:
    power: int
    fit: pd.DataFrame            # soft-threshold table
    modules: pd.Series           # feature -> colour name ("grey" = unassigned)
    eigengenes: pd.DataFrame     # samples x modules
    module_trait: pd.DataFrame   # r, p, adj_p, n_features per module
    kme: pd.Series               # correlation with own module eigengene
    trait_cor: pd.Series         # feature-MetS correlation (gene significance)
    tom: np.ndarray
    corr: np.ndarray


def _adjacency(C: np.ndarray, power: float, network_type: str) -> np.ndarray:
    if network_type == "signed":
        A = ((1 + C) / 2) ** power
    elif network_type == "signed hybrid":
        A = np.clip(C, 0, None) ** power
    elif network_type == "unsigned":
        A = np.abs(C) ** power
    else:
        raise ValueError(f"unknown network_type {network_type!r}")
    np.fill_diagonal(A, 0.0)
    return A


def scale_free_fit(k: np.ndarray, n_breaks: int = 10) -> tuple:
    """Signed R^2 of log10 p(k) ~ log10 k (WGCNA scaleFreeFitIndex)."""
    if k.max() <= k.min():
        return 0.0, 0.0
    counts, edges = np.histogram(k, bins=n_breaks)
    which = np.clip(np.digitize(k, edges[1:-1]), 0, n_breaks - 1)
    dk = np.array([k[which == b].mean() for b in range(n_breaks) if counts[b] > 0])
    pk = counts[counts > 0] / len(k)
    if len(dk) < 3:
        return 0.0, 0.0
    lx, ly = np.log10(dk), np.log10(pk)
    slope, intercept = np.polyfit(lx, ly, 1)
    ss_res = np.sum((ly - (slope * lx + intercept)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-np.sign(slope) * r2), float(slope)


def pick_soft_threshold(C: np.ndarray, network_type: str, r2_cut: float):
    rows = []
    for b in POWERS:
        A = _adjacency(C, b, network_type)
        k = A.sum(0)
        r2, slope = scale_free_fit(k)
        rows.append({"power": b, "SFT.R.sq": r2, "slope": slope, "mean.k": k.mean(),
                     "median.k": float(np.median(k)), "max.k": k.max()})
    fit = pd.DataFrame(rows)
    ok = fit[(fit["SFT.R.sq"] >= r2_cut) & (fit["slope"] < 0)]
    if len(ok):
        return int(ok["power"].iloc[0]), fit, True
    return int(fit.loc[fit["SFT.R.sq"].idxmax(), "power"]), fit, False


def tom_matrix(A: np.ndarray) -> np.ndarray:
    k = A.sum(0)
    L = A @ A
    denom = np.minimum.outer(k, k) + 1 - A
    T = (L + A) / denom
    np.fill_diagonal(T, 1.0)
    return T


def dynamic_cut(Z: np.ndarray, n: int, min_size: int, deep_split: int = 2,
                cut_quantile: float = 0.99) -> np.ndarray:
    """Simplified dynamic tree cut. A branch is a module if it has >= min_size leaves and
    joins its sibling well above its own merge height (gap relative to the tree's height
    range, tightened by deep_split). Small branches attach to a neighbouring module that
    is not split further; leftovers are unassigned (label 0)."""
    frac = {0: 0.30, 1: 0.20, 2: 0.12, 3: 0.08, 4: 0.05}[int(deep_split)]
    h = Z[:, 2]
    gap = frac * (h.max() - h.min())
    cap = h.min() + cut_quantile * (h.max() - h.min())

    def children(i): return int(Z[i - n, 0]), int(Z[i - n, 1])
    def size(i): return 1 if i < n else int(Z[i - n, 3])
    def height(i): return 0.0 if i < n else float(Z[i - n, 2])

    def leaves(i):
        out, st = [], [i]
        while st:
            j = st.pop()
            if j < n:
                out.append(j)
            else:
                st.extend(children(j))
        return out

    clusters = []
    stack = [(2 * n - 2, [])]
    while stack:
        node, carry = stack.pop()
        if node < n:
            continue
        a, b = children(node)
        H = height(node)
        if H > cap:
            stack += [(a, []), (b, [])]
            continue
        va = size(a) >= min_size and H - height(a) >= gap
        vb = size(b) >= min_size and H - height(b) >= gap
        if va and vb:
            stack += [(a, []), (b, [])]
        elif size(node) < min_size:
            continue
        elif not va and not vb:
            clusters.append(leaves(node) + carry)
        else:
            big, small = (a, b) if va else (b, a)
            if size(small) >= min_size:
                clusters.append(leaves(node) + carry)
            else:
                stack.append((big, carry + leaves(small)))
    labels = np.zeros(n, dtype=int)
    for i, c in enumerate(sorted(clusters, key=len, reverse=True), start=1):
        labels[c] = i
    return labels


def eigengene(Xm: np.ndarray) -> np.ndarray:
    """First principal component of the standardised module matrix, oriented to the mean profile."""
    Z = (Xm - Xm.mean(0)) / np.where(Xm.std(0) > 0, Xm.std(0), 1)
    u, s, _ = np.linalg.svd(Z, full_matrices=False)
    me = u[:, 0] * s[0]
    me = me / (me.std() or 1.0)
    if np.corrcoef(me, Z.mean(1))[0, 1] < 0:
        me = -me
    return me


def _eigengenes(X: np.ndarray, labels: np.ndarray) -> dict:
    return {m: eigengene(X[:, labels == m]) for m in sorted(set(labels) - {0})}


def merge_close_modules(X: np.ndarray, labels: np.ndarray, cut_height: float) -> np.ndarray:
    labels = labels.copy()
    while True:
        me = _eigengenes(X, labels)
        ids = list(me)
        if len(ids) < 2:
            return labels
        M = np.column_stack([me[i] for i in ids])
        D = 1 - np.corrcoef(M.T)
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if D[i, j] >= cut_height:
            return labels
        labels[labels == ids[j]] = ids[i]


def run_wgcna(L: pd.DataFrame, y: pd.Series, network_type: str = "signed",
              soft_power: Optional[int] = None, r2_cut: float = 0.85, min_module_size: int = 10,
              deep_split: int = 2, merge_cut_height: float = 0.25, n_cov: int = 0,
              log=print) -> WGCNAResult:
    X = L.to_numpy(dtype=float)
    n, p = X.shape
    C = np.corrcoef(X.T)
    C = np.nan_to_num(C)
    if soft_power is None:
        power, fit, reached = pick_soft_threshold(C, network_type, r2_cut)
        if not reached:
            log(f"  warning: no power reached scale-free R^2 >= {r2_cut}; using best fit, power={power}")
    else:
        power, reached = int(soft_power), True
        _, fit, _ = pick_soft_threshold(C, network_type, r2_cut)
    A = _adjacency(C, power, network_type)
    T = tom_matrix(A)
    D = 1 - T
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), method="average")
    raw = dynamic_cut(Z, p, min_module_size, deep_split)
    merged = merge_close_modules(X, raw, merge_cut_height)

    # rename by module size (largest = turquoise, ...); unassigned = grey
    sizes = pd.Series(merged[merged > 0]).value_counts()
    rename = {0: GREY[0]}
    for rank, mid in enumerate(sizes.index, start=1):
        rename[int(mid)] = module_color_name(rank)
    modules = pd.Series([rename[int(m)] for m in merged], index=L.columns, name="module")

    me = pd.DataFrame({rename[m]: v for m, v in _eigengenes(X, merged).items()}, index=L.index)
    yv = y.loc[L.index].to_numpy(dtype=float)
    rows = {}
    for m in me.columns:
        r = float(np.corrcoef(me[m], yv)[0, 1])
        rows[m] = {"r": r, "p": float(corr_to_p(r, n - 2 - n_cov)), "n_features": int((modules == m).sum())}
    mt = pd.DataFrame(rows).T.astype({"n_features": int})
    if len(mt):
        mt["adj_p"] = bh_adjust(mt["p"].to_numpy())
        mt = mt.sort_values("adj_p")

    kme = pd.Series(np.nan, index=L.columns, name="kME")
    for m in me.columns:
        cols = modules.index[modules == m]
        kme[cols] = [np.corrcoef(L[c], me[m])[0, 1] for c in cols]
    trait_cor = pd.Series([np.corrcoef(X[:, j], yv)[0, 1] for j in range(p)], index=L.columns, name="traitCor")
    return WGCNAResult(power, fit, modules, me, mt, kme, trait_cor, T, C)
