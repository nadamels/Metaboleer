"""Steps 1-2: missingness filter, MissForest / zero-fill imputation, log2, covariate regression."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def missingness_filter(X: pd.DataFrame, max_missing: float) -> Tuple[pd.DataFrame, pd.Series]:
    frac = X.isna().mean(axis=0)
    return X.loc[:, frac <= max_missing], frac


def impute(X: pd.DataFrame, xenobiotics: Iterable[str], trees: int = 100, max_iter: int = 5,
           seed: int = 1, n_jobs: int = -1) -> pd.DataFrame:
    """Non-xenobiotics: MissForest-style iterative random-forest imputation.
    Xenobiotics: missing = not detected -> filled with 0."""
    X = X.copy()
    xeno = [c for c in X.columns if c in set(xenobiotics)]
    other = [c for c in X.columns if c not in set(xeno)]
    if xeno:
        X[xeno] = X[xeno].fillna(0.0)
    if other and X[other].isna().any().any():
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        rf = RandomForestRegressor(n_estimators=trees, max_features="sqrt", min_samples_leaf=1,
                                   n_jobs=n_jobs, random_state=seed)
        imp = IterativeImputer(estimator=rf, max_iter=max_iter, initial_strategy="median",
                               imputation_order="ascending", tol=1e-3, random_state=seed,
                               skip_complete=True)
        X[other] = imp.fit_transform(X[other].to_numpy())
    return X


def log2_transform(X: pd.DataFrame) -> pd.DataFrame:
    """log2; non-positive values (zero-filled xenobiotics) become half the feature's
    smallest positive value before the log so they stay finite but below the detected range."""
    A = X.to_numpy(dtype=float).copy()
    A[A <= 0] = np.nan
    floor = np.nanmin(A, axis=0) / 2.0
    floor = np.where(np.isfinite(floor), floor, 1.0)
    bad = np.isnan(A)
    A[bad] = np.broadcast_to(floor, A.shape)[bad]
    return pd.DataFrame(np.log2(A), index=X.index, columns=X.columns)


def regress_out(L: pd.DataFrame, covariates: pd.DataFrame, y: pd.Series = None,
                protect_trait: bool = False) -> pd.DataFrame:
    """Remove covariate effects from every feature; feature means are kept (residual + mean)."""
    C = covariates.loc[L.index].to_numpy(dtype=float)
    D = np.column_stack([np.ones(len(L)), C])
    if protect_trait:
        D = np.column_stack([D, y.loc[L.index].to_numpy(dtype=float)])
    B, *_ = np.linalg.lstsq(D, L.to_numpy(), rcond=None)
    k = C.shape[1]
    fitted_cov = (C - C.mean(axis=0)) @ B[1:1 + k]   # covariate part only; feature means kept
    out = L.to_numpy() - fitted_cov
    return pd.DataFrame(out, index=L.index, columns=L.columns)
