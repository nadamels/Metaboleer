"""Small shared statistics helpers."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def corr_to_p(r, df):
    """Two-sided p-value for correlation(s) r with df degrees of freedom."""
    r = np.clip(np.asarray(r, dtype=float), -1 + 1e-15, 1 - 1e-15)
    t = r * np.sqrt(df / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), df)
