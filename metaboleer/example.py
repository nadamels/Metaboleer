"""Synthetic MetS-like dataset with planted structure (for demo and tests)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def make_example(outdir: str, n: int = 300, seed: int = 7, n_features: int = 220):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    mets = rng.binomial(1, 0.4, n)
    age = rng.normal(55, 10, n).round(0)
    sex = rng.choice(["F", "M"], n)
    fasting = rng.choice([8, 10, 12, 14], n, p=[.1, .2, .5, .2]).astype(float)
    sexn = (sex == "M").astype(float)

    # latent factors: (loading on MetS, size)
    modules = [("lipid", 0.9, 30), ("amino", 0.7, 25), ("bile", -0.6, 20), ("inert", 0.0, 25), ("tca", 0.5, 15)]
    cols, data = [], []
    for name, beta, size in modules:
        f = rng.normal(size=n) + beta * mets
        e = rng.normal(size=n)
        for k in range(size):
            # AR(1) chain of residuals -> direct (partial-correlation) edges between neighbours
            e = 0.8 * e + 0.6 * rng.normal(size=n)
            data.append(rng.uniform(0.5, 0.8) * f + e)
            cols.append(f"{name}_{k + 1}")
        if name == "lipid":
            # planted p-gain pairs: a pair-specific factor cancels in the ratio while MetS
            # pushes the two members in opposite directions
            for j in range(4):
                s_j = rng.normal(scale=0.7, size=n)
                data.append(1.0 * f + s_j + 0.4 * mets + rng.normal(scale=0.15, size=n)); cols.append(f"pg_A{j + 1}")
                data.append(1.0 * f + s_j - 0.4 * mets + rng.normal(scale=0.15, size=n)); cols.append(f"pg_B{j + 1}")
    n_noise = n_features - len(cols) - 20
    for k in range(max(n_noise, 0)):
        data.append(rng.normal(size=n)); cols.append(f"noise_{k + 1}")
    M = np.column_stack(data)
    # age / sex effects on everything (to be regressed out)
    M = M + np.outer((age - 55) / 10, rng.normal(0.3, 0.2, M.shape[1])) + np.outer(sexn, rng.normal(0.4, 0.3, M.shape[1]))
    A = 2.0 ** (M * 0.8 + 14)                                   # raw abundance scale

    # xenobiotics: mostly undetected (NaN), intensity driven by a diet/drug factor
    xcols = [f"xeno_{k + 1}" for k in range(20)]
    X = np.where(rng.random((n, 20)) < 0.55, np.nan, 2.0 ** rng.normal(12, 1.5, (n, 20)))
    # a nearly empty feature to exercise the missingness filter
    sparse = np.where(rng.random(n) < 0.9, np.nan, 2.0 ** rng.normal(10, 1, n))

    D = np.column_stack([A, X, sparse])
    cols = cols + xcols + ["sparse_feature"]
    miss = rng.random(D.shape) < 0.04                           # random missingness
    D[miss] = np.nan
    ids = [f"S{i + 1:04d}" for i in range(n)]
    pd.DataFrame(D, index=pd.Index(ids, name="sample_id"), columns=cols).to_csv(os.path.join(outdir, "data.csv"))
    pd.DataFrame({"sample_id": ids, "MetS": mets, "age": age, "sex": sex, "fasting_hours": fasting}) \
        .to_csv(os.path.join(outdir, "metadata.csv"), index=False)
    pd.DataFrame({"feature": cols, "super_pathway": ["Xenobiotics" if c.startswith("xeno_") else "Lipid" for c in cols]}) \
        .to_csv(os.path.join(outdir, "annotation.csv"), index=False)
    return outdir
