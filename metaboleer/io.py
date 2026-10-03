"""Step 0: load the abundance matrix and sample metadata, align them, build covariates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config

_POS = {"1", "yes", "y", "true", "t", "mets", "case", "cases", "metabolic syndrome"}
_NEG = {"0", "no", "n", "false", "f", "control", "controls", "ctrl", "non-mets", "nomets", "healthy"}


def read_table(path: str, id_col: Optional[str] = None) -> pd.DataFrame:
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt", ".tab")) else ","
    df = pd.read_csv(path, sep=sep)
    col = id_col if id_col else df.columns[0]
    if col not in df.columns:
        raise ValueError(f"id column '{col}' not found in {path}; columns: {list(df.columns)[:10]}")
    df[col] = df[col].astype(str)
    if df[col].duplicated().any():
        raise ValueError(f"duplicate sample ids in {path}")
    return df.set_index(col)


def parse_binary(series: pd.Series) -> pd.Series:
    """Map MetS labels (0/1, yes/no, case/control, ...) to 0/1."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        k = str(v).strip().lower()
        if k.endswith(".0"):
            k = k[:-2]
        if k in _POS:
            out.append(1.0)
        elif k in _NEG:
            out.append(0.0)
        else:
            raise ValueError(f"cannot interpret MetS value {v!r}; use 0/1, yes/no or case/control")
    return pd.Series(out, index=series.index, name=series.name)


@dataclass
class Dataset:
    X: pd.DataFrame          # samples x metabolites (raw abundances, NaN = missing)
    y: pd.Series             # MetS, 0/1
    covariates: pd.DataFrame  # numeric design columns (age, sex dummies, [fasting])
    metadata: pd.DataFrame
    dropped_samples: List[str]


def covariate_frame(meta: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cov = pd.DataFrame(index=meta.index)
    cov["age"] = pd.to_numeric(meta[cfg.age_col], errors="raise")
    sex = meta[cfg.sex_col]
    if pd.api.types.is_numeric_dtype(sex):
        cov["sex"] = sex.astype(float)
    else:
        d = pd.get_dummies(sex.astype(str).str.strip().str.lower(), drop_first=True, dtype=float)
        for c in d.columns:
            cov[f"sex_{c}"] = d[c]
    if cfg.adjust_fasting and cfg.fasting_col:
        cov["fasting_hours"] = pd.to_numeric(meta[cfg.fasting_col], errors="raise")
    return cov


def load_dataset(cfg: Config) -> Dataset:
    X = read_table(cfg.data, cfg.id_col)
    meta = read_table(cfg.metadata, cfg.id_col)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any(axis=0)]

    need = [cfg.mets_col, cfg.age_col, cfg.sex_col]
    if (cfg.adjust_fasting or cfg.min_fasting_hours is not None):
        if not cfg.fasting_col:
            raise ValueError("fasting options require --fasting-col")
        need.append(cfg.fasting_col)
    missing = [c for c in need if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata is missing columns {missing}; has {list(meta.columns)}")

    common = [s for s in X.index if s in meta.index]
    if len(common) < 10:
        raise ValueError(f"only {len(common)} samples shared between data and metadata ids")
    dropped = [s for s in X.index if s not in meta.index]
    X, meta = X.loc[common], meta.loc[common].copy()

    meta["_y"] = parse_binary(meta[cfg.mets_col])
    keep = meta[[cfg.age_col, cfg.sex_col, "_y"]].notna().all(axis=1)
    if cfg.min_fasting_hours is not None:
        fh = pd.to_numeric(meta[cfg.fasting_col], errors="coerce")
        keep &= fh >= cfg.min_fasting_hours
    dropped += list(meta.index[~keep])
    X, meta = X.loc[keep], meta.loc[keep]

    y = meta["_y"].astype(int)
    if y.nunique() != 2:
        raise ValueError("MetS must have both cases and controls after filtering")
    cov = covariate_frame(meta, cfg)
    return Dataset(X=X, y=y.rename("MetS"), covariates=cov, metadata=meta.drop(columns="_y"),
                   dropped_samples=dropped)


def load_xenobiotics(cfg: Config, features) -> set:
    """Features to zero-fill instead of impute (step 1 right branch)."""
    xeno = set()
    if cfg.xenobiotics_file:
        with open(cfg.xenobiotics_file) as fh:
            xeno |= {ln.strip() for ln in fh if ln.strip()}
    if cfg.annotation:
        ann = read_table(cfg.annotation)
        cols = {c.lower(): c for c in ann.columns}
        pw = next((cols[k] for k in ("super_pathway", "super.pathway", "superpathway", "class") if k in cols), None)
        if pw is None:
            raise ValueError("annotation needs a 'super_pathway' column to flag xenobiotics")
        flagged = ann.index[ann[pw].astype(str).str.strip().str.lower().isin({"xenobiotics", "xenobiotic"})]
        xeno |= set(flagged)
    return xeno & set(features)


def load_labels(cfg: Config) -> dict:
    """Optional pretty labels (annotation column 'label' or 'biochemical')."""
    if not cfg.annotation:
        return {}
    ann = read_table(cfg.annotation)
    cols = {c.lower(): c for c in ann.columns}
    lab = next((cols[k] for k in ("label", "biochemical", "name", "compound") if k in cols), None)
    return {} if lab is None else ann[lab].dropna().astype(str).to_dict()
