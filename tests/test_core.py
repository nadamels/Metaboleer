import numpy as np
import pandas as pd
import pytest

from metaboleer import differential, ggm, pgain, preprocess, wgcna
from metaboleer.stats import bh_adjust


def test_bh_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    q = bh_adjust(p)
    assert np.all(q >= p) and q.max() <= 1


def test_log2_handles_zero_fill():
    X = pd.DataFrame({"a": [0.0, 4.0, 8.0]})
    L = preprocess.log2_transform(X)
    assert np.isfinite(L.to_numpy()).all() and L["a"].iloc[0] < L["a"].iloc[1]


def test_regress_out_removes_covariate():
    rng = np.random.default_rng(0)
    age = rng.normal(size=200)
    L = pd.DataFrame({"f": 3 * age + rng.normal(size=200)})
    out = preprocess.regress_out(L, pd.DataFrame({"age": age}))
    assert abs(np.corrcoef(out["f"], age)[0, 1]) < 1e-8
    assert out["f"].mean() == pytest.approx(L["f"].mean())


def test_opls_vip_mean_square_is_one_and_finds_signal():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.5, 200)
    X = rng.normal(size=(200, 30))
    X[:, 0] += 1.5 * y
    r = differential.opls_da(X, y)
    assert np.mean(r["vip"] ** 2) == pytest.approx(1.0)
    assert np.argmax(r["vip"]) == 0


def test_limma_detects_shift_and_controls_null():
    rng = np.random.default_rng(2)
    g = rng.binomial(1, 0.5, 120)
    Y = rng.normal(size=(120, 200))
    Y[:, :10] += 1.0 * g[:, None]
    r = differential.limma_two_group(Y, g)
    assert (r["adj.P.Val"][:10] < 0.05).all()
    assert (r["adj.P.Val"][10:] < 0.05).sum() <= 4


def test_partial_correlation_chain():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    b = a + 0.5 * rng.normal(size=2000)
    c = b + 0.5 * rng.normal(size=2000)
    L = pd.DataFrame({"a": a, "b": b, "c": c})
    e = ggm.ggm_edges(L, ["a", "b", "c"], alpha=0.01)
    pairs = {frozenset((r.source, r.target)) for r in e.itertuples()}
    assert frozenset(("a", "b")) in pairs and frozenset(("b", "c")) in pairs
    assert frozenset(("a", "c")) not in pairs      # conditional independence given b


def test_pgain_ratio_beats_single_metabolites():
    rng = np.random.default_rng(4)
    n = 400
    y = pd.Series(rng.binomial(1, 0.5, n))
    s = rng.normal(scale=2, size=n)
    L = pd.DataFrame({"A": s + 0.5 * y + 0.1 * rng.normal(size=n),
                      "B": s - 0.5 * y + 0.1 * rng.normal(size=n),
                      "C": rng.normal(size=n)})
    d = pgain.pgain_pairs(L, y, np.array([[0, 1], [0, 2]]), list(L.columns), 0, 0.05, 10, 2)
    assert d.loc[0, "pgain"] > 1e3 and d.loc[0, "pgainDir"] == 1 and d.loc[0, "pgainTier"] == 2
    assert d.loc[1, "pgainTier"] == 0
    allp, _, B = pgain.pgain_all_pairs(L, y, list(L.columns), 0, 0.05, 10)
    assert B == 3
    assert allp.iloc[0][["metabolite_a", "metabolite_b"]].tolist() == ["A", "B"]
    assert allp.iloc[0]["pgain"] == pytest.approx(d.loc[0, "pgain"], rel=1e-6)


def test_wgcna_recovers_planted_modules():
    rng = np.random.default_rng(5)
    n = 200
    cols, data = [], []
    for m in range(3):
        f = rng.normal(size=n)
        for k in range(20):
            data.append(f + 0.6 * rng.normal(size=n)); cols.append(f"m{m}_{k}")
    for k in range(20):
        data.append(rng.normal(size=n)); cols.append(f"noise_{k}")
    L = pd.DataFrame(np.column_stack(data), columns=cols)
    y = pd.Series(rng.binomial(1, 0.5, n))
    res = wgcna.run_wgcna(L, y, min_module_size=8, log=lambda *_: None)
    for m in range(3):
        labs = res.modules[[c for c in cols if c.startswith(f"m{m}_")]]
        assert labs.value_counts().iloc[0] >= 17
    assert len({res.modules[f"m{m}_0"] for m in range(3)}) == 3
