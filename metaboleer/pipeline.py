"""Orchestration of the Metaboleer flowchart (steps 0-9, plus 3, 5 and 8b side branches)."""
from __future__ import annotations

import json
import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from . import differential, export, ggm, io, pgain, plots, preprocess, wgcna
from .config import Config


def run_pipeline(cfg: Config, log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    out = cfg.outdir
    for d in ("00_metadata", "01_missingness", "03_differential", "04_wgcna", "05_modules",
              "07_ggm", "08_pgain", "09_export/cytoscape", "09_export/figures"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    P = lambda *a: os.path.join(out, *a)  # noqa: E731
    np.random.seed(cfg.seed)

    # ---- 0 metadata
    log("[0] Metadata: MetS status, age, sex, fasting hours")
    ds = io.load_dataset(cfg)
    n_cov = ds.covariates.shape[1]
    log(f"    {ds.X.shape[0]} samples ({int(ds.y.sum())} MetS / {int((1 - ds.y).sum())} control), "
        f"{ds.X.shape[1]} features; {len(ds.dropped_samples)} samples dropped")
    ds.metadata.assign(MetS_01=ds.y).to_csv(P("00_metadata", "samples_used.csv"))

    # ---- 1 missingness filter + imputation
    log(f"[1] Missingness filter: drop features > {cfg.max_missing:.0%} missing")
    X, frac = preprocess.missingness_filter(ds.X, cfg.max_missing)
    frac.rename("fraction_missing").to_csv(P("01_missingness", "feature_missingness.csv"))
    log(f"    kept {X.shape[1]} / {ds.X.shape[1]} features")
    xeno = io.load_xenobiotics(cfg, X.columns)
    log(f"    imputation: MissForest for {X.shape[1] - len(xeno)} non-xenobiotics, zero-fill for {len(xeno)} xenobiotics")
    Xi = preprocess.impute(X, xeno, cfg.imputer_trees, cfg.imputer_iter, cfg.seed, cfg.n_jobs)
    Xi.to_csv(P("01_missingness", "imputed_raw.csv"))

    # ---- 2 preprocessing
    log("[2] Preprocessing: log2; regress out " + ", ".join(ds.covariates.columns))
    L = preprocess.log2_transform(Xi)
    L = preprocess.regress_out(L, ds.covariates, ds.y, cfg.protect_trait)
    L.to_csv(P("02_preprocessed.csv"))
    y = ds.y
    labels = io.load_labels(cfg)

    # ---- 3 differential candidates (side branch)
    log(f"[3] Differential candidates: OPLS-DA VIP >= {cfg.vip_cutoff:g}, limma FDR < {cfg.fdr_cutoff:g}")
    diff = differential.differential_candidates(L, y, n_cov, cfg.vip_cutoff, cfg.fdr_cutoff, cfg.opls_ortho)
    diff.to_csv(P("03_differential", "differential_candidates.csv"), index_label="feature")
    log(f"    {int(diff['differential'].sum())} differential candidates (OPLS-DA R2Y = {diff.attrs['opls_R2Y']:.2f})")

    # ---- 4 WGCNA
    log(f"[4] WGCNA: {cfg.network_type} co-abundance modules vs. MetS")
    res = wgcna.run_wgcna(L, y, cfg.network_type, cfg.soft_power, cfg.r2_cut, cfg.min_module_size,
                          cfg.deep_split, cfg.merge_cut_height, n_cov, log)
    n_mod = int((res.module_trait.index != "grey").sum()) if len(res.module_trait) else 0
    log(f"    soft power {res.power}; {n_mod} modules; {int((res.modules == 'grey').sum())} unassigned (grey)")
    res.fit.to_csv(P("04_wgcna", "soft_threshold.csv"), index=False)
    plots.soft_threshold_plot(res.fit, res.power, P("04_wgcna", "soft_threshold.pdf"))
    pd.DataFrame({"module": res.modules, "kME": res.kme, "traitCor": res.trait_cor}).to_csv(
        P("04_wgcna", "module_assignment.csv"), index_label="feature")
    res.module_trait.to_csv(P("04_wgcna", "module_trait.csv"), index_label="module")
    res.eigengenes.to_csv(P("04_wgcna", "module_eigengenes.csv"))

    # ---- 5 per-module outputs (side branch)
    log("[5] Per-module outputs: network PDFs, MetaboAnalyst lists")
    plots.module_network_pdfs(res.modules, res.tom, res.kme, diff, res.module_trait, labels, P("05_modules"))
    export.write_metaboanalyst_lists(res.modules, labels, P("05_modules"))

    # ---- 6 module selection
    log("[6] Module selection: top MetS modules with features <= samples")
    mods, feats = ggm.select_modules(res, len(L), cfg.module_alpha, cfg.modules, log)
    log(f"    selected {mods} -> {len(feats)} features (samples = {len(L)})")
    pd.Series(feats, name="feature").to_csv(P("07_ggm", "selected_features.csv"), index=False)

    # ---- 7 GGM
    log(f"[7] GGM: partial correlations, Bonferroni alpha = {cfg.ggm_alpha:g}")
    edges = ggm.ggm_edges(L, feats, cfg.ggm_alpha)
    log(f"    {len(edges)} edges among {len(feats)} metabolites (p < {edges.attrs['threshold']:.2e})")

    # ---- 8 p-gain on GGM edges
    B = max(len(edges), 1)
    log(f"[8] p-gain: log-ratio vs. single metabolites, strong cutoff B/2alpha = {B / (2 * cfg.pgain_alpha):g} (B = {B} edges)")
    cols = list(L.columns)
    ix = {c: i for i, c in enumerate(cols)}
    if len(edges):
        pairs = np.array([[ix[a], ix[b]] for a, b in zip(edges["source"], edges["target"])])
        pg = pgain.pgain_pairs(L, y, pairs, cols, n_cov, cfg.pgain_alpha, cfg.pgain_suggestive, B)
        for c in ("p_a", "p_b", "p_ratio", "pgain", "pgainDir", "pgainTier", "pgainPass"):
            edges[c] = pg[c].to_numpy()
    else:
        for c, v in (("p_a", 1.0), ("p_b", 1.0), ("p_ratio", 1.0), ("pgain", 0.0), ("pgainDir", 0),
                     ("pgainTier", 0), ("pgainPass", False)):
            edges[c] = v
    edges.to_csv(P("08_pgain", "ggm_edges_pgain.csv"), index=False)
    log(f"    p-gain edges: {int((edges['pgainTier'] == 2).sum())} strong, {int((edges['pgainTier'] == 1).sum())} suggestive")

    # ---- 8b p-gain on every pair
    scope = cols if cfg.pgain_scope == "all" else feats
    log(f"[8b] p-gain on all pairs ({len(scope)} metabolites -> {len(scope) * (len(scope) - 1) // 2:,} pairs)")
    allp, sample, n_pairs = pgain.pgain_all_pairs(L, y, scope, n_cov, cfg.pgain_alpha, cfg.pgain_suggestive,
                                                  cfg.keep_all_pairs)
    allp.to_csv(P("08_pgain", "pgain_all_pairs.csv"), index=False)
    log(f"    {int((allp['pgainTier'] == 2).sum())} strong, {int((allp['pgainTier'] == 1).sum())} suggestive pairs")

    # ---- 9 export & figures
    log("[9] Export & figures: D3 network, Cytoscape, p-gain plots")
    graph = export.build_graph(feats, res.modules, diff, res.trait_cor, edges, res.module_trait, mods, labels)
    html = P("09_export", "network.html")
    export.write_html(graph, html, cfg.title)
    export.write_cytoscape(graph, P("09_export", "cytoscape"))
    plots.pgain_plots(sample, n_pairs, cfg.pgain_alpha, edges, P("09_export", "figures"), B)
    summary = {
        "samples": int(len(L)), "features_input": int(ds.X.shape[1]), "features_after_filter": int(X.shape[1]),
        "xenobiotics": len(xeno), "differential": int(diff["differential"].sum()),
        "soft_power": res.power, "modules": n_mod, "selected_modules": mods, "ggm_nodes": len(feats),
        "ggm_edges": int(len(edges)), "pgain_strong_edges": int((edges["pgainTier"] == 2).sum()),
        "pgain_suggestive_edges": int((edges["pgainTier"] == 1).sum()),
        "pgain_pairs_strong": int((allp["pgainTier"] == 2).sum()), "seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
    }
    with open(P("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"Done in {summary['seconds']}s -> {html}")
    return {"summary": summary, "graph": graph, "wgcna": res, "diff": diff, "edges": edges,
            "pgain_all": allp, "html": html}
