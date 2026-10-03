# Metaboleer

Metabolomics network pipeline: **WGCNA co-abundance modules → Gaussian graphical model → p-gain**,
exported as an interactive D3 network (modules as colours, partial correlations as edges,
p-gain ratios as gold-haloed edges), Cytoscape tables and figures.

Input: a samples × metabolites abundance table and a metadata table. Output: `09_export/network.html`
(same viewer as the reference `ggm_wgcna_d3.html`: cluster-pull slider, trait-size slider, VIP outlines,
node/edge data tables).

## Install & run

```bash
pip install -e .
metaboleer example example_data          # synthetic demo data
metaboleer run --data example_data/data.csv --metadata example_data/metadata.csv \
               --annotation example_data/annotation.csv --outdir out
```

Open `out/09_export/network.html` in a browser (needs internet for the d3 / DataTables CDNs, as the reference page does).

### Input files

* `--data` CSV/TSV, rows = samples, columns = metabolites, first column = sample id, empty = missing.
* `--metadata` CSV/TSV with sample id plus columns `MetS` (0/1, yes/no, case/control), `age`, `sex`,
  `fasting_hours` (override names with `--mets-col --age-col --sex-col --fasting-col`).
* `--annotation` (optional): `feature`, `super_pathway` (rows labelled `Xenobiotics` are zero-filled instead of
  imputed) and optionally `label` for display names. Alternatively `--xenobiotics-file` (one feature per line).

## Pipeline

| Step | What | Output folder |
|---|---|---|
| 0 | Load metadata: MetS, age, sex, fasting hours (`--min-fasting-hours` filters, `--adjust-fasting` adds it as covariate) | `00_metadata` |
| 1 | Drop features > 80 % missing; **MissForest**-style RF imputation (non-xenobiotics), **zero-fill** (xenobiotics) | `01_missingness` |
| 2 | log2 (zeros → half the smallest detected value); regress out age + sex | `02_preprocessed.csv` |
| 3 | Differential candidates: OPLS-DA VIP ≥ 1 **and** limma (empirical-Bayes) FDR < 0.05 | `03_differential` |
| 4 | WGCNA: soft threshold (scale-free R² ≥ 0.85), TOM, dynamic cut, eigengene merge, module–MetS association (BH) | `04_wgcna` |
| 5 | Per-module network PDFs and MetaboAnalyst lists | `05_modules` |
| 6 | Module selection: top MetS modules, total features ≤ samples | `07_ggm/selected_features.csv` |
| 7 | GGM: partial correlations, Bonferroni α = 0.01 | `07_ggm` |
| 8 | p-gain of each GGM edge's log-ratio vs. its single metabolites; strong cutoff B/2α (= 10·B at α = 0.05), suggestive ≥ 10 | `08_pgain/ggm_edges_pgain.csv` |
| 8b | Same test for every metabolite pair | `08_pgain/pgain_all_pairs.csv` |
| 9 | D3 network, Cytoscape (`nodes.csv`, `edges.csv`, `network.cyjs`), p-gain plots | `09_export` |

`p-gain = min(p_A, p_B) / p_ratio`, where `p_ratio` is the association of log2(A) − log2(B) with MetS.
Edge colour in the viewer is the sign of the partial correlation; gold halo = p-gain; `pgainDir` = direction of the ratio in MetS.

## notes if you use!!!

* R packages are replaced by native implementations: limma's moderated t (`fitFDist` + `squeezeVar`),
  single-response OPLS-DA (one orthogonal component by default, `--vip-cutoff`), WGCNA soft-threshold/TOM,
  and a simplified dynamic tree cut (branch size + height gap, `--deep-split 0–4`). Results will be close to,
  not bit-identical with, the R tools.
* Step 6 keeps `features ≤ samples − 2` so partial correlations are estimable. If the top module alone is
  larger, its most MetS-correlated metabolites are kept. Force modules with `--modules blue,brown`.
* Association tests use residuals after covariate adjustment and subtract the covariates from the residual df.
* `vipTier` is reported only for differential candidates (`none` otherwise), as in the reference viewer.

* Imputation is the slow step (random forests; ~4 min for 300 samples × 220 features at the defaults).
  For quick looks use `--imputer-trees 20 --imputer-iter 2`.
* The viewer loads d3 / jQuery / DataTables from CDNs, exactly like the reference page.

Run the tests with `pytest`.

`Credits`: Fouad Azar, Nada Elsharkawy, Imane Mourjane
