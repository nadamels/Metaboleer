from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class Config:
    """All tunable parameters; defaults follow the Metaboleer flowchart."""
    # inputs / step 0
    data: str = ""
    metadata: str = ""
    outdir: str = "metaboleer_out"
    annotation: Optional[str] = None        # optional feature table (feature, super_pathway, label)
    xenobiotics_file: Optional[str] = None  # optional text file: one xenobiotic feature per line
    id_col: Optional[str] = None            # sample-id column (default: first column of each file)
    mets_col: str = "MetS"
    age_col: str = "age"
    sex_col: str = "sex"
    fasting_col: Optional[str] = "fasting_hours"
    min_fasting_hours: Optional[float] = None  # drop samples fasted for less than this
    adjust_fasting: bool = False                # also regress out fasting hours in step 2
    # step 1
    max_missing: float = 0.80
    # imputation
    imputer_trees: int = 100
    imputer_iter: int = 5
    # step 2
    protect_trait: bool = False  # keep MetS in the covariate model so it is not regressed out
    # step 3
    vip_cutoff: float = 1.0
    fdr_cutoff: float = 0.05
    opls_ortho: int = 1
    # step 4
    network_type: str = "signed"
    soft_power: Optional[int] = None
    r2_cut: float = 0.85
    min_module_size: int = 10
    deep_split: int = 2
    merge_cut_height: float = 0.25
    # step 6
    module_alpha: float = 0.05
    modules: Optional[List[str]] = None  # force modules instead of auto selection
    # step 7
    ggm_alpha: float = 0.01
    # step 8
    pgain_alpha: float = 0.05
    pgain_suggestive: float = 10.0
    pgain_scope: str = "all"      # 8b scope: "all" features or "ggm" nodes only
    keep_all_pairs: bool = False  # write every pair (not only p-gain >= suggestive)
    # misc
    seed: int = 1
    n_jobs: int = -1
    title: str = "WGCNA + GGM network"

    def to_dict(self):
        return asdict(self)
