"""Step 9 (data): D3 network page, Cytoscape tables / .cyjs, MetaboAnalyst lists."""
from __future__ import annotations

import json
import math
import os
from importlib import resources

import numpy as np
import pandas as pd

from .colors import hex_for


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def build_graph(features, modules, diff: pd.DataFrame, trait_cor: pd.Series, edges: pd.DataFrame,
                module_trait: pd.DataFrame, selected_modules, labels: dict) -> dict:
    nodes = []
    for f in features:
        m = modules[f]
        nodes.append({
            "id": f, "label": labels.get(f, f), "module": m, "color": hex_for(m),
            "vip": _clean(diff.at[f, "VIP"]), "vipTier": diff.at[f, "vipTier"],
            "diff": bool(diff.at[f, "differential"]), "traitCor": _clean(trait_cor[f]),
        })
    links = []
    for r in edges.itertuples(index=False):
        links.append({"source": r.source, "target": r.target, "cor": _clean(r.cor), "p": _clean(r.p),
                      "pgain": _clean(r.pgain), "pgainDir": int(r.pgainDir),
                      "pgainTier": int(r.pgainTier), "pgainPass": bool(r.pgainPass)})
    modstats = {m: {"p": _clean(module_trait.at[m, "adj_p"]), "r": _clean(module_trait.at[m, "r"])}
                for m in selected_modules}
    return {"nodes": nodes, "links": links, "modstats": modstats}


def write_html(graph: dict, path: str, title: str) -> None:
    tpl = resources.files("metaboleer").joinpath("templates/network.html").read_text(encoding="utf-8")
    blob = json.dumps(graph, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("__GRAPH_JSON__", blob).replace("__TITLE__", title)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def write_cytoscape(graph: dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    nodes = pd.DataFrame(graph["nodes"]).rename(columns={"id": "name"})
    edges = pd.DataFrame(graph["links"], columns=["source", "target", "cor", "p", "pgain", "pgainDir",
                                                  "pgainTier", "pgainPass"])
    edges["interaction"] = "partial_correlation"
    edges["pgainLevel"] = edges["pgainTier"].map({0: "none", 1: "suggestive", 2: "strong"})
    edges["abs_cor"] = edges["cor"].abs()
    nodes.to_csv(os.path.join(outdir, "nodes.csv"), index=False)
    edges.to_csv(os.path.join(outdir, "edges.csv"), index=False)
    cyjs = {
        "format_version": "1.0", "generated_by": "metaboleer", "target_cytoscapejs_version": "~2.1",
        "data": {"name": "Metaboleer WGCNA-GGM"},
        "elements": {
            "nodes": [{"data": {"id": n["id"], "name": n["label"], **{k: v for k, v in n.items() if k not in ("id", "label")}}}
                      for n in graph["nodes"]],
            "edges": [{"data": {"id": f"e{i}", **e}} for i, e in enumerate(graph["links"])],
        },
    }
    with open(os.path.join(outdir, "network.cyjs"), "w") as fh:
        json.dump(cyjs, fh, allow_nan=False)


def write_metaboanalyst_lists(modules: pd.Series, labels: dict, outdir: str) -> None:
    """One plain-text list per module (paste into MetaboAnalyst's enrichment / pathway tools)."""
    os.makedirs(outdir, exist_ok=True)
    for m in sorted(set(modules)):
        names = [labels.get(f, f) for f in modules.index[modules == m]]
        with open(os.path.join(outdir, f"module_{m}_metaboanalyst.txt"), "w") as fh:
            fh.write("\n".join(names) + "\n")
