"""Figures: soft-threshold, per-module network PDFs, p-gain plots."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .colors import hex_for  # noqa: E402

GOLD, BLUE, RED = "#E6AB02", "#1f78b4", "#e31a1c"


def soft_threshold_plot(fit: pd.DataFrame, power: int, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(fit["power"], fit["SFT.R.sq"], "o-", color="#134e5e")
    ax[0].axhline(0.85, ls="--", color="grey")
    ax[0].axvline(power, ls=":", color=RED)
    ax[0].set(xlabel="soft power", ylabel="scale-free fit (signed R²)", title="Scale-free topology")
    ax[1].plot(fit["power"], fit["mean.k"], "o-", color="#134e5e")
    ax[1].axvline(power, ls=":", color=RED)
    ax[1].set(xlabel="soft power", ylabel="mean connectivity", title="Mean connectivity", yscale="log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def module_network_pdfs(modules: pd.Series, tom: np.ndarray, kme: pd.Series, diff: pd.DataFrame,
                        module_trait: pd.DataFrame, labels: dict, outdir: str, max_edges: int = 200):
    """Per-module co-abundance network (strongest TOM edges); node size = |kME|, outline = VIP tier."""
    os.makedirs(outdir, exist_ok=True)
    idx = {f: i for i, f in enumerate(modules.index)}
    tier_col = {"none": "#cfcfcf", "VIP1-2": "#FDD0A2", "VIP2-3": "#FD8D3C", "VIP>=3": "#B2182B"}
    for m in sorted(set(modules)):
        feats = list(modules.index[modules == m])
        if len(feats) < 2 or m == "grey":
            continue
        ii = [idx[f] for f in feats]
        T = tom[np.ix_(ii, ii)]
        iu = np.triu_indices(len(feats), 1)
        w = T[iu]
        top = np.argsort(w)[::-1][:min(max_edges, max(len(feats) * 3, 1))]
        G = nx.Graph()
        G.add_nodes_from(feats)
        for t in top:
            G.add_edge(feats[iu[0][t]], feats[iu[1][t]], weight=float(w[t]))
        pos = nx.spring_layout(G, weight="weight", seed=1, k=1.5 / np.sqrt(len(feats)))
        fig, ax = plt.subplots(figsize=(9, 9))
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=[0.5 + 3 * G[u][v]["weight"] for u, v in G.edges])
        sizes = [60 + 500 * abs(kme.get(f, 0.3) or 0.3) ** 2 for f in feats]
        ax.scatter([pos[f][0] for f in feats], [pos[f][1] for f in feats], s=sizes, c=hex_for(m),
                   edgecolors=[tier_col[diff.at[f, "vipTier"]] for f in feats],
                   linewidths=np.array([2.5 if diff.at[f, "differential"] else 0.8 for f in feats], dtype=float), zorder=3)
        for f in feats:
            if len(feats) <= 60 or diff.at[f, "differential"]:
                ax.annotate(labels.get(f, f), pos[f], fontsize=6, ha="left", va="bottom", zorder=4)
        mt = module_trait.loc[m] if m in module_trait.index else None
        sub = f"  (MetS r={mt['r']:.2f}, adj p={mt['adj_p']:.1e})" if mt is not None else ""
        ax.set_title(f"Module {m} — {len(feats)} metabolites{sub}")
        ax.axis("off")
        fig.savefig(os.path.join(outdir, f"module_{m}_network.pdf"), bbox_inches="tight")
        plt.close(fig)


def pgain_plots(sample: pd.DataFrame, n_tests: int, alpha: float, edges: pd.DataFrame,
                outdir: str, n_edges_tests: int):
    os.makedirs(outdir, exist_ok=True)
    cmap = {0: ("#b8bfcc", "other pairs"), 1: (GOLD, "p-gain ≥ 10 (suggestive)"), 2: ("#B2182B", f"p-gain ≥ B/2α (strong)")}
    # --- all pairs: ratio significance vs best single-metabolite significance
    fig, ax = plt.subplots(figsize=(6.4, 6))
    x = -np.log10(np.minimum(sample["p_a"], sample["p_b"]).clip(lower=1e-300))
    y = -np.log10(sample["p_ratio"].clip(lower=1e-300))
    for t in (0, 1, 2):
        m = sample["pgainTier"] == t
        ax.scatter(x[m], y[m], s=6 if t == 0 else 14, c=cmap[t][0], label=f"{cmap[t][1]} (n={int(m.sum())})",
                   alpha=0.5 if t == 0 else 0.9, linewidths=0, rasterized=True)
    lim = max(x.max(), y.max()) * 1.03
    ax.plot([0, lim], [0, lim], color="k", lw=0.6, ls="--")
    ax.set(xlabel="−log10 p, better single metabolite", ylabel="−log10 p, log-ratio",
           title=f"p-gain, every pair (B = {n_tests:,} tests)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"pgain_all_pairs.{ext}"), dpi=200)
    plt.close(fig)

    # --- GGM edges: p-gain bars
    e = edges[edges["pgainTier"] >= 1].sort_values("pgain", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(e) + 1.2)))
    if len(e):
        names = [f"{a} / {b}" for a, b in zip(e["source"], e["target"])]
        ax.barh(range(len(e)), np.log10(e["pgain"]), color=[BLUE if d > 0 else RED for d in e["pgainDir"]])
        ax.set_yticks(range(len(e)))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(np.log10(n_edges_tests / (2 * alpha)), color="k", ls="--", lw=0.8)
        ax.set_xlabel("log10 p-gain   (dashed: B/2α;  blue = ratio↑ in MetS, red = ratio↓)")
    else:
        ax.text(0.5, 0.5, "no GGM edge reached p-gain ≥ 10", ha="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("Top p-gain GGM edges")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"pgain_ggm_edges.{ext}"), dpi=200)
    plt.close(fig)
