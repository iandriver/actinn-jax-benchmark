"""What the Cell-Ontology hierarchy is, what it produces, and whether the structure matters.

The coarse->fine reference build has exactly one step that wants a GPU: it Ward-clusters
scPRINT embeddings of per-type expression centroids to discover the coarse groups. Swapping
that one step for the Cell Ontology -- describe each type by the CL terms it descends from,
cluster *those* -- makes the build free, deterministic and organism-independent, which is
what lets a pan-mouse reference exist at all (Pan-human Azimuth is human-only, so there is
no mouse teacher to distill).

Three panels, because the claim has three parts a table cannot carry:
  A  the substitution -- both routes side by side, with the one box that differs marked.
  B  what it produces -- the 21 mouse coarse groups, each labeled with the CL lineage its
     members actually share. The point is that groups are lineages, not arbitrary splits,
     and that the two generic groups (types sitting on `cell` / `native cell`) are visible
     rather than hidden.
  C  whether the structure is doing the work -- ontology vs no hierarchy vs a random
     grouping with the same group sizes, on the held-out human lung atlas.

    .venv-protocloud/bin/python benchmark/explore/plot_ontology_hierarchy.py
"""

import argparse
import collections
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from benchmark import metrics

GPU = "#D55E00"      # the step that needs one
CPU = "#0072B2"      # the step that replaces it
INK = "#222222"
MUTE = "#7A7A7A"


def cl_names(obo):
    """CL id -> primary name, straight out of the obo stanzas."""
    names, cur = {}, None
    with open(obo) as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]":
                cur = None
            elif line.startswith("id: "):
                cur = line[4:]
            elif line.startswith("name: ") and cur:
                names[cur] = line[6:]
                cur = None
    return names


def label_group(members_cl, anc, names, corpus_freq):
    """Name a group by the most specific CL term most of its members descend from.

    Uninformative terms are dropped before ranking, not after: every cell descends from
    `cell` and `nucleate cell`, so those score 100% coverage in every group and would win
    the sort in all 21 of them. Among what is left, coverage decides (the label has to
    describe the group) and specificity breaks ties -- rarer across the corpus means
    further down the ontology, so `lymphocyte` beats `hematopoietic cell` when both cover
    every member.

    Specificity is depth in the ontology (how many terms this one descends from), not
    rarity in the corpus: the two disagree because the ancestor map follows `is_a` only,
    which left `neuron` scoring as no more specific than `electrically signaling cell`.
    """
    sets = [set(anc.get(c, ())) | {c} for c in members_cl if c]
    if not sets:
        return "unlabeled", 0.0
    votes = collections.Counter(t for s in sets for t in s)
    best, best_key = None, None
    for term, hits in votes.items():
        name = names.get(term, term)
        if name in GENERIC:
            continue
        cov = hits / len(sets)
        if cov < 0.60:                       # not a description of this group
            continue
        key = (round(cov, 2), len(anc.get(term, ())), -corpus_freq.get(term, 0))
        if best_key is None or key > best_key:
            best, best_key = term, key
    if best is None:                         # no shared lineage below the generic terms
        return "no shared lineage", 0.0
    return names.get(best, best), votes[best] / len(sets)


def panel_a(ax):
    """The two routes to a coarse hierarchy, differing in exactly one box."""
    # ylim crops to the drawn content; a taller box just prints whitespace above panel B
    ax.set_xlim(0, 10); ax.set_ylim(0.30, 3.20); ax.axis("off")

    def box(x, y, w, text, edge, face="white", bold=False, h=0.72):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045",
                                    linewidth=1.7 if bold else 1.1,
                                    edgecolor=edge, facecolor=face, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6,
                color=INK, zorder=3, linespacing=1.35,
                fontweight="bold" if bold else "normal")

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=9, color=MUTE, lw=1.0, zorder=1))

    rows = [
        (2.25, "human reference", GPU, "scPRINT embedding\nof the centroid",
         "needs a GPU", "29 groups"),
        (0.75, "any organism", CPU, "Cell Ontology terms\nthe type descends from",
         "free, deterministic", "21 groups (mouse)"),
    ]
    for y, tag, col, mid, note, out in rows:
        ax.text(0.02, y + 0.36, tag, ha="left", va="center", fontsize=8.2,
                color=col, fontweight="bold")
        box(1.55, y, 1.85, "per cell type in\nthe census sample", MUTE)
        arrow(3.45, 3.95, y + 0.36)
        box(3.98, y, 2.25, mid, col, bold=True)
        ax.text(5.10, y - 0.20, note, ha="center", va="center", fontsize=6.8, color=col)
        arrow(6.28, 6.78, y + 0.36)
        box(6.81, y, 1.30, "Ward\nlinkage", MUTE)
        arrow(8.16, 8.66, y + 0.36)
        box(8.69, y, 1.28, out, MUTE)

    # stop short of the "needs a GPU" note at y=2.05 rather than running through it
    ax.annotate("", xy=(5.10, 1.96), xytext=(5.10, 1.53),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(5.30, 1.75, "the only step that changes", fontsize=7.2, color=INK, va="center")
    ax.set_title("A  One substitution removes the GPU from the build",
                 fontsize=10, loc="left", pad=6)


def panel_b(ax, groups):
    """The 21 mouse coarse groups: size, and the lineage the members share."""
    groups = groups.sort_values("n", ascending=True)
    generic = groups["lineage"].isin(GENERIC)
    colors = [MUTE if g else CPU for g in generic]
    ax.barh(range(len(groups)), groups["n"], color=colors, height=0.74)
    ax.set_yticks(range(len(groups)))
    # show coverage only where the lineage does not cover the whole group, so the reader
    # can tell "every member is a lymphocyte" from "most members are epithelial"
    ax.set_yticklabels([r.lineage if r.coverage >= 0.995 or r.lineage in GENERIC
                        else f"{r.lineage}  ({r.coverage:.0%})"
                        for r in groups.itertuples()], fontsize=7)
    for i, r in enumerate(groups.itertuples()):
        ax.text(r.n + 0.7, i, f"{r.n}", va="center", fontsize=6.6, color=INK)
    ax.set_xlabel("cell types in the group", fontsize=8.5)
    ax.set_xlim(0, groups["n"].max() * 1.16)
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("B  What it produces: the 21 mouse coarse groups,\n"
                 "     each named by the Cell Ontology term its members share",
                 fontsize=10, loc="left", pad=6)
    ax.text(0.99, 0.02, "gray: no lineage below the generic terms — the census's vague labels\n"
                        "(\"cell\", \"blood cell\") collect here",
            transform=ax.transAxes, ha="right", fontsize=6.6, color=MUTE, linespacing=1.4)


def panel_c(ax, abl):
    """Ontology vs flat vs random, same group sizes."""
    order = ["ontology", "flat", "random"]
    pretty = {"ontology": "Cell Ontology\nlineage", "flat": "no hierarchy",
              "random": "random grouping,\nsame group sizes"}
    vals = [float(abl.loc[abl.hierarchy == h, "ontology"].iloc[0]) for h in order]
    cols = [CPU, MUTE, MUTE]
    ax.bar(range(3), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK,
                fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(3))
    ax.set_xticklabels([pretty[h] for h in order], fontsize=7.4)
    ax.set_ylabel("ontology concordance", fontsize=8.5)
    ax.set_ylim(0.50, max(vals) + 0.035)
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # the control is the whole point: random == flat, so it is not the splitting that helps
    # legs on the outer bar edges, not the centers, so they miss the value labels
    bracket = max(vals[1], vals[2]) + 0.016
    ax.plot([0.72, 0.72, 2.28, 2.28], [vals[1] + 0.004, bracket, bracket, vals[2] + 0.004],
            color=MUTE, lw=0.9)
    ax.text(1.5, bracket + 0.003, "splitting alone buys nothing", ha="center",
            va="bottom", fontsize=6.9, color=MUTE)
    ax.set_title("C  It is which types share a group,\n     not that the problem was split",
                 fontsize=10, loc="left", pad=6)


# true of essentially every cell, so they describe no group in particular
GENERIC = {"cell", "native cell", "animal cell", "eukaryotic cell", "nucleate cell",
           "motile cell", "somatic cell", "no shared lineage"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.expanduser(
        "~/Downloads/actinn-jax/actinn_jax/references/broad_mouse_v1/manifest.json"))
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--ablation", default="docs/results_hierarchy_ablation.csv")
    ap.add_argument("--out", default="docs/figures/fig_ontology_hierarchy.png")
    a = ap.parse_args()

    man = json.load(open(a.ref))
    t2g, c2cl = man["type_to_group"], man["class_to_cl"]
    anc, names = metrics.load_cl_ancestors(a.obo), cl_names(a.obo)

    corpus_freq = collections.Counter()
    for cl in c2cl.values():
        if cl:
            corpus_freq.update(set(anc.get(cl, ())) | {cl})

    rows = []
    for grp in sorted(set(t2g.values())):
        members = [t for t, g in t2g.items() if g == grp]
        lineage, cov = label_group([c2cl.get(t, "") for t in members], anc, names,
                                   corpus_freq)
        rows.append({"group": grp, "n": len(members), "lineage": lineage,
                     "coverage": cov})
    groups = pd.DataFrame(rows)
    print(f"{len(t2g)} types -> {len(groups)} groups")
    print(groups.sort_values("n", ascending=False).to_string(index=False))

    abl = pd.read_csv(a.ablation)

    fig = plt.figure(figsize=(11.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 2.25], width_ratios=[1.9, 1.0],
                          hspace=0.24, wspace=0.30)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, 0]), groups)
    panel_c(fig.add_subplot(gs[1, 1]), abl)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("ONTOLOGY_HIERARCHY_DONE", flush=True)


if __name__ == "__main__":
    main()
