"""Agreement between broad references as a confidence signal.

Three broad annotators over the same 3,396 withheld liver cells. The bars are ontology
concordance within each agreement tier; the dashed lines are each model's score over the whole
query. Every model roughly doubles on the cells where all three agree, which is the point: the
partition is not picking out cells one model happens to get right, it is picking out cells that
are unambiguous, and it can be computed on an unlabelled query.

    .venv-protocloud/bin/python benchmark/explore/plot_consensus.py
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIERS = ["3 of 3 references agree", "2 of 3 references agree", "1 of 3 references agree"]
LABEL = {"3 of 3 references agree": "all three agree",
         "2 of 3 references agree": "two agree",
         "1 of 3 references agree": "none agree"}
COLOUR = {"census (broad_human_v1)": "#999999",
          "distilled (panhuman_distill_v1)": "#0072B2",
          "Pan-human Azimuth": "#D55E00"}
SHORT = {"census (broad_human_v1)": "census reference",
         "distilled (panhuman_distill_v1)": "distilled reference",
         "Pan-human Azimuth": "Pan-human Azimuth"}


def panel(ax, d, title, show_legend):
    models = [m for m in COLOUR if m in set(d.model)]
    width = 0.26
    for j, m in enumerate(models):
        vals, covs = [], []
        for t in TIERS:
            r = d[(d.model == m) & (d.subset == t)]
            vals.append(float(r.ontology.iloc[0]) if len(r) else np.nan)
            covs.append(float(r.coverage.iloc[0]) if len(r) else np.nan)
        x = np.arange(len(TIERS)) + (j - 1) * width
        ax.bar(x, vals, width * 0.92, color=COLOUR[m], label=SHORT[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.014, f"{v:.2f}", ha="center", fontsize=7.6, color="#333333")
        whole = d[(d.model == m) & (d.subset == "all cells")]
        if len(whole):
            ax.axhline(float(whole.ontology.iloc[0]), color=COLOUR[m], lw=1.0,
                       ls=(0, (5, 4)), alpha=0.85)

    covs = [float(d[(d.model == models[0]) & (d.subset == t)].coverage.iloc[0]) for t in TIERS]
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([f"{LABEL[t]}\n{c:.0%} of cells" for t, c in zip(TIERS, covs)],
                       fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=8.6, loc="lower left", ncol=1)
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liver", default="docs/results_consensus_broad.csv")
    ap.add_argument("--lung", default="docs/results_consensus_broad_lung.csv")
    ap.add_argument("--brain", default="docs/results_consensus_broad_brain.csv")
    ap.add_argument("--out", default="docs/figures/fig_consensus.png")
    a = ap.parse_args()

    have = [(p, t) for p, t in (
        (a.liver, "A  liver — 3,396 cells, 34 truth types"),
        (a.lung, "B  lung — 65,662 cells, 46 truth types"),
        (a.brain, "C  brain (MTG) — 156,285 cells, 18 truth types"))
        if os.path.exists(p)]
    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, ((path, title), ax) in enumerate(zip(have, axes)):
        panel(ax, pd.read_csv(path), title, show_legend=(i == 0))
    axes[0].set_ylabel("ontology concordance", fontsize=9.6)
    fig.suptitle("Where independent broad references agree, every one of them is right more "
                 "often", fontsize=12.5, y=1.02)
    fig.text(0.5, -0.045, "dashed lines: the same model over the whole query",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} ({len(have)} tissue panels)")
    print("CONSENSUS_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
