"""Annotating an atlas: cost against query size, with the reference held fixed.

The companion to the reference-side sweep. One reference (17,753 cells, 36 types, seven
studies), one query grown from 50k to the whole 524,699-cell HLiCA liver atlas, three
repeats at every point.

Wall-clock on a laptop is noisy enough that a single run misleads, so each point is the best
of three runs for time and the worst of three for memory; see `series` for why the two axes
are summarised in opposite directions.

An earlier version of this figure carried an open "did not finish" marker for the linear
pipeline at the full atlas. That was our adapter, not the method: its `predict` densified
20,000 genes for the whole query at once (42 GB, and as much again per transform copy) where
it could block. Blocked, it finishes the atlas in a little over two minutes. The lesson is
recorded here because the failed figure was more interesting than the fixed one is: peak RSS
on a machine that is already swapping undercounts what a run actually needs, so cost figures
have to be measured on a quiet machine or not at all.

    .venv-protocloud/bin/python benchmark/explore/plot_query_scaling.py
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

STYLE = {
    "actinn-jax":       dict(color="#0072B2", marker="o", label="actinn-jax"),
    "linear-anova-pca": dict(color="#009E73", marker="s", label="linear pipeline"),
}


def axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([50, 125, 250, 525])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("query cells (thousands, log scale)", fontsize=8.8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def series(g, col, scale=1.0, stat="min"):
    """Repeats collapsed to one point per size.

    Neither axis takes a mean, and for opposite reasons.

    Wall-clock takes the *fastest* run. A competing process can only add time, never remove
    it, so on a machine that is not exclusively ours the minimum is the closest estimate of
    what the method costs; a mean reports how loaded the laptop was. This matters here: one
    repeat hit a memory-pressure stall that turned a 55 s fit into 979 s, and averaging that
    in would have published contention as though it were the linear pipeline's scaling.

    Peak memory takes the *largest* run, for the mirror-image reason. ``ru_maxrss`` counts
    resident pages, so a run squeezed by the OS reports less than it needed. The maximum
    cannot be dragged down by a repeat that got evicted.
    """
    m = (g[col].max() if stat == "max" else g[col].min()) * scale
    return m.index / 1000, m.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/results_query_scaling.csv")
    ap.add_argument("--out", default="docs/figures/fig_query_scaling.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    n_rep = int(d.groupby(["method", "n_query"]).size().min())

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))
    panels = [(0, "predict_s", 1.0, "min"), (1, "cells_per_s", 1e-3, "max"),
              (2, "peak_gb", 1.0, "max")]

    for meth, st in STYLE.items():
        g = d[d.method == meth].groupby("n_query")
        for i, col, scale, stat in panels:
            x, y = series(g, col, scale, stat)
            axes[i].plot(x, y, lw=1.9, markersize=5.4, markeredgecolor="white",
                         markeredgewidth=0.7, **st)

    axes[0].set_ylabel("predict wall-clock (s)", fontsize=8.8)
    axes[0].set_yscale("log")
    axes[0].set_title("A  Time to annotate the query", fontsize=10, loc="left", pad=7)
    axes[1].set_ylabel("throughput (thousand cells/s)", fontsize=8.8)
    axes[1].set_ylim(0, None)
    axes[1].set_title("B  Throughput against query size", fontsize=10, loc="left", pad=7)
    axes[2].set_ylabel("peak memory (GB)", fontsize=8.8)
    axes[2].set_ylim(0, None)
    axes[2].set_title("C  Peak memory", fontsize=10, loc="left", pad=7)
    for ax in axes:
        axis(ax)
    axes[0].legend(frameon=False, fontsize=8.2, loc="upper left")

    fig.suptitle("Annotating half a million cells: the reference is fixed, the query grows",
                 fontsize=12, y=1.015)
    fig.text(0.5, -0.035,
             f"{n_rep} runs per point on a shared laptop: A and B show the fastest run, "
             "C the largest peak. Per-run values are in results_query_scaling.csv.",
             ha="center", fontsize=7.8, color="#444444")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} ({n_rep} repeats per point)")
    print("QUERY_SCALING_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
