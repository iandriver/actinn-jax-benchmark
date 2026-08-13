"""Annotating an atlas: cost against query size, with the reference held fixed.

The companion to the reference-side sweep. One reference (17,753 cells, 36 types, seven
studies), one query grown from 50k to the whole 524,699-cell HLiCA liver atlas.

The linear pipeline's largest point is an open marker, unjoined to its curve: it did not
finish. After 76 minutes it held 15.3 GB resident with ~14 GB paged out and was still
running, so it was stopped. Plotting a wall-clock there would be inventing a number, and
dropping the point would hide the result, so it appears on the memory panel alone. It is not
connected to the line because resident-at-kill is *below* the peak the run needed -- the
rest had been evicted to swap -- and a connecting segment would read as memory falling.

    .venv-protocloud/bin/python benchmark/explore/plot_query_scaling.py
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/results_query_scaling.csv")
    ap.add_argument("--out", default="docs/figures/fig_query_scaling.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))

    for m, st in STYLE.items():
        g = d[(d.method == m) & d.completed].sort_values("n_query")
        x = g.n_query / 1000
        axes[0].plot(x, g.predict_s, lw=1.9, markersize=5.4, markeredgecolor="white",
                     markeredgewidth=0.7, **st)
        axes[1].plot(x, g.cells_per_s / 1000, lw=1.9, markersize=5.4,
                     markeredgecolor="white", markeredgewidth=0.7, **st)
        axes[2].plot(x, g.peak_gb, lw=1.9, markersize=5.4,
                     markeredgecolor="white", markeredgewidth=0.7, **st)

    # the point that did not finish, marked where it stopped rather than dropped
    dnf = d[~d.completed]
    if len(dnf):
        r = dnf.iloc[0]
        axes[2].plot([r.n_query / 1000], [r.peak_gb], marker="s", markersize=9,
                     markerfacecolor="white", markeredgecolor=STYLE[r.method]["color"],
                     markeredgewidth=1.8, linestyle="none", zorder=5)
        axes[2].annotate("did not finish — stopped at 76 min\n"
                         "15.3 GB resident, ~14 GB swapped",
                         (r.n_query / 1000, r.peak_gb), textcoords="offset points",
                         xytext=(-14, 30), fontsize=7.2, ha="right",
                         color=STYLE[r.method]["color"],
                         arrowprops=dict(arrowstyle="->", lw=0.9,
                                         color=STYLE[r.method]["color"]))
        for ax in axes[:2]:
            ax.axvline(r.n_query / 1000, color=STYLE[r.method]["color"], lw=0.9,
                       ls=":", alpha=0.7)

    axes[0].set_ylabel("predict wall-clock (s)", fontsize=8.8)
    axes[0].set_yscale("log")
    axes[0].set_title("A  Time to annotate the query", fontsize=10, loc="left", pad=7)
    axes[1].set_ylabel("throughput (thousand cells/s)", fontsize=8.8)
    axes[1].set_ylim(0, None)
    axes[1].set_title("B  Throughput holds, or does not", fontsize=10, loc="left", pad=7)
    axes[2].set_ylabel("peak memory (GB)", fontsize=8.8)
    axes[2].set_ylim(0, None)
    axes[2].set_title("C  Peak memory", fontsize=10, loc="left", pad=7)
    for ax in axes:
        axis(ax)
    axes[0].legend(frameon=False, fontsize=8.2, loc="upper left")

    fig.suptitle("Annotating half a million cells: the reference is fixed, the query grows",
                 fontsize=12, y=1.015)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("QUERY_SCALING_PLOT_DONE", flush=True)


if __name__ == "__main__":
    main()
