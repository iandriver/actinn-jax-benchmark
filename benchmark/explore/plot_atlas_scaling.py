"""Accuracy and memory against reference size, on two atlases.

The paper's second headline claim is that method rankings are not stable in reference size:
a prototype VAE that sits below the cluster on subsampled references becomes the most
accurate method of all once given a real atlas. That claim appeared in the abstract, the
Discussion and the Limitations, but had no figure and no table anywhere -- and §3.3, which is
where a reader is sent for it, reports only fit and predict *time*.

Three panels:
  A  lung, 3k -> 49k reference cells: ProtoCloud crosses from worst to best.
  B  the HLiCA liver atlas, 2.7k -> 47k: the same reversal, independently.
  C  peak memory over the same sweeps -- the band stays bounded at ~2x rather than widening,
     and scTOP crosses over from lightest to heaviest.

Fit time is deliberately not plotted. The 47k liver point was measured under contention
(actinn-jax 359 s against 59 s at 25k, scTOP 10.7 s against 1.5 s), so the cost axis there
would show scheduling, not scaling; the fit-cost ratios quoted in the text come from the
controlled matrix instead.

    .venv-protocloud/bin/python benchmark/explore/plot_atlas_scaling.py
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
    "protocloud":       dict(color="#D55E00", marker="o", label="ProtoCloud"),
    "linear-anova-pca": dict(color="#009E73", marker="s", label="linear pipeline"),
    "actinn-jax":       dict(color="#0072B2", marker="D", label="actinn-jax"),
    "sctop":            dict(color="#CC79A7", marker="^", label="scTOP"),
}
ORDER = ["protocloud", "linear-anova-pca", "actinn-jax", "sctop"]


def curve(ax, d, value, scale=1.0):
    for m in ORDER:
        g = d[d.method == m].sort_values("n_ref")
        if g.empty:
            continue
        st = STYLE[m]
        ax.plot(g["n_ref"] / 1000, g[value] * scale, lw=1.9, markersize=5.2,
                markeredgecolor="white", markeredgewidth=0.7, **st)
    ax.set_xscale("log")
    ax.set_xticks([3, 8, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # a log axis keeps its own minor ticks, which print "4x10^0" straight through the
    # sweep sizes we actually measured
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.22, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def mark_reversal(ax, d):
    """Call out the two endpoints of the claim rather than leaving them to be read off."""
    p = d[d.method == "protocloud"].sort_values("n_ref")
    lo, hi = p.iloc[0], p.iloc[-1]
    # both labels sit above their point: the low one is near the axis floor, and below it
    # the text runs into the tick labels
    ax.annotate(f"worst\n{lo.accuracy:.3f}", (lo.n_ref / 1000, lo.accuracy),
                textcoords="offset points", xytext=(9, 3), fontsize=7,
                color=STYLE["protocloud"]["color"], va="bottom")
    ax.annotate(f"best\n{hi.accuracy:.3f}", (hi.n_ref / 1000, hi.accuracy),
                textcoords="offset points", xytext=(-6, -20), fontsize=7,
                color=STYLE["protocloud"]["color"], ha="right", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lung", default="results/scaling_memory/results.csv")
    ap.add_argument("--liver", default="results/scaling_memory_hlica/results.csv")
    ap.add_argument("--out", default="docs/figures/fig_atlas_scaling.png")
    a = ap.parse_args()

    lung, liver = pd.read_csv(a.lung), pd.read_csv(a.liver)
    for tag, d in (("lung", lung), ("liver", liver)):
        p = d[d.method == "protocloud"].sort_values("n_ref")
        print(f"{tag}: ProtoCloud {p.accuracy.iloc[0]:.3f} at {p.n_ref.iloc[0]:,} "
              f"-> {p.accuracy.iloc[-1]:.3f} at {p.n_ref.iloc[-1]:,}")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15))

    curve(axes[0], lung, "accuracy")
    mark_reversal(axes[0], lung)
    axes[0].set_title("A  Lung atlas: the ranking inverts with scale", fontsize=10,
                      loc="left", pad=7)
    axes[0].set_ylabel("accuracy", fontsize=8.8)

    curve(axes[1], liver, "accuracy")
    mark_reversal(axes[1], liver)
    axes[1].set_title("B  HLiCA liver atlas: the same reversal", fontsize=10,
                      loc="left", pad=7)

    # both sweeps on one memory axis; the point is the band, not either curve alone
    curve(axes[2], lung, "peak_mem_mb", scale=1 / 1000)
    for m in ORDER:
        g = liver[liver.method == m].sort_values("n_ref")
        axes[2].plot(g["n_ref"] / 1000, g["peak_mem_mb"] / 1000, lw=1.3, ls=":",
                     color=STYLE[m]["color"], alpha=0.85)
    axes[2].set_title("C  Peak memory stays within a bounded band", fontsize=10,
                      loc="left", pad=7)
    axes[2].set_ylabel("peak memory (GB)", fontsize=8.8)
    axes[2].text(0.97, 0.05, "solid: lung    dotted: liver", transform=axes[2].transAxes,
                 ha="right", fontsize=6.9, color="#666")

    for ax in axes:
        ax.set_xlabel("reference cells (thousands, log scale)", fontsize=8.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Conclusions from subsampled references do not carry to atlas scale",
                 fontsize=12, y=1.015)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("ATLAS_SCALING_DONE", flush=True)


if __name__ == "__main__":
    main()
