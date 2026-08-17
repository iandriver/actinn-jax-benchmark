"""Per-method abstain behaviour: what a threshold does to each of the eight methods.

The companion to the cross-method trade-off figure. That one answers "which method trades
coverage for accuracy best"; this one answers "what does turning the knob actually do to
method X", which was previously a table of eight rows by three thresholds with three numbers
crammed into every cell -- seventy-two figures to read in order to notice that three of the
methods do not respond to the knob at all.

All three quantities are fractions, so one shared 0-1 axis carries them and the failure modes
become shapes rather than numbers: CellTypist's lines go flat after the first step, scTOP's
coverage falls off a cliff, ProtoCloud's novelty line stays on the floor until the very end.

    .venv-protocloud/bin/python benchmark/explore/plot_abstain_grid.py
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Table 8's order: the five that respond to a threshold first, then the three that do not.
ORDER = ["actinn-jax", "scarches", "scanvi", "linear-anova-pca", "knn",
         "celltypist", "protocloud", "sctop"]
NICE = {"actinn-jax": "actinn-jax", "scarches": "scArches", "scanvi": "scANVI",
        "linear-anova-pca": "linear pipeline", "knn": "kNN", "celltypist": "CellTypist",
        "protocloud": "ProtoCloud", "sctop": "scTOP"}
# Okabe-Ito, consistent with the rest of the figures.
SERIES = [("indist_acc_kept", "accuracy on kept cells", "#0072B2", "o"),
          ("coverage", "coverage (cells kept)", "#009E73", "s"),
          ("ood_flagged", "novel cells flagged", "#D55E00", "^")]
# (text, y in axes fraction) -- scTOP's coverage line runs along the floor, so its note sits
# in the gap the collapse opens up instead.
NOTE = {"celltypist": ("saturated probabilities:\nevery threshold is one point", 0.06),
        "sctop": ("not a calibrated probability:\ndiscards the query", 0.42),
        "protocloud": ("ambiguity flag barely\nmoves until 0.9", 0.06)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/results_rejection.csv")
    ap.add_argument("--out", default="docs/figures/fig_abstain_grid.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), sharex=True, sharey=True)
    for ax, meth in zip(axes.ravel(), ORDER):
        s = d[d.method == meth].sort_values("min_prob")
        for col, label, colour, marker in SERIES:
            ax.plot(s.min_prob, s[col], marker=marker, color=colour, lw=1.8,
                    markersize=4.6, markeredgecolor="white", markeredgewidth=0.6,
                    label=label)
        works = meth not in NOTE
        ax.set_title(NICE[meth], fontsize=10.5, pad=6,
                     fontweight="bold" if meth == "actinn-jax" else "normal",
                     color="#222222" if works else "#B03000")
        if meth in NOTE:
            txt, ypos = NOTE[meth]
            ax.text(0.5, ypos, txt, transform=ax.transAxes, ha="center",
                    fontsize=7.4, color="#B03000", linespacing=1.35)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0.0, 0.3, 0.5, 0.7, 0.9])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("confidence threshold", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.6, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("What a confidence threshold does to each method — "
                 "9 of 36 liver cell types withheld", fontsize=12.5, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("ABSTAIN_GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
