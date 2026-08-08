"""Confusion matrices for the broad and focused passes, marking ontology-correct errors.

The UMAP figure shows *that* the broad pass disagrees with the truth and the focused pass
agrees. It cannot show *how* they disagree, which is the interesting part: the paper's whole
case for reporting ontology-aware concordance is that a large share of "wrong" calls are the
right lineage at the wrong depth, and exact match scores those identically to calling a
hepatocyte a T cell.

So each cell is shaded by the fraction of a truth type receiving that label, and any
off-diagonal cell whose label is an ancestor or descendant of the truth in the Cell Ontology
is outlined. Outlined mass is error under exact match and credit under concordance; the
difference between the two metrics is visible rather than asserted.

    .venv/bin/python benchmark/explore/plot_confusion.py --out docs/figures/fig_confusion.png
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.patches import Rectangle

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics
from benchmark.explore.plot_workflow_umap import short_names, top_labels


def related(a, b, anc):
    """True when two ontology ids are the same node or one descends from the other."""
    if not a or not b or a in ("unknown", "nan", "") or b in ("unknown", "nan", ""):
        return False
    return a == b or a in anc.get(b, ()) or b in anc.get(a, ())


def confusion(truth, pred, rows, cols):
    m = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        sel = truth == r
        n = int(sel.sum())
        if not n:
            continue
        for j, c in enumerate(cols):
            m[i, j] = float((pred[sel] == c).sum()) / n
    return m


def panel(ax, mat, rows, cols, row_cl, col_cl, anc, title, subtitle):
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    rn, cn = short_names(rows, 30), short_names(cols, 26)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([cn[c] for c in cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rn[r] for r in rows], fontsize=7)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.text(0.5, -0.52, subtitle, transform=ax.transAxes, ha="center", fontsize=8.5,
            color="#444")

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = mat[i, j]
            exact = (r == c)
            onto = (not exact) and related(row_cl.get(r), col_cl.get(c), anc)
            if onto and v >= 0.03:      # outlining 1% mass just adds boxes round white cells
                # outline, not fill: this mass is an error to exact match and credit to
                # concordance, and the reader should see which cells carry the difference
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="#D55E00", lw=1.6, zorder=3))
            if v >= 0.08:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.2,
                        color="white" if v > 0.55 else "#222", zorder=4)
    ax.set_xlabel("predicted", fontsize=8.5)
    ax.set_ylabel("true label", fontsize=8.5)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="/Volumes/IanSSD/hlica/liver_query_xstudy.h5ad")
    ap.add_argument("--broad", default="broad_human_v1")
    ap.add_argument("--focused-train", default="/Volumes/IanSSD/hlica/liver_ref_xstudy.h5ad")
    ap.add_argument("--focused-label", default="cell_type")
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="docs/figures/fig_confusion.png")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    q = sc.read_h5ad(a.query)
    truth = q.obs["cell_type"].astype(str).to_numpy()
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
    row_cl = dict(zip(truth, true_cl))
    print(f"query {q.shape} | {len(set(truth))} truth types", flush=True)

    calls, col_cl = {}, {}
    for tag in ("broad", "focused"):
        if tag == "broad":
            model = aj.bundled_reference(a.broad)
        else:
            ref = sc.read_h5ad(a.focused_train)
            raw = ref.copy()
            sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
            sc.pp.highly_variable_genes(raw, n_top_genes=min(4000, raw.n_vars))
            ref = ref[:, raw.var["highly_variable"].values].copy()
            model = aj.train_reference(ref, train_label_name=a.focused_label,
                                       print_cost=False)
        try:
            frame = model.predict_frame(q)[0]
        except TypeError:
            frame = model.predict_frame(q, use_raw="auto")[0]
        calls[tag] = frame["celltype"].to_numpy().astype(str)
        cmap_cl = dict(getattr(model, "class_to_cl", None) or {})
        col_cl[tag] = cmap_cl or dict(zip(truth, true_cl))
        exact = float((calls[tag] == truth).mean())
        print(f"{tag:<8} exact={exact:.3f} classes={len(model.classes)}", flush=True)

    rows = top_labels(truth, a.top)
    # wspace: the right panel's y-tick labels are long cell-type names and ran into the left
    # panel's plot area at the default spacing.
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6),
                             gridspec_kw={"wspace": 0.62})
    for ax, tag, title in ((axes[0], "broad", "broad pass — census reference"),
                           (axes[1], "focused", "focused pass — liver reference")):
        pred = calls[tag]
        cols = top_labels(pred[np.isin(truth, rows)], a.top)
        mat = confusion(truth, pred, rows, cols)
        onto = sum(mat[i, j] for i, r in enumerate(rows) for j, c in enumerate(cols)
                   if r != c and related(row_cl.get(r), col_cl[tag].get(c), anc))
        diag = sum(mat[i, j] for i, r in enumerate(rows) for j, c in enumerate(cols) if r == c)
        im = panel(ax, mat, rows, cols, row_cl, col_cl[tag], anc, title,
                   f"exact on shown cells {diag / len(rows):.2f} · "
                   f"+{onto / len(rows):.2f} ontology-correct (outlined)")
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.03)
    cb.set_label("fraction of the true type given this label", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Where each pass disagrees with the truth — outlined cells are the right "
                 "lineage at the wrong depth", fontsize=12, y=1.0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("CONFUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
