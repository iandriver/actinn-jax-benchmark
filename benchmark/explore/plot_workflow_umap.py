"""Figure: what the broad pass and the focused pass actually do to one dataset.

"Tier 1 / Tier 2" was jargon the paper never defined. This makes the two passes concrete on
a single query -- the withheld HLiCA liver study -- by showing the same UMAP three times:

  broad pass    the shipped census-wide reference (~800 human types). Identifies roughly
                what is present and which focused reference to load; abstains (grey) where
                confidence is below min_prob.
  focused pass  a 48-type HLiCA liver reference re-annotating the same cells.
  truth         the study's own labels.

The palettes deliberately differ between the broad panel and the other two: the broad
reference answers in the census vocabulary, the focused one in the study's. That mismatch is
the reason the hand-off exists, so hiding it would misrepresent the figure.

    .venv/bin/python benchmark/explore/plot_workflow_umap.py [--out docs/figures/fig_workflow_umap.png]
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
from matplotlib.lines import Line2D

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics

GREY = "#d9d9d9"


def concordance(true_cl, pred_cl, anc):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not t or t in ("unknown", "nan", ""):
            continue
        n += 1
        ok += bool(p and p not in ("unknown", "nan", "") and
                   (p == t or p in anc.get(t, ()) or t in anc.get(p, ())))
    return ok / n if n else float("nan")


def panel(ax, xy, labels, title, palette, subtitle=None, max_legend=8):
    order = [l for l, _ in sorted(
        ((l, int((labels == l).sum())) for l in set(labels)),
        key=lambda kv: -kv[1])]
    for lab in order:
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=3, linewidths=0,
                   c=palette.get(lab, GREY), label=lab if lab in order[:max_legend] else None)
    ax.set_title(title, fontsize=11, pad=6)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes, ha="center", fontsize=8.5,
                color="#444")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      color=palette.get(l, GREY),
                      label=(l[:26] + "…") if len(l) > 27 else l)
               for l in order[:max_legend]]
    extra = len(order) - max_legend
    if extra > 0:
        handles.append(Line2D([], [], marker="", linestyle="",
                              label=f"+{extra} more label{'s' if extra > 1 else ''}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.10),
              frameon=False, fontsize=7.5, ncol=2, handletextpad=0.4,
              columnspacing=0.9, labelspacing=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="/Volumes/IanSSD/hlica/liver_query_xstudy.h5ad")
    ap.add_argument("--broad", default="broad_human_v1")
    # The SHIPPED liver_hlica_v2 is trained on all HLiCA studies including the one withheld
    # here, so scoring it on this query leaks: it reports 0.936 against the 0.86 the paper
    # quotes from a leakage-free model. Train the focused reference on the 6-study reference
    # instead, which is what section 3.5 measures.
    ap.add_argument("--focused-train", default="/Volumes/IanSSD/hlica/liver_ref_xstudy.h5ad",
                    help="h5ad to train the focused reference from (leakage-free)")
    ap.add_argument("--focused-label", default="cell_type")
    ap.add_argument("--focused", default=None,
                    help="instead load a bundled reference by name (may leak; see above)")
    # No abstain by default. With min_prob set, abstained cells go grey but still count as
    # misses in the concordance below, which prints a number far under the one the text
    # quotes for the same models (0.21 vs 0.58) -- a figure that appears to contradict the
    # paper. Abstain has its own section; this figure is about what the two passes label.
    ap.add_argument("--min-prob", type=float, default=None)
    ap.add_argument("--obo", default="/tmp/cl-basic.obo")
    ap.add_argument("--out", default="docs/figures/fig_workflow_umap.png")
    a = ap.parse_args()

    anc = metrics.load_cl_ancestors(a.obo)
    q = sc.read_h5ad(a.query)
    truth = q.obs["cell_type"].astype(str).to_numpy()
    true_cl = q.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
    print(f"query {q.shape} | {len(set(truth))} truth types", flush=True)

    def focused_model():
        if a.focused:
            return aj.bundled_reference(a.focused), a.focused
        ref = sc.read_h5ad(a.focused_train)
        raw = ref.copy()
        sc.pp.normalize_total(raw, target_sum=1e4)
        sc.pp.log1p(raw)
        sc.pp.highly_variable_genes(raw, n_top_genes=min(4000, raw.n_vars))
        ref = ref[:, raw.var["highly_variable"].values].copy()
        print(f"training focused reference on {ref.shape} "
              f"({ref.obs[a.focused_label].nunique()} types)", flush=True)
        m = aj.train_reference(ref, train_label_name=a.focused_label, print_cost=False)
        return m, f"trained on {os.path.basename(a.focused_train)}"

    calls, scores = {}, {}
    for tag in ("broad", "focused"):
        if tag == "broad":
            model, name = aj.bundled_reference(a.broad), a.broad
        else:
            model, name = focused_model()
        kw = {"min_prob": a.min_prob} if a.min_prob else {}
        try:
            frame = model.predict_frame(q, **kw)[0]
        except TypeError:                 # flat ReferenceModel: no min_prob, needs use_raw
            frame = model.predict_frame(q, use_raw="auto")[0]
        lab = frame["celltype"].to_numpy().astype(str)
        # Hierarchical references carry a class->CL map; a flat one trained here does not,
        # but it predicts into the query's own vocabulary, so the query supplies the mapping.
        cl_map = dict(getattr(model, "class_to_cl", None) or {})
        if not cl_map:
            cl_map = dict(zip(truth, true_cl))
        pred_cl = np.array([cl_map.get(p, "unknown") for p in lab])
        calls[tag] = lab
        scores[tag] = (concordance(true_cl, pred_cl, anc),
                       float((lab == "unknown").mean()), len(model.classes))
        print(f"{tag:<8} {name:<20} ontology={scores[tag][0]:.3f} "
              f"abstained={scores[tag][1]:.1%} classes={scores[tag][2]}", flush=True)

    # UMAP on the query alone: the figure is about how the same cells get labelled, so the
    # embedding must not depend on either reference.
    print("computing UMAP", flush=True)
    e = q.copy()
    sc.pp.normalize_total(e, target_sum=1e4)
    sc.pp.log1p(e)
    sc.pp.highly_variable_genes(e, n_top_genes=2000)
    e = e[:, e.var.highly_variable].copy()
    sc.pp.scale(e, max_value=10)
    sc.tl.pca(e, n_comps=50)
    sc.pp.neighbors(e, n_neighbors=15)
    sc.tl.umap(e)
    xy = e.obsm["X_umap"]

    # Focused and truth share the study's vocabulary, so they share colours and can be read
    # against each other. The broad panel gets its own palette by necessity.
    cmap = plt.get_cmap("tab20")
    shared = sorted(set(truth) | set(calls["focused"]) - {"unknown"})
    pal_shared = {l: cmap(i % 20) for i, l in enumerate(shared)}
    pal_shared["unknown"] = GREY
    broad_labels = sorted(set(calls["broad"]) - {"unknown"})
    cmap2 = plt.get_cmap("tab20b")
    pal_broad = {l: cmap2(i % 20) for i, l in enumerate(broad_labels)}
    pal_broad["unknown"] = GREY

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
    ob, ab, nb = scores["broad"]
    of, af, nf = scores["focused"]
    ab_txt = f" · {ab:.0%} abstained (grey)" if a.min_prob else ""
    af_txt = f" · {af:.0%} abstained" if a.min_prob else ""
    panel(axes[0], xy, calls["broad"],
          f"broad pass — census reference ({nb} types)",
          pal_broad,
          f"ontology concordance {ob:.2f}{ab_txt}")
    panel(axes[1], xy, calls["focused"],
          f"focused pass — liver reference ({nf} types)",
          pal_shared,
          f"ontology concordance {of:.2f}{af_txt}")
    panel(axes[2], xy, truth, "study's own labels", pal_shared,
          f"{len(set(truth))} types")
    fig.suptitle("One query, two passes: the broad pass routes, the focused pass resolves",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print("WORKFLOW_UMAP_DONE", flush=True)


if __name__ == "__main__":
    main()
