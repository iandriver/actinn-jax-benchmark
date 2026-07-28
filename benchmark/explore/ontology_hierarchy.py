"""A coarse→fine hierarchy derived from the Cell Ontology instead of a foundation model.

The shipped human reference gets its coarse groups by Ward-clustering scPRINT embeddings
of per-type centroids -- the one step of the build that wants a GPU and an hour. But the
census already labels every cell with a Cell Ontology term, and CL encodes exactly the
relation the clustering is trying to recover: which cell types are kinds of the same thing.

So: describe each cell type by the set of CL terms it descends from, and cluster *those*.
Same `discover_hierarchy` call, same Ward linkage, ontology indicator vectors instead of
embeddings. That makes the hierarchy free, deterministic, and organism-independent -- the
last property is why the pan-mouse reference can be built at all, since the pretrained
annotator we distil for human (Pan-human Azimuth) is human-only and scPRINT's mouse support
is untested here.

The cost is that CL describes *nomenclature*, not expression: two types that look alike but
sit in different branches will not be grouped, and the ontology's granularity is uneven.
Whether that matters is measurable -- build both hierarchies on the same human corpus and
score them on the same query (see docs/PAN_MOUSE.md).

    from ontology_hierarchy import ontology_hierarchy
    grp = ontology_hierarchy(cl_ids, labels, n_groups=28, obo="/tmp/cl-basic.obo")
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("ACTINN_JAX_REPO",
                                  os.path.expanduser("~/Downloads/actinn-jax")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import actinn_jax as aj
from benchmark import metrics


def ontology_features(cl_ids, labels, obo="/tmp/cl-basic.obo", min_freq=2):
    """One row per cell type: a binary indicator over the CL terms it descends from.

    Terms shared by nearly every type (``cell``, ``native cell``) carry no grouping signal
    but dominate a Euclidean distance, so terms appearing in fewer than ``min_freq`` types
    or in *all* of them are dropped.
    """
    anc = metrics.load_cl_ancestors(obo)
    labels = np.asarray([str(x) for x in labels])
    cl_ids = np.asarray([str(x) for x in cl_ids])

    # one CL id per type (types are 1:1 with CL ids in census data)
    type_cl = {}
    for lab, cl in zip(labels, cl_ids):
        if cl and cl not in ("unknown", "nan", "") and lab not in type_cl:
            type_cl[lab] = cl
    types = sorted(type_cl)
    if not types:
        raise ValueError("no cell type carried a usable Cell Ontology id")

    term_sets = [set(anc.get(type_cl[t], ())) | {type_cl[t]} for t in types]
    counts = {}
    for s in term_sets:
        for term in s:
            counts[term] = counts.get(term, 0) + 1
    vocab = sorted(term for term, n in counts.items()
                   if min_freq <= n < len(types))
    if not vocab:
        raise ValueError("no informative CL terms: every term is universal or unique")

    idx = {term: i for i, term in enumerate(vocab)}
    X = np.zeros((len(types), len(vocab)), dtype=np.float32)
    for r, s in enumerate(term_sets):
        for term in s:
            if term in idx:
                X[r, idx[term]] = 1.0
    return X, np.array(types), len(vocab)


def ontology_hierarchy(cl_ids, labels, n_groups=28, obo="/tmp/cl-basic.obo"):
    """``{cell_type: group_id}`` from CL lineage. Types without a CL id are left out;
    ``build_hierarchical_reference`` puts those in its catch-all group."""
    X, types, n_terms = ontology_features(cl_ids, labels, obo=obo)
    grp = aj.discover_hierarchy(X, types, n_groups=n_groups)
    return grp, {"n_types_with_cl": len(types), "n_cl_terms": n_terms}


if __name__ == "__main__":
    import collections

    import scanpy as sc

    path = sys.argv[1]
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    a = sc.read_h5ad(path)
    grp, info = ontology_hierarchy(a.obs["cell_type_ontology_term_id"],
                                   a.obs["cell_type"], n_groups=n_groups)
    sizes = collections.Counter(grp.values())
    print(f"{info['n_types_with_cl']} types / {info['n_cl_terms']} CL terms "
          f"-> {len(sizes)} groups")
    print("group sizes:", sorted(sizes.values(), reverse=True))
    for g, _ in sizes.most_common(5):
        members = [t for t, gg in grp.items() if gg == g][:6]
        print(f"  {g} ({sizes[g]}): {', '.join(members)}")
