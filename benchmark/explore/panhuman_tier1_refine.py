"""Does the broad->refined hand-off still pay when tier 1 is Pan-human Azimuth?

PAPER.md section 3.5 claims the value of actinn-jax's workflow is the hand-off from a
broad model to a focused reference. That claim was measured with *our* broad model as
tier 1. Pan-human Azimuth is a stronger broad model (docs/PAN_HUMAN_AZIMUTH.md), so the
honest test is whether the hand-off still adds anything on top of it.

Everything runs on the leakage-free cross-study liver split: reference = 6 HLiCA studies,
query = a withheld study. Tier 2 is trained only on the reference.

Arms
----
tier1_only          Pan-human Azimuth's own fine call.
tier2_only          actinn-jax trained on the reference, unrestricted (the control that
                    decides whether tier 1 contributes anything at all).
tier1_scopes_tier2  tier 2's class probabilities masked to the classes compatible with
                    tier 1's *coarse* call for that cell, then renormalized -- the
                    zero-retrain narrowing actinn-jax ships, driven by Pan-human Azimuth.
oracle_scope        the same masking driven by the TRUE coarse lineage. Not achievable in
                    practice; it is the ceiling on what any tier 1 could buy.

Scored by ontology-aware concordance, the only metric comparable across the two label
vocabularies (see docs/PAN_HUMAN_AZIMUTH.md).

    .venv-protocloud/bin/python benchmark/explore/panhuman_tier1_refine.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")

import numpy as np
import pandas as pd
import yaml

import actinn_jax as aj
import benchmark.driver as drv
from benchmark import metrics

OBO = "/tmp/cl-basic.obo"
TIER1 = "/tmp/panhuman_tier1_liver_cross.parquet"
LABEL = "cell_type"
CL_COL = "cell_type_ontology_term_id"

anc = metrics.load_cl_ancestors(OBO)


def related(a, b):
    """True if a and b are the same CL term or one is an ancestor of the other."""
    if not a or not b or a == "unknown" or b == "unknown":
        return False
    return a == b or a in anc.get(b, ()) or b in anc.get(a, ())


def concordance(true_cl, pred_cl):
    ok = n = 0
    for t, p in zip(true_cl, pred_cl):
        if not isinstance(t, str) or not t:
            continue
        n += 1
        ok += related(t, p)
    return ok / max(n, 1)


cfg = yaml.safe_load(open("configs/panhuman_compare.yaml"))
ds = [d for d in cfg["datasets"] if d["name"] == "liver_cross"][0]
ref, query = drv.build_pair(ds, LABEL)
true_cl = query.obs[CL_COL].astype(str).to_numpy()
print(f"ref {ref.n_obs} cells / query {query.n_obs} cells "
      f"({ref.obs[LABEL].nunique()} reference types)\n")

tier1 = pd.read_parquet(TIER1).loc[list(query.obs_names)]

# ---- tier 2: actinn-jax trained on the reference only ----------------------------
model = aj.train_reference(ref, train_label_name=LABEL, print_cost=False)
classes = list(model.classes)
P = np.asarray(model.predict_proba(query))
print(f"tier 2: {len(classes)} classes, probability matrix {P.shape}")

# CL id for each tier-2 class, taken from the reference's own annotations.
cls_cl = (ref.obs.drop_duplicates(LABEL).set_index(LABEL)[CL_COL].astype(str).to_dict())
class_cl = np.array([cls_cl.get(c, "") for c in classes], dtype=object)

argmax_cl = np.array([class_cl[i] for i in P.argmax(axis=1)], dtype=object)

# ---- masking -----------------------------------------------------------------------
def scoped(coarse_cl):
    """Mask each cell's tier-2 classes to those compatible with a coarse CL call."""
    out = np.empty(len(coarse_cl), dtype=object)
    cache = {}
    for i, c in enumerate(coarse_cl):
        if c not in cache:
            cache[c] = np.array([related(c, cc) for cc in class_cl])
        m = cache[c]
        row = P[i] * m if m.any() else P[i]     # empty mask -> leave the cell alone
        out[i] = class_cl[int(row.argmax())]
    return out


broad_cl = tier1["azimuth_broad_CL_ID"].astype(str).to_numpy()
fine_cl = tier1["azimuth_fine_CL_ID"].astype(str).to_numpy()

# Oracle coarse call: the *correct* answer expressed at tier 1's own granularity, i.e.
# whichever of tier 1's coarse terms is an ancestor of the true type. Passing the true
# fine CL id here instead would leak the answer -- masking to classes related to the
# truth makes the argmax trivially right and scores 1.000 by construction.
coarse_vocab = [c for c in pd.unique(broad_cl) if c and c != "unknown"]
oracle_cl = np.array(
    [next((c for c in coarse_vocab if related(c, t)), "unknown") for t in true_cl],
    dtype=object,
)
print(f"coarse vocabulary: {len(coarse_vocab)} terms; "
      f"oracle resolves {np.mean(oracle_cl != 'unknown'):.1%} of cells")
print(f"tier-1 coarse call agrees with oracle on "
      f"{np.mean(broad_cl == oracle_cl):.1%} of cells\n")

# Our own shipped broad model, as the tier-1 incumbent Pan-human Azimuth is compared to.
broad_ours = aj.bundled_reference("broad_human_v1")
ours_frame, _ = broad_ours.predict_frame(query)
ours_name = ours_frame.loc[list(query.obs_names), "celltype"].to_numpy()
ours_cl = np.array([broad_ours.class_to_cl.get(p, "") for p in ours_name], dtype=object)

arms = {
    "tier1_only  (broad_human_v1, ours)": ours_cl,
    "tier1_only  (Pan-human Azimuth fine)": fine_cl,
    "tier2_only  (actinn-jax, unrestricted)": argmax_cl,
    "tier1_scopes_tier2 (PHA coarse -> mask)": scoped(broad_cl),
    "oracle_scope (perfect coarse -> mask)": scoped(oracle_cl),
}

n_empty = sum(1 for c in np.unique(broad_cl)
              if not any(related(c, cc) for cc in class_cl))
print(f"tier-1 coarse calls with no compatible tier-2 class: {n_empty} "
      f"of {len(np.unique(broad_cl))} distinct\n")

print(f"{'arm':42s} {'ontology concordance':>20s}")
print("-" * 64)
for name, pred in arms.items():
    print(f"{name:42s} {concordance(true_cl, pred):>20.3f}")
