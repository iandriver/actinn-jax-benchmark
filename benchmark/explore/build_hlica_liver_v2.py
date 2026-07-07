"""liver_hlica_v2: fixes the two gaps found in docs/HLICA_EDGE_CASES.md.

v1 (build_hlica_liver.py) used `author_cell_type` only for the hepatocyte lineage and
`cell_type` (standardized, coarser) for the other 5 -- silently losing HLiCA's own
headline findings (NRXN1+ stromal cells, MAMLD1+ trans monocytes, TREM2+ macrophages,
Type 1/2 cDCs, Bright/Dim NK, finer vascular-bed distinctions) to collapsing into
generic parent labels. v1 also excluded plasmacytoid dendritic cells entirely, since
they're structurally absent from both the myeloid.h5ad and lymphocyte.h5ad lineage
files (HLiCA's own text: "transcriptomically distinct from both").

v2:
  1. author_cell_type uniformly across all 6 lineages (v1's inconsistency, fixed).
  2. class_to_cl built directly from data (per-row cell_type_ontology_term_id), not a
     hand-typed dict -- for a collapsed label this naturally recovers its parent
     standardized CL id, exactly matching what a human would have typed by hand.
  3. "Cycling" cells (cell-cycle-dominated transcriptomes; found in 4 of 6 lineage
     files, always resolving to cell_type="lymphocyte" regardless of host file) are
     RE-ROUTED to the lymphocyte lineage using their own cell_type/CL id, not dropped
     and not left in a possibly-wrong lineage by file-of-origin.
  4. The 790 plasmacytoid dendritic cells, pulled from all_cells.h5ad (absent from
     every lineage file), added as their own 7th coarse group.
  5. hierarchy is built via majority-vote per fine label (robust), not dict(zip(...))
     last-write-wins (v1's fragile, order-dependent construction).

Same cross-study validation as v1: train on 6 studies, test on the withheld 7th
(Andrews_2022) -- confirmed via the STUDY column to be the source of our own earlier
liver_zonation_query.h5ad, so it cannot be reused as an "external" test.
"""
import os, sys, gc, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark/benchmark")
import actinn_jax as aj
from metrics import load_cl_ancestors, _ontology

SRC = "/Volumes/IanSSD/hlica"
OUT = "/Users/iandriver/Downloads/actinn-jax/actinn_jax/references/liver_hlica_v2"
OBO = "/tmp/cl-basic.obo"
N_HVG = 4000
HELD_OUT_STUDY = "Andrews_2022"
LINEAGES = ["hepatocyte", "cholangiocyte", "endothelial", "myeloid", "mesenchyme", "lymphocyte"]

t0 = time.time()
parts = []
for lin in LINEAGES:
    a = sc.read_h5ad(f"{SRC}/{lin}.h5ad")
    r = a.raw.to_adata()
    r.obs = a.obs.copy()
    r.obs["_fine"] = r.obs["author_cell_type"].astype(str)
    r.obs["_cl"] = r.obs["cell_type_ontology_term_id"].astype(str)
    r.obs["_lineage"] = lin
    # re-route Cycling cells to their true lineage (lymphocyte), using their OWN
    # cell_type/CL id -- not the file-of-origin's author_cell_type/lineage
    cyc = r.obs["_fine"] == "Cycling"
    if cyc.any():
        r.obs.loc[cyc, "_fine"] = r.obs.loc[cyc, "cell_type"].astype(str)
        r.obs.loc[cyc, "_lineage"] = "lymphocyte"
    r.obs = r.obs[["_fine", "_cl", "_lineage", "STUDY"]]
    parts.append(r)
    print(f"{lin}: {r.n_obs} cells, {r.obs['_fine'].nunique()} fine types "
          f"({cyc.sum()} Cycling re-routed) ({time.time()-t0:.0f}s)", flush=True)
    del a; gc.collect()

# add plasmacytoid dendritic cells (absent from every lineage file) as a 7th group.
# all_cells.h5ad is 5.3GB -- read backed and slice to just the ~790 pDC rows so we
# never materialize the full 524,699-cell matrix in memory.
allc = sc.read_h5ad(f"{SRC}/all_cells.h5ad", backed="r")
pdc_idx = np.where(allc.obs["cell_type"].astype(str) == "plasmacytoid dendritic cell")[0]
pdc = allc[pdc_idx].to_memory()
pdc = pdc.raw.to_adata()
pdc.obs["_fine"] = "pDC"
pdc.obs["_cl"] = pdc.obs["cell_type_ontology_term_id"].astype(str)
pdc.obs["_lineage"] = "pDC"
pdc.obs = pdc.obs[["_fine", "_cl", "_lineage", "STUDY"]]
parts.append(pdc)
del allc
print(f"pDC: {pdc.n_obs} cells (from all_cells.h5ad, absent from every lineage file) "
      f"({time.time()-t0:.0f}s)", flush=True)

ref = ad.concat(parts, join="outer", merge="first")
print(f"\ncombined: {ref.shape}, {ref.obs['_fine'].nunique()} total fine types across "
      f"{ref.obs['_lineage'].nunique()} coarse groups ({time.time()-t0:.0f}s)", flush=True)

# robust hierarchy: majority-vote lineage per fine label (not last-write-wins dict order)
hierarchy = ref.obs.groupby("_fine")["_lineage"].agg(lambda s: s.value_counts().index[0]).to_dict()
class_to_cl_all = ref.obs.groupby("_fine")["_cl"].agg(lambda s: s.value_counts().index[0]).to_dict()

held_out = ref.obs["STUDY"] == HELD_OUT_STUDY
train_full, test = ref[~held_out].copy(), ref[held_out].copy()
print(f"cross-study split: {train_full.n_obs} train (all studies except {HELD_OUT_STUDY}) / "
      f"{test.n_obs} held-out test ({HELD_OUT_STUDY})", flush=True)


def hvg_subset(adata, n=N_HVG):
    raw = adata.copy()
    sc.pp.normalize_total(raw, target_sum=1e4)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=min(n, raw.n_vars))
    return adata[:, raw.var["highly_variable"].values].copy()


# ---- validate: train on non-Andrews_2022 studies, test on the held-out study ----
t = time.time()
val_model = aj.build_hierarchical_reference(hvg_subset(train_full), "_fine",
                                            hierarchy={k: v for k, v in hierarchy.items()
                                                       if k in set(train_full.obs["_fine"])},
                                            ontology_key="_cl", print_cost=False)
print(f"built validation model ({len(val_model.classes)} types) in {time.time()-t:.0f}s", flush=True)

frame, _ = val_model.predict_frame(test)
pred, true = frame["celltype"].values, test.obs["_fine"].astype(str).values
exact = float((pred == true).mean())
anc = load_cl_ancestors(OBO)
pred_cl = np.array([val_model.class_to_cl.get(p, "") for p in pred])
true_cl = test.obs["_cl"].astype(str).values
keep = np.array([bool(c) for c in true_cl])
onto = _ontology(true_cl, pred_cl, anc, keep)
print(f"\nheld-out ({HELD_OUT_STUDY}, {test.n_obs} cells, cross-study): "
      f"exact {exact:.3f} | ontology-concordant {onto:.3f}", flush=True)

# pDC recall specifically (4 pDC cells happen to be in the held-out Andrews_2022 study)
pdc_test = test.obs["_fine"] == "pDC"
if pdc_test.sum():
    pdc_recall = float((pred[pdc_test.values] == "pDC").mean())
    print(f"pDC held-out recall: {pdc_recall:.2f} ({pdc_test.sum()} cells)", flush=True)

# zonation cross-tabs (same as v1, for direct comparison)
for lineage, zones in [("hepatocyte", ["Periportal Hepatocyte", "Pericentral Hepatocyte"]),
                       ("endothelial", ["Periportal LSEC", "Central Venous LSEC"])]:
    sub_true = test.obs["_fine"].isin(zones)
    if sub_true.sum() == 0:
        continue
    t_zone, p_zone = true[sub_true.values], pred[sub_true.values]
    exact_zone = float((t_zone == p_zone).mean())
    flip = float(((t_zone == zones[0]) & (p_zone == zones[1])).sum() +
                ((t_zone == zones[1]) & (p_zone == zones[0])).sum()) / max(sub_true.sum(), 1)
    print(f"{lineage} zonation ({sub_true.sum()} cells): exact-zone {exact_zone:.3f} | "
          f"flip rate {flip:.3f}", flush=True)

# ---- compare vs v1 and broad_human_v1 on the SAME held-out cells ----
broad = aj.bundled_reference("broad_human_v1")
bframe, _ = broad.predict_frame(test)
bpred_cl = np.array([broad.class_to_cl.get(p, "") for p in bframe["celltype"].values])
print(f"\nbroad_human_v1 (798 types): exact-CL {float((bpred_cl==true_cl)[keep].mean()):.3f} "
      f"| ontology {_ontology(true_cl, bpred_cl, anc, keep):.3f}", flush=True)

v1 = aj.bundled_reference("liver_hlica_v1")
v1frame, _ = v1.predict_frame(test)
v1pred_cl = np.array([v1.class_to_cl.get(p, "") for p in v1frame["celltype"].values])
print(f"liver_hlica_v1 (38 types): exact-CL {float((v1pred_cl==true_cl)[keep].mean()):.3f} "
      f"| ontology {_ontology(true_cl, v1pred_cl, anc, keep):.3f}  "
      f"(note: v1 was trained WITH Andrews_2022 -> this comparison favors v1 unfairly, "
      f"shown only as a sanity floor, not a fair benchmark)", flush=True)
print(f"liver_hlica_v2 (this build, cross-study honest): exact-CL {exact:.3f} | ontology {onto:.3f}",
      flush=True)

# ---- ship: retrain on ALL data (train+test) for the final shipped reference ----
t = time.time()
model = aj.build_hierarchical_reference(hvg_subset(ref), "_fine", hierarchy=hierarchy,
                                        ontology_key="_cl", print_cost=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
ng = len(set(model.type_to_group.values()))
print(f"\nshipped {len(model.classes)} types / {ng} coarse groups / {sz:.1f}MB "
      f"in {time.time()-t:.0f}s -> {OUT}", flush=True)
print("HLICA_LIVER_V2_DONE", flush=True)
