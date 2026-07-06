"""Build a focused liver reference from HLiCA (Edgar et al. 2026, bioRxiv 10.64898/
2026.06.30.735539), a 522k-cell, 110-donor, expert-curated integrated human liver atlas
-- a major upgrade over the ~26k-cell, 1-2-dataset zonation reference used previously.

Coarse hierarchy needs no scPRINT: HLiCA already ships as 6 expert-curated lineage files
(hepatocyte/cholangiocyte/endothelial/myeloid/mesenchyme/lymphocyte) -- that split IS the
coarse grouping. Fine labels: HLiCA's own cell_type for 5 lineages; for hepatocytes, the
richer `author_cell_type` (7 substates incl. Periportal/Pericentral zonation + metabolic
states), since the standardized `cell_type` collapses those extra states to generic
"hepatocyte". One contaminant cluster ("Cycling", 1179 cells, actually lymphocyte per its
own cell_type_ontology_term_id) is dropped from the hepatocyte lineage.

Validation is a CROSS-STUDY held-out split, not a random one: our previous liver query
(benchmark/explore/fetch_liver_query.py, CELLxGENE dataset ddb22b3d-...) turns out to BE
HLiCA's "Andrews_2022" component study -- reusing it as an "external" query would leak
training data. So Andrews_2022 is held out entirely as test; everything else trains.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark/benchmark")
import actinn_jax as aj
from metrics import load_cl_ancestors, _ontology

SRC = "/Volumes/IanSSD/hlica"
OUT = "/Users/iandriver/Downloads/actinn-jax/actinn_jax/references/liver_hlica_v1"
OBO = "/tmp/cl-basic.obo"
N_HVG = 4000
HELD_OUT_STUDY = "Andrews_2022"
LINEAGES = ["hepatocyte", "cholangiocyte", "endothelial", "myeloid", "mesenchyme", "lymphocyte"]
HEP_CL = {
    "Periportal Hepatocyte": "CL:0019026",
    "Pericentral Hepatocyte": "CL:0019029",
    "Ribosomal+ Hepatocyte": "CL:0000182",
    "Mito+ Hepatocyte": "CL:0000182",
    "SERPINE1+ Hepatocyte": "CL:0000182",
    "UGT+ Hepatocyte": "CL:0000182",
}

t0 = time.time()
parts = []
for lin in LINEAGES:
    a = sc.read_h5ad(f"{SRC}/{lin}.h5ad")
    r = a.raw.to_adata()
    r.obs = a.obs.copy()
    if lin == "hepatocyte":
        r = r[r.obs["author_cell_type"] != "Cycling"].copy()   # mislabeled lymphocyte contaminant
        r.obs["_fine"] = r.obs["author_cell_type"].astype(str)
        r.obs["_cl"] = r.obs["_fine"].map(HEP_CL)
    else:
        r.obs["_fine"] = r.obs["cell_type"].astype(str)
        r.obs["_cl"] = r.obs["cell_type_ontology_term_id"].astype(str)
    r.obs["_lineage"] = lin
    r.obs = r.obs[["_fine", "_cl", "_lineage", "STUDY"]]
    parts.append(r)
    print(f"{lin}: {r.n_obs} cells, {r.obs['_fine'].nunique()} fine types "
          f"({time.time()-t0:.0f}s)", flush=True)

ref = ad.concat(parts, join="outer", merge="first")
print(f"\ncombined: {ref.shape}, {ref.obs['_fine'].nunique()} total fine types across "
      f"{len(LINEAGES)} lineages ({time.time()-t0:.0f}s)", flush=True)

hierarchy = dict(zip(ref.obs["_fine"], ref.obs["_lineage"]))
class_to_cl_all = dict(zip(ref.obs["_fine"], ref.obs["_cl"]))

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

# zonation-specific: hepatocyte AND endothelial cross-tabs
for lineage, zones in [("hepatocyte", ["Periportal Hepatocyte", "Pericentral Hepatocyte"]),
                       ("endothelial", ["endothelial cell of periportal hepatic sinusoid",
                                        "endothelial cell of pericentral hepatic sinusoid"])]:
    sub_true = test.obs["_fine"].isin(zones)
    if sub_true.sum() == 0:
        print(f"\n{lineage} zonation: no held-out cells")
        continue
    t_zone = true[sub_true.values]
    p_zone = pred[sub_true.values]
    exact_zone = float((t_zone == p_zone).mean())
    flip = float(((t_zone == zones[0]) & (p_zone == zones[1])).sum() +
                ((t_zone == zones[1]) & (p_zone == zones[0])).sum()) / max(sub_true.sum(), 1)
    print(f"\n{lineage} zonation ({sub_true.sum()} cells): exact-zone {exact_zone:.3f} | "
          f"portal<->central flip rate {flip:.3f}")
    print(pd.crosstab(pd.Series(t_zone, name="true"), pd.Series(p_zone, name="pred")).to_string())

# ---- compare vs the big 798-type broad reference on the SAME held-out cells ----
broad = aj.bundled_reference("broad_human_v1")
bframe, _ = broad.predict_frame(test)
bpred = bframe["celltype"].values
bpred_cl = np.array([broad.class_to_cl.get(p, "") for p in bpred])
bexact = float((bpred_cl == true_cl)[keep].mean())
bonto = _ontology(true_cl, bpred_cl, anc, keep)
print(f"\nbroad_human_v1 (798 types) on same held-out set: exact-CL {bexact:.3f} | "
      f"ontology {bonto:.3f}  <- vs. new focused model exact {exact:.3f} / ontology {onto:.3f}",
      flush=True)

# ---- ship: retrain on ALL data (train+test) for the final shipped reference ----
t = time.time()
model = aj.build_hierarchical_reference(hvg_subset(ref), "_fine", hierarchy=hierarchy,
                                        ontology_key="_cl", print_cost=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model.save(OUT)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
print(f"\nshipped {len(model.classes)} types / {len(LINEAGES)} lineages / {sz:.1f}MB "
      f"in {time.time()-t:.0f}s -> {OUT}", flush=True)
print("HLICA_LIVER_DONE", flush=True)
