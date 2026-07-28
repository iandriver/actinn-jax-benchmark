"""Stage 1 of Pan-human Azimuth distillation: label a corpus with the teacher.

Pan-human Azimuth (``panhumanpy``, MIT) is a pretrained pan-human classifier over a fixed
8-level typology. Distilling it into actinn-jax gives a student with the *same harmonized,
CL-mapped vocabulary* but actinn-jax's cost profile -- and, critically, a build that needs
**no GPU and no labels**: the teacher supplies both the labels and the coarse->fine
hierarchy that the shipped ``broad_human_v1`` currently gets from scPRINT embeddings.

Keras/TF and JAX cannot share a process, so this writes teacher labels to disk for
``distill_train.py`` to consume. It also writes the exact count matrix it scored, so the
student trains on byte-identical input and any student/teacher disagreement is the model,
not the preprocessing.

    .venv-panhuman/bin/python benchmark/explore/distill_dump.py [--cap 400] [--out DIR]
"""

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# (name, path, label column, CL-id column or None). These are the local atlases the paper
# already uses; the labels are carried only to *score* the student, never to train it.
CORPORA = [
    ("lung", os.path.expanduser("~/Downloads/krasnow_lung_atlas_10x.h5ad"),
     "cell_type", "cell_type_ontology_term_id"),
    ("liver", "/Volumes/IanSSD/hlica/liver_intra.h5ad",
     "cell_type", "cell_type_ontology_term_id"),
    ("blood_gut", "/Volumes/IanSSD/hlica/blood_gut_intra.h5ad",
     "cell_type", None),
    # Census-wide pull from fetch_census_wide.py -- the breadth arm. Skipped silently when
    # absent, so the three local atlases still work without it.
    ("census", os.path.join(os.environ.get("ACTINN_REF_WORK", "/tmp/actinn_ref_build"),
                            "census_wide_ref.h5ad"),
     "cell_type", "cell_type_ontology_term_id"),
]

# Symbol columns to normalise to `feature_name`; panhumanpy keys on gene symbols while
# every corpus here is Ensembl-keyed.
_SYMBOL_COLS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name", "symbol")

LEVELS = ("azimuth_broad", "azimuth_medium", "azimuth_fine", "final_level_labels")


def counts_with_symbols(a):
    """Raw counts + a var frame that still carries gene symbols, keyed by Ensembl id."""
    if a.raw is not None:
        X, var = a.raw.X, a.raw.var.copy()
        var.index = a.raw.var_names
    else:
        X, var = a.X, a.var.copy()
    col = next((c for c in _SYMBOL_COLS if c in var.columns), None)
    if col is None:
        raise ValueError(f"no gene-symbol column in .var (looked for {_SYMBOL_COLS})")
    out = ad.AnnData(X=sp.csr_matrix(X).astype("float32"), obs=a.obs.copy(), var=var)
    out.var["feature_name"] = var[col].astype(str).values
    return out


def stratified(labels, cap, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep.append(rng.choice(idx, cap, replace=False) if len(idx) > cap else idx)
    return np.sort(np.concatenate(keep))


def teacher_labels(a, batch=8192):
    """Run Pan-human Azimuth over `a` and return every level plus the fine CL id."""
    import panhumanpy as ph

    az = ph.AzimuthNN_base(eval_batch_size=batch)
    az.query_adata(a, feature_names_col="feature_name")
    az.process_query()
    az.run_inference_model()
    az.calibrate_predictions()
    az.process_outputs(mode="minimal")
    for lvl in ("broad", "medium", "fine"):
        az.refine_labels(lvl)
    az.update_cells_meta()
    meta = az.cells_meta.copy()

    # panhumanpy blanks a refined level to boolean False when it cannot reconcile the
    # hierarchy; those cells still carry a deepest-level call. Falling back keeps them in
    # the training set instead of silently distilling only the easy cells.
    for col in LEVELS:
        if col not in meta:
            continue
        s = meta[col]
        blank = ~s.map(lambda v: isinstance(v, str)) | s.astype(str).isin(("False", "nan", ""))
        meta[col] = s.astype(str).where(~blank, meta["final_level_labels"].astype(str))

    for col in LEVELS:
        if col in meta:
            az.cells_meta = meta
            az.map_to_cell_ontology(col, include_cl_id=True)
            meta = az.cells_meta

    keep = [c for c in meta.columns
            if c.startswith("azimuth_") or c.startswith("final_level")]
    meta = meta[keep].astype(str)
    meta.index = list(a.obs_names)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=400, help="cells per label per corpus")
    ap.add_argument("--out", default="/tmp/distill")
    ap.add_argument("--only", default="", help="comma-separated corpus names to run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    for name, path, label_key, cl_key in CORPORA:
        if only and name not in only:
            continue
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found", flush=True)
            continue
        t = time.time()
        adata = sc.read_h5ad(path)
        idx = stratified(adata.obs[label_key].astype(str).to_numpy(), a.cap)
        sub = counts_with_symbols(adata[idx].copy())
        del adata

        cols = {"truth": sub.obs[label_key].astype(str).values}
        cols["truth_cl"] = (sub.obs[cl_key].astype(str).values if cl_key
                            else np.array(["unknown"] * sub.n_obs))
        sub.obs = pd.DataFrame(cols, index=sub.obs_names)
        sub.obs["corpus"] = name
        sub.var = sub.var[["feature_name"]]
        sub.write_h5ad(f"{a.out}/{name}_corpus.h5ad")
        print(f"{name}: {sub.shape} / {sub.obs.truth.nunique()} truth types "
              f"(loaded+subset {time.time()-t:.0f}s)", flush=True)

        t = time.time()
        meta = teacher_labels(sub)
        meta.to_parquet(f"{a.out}/{name}_teacher.parquet")
        rate = sub.n_obs / max(time.time() - t, 1e-9)
        print(f"{name}: teacher {time.time()-t:.0f}s ({rate:.0f} cells/s), "
              f"{meta['azimuth_fine'].nunique()} fine labels, "
              f"{(meta['azimuth_fine'] == 'Unassigned').mean():.1%} Unassigned", flush=True)

    print("DISTILL_DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
