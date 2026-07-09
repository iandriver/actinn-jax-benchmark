"""Preprocessing/normalization/scaling ablations for actinn-jax on the OP datasets.

Motivated by scANVI+scArches: its edge over the gene-MLP is NOT the VAE (plain scanvi
0.826 < actinn-jax 0.837) but (a) count-model normalization and (b) query-side domain
adaptation. This tests CPU-cheap analogs, reusing actinn-jax's exact MLP (au.train /
au.predict_proba, same epochs/seed/layers) and swapping ONLY the feature pipeline:

  baseline : log2(CP10k+1)  + ACTINN percentile expr/CV gene filter        (current)
  E1       : baseline features + per-gene standardization (ref μ,σ frozen)
  E2       : analytic Pearson residuals (ref-frozen) on the filtered genes
  E3       : E1 features but genes chosen by Pearson-residual variance (HVG)
  E4       : self-training / pseudo-labeling on the query (best base config)

Each mode changes one thing vs its stated base, so effects are isolable. Reference stats
(scaler, residual expectations, gene set) are fit on the REFERENCE and frozen for the
query -- standard reference-mapping discipline, no query leakage.

    python preproc_ablation.py <dataset_dir> [out_csv] [--modes baseline,E1,E2,E3,E4]
"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, anndata as ad
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax")
import actinn_jax.actinn_utils as au
from actinn_jax.actinn_predict import _normalize, _gene_filter, _encode_labels
from sklearn.metrics import accuracy_score, f1_score

THETA = 100.0            # Pearson-residual NB dispersion (Lause/Kobak/Berens 2021)
SELF_TRAIN_CONF = 0.90   # confidence threshold for pseudo-labels
SEED = au.DEFAULT_SEED


def load_counts(path):
    a = sc.read_h5ad(path, backed="r")
    X = a.layers["counts"][:]
    obs, var, names = a.obs.copy(), a.var.copy(), list(a.obs_names)
    try: a.file.close()
    except Exception: pass
    b = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var); b.obs_names = names
    return b


def pearson_residuals(counts_csr, gene_sum_ref, total_ref, n_clip):
    """Analytic NB Pearson residuals with reference-frozen gene expectations.
    mu_ij = cell_total_i * (gene_sum_ref_j / total_ref); residual = (x-mu)/sqrt(mu+mu^2/theta)."""
    X = counts_csr.astype(np.float64)
    cell_tot = np.asarray(X.sum(1)).ravel()
    gene_prop = gene_sum_ref / max(total_ref, 1.0)
    mu = np.outer(cell_tot, gene_prop)                      # dense (cells x genes)
    mu = np.maximum(mu, 1e-8)
    Z = (X.toarray() - mu) / np.sqrt(mu + mu * mu / THETA)
    clip = np.sqrt(n_clip)
    return np.clip(Z, -clip, clip).astype(np.float32)


def fit_predict(Xtr_dense, ytr_int, n_classes, Xte_dense, classes, true, epochs=au.DEFAULT_NUM_EPOCHS):
    Y = au.one_hot(ytr_int, n_classes)
    params = au.train(Xtr_dense, Y, num_epochs=epochs, seed=SEED, print_cost=False)
    proba = au.predict_proba(params, Xte_dense)
    pred = np.array([classes[i] for i in proba.argmax(1)])
    return params, proba, accuracy_score(true, pred), f1_score(true, pred, average="macro")


def main():
    ds_dir = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    modes = ["baseline", "E1", "E2", "E3", "E4"]
    for a in sys.argv:
        if a.startswith("--modes"): modes = a.split("=")[1].split(",")
    name = os.path.basename(ds_dir.rstrip("/"))

    tr = load_counts(f"{ds_dir}/train.h5ad"); tr.obs["label"] = tr.obs["label"].astype(str)
    te = load_counts(f"{ds_dir}/test.h5ad")
    sol = sc.read_h5ad(f"{ds_dir}/solution.h5ad", backed="r")
    if "hvg" in tr.var.columns:
        m = tr.var["hvg"].astype(bool).to_numpy(); tr = tr[:, m].copy(); te = te[:, m].copy()
    Xr, Xq = tr.X.tocsr(), te.X.tocsr()
    labels = tr.obs["label"].to_numpy()
    yint, classes = _encode_labels(labels)
    true = sol.obs.loc[list(te.obs_names), "label"].astype(str).to_numpy()
    print(f"{name}: train {Xr.shape}, test {Xq.shape[0]}, {len(classes)} labels", flush=True)

    # shared log-norm + baseline gene filter
    Xr_ln, Xq_ln = _normalize(Xr), _normalize(Xq)
    base_mask = _gene_filter(Xr_ln); base_idx = np.where(base_mask)[0]
    k = len(base_idx)

    rows = []
    def emit(mode, acc, f1, ng, note=""):
        r = {"dataset": name, "mode": mode, "accuracy": round(acc, 4),
             "f1_macro": round(f1, 4), "n_genes": ng, "note": note}
        rows.append(r); print("  RESULT", r, flush=True)

    # cache for E4
    best = {"acc": -1, "Xtr": None, "Xte": None}

    for mode in modes:
        t = time.time()
        if mode == "baseline":
            Xtr = Xr_ln[:, base_idx].toarray(); Xte = Xq_ln[:, base_idx].toarray()
            _, _, acc, f1 = fit_predict(Xtr, yint, len(classes), Xte, classes, true)
            emit(mode, acc, f1, k)
        elif mode == "E1":
            Xtr = Xr_ln[:, base_idx].toarray(); Xte = Xq_ln[:, base_idx].toarray()
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            _, _, acc, f1 = fit_predict(Xtr, yint, len(classes), Xte, classes, true)
            emit(mode, acc, f1, k, "ref-std")
        elif mode == "E2":
            gsum = np.asarray(Xr.sum(0)).ravel(); tot = float(Xr.sum())
            Rr = pearson_residuals(Xr, gsum, tot, Xr.shape[0])[:, base_idx]
            Rq = pearson_residuals(Xq, gsum, tot, Xr.shape[0])[:, base_idx]
            _, _, acc, f1 = fit_predict(Rr, yint, len(classes), Rq, classes, true)
            emit(mode, acc, f1, k, "pearson-resid")
        elif mode == "E3":
            # pick top-k genes by Pearson-residual variance on the reference
            gsum = np.asarray(Xr.sum(0)).ravel(); tot = float(Xr.sum())
            Rr_full = pearson_residuals(Xr, gsum, tot, Xr.shape[0])
            var = Rr_full.var(0); idx = np.argsort(var)[::-1][:k]
            Xtr = Xr_ln[:, idx].toarray(); Xte = Xq_ln[:, idx].toarray()
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            _, _, acc, f1 = fit_predict(Xtr, yint, len(classes), Xte, classes, true)
            emit(mode, acc, f1, k, "pearsonHVG+std")
        elif mode == "E4":
            # self-training on the best base config seen so far (fallback: E1-style std)
            Xtr = Xr_ln[:, base_idx].toarray(); Xte = Xq_ln[:, base_idx].toarray()
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            _, proba, acc0, f10 = fit_predict(Xtr, yint, len(classes), Xte, classes, true)
            conf = proba.max(1); keep = conf >= SELF_TRAIN_CONF
            pseudo = proba.argmax(1)[keep]
            Xtr2 = np.vstack([Xtr, Xte[keep]]); y2 = np.concatenate([yint, pseudo])
            _, _, acc, f1 = fit_predict(Xtr2, y2, len(classes), Xte, classes, true)
            emit(mode, acc, f1, k, f"selftrain +{int(keep.sum())}c@{SELF_TRAIN_CONF} (base {acc0:.3f})")
        print(f"    [{mode} {time.time()-t:.0f}s]", flush=True)

    if out_csv:
        hdr = not os.path.exists(out_csv)
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=hdr, index=False)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
