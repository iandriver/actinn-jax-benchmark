"""Two-stage comparison: scPRINT (broad) shaping a fast actinn-jax (fine) classifier.

Runs off the cached scPRINT embeddings/predictions (two_stage_embed.py). All small
models are actinn-jax (fast, CPU). Methods:
  flat-full     : actinn-jax on the full reference                (baseline)
  random-K      : actinn-jax on a random K/type subsample          (size control)
  coreset-K     : actinn-jax on a scPRINT-embedding medoid coreset  (A, offline)
  hierarchy     : scPRINT-embedding coarse groups -> coarse + per-group fine (B, offline)
  scoping       : restrict ref classes to scPRINT's query calls     (C, online)
  routing       : scPRINT coarse call routes each query cell to a fine model (C, online)
  scprint-alone : scPRINT zero-shot CL predictions                  (reference point)

Run in the core env: .venv/bin/python benchmark/explore/two_stage_compare.py
"""
import os, time, warnings; warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, numpy as np, scanpy as sc, pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, "/Users/iandriver/Downloads/actinn-jax-benchmark")
from benchmark import metrics
import actinn_jax as aj

LABEL = "cell_type"
OBO = "/tmp/cl-basic.obo"
ref = sc.read_h5ad("/tmp/ts_ref.h5ad")
qry = sc.read_h5ad("/tmp/ts_query.h5ad")
EMB = ref.obsm["scprint_emb"]
truth = qry.obs[LABEL].astype(str).to_numpy()
anc = metrics.load_cl_ancestors(OBO) if os.path.exists(OBO) else None
name2cl = dict(zip(ref.obs[LABEL].astype(str), ref.obs["cell_type_ontology_term_id"].astype(str)))
truth_cl = qry.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
rows = []


def score(name, pred, train_s, infer_s, n_ref, extra_s=0.0):
    pred = np.asarray(pred, dtype=object)
    pred_cl = np.array([name2cl.get(str(p), "") for p in pred])
    m = metrics.compute(truth, pred, ontology=anc, truth_cl=truth_cl, pred_cl=pred_cl)
    rows.append({"method": name, "accuracy": round(m["accuracy"], 3),
                 "macro_f1": round(m["macro_f1"], 3),
                 "ontology": round(m.get("ontology_concordance", float("nan")), 3),
                 "train_s": round(train_s, 1), "infer_s": round(infer_s, 2),
                 "scprint_s": round(extra_s, 1), "n_ref": int(n_ref)})
    print(f"  {name:14} acc {m['accuracy']:.3f} f1 {m['macro_f1']:.3f} "
          f"onto {m.get('ontology_concordance',0):.3f} | train {train_s:.1f}s "
          f"infer {infer_s:.2f}s nref {n_ref}", flush=True)


def fit_predict(ref_ad, label, query_ad=qry):
    t = time.time(); model = aj.train_reference(ref_ad, train_label_name=label, print_cost=False)
    tr = time.time() - t
    t = time.time(); frame, _ = model.predict_frame(query_ad); inf = time.time() - t
    return frame["celltype"].to_numpy(), tr, inf, model


# ---- flat-full baseline ----
print("flat-full...", flush=True)
pred, tr, inf, _ = fit_predict(ref, LABEL)
score("flat-full", pred, tr, inf, ref.n_obs)

K = 50
# ---- random-K subsample ----
rng = np.random.default_rng(0)
labels = ref.obs[LABEL].astype(str).to_numpy()
ridx = np.concatenate([rng.choice(np.where(labels == c)[0], min(K, (labels == c).sum()), replace=False)
                       for c in np.unique(labels)])
pred, tr, inf, _ = fit_predict(ref[np.sort(ridx)].copy(), LABEL)
score(f"random-{K}", pred, tr, inf, len(ridx))

# ---- coreset-K (scPRINT-embedding medoids) ----
def coreset_idx(emb, labels, k, seed=0):
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= k:
            keep += list(idx); continue
        km = KMeans(k, n_init=3, random_state=seed).fit(emb[idx])
        for ci in range(k):
            members = idx[km.labels_ == ci]
            if len(members):
                d = ((emb[members] - km.cluster_centers_[ci]) ** 2).sum(1)
                keep.append(int(members[d.argmin()]))
    return np.array(sorted(set(keep)))

cidx = coreset_idx(EMB, labels, K)
pred, tr, inf, _ = fit_predict(ref[cidx].copy(), LABEL)
score(f"coreset-{K}", pred, tr, inf, len(cidx))

# ---- hierarchy (coarse groups) ----
NG = 8
types = np.unique(labels)
cent = np.vstack([EMB[labels == t].mean(0) for t in types])
grp_scprint = dict(zip(types, fcluster(linkage(cent, "ward"), NG, criterion="maxclust")))
# control: random grouping of fine types into the same number of groups
_rng = np.random.default_rng(0)
grp_random = dict(zip(types, _rng.integers(1, NG + 1, size=len(types))))


def run_hierarchy(name, grp):
    ref.obs["coarse"] = [str(grp[t]) for t in labels]
    t0 = time.time()
    coarse_model = aj.train_reference(ref, train_label_name="coarse", print_cost=False)
    fine_models = {}
    for g in np.unique(list(grp.values())):
        sub = ref[ref.obs["coarse"] == str(g)]
        gt = sub.obs[LABEL].unique()
        fine_models[str(g)] = (str(gt[0]) if len(gt) == 1
                               else aj.train_reference(sub.copy(), train_label_name=LABEL, print_cost=False))
    htrain = time.time() - t0
    t0 = time.time()
    cpred = coarse_model.predict_frame(qry)[0]["celltype"].to_numpy()
    hpred = np.empty(qry.n_obs, dtype=object)
    for g in np.unique(cpred):
        mask = cpred == g; fm = fine_models.get(str(g))
        hpred[mask] = fm if isinstance(fm, str) else fm.predict_frame(qry[mask])[0]["celltype"].to_numpy()
    score(name, hpred, htrain, time.time() - t0, ref.n_obs)
    return coarse_model, fine_models, grp, htrain


coarse_model, fine_models, grp, hier_train_s = run_hierarchy(f"hierarchy-scprint(G{NG})", grp_scprint)
run_hierarchy(f"hierarchy-random(G{NG})", grp_random)  # control: structure without scPRINT

# ---- scprint-alone (zero-shot CL preds, scored in CL space) ----
spred_cl = qry.obs["scprint_pred_cl"].astype(str).to_numpy()
if anc is not None:
    msp = metrics.compute(truth_cl, spred_cl, ontology=anc, truth_cl=truth_cl, pred_cl=spred_cl)
    rows.append({"method": "scprint-alone", "accuracy": round(msp["accuracy"], 3),
                 "macro_f1": round(msp["macro_f1"], 3),
                 "ontology": round(msp.get("ontology_concordance", float("nan")), 3),
                 "train_s": 0.0, "infer_s": float("nan"), "scprint_s": float("nan"), "n_ref": 0})
    print(f"  scprint-alone  acc {msp['accuracy']:.3f} onto {msp.get('ontology_concordance',0):.3f}", flush=True)

# ---- class scoping (online: restrict ref to scPRINT-implicated types) ----
def lineage_related(a, b):
    return a == b or (anc and (a in anc.get(b, ()) or b in anc.get(a, ())))

if anc is not None:
    sp_cls = set(spred_cl)
    scope_types = [t for t in types if any(lineage_related(name2cl.get(t, ""), pc) for pc in sp_cls)]
    if len(scope_types) >= 2:
        sref = ref[ref.obs[LABEL].isin(scope_types)].copy()
        pred, tr, inf, _ = fit_predict(sref, LABEL)
        score(f"scoping({len(scope_types)}/{len(types)}t)", pred, tr, inf, sref.n_obs)

# ---- routing (online: scPRINT's coarse call routes each query cell to a fine model) ----
if anc is not None:
    def sp_to_group(pcl):
        for t in types:
            if lineage_related(name2cl.get(t, ""), pcl):
                return str(grp[t])
        return None
    t0 = time.time()
    sp_groups = np.array([sp_to_group(pc) for pc in spred_cl], dtype=object)
    unmapped = np.array([g is None for g in sp_groups])
    if unmapped.any():  # fall back to the trained coarse model where scPRINT didn't map
        sp_groups[unmapped] = coarse_model.predict_frame(qry[unmapped])[0]["celltype"].to_numpy()
    rpred = np.empty(qry.n_obs, dtype=object)
    for g in np.unique(sp_groups):
        mask = sp_groups == g; fm = fine_models.get(str(g))
        rpred[mask] = ("unknown" if fm is None else
                       fm if isinstance(fm, str) else
                       fm.predict_frame(qry[mask])[0]["celltype"].to_numpy())
    score(f"routing(G{NG})", rpred, hier_train_s, time.time() - t0, ref.n_obs)

print()
df = pd.DataFrame(rows)
df.to_csv("/tmp/two_stage_results.csv", index=False)
print(df.to_string(index=False))
print("TWO_STAGE_DONE", flush=True)
