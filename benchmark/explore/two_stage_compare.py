"""Two-stage comparison: scPRINT (broad) shaping a fast actinn-jax (fine) classifier.

Runs off cached scPRINT embeddings (two_stage_embed.py). All small models = actinn-jax.
Env: TS_OUT (cache prefix), TS_LABEL (fine label), TS_BIO (optional biological coarse
column for a control hierarchy, e.g. Lineage), TS_NG (groups), TS_K (coreset/random per
type), TS_RESULTS (csv out).

Methods: flat-full, random-K, coreset-K (A), hierarchy-scprint / -random / -biological
(B), scoping + routing (C, needs CL ids), scprint-alone. Run in the core env.
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

OUT = os.environ.get("TS_OUT", "/tmp/ts")
LABEL = os.environ.get("TS_LABEL", "cell_type")
BIO = os.environ.get("TS_BIO", "")
NG = int(os.environ.get("TS_NG", 8))
K = int(os.environ.get("TS_K", 50))
RESULTS = os.environ.get("TS_RESULTS", "/tmp/two_stage_results.csv")
OBO = "/tmp/cl-basic.obo"

ref = sc.read_h5ad(f"{OUT}_ref.h5ad")
qry = sc.read_h5ad(f"{OUT}_query.h5ad")
EMB = ref.obsm["scprint_emb"]
labels = ref.obs[LABEL].astype(str).to_numpy()
truth = qry.obs[LABEL].astype(str).to_numpy()
types = np.unique(labels)
has_cl = ("cell_type_ontology_term_id" in ref.obs and os.path.exists(OBO)
          and (ref.obs["cell_type_ontology_term_id"].astype(str) != "unknown").mean() > 0.5)
anc = metrics.load_cl_ancestors(OBO) if has_cl else None
if has_cl:
    name2cl = dict(zip(labels, ref.obs["cell_type_ontology_term_id"].astype(str)))
    truth_cl = qry.obs["cell_type_ontology_term_id"].astype(str).to_numpy()
print(f"ref {ref.n_obs} / query {qry.n_obs} | {len(types)} types | CL-ids={has_cl} | bio={BIO or None}", flush=True)
rows = []


def score(name, pred, train_s, infer_s, n_ref):
    pred = np.asarray(pred, dtype=object)
    kw = {}
    if has_cl:
        kw = dict(ontology=anc, truth_cl=truth_cl,
                  pred_cl=np.array([name2cl.get(str(p), "") for p in pred]))
    m = metrics.compute(truth, pred, **kw)
    rows.append({"method": name, "accuracy": round(m["accuracy"], 3),
                 "macro_f1": round(m["macro_f1"], 3),
                 "ontology": round(m.get("ontology_concordance", float("nan")), 3),
                 "train_s": round(train_s, 1), "infer_s": round(infer_s, 2), "n_ref": int(n_ref)})
    print(f"  {name:22} acc {m['accuracy']:.3f} f1 {m['macro_f1']:.3f} "
          f"onto {m.get('ontology_concordance', float('nan')):.3f} | "
          f"train {train_s:.1f}s infer {infer_s:.2f}s nref {n_ref}", flush=True)


def fit_predict(ref_ad, label):
    t = time.time(); model = aj.train_reference(ref_ad, train_label_name=label, print_cost=False)
    tr = time.time() - t
    t = time.time(); frame, _ = model.predict_frame(qry); inf = time.time() - t
    return frame["celltype"].to_numpy(), tr, inf, model


# flat baseline
pred, tr, inf, _ = fit_predict(ref, LABEL); score("flat-full", pred, tr, inf, ref.n_obs)

# random-K vs coreset-K
rng = np.random.default_rng(0)
ridx = np.concatenate([rng.choice(np.where(labels == c)[0], min(K, (labels == c).sum()), replace=False)
                       for c in types])
pred, tr, inf, _ = fit_predict(ref[np.sort(ridx)].copy(), LABEL); score(f"random-{K}", pred, tr, inf, len(ridx))


def coreset_idx(emb, labels, k, seed=0):
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= k:
            keep += list(idx); continue
        km = KMeans(k, n_init=3, random_state=seed).fit(emb[idx])
        for ci in range(k):
            mem = idx[km.labels_ == ci]
            if len(mem):
                keep.append(int(mem[((emb[mem] - km.cluster_centers_[ci]) ** 2).sum(1).argmin()]))
    return np.array(sorted(set(keep)))

cidx = coreset_idx(EMB, labels, K)
pred, tr, inf, _ = fit_predict(ref[cidx].copy(), LABEL); score(f"coreset-{K}", pred, tr, inf, len(cidx))


def run_hierarchy(name, grp):
    ref.obs["coarse"] = [str(grp[t]) for t in labels]
    t0 = time.time()
    cm = aj.train_reference(ref, train_label_name="coarse", print_cost=False)
    fine = {}
    for g in np.unique(list(grp.values())):
        sub = ref[ref.obs["coarse"] == str(g)]; gt = sub.obs[LABEL].unique()
        fine[str(g)] = (str(gt[0]) if len(gt) == 1
                        else aj.train_reference(sub.copy(), train_label_name=LABEL, print_cost=False))
    htrain = time.time() - t0
    t0 = time.time()
    cpred = cm.predict_frame(qry)[0]["celltype"].to_numpy()
    hpred = np.empty(qry.n_obs, dtype=object)
    for g in np.unique(cpred):
        mask = cpred == g; fm = fine.get(str(g))
        hpred[mask] = fm if isinstance(fm, str) else fm.predict_frame(qry[mask])[0]["celltype"].to_numpy()
    score(name, hpred, htrain, time.time() - t0, ref.n_obs)
    return cm, fine, grp, htrain


# hierarchy: scPRINT-embedding groups, random control, biological control
cent = np.vstack([EMB[labels == t].mean(0) for t in types])
grp_scprint = dict(zip(types, fcluster(linkage(cent, "ward"), NG, criterion="maxclust")))
grp_random = dict(zip(types, np.random.default_rng(0).integers(1, NG + 1, size=len(types))))
cm, fine_models, grp, hier_train_s = run_hierarchy(f"hierarchy-scprint(G{NG})", grp_scprint)
run_hierarchy(f"hierarchy-random(G{NG})", grp_random)
if BIO and BIO in ref.obs:
    type_bio = dict(zip(labels, ref.obs[BIO].astype(str)))  # fine type -> biological group
    run_hierarchy(f"hierarchy-bio({BIO})", {t: type_bio[t] for t in types})

# scprint-alone + scoping + routing (only when CL ids available)
if has_cl:
    spred_cl = qry.obs["scprint_pred_cl"].astype(str).to_numpy()
    msp = metrics.compute(truth_cl, spred_cl, ontology=anc, truth_cl=truth_cl, pred_cl=spred_cl)
    rows.append({"method": "scprint-alone", "accuracy": round(msp["accuracy"], 3),
                 "macro_f1": round(msp["macro_f1"], 3),
                 "ontology": round(msp.get("ontology_concordance", float("nan")), 3),
                 "train_s": 0.0, "infer_s": float("nan"), "n_ref": 0})
    print(f"  scprint-alone          acc {msp['accuracy']:.3f} onto {msp.get('ontology_concordance',0):.3f}", flush=True)

    def related(a, b):
        return a == b or a in anc.get(b, ()) or b in anc.get(a, ())
    sp_cls = set(spred_cl)
    scope = [t for t in types if any(related(name2cl.get(t, ""), pc) for pc in sp_cls)]
    if len(scope) >= 2:
        pred, tr, inf, _ = fit_predict(ref[ref.obs[LABEL].isin(scope)].copy(), LABEL)
        score(f"scoping({len(scope)}/{len(types)}t)", pred, tr, inf, int(ref.obs[LABEL].isin(scope).sum()))

    def sp_group(pcl):
        for t in types:
            if related(name2cl.get(t, ""), pcl):
                return str(grp[t])
        return None
    t0 = time.time()
    g_of = np.array([sp_group(pc) for pc in spred_cl], dtype=object)
    un = np.array([g is None for g in g_of])
    if un.any():
        g_of[un] = cm.predict_frame(qry[un])[0]["celltype"].to_numpy()
    rpred = np.empty(qry.n_obs, dtype=object)
    for g in np.unique(g_of):
        mask = g_of == g; fm = fine_models.get(str(g))
        rpred[mask] = ("unknown" if fm is None else fm if isinstance(fm, str)
                       else fm.predict_frame(qry[mask])[0]["celltype"].to_numpy())
    score(f"routing(G{NG})", rpred, hier_train_s, time.time() - t0, ref.n_obs)

print()
df = pd.DataFrame(rows); df.to_csv(RESULTS, index=False)
print(df.to_string(index=False)); print("TWO_STAGE_DONE", flush=True)
