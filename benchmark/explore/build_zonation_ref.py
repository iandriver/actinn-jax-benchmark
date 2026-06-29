"""Build a human-liver hepatocyte ZONATION reference from GSE158723 (8x 10x samples).

No zonation labels ship with the data, so we derive them the standard (Halpern) way:
identify hepatocytes (ALB-high), score each on a central-vs-portal landmark-gene axis,
and bin into portal / mid / central tertiles. Saves a reference h5ad with raw counts,
Ensembl var_names (+ symbols), and obs['zone'].
"""
import glob, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, scipy.io as sio, scipy.sparse as sp

SRC = "/tmp/gse158723"
CENTRAL = ["GLUL", "CYP2E1", "CYP1A2", "CYP3A4", "ADH1B", "ADH4", "OAT", "CYP2C8"]
PORTAL = ["ASS1", "ASL", "SDS", "HAL", "CPS1", "PCK1", "ARG1", "AGT", "GLS2", "SLC7A2"]

samples = sorted({f.rsplit("_", 1)[0] for f in glob.glob(f"{SRC}/*_matrix.mtx.gz")})
parts = []
for s in samples:
    M = sio.mmread(f"{s}_matrix.mtx.gz").T.tocsr().astype(np.float32)  # cells x genes
    genes = pd.read_csv(f"{s}_genes.tsv.gz", sep="\t", header=None)
    barc = pd.read_csv(f"{s}_barcodes.tsv.gz", header=None)[0].values
    a = ad.AnnData(X=M)
    a.var_names = genes[0].astype(str).values            # Ensembl
    a.var["symbol"] = genes[1].astype(str).values
    tag = os.path.basename(s).split("_", 1)[1]
    a.obs_names = [f"{tag}_{b}" for b in barc]
    a.obs["sample"] = tag
    parts.append(a)
adata = ad.concat(parts, join="inner", merge="first")  # merge=first carries var['symbol']
adata.var_names_make_unique()
print(f"loaded {adata.shape} from {len(samples)} samples", flush=True)

# QC
adata.var["mt"] = adata.var["symbol"].str.startswith("MT-").fillna(False).values
sc.pp.filter_cells(adata, min_genes=300)
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
adata = adata[adata.obs.pct_counts_mt < 60].copy()       # hepatocytes are high-mito; lenient
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

sym2idx = {s: i for i, s in enumerate(adata.var["symbol"].values)}
def expr(sym):
    return np.asarray(adata.X[:, sym2idx[sym]].todense()).ravel() if sym in sym2idx else np.zeros(adata.n_obs)

# hepatocytes: very high ALB, low immune (PTPRC)
alb, ptprc = expr("ALB"), expr("PTPRC")
hep_mask = (alb > np.quantile(alb, 0.6)) & (alb > 2.0) & (ptprc < 0.5)
hep = adata[hep_mask].copy()
print(f"hepatocytes: {hep.n_obs} (ALB-high)", flush=True)

# zonation score = central - portal (scanpy score_genes), tertile bins
cen = [g for g in CENTRAL if g in sym2idx]; por = [g for g in PORTAL if g in sym2idx]
hep.var_names = hep.var["symbol"].astype(str).values; hep.var_names_make_unique()
sc.tl.score_genes(hep, cen, score_name="central")
sc.tl.score_genes(hep, por, score_name="portal")
hep.obs["zonation_score"] = hep.obs["central"] - hep.obs["portal"]
q1, q2 = hep.obs["zonation_score"].quantile([1/3, 2/3])
hep.obs["zone"] = pd.cut(hep.obs["zonation_score"], [-np.inf, q1, q2, np.inf],
                         labels=["portal", "mid", "central"]).astype(str)
print("zone counts:", dict(hep.obs["zone"].value_counts()), flush=True)
print(f"central markers used: {cen}\nportal markers used: {por}", flush=True)

# save reference: raw counts in X, Ensembl var_names (+ symbol), zone label
out = ad.AnnData(X=hep.layers["counts"], obs=hep.obs[["sample", "zone", "zonation_score"]].copy())
out.var_names = adata.var_names.values
out.var["symbol"] = adata.var["symbol"].values
out.write_h5ad("/tmp/liver_zonation_ref.h5ad")
print(f"ZONATION_REF_DONE {out.shape}", flush=True)
