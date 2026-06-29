"""Reproduce the scPRINT-discovered coarse->fine hierarchy from CACHED embeddings.

No GPU, no raw counts needed: loads a committed embedding artifact
(data/embeddings/<dataset>.npz), clusters cell-type centroids in scPRINT space into
coarse groups, prints the hierarchy, and — where a biological grouping is available —
reports how well scPRINT's structure recovers it (Adjusted Rand Index over cell types).

    python benchmark/explore/discover_hierarchy.py lung
    python benchmark/explore/discover_hierarchy.py blood_gut --bio Lineage
    python benchmark/explore/discover_hierarchy.py tabula_sapiens --bio organ
"""
import argparse, os, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

LABELS = {"lung": "cell_type", "blood_gut": "Final_labels", "tabula_sapiens": "cell_type"}
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(LABELS))
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--bio", default=None, help="biological obs column to compare against")
    a = ap.parse_args()

    z = np.load(os.path.join(HERE, "data/embeddings", a.dataset + ".npz"), allow_pickle=True)
    lab = LABELS[a.dataset]
    emb = z["ref_emb"]
    labels = z[f"ref_{lab}"].astype(str)
    types = np.unique(labels)
    cent = np.vstack([emb[labels == t].mean(0) for t in types])
    grp = dict(zip(types, fcluster(linkage(cent, "ward"), a.groups, criterion="maxclust")))

    print(f"{a.dataset}: {len(types)} cell types -> {a.groups} scPRINT-discovered groups "
          f"({emb.shape[1]}-d embeddings, {len(labels)} ref cells)\n")
    for g in sorted(set(grp.values())):
        members = [t for t in types if grp[t] == g]
        print(f"  group {g} ({len(members)}): " + ", ".join(m[:32] for m in members[:6])
              + (" ..." if len(members) > 6 else ""))

    if a.bio and f"ref_{a.bio}" in z.files:
        bio = z[f"ref_{a.bio}"].astype(str)
        type_bio = {t: bio[labels == t][0] for t in types}  # modal-ish biological group per type
        gs = [grp[t] for t in types]
        bs = [type_bio[t] for t in types]
        ari = adjusted_rand_score(bs, gs)
        print(f"\n  scPRINT grouping vs biological '{a.bio}' "
              f"({len(set(bs))} biological groups): ARI = {ari:.3f}")


if __name__ == "__main__":
    main()
