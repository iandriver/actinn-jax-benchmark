"""Build a class->tissue map from a metadata-only census query and bake it into a
bundled model manifest. Pan-tissue classes get the sentinel ["*"] (always allowed);
organ-specific classes get the list of tissue_general categories where they
meaningfully occur; classes with no census match are omitted (treated as always
allowed at filter time). Keyed by cell_type name, with ontology-ID fallback.
"""
import json, sys, os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cellxgene_census
import actinn_jax as aj

REFDIR = sys.argv[1] if len(sys.argv)>1 else "actinn_jax/references/broad_human_v1"
SPECIFIC_FLOOR = 0.05   # a tissue is "meaningful" for a type at >=5% of its cells
BREADTH_FLOOR  = 0.01   # count tissues at >=1% to decide pan-tissue
PAN_MIN        = 8      # >= this many tissues @1% -> pan-tissue (always allowed)

man = json.load(open(os.path.join(REFDIR, "manifest.json")))
classes = list(man["classes"]); class_to_cl = man.get("class_to_cl", {})
print(f"{REFDIR}: {len(classes)} classes", flush=True)

t=time.time()
with cellxgene_census.open_soma(census_version="2025-11-08") as census:
    obs = cellxgene_census.get_obs(
        census, "homo_sapiens",
        value_filter="is_primary_data == True and disease == 'normal'",
        column_names=["cell_type", "cell_type_ontology_term_id", "tissue_general"])
print(f"pulled {len(obs):,} cells in {time.time()-t:.0f}s", flush=True)

def tissue_sets(keycol):
    g = obs.groupby([keycol, "tissue_general"], observed=True).size().rename("n").reset_index()
    tot = g.groupby(keycol)["n"].transform("sum"); g["frac"]=g["n"]/tot
    out={}
    for k, gg in g.groupby(keycol, observed=True):
        breadth = sorted(gg.loc[gg.frac>=BREADTH_FLOOR, "tissue_general"])
        specific = sorted(gg.loc[gg.frac>=SPECIFIC_FLOOR, "tissue_general"])
        out[k] = ["*"] if len(breadth) >= PAN_MIN else specific
    return out

by_name = tissue_sets("cell_type")
by_cl   = tissue_sets("cell_type_ontology_term_id")

class_to_tissue={}; n_name=n_cl=n_miss=0
for c in classes:
    if c in by_name:            class_to_tissue[c]=by_name[c]; n_name+=1
    elif class_to_cl.get(c) in by_cl:
        class_to_tissue[c]=by_cl[class_to_cl[c]]; n_cl+=1
    else:                       n_miss+=1
print(f"mapped by name={n_name}, by ontology={n_cl}, unmapped(always-allowed)={n_miss}", flush=True)
pan=sum(1 for v in class_to_tissue.values() if v==["*"])
print(f"pan-tissue(always-allowed)={pan}; organ-specific={len(class_to_tissue)-pan}", flush=True)

man["class_to_tissue"]=class_to_tissue
man["tissue_map_meta"]={"census_version":"2025-11-08","tissue_field":"tissue_general",
    "specific_floor":SPECIFIC_FLOOR,"breadth_floor":BREADTH_FLOOR,"pan_min":PAN_MIN}
json.dump(man, open(os.path.join(REFDIR,"manifest.json"),"w"))
print("wrote", os.path.join(REFDIR,"manifest.json"), flush=True)

# --- verification: what does a liver sample keep vs a lung sample? ---
def allowed(tissue):
    T=tissue
    keep=[c for c in classes if (c not in class_to_tissue) or class_to_tissue[c]==["*"]
          or T in class_to_tissue[c]]
    return keep
for T in ["liver","lung","blood","heart"]:
    k=allowed(T)
    print(f"tissue={T}: {len(k)}/{len(classes)} classes allowed", flush=True)
# spot checks
def st(name):
    for c in classes:
        if name.lower() in c.lower(): return f"{c!r}={class_to_tissue.get(c,'UNMAPPED')}"
    return f"(no class matching {name})"
for q in ["hepatocyte","pneumocyte","cardiac muscle","T cell","macrophage","kidney"]:
    print("  ", st(q), flush=True)
