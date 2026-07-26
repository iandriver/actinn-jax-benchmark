# Pan-human Azimuth: the closest thing yet to our broad→refined workflow

Sarkar, Li, Molla, … Satija, *Organism-scale annotation with Pan-human Azimuth*,
bioRxiv 2026, [doi:10.64898/2026.07.16.738997](https://doi.org/10.64898/2026.07.16.738997)
(NIH HuBMAP). Package `panhumanpy` ([GitHub](https://github.com/satijalab/panhumanpy),
PyPI), weights on [Zenodo](https://doi.org/10.5281/zenodo.20401417), R access via
`AzimuthAPI`, plus a hosted `CloudAzimuth` inference option.

**Why this matters to us:** it is a *shipped, pretrained, pan-human annotator with a
calibrated abstain that runs on a laptop* — the thing [`PAPER.md`](PAPER.md) described as
having "no counterpart in the baselines". That claim is now wrong and has been corrected.
At the same time the paper is strong independent support for this benchmark's central
argument, so it cuts both ways.

## What it is

A supervised hierarchical classifier over a single organism-wide cell-type tree.

| | |
|---|---|
| training corpus | 27.04M cells, 23 tissues, from Azimuth refs, DISCO, CELLxGENE, GTEx, HuBMAP |
| after QC/curation | **9,665,434** high-confidence cells (≤25k per type per tissue) |
| negative examples | ~145,000 empty droplets + simulated ambient profiles, labeled **"Unassigned"** |
| input | fixed **5,055-gene** panel |
| embedding | 5,055 → 1,024 → 512 → 256 → **128** (~5.87M params) |
| heads | **8 hierarchical levels**, each an MLP on the 128-d embedding *concatenated with compressed softmax outputs of all preceding levels* |
| classes per level | 13 (L1), 91 (L2), 209 (L3), 313 (L4) … **382** (L8) |
| total params | **6,981,993** |
| loss | focal loss, weighted sum across levels |
| calibration | post-hoc entropy-informed temperature scaling per head — **ECE 0.0044** |
| throughput | **~1,000 cells/s on a MacBook Air (M2, 16 GB)**, batch 8,192; linear in dataset size (100k→1M cells) |
| at scale | scBaseCamp 85.9M cells: 13.5 h on an A100, **23.5 h on an Intel Xeon CPU** |

Two design choices are worth stealing-or-at-least-noting. First, each head can emit a
**"blank"**, so the model stops descending when evidence for finer resolution runs out and
still returns a confident call at an intermediate level — graceful degradation built into
the architecture rather than bolted on as a threshold. Second, **quality control is a
trained class, not a cutoff**: empty droplets and simulated ambient RNA are training
examples, so low-quality profiles get "Unassigned" instead of being force-labeled. On
scBaseCamp 11% of profiles were flagged this way, with median UMI 19 vs 4,734 for assigned
cells.

## Where it agrees with us

- **A curated supervised model beats foundation models at annotation.** They benchmark
  against scGPT and SCimilarity via a "DE separability" metric and find both noisier —
  SCimilarity assigned lung epithelial cells to cardiomyocyte labels; scGPT's goblet cells
  lacked canonical secretory markers. This reproduces our §3.7/§3.8 finding on an
  independent corpus, from a group with no stake in our conclusion.
- **Data curation beats scale.** Their stated central theme is that "quality and
  organization of training data can be as important as model architecture or training
  scale", and accuracy **saturates past ~5M training cells**. That is the same saturation
  argument we make from the other direction in §4.
- **Laptop-scale inference is enough.** ~1,000 cells/s on a MacBook Air, and a CPU run of
  85.9M cells within 2× of an A100. The premise of this paper — that this work belongs on
  commodity hardware — is one they share.

## Where it beats our workflow

Honestly: on the broad-annotation tier, in several ways.

1. **The hierarchy is architectural, not procedural.** Eight heads conditioned on every
   preceding level enforce parent/child consistency inside one model. Our broad→refined is
   two separate models handed off in sequence (§3.5), which cannot guarantee the fine call
   is consistent with the broad one.
2. **Abstention is trained, ours is thresholded.** Their "blank" and "Unassigned" outputs
   come from the loss and from negative training data; our abstain is a softmax threshold
   swept post hoc (§3.6). Theirs is the better mechanism, and their ECE of 0.0044 is a
   calibration number we have not measured against.
3. **The label space is harmonized and ontology-mapped.** Every node and leaf has a
   one-to-one Cell Ontology mapping, with a published crosswalk. Our broad reference
   inherits CELLxGENE's vocabulary with its known fragmentation.

## What remains distinct to actinn-jax

The typology is **fixed**: 382 leaves, trained once, with incremental learning listed as
future work. That is the boundary of what Pan-human Azimuth can do, and it is where our
workflow still has something to offer.

- **Refinement against the user's *own* reference.** Tier 2 of our workflow hands off to
  any reference the user builds — the 48-type HLiCA liver reference (cross-study liver
  0.23/0.58 → 0.72/0.86, §3.5). Pan-human Azimuth cannot re-annotate into a label set it
  was not trained on.
- **Sub-leaf resolution.** Hepatocyte zonation (portal→central, ~0.99 within-one-zone) is a
  *state* below any cell-type leaf; no fixed typology contains it.
- **It is a trainer, not only a model.** actinn-jax's product is `train_reference` +
  a cached `ReferenceModel` for whatever labels you have; the shipped broad reference is one
  application of it. `panhumanpy` is inference against a fixed model.
- **Novel-type screening.** Their "Unassigned" is trained on *empty droplets and ambient
  RNA* — a low-quality detector. Flagging a genuine cell type absent from the typology (our
  withheld-ionocyte screen, §3.5) is a different problem, and their negative examples do not
  directly address it. Their per-level "blank" partly covers it by declining to go deeper.

## What this changes in the paper

- The claim that a ready-to-run broad reference has "no counterpart in the baselines" is
  **retracted**; Pan-human Azimuth is exactly that counterpart, and better resourced.
  The distinct piece is refinement into a user-defined label set, not the broad tier.
- It strengthens §3.7/§3.8 and §4: an independent group reaches the same conclusion about
  foundation-model labels and about training-data saturation.

## Measured head-to-head

Adapter: [`benchmark/adapters/panhuman_adapter.py`](../benchmark/adapters/panhuman_adapter.py),
env `.venv-panhuman`, config [`panhuman_compare.yaml`](../configs/panhuman_compare.yaml),
raw numbers in [`results_panhuman_compare.csv`](results_panhuman_compare.csv). One repeat,
same splits as the paper matrix.

| dataset | method | exact | **ontology** | fit (s) | predict (s) | peak mem (MB) |
|---|---|---:|---:|---:|---:|---:|
| lung_intra | actinn-jax | 0.894 | **0.917** | 16.0 | 0.36 | 1804 |
| lung_intra | Pan-human Azimuth | 0.408† | **0.700** | 7.3‡ | 4.8 | 1623 |
| liver_intra | actinn-jax | 0.802 | **0.846** | 8.3 | 0.29 | 1217 |
| liver_intra | Pan-human Azimuth | 0.227† | **0.521** | 4.6‡ | 2.6 | 1685 |
| liver_cross | actinn-jax | 0.686 | **0.731** | 9.8 | 0.38 | 2308 |
| liver_cross | Pan-human Azimuth | 0.153† | **0.408** | 4.4‡ | 3.5 | 2112 |

† Exact-CL-id match, **not** the same quantity as actinn-jax's exact-label-string match —
Pan-human Azimuth predicts into its own typology. Only the **ontology** column compares.
‡ No training happens; "fit" is model loading only.

**Read this comparison carefully — it is not a fair fight, and it is not news.** actinn-jax
is trained on a reference drawn from the same dataset and the same label vocabulary it is
scored against; Pan-human Azimuth has never seen these datasets and answers in a different
vocabulary. A reference-trained model *should* win here. What the numbers do establish:

- **It is far better than zero-shot foundation labels.** On lung it reaches 0.700 ontology
  concordance where scPRINT manages 0.206 (§3.7) — the difference between a curated
  supervised model and a zero-shot foundation head, which is the paper's thesis.
- **The gap is genuine disagreement, not a vocabulary ceiling.** We tested that: for every
  truth type in lung (46/46) and liver (36/36) there is a CL id reachable from Pan-human
  Azimuth's typology by an exact, ancestor or descendant relation, so the ceiling on
  ontology concordance is **1.000** on both. Our initial hypothesis — that HLiCA's
  liver-specialist vocabulary was simply not representable in 382 leaves — is **refuted**.
  The residual misses are *sibling-level*: `IgG memory B cell` → `Plasma cell`,
  `endothelial cell of sinusoid` → `Capillary EC`. Both are close, neither is an ancestor.
- **The right calls are common and the naming differs harmlessly.** `regulatory T cell` →
  `Treg cell`, `non-classical monocyte` → `CD16 monocyte`, `periportal region hepatocyte` →
  `Hepatocyte` all score as exact-match failures and ontology-match successes, which is what
  the metric is for. 253 of the 446 crosswalk entries are `skos:narrowMatch`, mapping their
  label to a *broader* CL term, so ancestor-crediting is doing real work here.
- **Cost:** ~2.6–4.8 s per query vs actinn-jax's ~0.3 s, at comparable memory. Its
  throughput claim (~1,000 cells/s) holds on our hardware.

Sanity checks that rule out adapter error: gene-panel overlap was **~90%** (4,598/5,055) on
Ensembl-keyed atlases via the `feature_name` column, and 1,330/1,332 predictions mapped to a
CL id.

## What has not been tested yet

`panhumanpy` is pip-installable with public weights, so a head-to-head against our shipped
`broad_human_v1` is feasible. One design point decides whether the comparison is meaningful:
**exact-label accuracy would be a vocabulary artifact**, exactly as in the lung_cross case
(§3.1 †) — their 382-leaf typology is not our datasets' vocabulary, and their own paper says
comparisons across label spaces are "inherently challenging". The fair metric is our
**ontology-aware concordance**, and they publish a Cell Ontology crosswalk for every node,
so both sides can be projected into CL and scored. The datasets carrying CL ids
(lung_intra, lung_cross, liver_intra, liver_cross) are the ones to use.

The interesting question is not which is more accurate on a broad call — theirs is trained
on 9.7M curated cells and ours on the census — but **whether the broad→refined hand-off
still buys anything when the broad tier is theirs instead of ours**: run Pan-human Azimuth
for tier 1, then refine with a focused actinn-jax reference, and see whether the liver
0.23/0.58 → 0.72/0.86 gain survives. If it does, the workflow claim stands on a stronger
broad model than our own.
