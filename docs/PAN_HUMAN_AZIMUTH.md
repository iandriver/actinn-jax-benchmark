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
| lung_intra | actinn-jax | 0.894 | **0.917** | 12.8 | 0.37 | 2092 |
| lung_intra | Pan-human Azimuth | 0.408† | **0.700** | 4.9‡ | 3.07 | 2025 |
| liver_intra | actinn-jax | 0.802 | **0.846** | 8.3 | 0.27 | 1660 |
| liver_intra | Pan-human Azimuth | 0.227† | **0.521** | 4.1‡ | 2.22 | 1652 |
| liver_cross | actinn-jax | 0.686 | **0.731** | 10.6 | 0.36 | 2311 |
| liver_cross | Pan-human Azimuth | 0.153† | **0.408** | 4.2‡ | 3.36 | 2105 |

† Exact-CL-id match, **not** the same quantity as actinn-jax's exact-label-string match —
Pan-human Azimuth predicts into its own typology. Only the **ontology** column compares.
‡ No training happens; "fit" is TensorFlow import plus weight loading.

**Peak memory is a tie** — 1.65–2.1 GB for both, on every dataset. Whatever separates these
two methods, it is not footprint.

## Cost as the query grows

[`results_panhuman_cost.csv`](results_panhuman_cost.csv), from
[`panhuman_cost.py`](../benchmark/explore/panhuman_cost.py): one lung reference of 4,000
cells, query size swept, each (method, size) in a fresh subprocess under the harness's own
`ResourceMonitor`.

| n query | actinn-jax predict | cells/s | PHA predict | cells/s |
|---:|---:|---:|---:|---:|
| 1,000 | 0.25 s | 3,978 | 2.15 s | 466 |
| 4,000 | 0.39 s | 10,305 | 3.72 s | 1,076 |
| 16,000 | 1.10 s | 14,610 | 11.07 s | 1,445 |
| 40,000 | 2.86 s | 13,997 | 25.59 s | 1,563 |

- **Their published throughput holds.** The paper quotes ~1,000 cells/s on a MacBook Air
  (M2, 16 GB); we measure **1,076–1,563 cells/s** once the query reaches 4k cells, so the
  claim is accurate and, at scale, slightly conservative. Below ~1k cells fixed overhead
  dominates and throughput falls to 466 cells/s — the quoted figure describes the amortized
  regime, which is the regime that matters. Extrapolating 1M cells at 1,563 cells/s gives
  ~11 minutes, the same order as their reported ~1,100 s.
- **actinn-jax predicts ~9× faster** at 40k cells (2.86 s vs 25.59 s). Both are linear in
  query size; neither degrades.
- **Peak memory in that sweep is not a clean per-method number** — each worker loads the
  full 65k-cell lung atlas (596 MB on disk, ~4 GB resident) before subsetting, and that
  baseline is identical for both methods. Use the per-dataset table above for footprint.

One adapter note that affects these numbers: the high-level `AzimuthNN` class takes the
query in its constructor and so reloads the weights on **every** call. An earlier version of
this adapter paid that per prediction, inflating predict time by ~0.4–1.7 s. It now drives
the low-level `AzimuthNN_base`, loading once in `fit` — verified to yield labels identical to
the high-level path, and the accuracy columns above are unchanged from before the fix.

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

## Pan-human Azimuth as tier 1: the hand-off does not stack

Script: [`panhuman_tier1_refine.py`](../benchmark/explore/panhuman_tier1_refine.py) (tier-1
predictions dumped by [`panhuman_tier1_dump.py`](../benchmark/explore/panhuman_tier1_dump.py),
since Keras and JAX cannot share a process). Numbers in
[`results_panhuman_tier1.csv`](results_panhuman_tier1.csv). Leakage-free cross-study liver
split: reference = 6 HLiCA studies, query = a withheld study, tier 2 trained only on the
reference. Ontology-aware concordance throughout.

| arm | ontology |
|---|---:|
| tier 1 only — `broad_human_v1` (ours) | 0.338 |
| tier 1 only — Pan-human Azimuth | 0.380 |
| **tier 2 only — actinn-jax on the liver reference** | **0.731** |
| tier 1 scopes tier 2 — PHA's coarse call masks tier-2 classes | 0.708 |
| oracle scope — a *perfect* coarse call masks tier-2 classes | 0.759 |

Three results, one of them against us:

1. **Pan-human Azimuth is the better broad model** — 0.380 vs 0.338 for our shipped
   `broad_human_v1` on the same cells. Modest, but it is better resourced and it shows.
2. **The hand-off is worth a lot, and that reproduces.** Either broad model scores ~0.34–0.38;
   the focused reference reaches **0.731**. Switching to a tissue-specific reference roughly
   doubles concordance, which is the §3.5 claim, here on a leakage-free split.
3. **But the two stages do not stack.** Using tier 1's coarse call to narrow tier 2's classes
   — the zero-retrain masking actinn-jax ships — makes things **worse**: 0.731 → **0.708**.
   Pan-human Azimuth's coarse call agrees with the truth's coarse lineage on 85.8% of cells,
   and the 14% it gets wrong cost more than the 86% it gets right can gain, because a wrong
   mask removes the correct class from consideration entirely.

The ceiling arm is the informative one: even a **perfect** coarse call buys only
**+2.8 points** (0.731 → 0.759). Once tier 2 is trained on a reference that covers the
tissue, there is almost nothing left for a broad model to contribute to *accuracy*.

**What this means for the paper.** §3.5 already says the broad model's job "is to route to
[the focused reference], not to be right about subtypes itself" — this measures that and
finds it is the whole story. The value of the two-tier workflow is **switching**: knowing
which focused reference to load, and covering cells no focused reference claims. It is not
fusion, and we should not imply the tiers combine to beat either alone. Substituting a
stronger tier 1 improves the broad pass and changes nothing downstream.

**Caveats.** One split, one tissue, one fusion mechanism (a hard mask by coarse CL lineage).
A softer combination — probability blending, or routing only low-confidence cells — might
recover the +2.8, but it cannot exceed it. The tier-1-only arms are also still scored against
a liver-specialist vocabulary neither broad model was built for.

## Still open

- **Routing across tissues.** Everything above assumes we already know to load the liver
  reference. The workflow's real tier-1 job is choosing *which* focused reference to load on
  a query of unknown provenance, and neither the paper nor this doc measures routing accuracy
  directly. Pan-human Azimuth's coarse call was right about lineage on 85.8% of cells, which
  is a lower bound on how well it would route.
- **Cells no focused reference claims.** The other stated tier-1 job is coverage — flagging
  cells outside the focused reference's scope. Pan-human Azimuth's trained `Unassigned` class
  is a better mechanism for this than our threshold, and comparing the two is untested.
- **Softer fusion.** The masking here is a hard lineage filter. Probability blending or
  routing only low-confidence cells might recover part of the +2.8-point oracle headroom,
  but cannot exceed it.
- **Other tissues.** One split, one tissue. Whether the +2.8 ceiling is liver-specific or
  general is unknown; lung (46 types, CL-annotated) is the obvious second test.
