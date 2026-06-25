# Reference-based cell-type annotation: methods survey

A cited landscape of supervised / reference-based single-cell annotation methods,
assembled to inform this benchmark's design. Claims below were extracted from primary
sources and adversarially verified (multi-vote); a few widely-repeated claims **did
not survive verification** and are listed in [Refuted](#refuted-do-not-cite) so we
don't propagate them.

## Bottom line

No single method family dominates every axis:

- **Classical task-specific classifiers lead on accuracy and robustness.** The
  general-purpose **SVM** is the overall accuracy leader across the largest head-to-head
  benchmark ([Abdelaal et al. 2019, *Genome Biology*](https://pubmed.ncbi.nlm.nih.gov/31500660/) —
  22 classifiers × 27 datasets), with most classifiers degrading on complex datasets
  with overlapping classes or deep/fine-grained labels.
- **Foundation models can tie, but not consistently beat, task-specific methods.**
  In the 18-method immune benchmark ([Huang et al. 2024, *Brief. Bioinform.* bbae392](https://academic.oup.com/bib/article/25/5/bbae392/7730135)),
  **SVM, scBERT, scDeepSort** rank top (accuracy up to ~0.95), but fine-grained T-cell
  subtype accuracy stays low. Recent work finds the foundation-model advantage is
  **"biologically stratified"** — strong on recognition (annotation), weaker on
  quantification ([CellBench-LS 2026](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full);
  [Liu et al. 2026, *Adv. Sci.*](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490)).
- **Consensus / probabilistic methods add calibrated uncertainty**, not necessarily
  higher accuracy (popV).

**Implication for this benchmark:** a fast, GPU-free **actinn-jax** belongs alongside
**SVM and CellTypist** as a competitive task-specific baseline — foundation models
offer no guaranteed accuracy win to justify their GPU dependence on a CPU-first setup.

## Verified findings by category

### Classical classifiers
- **SVM** — overall accuracy leader ([Abdelaal 2019](https://pubmed.ncbi.nlm.nih.gov/31500660/));
  **SVM_rejection** adds unseen-type rejection. Top-3 in [Huang 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135).
- **CellTypist** — regularized (L2) logistic regression, SGD-trainable for >500k cells;
  fast, CPU-friendly, Python ([Domínguez Conde et al. 2022, *Science*](https://www.science.org/doi/10.1126/science.abl5197)).
- **scmap-cell/cluster, CHETAH, Cell BLAST** — provide rejection of unknown types
  ([Huang 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135)).
- **mtANN** — supervised neural-network ensemble integrating multiple references via 8
  gene-selection methods; designed to annotate *and* flag unseen types
  ([Xiong et al. 2023, *PLoS Comput. Biol.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10335708/)).
- ACTINN (and **actinn-jax**) sit here as neural-net task-specific classifiers.

### Deep / probabilistic reference mapping & consensus
- **popV** — consensus of **8** methods (RF, SVM, scANVI, OnClass, CellTypist, and k-NN
  after scVI / BBKNN / Scanorama batch correction) by majority vote
  ([Ergen et al. 2024, *Nat. Genet.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11631762/)).
  Its **per-cell consensus score (0–8) is calibrated uncertainty**: score 8 → 98% exact
  matches; ≤3 → <50% accuracy; low scores flag novel/query-specific types for manual
  review. Goal is calibrated uncertainty, **not** higher accuracy than single predictors.

### Transformer / foundation models
- **scBERT, scDeepSort** — competitive with top classifiers (~0.95) in
  [Huang 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135); both support rejection.
- **Geneformer** — masked-pretraining transformer (15% genes masked) on ~30M cells;
  **GPU required for efficient use** ([model card](https://huggingface.co/ctheodoris/Geneformer)).
- Across tasks, foundation models **do not consistently outperform** task-specific
  methods ([Liu et al. 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490);
  [CellBench-LS](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full)).

### LLM-based annotation
- **CASSIA** — reference-free, marker-driven multi-agent LLM (5 agents) with a built-in
  0–100 cell-ontology-aware confidence score and majority voting; reports gains over
  *other reference-free methods* over 970 cell types
  ([*Nat. Commun.* 2025](https://www.nature.com/articles/s41467-025-67084-x)).
  **Caveat:** author-reported, scoped to reference-free methods, not independently
  benchmarked against supervised classifiers.

## Rejection / uncertainty support

Methods with explicit unseen-type rejection: **SVM_rejection, CHETAH, scmap-cell,
scmap-cluster, Cell BLAST, scBERT, scDeepSort** ([Huang 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135)).
**popV** and **CASSIA** provide calibrated confidence scores. Even so, simultaneously
annotating known + unknown types is unsolved — with a type held out of training, the
best rejection methods scored <0.5 on the unknown type.

## Refuted (do NOT cite)

These claims failed adversarial verification and should not be used as established:
- GPT-4 matched expert annotation in ">75% of cell types" across 10 datasets/5 species — **refuted (1–2)**.
- GPT-4 "surpasses existing automatic annotation algorithms" — **refuted (1–2)**.
- GPT-4 distinguished pure vs mixed types at 93% / known vs unknown at 99% — **refuted (1–2)**.
- popV combines "ten classifiers incl. XGBoost/Harmony" — **refuted (0–3)** (it is 8).

(Source for the GPT-4 claims: [Hou & Ji 2024, *Nat. Methods*](https://www.nature.com/articles/s41592-024-02235-4) — the paper exists; these specific quantitative claims did not survive verification.)

## Caveats

- Abdelaal 2019 predates foundation/LLM methods; Huang 2024's ~0.95 is an immune-subtype
  ceiling where fine T-cell accuracy stayed low — rankings may not transfer across tissues.
- popV calibration and CASSIA's outperformance are **authors' own evaluations**, not
  independent third-party benchmarks.
- Several requested methods (SingleR, scPred, Garnett, scID, scANVI, scArches, Symphony,
  Azimuth, scGPT, TOSICA, GPTCelltype) appear only incidentally here — they are in the
  benchmark plan but were not independently verified in this pass.

## Open questions this benchmark should answer

1. Where does CPU-first actinn-jax land on the **accuracy × runtime Pareto** vs SVM,
   CellTypist, SingleR on Apple Silicon? (No existing claim benchmarks ACTINN's
   accuracy or speed directly.)
2. Verified accuracy / rejection / runtime for the requested-but-unverified methods
   (SingleR, scPred, scANVI, scArches, Symphony, Azimuth, scGPT, TOSICA, …).
3. Do foundation models' few-shot advantages hold under the **abundant-label** regime
   typical of reference atlases, where task-specific methods are actually deployed?

## Key sources
- [Abdelaal et al. 2019, *Genome Biology*](https://pubmed.ncbi.nlm.nih.gov/31500660/) — 22-classifier benchmark
- [Huang et al. 2024, *Brief. Bioinform.* bbae392](https://academic.oup.com/bib/article/25/5/bbae392/7730135) — 18-method immune benchmark
- [Ergen et al. 2024, *Nature Genetics* (popV)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11631762/)
- [Domínguez Conde et al. 2022, *Science* (CellTypist)](https://www.science.org/doi/10.1126/science.abl5197)
- [CellBench-LS 2026, bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full) · [Liu et al. 2026, *Adv. Sci.*](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490)
- [Geneformer model card](https://huggingface.co/ctheodoris/Geneformer) · [mtANN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10335708/) · [CASSIA](https://www.nature.com/articles/s41467-025-67084-x)
- [single-cell best practices: annotation](https://www.sc-best-practices.org/cellular_structure/annotation.html)

---
*Generated via a fan-out, adversarially-verified deep-research pass (5 angles, 20
sources fetched, 93 claims extracted, 25 verified, 21 confirmed / 4 killed).*
