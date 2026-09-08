# Reference-based cell-type annotation: methods survey

A cited survey of supervised and reference-based single-cell annotation methods,
assembled to inform this benchmark's design. Claims below were extracted from primary
sources and checked against them. A few widely-repeated claims did not survive that check
and are listed under [Refuted](#refuted-do-not-cite) so they are not propagated here.

## Bottom line

No single method family dominates every axis:

- Classical task-specific classifiers lead on accuracy and robustness. The general-purpose
  SVM is the overall accuracy leader across the largest head-to-head benchmark
  ([Abdelaal et al. 2019, *Genome Biology*](https://pubmed.ncbi.nlm.nih.gov/31500660/),
  22 classifiers over 27 datasets), with most classifiers degrading on complex datasets
  that have overlapping classes or deep, fine-grained labels.
- Foundation models can tie task-specific methods but do not consistently beat them. In
  the 18-method immune benchmark ([Fu et al. 2024, *Brief. Bioinform.* bbae392](https://academic.oup.com/bib/article/25/5/bbae392/7730135)),
  SVM, scBERT and scDeepSort rank top at accuracy up to about 0.95, while fine-grained
  T-cell subtype accuracy stays low. Recent work finds the foundation-model advantage is
  biologically stratified, being strong on recognition and weaker on quantification
  ([CellBench-LS 2026](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full);
  [Liu et al. 2026, *Adv. Sci.*](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490)).
- Consensus and probabilistic methods add calibrated uncertainty rather than higher
  accuracy, as popV does.

For this benchmark, that puts a fast, GPU-free actinn-jax alongside SVM and CellTypist as a
competitive task-specific baseline, since foundation models offer no guaranteed accuracy
win to justify their GPU dependence on a CPU-first setup.

## Verified findings by category

### Classical classifiers
- SVM is the overall accuracy leader ([Abdelaal 2019](https://pubmed.ncbi.nlm.nih.gov/31500660/)),
  and SVM_rejection adds unseen-type rejection. Top three in [Fu 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135).
- CellTypist is L2-regularized logistic regression, SGD-trainable past 500k cells, fast,
  CPU-friendly and in Python ([Domínguez Conde et al. 2022, *Science*](https://www.science.org/doi/10.1126/science.abl5197)).
- scmap-cell, scmap-cluster, CHETAH and Cell BLAST provide rejection of unknown types
  ([Fu 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135)).
- mtANN is a supervised neural-network ensemble integrating multiple references via 8
  gene-selection methods, designed to annotate and to flag unseen types
  ([Xiong et al. 2023, *PLoS Comput. Biol.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10335708/)).
- ACTINN and actinn-jax sit here as neural-net task-specific classifiers.

### Deep / probabilistic reference mapping & consensus
- popV is a consensus of 8 methods (RF, SVM, scANVI, OnClass, CellTypist, and k-NN after
  scVI, BBKNN or Scanorama batch correction) by majority vote
  ([Ergen et al. 2024, *Nat. Genet.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11631762/)).
  Its per-cell consensus score, from 0 to 8, is calibrated uncertainty: a score of 8 gives
  98% exact matches, 3 or below gives under 50% accuracy, and low scores flag novel or
  query-specific types for manual review. The goal is calibrated uncertainty rather than
  higher accuracy than single predictors.

### Transformer / foundation models
- scBERT and scDeepSort are competitive with top classifiers at about 0.95 in
  [Fu 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135), and both support rejection.
- Geneformer is a masked-pretraining transformer, masking 15% of genes, trained on about
  30M cells, and needs a GPU for efficient use ([model card](https://huggingface.co/ctheodoris/Geneformer)).
- Across tasks, foundation models do not consistently outperform task-specific methods
  ([Liu et al. 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490);
  [CellBench-LS](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full)).

### LLM-based annotation
- CASSIA is a reference-free, marker-driven multi-agent LLM using 5 agents, with a built-in
  0 to 100 cell-ontology-aware confidence score and majority voting. It reports gains over
  other reference-free methods across 970 cell types
  ([*Nat. Commun.* 2025](https://www.nature.com/articles/s41467-025-67084-x)). Those results
  are author-reported, scoped to reference-free methods, and not independently benchmarked
  against supervised classifiers.

## Rejection / uncertainty support

Methods with explicit unseen-type rejection: SVM_rejection, CHETAH, scmap-cell,
scmap-cluster, Cell BLAST, scBERT and scDeepSort ([Fu 2024](https://academic.oup.com/bib/article/25/5/bbae392/7730135)).
popV and CASSIA provide calibrated confidence scores. Even so, annotating known and unknown
types at the same time is unsolved: with a type held out of training, the best rejection
methods scored below 0.5 on the unknown type.

## Refuted (do NOT cite)

These claims failed adversarial verification and should not be used as established:
- GPT-4 matched expert annotation in "more than 75% of cell types" across 10 datasets and 5
  species. Refuted, 1 vote to 2.
- GPT-4 "surpasses existing automatic annotation algorithms". Refuted, 1 to 2.
- GPT-4 distinguished pure from mixed types at 93% and known from unknown at 99%. Refuted,
  1 to 2.
- popV combines "ten classifiers including XGBoost and Harmony". Refuted, 0 to 3; it is 8.

Source for the GPT-4 claims: [Hou & Ji 2024, *Nat. Methods*](https://www.nature.com/articles/s41592-024-02235-4). The paper exists; these specific quantitative claims did not survive checking.

## Caveats

- Abdelaal 2019 predates foundation and LLM methods, and Fu 2024's 0.95 is an immune-subtype
  ceiling where fine T-cell accuracy stayed low, so rankings may not transfer across tissues.
- popV's calibration and CASSIA's outperformance are the authors' own evaluations rather
  than independent third-party benchmarks.
- Several methods (SingleR, scPred, Garnett, scID, scANVI, scArches, Symphony, Azimuth,
  scGPT, TOSICA, GPTCelltype) appear only incidentally here. They are in the benchmark plan
  but were not independently checked in this pass.

## Open questions this benchmark should answer

1. Where does CPU-first actinn-jax land on the accuracy against runtime frontier, next to
   SVM, CellTypist and SingleR on Apple Silicon? No existing claim benchmarks ACTINN's
   accuracy or speed directly.
2. Verified accuracy / rejection / runtime for the requested-but-unverified methods
   (SingleR, scPred, scANVI, scArches, Symphony, Azimuth, scGPT, TOSICA, …).
3. Do foundation models' few-shot advantages hold under the abundant-label regime typical
   of reference atlases, where task-specific methods are actually deployed?

## Key sources
- [Abdelaal et al. 2019, *Genome Biology*](https://pubmed.ncbi.nlm.nih.gov/31500660/), the 22-classifier benchmark
- [Fu et al. 2024, *Brief. Bioinform.* bbae392](https://academic.oup.com/bib/article/25/5/bbae392/7730135), the 18-method immune benchmark
- [Ergen et al. 2024, *Nature Genetics* (popV)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11631762/)
- [Domínguez Conde et al. 2022, *Science* (CellTypist)](https://www.science.org/doi/10.1126/science.abl5197)
- [CellBench-LS 2026, bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.01.714123v1.full) · [Liu et al. 2026, *Adv. Sci.*](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202514490)
- [Geneformer model card](https://huggingface.co/ctheodoris/Geneformer) · [mtANN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10335708/) · [CASSIA](https://www.nature.com/articles/s41467-025-67084-x)
- [single-cell best practices: annotation](https://www.sc-best-practices.org/cellular_structure/annotation.html)

---
*Claims here were extracted from primary sources and checked against them: 93 claims
extracted, 25 checked in depth, 21 confirmed and 4 rejected. The rejected ones are listed
under Refuted.*
