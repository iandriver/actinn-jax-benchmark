# What actinn-jax can borrow from scANVI+scArches — without a GPU

**Question.** scanvi_scarches tops the OP label_projection leaderboard (0.939). Is there a
transferable lesson for a CPU gene-space MLP around **preprocessing / normalization /
scaling**, short of adopting a VAE?

**Framing.** The win is *not* the VAE: plain `scanvi` scores 0.826 — **below** actinn-jax's
0.837. The jump to 0.939 is `scArches` **reference surgery** = query-side *domain
adaptation*. So the transferable ingredients are (a) count-model normalization and (b)
aligning the query into the reference's feature space. Both have CPU-cheap analogs.

**Method.** `benchmark/explore/preproc_ablation.py` reuses actinn-jax's exact MLP
(`au.train`/`au.predict_proba`, same epochs/seed/layers) and swaps ONLY the feature
pipeline. Reference stats (scaler, residual expectations, gene set) are fit on the
reference and **frozen** for the query — no leakage. Scored on all 6 OP datasets by
accuracy and macro-F1; baseline reproduces the same-hardware AWS numbers exactly
(dkd 0.9471, tabula_sapiens 0.3944).

| lever | what changes vs baseline |
|---|---|
| baseline | `log2(CP10k+1)` + ACTINN percentile expr/CV gene filter (shipped) |
| **E1** | + per-gene **standardization** (reference μ,σ, frozen) |
| E2 | **analytic Pearson residuals** (reference-frozen) as features |
| E3 | genes chosen by **Pearson-residual variance** (HVG) + standardization |
| E4 | **self-training** / pseudo-labels on the query (conf ≥ 0.90), on the E1 base |

## Results (mean over 6 datasets, Δ vs baseline in points)

| lever | mean acc | Δacc | mean macro-F1 | ΔF1 |
|---|---|---|---|---|
| baseline | 0.8368 | — | 0.6561 | — |
| **E1 standardize** | 0.8400 | **+0.32** | 0.6670 | **+1.10** |
| E2 pearson-resid | 0.8424 | +0.56 | 0.6563 | +0.02 |
| E3 pearsonHVG+std | 0.8374 | +0.06 | 0.6701 | +1.40 |
| E4 selftrain (+E1) | 0.8430 | +0.62 | 0.6624 | +0.63 |

Per-dataset accuracy Δ (points): full matrix in `results_preproc_ablation.csv`. Highlights:
- **tabula_sapiens** (hardest; 284-cell test batch, 160 fine types): E1 **+1.76**, E4 **+2.81**;
  E3 **−1.06** (Pearson-HVG drops genes the fine types need).
- **gtex_v9** (batch-shifted): E2 +1.66, E4 +0.90; macro-F1 E1 **+6.6**, E3 **+6.8**.
- **immune_cell_atlas**: E2 **−1.52 / −5.5 F1** — the one clear regression.

## Findings

1. **The humble z-score (E1) is the robust win, not the fancy count model (E2).** E1 gives
   +0.32 acc / +1.1 macro-F1, **never a meaningful loss**, and is the single biggest lift on
   the hardest dataset (tabula_sapiens +1.8). Its macro-F1 gains land exactly on the
   batch-shifted datasets (gtex +6.6, immune +2.5) — i.e. **rare-class recall**. One line,
   zero hyperparameters.
2. **Pearson-residual *features* (E2) are unreliable.** Big on dkd/gtex but −1.5 pts on
   immune and zero net F1 — the highest-variance lever. The "obvious" scVI analog does not
   generalize; reject as a default.
3. **Standardization is the cheap `scArches` analog.** Aligning the query to frozen
   reference μ,σ *is* domain adaptation into the reference space — and it's what actinn-jax
   is missing (features currently go into the MLP unscaled). This is the mechanism behind
   scANVI+scArches's edge, captured on CPU in one transform.
4. **Self-training (E4) helps only under domain shift** (gtex +0.9, TS +2.8) and adds a
   confidence threshold + confirmation-bias risk. Worth offering as an option, not a default.
5. **Pearson-HVG selection (E3)** gives the best mean macro-F1 but regresses TS accuracy —
   not a safe blanket change.

## Recommendation

Adopt **E1 (reference-fit per-gene standardization)** as actinn-jax's default (or opt-in)
scaling step: fit μ,σ on the reference after normalization + gene filtering, store them in
the `ReferenceModel`, apply to every query block. Offer **E4 self-training** as an optional
flag for hard cross-batch cases. Skip E2/E3 as defaults.

## Adopted: `standardize=True` (validated through the package API)

E1 was implemented in actinn-jax as an opt-in `train_reference(..., standardize=True)`
flag (frozen reference μ,σ, applied per-minibatch so atlas training stays memory-bounded;
scaler persisted with the model). Re-run end-to-end through the shipped API on all 6
datasets (`results_standardize_packageapi.csv`):

| dataset | base acc | std acc | Δacc | base F1 | std F1 | ΔF1 |
|---|---|---|---|---|---|---|
| dkd | 0.9471 | 0.9451 | −0.20 | 0.9362 | 0.9300 | −0.62 |
| gtex_v9 | 0.8576 | 0.8613 | +0.37 | 0.3309 | 0.4095 | **+7.86** |
| immune_cell_atlas | 0.8590 | 0.8574 | −0.16 | 0.7342 | 0.7593 | +2.51 |
| mouse_pancreas_atlas | 0.9653 | 0.9659 | +0.06 | 0.7840 | 0.7695 | −1.45 |
| hypomap | 0.9973 | 0.9982 | +0.09 | 0.9802 | 0.9950 | +1.48 |
| tabula_sapiens | 0.3944 | 0.4049 | **+1.05** | 0.1709 | 0.1468 | −2.41 |
| **MEAN** | **0.8368** | **0.8388** | **+0.20** | **0.6561** | **0.6684** | **+1.23** |

Net positive but not free: mean +0.20 acc / +1.23 macro-F1, big F1 wins on the
batch-shifted references and the best single-lever accuracy gain on the hardest dataset
(tabula_sapiens +1.05), against a few small F1 dips and ~24% fit-time overhead. **Shipped
opt-in** (default off): standardization shifts softmax calibration, which the two-stage
refine/abstain thresholds are tuned against — default-on regresses that path. Enable for a
one-stage accuracy win, or re-tune the refine thresholds before combining.

## Negative result: UCE-style protein-embedding featurization (CPU) does not help

UCE (snap-stanford/UCE, Nature 2026) represents each gene by its **ESM2 protein-language-model
embedding** and runs a transformer over the expressed genes. We tested the cheap, CPU-only
version of that idea for actinn-jax: represent each cell as the **expression-weighted average
of its genes' ESM2 embeddings** (5120-d, ~18k embeddable genes; human `gene_symbol_to_embedding_ESM2.pt`
from the UCE Figshare bundle), then train actinn-jax's own MLP on it.
`benchmark/explore/protein_embed_probe.py`; results `docs/results_protein_embed_probe.csv`.

| dataset | protein-ESM2 (weighted mean) | raw 1000 HVG | raw 5000 HVG |
|---|---|---|---|
| immune_cell_atlas | 0.860 / F1 0.730 | 0.857 / 0.759 | 0.891 / 0.816 |
| gtex_v9 | 0.851 / F1 0.321 | 0.861 / 0.410 | 0.891 / 0.455 |
| tabula_sapiens | **0.225 / F1 0.117** | 0.405 / 0.147 | 0.303 / 0.135 |

**It doesn't help and hurts the hard case.** Protein-mean ≈ raw-1000 accuracy on the easy
datasets, is *below* raw-5000 everywhere, and **collapses on tabula_sapiens (0.225 < 0.405)** —
the opposite of the domain-robustness hoped for. The weighted mean pools all genes into a
single centroid in protein space, discarding the per-gene expression detail that separates
fine cell types (gtex rare-class F1 0.321 vs 0.410 confirms the loss). UCE's value is in its
**attention over the gene set**, not a portable averaging trick — capturing it needs the GPU
transformer, not a CPU pooling. The protein-embedding table *does* map genes with zero
name-matching (symbols → proteins), so it remains useful for cross-panel/species gene
alignment — just not as an accuracy lever here.
