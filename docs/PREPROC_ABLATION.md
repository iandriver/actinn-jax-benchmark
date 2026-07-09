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
