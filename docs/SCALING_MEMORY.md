# Atlas-scale behaviour: does anything break, and does the memory gap grow?

In [`SIMPLE_BASELINES.md`](SIMPLE_BASELINES.md) we argued that actinn-jax's remaining
advantage over a tuned linear pipeline was **bounded sparse memory**, and *projected* that
the gap would widen at atlas scale because ANOVA/PCA densify a cells × genes matrix. That
projection was explicitly flagged as untested. This measures it.

Same dataset (Krasnow lung, 65,662 cells, 46 types) at five reference sizes, all four
methods, CPU, 48 GB machine. Config: [`configs/scaling_memory.yaml`](../configs/scaling_memory.yaml).

## Peak memory (MB)

| n_ref | actinn-jax | scTOP | linear-anova-pca | ProtoCloud |
|---:|---:|---:|---:|---:|
| 3,114 | 1,132 | **1,009** | 2,072 | 1,388 |
| 8,109 | 1,955 | **1,599** | 5,308 | 2,014 |
| 19,569 | 3,460 | **2,819** | 9,556 | 3,380 |
| 32,870 | **4,117** | 4,604 | 11,897 | 4,928 |
| 49,264 | **6,122** | 7,480 | 13,156 | 7,138 |

**linear / actinn-jax memory ratio:** 1.83 → 2.71 → 2.76 → 2.89 → **2.15**

## Accuracy

| n_ref | actinn-jax | scTOP | linear-anova-pca | ProtoCloud |
|---:|---:|---:|---:|---:|
| 3,114 | 0.881 | 0.819 | **0.899** | 0.722 |
| 8,109 | 0.894 | 0.828 | 0.898 | **0.932** |
| 19,569 | 0.904 | 0.826 | 0.908 | **0.959** |
| 32,870 | 0.912 | 0.813 | 0.918 | **0.967** |
| 49,264 | 0.936 | 0.846 | 0.939 | **0.976** |

## Fit time (s)

| n_ref | actinn-jax | scTOP | linear-anova-pca | ProtoCloud |
|---:|---:|---:|---:|---:|
| 3,114 | 6.8 | **1.0** | 2.2 | 64.0 |
| 49,264 | 81.5 | **1.8** | 30.2 | 1,541.3 |

## What this changes

**1. Nothing broke, and our memory projection was wrong.** All four methods completed at
49,264 reference cells. We predicted the linear pipeline would need ~32 GB and likely fail;
it used **13.2 GB**. The naive extrapolation from the 8k point over-counted, because
`SelectKBest(k=20000)` caps the dense matrix at cells × 20,000 rather than cells × all
genes. **Measure, don't extrapolate.**

**2. The memory gap does not diverge — it is bounded at ~2–3×.** The linear/actinn-jax
ratio rises to 2.89× mid-range then falls back to **2.15×** at full scale. actinn-jax is
genuinely lighter, consistently, but there is no scaling cliff and no regime here where the
linear pipeline stops being usable. The earlier claim that the gap "grows structurally with
the data" is retracted.

**3. ProtoCloud is the decisive accuracy winner at scale.** It goes from *worst* at 3k
cells (0.722) to *best* at every size above that, reaching **0.976** at 49k vs 0.936–0.939
for actinn-jax and the linear pipeline. This confirms the caveat we put in
[`PROTOCLOUD.md`](PROTOCLOUD.md): our earlier "accuracy is close and splits both ways" was
an artifact of subsampling, and a data-hungry prototype VAE is exactly the model that
should — and does — pull ahead when given a real atlas. Its cost is 1,541 s vs 81.5 s
(19×) on CPU; on the GPU it targets, that gap would largely close while the accuracy
advantage would not change.

**4. scTOP does not scale, on either axis.** Its accuracy is flat-to-noisy across a 16×
increase in data (0.819 → 0.846, dipping to 0.813) — it extracts almost nothing from more
cells, which is what a projection onto class averages should do. And it *loses* its memory
advantage: 0.89× actinn-jax at 3k, **1.22× at 49k**. It is a small-data, low-cardinality
tool, and cheapest-by-far on fit time (1.8 s at 49k), but not an atlas-scale method.

**5. actinn-jax and the linear pipeline are statistically indistinguishable on accuracy at
every scale** (0.936 vs 0.939 at full; the linear pipeline is ahead by 0.003–0.018
throughout), with the linear pipeline ~2.7× faster to fit and actinn-jax ~2.15× lighter.

## Honest bottom line

At atlas scale on this dataset the ranking is: **ProtoCloud for accuracy** (at 19× the CPU
fit cost), **linear-anova-pca for accuracy-per-second**, **actinn-jax for accuracy-per-byte**,
and **scTOP only for speed at small scale**. actinn-jax is not the accuracy leader at any
scale here. Its defensible claim is the memory/throughput profile plus the workflow layer
(cached references, broad→refined, abstain, novel-type detection) — and the memory edge is
~2×, not the widening structural advantage we previously asserted.
