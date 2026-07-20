# actinn-jax vs ProtoCloud

[ProtoCloud](https://doi.org/10.1016/j.xgen.2026.101217) (Guo & Ding, *Cell Genomics*
2026; code [Ding-Group/ProtoCloud](https://github.com/Ding-Group/ProtoCloud)) is a
prototype-based, self-explaining variational autoencoder for single-cell annotation. It
overlaps actinn-jax's territory almost point-for-point — reference annotation, built-in
uncertainty, rare/novel-population discovery, marker-gene explainability — but its paper
benchmarks only heavyweight and foundation models (Seurat, CellTypist, scANVI, scPoli,
TOSICA, SIMS, scGPT, scBERT). There is **no fast, lightweight, CPU baseline** in that
comparison. actinn-jax is exactly that missing point of reference.

## Setup

ProtoCloud is added as a benchmark adapter (`benchmark/adapters/protocloud_adapter.py`,
env in [`envs/README.md`](../envs/README.md)) and run through the same harness as every
other method. We use ProtoCloud's **published defaults** (encoder 1024/512/256, 20-d
latent, 6 prototypes/class, 100 epochs, AdamW, raw UMI input, rare-type multinomial
augmentation) and its paper's **3000 HVGs**. Its `pc_certainty == 'ambiguous'` flag is
surfaced as the harness's rejection column.

**One difference matters and is stated up front:** ProtoCloud is written for CUDA-or-CPU
(no Apple MPS) and its paper runs on an NVIDIA V100. Here it runs on **CPU**. Accuracy is
hardware-independent — the numbers below are what ProtoCloud produces regardless of
device — but its *training time* reflects the lack of a GPU. That is precisely the axis
actinn-jax exists to remove: it needs no GPU at all.

Config: [`configs/protocloud_compare.yaml`](../configs/protocloud_compare.yaml). Intra
datasets are subsampled per label (200–300 cells/type) so ProtoCloud's CPU VAE training
stays tractable; **both methods see the identical subsample and split**, so the accuracy
comparison is fair. Results: [`results/protocloud_compare/results.csv`](../results/protocloud_compare/results.csv).

## Results (our datasets, CPU, 3000 HVGs)

| dataset (n types) | metric | actinn-jax | ProtoCloud |
|---|---|---|---|
| pbmc3k (8) | accuracy / macro-F1 | **0.913 / 0.795** | 0.880 / 0.770 |
| lung, Krasnow (46) | accuracy / macro-F1 | 0.894 / 0.901 | **0.932 / 0.932** |
| liver, HLiCA (36) | accuracy / macro-F1 | **0.802 / 0.798** | 0.695 / 0.689 |
| blood+gut, HLiCA (86) | accuracy / macro-F1 | **0.860 / 0.860** | 0.841 / 0.839 |
| **mean** | accuracy / macro-F1 | **0.867 / 0.839** | 0.837 / 0.807 |

| dataset | actinn-jax fit | ProtoCloud fit | speedup | peak mem (aj / PC) | PC abstained |
|---|---|---|---|---|---|
| pbmc3k | 6.8 s | 62 s | 9× | 0.9 / 1.1 GB | 12% |
| lung | 18 s | 246 s | 13× | 2.0 / 2.0 GB | 19% |
| liver | 29 s | 123 s | 4× | 1.5 / 1.6 GB | 13% |
| blood+gut | 24 s | 236 s | 10× | 2.5 / 2.1 GB | 17% |
| **mean** | **19.5 s** | **167 s** | **~8.5×** | comparable | 12–19% |

Both methods are deterministic given the fixed split, so the three repeats agree exactly.
Predict time is sub-2 s for both. actinn-jax never abstains here (default `min_prob`
off); ProtoCloud rejects 12–19% of cells as "ambiguous" via its uncertainty flag — a
feature actinn-jax matches with `min_prob` when desired.

## Reading

- **Accuracy is close and splits both ways.** actinn-jax is higher on 3 of 4 datasets
  (pbmc, liver, blood+gut) and on the mean; ProtoCloud is clearly higher on the
  finest-grained set (lung, 46 types), where a deep model's extra capacity helps. This is
  the expected shape: a small MLP and a prototype VAE trade the lead depending on how much
  structure there is to model.
- **actinn-jax trains ~8.5× faster, on CPU.** Even setting the GPU question aside, the
  compute asymmetry is large — and actinn-jax's whole premise is that you never reach for
  a GPU. ProtoCloud on a V100 would close most of the *time* gap, but not the hardware
  requirement.
- **Comparable memory** at these sizes; both 1–2.5 GB.
- **Feature parity where it counts:** ProtoCloud's headline extras — built-in uncertainty
  and per-cell gene explainability — have lighter analogues in actinn-jax (`min_prob`
  abstain; per-cluster markers from `detect_novel_celltypes`). ProtoCloud's per-cell LRP
  gene relevance is genuinely richer.

## Honest caveats

This comparison deliberately probes the **lightweight-CPU corner**, which is *not*
ProtoCloud's home turf:

1. **Scale — since confirmed, and it reverses the conclusion.** ProtoCloud's paper uses
   10k–400k-cell atlases; the table above subsamples to 200–300 cells/type. We predicted
   its relative accuracy should improve with more cells, and a scaling sweep
   ([`SCALING_MEMORY.md`](SCALING_MEMORY.md)) confirms it emphatically: on the lung
   reference from 3k → 49k cells ProtoCloud goes from **worst (0.722) to best (0.976)**,
   clear of actinn-jax (0.936) and a tuned linear pipeline (0.939). **At atlas scale
   ProtoCloud is the most accurate method we have benchmarked**; the "close and splits both
   ways" reading below holds only for subsampled references. Its cost at 49k cells is
   1,541 s vs 81.5 s on CPU (19×) — a gap the GPU it targets would largely close.
2. **Hardware.** CPU, not the V100 the method targets. Fair for accuracy, unflattering for
   ProtoCloud's wall-clock.
3. **Their datasets, not ours.** ProtoCloud reports matching/beating 8 methods on 8
   specific atlases (PBMC10K/30K, AtlasRGC, TSCA lung/oesophagus/spleen, AtlasEoE, ICA;
   accessions in the paper's Key Resources Table). We did not reproduce those; a true
   apples-to-apples on that suite — ideally on GPU — is the natural follow-up.

**Takeaway.** On small-to-moderate data on a laptop CPU, actinn-jax matches or beats
ProtoCloud on most datasets while training ~8.5× faster and needing no GPU; ProtoCloud
pulls ahead where fine-grained structure and (presumably) scale reward a deep generative
model, and it offers richer built-in interpretability. They sit at opposite, complementary
ends of the accuracy/interpretability-vs-speed/simplicity spectrum — which is the point of
adding the baseline ProtoCloud's own paper omits.
