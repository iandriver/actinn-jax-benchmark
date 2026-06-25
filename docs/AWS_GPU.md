# Deferred plan: optional cloud-GPU (AWS) runtime comparison

**Status: deferred.** The primary benchmark runs CPU-first on the Mac M5 Pro (plus
Apple MPS where supported). This GPU pass is held until the CPU/MPS results are in,
so we only pay for cloud GPU if it would actually change a conclusion.

## When it's worth doing

Run it only if the CPU/MPS results show that **runtime is the deciding factor** for
the GPU-native methods — i.e., a deep / foundation method is accuracy-competitive but
looks unusably slow on CPU/MPS, and a CUDA number would change the recommendation.
If the foundation models are not accuracy-competitive, or MPS is adequate, skip it.

## What it would measure

A second, clearly separated runtime table for **Tier 2 (deep)** and **Tier 3
(foundation)** methods on CUDA:
- scANVI / scArches, scGPT / scBERT / scDeepSort / TOSICA — fit and predict time,
  peak GPU memory, scaling vs #cells.
- Classical Tier-1 methods are CPU-bound; re-run them on the box's CPU only as a
  cross-machine sanity check, **never** compared against the GPU numbers.

## Proposed setup

- **Instance:** a single `g5.xlarge` (1× NVIDIA A10G, 24 GB) — enough VRAM for
  scGPT/scBERT inference and scvi-tools training at benchmark scale. Step up to
  `g5.2xlarge`/`g6` only if VRAM-bound.
- **Image:** AWS Deep Learning AMI (Ubuntu, CUDA + PyTorch preinstalled).
- **Repro:** same `benchmark.runner` CLI and per-method env files; only the device
  changes. Push input h5ads to S3, pull results back.
- **Cost control:** spot instance, auto-stop on completion, one dataset at a time.

## Rough cost sketch (to refine before committing)

`g5.xlarge` is ≈ $1.0/hr on-demand, ≈ $0.40/hr spot (us-east-1; **verify current
pricing**). A full Tier-2 + Tier-3 sweep on the chosen datasets is estimated at a
few GPU-hours, so order-of-magnitude **$5–20**. The decision point: is removing the
CPU/MPS timing caveat for foundation models worth that and the setup time? Revisit
once Phase 3 (MPS) numbers exist.

## Decision checklist (fill in after CPU/MPS phases)

- [ ] Which methods are accuracy-competitive enough that their runtime matters?
- [ ] Is MPS timing already acceptable / representative for those?
- [ ] Estimated GPU-hours for the needed runs → estimated $ at current spot price.
- [ ] Go / no-go.
