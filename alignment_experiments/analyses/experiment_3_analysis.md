# Experiment 3 — Two-point (a,b) vs single-point M calibration on UCI drift

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_3_calibration.py`, `experiment_3_calibration_result.txt`,
`experiment_3_calibration_metrics.json`.

## Question (Phase 3)

Experiment 2 showed single-point M cannot undo exponent (b) mismatch — a single
per-channel gain cannot invert a concentration-dependent power distortion
`a_B = g·a_A^δ`. Does a **two-point log-log power-law** fit `(g, δ)` per channel
remove that δ penalty, under *real* drift where b differences and batch
structure are known? Venue: the UCI Gas Sensor Array Drift dataset (10 batches =
one rig measured over 36 months, so later batches behave as a different device
in the transfer sense).

## Design

- **Devices:** primary split A = batches 1–3 vs B = batches 8–10 (max drift);
  sensitivity split A = batches 1–5 vs B = batches 6–10 (standard split).
- **Magnitude subspace:** the two steady-state amplitude features per sensor
  (raw + normalized, feature indices `8·s+0`, `8·s+1`) → 32 dims.
- **Alignment (per amplitude feature-channel):**
  - single-point M: `a_B' = M·a_B`, `M = median(A)/median(B)`.
  - two-point power: `log a_B = log g + δ·log a_A` by least squares, using
    **quantile pairing** — A and B samples of a reference gas are paired by
    rank within that gas (histogram matching; concentrations are unlabelled in
    the `.dat` files, so no measured `(C₁, C₂)` pairs exist). Apply inverse
    `a_B' = (a_B/g)^(1/δ)`.
- **Discipline:** leave-one-out — the tested gas's responses never enter the
  fit. A pooled fit (all 6 gases) is reported as an optimistic bound.
- **Classifiers (RF, trained on device A):** magnitude-only (32 dims) isolates
  the calibration subspace; full 128-dim shows dilution by unaligned
  kinetic/derivative features. Chance = 16.7%.

## Headline results

Within-device ceiling (train + test on A) is **100%** for both models — the
features fully separate the 6 gases on the source device; all cross-batch
accuracy loss is drift-induced distribution shift, not feature insufficiency.

### Primary split (A = 1–3, B = 8–10)

| Model | RAW | ALIGNED M | ALIGNED power |
|---|---|---|---|
| magnitude-only (32d) | **37.8%** | 20.4% (−17.4 pp) | 27.9% (−9.9 pp) |
| full 128-dim | 38.8% | 38.8% (+0.0 pp) | 38.8% (+0.0 pp) |

Pooled (optimistic): mag RAW 37.8% → M 39.7% → power 35.0%; full 38.9% (all).

### Sensitivity split (A = 1–5, B = 6–10)

| Model | RAW | ALIGNED M | ALIGNED power |
|---|---|---|---|
| magnitude-only (32d) | 42.7% | 34.4% (−8.3 pp) | **43.1%** (+0.4 pp) |
| full 128-dim | 50.0% | 50.0% (+0.0 pp) | 50.0% (+0.0 pp) |

Pooled (optimistic): mag RAW 44.5% → M 47.5% → power **49.5%** (+5.0 pp); full
51.3% (all).

Fitted exponents (primary split, pooled): median **δ = 0.727** (range ≈
0.6–0.98), 0/32 degenerate — real drift *does* contain an exponent-like (b)
distortion, so a power-law correction is in principle the right family.

## Mechanism

1. **The full model is invariant to both calibrations (+0.0 pp in both
   splits).** It separates gases using kinetic/derivative features the
   magnitude transforms never touch — the same mechanism Exp 2 measured on
   SmellNet, now reproduced on real UCI drift.
2. **Single-point M actively hurts the magnitude model** (−17.4 pp / −8.3 pp
   LOO). M = median-ratio assumes B's amplitude manifold is A's times a
   constant gain. Real drift violates this twice: (a) δ ≈ 0.73 ≠ 1 (exponent
   drift, which M structurally cannot invert), and (b) batch-to-batch
   **concentration-distribution shifts** — batch 10 is 600 samples/gas vs
   batch 1–3's irregular per-gas concentrations — so the median ratio encodes
   concentration change, not device change, and erases legitimate
   between-gas signal.
3. **Two-point power is nearer-neutral than M** (−9.9 pp / +0.4 pp LOO; +5.0 pp
   pooled in the sensitivity split) because it at least fits the exponent. But
   it is **not decisive**: the pooled result flips sign between splits
   (primary −2.8 pp, sensitivity +5.0 pp), and the LOO gain never approaches
   the 100% ceiling. Per-gas instability is large (e.g., held-out acetone
   jumps to ~97% while other gases drop), a signature of over-fitting the
   reference manifold: without concentration labels, quantile pairing cannot
   separate device gain/exponent from gas-identity concentration structure,
   and extrapolation to a held-out gas's amplitude regime is unreliable.
4. **Consistent cross-split facts:** ceiling 100%, full-invariant, mag capped
   at 38–50% cross-batch, toluene ≈ 0% (only ~79 early-batch samples vs 600 in
   batch 10 — a class-balance artifact of the split, not a calibration effect).

## Interpretation (honest)

- **Two-point (a,b) does not remove the δ penalty on real UCI drift.** It is
  directionally better than single-point M under exponent drift (consistent
  with Exp 2), but the recovery is ~0–5 pp on the magnitude-only model and 0 pp
  on the full model — nowhere near the within-device ceiling.
- The **binding constraint remains feature discipline, not calibration order.**
  Magnitude-only transfer is capped at ~40–50% even with calibration; the full
  model is calibration-invariant and also capped (~50%).
- Real cross-batch transfer on UCI needs either (a) matched-concentration
  calibration data (two *known* concentrations per gas, which the `.dat` files
  do not provide — the quantile-pairing workaround is weaker than a true
  two-point fit), or (b) a model that trains on device-agnostic + kinetic
  features with calibration applied *inside* the training loop, or (c) the
  session-invariance route (learning over sessions, 81.78% on SmellNet) rather
  than zero-shot calibration.

## Strategic readout

- The three-experiment arc is now internally consistent:
  - **Exp 1** (real SmellNet→user): M gives +0.0 pp; full model dominates via
    device-bound features.
  - **Exp 2** (synthetic): M is mathematically sound for pure gain but cannot
    undo exponent distortion or intensity variance; full model is 100%
    invariant to magnitude alignment.
  - **Exp 3** (UCI real drift): M hurts the magnitude model; two-point power
    reduces but does not remove the penalty; full model still invariant.
- **Conclusion for the paper:** zero-shot cross-device calibration — single-
  point or two-point — is not a transfer solution for MOX e-noses. The honest,
  publishable position is: (i) features must be device-agnostic by *selection*
  (magnitude + selectivity only), (ii) calibration needs known-concentration
  reference points, (iii) the measured session-invariance (81.78%) is the
  actual interoperability result. This is a clean **negative result for the
  calibration path**, completing §8.7's "calibration is reference-point
  strategy, not universal."

## Limitations

- UCI `.dat` files carry **128 precomputed features and no concentration
  labels**; two-point calibration is approximated by within-gas quantile
  pairing (histogram matching), which conflates device drift with
  concentration-distribution shift. A true `(C₁, C₂)` two-point test requires a
  dataset with measured concentrations.
- Toluene (~79 samples in A vs 600 in B) and the irregular per-batch gas mix
  make A imbalanced; `class_weight="balanced"` mitigates but does not remove
  the artifact.
- The magnitude subspace (indices `8s+0/8s+1`) is assumed to be the two
  amplitude features per the dataset's documented layout; kinetics/transients
  live in the other six and are intentionally left unaligned (that is the
  "full model" condition).
- RandomForest; no hyperparameter search; alignment is applied
  out-of-loop (not refit jointly with the classifier), matching Exp 1/2 and
  keeping the calibration effect isolated.
