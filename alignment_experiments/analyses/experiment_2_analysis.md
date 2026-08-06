# Experiment 2 — Synthetic mechanism probe

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_2_synthetic_probe.py`, `experiment_2_synthetic_metrics.json`.

## Question

Phase 1 showed single-point M alignment gives +0.0 pp on real data. Is that a
failure of M itself, or of the confounds around it? This probe removes the
confounds by injecting a *known* per-channel transform into SmellNet and
measuring how much single-point M can undo.

## Design

- Model: RandomForest on clean SmellNet (device A), scored on simulated
  device-B windows (train windows = same recording family — this is a
  mechanism probe, not a real transfer estimate; all cells are relative to
  the identity ceiling, which is ~99–100%).
- Device-B transform (per live channel, per window):
  `mult = g · e^ε · a^δ` applied to magnitude features (`a` = relative
  amplitude); AUC/endpoint_delta scaled by the same mult; selectivity ratios
  and amplitude globals recomputed.
  - δ = 0  → pure per-channel gain (M's designed regime).
  - δ > 0  → concentration-dependent distortion (exponent/b mismatch).
  - σ = 0.25 → per-window log-normal intensity wobble (batch variance).
- M fit two ways: **oracle** (test substance's own windows) and **ref**
  (other reference substances, honest, Phase-1-style).
- Two models: **magnitude-only (15 dims)** isolates M's mathematical adequacy;
  **full 91-dim** shows dilution by device-bound/kinetic features.

## Results (magnitude-only model, mean over 3 seeds)

| Cell | RAW | ALIGNED oracle | ALIGNED ref |
|---|---|---|---|
| identity (ceiling) | 99.3% | 99.3% | 99.3% |
| pure gain 0.5× | 29.5% | **99.3%** | **99.3%** |
| exponent-shift 0.2 | 52.6% | 80.2% | 69.9% |
| exponent-shift 0.5 | 35.5% | 67.4% | 61.2% |
| gain 0.5 + exp-shift 0.5 | 11.3% | 67.4% | 61.2% |
| batch noise 25% (δ=0) | 66.9% | 68.4% | 61.8% |
| exp-shift 0.5 + batch noise | 38.0% | 59.0% | 57.8% |

Full 91-dim model: **RAW = 100% in every cell** — the magnitude transform does
not touch the abs/hw/kinetic features the full model leans on.

## Conclusions (mechanism decomposed)

1. **M is mathematically sound for pure per-channel gain.** A 0.5× uniform gain
   collapses magnitude-only accuracy to 29.5%; single-point M restores it to
   **99.3% ≈ ceiling — even when fit honestly on reference substances**. The
   concept works exactly in its designed regime.
2. **M degrades when exponent differences (b) dominate.** δ = 0.2 → oracle
   recovers 80.2% (ref 69.9%); δ = 0.5 → oracle 67.4% (ref 61.2%). A single
   scale cannot invert a concentration-dependent power distortion. This is the
   precise sense in which "single-point M breaks when b differences dominate."
   Cross-substance fitting (ref) loses ~8–10 pp more than oracle because the
   power-law distortion's effective scale depends on the substance's
   amplitude/concentration distribution.
3. **M cannot remove per-window intensity variance.** With 25% batch noise even
   the oracle M adds only +1.5 pp (68.4% vs 66.9%). A single median-ratio gain
   leaves per-window residuals. This is the floor that the ADC-clipped banana
   file and real exposure-intensity variance impose.
4. **The Phase 1 +0.0 pp is reproduced synthetically and explained.** The full
   91-dim model is 100% invariant to the magnitude transform, because it leans
   on abs_*/hw_*/kinetic features that M never touches. When the transfer model
   is fed the full canonical feature set, magnitude alignment cannot change its
   predictions — exactly what Phase 1 measured.

## Strategic readout

- The M concept is **vindicated as a first-order, honest correction** for
  per-channel gain differences — not falsified. It fails precisely when
  (a) exponent differences dominate and (b) per-recording intensity variance
  is large.
- The real lever for cross-device transfer is **feature discipline** (train on
  device-agnostic magnitude + selectivity only), not calibration alone.
- Phase 3 (UCI controlled batches) is now well-posed: test whether two-point
  (a, b) calibration removes the δ penalty that single-point M cannot, on data
  where b differences and batch structure are known.

## Limitations

- Train/test overlap (same windows) inflates absolute levels; all conclusions
  are *relative* to the identity ceiling and are therefore about M's
  corrective power, not absolute transfer accuracy.
- The transform acts on magnitude features only; kinetics/temporal/abs/hw are
  held identical between A and B (a controlled simplification that isolates
  the magnitude subspace M lives in).
- RandomForest; no concentration labels — δ is a stand-in for exponent/b
  mismatch via power-law amplitude distortion, not a measured b.
