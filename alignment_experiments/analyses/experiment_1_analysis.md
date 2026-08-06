# Experiment 1 — Deciding cross-device calibration test

**Status:** COMPLETE — decisive negative for the single-point M path on real data.
**Date:** 2026-08-05
**Artifacts:** `experiment_1_calibration.py`, `experiment_1_calibration_result.txt`,
`experiment_1_calibration_metrics.json`, `experiment_1_features_cache.npz`.

## Design (as planned)

- Train RandomForest on **SmellNet (Rig A)** restricted to the three live
  channels shared with the user rig — VOC, Alcohol, LPG (canonical ch 2, 4, 5).
- Feature set: the **canonical 91-dim extractor subset** for those channels
  (6 per-channel dimensions × 3 + 3 selectivity ratios + 4 globals).
- Test set: the user's real V2 recordings (garlic ×2, cinnamon ×1, banana ×1).
  lime is user-only and used as an out-of-vocabulary probe.
- 4 train classes {cinnamon, garlic, ginger, banana} → chance = 25%, matching
  the documented 33.3% external baseline's setup (4-class, chance 25%).
- Alignment: per-live-channel single-point M = median(SmellNet rel-amp) /
  median(user rel-amp), fit on reference substances present in **both**
  datasets (cinnamon, garlic, banana), applied to magnitude features only
  (relative_amplitude, auc, endpoint_delta) + recomputed amplitude globals.
  Leave-one-out discipline: evaluating substance t fits M on the references
  minus t (no test leakage). A pooled M is reported as an optimistic bound.
- Fixed seed 42; 5-fold CV sanity on SmellNet.

## Headline result

| Condition | Accuracy | Detail |
|---|---|---|
| SmellNet 5-fold CV (shared 3-ch subset) | **96.4%** | extractor + features are sound within-device |
| RAW (no alignment), user | **8.3%** (1/12) | cinnamon 0/3, garlic 0/6, banana 1/3 — all collapse to banana |
| ALIGNED, leave-one-out M | **8.3%** (1/12) | identical confusion |
| ALIGNED, pooled M (optimistic) | **8.3%** (1/12) | identical confusion |
| **Delta (aligned − raw)** | **+0.0 pp** | |

External reference (documented, 30-dim paradigm): overall **33.3%** (garlic
100%, ginger 0%, cinnamon 0%).

## Sensitivity: the banana recording is degenerate

The user banana file's VOC and LPG channels **clip to exactly 0.0** (ADC rail).
`(series − R0)/R0` then hits −1.0 → relative_amplitude = 1.0, a saturation
artifact that makes banana look like an extreme responder and collapses all
test predictions to banana.

With the banana user file excluded (test = garlic + cinnamon only):
RAW **0.0%** (0/9), ALIGNED **0.0%** (0/9). The zero-delta conclusion is
robust to removing it.

## Mechanism diagnosis

1. **The full canonical model is dominated by device-bound features.** Top-14
   RandomForest importances on the SmellNet training set are almost all
   `ch*_abs_raw_resistance`, `ch*_abs_voltage`, `ch*_abs_baseline_resistance`,
   `ch*_hw_circuit_response`. `ch4_da_relative_amplitude` ranks #13
   (importance ≈ 0.023). The model separates classes using absolute resistance
   scales that are meaningless on another rig → user samples land in
   out-of-distribution space → degenerate predictions.
2. **Magnitude rescaling cannot move predictions the model doesn't rely on.**
   Because the RF barely uses the magnitude features that M corrects, aligning
   them leaves predictions unchanged (Δ = 0.0 pp).
3. **Magnitude-only model (15 dims) also fails (0.0%).** The user rig's
   magnitude manifold is offset from SmellNet's beyond a single gain: measured
   per-channel relative-amplitude medians differ by up to ~9× on individual
   recordings (e.g., user banana LPG rel-amp = 1.0 vs SmellNet banana = 0.11),
   and the offset is a mix of per-channel gain (a), concentration-exponent
   differences (b), and **per-recording exposure-intensity variance** that M
   cannot encode.
4. **Consistent with prior negatives.** 47%→33% affine-calibration dead end,
   and now canonical zero-shot ≈ chance. Notably, the device-agnostic-only
   30-dim paradigm set (33.3%) transfers **better** than the full canonical
   set (8.3%): feeding device-bound absolute/hardware features to a zero-shot
   cross-device model actively hurts.

## Interpretation (honest)

- The deciding experiment is **falsified for the single-point M path as
  implemented**: on the real 13-file cross-device test, M alignment yields
  +0.0 pp. 13 files / 3 shared substances is directional, not proof.
- The failure is **not** evidence that M is mathematically wrong; it is
  evidence that (a) the transfer model must consume only device-agnostic
  features, and (b) real user recordings carry exposure-intensity variance +
  saturation artifacts that a single per-channel gain cannot correct.
- This sharpens the plan for **Phase 2** (synthetic mechanism probe): inject
  *known* per-channel gain (a) vs exponent (b) vs batch variance into SmellNet,
  with an A/B split, to find precisely where single-point M breaks and whether
  feature selection (device-agnostic-only) is the binding constraint.
- **Phase 3** (UCI TGS drift, controlled batches) is the right venue to test
  M vs two-point (a, b) calibration under known conditions.

## Limitations

- Tiny test set (12 shared-class windows across 4 user files; 3 shared
  substances); no user ginger in V2 format.
- sr mismatch: user rig ≈ 2.1 Hz, SmellNet assumed 10 Hz → kinetic features
  (rise/decay seconds) are not cross-device comparable; they were used
  unaligned per plan.
- One test recording (banana) is ADC-clipped; sensitivity analysis provided.
- Canonical 8.3% vs legacy 33.3% comparison is directional only (different
  test-set composition and feature set).
