# Experiment 7 — Chemoprint as a cross-device prior

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_7_chemoprint.py`, `experiment_7_chemoprint_result.txt`,
`experiment_7_chemoprint_metrics.json`.

## Question (Phase 4, Path B)

The Chemoprint README claims a physical sensor array can be calibrated to
output the 29-dim chemoprint with variance-weighted **R² = 0.982** (UCI, 128
precomputed features per sample). That number came from a random 80/20 split
over all 10 drift batches — an in-sample claim that lets the model see every
gas's pattern and simply reproduce it. The interoperability question, which
the README itself flags as untested, is: does the **sensor → chemoprint map
transfer across a device boundary**? If yes, chemoprint is a genuinely
device-agnostic prior for digital olfaction; if not, it is device-bound like
every other learned sensor mapping.

## Design

- Venue: UCI Gas Sensor Array Drift (A = early batches, B = late batches =
  "different device"), same splits as Exps 3/4/6.
- Target: the 29-dim chemoprint computed from each gas's SMILES (constant
  within a gas, so R² measures whether the A-learned gas→chemoprint mapping
  preserves gas clusters in B's feature manifold).
- Regressor: RandomForest (200 trees), full 128-dim (venue of the original
  claim) and magnitude-only 32-dim.
- Conditions: `mixed80` (reproduces the README claim), `within_A`, `within_B`,
  `cross_AB`, `cross_BA`.

## Headline results

| Condition (full model) | R² variance-weighted |
|---|---|
| mixed 80/20 (all batches) — README claim | **0.973** (README: 0.982) |
| within_A (train+test on A) | **99.6%** |
| within_B (train+test on B) | **99.6–99.7%** |
| **cross A→B (primary 1–3→8–10)** | **2.4%** (drop −97.2 pp) |
| **cross A→B (sensitivity 1–5→6–10)** | **−5.0%** (drop −104.6 pp) |
| cross B→A (primary / sensitivity) | 49.6% / 58.2% |
| cross A→B, magnitude-only (primary / sensitivity) | 13.7% / 2.6% |

Per-dim R² (cross A→B) is ≈ 0 or negative across the board (MW +0.19/+0.04,
heavy atoms +0.11/−0.03, TPSA-ish −0.44/−0.42, rings −0.76/−0.36, Wiener
−0.08/−0.16). Per-gas MAE on B is largest for toluene (4.77/4.96) vs ethanol
(0.35/0.39) — the heaviest gas is systematically misread cross-device.

## Mechanism

1. **The sensor→chemoprint map is device-bound.** Within-device the map is
   near-perfect (99.6%): the regressor learns which absolute feature patterns
   correspond to which gas on a given rig. Across the drift boundary it
   collapses to ~0–2% R², and on the sensitivity split it is *worse than
   predicting B's mean* (R² < 0). The A-learned gas→chemoprint couplings are
   keyed to A's absolute feature scales — the same device-bound mechanism that
   collapsed Exp 1 (everything→banana), Exp 5 (everything→Woody) and the
   chemoprint target is no exception.
2. **The in-sample 0.982 is a leak artifact, not a device-agnostic property.**
   The random 80/20 split mixes batches, so train and test contain the same
   gases from overlapping drift periods; the regressor reproduces gas→chemoprint
   from within-sample patterns. The cross-device number (2.4%/−5.0%) is the
   honest statistic and it destroys the "hardware-agnostic representation"
   claim as a *transfer* claim. (The chemoprint as a pure SMILES descriptor
   remains exact — this experiment tests the *mapping*, not the descriptor.)
3. **Reverse direction (B→A) transfers better** (49.6%/58.2%) — consistent
   with Exp 3/4's asymmetry (A is early/cleaner, B is late/more-concentrated;
   the direction matters but neither direction is device-agnostic).

## Interpretation (honest)

- **Chemoprint-as-prior is falsified for cross-device transfer on real drift.**
  99.6% within → ~0–2% across is a decisive negative. If chemoprint were a
  device-agnostic intermediate, the A-trained map would need to place B's gas
  clusters at their true chemoprints; it does not.
- This closes the last proposed path in the Qwen brief: scalar calibration
  (Exps 1–3), covariance alignment (Exp 4), supervised anchors (Exp 5),
  ontology coarsening (Exp 6), and chemoprint-as-prior (Exp 7) all fail the
  zero-shot cross-device test. The shared root cause across all seven: the
  learned mappings key to device-bound feature scales, and every "device-
  agnostic" intermediate (paradigm features, chemoprint) still requires
  per-device calibration of its *sensor→representation map*.
- The one legitimate caveat: chemoprint is a *descriptor* of molecular
  structure; it is exact for pure compounds and is a fine interpretive layer
  *within* a calibrated device. It is not a transfer mechanism.
- Consistent with the impossibility theorem: with no target-device reference
  samples, no feature representation or mapping recovers gas identity across
  the boundary; with reference samples, everything (calibration, CORAL,
  anchors) helps partially. The measured 81.78% within-rig session-invariance
  remains the only demonstrated interoperability.

## Limitations

- Negative R² on cross splits is a legitimate domain-shift outcome (prediction
  worse than the mean baseline), not a code artifact; reported honestly.
- The chemoprint target is constant per gas, so R² is dominated by
  between-gas variance (esp. toluene's MW=92); per-gas MAE is provided so the
  effect is not attributed to a single dimension.
- mixed80 reproduces 0.973 vs README 0.982 (n_estimators 200 vs 100, seed);
  the ~0.01 gap does not change the conclusion.
- Magnitude-only cross R² (13.7%/2.6%) is reported for the drift-subspace
  comparison; the original claim used full 128-dim.
