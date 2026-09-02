# Universal Model Architecture (Milestones 3-4)

## Goal

A single model that classifies odors across **different hardware** by taking
both the physicochemical features AND the sensor's identity parameters as
input. This is the "hardware-aware" (a.k.a. conditional) model.

## Input Vector

```
x = [ F_physicochemical (187-dim)  ⊕  P_sensor_params ]
```

Where `P_sensor_params` is built from the given device's `sensor_profile.json`:

- per channel: `[a, b]` (sensitivity, exponent)
- optionally: operating temperature, doping type (one-hot), ADC bits

## Why This Helps (vs. ignoring hardware)

- Raw approach (ignore hardware): model must "average out" unit differences
  → fits the majority device, fails minority devices (zero-shot failure).
- Hardware-aware (conditional): the model learns a **family of lines**, one
  per device, sharing physics. A new device needs only its profile → no
  retraining, just parameter conditioning.

## Validation Protocol (device-LOO)

**Critical:** must be held-out *device*, not held-out session. This is what
distinguishes true generalization from within-device memorization.

```
for each device D in devices:
    train on all devices except D
    test on D only (never saw D in training)
report: mean + per-device accuracy, confusion matrix
```

This is analogous to alignment experiment 7 (chemoprint) but at the model
level, and it's the real target.

## Model Candidates

1. **Baseline**: RandomForest on `[F | P]` — fastest, interpretable, good
   starting point.
2. **Gradient boosting** — often best tabular performance.
3. **Small MLP** on `[F | P]` — captures nonlinear sensor-parameter
   interactions.
4. **(Later) Modality-style encoder** — if extending cross-modality, treat
   `F` as the modality representation and `P` as a conditioning vector.

Always: fixed seeds, StandardScaler fit on train only, device-LOO.

## Milestone Roadmap

| # | Milestone | Criterion | Data needed | Status |
|---|-----------|-----------|-------------|--------|
| 1 | Feature foundation on protocol data | Physicochemical features separate functional groups (alcohols/aldehydes/acids) on baseline→exposure→recovery data | **Osmograph protocol recordings** | NEEDED |
| 2 | Sensor profiles | Machine-readable `(a, b)` + protocol + circuit per device | Sensor datasheets / contributor rigs | Template ready |
| 3 | Train universal model | Accuracy on device-LOO > chance across ≥2 devices | Multi-device protocol data | Blocked on M1+M2 |
| 4 | Validate new device | Held-out device accuracy; iterate feature set | A brand-new device | Blocked |
| 5 | Toward genotype | Map phenotype→molecular structure (MIRIS/richer data) | Precise sensors | Future |

## Files

- `experiments/run_full_ablation.py` — Milestone 1 feature validation
  (current, protocol-gated)
- `sensor_profiles/` — Milestone 2
- (future) `experiments/train_universal_model.py` — Milestone 3, once data
  exists
