# Contributing to the Hardware-Agnostic Perception Layer

This research direction can only be completed **as a community**: the theory
and framework are in place, but the decisive evidence requires data and
hardware many of us don't have alone. Here is how to contribute — each item is
tied to a specific open milestone.

## What's Blocking, Ranked by Impact

> **Read this first:** `docs/OPEN_QUESTIONS.md` is the authoritative list of what
> the ontology cannot yet answer, stated as precise, falsifiable questions with
> the *exact data each one needs*. Filing a dataset against one of those
> questions (not against a vague "we want more data") is the highest-value
> contribution. Each P0 item below is a concrete instance of one of those
> questions.

### P0 — The decisive bottleneck: Protocol-Data (Milestone 1)

The feature foundation cannot be fully proven until we have recordings that
follow the `baseline → exposure → recovery` protocol. This is now *measured*,
not assumed (see `results/validation_piecewise.json`): the recovery-dependent
axes of the phenotype layer cannot be trained or validated from any current
public dataset — UCI Gas Drift (Vergara) is pre-extracted features with no raw
curves, UCI Home Activity (Hu) does not separate its targets ("banana"/"wine"),
and SmellNet recordings are inconsistent for A2 (of 68 inspected, 51 had a
genuine recovery return and 17 were ramp-only — so A2 must be judged per
recording, not per dataset).

**If you have an OpenSmell rig (or any MOX e-nose) that records per
`PROTOCOL.md`:** record clean-air baseline (30s) → exposure (45s) →
recovery (120s) for a small set of known substances (e.g. ethanol, vinegar/
acetic acid, acetone, ammonia, a clean alcohol like propanol), with the
substance labelled. This single contribution unlocks Milestone 1 and makes the
**A2 (kinetic) percept mapping** trainable for the first time.

- Format: `.osmell` (preferred) or raw CSV with a manifest describing
  baseline/exposure/recovery times.
- Submit via the `data-commons/` pipeline.
- What we'll do: run the full ablation, validate the recovery features, learn
  the A2 percept map, and credit you in the dataset + paper.

### P0 — Diverse-chemistry protocol recordings (A2/A3 percept map)

The measured percept map (`ontology/percept_map.py`) is currently **A3-only**
and anchored on Vergara's 6 gases. To complete it we need protocol recordings
(above) spanning **diverse chemistry**: more reducing VOCs AND at least one
oxidizing gas (e.g. NO2/O3) and one redox-inert gas (CO2/N2) — so the A1 redox
axis, A2 kinetics, and A3 fingerprint can all be separated and mapped to
PERCEPTS with confidence. Raw curves on a **known-dopant reference rig** (n vs
p) additionally pin the absolute A1 sign (see `ontology/detect.py`).

### P0 — Labelled concentration data (Theorem 3 validation)

The reference-point calibration path is implemented but unvalidated on real
data. If you can expose a sensor to **known ppm** values of a gas, the
`(a, b)` constants become measured rather than nominal, and Milestones 2-3
become possible.

- Even 4-6 concentration points × 1 gas is enough to start.

### P0 — Known-good vs known-poisoned sensor recordings (Diagnostics)

The diagnostic features (poisoning, drift) are currently validated only on
*synthetic* faults (see `run_full_ablation.py` →
`validate_diagnostics_on_synthetic`). Real validation needs recordings where
the sensor's health is **labelled**: known-good, known-poisoned (e.g. after
exposure to a known poisoner like silicone / sulfur compounds), known-drifted
(after long aging). This settles the open "poisoning theory" question in
`FEATURE_ROLES.md`.

### P1 — A new sensor profile (Milestone 2)

Fill `hardware_aware/sensor_profiles/templates/TEMPLATE.json` with your rig's
parameters: sensor models, `(a, b)` if you have them, circuit (Vcc, RL, ADC
bits), gas delivery, and whether it records per protocol. This is lightweight
(10 min) and immediately grows the device pool needed for the universal model.

### P1 — Cross-device validation experiment (Milestone 3-4)

If you have **two devices** (even two rigs of the same model, better if
different sensor models), record the same substances on both. This is the raw
material for device-LOO validation — the proof of hardware-agnostic
generalization.

## Roles and Their Validity Criterion

When you contribute or review features, remember the two roles
(`feature_catalog/FEATURE_ROLES.md`):

| Role | Question | Validated by | Examples |
|------|----------|--------------|----------|
| Discrimination | WHAT substance? | Session/device-LOO accuracy | amplitude, selectivity ratios |
| Diagnostic | Is sensor healthy? | Responsiveness to known faults | noise_floor, adc_noise, hysteresis |
| Absolute | What raw resistance/conc? | Calibration recovery | raw_resistance, concentration |

**Do not judge a diagnostic feature by classification accuracy, or vice
versa.** And **do not judge a recovery-phase feature on non-protocol data.**

## Adding or Removing a Feature

1. **Argue it physically first** (`feature_catalog/CATALOG.md`): what
   molecular/surface process does it capture? What is its
   device-transferability?
2. **Assign a role.**
3. **Add it to the SDK extractor** (`opensmell/opensmell/mox/features.py`) —
   do NOT create a parallel extractor (see IMPLEMENTATION_ALIGNMENT.md).
4. **Run the full ablation** — but only remove it if it fails its OWN role's
   criterion on PROTOCOL data.
5. Add a test asserting the feature behaves as expected.

## Community Norms

- **Recoverability**: fixed random seeds, documented data sources, results
  written as JSON.
- **No silent placeholders**: if a feature is a placeholder (like
  `sensitivity_decay` = always 0), say so in the catalog — don't let it imply
  real signal.
- **One source of truth for extraction**: the SDK. Report divergences, don't
  fork silently.
- **Open an issue** before a big PR; small focused PRs are preferred.

## Where to start

- Want the quickest high-impact win? **P0 protocol-data** or **P0 sensor
  profile**.
- Want to code? Improve `validate_diagnostics_on_synthetic`, add a
  `train_universal_model.py`, or implement a better protocol-compliance
  detector.

## Data submission

- `.osmell` / CSV protocol data, concentration data, sensor-health data →
  `data-commons/` pipeline.
- Sensor profiles → PR to
  `research/hardware_aware/sensor_profiles/`.
- Questions/ideas → GitHub issues on this repository.
