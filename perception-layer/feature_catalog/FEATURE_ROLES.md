# Feature Roles: Discrimination vs Diagnostic

Not every feature exists to discriminate substances. Conflating the two roles
leads to wrong conclusions (e.g. "noise_floor is useless because removing it
doesn't hurt classification accuracy" — when its real job is detecting a
failing ADC, not identifying a chemical).

## Two Jobs

### Role: Discrimination (`ROLE_DISCRIMINATION`)
Answers: **"WHAT substance is present?"**

Validated by: **classification accuracy** (session-invariance ablation).

These features map response patterns → chemical identity. This is the
perception-layer claim. Critique: removing them should hurt accuracy.

### Role: Diagnostic (`ROLE_DIAGNOSTIC`)
Answers: **"Is the SENSOR healthy / poisoned / drifting / degraded?"**

Validated by: **diagnostic validity** — does the feature correctly separate
a known-good sensor from a known-bad (fault-injected) one?

These features support app utilities:
- **Health tracking** (drift_rate, sensitivity_decay, hysteresis)
- **Poisoning detection** (sensitivity_decay, hysteresis — a poisoned sensor
  loses response/recovery)
- **ADC / hardware QA** (adc_noise, circuit_response, thermal_profile)
- **Noise monitoring** (noise_floor, hf_transient, oscillation_*)

Critique for diagnostics is the OPPOSITE of discrimination: a diagnostic
feature is worthless if it *doesn't* change when the sensor degrades.
Removing it should make fault detection WORSE, not classification.

### Role: Absolute (`ROLE_ABSOLUTE`)
Answers: "What is the raw resistance / concentration?"

Validated by: calibration recovery — with true sensor constants `(a, b)`,
does predicted concentration match known ppm?

Only meaningful after per-rig reference-point calibration (Theorem 3).
No calibration → no meaning.

## Why This Separation Matters

The full ablation study (run_full_ablation.py) treats these separately:

1. **Discrimination features** → judged by session-invariance holdout
   accuracy (remove → does accuracy drop?).
2. **Diagnostic features** → judged by responsiveness to injected faults
   (drift / noise / poisoning): does the feature move in the expected
   direction when the sensor is degraded?

A feature can be:
- Good at both (rare) — e.g. selectivity ratios carry chemical info AND
  degrade when sensors fail
- Good at discrimination only — e.g. relative_amplitude
- Good at diagnostics only — e.g. adc_noise (inert for chemistry, vital for
  hardware QA)

## The Poisoning Problem (theory still open)

Poisoning (sensor poisoning / cross-sensitization loss) is a real failure
mode: exposure to a strong poisoner or aging irreversibly degrades a sensor's
sensitivity. The candidate diagnostic features are:

- `sensitivity_decay` — decrease in response amplitude over time. Currently a
  placeholder (always 0 in the SDK — needs multi-recording measurement).
- `hysteresis` — permanent baseline shift after exposure (irreversible
  adsorption). Promising but underexplored.
- `noise_floor` — a poisoned sensor may show abnormal baseline noise.

The theory for *which* feature cleanly isolates poisoning from ordinary drift
is **not yet settled**. This is flagged as open research and a community data
need: **labelled known-good vs known-poisoned sensor recordings** (see
sensor_profiles and the data-commons pipeline).

## Validation Protocol (per role)

| Role | Criterion | Data needed | Status |
|------|-----------|-------------|--------|
| Discrimination | Session-invariance accuracy, leave-substance-out | SmellNet (for now) | Runs; protocol-gated |
| Discrimination (recovery) | Same, but on baseline→exposure→recovery data | **Osmograph protocol data** | **NEEDED** |
| Diagnostic | Fault-injection responsiveness | Synthetic (placeholder); real fault data needed | Placeholder |
| Absolute | Calibration recovery (pred. conc. vs known ppm) | Labelled concentration data | **NEEDED** |

## Conclusion

Do NOT remove a feature just because it fails the *discrimination* ablation —
first check its *role*. If it is a diagnostic feature, test it for diagnostic
validity instead. The full ablation script does both and reports them
separately so neither criterion gets silently applied to the wrong features.
