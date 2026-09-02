# Full 187-Feature Ablation (research/full_ablation)

This is a **separate, more rigorous** ablation than the canonical Experiment 5.
The canonical one only ablates Dimension 1 (the 55 device-agnostic features) on
SmellNet. This one ablates **all 187** SDK features across every dimension, and
adds two critical guardrails the canonical study lacks:

## Guardrail 1: Protocol-Dependence is Enforced

Kinetics, decay, and hysteresis features are only meaningful when the
recording follows the `baseline → exposure → recovery` protocol. SmellNet
does **not**: its sessions are ~60s monotonic exposure ramps with no recovery
phase (verified empirically — see below).

The script therefore **detects protocol compliance** and marks any
protocol-dependent group's verdict as `verdict_valid_on_this_data = false`
when the dataset lacks a recovery phase. You cannot judge a feature family on
data that cannot express it.

Evidence (from `load_all_smellnet_data`):
- Session lengths: median 600 samples (~60s at 10Hz)
- Example session (allspice): channel 0 rises monotonically 52 → 310 with no
  return toward baseline
- i.e. no desorption/recovery arc to fit `decay_time`, `decay_tau*`,
  `hysteresis`

## Guardrail 2: Features are Classified by ROLE

Not all features exist to discriminate substances. See
`feature_catalog/FEATURE_ROLES.md`. The script separates:

- **Discrimination** (97 feats) → judged by session-invariance accuracy
- **Absolute** (24 feats) → judged by calibration recovery
- **Diagnostic** (60 feats) → judged by responsiveness to injected faults
  (drift / noise / poisoning), NOT by classification accuracy

A diagnostic feature that's inert for classification (like `noise_floor`) is
not "useless" — its job is hardware QA. Judging it by classification
accuracy is a category error. The script reports `diagnostic_validity`
separately.

## Honest Caveats (why these numbers are NOT final)

1. **No protocol data yet.** The whole point of this research direction is
   the recovery-phase / protocol-dependent features, and SmellNet cannot
   exercise them. The single most important next data need is
   **Osmograph-protocol recordings** (baseline→exposure→recovery).
2. **Diagnostics are validated on synthetic faults only.** The
   `validate_diagnostics_on_synthetic` function injects drift/noise/poisoning
   synthetically as a placeholder. Real validation needs **labelled
   known-good vs known-bad sensor recordings**.
3. **Absolute features** (raw_resistance, voltage, concentration) mean
   nothing without per-rig calibration; their ablation "importance" on
   SmellNet is a dataset artifact, not a physics statement.

## How to Use

```bash
# Quick test (small data)
python experiments/run_full_ablation.py --limit 48

# Full run (slow: ~50 substances)
python experiments/run_full_ablation.py --limit 0

# Dry-run group/role definitions only (no data)
python experiments/run_full_ablation.py --no-smellnet
```

> **Note on `results/full_ablation.json`:** the committed file is a **sample
> result** from an 8-substance subset (baseline 96.9%) so it runs in seconds.
> The full 50-substance run is compute-heavy (RandomForest over 187 dims ÷
> many tree fits); reproduce it with `--limit 0`. The numbers in the sample
> file are illustrative of the *method*, not the final scientific result —
> the decisive evidence still requires **protocol data** (see below).

Output: `results/full_ablation.json` with:

- `feature_roles`: which role each feature plays
- `protocol_compliance`: recovery-phase detection per substance + composite
- `groups`: per-group ablation deltas + protocol validity flag
- `flagged_candidates`: individual flagged-feature verdicts
- `diagnostic_validity`: per-feature responsiveness to injected faults
- `removal_candidates`: features whose removal *improves* classification
  (only trustworthy for non-protocol-dependent, non-diagnostic features)

## Verdict interpretation rules

- **Never remove** a protocol-dependent feature based on SmellNet ablation.
- **Never remove** a diagnostic feature based on classification ablation.
- Only trust removal verdicts for **discrimination-role, protocol-valid**
  features on SmellNet.
- Re-run everything on protocol data before finalizing any removal.
