# Implementation Alignment: Three Feature Extractors

## The Problem

There are three implementations of MOX feature extraction in this repository:

1. **SDK** (`opensmell/opensmell/mox/features.py`) — 187 features, production code
2. **Canonical experiments** (`interoperability/canonical_experiments/framework_features.py`) — 145 features, paper experiments
3. **Smell-monitor** (`smell-monitor/src/core/features.py`) — 187 features, monitoring system

The SDK and canonical experiments are consistent (verified: identical values on test data). The smell-monitor diverges in feature names, algorithms, and global features. This document catalogs every divergence and specifies the canonical behavior.

## Canonical Implementation

**The SDK (`opensmell/opensmell/mox/features.py`) is the canonical implementation.** All other implementations must match it. Reasons:

- It is the production codepath (web, desktop, CLI all route through it)
- It has the most complete feature set (187 features)
- It has the rigorous `_r0_from_contract` fallback chain
- It has proper calibration integration via `calibration_for_channel`
- The canonical experiments already match it exactly

## Divergence Catalog

### D1: Feature Naming

| Feature | SDK / Canonical | Smell-monitor | Status |
|---------|----------------|---------------|--------|
| Device-agnostic amplitude | `ch{ch}_da_relative_amplitude` | `ch{ch}_delta_r_r0` | **DIVERGENT** |
| Device-agnostic direction | `ch{ch}_da_direction` | `ch{ch}_direction` | **DIVERGENT** |
| Device-agnostic rise_time | `ch{ch}_da_rise_time` | `ch{ch}_rise_time` | **DIVERGENT** |
| Device-agnostic decay_time | `ch{ch}_da_decay_time` | `ch{ch}_decay_time` | **DIVERGENT** |
| Device-agnostic auc | `ch{ch}_da_auc` | `ch{ch}_auc` | **DIVERGENT** |
| Device-agnostic endpoint_delta | `ch{ch}_da_endpoint_delta` | `ch{ch}_endpoint_delta` | **DIVERGENT** |
| Absolute raw_resistance | `ch{ch}_abs_raw_resistance` | `ch{ch}_raw_resistance` | **DIVERGENT** |
| Absolute baseline_resistance | `ch{ch}_abs_baseline_resistance` | `ch{ch}_baseline` | **DIVERGENT** |
| Absolute voltage | `ch{ch}_abs_voltage` | `ch{ch}_voltage` | **DIVERGENT** |
| Absolute calibrated_conc | `ch{ch}_abs_calibrated_concentration` | `ch{ch}_calibrated_conc` | **DIVERGENT** |
| Temporal hf_transient | `ch{ch}_temp_hf_transient` | `ch{ch}_high_freq_transient` | **DIVERGENT** |
| Temporal oscillation_freq | `ch{ch}_temp_oscillation_freq` | `ch{ch}_oscillation_freq` | **DIVERGENT** |
| Temporal oscillation_amp | `ch{ch}_temp_oscillation_amp` | `ch{ch}_oscillation_amp` | **DIVERGENT** |
| Temporal response_latency | `ch{ch}_temp_response_latency` | `ch{ch}_response_latency` | **DIVERGENT** |
| Health drift_rate | `ch{ch}_health_drift_rate` | `ch{ch}_drift_rate` | **DIVERGENT** |
| Health sensitivity_decay | `ch{ch}_health_sensitivity_decay` | `ch{ch}_sensitivity_decay` | **DIVERGENT** |
| Health noise_floor | `ch{ch}_health_noise_floor` | `ch{ch}_noise_floor` | **DIVERGENT** |
| Health hysteresis | `ch{ch}_health_hysteresis` | `ch{ch}_hysteresis` | **DIVERGENT** |
| Hardware circuit_response | `ch{ch}_hw_circuit_response` | `ch{ch}_circuit_response` | **DIVERGENT** |
| Hardware thermal_profile | `ch{ch}_hw_thermal_profile` | `ch{ch}_thermal_profile` | **DIVERGENT** |
| Hardware adc_noise | `ch{ch}_hw_adc_noise` | `ch{ch}_adc_noise` | **DIVERGENT** |
| Saturation index | `ch{ch}_advanced_saturation_index` | `ch{ch}_saturation_index` | **DIVERGENT** |
| Decay tau1 | `ch{ch}_decay_tau1` | `ch{ch}_tau1` | **DIVERGENT** |
| Decay tau2 | `ch{ch}_decay_tau2` | `ch{ch}_tau2` | **DIVERGENT** |
| Decay tau3 | `ch{ch}_decay_tau3` | `ch{ch}_tau3` | **DIVERGENT** |
| Decay a1 | `ch{ch}_decay_a1` | `ch{ch}_a1` | **DIVERGENT** |
| Decay a2 | `ch{ch}_decay_a2` | `ch{ch}_a2` | **DIVERGENT** |
| Decay a3 | `ch{ch}_decay_a3` | `ch{ch}_a3` | **DIVERGENT** |
| Selectivity | `sel_ratio_ch{i}_ch{j}` | `selectivity_ch{i}_ch{j}` | **DIVERGENT** |
| Global max | `global_max_delta_ratio` | `global_max_amplitude` | **DIVERGENT** |
| Global mean | `global_mean_delta_ratio` | `global_mean_response` | **DIVERGENT** |
| Global active channels | `global_n_active_channels` | (missing) | **DIVERGENT** |
| Global total AUC | `global_total_auc` | (missing) | **DIVERGENT** |
| Global entropy | (missing) | `global_array_entropy` | **DIVERGENT** |
| Global std | (missing) | `global_std_response` | **DIVERGENT** |

### D2: Algorithm Differences

#### Rise Time
- **SDK/Canonical**: 10%→90% of full span crossing (threshold on raw series)
- **Smell-monitor**: baseline→90% of peak (uses normalized data, different baseline estimation)

#### Direction
- **SDK/Canonical**: Compares `max(series - R0)` vs `min(series - R0)` — which deviation from baseline is larger
- **Smell-monitor**: `data[-1] > data[0]` — compares final vs initial value of normalized signal

#### AUC
- **SDK/Canonical**: `np.trapezoid(np.abs(norm))` — integral of absolute normalized signal
- **Smell-monitor**: `np.trapz(np.abs(data)) / len(data)` — integral divided by length (normalizes by duration)

#### Endpoint Delta
- **SDK/Canonical**: `(series[-1] - R0) / R0` — relative to baseline
- **Smell-monitor**: `data[-1] - data[0]` — difference of first and last normalized values

#### Drift Rate
- **SDK/Canonical**: `(mean(series[-10:]) - R0) / R0` — end-vs-baseline ratio
- **Smell-monitor**: `stats.linregress(time, data).slope` — linear regression slope

#### Noise Floor
- **SDK/Canonical**: `std(series[:r0_samples]) / R0` — baseline standard deviation normalized by R0
- **Smell-monitor**: Welch PSD high-frequency power (>5 Hz) — spectral noise estimate

#### Hysteresis
- **SDK/Canonical**: Trapezoid area difference between adsorption and desorption paths, normalized by adsorption area
- **Smell-monitor**: Interpolation-aligned mean absolute difference between rising and falling edges

#### Saturation Index
- **SDK/Canonical**: `current_response / (current_response + noise_floor * 10)` — noise-floor-scaled estimator
- **Smell-monitor**: `peak / (peak * 2)` — initial-slope-based theoretical maximum estimator

#### Decay Fitting
- **SDK/Canonical**: Subtracts endpoint first (`y = recovery - recovery[-1]`), fits bi-exponential by default
- **Smell-monitor**: Fits raw decay data, tries bi-exponential then tri-exponential

#### Thermal Profile
- **SDK/Canonical**: `std(series)` — standard deviation of entire signal
- **Smell-monitor**: Exponential fit rate coefficient — `curve_fit(lambda t, a, b: a*exp(b*t), ...)`

#### ADC Noise
- **SDK/Canonical**: `std(series - smooth)` — residual std after 5-sample moving average
- **Smell-monitor**: `min(diff(sort(unique(round(data, 6)))))` — minimum quantization step

### D3: Global Features

| Feature | SDK / Canonical | Smell-monitor |
|---------|----------------|---------------|
| `global_max_delta_ratio` | `max(relative_amplitudes)` across active channels | — |
| `global_mean_delta_ratio` | `mean(relative_amplitudes)` across active channels | — |
| `global_n_active_channels` | count of channels with `relative_amplitude > 0` and `!is_dead` | — |
| `global_total_auc` | `sum(auc)` across active channels | — |
| `global_max_amplitude` | — | `max(abs(normalized_data))` |
| `global_mean_response` | — | `mean(normalized_data)` |
| `global_std_response` | — | `std(normalized_data)` |
| `global_array_entropy` | — | Shannon entropy of channel peak ratios |

## Required Actions

### Immediate (P0)
1. **Smell-monitor must adopt SDK naming conventions** — all 34 feature names must match
2. **Smell-monitor must adopt SDK algorithms** — all 12 algorithm differences must be resolved
3. **Smell-monitor must adopt SDK global features** — replace its 4 global features with SDK's 4

### Short-term (P1)
4. **Add a cross-validation test** that runs all three extractors on the same data and asserts feature equality
5. **Document the 145→187 gap** — the canonical experiments use 145 features; the full ablation must use 187

### Medium-term (P2)
6. **Unify extraction into a single module** — eliminate the duplication entirely
7. **Add feature-level property tests** (e.g., `relative_amplitude >= 0`, `direction ∈ {-1, 0, 1}`)

## Verification Script

```python
"""Verify all three extractors produce identical features on the same data."""
import numpy as np
from opensmell.opensmell.mox.features import extract_all_framework_features as sdk_extract
from interoperability.canonical_experiments.framework_features import extract_all_framework_features as exp_extract
# from smell_monitor.features import MOXFeatureExtractor  # After alignment

np.random.seed(42)
test_data = np.random.randn(100, 6) * 100 + 1000

sdk_feats = sdk_extract(test_data, r0_samples=15, sr=10)
exp_feats = exp_extract(test_data, r0_samples=15, sr=10)

# SDK and canonical experiments must match exactly
common_keys = set(sdk_feats.keys()) & set(exp_feats.keys())
for k in common_keys:
    sv, ev = sdk_feats[k], exp_feats[k]
    if isinstance(sv, (int, float)):
        assert abs(sv - ev) < 1e-6, f"Mismatch on {k}: SDK={sv} EXP={ev}"

print(f"All {len(common_keys)} common features match between SDK and canonical experiments.")
```
