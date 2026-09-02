# Exhaustive Feature Catalog

Every feature in the canonical 187-dimensional SDK extractor, with:
- **Definition**: exact mathematical form
- **Physics**: why it captures what it captures
- **Device-transferability**: whether it survives cross-device transfer
- **Falsifiability**: how to test whether it carries unique information
- **Status**: canonical / flagged in the literature

Each feature is argued from first principles. Features that cannot be given a
physical interpretation, or whose interpretation depends on hardware, are
flagged as **hardware-bound** and are candidates for removal via ablation.

---

## Conventions

- `series`: raw resistance time-series for one channel, length N
- `R0`: baseline resistance = median of first `r0_samples` (default 15) finite samples
- `norm_i = (series_i - R0) / R0`: the normalized series
- `sr`: sampling rate in Hz (default 10)
- `peak_idx = argmax(|series - R0|)`: index of peak deviation from baseline
- `t_i = i / sr`: time of sample i in seconds

A channel is **dead** if it has <2 finite samples or `std(finite)/R0 < 0.001`.

---

# Dimension 1 — Device-Agnostic (per-channel)

These are the transferable features. They depend only on the gas-surface
interaction, not on hardware. **This is the dimension that enables
interoperability.**

## 1.1 `relative_amplitude` (`ch{X}_da_relative_amplitude`)

- **Definition**: `max(|delta_max|, |delta_min|) / R0` where
  `delta_max = max(series) - R0`, `delta_min = min(series) - R0`.
- **Physics**: The peak fractional change in resistance. For an n-type MOX
  sensor, exposure to a reducing gas lowers resistance proportionally to
  surface coverage of adsorbates. This is the primary "how much does the gas
  interact" metric. In the power-law regime `Rs/R0 = a·C^b`, the amplitude
  carries the concentration signal folded through the sensor sensitivity `b`.
- **Device-transferability**: **HIGH** — `R0` cancels the loading-resistor and
  supply-voltage dependence (Theorem 1). What remains is `a·C^b` which is
  sensor-chemistry specific (`a`, `b` are the sensor constants).
- **Falsifiability**: Across two devices with identical sensor models, the
  same sample should give the same relative amplitude (up to drift). If it
  does not, the feature is not device-agnostic.
- **Known limit**: `b` differs between sensor materials, so amplitude does NOT
  transfer across *different* sensor models. This is precisely the
  cross-device failure (Experiment 7).

## 1.2 `direction` (`ch{X}_da_direction`)

- **Definition**: `+1` if the larger deviation is positive (resistance rises
  from baseline), `-1` if larger deviation is negative (resistance falls).
  0 for dead channels.
- **Physics**: Whether the analyte is an oxidizing agent (raises resistance
  for n-type MOX) or reducing agent (lowers resistance). For an n-type sensor,
  reducing gases donate electrons → lower resistance → `direction = -1`;
  oxidizing gases withdraw electrons → higher resistance → `direction = +1`.
- **Device-transferability**: **HIGH** — the sign of the response is a
  chemical property, not a hardware property, for a fixed sensor doping. (An
  n-type vs p-type sensor inverts it — that is a sensor-chemistry constant.)
- **Falsifiability**: Same functional group should give consistent sign on
  the same sensor material across devices.

## 1.3 `rise_time` (`ch{X}_da_rise_time`)

- **Definition**: Time (s) from the 10% to 90% crossing of the full span of
  the response, in the direction of the peak. `-1` if not computable.
- **Physics**: The adsorption kinetics. The time to reach 90% of the peak
  deviation reflects the adsorption rate constant `k_ads`. For Langmuir
  adsorption, coverage `θ(t) = θ_eq(1 - e^{-k t})`, so the rise time scales as
  `≈ ln(9)/k`. Different molecules bind with different rate constants due to
  their size, polarity, and electron affinity.
- **Device-transferability**: **HIGH** — a rate constant is intrinsic to the
  molecule-sensor-surface system, independent of readout electronics.
- **Falsifiability**: Two devices with the same sensor model should give the
  same rise time for the same sample.
- **Note**: Requires the full `baseline → exposure → recovery` protocol. On
  SmellNet (no explicit phases), this feature is unreliable/zero — the
  ablation study flagged this.

## 1.4 `decay_time` (`ch{X}_da_decay_time`)

- **Definition**: Time (s) from the 90% to 10% crossing of the full span
  *after the peak* (desorption phase). `-1` if not computable.
- **Physics**: The desorption kinetics. The time to return toward baseline
  reflects the desorption rate constant `k_des` and the activation energy of
  the binding site. The literature shows desorption-phase features are
  **independent of the sensor-sample reaction time** — they carry unique
  chemical information.
- **Device-transferability**: **HIGH** (same argument as rise time).
- **Falsifiability**: Same sample, same sensor model, different devices →
  same decay time.

## 1.5 `auc` (`ch{X}_da_auc`)

- **Definition**: `trapezoid(|norm|)` — integral of the absolute normalized
  series over the whole recording.
- **Physics**: The total integrated exposure `∫|R-R0|/R0 dt`. This folds
  together amplitude AND duration — a proxy for total surface-molecule
  interaction. For a constant concentration, AUC ∝ amplitude × exposure time.
- **Device-transferability**: **HIGH** (product of two transferable
  quantities, in normalized units).
- **Caveat**: Depends on recording window length; only comparable across
  recordings of equal duration. Flagged for this reason.

## 1.6 `endpoint_delta` (`ch{X}_da_endpoint_delta`)

- **Definition**: `(series[-1] - R0) / R0` — normalized value of the final
  sample.
- **Physics**: Whether the sensor returned to baseline by the end of the
  recording. Nonzero endpoint indicates residual adsorption (incomplete
  desorption) — relevant for detecting sensor "memory" / contamination.
- **Device-transferability**: **PARTIAL** — depends on whether the recording
  protocol fully flushes the chamber. A protocol-controlled quantity, hence
  weakly device-dependent.
- **Falsifiability**: Poorly flushed chambers produce nonzero endpoints on
  all devices; the meaning is protocol-bound.

---

# Dimension 2 — Absolute (per-channel)

These require device-specific calibration constants. They are **not**
device-transferable as-is but enable concentration estimation *after*
per-rig calibration.

## 2.1 `raw_resistance` (`ch{X}_abs_raw_resistance`)

- **Definition**: `mean(series[-10:])` (mean of last 10 samples).
- **Physics**: The absolute sensor resistance at (near) the endpoint. Absolute
  resistance depends on circuit (R_L) and sensor material — **hardware-bound**.
- **Device-transferability**: **NONE** without calibration.

## 2.2 `baseline_resistance` (`ch{X}_abs_baseline_resistance`)

- **Definition**: `R0`.
- **Physics**: The clean-air resistance. Used as the denominator of all
  normalizations and as the reference for calibration.
- **Device-transferability**: **NONE** (device-specific).

## 2.3 `voltage` (`ch{X}_abs_voltage`)

- **Definition**: `raw_resistance` (placeholder — requires Vcc and R_L).
- **Physics**: The raw voltage read. **Hardware-bound** by definition.
- **Device-transferability**: **NONE**.
- **Note**: This is a placeholder. In the canonical SDK it is set equal to raw
  resistance. Flagged as low-information / ablation candidate.

## 2.4 `calibrated_concentration` (`ch{X}_abs_calibrated_concentration`)

- **Definition**: Inverted power law `C = (rr / a)^(1/b)` with nominal
  `(a, b) = (1.0, -0.5)`, `rr = raw_resistance / R0`.
- **Physics**: A concentration estimate from the power-law response. Only
  meaningful when `(a, b)` are the true sensor constants from per-rig
  reference-point calibration (Theorem 3).
- **Device-transferability**: **NONE** with nominal constants; **HIGH** after
  proper per-rig calibration.
- **Falsifiability**: With true `(a, b)`, predicted C should match known ppm
  of a controlled gas. This is the reference-point calibration path.

---

# Dimension 3 — Temporal (per-channel)

## 3.1 `hf_transient` (`ch{X}_temp_hf_transient`)

- **Definition**: `mean(|diff(series)|)` — mean absolute first difference.
- **Physics**: A crude high-frequency activity measure. Captures the rate of
  change during fast transients and noise. **Partially hardware-bound** (noise
  level depends on ADC).
- **Device-transferability**: **LOW-MEDIUM** — the fast-transient signal is
  chemical, but the noise component is hardware.
- **Caveat**: Confounds true fast transients with electronic noise.
  Ablation candidate: may be redundant with `noise_floor`.

## 3.2 `oscillation_freq` (`ch{X}_temp_oscillation_freq`)

- **Definition**: Dominant frequency of the detrended series (periodogram
  peak, excluding DC), in Hz.
- **Physics**: Oscillatory behavior sometimes observed in MOX responses due to
  surface-reaction cycling or gas-turbulence coupling. Meaningful only if a
  stable oscillation exists.
- **Device-transferability**: **MEDIUM**.
- **Caveat**: Often dominated by noise → unreliable. Ablation candidate.

## 3.3 `oscillation_amp` (`ch{X}_temp_oscillation_amp`)

- **Definition**: `sqrt(PSD at dominant freq)` — amplitude of the dominant
  oscillation.
- **Physics**: Strength of the dominant oscillatory component.
- **Device-transferability**: **MEDIUM**.
- **Falsifiability**: Requires non-noise oscillation; weak on clean data.

## 3.4 `response_latency` (`ch{X}_temp_response_latency`)

- **Definition**: Time (s) for the signal to first exceed `3σ` of the
  baseline window (first 10 samples), measured from sample 10.
- **Physics**: How quickly the sensor first detects a significant signal.
  Related to gas delivery delay + adsorption onset. **Protocol-bound** (depends
  on how fast gas reaches the chamber, which is hardware/plumbing-specific).
- **Device-transferability**: **LOW** — strongly hardware/protocol dependent.

---

# Dimension 4 — Health (per-channel)

## 4.1 `drift_rate` (`ch{X}_health_drift_rate`)

- **Definition**: `(mean(series[-10:]) - R0) / R0`.
- **Physics**: Within-recording baseline drift — a proxy for slow sensor drift.
  The literature recommends tracking drift over days/weeks (multi-recording),
  which this single-recording proxy does not do. **Flagged as placeholder**.
- **Device-transferability**: **MEDIUM** — normalized, but slow-drift
  measurement requires long baselines.
- **Caveat**: Confounds real drift with incomplete desorption.
  Ablation candidate (weak signal within a single recording).

## 4.2 `sensitivity_decay` (`ch{X}_health_sensitivity_decay`)

- **Definition**: Currently `0.0` (placeholder — requires multi-recording to
  measure true degradation).
- **Physics**: Whether the sensor responds less over months of use. **Cannot
  be measured from a single recording.** Flagged as placeholder.
- **Device-transferability**: N/A (not yet implemented).
- **Caveat**: Strong **ablation candidate** — currently always zero and
  contributes nothing.

## 4.3 `noise_floor` (`ch{X}_health_noise_floor`)

- **Definition**: `std(series[:r0_samples]) / R0`.
- **Physics**: Baseline RMS noise normalized by R0. A low noise floor means a
  stable, healthy sensor; high noise indicates degradation or EMI.
- **Device-transferability**: **PARTIAL** — normalized to R0, but noise level
  depends on ADC quality (hardware).
- **Falsifiability**: Same sensor model on different devices → different noise
  floors if ADCs differ. Used to detect sensor health, not chemistry.

## 4.4 `hysteresis` (`ch{X}_health_hysteresis`)

- **Definition**: `|∫ads - ∫des| / max(∫ads, ε)` where adsorption path is
  `|series[:peak]-R0|` and desorption is `|series[peak:]-R0|`.
- **Physics**: Asymmetry between the adsorption and desorption paths (the
  loop area). Large hysteresis indicates irreversible trapping — a real,
  physically meaningful health signal.
- **Device-transferability**: **HIGH** — it is a normalized ratio of
  integrated responses.
- **Falsifiability**: Irreversible-binding analytes produce large hysteresis
  on all devices of the same sensor type.

---

# Dimension 5 — Hardware (per-channel)

## 5.1 `circuit_response` (`ch{X}_hw_circuit_response`)

- **Definition**: `mean(series)` — channel mean.
- **Physics**: **Hardware-bound**. The mean raw signal is dominated by the
  absolute resistance, which is circuit- and material-dependent. The name
  overstates what this measures (a true circuit characterization would require
  a step response + RC fit).
- **Device-transferability**: **NONE**.
- **Caveat**: Placeholder. **Strong ablation candidate** — redundant with
  `raw_resistance`.

## 5.2 `thermal_profile` (`ch{X}_hw_thermal_profile`)

- **Definition**: `std(series)`.
- **Physics**: **Hardware-bound**. Total signal variability, which mixes
  chemical response, noise, and thermal drift. Not a clean thermal
  measurement. **Flagged as placeholder** — smell-monitor instead fits an
  exponential drift rate (divergent, see IMPLEMENTATION_ALIGNMENT.md).
- **Device-transferability**: **NONE**.
- **Caveat**: **Ablation candidate** (redundant with `noise_floor` + amplitude).

## 5.3 `adc_noise` (`ch{X}_hw_adc_noise`)

- **Definition**: `std(series - smooth5)` — residual std after 5-sample moving
  average.
- **Physics**: **Hardware-bound** — the quantization/electronic noise of the
  ADC. Useful for hardware QA, not chemistry.
- **Device-transferability**: **NONE** (by definition — it measures the ADC).
- **Falsifiability**: Same sensor, better ADC → lower adc_noise.
  Useful for detecting bad ADC channels.

---

# Advanced (per-channel)

## 6.1 `saturation_index` (`ch{X}_advanced_saturation_index`)

- **Definition**: `current_response / (current_response + noise_floor*10)`,
  clamped to [0,1]; 0 if `current_response < 2*noise_floor`.
- **Physics**: Approximate Langmuir surface-coverage estimate — how close the
  response is to full surface saturation. In the Langmuir isotherm
  `θ = KP/(1+KP)`, high concentration → θ→1 → saturated. This feature tracks
  proximity to saturation.
- **Device-transferability**: **HIGH** — normalized response ratio.
- **Falsifiability**: Elevated concentrations of a strongly-adsorbing gas on
  any device of the same sensor type give saturation_index near 1.
- **Note**: The estimator is empirical (uses max observed response as a proxy
  for saturation capacity). The smell-monitor uses a different (divergent)
  estimator.

## 6.2 `decay_tau1`, `decay_a1` (`ch{X}_decay_tau1`, `ch{X}_decay_a1`)

- **Definition**: Fast exponential decay constant and amplitude from fitting
  `y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + c` (after endpoint subtraction)
  to the recovery phase.
- **Physics**: Rate constant of the fast desorption process. MOX surfaces have
  heterogeneous binding sites: weakly adsorbed (labile) species desorb
  quickly (tau1 ~ 1-3s); strongly bound species desorb slowly (tau2 ~ 10-30s).
  The fast component is dominated by physisorbed species.
- **Device-transferability**: **HIGH** — a desorption rate constant, intrinsic
  to the molecule-surface system.
- **Falsifiability**: Same analyte, same sensor chemistry, different devices →
  same tau1.

## 6.3 `decay_tau2`, `decay_a2` (`ch{X}_decay_tau2`, `ch{X}_decay_a2`)

- **Definition**: Slow exponential decay constant and amplitude (bi-exponential
  model).
- **Physics**: The slow desorption component — chemisorbed / strongly-bound
  species. The ratio `tau2/tau1` is a direct measure of surface heterogeneity.
- **Device-transferability**: **HIGH**.

## 6.4 `decay_tau3`, `decay_a3` (`ch{X}_decay_tau3`, `ch{X}_decay_a3`)

- **Definition**: Third exponential component (tri-exponential, opt-in only).
- **Physics**: The tri-exponential is **over-parameterized** for MOX recovery —
  the three components are not uniquely identifiable, causing unreliable
  fits. **Flagged in the SDK docstring** as opt-in and potentially
  nondeterministic.
- **Device-transferability**: **LOW** — identifiability problems make this
  unreliable.
- **Caveat**: **Ablation candidate** — the default bi-exponential fit leaves
  these at `-1`; they carry no information in the default path.

---

# Cross-Channel — Selectivity Ratios

## 7.x `sel_ratio_ch{i}_ch{j}` (15 features for 6 channels)

- **Definition**: `relative_amplitude[i] / relative_amplitude[j]` for active
  channels; 0 if either is inactive or the denominator is 0.
- **Physics**: The **relative sensitivity fingerprint** of the array to a
  given analyte. Different sensors (different MOX materials) respond with
  different relative strengths to different molecules; the pattern of ratios
  is the array's chemical fingerprint. This is the most discriminative
  dimension for substance identification.
- **Device-transferability**: **PARTIAL** — ratios of amplitudes cancel `R0`,
  but the absolute sensitivities `a` differ per sensor material. Two devices
  with the *same sensor models* give matching fingerprints; devices with
  *different* sensor models give different fingerprints (Experiment 7).
- **Falsifiability**: The canonical ablation study found removing these
  causes the largest accuracy drop (−4.9%). This is the single most
  information-dense dimension.

---

# Global

## 8.1 `global_max_delta_ratio`

- **Definition**: `max(relative_amplitude)` over active channels.
- **Physics**: The strongest single-channel response — a coarse "how strong
  overall" metric.
- **Device-transferability**: HIGH.

## 8.2 `global_mean_delta_ratio`

- **Definition**: `mean(relative_amplitude)` over active channels.
- **Physics**: Average response strength across the array.
- **Device-transferability**: HIGH.

## 8.3 `global_n_active_channels`

- **Definition**: Count of channels with `relative_amplitude > 0` and not dead.
- **Physics**: How many sensors the analyte triggers — a coarse selectivity /
  breadth metric.
- **Device-transferability**: HIGH.

## 8.4 `global_total_auc`

- **Definition**: `sum(auc)` over active channels.
- **Physics**: Total integrated exposure across the array.
- **Device-transferability**: HIGH.

---

# Ablation Candidates (to test for removal)

The following features are **flagged as low-value** and should be tested in
the full ablation study for removal:

1. **`voltage`** (`abs_voltage`) — placeholder, equal to raw_resistance;
   fully redundant. **Remove if ablation shows zero contribution.**
2. **`circuit_response`** (`hw_circuit_response`) — placeholder channel mean;
   redundant with raw_resistance. **Remove if zero contribution.**
3. **`thermal_profile`** (`hw_thermal_profile`) — placeholder std; mixes
   noise+signal, not a true thermal measure. **Remove if zero contribution.**
4. **`sensitivity_decay`** (`health_sensitivity_decay`) — always 0 in current
   impl; **cannot contribute** until multi-recording measurement added.
   **Remove.**
5. **`oscillation_freq`/`oscillation_amp`** — noise-dominated on clean data.
   **Remove if zero contribution.**
6. **`decay_tau3`/`decay_a3`** — tri-exponential not identifiable / -1 in
   default path. **Remove.**

## The Core Transferable Set

If ablation confirms the theory, the **minimal device-transferable core** is:

- Per-channel: `relative_amplitude`, `direction`, `rise_time`, `decay_time`,
  `hysteresis`, `decay_tau1`, `decay_tau2`, `saturation_index`
- Cross-channel: all `sel_ratio_*`
- Global: `max`, `mean`, `n_active`

Everything else either requires calibration (Dimension 2), is protocol-bound
(temporal latency), or is hardware-bound (Dimension 5) — contribute to
discrimination only within a single device, and should be dropped for
cross-device work.
