# Hardware-Aware Modeling: Sensor Digital Twins

## The Core Idea

The zero-shot cross-device transfer failure (`canonical 07`, `alignment 1-7`)
is not because features are bad — it's because **sensor constants `(a, b)`
differ between devices**, and those constants stay embedded in the feature
distributions even after `Rs/R0` normalization (Theorem 2).

The universal-model strategy is **not** to pretend hardware differences don't
exist. It is to **explicitly feed them to the model as conditional
information**. This converts a pathological unit-difference into a
well-conditioned input feature.

## The Two Linearization Steps

### Step 1: Calibrate the concentration law (per-sensor)

Each MOX channel obeys, in its linear region:

```
rs_r0 = a · C^b
```

- `a`: sensor sensitivity (geometry, doping, operating temperature)
- `b`: sensor exponent (~ -0.3 to -0.7 for common MOX; concentration-dependent)

These two numbers are the sensor's **identity parameters**. They are what
fail to transfer.

### Step 2: Concatenate sensor parameters into the feature vector

For each recording, build the **hardware-aware feature vector** as:

```
x_hwa = [ physico_features (187-dim)  |  sensor_params (a, b per channel) ]
```

The model learns: "given that *this* sensor has constants (a, b), this
feature pattern maps to this chemical class." This is calibration transfer at
the feature level — higher-level than signal-level correction, and it is what
makes a single model usable across many devices.

## Sensor Profile Schema

The `sensor_profiles/` folder stores machine-readable "digital twins." Each
profile records everything needed to construct the `sensor_params` portion of
the feature vector and, critically, which parts of the recording protocol the
device follows.

Key fields (see `TEMPLATE.json`):

| Field | Purpose for hardware-aware modeling |
|-------|-------------------------------------|
| `channels[].nominal_sensitivity_a` | the `a` parameter |
| `channels[].nominal_exponent_b` | the `b` parameter |
| `channels[].doping_type` | p/n-type → sign convention for `direction` |
| `channels[].operating_surface_temp_c` | affects kinetics (adsorption/desorption rates) |
| `circuit.supply_voltage_vcc` | confirmed cancelled by Rs/R0 (Theorem 1) |
| `circuit.load_resistor_rl_ohm` | confirmed cancelled by Rs/R0 (Theorem 1) |
| `circuit.adc_bits` | diagnostic: noise floor expectation |
| `recording_protocol.*` | which features are valid vs protocol-dependent |
| `gas_delivery.*` | delivery latency, affects `response_latency` |

## From Individual Profiles to a Universal Model

1. Each contributor adds their `sensor_profile.json` + recordings.
2. `load_profiles()` merges all profiles; each recording is tagged with its
   device's parameters.
3. Feature vector = physicochemical (187) ⊕ sensor params.
4. Train one supervised model (functional-group classification first, then
   substance) across all devices.
5. **Validate on an entirely held-out device** (device-LOO) — not just
   held-out sessions.

The device-LOO validation is the milestone that proves hardware-agnostic
generalization, versus the session-LOO which only proves within-device
stability (canonical 01).

## Why This Is Milestone 2-3 (not 1)

Before building the universal model, we need the **feature foundation**
(Milestone 1) proven on protocol data: the set of physicochemical features
that reliably separate functional groups. Without that, concatenating sensor
params just decorates noise.

## Open Questions / Research Needs

1. **Exponent `b` uncertainty** — `b` is concentration-dependent and drifts;
   are the *nominal* `(a, b)` from datasheets good enough, or must `(a, b)`
   be measured per-rig? (Per-rig calibration is Theorem 3.)
2. **Temperature dependence** — kinetics features depend on operating
   temperature, which drifts with heater voltage. Does the profile need a
   temperature-correction term?
3. **Cross-modality** — can the same `[features | sensor_params]` template
   extend to electrochemical / MIRIS (optical) modalities, where the
   "sensor params" are different (e.g. spectrum resolution, IR source)?
   This is the long-term modality-agnostic goal.

## Files

- `sensor_profiles/templates/TEMPLATE.json` — blank template for contributors
- `sensor_profiles/templates/opensmell_reference_rig.json` — populated example
- `docs/universal_model.md` — architecture for the multi-device model
