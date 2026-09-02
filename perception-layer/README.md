# Research: Hardware-Agnostic Olfactory Perception Layer

> The frontier direction: turn the OpenSmell `.osmell` recording protocol into
> a **physics-based, hardware-agnostic perception layer** — first for MOX
> electronic noses, later for any gas-sensing modality (electrochemical,
> MIRIS/optical).

This directory is the **research spine** for the interoperability agenda
described in *"Towards Interoperable Digital Olfaction: A Modular Feature
Framework for Electronic Noses."* It complements (does NOT duplicate) the
`canonical_experiments/` and `alignment_experiments/` folders, which proved the
*bounds*; this directory pushes the *actual goal*: a model that generalizes
across devices because it reasons about **features and sensor properties**
rather than raw signals.

## The One-Line Thesis

> Sensor responses distinguish devices because `(a, b)` constants differ;
> extract the physicochemical features that survive `Rs/R0` normalization,
> feed the device's `(a, b)` profile to the model as conditioning input, and
> a single universal model can classify odors across arbitrarily different
> hardware.

## How This Directory Sits With Existing Work

| Existing | Role | This directory complements with |
|----------|------|----------------------------------|
| `canonical/01-02` | Within-device session stability (proved) | — |
| `canonical/07`, `alignment/1-7` | Proved zero-shot transfer FAILS | Why: Theorem 2; and the fix (device-LOO + sensor params) |
| `canonical/05` | Dimension-1-only ablation | **Full 187-feature ablation** with protocol + role gating |
| SDK `features.py` | Canonical extractor | Consumed directly (never reimplemented) so results reflect production |

## The Two Critical Scientific Guardrails

These are the methodological corrections this directory adds over the naive
"build more features" approach:

### 1. Protocol-Dependence
Features that capture the **recovery/desorption phase** (decay times, tau
constants, hysteresis) are only meaningful on `baseline → exposure →
recovery` data. Recordings without a genuine recovery return make those
features degenerate, and their ablation verdicts are invalid. The ablation
script **detects** this and marks such verdicts as invalid; `phenotype.py`
gates `A2.recovery` the same way (reported `unknown`, never forced).
See `full_ablation/README.md`.

> Correction (measured, not assumed): SmellNet recordings are NOT all ramp-only.
> Of 68 inspected, 51 contain a genuine recovery return and 17 are ramp-only;
> the set also contains at least one saturated (dead) channel. Any protocol
> gate must therefore be decided *per recording*, not per dataset. See
> `results/validation_piecewise.json`.

### 2. Feature Roles
Not all features discriminate substances. Some are **diagnostic** (sensor
health, poisoning, ADC QA) and must be validated by *responsiveness to
faults*, not by classification accuracy. See `feature_catalog/FEATURE_ROLES.md`.

Conflating these two is the fastest way to draw a wrong conclusion.

## Contents

```
research/
├── README.md                                # This file
├── feature_catalog/
│   ├── CATALOG.md                           # All 187 features, argued from physics
│   └── FEATURE_ROLES.md                     # discrimination vs diagnostic vs absolute
├── full_ablation/
│   └── README.md                            # Full-187 ablation methodology + caveats
├── ontology/
│   ├── PHENOTYPE_ONTO.md                    # Measured-phenotype physics-axes ontology (review this)
│   ├── normalize.py                         # Topology-aware ADC->resistance (hard-to-vary sign)
│   ├── detect.py                            # Auto-detect response convention + single user binary (no hardware form)
│   ├── phenotype.py                         # A1/A2/A3 evidence-chain verdicts from recordings
│   ├── percept_map.py                       # Measured A3 -> response-type clusters (honest, k=2)
│   ├── response_type.py                     # Assign a new A3 fingerprint to a validated response-type cluster
│   └── percept_guidance.py                  # Response-type -> percept label + use-case guidance (scoped-down)
├── experiments/
│   ├── run_full_ablation.py                 # Runnable full ablation (consumes SDK)
│   ├── validate_perception.py               # Piecewise cross-dataset validation (Vergara/Hu/SmellNet)
│   └── falsify_four_category.py             # Falsification attempt on the 4-category claim (see docs/)
├── docs/
│   ├── OPEN_QUESTIONS.md                    # Decision-theoretic questions + community data requests
│   ├── DEVICE_INVARIANCE_DISTINCTION.md     # response-type vs cross-device vector transfer
│   └── FALSIFYING_FOUR_CATEGORY.md          # Why the 4-category scheme is not validated
├── hardware_aware/
│   ├── README.md                            # Sensor digital-twin strategy
│   ├── universal_model.md                   # Milestones 1-5 roadmap
│   └── sensor_profiles/templates/           # Machine-readable device profiles
├── docs/
│   └── IMPLEMENTATION_ALIGNMENT.md          # 3-extractor divergence catalog
└── results/
    ├── full_ablation.json                   # Latest ablation output
    ├── validation_piecewise.json            # Ontology piecewise validation (Vergara/Hu/SmellNet)
    └── percept_map_A3.json                  # A3 fingerprint clustering -> percept hypotheses (A2 pending protocol data)
```

## Piecewise Validation Results (measured, not assumed)

No single dataset has both diverse controlled chemistry AND raw dynamic curves,
so each axis is validated on the dataset(s) that actually support it
(`experiments/validate_perception.py` → `results/validation_piecewise.json`):

| Axis / claim | Dataset | Finding |
|---|---|---|
| **A3** cross-sensor selectivity fingerprint | UCI Gas Drift (Vergara) 13,910 samples × 16 sensors × 10 batches | **Device-invariant**. k=2 (silhouette 0.598, data-driven). Two homogeneous *response-type* clusters: {Acetone/Ethanol/Acetaldehyde/Toluene} (impurity 0.03) and {Ethylene/Ammonia} (impurity 0.03). **But gas-level purity only 0.405** — the array resolves *response type*, not individual gases. Same k=2 on batch-1-only and batches-2-10-only |
| **A1** redox valence | Vergara | All six targets are known reducing VOCs on n-type MOX (invariant); caveat: Vergara gives pre-extracted features, no raw curves for absolute sign confirmation |
| **A2** recovery honesty | SmellNet (68 recordings) | Must be gated **per recording**: 51 have genuine recovery, 17 are ramp-only; ≥1 dead (saturated) channel — `phenotype.py` excludes dead channels and reports `A2.recovery=unknown` where absent |
| **Robustness/negative** | UCI Home Activity (Hu 2016) | `not_valid_for_A1/A2_dynamics`: within-class spread (banana 5.7, wine 7.1) exceeds banana–wine distance (1.2); 3.6-s drift-contaminated snippets do not separate the targets |

**Honest conclusion:** A3 (array fingerprint) is the axis the current real data
can validate as device-invariant — but only at the **response-type** level
(2 clusters, gas-level purity 0.405), not at the individual-gas/percept level.
A1 is chemistry-grounded. A2's recovery axis cannot be validated from current
data — it needs genuine `baseline → exposure → recovery` protocol recordings,
which remain the single most valuable contribution (see below). The percept map
therefore reports device-invariant *response-type* clusters with their measured
purity, and refuses to claim finer percepts the array cannot resolve.

**On the proposed "four device-invariant categories" scheme:** we ran a
falsification experiment (`experiments/falsify_four_category.py` →
`docs/FALSIFYING_FOUR_CATEGORY.md` →
`results/falsify_four_category.json`). Verdict: the **Strongly/Weakly Reducing
amplitude split is FALSIFIED** (continuous magnitude gradient, 2-group separation
Z≈2.9 < 3.0; the two measured clusters are separated by A3 selectivity *direction*,
not magnitude), and the **Strongly/Weakly Oxidizing + kinetics-grid halves are
untestable** with current data (no oxidizing targets, no protocol dynamics). So
the full four-category scheme must NOT be presented as device-invariant measured
structure; only the 2 reducing response-type clusters are data-backed.

## The Scoped-Down Ecosystem (engine → dashboard)

The honest build ships what the data supports, as a layered engine→dashboard
stack that consumes the SDK extractor (never reimplements it):

| Layer | Module | What it does | Status |
|---|---|---|---|
| Sensor | `normalize.py` | Topology-aware ADC→resistance (`Rs/R0`) | Theoretic; needs `.osmell` recordings to validate empirically |
| Axes | `phenotype.py` | A1 polarity, A2 kinetics (gated), A3 selectivity | Consumes SDK `extract_all_framework_features` |
| Response-type | `response_type.py` | Assign A3 fingerprint → 1 of 2 validated clusters | Fitted + validated on Vergara (99.98% reproduce percept_map); centroid reference saved |
| Percept | `percept_guidance.py` | Response-type → human label + use-case guidance | Scoped-down; carries explicit `cannot` boundaries |
| Reference | `results/response_type_reference.json` | The learned centroids + metadata | Reproducible artifact |

What is deliberately **withheld** (not implemented as classes): the strong/weak
amplitude split (falsified), oxidizing categories (untested), kinetics grid
(untested), molecule ID (purity 0.405 prevents it). Each withheld capability is
an **open question** with the exact data required to test it —
see `docs/OPEN_QUESTIONS.md` (the community data request) and
`docs/DEVICE_INVARIANCE_DISTINCTION.md` (why "response-type" invariance does not
contradict the monitoring paper's cross-device-transfer impossibility).

## Run It

```bash
# Full-187 ablation (protocol + role aware)
python experiments/run_full_ablation.py --limit 60   # quick test
python experiments/run_full_ablation.py              # full
python experiments/run_full_ablation.py --no-smellnet # group/role defs only

# Piecewise cross-dataset ontology validation
python experiments/validate_perception.py

# Falsify the proposed "four device-invariant categories" scheme
python experiments/falsify_four_category.py

# Fit + save the response-type reference centroids (validates reproduce percept_map)
python ontology/response_type.py

# Percept + guidance self-test (scoped-down honest labels)
python ontology/percept_guidance.py
```

## The Single Most Important Next Step

Everything here is currently bounded by one limitation: **we lack data that
follows the `baseline → exposure → recovery` protocol**. This is now *measured*
(see `results/validation_piecewise.json`) and it is the **only** thing standing
between the A3-only percept map we have today and a complete
A1+A2+A3 → PERCEPTS map. Milestone 1 (proving the feature foundation on real
protocol data) is blocked on getting such recordings, and the **A2 kinetic
percept map cannot be learned at all** until they exist. If you have a rig that
records according to `PROTOCOL.md` / `.osmell`, your data is the most valuable
contribution possible right now — see `contributor_guide/CONTRIBUTING.md` (P0
protocol-data + diverse-chemistry asks).

## Status

- [x] Document extractor divergence (IMPLEMENTATION_ALIGNMENT.md)
- [x] Exhaustive feature catalog with physical justifications
- [x] Feature-role taxonomy + protocol gating
- [x] Full 187-feature ablation (runs, SmellNet)
- [x] Sensor profile schema + templates
- [x] Measured-phenotype ontology spec (PHENOTYPE_ONTO.md) — review
- [x] Topology-aware normalization + A1/A2/A3 phenotype layer (normalize.py, phenotype.py)
- [x] Response-convention auto-detect (detect.py) — mirror-pair + single observation binary
- [x] Piecewise cross-dataset validation (Vergara/Hu/SmellNet) → results/validation_piecewise.json
- [x] A3 percept map: data-driven k-means (silhouette), reports device-invariant response-type clusters + gas purity 0.405 (percept_map.py) → results/percept_map_A3.json
- [x] Falsification attempt on the "four device-invariant categories" claim (falsify_four_category.py) → results/falsify_four_category.json: reducing strong/weak split FALSIFIED; oxidizing + kinetics-grid OPEN (no data)
- [ ] A2 percept map (blocked: needs baseline→exposure→recovery protocol data; see contributor_guide/CONTRIBUTING.md)
- [ ] Oxidizing-target validation (blocked: needs external dataset with NO₂/O₃/Cl₂ on same array across devices)
- [ ] Milestone 2-4: universal model training + device-LOO validation
- [ ] Milestone 5: phenotype → genotype (molecular structure)
