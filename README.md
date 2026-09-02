# Digital Olfaction Framework: Canonical Experiments and Cross-Device Bounds

This repository contains the canonical experiments for the modular feature framework for digital olfaction, as described in the paper "Towards Interoperable Digital Olfaction: A Modular Feature Framework for Electronic Noses": session-invariance on a single rig (the demonstrated result), the failure of zero-shot cross-device transfer, and the reference-point calibration path to a concentration-reading instrument.

## TL;DR — the honest one-paragraph summary

OpenSmell is building the engineering stack for smell (the way we have one for images and audio): open SDK, open protocol, open data. **What this repo proves:** a framework that extracts physically-grounded features from MOX sensor time-series, with session-invariance on a single rig (88.5%), and a formal proof (`R_s/R_0` normalization) bounding what *can* and *cannot* transfer across devices. **What it does NOT prove:** that a MOX array can identify individual molecules device-invariantly — it can't without per-rig calibration; and the four-category "strong/weak × reducing/oxidizing" scheme is *not* validated (see below). The honest, data-backed result is a **device-invariant response-type structure** (2 clusters among reducing targets). The **only** axis that transfers across a device family is the coarse A3 cross-sensor selectivity fingerprint — everything finer is provably device-bound.

## Start here

- **New to this repo?** See [What this repo proves / does not prove](#what-this-repo-proves--does-not-prove), then [Quick Start](#quick-start).
- **Want the newest research?** The measured-phenotype ontology (response-type clusters, falsification attempt on the 4-category scheme, open research questions + data requests) lives in [`perception-layer/`](perception-layer/README.md).
- **Looking for the SDK?** `pip install opensmell` — feature extractor, monitor, and calibration live in the SDK, not here.
- **Want to contribute data?** See [Contributing](#contributing); priority gaps are protocol recordings, oxidizing targets, and labelled concentrations.

## What this repo proves / does not prove

**Proven (with real data, measured — not asserted):**

| Claim | Where | Result |
|-------|-------|--------|
| Session-invariance on one rig | Experiment 1 | 88.5% across held-out sessions |
| Leave-substance-out consistency | Experiment 2 | 67.4% on novel substances |
| `R_s/R_0` cancels circuit params (`Vcc`, `R_L`) | Overview + SDK | Theorem 2: only sensor constants `(a,b)` survive |
| Zero-shot cross-device transfer fails | Experiment 7 | 10–18% vs 25% chance; consistent with Theorem 2 |
| Device-invariant *response-type* structure (A3 selectivity) | `perception-layer/` | 2 clusters across 10 UCI drift batches; gas purity 0.405 |

**Does NOT prove (and why):**

| Claim | Status |
|-------|--------|
| Individual molecule ID device-invariantly | **Impossible without per-rig calibration** (Theorem 2) |
| "Strong vs weak" amplitude split | **Falsified** on real data — continuous gradient, not two classes (see `perception-layer/docs/FALSIFYING_FOUR_CATEGORY.md`) |
| Oxidizing categories (NO₂/O₃/Cl₂) | **Untested** — no oxidizing data in corpus (open question, `perception-layer/docs/OPEN_QUESTIONS.md`) |
| Kinetics as an orthogonal axis | **Untested** — no `baseline→exposure→recovery` protocol data (open question) |

> These limits are not dismissals; they are the *open research questions* the project actively wants data to resolve. See `perception-layer/docs/OPEN_QUESTIONS.md` for the exact data each one needs.

## Overview

The framework extracts device-agnostic features from MOX sensor time-series along five dimensions (device-agnostic, absolute, temporal, health, hardware). The key contribution is the proof that $R_s/R_0$ normalization cancels both $V_{cc}$ and $R_L$ completely — which bounds what can and cannot transfer across devices: sensor constants $(a, b)$ do not cancel, so zero-shot transfer fails (Experiment 7) and calibration is per-rig.

This repository validates the framework through seven experiments:

| # | Experiment | Key Result |
|---|-----------|------------|
| 1 | **Session-Invariance** | 88.5% accuracy across held-out sessions (t=60.78, p<0.001) |
| 2 | **Leave-Substance-Out** | 67.4% consistency on novel substances (σ=7.6%) |
| 3 | **UCI Drift Stability** | All feature subsets maintain separation ratio > 1.0 across 36 months |
| 4 | **Normalization Comparison** | Framework features (67.3%) > raw voltages (63.3%) > R/R₀ alone (37.5%) |
| 5 | **Ablation Study** | All feature subgroups contribute < 1% to accuracy — framework is highly redundant |
| 6 | **Baseline Comparison** | Framework (88.5%) > 1D-CNN (81.78%) > raw voltages (63.3%) > ScentFormer (53.0%) > R/R₀ (37.5%) |
| 7 | **Cross-Device Sanity** | Zero-shot transfer fails (10-18% vs 25% chance; best single channel 40.7% borderline), consistent with Theorem 2 |

## Cross-Device Calibration / Alignment Arc (Exps 1-7)

Beyond the seven canonical experiments above, a second numbered series in
`alignment_experiments/` attacks the remaining cross-device transfer paths from
the paper's interoperability agenda. Each is falsifiable, LOO-fair, and
documented with an `*_analysis.md`. Every reference-free alignment is at or
near chance; gains appear only when the target substance's own reference
samples enter the fit (calibration data), consistent with the framework's
per-rig calibration bound.

| # | Script | Venue | LOO-fair result |
|---|--------|-------|-----------------|
| 1 | `experiment_1_calibration.py` | SmellNet → Praise James' rig, single-point M | +0.0 pp |
| 2 | `experiment_2_synthetic_probe.py` | synthetic gain/exponent/batch | M sound for gain only |
| 3 | `experiment_3_calibration.py` | UCI two-point (a,b) vs M | mag ≤ +0.4 pp, full +0.0 |
| 4 | `experiment_4_coral.py` | UCI CORAL covariance map | coral_ref ≤ +2.8 pp LOO-fair; coral_all +3.3–7.3 pp (test-gas samples in fit) |
| 5 | `experiment_5_anchors.py` | Praise James' rig, 3-anchor affine/Procrustes | +0.0 pp |
| 6 | `experiment_6_taxonomy.py` | UCI classes + SmellNet→OSMO | within-class error share below chance; rig→Woody |
| 7 | `experiment_7_chemoprint.py` | UCI sensor→chemoprint | within 99.6% → cross 2.4%/−5.0% |

Run individually from this repository with the project venv, e.g.
`python ../venv/bin/python alignment_experiments/experiment_4_coral.py` (or run
`run_alignment_experiments.py` from `alignment_experiments/` to run all seven).
Every script resolves its own paths from its file location and writes
`result.txt` + `metrics.json` into `alignment_experiments/results/`. Details in
each `*_analysis.md`;

## The reference-point calibration path

Since every reference-free alignment above is at or near chance, the route to
a concentration-reading instrument is per-rig reference-point
calibration (`rr = R/R0 = a·C^b`, fitted in log-log space; invert as
`C = (rr/a)^(1/b)`). It is implemented in the SDK
(`opensmell/opensmell/calibration.py`), falsified numerically, and gated by the
`HardwareInsufficiencyWarning` (`opensmell/opensmell/hardware.py`):

| Item | Venue | Result |
|---|---|---|
| Method recovery under noise | `research/calibration-experiments/reference-point-calibration/experiment.py` | unbiased; σ=5%, 6 pts, 2 decades → LOOCV ≈7.1% median conc. error |
| Data budget (points × noise) | same | σ=5% → 4 pts ≈9.5%; σ=10% → 4 pts ≈19% |
| Real rig repeatability | Praise James' rig session cache (`experiment_1_features_cache.npz`) | σ_session ≈ 12% ⇒ replicates required (4/point → σ≈5%) |
| Extrapolation penalty | same | predicting 100 ppm from a 1–30 ppm fit is worse — never extrapolate |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/opensmell/interoperability.git
cd interoperability/canonical_experiments

# Install dependencies
pip install numpy scipy scikit-learn pandas matplotlib

# Download UCI dataset (required for Experiment 3)
# Download from: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
# Place in: canonical_experiments/data/uci/gas+sensor+array+drift+dataset/Dataset/

# Run all experiments
python3 run_all.py
```

## Dependencies

- Python 3.8+
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib (optional, for figures)

## Data Sources

> **Data download requirements (what auto-fetches vs. what's manual):**

> - **SmellNet** — *auto-fetched* from HuggingFace (`DeweiFeng/smell-net`). No manual setup. Used by canonical 1, 2 and alignment 1, 2, 5, 6.
> - **UCI Gas Drift + UCI mixtures** — *manual download* required (research-only license). Without these, experiments 3, 7 and alignment 3, 4, 7 fail with a clear "data not found" message.
> - **Praise James' rig / OSMO taxonomy** — results are *pre-computed and documented*; no re-run needed.

### SmellNet (canonical 1, 2; alignment 1, 2, 5, 6)

- **Auto-fetched** from HuggingFace: `DeweiFeng/smell-net` — no manual setup required
- 50 food substances, 6 MOX sensors, multiple sessions
- Citation: Feng, D., Dai, W., Li, C., Pernigo, A., Wen, Y. & Liang, P. P.
  "SmellNet: A Large-scale Dataset for Real-world Smell Recognition."
  arXiv:2506.00239 (2025); ICLR 2026.
- Alignment experiments use a local mirror at `../SmellNet/neurips-data-processed/`
  (also mirrored on HuggingFace); the Praise James' rig session cache is
  `alignment_experiments/experiment_1_features_cache.npz` (gitignored).

### OSMO taxonomy (alignment 6)
- Osmo Labs, PBC. "The Osmo Scent Taxonomy," v1.1 (2025):
  <https://github.com/osmoai/taxonomy>

### UCI Gas Sensor Array Drift (canonical 3; alignment 3, 4, 7)
- **Manual download required** (research-only license): https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
- Extract to: `data/uci/gas+sensor+array+drift+dataset/Dataset/`
- Should contain `batch1.dat` through `batch10.dat`
- 6 pure gases, 16 MOX sensors, 10 batches over 36 months

### UCI Dynamic / Turbulent Mixtures (alignment 3, 4)
- **Manual download required**: 
  - Dynamic mixtures: https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures
  - Turbulent mixtures: https://archive.ics.uci.edu/dataset/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures
- Extract into `alignment_experiments/data/{dynamic,turbulent}-mixtures/`
- 16 MOX sensors under varying concentration/gas-composition profiles; research-only license (validation, not product training)

### Praise James' rig (alignment 1, 2, 5)
- 3-sensor rig recordings from `~/Osmograph_Recordings/`
- Substances: garlic, ginger, cinnamon, banana
- Results are pre-computed and documented — no re-run needed

## Repository Structure

```
interoperability/
├── canonical_experiments/
│   ├── 01_session_invariance.py          # Experiment 1
│   ├── 02_leave_substance_out.py         # Experiment 2
│   ├── 03_uci_drift.py                   # Experiment 3
│   ├── 04_normalization_comparison.py    # Experiment 4
│   ├── 05_ablation.py                    # Experiment 5
│   ├── 06_baseline_comparison.py         # Experiment 6
│   ├── 07_cross_device_sanity.py         # Experiment 7
│   ├── config.py                         # Configuration
│   ├── load_data.py                      # SmellNet data loader
│   ├── load_uci.py                       # UCI data loader
│   ├── framework_features.py             # Feature extraction (145-dim)
│   ├── run_all.py                        # Run all canonical experiments
│   ├── generate_figures.py               # Generate Figs 1-5
│   ├── generate_tables.py                # Generate Tables 1-9
│   ├── data/
│   │   └── uci/                          # UCI dataset (user-provided)
│   └── results/                          # Canonical results/metrics
└── alignment_experiments/
    ├── experiment_1_calibration.py       # Session invariance of features
    ├── experiment_2_synthetic_probe.py   # Synthetic gain/exponent/batch probe
    ├── experiment_3_calibration.py       # UCI two-point (a,b) vs single-point M
    ├── experiment_4_coral.py             # UCI CORAL covariance alignment
    ├── experiment_5_anchors.py           # Real-rig 3-anchor affine/Procrustes
    ├── experiment_6_taxonomy.py          # SmellNet→OSMO taxonomy transfer
    ├── experiment_7_chemoprint.py        # UCI sensor→chemoprint regression
    ├── run_alignment_experiments.py      # Run all 1-7 sequentially
    ├── analyses/                         # *_analysis.md per experiment
    ├── results/                          # *_result.txt + *_metrics.json
    └── experiment_1_features_cache.npz   # Praise James' rig session cache (gitignored)
└── README.md                             # This file
```

## Expected Results

### Experiment 1: Session-Invariance
- Mean accuracy: 88.5% (145-dim full framework)
- Per-iteration: ~83-93%
- t-statistic: ~60.78, p < 0.000001
- Chance level: 2%

### Experiment 2: Leave-Substance-Out
- Mean consistency: 67.4% (σ=7.6%)
- Per-fold: ~53-75%
- Random baseline: 2%

### Experiment 3: UCI Drift Stability
- Full 128 features: ratio 1.18 (STABLE)
- Steady-state amplitude: ratio 1.18 (STABLE)
- Transient dynamics: ratio 1.15 (STABLE)
- ΔR only: ratio 1.18 (STABLE)

### Experiment 4: Normalization Comparison
- Framework features: 67.3% consistency (145 dims)
- Raw voltages: 63.3% consistency (600 dims)
- R/R₀ normalization: 37.5% consistency (600 dims)

### Experiment 5: Ablation Study (Dimension 1 — Device-Agnostic Features)
Tests which subgroups of the 55 device-agnostic features matter most.

| Feature Subgroup Removed | Accuracy | Δ from Dim 1 |
|--------------------------|----------|-------------|
| None (all 55 Dim 1 feats) | 65.9% | — |
| Selectivity Ratios | 61.0% | **-4.9%** |
| Global Metrics | 64.9% | -1.0% |
| Kinetics | 65.8% | -0.1% |
| Amplitude | 66.1% | +0.2% |
| Integral/Endpoint | 66.1% | +0.2% |

Selectivity ratios (cross-channel relative amplitude comparisons) carry the most discriminative power. Per-channel features (amplitude, kinetics, integral) are individually replaceable due to correlation.

Note: The ablation study was conducted on SmellNet data, which lacks explicit baseline-exposure-recovery phases. Consequently, kinetics features (rise/decay time) are unreliable or zero in this dataset. The contribution of kinetics features may be higher when tested with data collected using the standardized Osmograph protocol. A follow-up ablation study on Osmograph-collected data is planned.

### Experiment 6: Baseline Comparison
| Rank | Method | Accuracy |
|------|--------|----------|
| 1 | Framework features (ours, 145-dim) | 88.5% |
| 2 | Contrastive 1D-CNN (ours) | 81.78% |
| 3 | Raw voltages + RandomForest | 63.3% |
| 4 | ScentFormer (Transformer) | 53.0% |
| 5 | R/R₀ time series + RandomForest | 37.5% |

### Experiment 7: Cross-Device Sanity Check
| Feature Set | Accuracy | Chance | Significance |
|------------|----------|--------|-------------|
| 30-dim paradigm (6 ch) | 11.9% | 25% | n.s. |
| 145-dim framework | 18.6% | 25% | n.s. |
| 15-dim shared sensors | 10.2% | 25% | n.s. |
| MQ-135 VOC only | 40.7% | 25% | borderline |
| MQ-3 only | 6.8% | 25% | n.s. |

These informal tests between a 3-sensor OpenSmell rig (Praise James' rig) and 6-sensor SmellNet device are consistent with Theorem 2: different MQ sensor models (different a,b constants) produce incompatible feature distributions even after R/R₀ normalization.

## Feature Extraction

The canonical experiments extractor produces 145 features per 6-sensor
recording:
- 21 per-channel features (6 device-agnostic + 4 absolute + 4 temporal + 4
  health + 3 hardware) × 6 channels = 126
- 15 cross-channel selectivity ratios
- 4 global features

The current SDK (`opensmell`) adds 7 more per channel (1 saturation index +
6 decay constants), giving 187 features on 6 channels and 91 on the 3-channel
rig. This repository's canonical experiments use the 145-dim extractor in
`canonical_experiments/framework_features.py`.

Key features include:
- Relative amplitude (ΔR/R₀)
- Rise time (10%→90%)
- Decay time (90%→10%)
- Area under curve
- Selectivity ratios

See `framework_features.py` for the complete implementation.

## Reproducibility

All experiments use fixed random seeds (`RANDOM_STATE = 42`) for reproducibility. Results are saved to `results/metrics.json` after each run.

To reproduce a specific experiment:
```bash
python3 01_session_invariance.py
python3 02_leave_substance_out.py
python3 03_uci_drift.py
python3 04_normalization_comparison.py
python3 05_ablation.py
python3 06_baseline_comparison.py
python3 07_cross_device_sanity.py
```

## Paper

The full paper will be linked here once published on arXiv.

## Citation

James, P. (2026). Towards Interoperable Digital Olfaction: A Modular Feature Framework for Metal‑Oxide Semiconductor Electronic Noses. Zenodo. https://doi.org/10.5281/zenodo.21243013

## License

Open-source under MIT License. See LICENSE file for details.

## Contact

- GitHub: https://github.com/opensmell

## Contributing

We welcome contributions that help close the gaps between the framework and real-world deployment. Priority areas include:

**High Priority (Directly addresses blockers in the README):**

    Labelled concentration data — The reference-point calibration path is ready (run_calibration.py) but lacks a dataset with known ppm values. If you have access to controlled gas mixtures or can record sensor responses to known concentrations, your data would unlock empirical validation of the two-point calibration method.

    Cross-device validation experiments — The zero-shot transfer failure is proven theoretically and observed informally (Experiment 7). We need controlled experiments with two devices, known substances, and matched concentrations to empirically validate the two-point calibration approach described in the paper.

**Medium Priority (Extends the framework's utility):**

    Additional sensor modalities — The framework is modular and sensor-agnostic. Contributions for electrochemical, optical, or MEMS sensors are welcome, especially if you can provide the equivalent of the device-agnostic normalization for your modality.

    Temporal feature validation — The ablation study's kinetics features (rise/decay time) were limited by SmellNet's lack of explicit baseline-exposure-recovery phases. If you collect data using the standardized protocol (see PROTOCOL.md), we can retest the ablation and validate whether kinetics features become more discriminative.

**General (Always welcome):**

    Bug fixes and improvements — Open an issue before submitting a PR to discuss the fix.

    Documentation enhancements — Clarity, examples, quickstart guides, and translations.

**How to contribute:**

    Open an issue describing what you'd like to work on.

    Fork the repository and create a branch for your work.

    Submit a pull request with a clear description of the changes.

    Ensure experiments remain reproducible (fixed random seeds, documented data sources).

Data contribution: If you have sensor data, please see data-commons/ for the contribution pipeline. For labelled concentration data, contact us directly to coordinate calibration experiments.
