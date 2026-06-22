# Interoperability: Canonical Experiments for Digital Olfaction Framework

This repository contains the canonical experiments validating the modular feature framework for interoperable digital olfaction, as described in the paper "Towards Interoperable Digital Olfaction: A Modular Feature Framework for Electronic Noses."

## Overview

The framework extracts device-agnostic features from MOX sensor time-series along five dimensions (device-agnostic, absolute, temporal, health, hardware). The key contribution is the mathematical proof that $\frac{R_s}{R_0}$ normalization cancels both $V_{cc}$ and $R_L$ completely, enabling theoretical cross-device interoperability.

This repository validates the framework through four experiments:

1. **Session-Invariance**: 88.5% accuracy across held-out sessions (t=60.78, p<0.000001)
2. **Leave-Substance-Out Generalization**: 67.4% consistency on novel substances
3. **UCI Drift Stability**: All framework-like features maintain separation ratio > 1.0 across 36 months
4. **Normalization Comparison**: Framework features (67.3%) outperform raw voltages (63.3%) and R/R₀ alone (37.5%)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/opensmell/interoperability.git
cd interoperability/canonical_experiments

# Install dependencies
pip install numpy scipy scikit-learn pandas

# Download UCI dataset (required for Experiment 3)
# Download from: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
# Place in: data/uci/gas+sensor+array+drift+dataset/Dataset/

# Run all experiments
python3 run_all.py
```

## Dependencies

- Python 3.8+
- numpy
- scipy
- scikit-learn
- pandas

## Data Sources

### SmellNet (Experiments 1, 2, 4)
- Automatically fetched from HuggingFace: `DeweiFeng/smell-net`
- 50 food substances, 6 MOX sensors, multiple sessions
- No manual setup required

### UCI Gas Sensor Array Drift (Experiment 3)
- Download from: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
- Extract to: `data/uci/gas+sensor+array+drift+dataset/Dataset/`
- Should contain `batch1.dat` through `batch10.dat`
- 6 pure gases, 16 MOX sensors, 10 batches over 36 months

## Repository Structure

```
interoperability/
├── canonical_experiments/
│   ├── 01_session_invariance.py          # Experiment 1
│   ├── 02_leave_substance_out.py         # Experiment 2
│   ├── 03_uci_drift.py                   # Experiment 3
│   ├── 04_normalization_comparison.py    # Experiment 4
│   ├── config.py                         # Configuration
│   ├── load_data.py                      # SmellNet data loader
│   ├── load_uci.py                       # UCI data loader
│   ├── framework_features.py              # Feature extraction (145-dim)
│   ├── run_all.py                        # Run all experiments
│   ├── data/
│   │   └── uci/                          # UCI dataset (user-provided)
│   ├── results/
│   │   ├── metrics.json                  # All experiment results
│   │   ├── figures/                      # Generated plots
│   │   └── tables/                       # Generated tables
│   └── README.md                         # This file
└── README.md                             # Top-level README
```

## Expected Results

After running `python3 run_all.py`, you should see:

### Experiment 1: Session-Invariance
- Mean accuracy: 88.5%
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

## Feature Extraction

The framework extracts 145 features per recording (6 sensors):
- 6 per-channel device-agnostic features × 6 channels = 36
- 15 cross-channel selectivity ratios
- 4 global features
- Additional derived features

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
```

## Paper

The full paper is available on arXiv: [link to be added after submission]

## Citation

```bibtex
@article{james2026interoperable,
  title={Towards Interoperable Digital Olfaction: A Modular Feature Framework for Electronic Noses},
  author={James, Praise},
  year={2026},
  journal={arXiv preprint}
}
```

## License

Open-source under MIT License. See LICENSE file for details.

## Contact

- GitHub: https://github.com/opensmell
- Email: praisejx@protonme

## Contributing

We welcome contributions, especially:
- Cross-device validation experiments
- Additional sensor modalities
- Bug fixes and improvements
- Documentation enhancements

Please open an issue or pull request on GitHub.