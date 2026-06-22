"""Configuration: paths, constants, and parameters for all canonical experiments."""

from pathlib import Path
import numpy as np

# === Experiment Output Paths ===
CANONICAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CANONICAL_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# === Experiment Parameters ===
R0_SAMPLES = 15
WINDOW_SIZE = 100
WINDOW_STRIDE = 10

BASELINE_SECONDS = 30
EXPOSURE_SECONDS = 45
RECOVERY_SECONDS = 120

N_ESTIMATORS = 200
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === UCI Gas Sensor Array Drift Constants ===
UCI_ETHANOL = np.array([1, 2, 3, 4, 5, 6])
UCI_ETHYLENE = np.array([7, 8, 9, 10, 11, 12])
UCI_AMMONIA = np.array([13, 14, 15, 16, 17, 18])
UCI_ACETALDEHYDE = np.array([19, 20, 21, 22, 23, 24])
UCI_ACETONE = np.array([25, 26, 27, 28, 29, 30])
UCI_TOLUENE = np.array([31, 32, 33, 34, 35, 36])

UCI_BATCH_GAS_MAP = {
    1: "Ethanol", 2: "Ethanol", 3: "Ethanol", 4: "Ethanol", 5: "Ethanol", 6: "Ethanol",
    7: "Ethylene", 8: "Ethylene", 9: "Ethylene", 10: "Ethylene", 11: "Ethylene", 12: "Ethylene",
    13: "Ammonia", 14: "Ammonia", 15: "Ammonia", 16: "Ammonia", 17: "Ammonia", 18: "Ammonia",
    19: "Acetaldehyde", 20: "Acetaldehyde", 21: "Acetaldehyde", 22: "Acetaldehyde",
    23: "Acetaldehyde", 24: "Acetaldehyde",
    25: "Acetone", 26: "Acetone", 27: "Acetone", 28: "Acetone", 29: "Acetone", 30: "Acetone",
    31: "Toluene", 32: "Toluene", 33: "Toluene", 34: "Toluene", 35: "Toluene", 36: "Toluene",
}
