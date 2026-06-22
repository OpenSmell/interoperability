"""Load real UCI Gas Sensor Array Drift dataset (16 MOX sensors, 6 gases, 10 batches over 36 months).

Format: 128 features per sample = 8 extracted features × 16 sensors.
Label: 1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene.
Batches 1-10 span ~36 months of drift.
"""

import numpy as np
from glob import glob
from pathlib import Path

UCI_DIR = Path(__file__).resolve().parent / "data" / "uci" / "gas+sensor+array+drift+dataset" / "Dataset"

GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
GAS_IDS = {v: k for k, v in GAS_NAMES.items()}

N_SENSORS = 16
N_FEATURES_PER_SENSOR = 8
N_FEATURES = 128


def load_all_batches(data_dir=None):
    """Load all 10 batches. Returns dict: batch_num -> (X, y)"""
    if data_dir is None:
        data_dir = UCI_DIR
    data_dir = Path(data_dir)
    batches = {}
    for filepath in sorted(glob(str(data_dir / "batch*.dat"))):
        batch_num = int(Path(filepath).stem.split("batch")[1])
        X_batch, y_batch = [], []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0])
                features = [float(val.split(":")[1]) for val in parts[1:129]]
                if len(features) != 128:
                    continue
                X_batch.append(features)
                y_batch.append(label)
        batches[batch_num] = (np.array(X_batch, dtype=np.float64),
                              np.array(y_batch, dtype=np.int32))
    return batches


def load_split(data_dir=None, source_batches=(1, 2, 3, 4, 5), target_batches=(6, 7, 8, 9, 10)):
    """Load source batches and target batches as two virtual devices."""
    batches = load_all_batches(data_dir)
    X_source, y_source = [], []
    for b in source_batches:
        if b in batches:
            X, y = batches[b]
            X_source.append(X)
            y_source.append(y)
    X_target, y_target = [], []
    for b in target_batches:
        if b in batches:
            X, y = batches[b]
            X_target.append(X)
            y_target.append(y)
    return (np.vstack(X_source) if X_source else np.array([]),
            np.concatenate(y_source) if y_source else np.array([]),
            np.vstack(X_target) if X_target else np.array([]),
            np.concatenate(y_target) if y_target else np.array([]))


def summary():
    batches = load_all_batches()
    print(f"UCI Dataset: {len(batches)} batches, 16 sensors × 8 features = 128 dims")
    print(f"Gases: {GAS_NAMES}")
    for b in sorted(batches):
        X, y = batches[b]
        unique, counts = np.unique(y, return_counts=True)
        gas_counts = {GAS_NAMES[g]: c for g, c in zip(unique, counts)}
        print(f"  Batch {b}: {X.shape[0]} samples, gases={gas_counts}")
    return batches


if __name__ == "__main__":
    summary()
