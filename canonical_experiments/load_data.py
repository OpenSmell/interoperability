"""Data loading: unified interface for all canonical experiment data sources.

SmellNet data is fetched from HuggingFace (DeweiFeng/smell-net) and cached
at ~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net/.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
N_CHANNELS = 6


def _find_hf_snapshot_dir():
    """Locate the HuggingFace cache snapshot for DeweiFeng/smell-net.

    Returns path to base_data/ directory containing training/ and testing/
    subdirectories, or None if not found.
    """
    hub_candidates = [
        os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net"),
    ]

    for candidate in hub_candidates:
        snapshots = os.path.join(candidate, "snapshots")
        if os.path.isdir(snapshots):
            revisions = sorted(os.listdir(snapshots), reverse=True)
            for rev in revisions:
                base = os.path.join(snapshots, rev, "base_data")
                if os.path.isdir(base):
                    return base

    # Try to trigger download via datasets library
    try:
        from datasets import load_dataset
        load_dataset("DeweiFeng/smell-net", "base_data", split="train")
        # Retry after download
        for candidate in hub_candidates:
            snapshots = os.path.join(candidate, "snapshots")
            if os.path.isdir(snapshots):
                revisions = sorted(os.listdir(snapshots), reverse=True)
                for rev in revisions:
                    base = os.path.join(snapshots, rev, "base_data")
                    if os.path.isdir(base):
                        return base
    except Exception:
        pass

    return None


def _walk_csv_files(base_dir):
    """Walk training/ and testing/ dirs, yield (substance, session_num, filepath)."""
    for split_name in ["training", "testing"]:
        split_dir = os.path.join(base_dir, split_name)
        if not os.path.isdir(split_dir):
            continue
        for substance_name in sorted(os.listdir(split_dir)):
            sub_dir = os.path.join(split_dir, substance_name)
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if not fname.endswith(".csv"):
                    continue
                stem = fname[:-4]
                parts = stem.split("_")
                session_str = parts[-1]
                if not session_str.isdigit():
                    continue
                session_num = int(session_str)
                yield substance_name, session_num, os.path.join(sub_dir, fname)


def load_all_smellnet_data():
    """Load all SmellNet data from HuggingFace cache.

    Returns dict: {substance_name: [(session_num, np.ndarray), ...]}
    where each ndarray is shape (T, 6) with columns matching SENSOR_NAMES.
    """
    base = _find_hf_snapshot_dir()
    if base is None:
        raise RuntimeError(
            "SmellNet data not found in HuggingFace cache. "
            "Run: python -c \"from datasets import load_dataset; "
            "load_dataset('DeweiFeng/smell-net', 'base_data', split='train')\""
        )

    result = {}
    for substance, session_num, fpath in _walk_csv_files(base):
        try:
            df = pd.read_csv(fpath)
            avail = [c for c in SENSOR_NAMES if c in df.columns]
            if not avail:
                continue
            data = df[avail].values.astype(np.float32)
            if np.any(np.isnan(data)):
                for c in range(data.shape[1]):
                    col_str = df[avail[c]].astype(str).str.replace(",", ".")
                    data[:, c] = pd.to_numeric(col_str, errors="coerce").values
                data = np.nan_to_num(data, nan=0.0).astype(np.float32)
            result.setdefault(substance, []).append((session_num, data))
        except Exception:
            continue

    return result


def find_smellnet_files():
    """Return list of all SmellNet CSV file paths (for backward compatibility).

    Uses HuggingFace cache.
    """
    base = _find_hf_snapshot_dir()
    if base is None:
        return []
    return [fpath for _, _, fpath in _walk_csv_files(base)]


def load_smellnet_csv(path):
    """Load a single SmellNet CSV file and return 6-channel array."""
    path = Path(path)
    df = pd.read_csv(path)
    rename = {"C2H50H": "C2H5OH"}
    df.columns = [rename.get(c.strip(), c.strip()) for c in df.columns]
    avail = [c for c in SENSOR_NAMES if c in df.columns]
    if not avail:
        raise ValueError(f"No sensor columns found in {path}")
    raw = df[avail].values.astype(np.float32)
    if np.any(np.isnan(raw)):
        for c in range(raw.shape[1]):
            col_str = df[avail[c]].astype(str).str.replace(",", ".")
            raw[:, c] = pd.to_numeric(col_str, errors="coerce").values
    return np.nan_to_num(raw, nan=0.0).astype(np.float32)
