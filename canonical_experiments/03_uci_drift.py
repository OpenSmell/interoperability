"""Experiment 3: UCI Drift Stability — Framework-Like Feature Analysis

UCI Gas Sensor Array Drift dataset (16 MOX sensors, 6 gases, 10 batches, 36 months).

The 128-dim feature vector per sample is organized as 16 sensors × 8 features:
  [ΔR, |ΔR|, EMAi_0.001, EMAi_0.01, EMAi_0.1, EMAd_0.001, EMAd_0.01, EMAd_0.1]×16

These are framework-like features — they are extracted from full sensor response
curves (steady-state + transient dynamics), analogous to our framework taxonomy:
  - ΔR / |ΔR|  → framework's relative_amplitude (Device-Agnostic)
  - EMAi_α      → framework's rising-transient temporal features (rise_time, etc.)
  - EMAd_α      → framework's decaying-transient temporal features (decay_time, etc.)

We test each subset independently to show framework-like features survive drift.

Method:
- Load all 10 UCI batches
- For each subset, compute intra-gas cosine similarity (early→late batches)
  and inter-gas cosine similarity
- Ratio intra/inter > 1 means same-gas similarity survives drift
"""

import sys, json, time
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, METRICS_PATH
from load_uci import load_all_batches, GAS_NAMES

# Feature index layout: for sensor j (0..15), 8 features:
#   idx = j*8 + 0: ΔR          (steady-state amplitude)
#   idx = j*8 + 1: |ΔR|        (normalized amplitude)
#   idx = j*8 + 2: EMAi_0.001  (rising transient, α=0.001)
#   idx = j*8 + 3: EMAi_0.01   (rising transient, α=0.01)
#   idx = j*8 + 4: EMAi_0.1    (rising transient, α=0.1)
#   idx = j*8 + 5: EMAd_0.001  (decaying transient, α=0.001)
#   idx = j*8 + 6: EMAd_0.01   (decaying transient, α=0.01)
#   idx = j*8 + 7: EMAd_0.1    (decaying transient, α=0.1)

FEATURE_SETS = {
    "full_128": "All 128 UCI features (baseline)",
    "steady_state_amplitude": "ΔR + |ΔR| (framework: relative_amplitude)",
    "all_transient_dynamics": "EMAi + EMAd (framework: temporal features)",
    "raw_delta_R": "ΔR only (framework: minimal amplitude)",
}

N_SENSORS = 16
FEATS_PER_SENSOR = 8


def _subset_indices(name):
    if name == "full_128":
        return list(range(128))
    elif name == "steady_state_amplitude":
        return [j * FEATS_PER_SENSOR + k for j in range(N_SENSORS) for k in (0, 1)]
    elif name == "all_transient_dynamics":
        return [j * FEATS_PER_SENSOR + k for j in range(N_SENSORS) for k in range(2, 8)]
    elif name == "raw_delta_R":
        return [j * FEATS_PER_SENSOR + 0 for j in range(N_SENSORS)]
    return list(range(128))


def _cosine_similarities(X_early, X_late, max_samples=100):
    """Compute pairwise cosine similarities between early and late sets."""
    n_early = min(len(X_early), max_samples)
    n_late = min(len(X_late), max_samples)
    vals = []
    for i in range(n_early):
        a = X_early[i]
        for j in range(n_late):
            b = X_late[j]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                vals.append(float(np.dot(a, b) / (na * nb)))
    return vals


def _subset_analysis(X_all, y_all, b_all, indices, gases, early_set, late_set):
    """Run intra/inter cosine similarity analysis on a feature subset."""
    X_sub = X_all[:, indices]

    gas_early = defaultdict(list)
    gas_late = defaultdict(list)
    for i in range(len(y_all)):
        g = int(y_all[i])
        if b_all[i] in early_set:
            gas_early[g].append(X_sub[i])
        elif b_all[i] in late_set:
            gas_late[g].append(X_sub[i])

    intra = {}
    for g in gases:
        early_vecs = np.array(gas_early.get(g, []))
        late_vecs = np.array(gas_late.get(g, []))
        if len(early_vecs) < 2 or len(late_vecs) < 2:
            continue
        cos_vals = _cosine_similarities(early_vecs, late_vecs)
        intra[g] = {
            "gas": GAS_NAMES[g],
            "mean_cosine": float(np.mean(cos_vals)) if cos_vals else 0.0,
            "std_cosine": float(np.std(cos_vals)) if cos_vals else 0.0,
            "n_pairs": len(cos_vals),
        }

    inter = {}
    for gi in gases:
        for gj in gases:
            if gi >= gj:
                continue
            early_i = np.array(gas_early.get(gi, []))[:100]
            late_j = np.array(gas_late.get(gj, []))[:100]
            if len(early_i) < 2 or len(late_j) < 2:
                continue
            cos_vals = _cosine_similarities(early_i, late_j)
            key = f"{GAS_NAMES[gi]}_vs_{GAS_NAMES[gj]}"
            inter[key] = {
                "mean_cosine": float(np.mean(cos_vals)) if cos_vals else 0.0,
                "std_cosine": float(np.std(cos_vals)) if cos_vals else 0.0,
                "n_pairs": len(cos_vals),
            }

    separation = {}
    for g in gases:
        if g not in intra:
            continue
        intra_val = intra[g]["mean_cosine"]
        inter_vals = [v["mean_cosine"] for k, v in inter.items() if GAS_NAMES[g] in k]
        mean_inter = np.mean(inter_vals) if inter_vals else 0
        ratio = intra_val / max(abs(mean_inter), 1e-10)
        separation[GAS_NAMES[g]] = {
            "intra": float(intra_val),
            "inter": float(mean_inter),
            "ratio": float(ratio),
        }

    mean_ratio = float(np.mean([v["ratio"] for v in separation.values()])) if separation else 0.0
    return intra, inter, separation, mean_ratio


def run():
    print("=" * 70)
    print("Experiment 3: UCI Drift Stability — Framework-Like Feature Analysis")
    print("36 months of real sensor aging, 16 MOX sensors, 6 gases")
    print("=" * 70)
    t0 = time.time()
    results = {"experiment": "uci_drift_stability", "data_source": "UCI_gas_sensor_array_drift"}

    batches = load_all_batches()
    if not batches:
        print("  ERROR: No UCI data found.")
        results["runnable"] = False
        results["error"] = "No UCI data"
        return results

    all_X, all_y, all_b = [], [], []
    for b in sorted(batches):
        X, y = batches[b]
        all_X.append(X)
        all_y.append(y)
        all_b.append(np.full(len(y), b))
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    b_all = np.concatenate(all_b)

    gases = sorted(set(int(y) for y in y_all))
    early_batches = {1, 2, 3, 4, 5}
    late_batches = {6, 7, 8, 9, 10}

    print(f"\n  Loaded {len(batches)} batches, {X_all.shape[0]} samples, "
          f"{len(gases)} gases: {[GAS_NAMES[g] for g in gases]}")
    print(f"  Early batches: {sorted(early_batches)}")
    print(f"  Late batches:  {sorted(late_batches)}")
    print(f"  Feature dims:  {X_all.shape[1]}")

    analyses = {}
    print(f"\n  {'─' * 65}")

    for set_name, set_desc in FEATURE_SETS.items():
        indices = _subset_indices(set_name)
        n_dims = len(indices)
        print(f"\n  >>> {set_name} ({n_dims} dims): {set_desc}")

        intra, inter, separation, mean_ratio = _subset_analysis(
            X_all, y_all, b_all, indices, gases, early_batches, late_batches
        )

        print(f"  {'─' * 55}")
        for g in gases:
            if g in intra:
                print(f"  {GAS_NAMES[g]:15s}: intra={intra[g]['mean_cosine']:.4f} "
                      f"inter={separation[GAS_NAMES[g]]['inter']:.4f} "
                      f"ratio={separation[GAS_NAMES[g]]['ratio']:.2f}")

        stable = mean_ratio > 1.0
        print(f"  {'─' * 55}")
        print(f"  Mean separation ratio: {mean_ratio:.2f} "
              f"{'✓ STABLE' if stable else '✗ UNSTABLE'}")

        analyses[set_name] = {
            "description": set_desc,
            "n_dims": n_dims,
            "intra_similarities": {str(k): v for k, v in intra.items()},
            "inter_similarities": inter,
            "separation_ratios": separation,
            "mean_separation_ratio": float(mean_ratio),
            "stable": stable,
        }

    # Summary table
    print(f"\n  {'=' * 55}")
    print(f"  SUMMARY: Framework-Like Feature Stability Across 36 Months")
    print(f"  {'=' * 55}")
    print(f"  {'Feature Subset':35s} {'Dims':5s} {'Ratio':6s} {'Status':8s}")
    print(f"  {'─' * 55}")
    for set_name, analysis in analyses.items():
        desc = FEATURE_SETS[set_name].split("(")[0].strip()
        short = desc[:33]
        ratio = analysis["mean_separation_ratio"]
        status = "STABLE" if analysis["stable"] else "UNSTABLE"
        print(f"  {short:35s} {analysis['n_dims']:5d} {ratio:6.2f} {status:8s}")

    results["runnable"] = True
    results["n_batches"] = len(batches)
    results["n_samples"] = X_all.shape[0]
    results["n_gases"] = len(gases)
    results["early_batches"] = sorted(early_batches)
    results["late_batches"] = sorted(late_batches)
    results["feature_set_analysis"] = analyses
    results["duration_s"] = round(time.time() - t0, 2)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_3"] = results
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Elapsed: {results['duration_s']:.1f}s")
    return results


if __name__ == "__main__":
    run()
