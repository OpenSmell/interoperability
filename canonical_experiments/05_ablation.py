"""Experiment 5: Ablation Study — Dimension 1 (Device-Agnostic Features)

Tests contribution of each feature subgroup within Dimension 1 only
(device-agnostic per-channel features + selectivity ratios + global metrics).

Baseline: all 55 device-agnostic features (Dimension 1).
Each ablation removes one subgroup and measures session-invariance drop.

Subgroups (all within Dimension 1):
  - Amplitude:          relative_amplitude, direction          (12 feats)
  - Kinetics:           rise_time, decay_time                  (12 feats)
  - Integral/Endpoint:  auc, endpoint_delta                    (12 feats)
  - Selectivity Ratios: sel_ratio_*                            (15 feats)
  - Global Metrics:     global_*                               (4 feats)

Total Dimension 1: 36 per-channel + 15 selectivity + 4 global = 55 feats.
"""

import sys, json, time, re
from pathlib import Path
from collections import defaultdict
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, METRICS_PATH, RANDOM_STATE, R0_SAMPLES, WINDOW_SIZE, WINDOW_STRIDE
from load_data import load_all_smellnet_data
from framework_features import extract_all_framework_features, feature_vector, feature_names


N_ITERATIONS = 3
ABLATION_N_ESTIMATORS = 100


def extract_features(data, sr=10):
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    W = min(WINDOW_SIZE, N)
    if N < W:
        padded = np.pad(data, ((0, W - N), (0, 0)), mode="edge")
        feats_dict = extract_all_framework_features(padded, r0_samples=R0_SAMPLES, sr=sr)
        vec, _ = feature_vector(feats_dict)
        return vec.reshape(1, -1)
    stride = WINDOW_STRIDE
    windows = [data[i:i + W] for i in range(0, N - W + 1, stride)]
    feat_vecs = []
    for w in windows:
        feats_dict = extract_all_framework_features(w, r0_samples=min(R0_SAMPLES, W), sr=sr)
        vec, _ = feature_vector(feats_dict)
        feat_vecs.append(vec)
    return np.array(feat_vecs, dtype=np.float32)


ALL_NAMES = feature_names()
N_FULL = len(ALL_NAMES)

# Dimension 1 includes: per-channel da_*, sel_ratio_*, global_*
DIM1_PATTERN = re.compile(r"(_da_|^sel_ratio_|^global_)")
dim1_mask = np.array([bool(DIM1_PATTERN.search(n)) for n in ALL_NAMES], dtype=bool)
DIM1_NAMES = [n for n, m in zip(ALL_NAMES, dim1_mask) if m]

# Subgroup patterns within Dimension 1
SUBGROUPS = {
    "Amplitude": re.compile(r"_da_(relative_amplitude|direction)$"),
    "Kinetics": re.compile(r"_da_(rise_time|decay_time)$"),
    "Integral_Endpoint": re.compile(r"_da_(auc|endpoint_delta)$"),
    "Selectivity_Ratios": re.compile(r"^sel_ratio_"),
    "Global_Metrics": re.compile(r"^global_"),
}


def get_dim1_keep_mask(exclude_group):
    """Mask on full 145-dim: keep only Dimension 1 features EXCEPT exclude_group."""
    pattern = SUBGROUPS[exclude_group]
    mask = dim1_mask.copy()
    for i, name in enumerate(ALL_NAMES):
        if mask[i] and pattern.search(name):
            mask[i] = False
    return mask


def run_session_invariance(X_all, y_all, y_enc, sess_all, le, substances):
    sub_ses_indices = defaultdict(lambda: defaultdict(list))
    for idx, (sub, ses) in enumerate(zip(y_all, sess_all)):
        sub_ses_indices[sub][ses].append(idx)

    rng = np.random.RandomState(RANDOM_STATE)
    session_accs = []

    for i in range(N_ITERATIONS):
        train_idx, test_idx = [], []
        for sub, ses_dict in sub_ses_indices.items():
            sessions = sorted(ses_dict.keys())
            n_test = max(1, len(sessions) // 3)
            test_sessions = set(rng.choice(sessions, size=n_test, replace=False))
            for ses, indices in ses_dict.items():
                if ses in test_sessions:
                    test_idx.extend(indices)
                else:
                    train_idx.extend(indices)

        if len(train_idx) < 10 or len(test_idx) < 2:
            continue

        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        if len(np.unique(y_tr)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=ABLATION_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr_s, y_tr)
        preds = rf.predict(X_te_s)
        acc = accuracy_score(y_te, preds)
        session_accs.append(float(acc))

    return session_accs


def run():
    print("=" * 70)
    print("Experiment 5: Ablation Study — Dimension 1 Feature Subgroups")
    print("=" * 70)
    t0 = time.time()

    results = {
        "experiment": "ablation_study",
        "description": "Dimension 1 (device-agnostic) only: 55 feats",
        "n_substances": 0,
        "n_dim1_features": int(dim1_mask.sum()),
        "n_full_features": N_FULL,
        "baseline_accuracy": None,
        "ablations": {},
    }

    # ── Load data ────────────────────────────────────────────────────
    all_data = load_all_smellnet_data()
    multi_ses = {s: g for s, g in all_data.items() if len(g) >= 2}
    substances = sorted(multi_ses.keys())
    results["n_substances"] = len(substances)

    if len(substances) < 3:
        print("  ERROR: insufficient substances")
        results["runnable"] = False
        return results

    # ── Extract full feature matrix ──────────────────────────────────
    X_chunks, y_list, sess_list = [], [], []
    for sub in substances:
        for ses_num, data_array in multi_ses[sub]:
            feats = extract_features(data_array)
            X_chunks.append(feats)
            for _ in range(len(feats)):
                y_list.append(sub)
                sess_list.append(ses_num)

    X_all = np.vstack(X_chunks)
    y_all = np.array(y_list)
    sess_all = np.array(sess_list)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    n_feats_full = X_all.shape[1]
    n_dim1 = int(dim1_mask.sum())
    print(f"\n  Full feature dim: {n_feats_full}  |  Dimension 1: {n_dim1}")

    # ── Build ablation configs ───────────────────────────────────────
    # First entry: Dimension 1 only (baseline)
    ablation_configs = [("Dimension_1_Only", dim1_mask)]
    for group_name in sorted(SUBGROUPS.keys()):
        ablation_configs.append((f"Remove_{group_name}", get_dim1_keep_mask(group_name)))

    # ── Run each ablation ────────────────────────────────────────────
    table_rows = []
    chance = 1.0 / len(substances)

    for label, mask in ablation_configs:
        kept = int(mask.sum())
        removed = int((~mask).sum() - (N_FULL - n_dim1))  # only Dim1 removals
        if label.startswith("Remove"):
            print(f"\n  {'─' * 50}")
            print(f"  Ablation: {label}  (keeping {kept}/{n_dim1} Dim1 feats)")
        else:
            print(f"\n  {'─' * 50}")
            print(f"  Baseline: {label} ({kept} features)")

        X_sub = X_all[:, mask]

        accs = run_session_invariance(X_sub, y_all, y_enc, sess_all, le, substances)

        if not accs:
            print(f"    No valid iterations — skipping")
            continue

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        t_stat, p_val = stats.ttest_1samp(accs, chance)

        print(f"    Iterations: {len(accs)}")
        for i, a in enumerate(accs):
            print(f"      Iteration {i+1}: {a*100:.1f}%")
        print(f"    Mean ± std: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
        print(f"    t({len(accs)-1}) = {t_stat:.3f}, p = {p_val:.6f}")

        if label == "Dimension_1_Only":
            results["baseline_accuracy"] = mean_acc
            baseline = mean_acc
            delta_str = "—"
        else:
            delta = mean_acc - baseline
            delta_str = f"{delta*100:+.1f}%"
            results["ablations"][label] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "delta_from_dim1_baseline": delta,
                "n_features_kept": int(mask.sum()),
                "per_iteration": accs,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            }

        display_label = label.replace("_", " ")
        table_rows.append({
            "Feature_Group": display_label,
            "Mean_Accuracy": round(mean_acc * 100, 1),
            "Std_Accuracy": round(std_acc * 100, 1),
            "Delta_from_Dim1": delta_str,
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 6),
        })

    # ── Save CSV ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    with open(csv_path, "w") as f:
        f.write("Feature_Group,Mean_Accuracy(%),Std_Accuracy(%),Delta_from_Dim1(%),t_stat,p_value\n")
        for row in table_rows:
            f.write(f"{row['Feature_Group']},{row['Mean_Accuracy']},{row['Std_Accuracy']},{row['Delta_from_Dim1']},{row['t_stat']},{row['p_value']}\n")
    print(f"\n  Saved {csv_path}")

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n  {'─' * 80}")
    print(f"  Ablation Summary (Dimension 1 — Device-Agnostic Features)")
    print(f"  {'─' * 80}")
    print(f"  {'Feature Group':<30s} | {'Mean Acc':>8s} | {'Std':>5s} | {'Δ from Dim1':>10s} | {'p-value':>8s}")
    print(f"  {'─' * 80}")
    for row in table_rows:
        print(f"  {row['Feature_Group']:<30s} | {row['Mean_Accuracy']:>7.1f}% | {row['Std_Accuracy']:>4.1f}% | {row['Delta_from_Dim1']:>10s} | {row['p_value']:>8.6f}")

    # ── Bar chart ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        labels = [r["Feature_Group"] for r in table_rows]
        means = [r["Mean_Accuracy"] for r in table_rows]
        stds = [r["Std_Accuracy"] for r in table_rows]
        colors = ["seagreen" if r["Delta_from_Dim1"] == "—" else "coral" for r in table_rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Session-Invariance Accuracy (%)")
        ax.set_title("Figure 5: Ablation Study — Dimension 1 (Device-Agnostic Features)")
        ax.axhline(y=chance * 100, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Chance ({chance*100:.0f}%)")
        ax.legend()
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) + 0.5,
                    f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        plt.tight_layout()
        outpath = FIGURES_DIR / "fig5_ablation.png"
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Saved {outpath}")
    except ImportError:
        print("  SKIP ablation chart: matplotlib not available")

    # ── Save metrics ─────────────────────────────────────────────────
    results["mean_chance"] = float(chance)
    results["runnable"] = True
    results["duration_s"] = round(time.time() - t0, 2)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_5"] = results
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Elapsed: {results['duration_s']:.1f}s")
    return results


if __name__ == "__main__":
    run()
