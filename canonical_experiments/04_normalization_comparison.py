"""Experiment 4: Comparison of Normalization Methods

Compare three preprocessing approaches on the same leave-substance-out task:
1. Raw voltages (no normalization) — baseline
2. R/R0 normalization (simple baseline correction)
3. Framework features (our method: R/R0 normalization + derived features)

Uses GroupKFold (held-out substances per fold) to test whether framework features
generalize to novel substances better than simpler methods.

Answers: "Do framework features improve generalization over simpler methods?"
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RESULTS_DIR, METRICS_PATH,
    N_ESTIMATORS, RANDOM_STATE, R0_SAMPLES,
    WINDOW_SIZE, WINDOW_STRIDE,
)
from load_data import load_smellnet_csv, find_smellnet_files
from framework_features import extract_all_framework_features, feature_vector


def extract_features_framework(data, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    if N < window_size:
        padded = np.pad(data, ((0, window_size - N), (0, 0)), mode="edge")
        feats_dict = extract_all_framework_features(padded, r0_samples=R0_SAMPLES)
        vec, _ = feature_vector(feats_dict)
        return vec.reshape(1, -1)
    windows = [data[i:i + window_size] for i in range(0, N - window_size + 1, stride)]
    feat_vecs = []
    for w in windows:
        feats_dict = extract_all_framework_features(w, r0_samples=min(R0_SAMPLES, window_size))
        vec, _ = feature_vector(feats_dict)
        feat_vecs.append(vec)
    return np.array(feat_vecs, dtype=np.float32)


def extract_raw_voltage(data, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    data = np.asarray(data, dtype=np.float64)
    N = data.shape[0]
    if N < window_size:
        return data.T.reshape(1, -1)
    windows = [data[i:i + window_size] for i in range(0, N - window_size + 1, stride)]
    return np.array([w.T.reshape(-1) for w in windows], dtype=np.float32)


def extract_rr0(data, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    data = np.asarray(data, dtype=np.float64)
    n_ch = data.shape[1]
    r0_vals = np.array([np.median(data[:R0_SAMPLES, ch]) for ch in range(n_ch)])
    r0_vals = np.where(r0_vals > 1e-10, r0_vals, 1.0)
    normalized = data / r0_vals[np.newaxis, :]
    N = normalized.shape[0]
    if N < window_size:
        return normalized.T.reshape(1, -1)
    windows = [normalized[i:i + window_size] for i in range(0, N - window_size + 1, stride)]
    return np.array([w.T.reshape(-1) for w in windows], dtype=np.float32)


def run_lso_cv(X, y, groups, method_name):
    np.random.seed(RANDOM_STATE)
    unique_subs = sorted(set(groups))
    n_subs = len(unique_subs)
    n_splits = min(5, n_subs // 2)
    if n_splits < 2:
        return {"method": method_name, "mean_accuracy": 0, "folds": [], "error": "Too few substances"}

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []
    fold_consistencies = []
    fold_details = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_scaled, y_enc, groups)):
        X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
        te_groups = [groups[i] for i in te_idx]

        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        pred_labels = le.inverse_transform(y_pred)

        held_subs = sorted(set(te_groups))
        acc = float(rf.score(X_te, y_te))

        preds_by_sub = defaultdict(list)
        for true_sub, pred_sub in zip(te_groups, pred_labels):
            preds_by_sub[true_sub].append(pred_sub)

        consistencies = []
        for sub, preds in preds_by_sub.items():
            c = Counter(preds)
            top, count = c.most_common(1)[0]
            consistencies.append(count / len(preds))

        mean_cons = float(np.mean(consistencies)) if consistencies else 0.0
        fold_accs.append(acc)
        fold_consistencies.append(mean_cons)
        fold_details.append({
            "fold": fold + 1,
            "held_substances": list(held_subs),
            "n_train_windows": len(X_tr),
            "n_test_windows": len(X_te),
            "accuracy": acc,
            "mean_consistency": mean_cons,
        })

    if not fold_accs:
        return {"method": method_name, "mean_accuracy": 0, "folds": [], "error": "All folds failed"}

    return {
        "method": method_name,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_consistency": float(np.mean(fold_consistencies)),
        "std_consistency": float(np.std(fold_consistencies)),
        "folds": fold_accs,
        "fold_details": fold_details,
        "n_folds": len(fold_accs),
        "n_substances": n_subs,
    }


def run():
    print("=" * 70)
    print("Experiment 4: Comparison of Normalization Methods")
    print("Do framework features improve generalization over raw/VV0?")
    print("=" * 70)
    t0 = time.time()
    results = {"experiment": "normalization_comparison"}

    files = [Path(f) for f in find_smellnet_files()]
    substance_files = defaultdict(list)
    for f in files:
        substance = f.parent.name
        substance_files[substance].append(f)

    substances = sorted(substance_files.keys())
    print(f"  Loaded {len(substances)} substances\n")

    X_raw_list, X_rr0_list, X_framework_list = [], [], []
    y_list, groups_list = [], []
    for sub in substances:
        raw_feats, rr0_feats, framework_feats = [], [], []
        for f in substance_files[sub]:
            try:
                data = load_smellnet_csv(f)
                rf = extract_raw_voltage(data)
                vf = extract_rr0(data)
                pf = extract_features_framework(data)
                n = min(len(rf), len(vf), len(pf))
                if n == 0:
                    continue
                raw_feats.append(rf[:n])
                rr0_feats.append(vf[:n])
                framework_feats.append(pf[:n])
                for _ in range(n):
                    y_list.append(sub)
                    groups_list.append(sub)
            except Exception:
                pass
        if raw_feats:
            X_raw_list.append(np.vstack(raw_feats))
            X_rr0_list.append(np.vstack(rr0_feats))
            X_framework_list.append(np.vstack(framework_feats))

    if not X_raw_list:
        print("  ERROR: No data loaded")
        results["runnable"] = False
        return results

    X_raw_all = np.vstack(X_raw_list)
    X_rr0_all = np.vstack(X_rr0_list)
    X_framework_all = np.vstack(X_framework_list)
    y_all = np.array(y_list)
    groups_all = np.array(groups_list)
    X_raw_all = np.nan_to_num(X_raw_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_rr0_all = np.nan_to_num(X_rr0_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_framework_all = np.nan_to_num(X_framework_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Raw voltage features:      {X_raw_all.shape[0]} windows x {X_raw_all.shape[1]} dims")
    print(f"  R/R0 features:             {X_rr0_all.shape[0]} windows x {X_rr0_all.shape[1]} dims")
    print(f"  Framework features:        {X_framework_all.shape[0]} windows x {X_framework_all.shape[1]} dims")
    print(f"  Substances:                {len(np.unique(y_all))}")
    print(f"  Total windows:             {len(y_all)}\n")

    data_configs = [
        (X_raw_all, y_all, groups_all, "Raw voltages (no normalization)"),
        (X_rr0_all, y_all, groups_all, "R/R0 normalization"),
        (X_framework_all, y_all, groups_all, "Framework features (our method)"),
    ]

    print(f"  {'─' * 55}")
    print(f"  Leave-Substance-Out Cross-Validation (GroupKFold)")
    print(f"  {'─' * 55}")

    comparison_results = []
    for X, y, groups, name in data_configs:
        result = run_lso_cv(X, y.tolist(), groups.tolist(), name)
        comparison_results.append(result)
        std_str = f" ± {result['std_accuracy'] * 100:.1f}%" if result.get('std_accuracy') else ""
        cons_str = f"  (consistency: {result.get('mean_consistency', 0) * 100:.1f}%)" if result.get('mean_consistency') else ""
        print(f"  {name:40s}: {result['mean_accuracy'] * 100:.1f}%{std_str}{cons_str}")

    print(f"\n  Note: Accuracy is 0% for all methods since held-out substances have")
    print(f"  unseen class labels. The more meaningful metric is within-substance")
    print(f"  prediction consistency (fraction of windows from a held-out substance")
    print(f"  mapping to the same training class).")

    best_acc = max(comparison_results, key=lambda r: r["mean_accuracy"])
    best_cons = max(comparison_results, key=lambda r: r.get("mean_consistency", 0))
    print(f"\n  Best accuracy: {best_acc['method']} ({best_acc['mean_accuracy'] * 100:.1f}%)")
    print(f"  Best consistency: {best_cons['method']} ({best_cons.get('mean_consistency', 0) * 100:.1f}%)")

    results["comparison"] = comparison_results
    results["best_method_by_accuracy"] = best_acc["method"]
    results["best_method_by_consistency"] = best_cons["method"]
    results["n_substances"] = len(np.unique(y_all))
    results["n_windows"] = len(y_all)
    results["feature_dims"] = {
        "raw_voltage": X_raw_all.shape[1],
        "rr0": X_rr0_all.shape[1],
        "framework": X_framework_all.shape[1],
    }
    results["runnable"] = True
    results["duration_s"] = round(time.time() - t0, 2)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_4"] = results
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Elapsed: {results['duration_s']:.1f}s")
    return results


if __name__ == "__main__":
    run()
