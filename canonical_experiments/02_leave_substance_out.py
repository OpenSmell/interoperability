"""Experiment 2: Leave-Substance-Out Generalization

Test whether framework features generalize to completely unseen substances.
For each fold, held-out substances are never seen during training (GroupKFold).
Within-substance prediction consistency measures how often windows from the same
held-out substance map to the same training class.

Answers: "Do framework features generalize to novel substances?"
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, METRICS_PATH, N_ESTIMATORS, RANDOM_STATE, R0_SAMPLES, WINDOW_SIZE, WINDOW_STRIDE
from load_data import load_all_smellnet_data
from framework_features import extract_all_framework_features, feature_vector


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


def run():
    print("=" * 70)
    print("Experiment 2: Leave-Substance-Out Generalization")
    print("Do framework features generalize to unseen substances?")
    print("=" * 70)
    t0 = time.time()
    results = {"experiment": "leave_substance_out"}

    all_data = load_all_smellnet_data()
    if not all_data:
        print("  ERROR: No SmellNet data found")
        results["runnable"] = False
        results["error"] = "No SmellNet data"
        return results

    substances = sorted(all_data.keys())
    print(f"  Total substances: {len(substances)}")

    # Build feature matrix by concatenating all windows across all sessions
    X_chunks, y_list, group_list = [], [], []
    for sub in substances:
        for session_num, data_array in all_data[sub]:
            feats = extract_features(data_array)
            X_chunks.append(feats)
            n = len(feats)
            y_list.extend([sub] * n)
            group_list.extend([sub] * n)

    if not X_chunks:
        print("  ERROR: No features extracted")
        results["runnable"] = False
        return results

    X_all = np.vstack(X_chunks)
    y_all = np.array(y_list)
    groups_all = np.array(group_list)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    print(f"  Total windows: {len(y_all)}")
    print(f"  Feature dim:   {X_all.shape[1]}")
    print(f"  Substances:    {len(np.unique(groups_all))}")

    # GroupKFold: each fold holds out one or more entire substances
    n_splits = min(5, len(substances) // 2)
    if n_splits < 2:
        print("  ERROR: Too few substances for cross-validation")
        results["runnable"] = False
        return results

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    print(f"\n  Using GroupKFold with {n_splits} splits — substances never shared across train/test")
    print(f"  {'─' * 55}")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_enc, groups_all)):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
        te_groups = groups_all[te_idx]
        unique_te_groups = sorted(set(te_groups))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_te_s)
        pred_labels = le.inverse_transform(y_pred)

        # Per-substance consistency
        preds_by_sub = defaultdict(list)
        for true_sub, pred_sub in zip(te_groups, pred_labels):
            preds_by_sub[true_sub].append(pred_sub)

        consistencies = []
        sub_details = []
        for sub, preds in sorted(preds_by_sub.items()):
            c = Counter(preds)
            top_label, top_count = c.most_common(1)[0]
            cons = top_count / len(preds)
            consistencies.append(cons)
            sub_details.append({
                "substance": sub,
                "n_windows": len(preds),
                "top_prediction": top_label,
                "top_fraction": round(cons, 4),
                "unique_predictions": len(c),
            })

        fold_cons = float(np.mean(consistencies)) if consistencies else 0.0
        fold_results.append({
            "fold": fold + 1,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_held_out_substances": len(unique_te_groups),
            "held_out_substances": list(unique_te_groups),
            "mean_consistency": fold_cons,
            "per_substance": sub_details,
        })

        print(f"  Fold {fold + 1}: train={len(X_tr)}, test={len(X_te)}, "
              f"held-out={len(unique_te_groups)} substances, "
              f"mean_consistency={fold_cons:.3f}")

    if not fold_results:
        print("  ERROR: No folds completed")
        results["runnable"] = False
        return results

    mean_consistency = float(np.mean([f["mean_consistency"] for f in fold_results]))
    std_consistency = float(np.std([f["mean_consistency"] for f in fold_results]))

    print(f"\n  {'─' * 55}")
    print(f"  Overall Results")
    print(f"  {'─' * 55}")
    print(f"  Folds:                    {len(fold_results)}")
    print(f"  Total substances:         {len(substances)}")
    print(f"  Mean consistency:         {mean_consistency:.3f} ({mean_consistency * 100:.1f}%)")
    print(f"  Std consistency:          {std_consistency:.3f} ({std_consistency * 100:.1f}%)")
    cons_strs = [f"{f['mean_consistency']:.3f}" for f in fold_results]
    print(f"  Per-fold consistencies:   {cons_strs}")

    results["n_substances"] = len(substances)
    results["n_folds"] = len(fold_results)
    results["n_windows"] = len(y_all)
    results["feature_dim"] = X_all.shape[1]
    results["mean_consistency"] = mean_consistency
    results["std_consistency"] = std_consistency
    results["fold_results"] = fold_results
    results["runnable"] = True
    results["duration_s"] = round(time.time() - t0, 2)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_2"] = results
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Elapsed: {results['duration_s']:.1f}s")
    return results


if __name__ == "__main__":
    run()
