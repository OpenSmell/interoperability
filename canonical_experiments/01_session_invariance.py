"""Experiment 1: Session-Invariance

Test if framework features preserve session-invariant information by training
a classifier on framework features extracted from some sessions and testing on
held-out sessions. Uses a one-sample t-test comparing 6 random session-holdout
accuracy values against chance level (1/N_substances).

Answers: "What is the substance, independent of when it was measured?"
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

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
    print("Experiment 1: Session-Invariance")
    print("Do framework features preserve session-invariant information?")
    print("=" * 70)
    t0 = time.time()
    results = {"experiment": "session_invariance"}

    all_data = load_all_smellnet_data()
    if not all_data:
        print("  ERROR: No SmellNet data found")
        results["runnable"] = False
        results["error"] = "No SmellNet data"
        return results

    # Filter to substances with >= 2 sessions
    multi_ses = {s: groups for s, groups in all_data.items() if len(groups) >= 2}
    substances = sorted(multi_ses.keys())
    print(f"  Substances with >=2 sessions: {len(substances)}")
    if len(substances) < 3:
        print("  Insufficient multi-session substances")
        results["runnable"] = False
        return results

    chance_level = 1.0 / len(substances)

    # Extract features for all windows
    X_chunks, y_list, sess_list = [], [], []
    for substance in substances:
        for session_num, data_array in multi_ses[substance]:
            feats = extract_features(data_array)
            X_chunks.append(feats)
            for _ in range(len(feats)):
                y_list.append(substance)
                sess_list.append(session_num)

    if not X_chunks:
        print("  No data loaded")
        results["runnable"] = False
        return results

    X_all = np.vstack(X_chunks)
    y_all = np.array(y_list)
    sess_all = np.array(sess_list)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    print(f"  Total windows: {len(y_all)}, classes: {len(le.classes_)}")
    print(f"  Feature dim: {X_all.shape[1]}")

    # Build per-substance session-to-indices map
    sub_ses_indices = defaultdict(lambda: defaultdict(list))
    for idx, (sub, ses) in enumerate(zip(y_list, sess_list)):
        sub_ses_indices[sub][ses].append(idx)

    N_ITERATIONS = 6
    rng = np.random.RandomState(RANDOM_STATE)

    session_accs = []
    print(f"\n  Chance level: {chance_level:.4f} ({chance_level * 100:.1f}%)")
    print(f"  Running {N_ITERATIONS} iterations — for each substance, hold out ~1/3 of its sessions as test\n")

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
            print(f"    Iteration {i + 1}: SKIP (train={len(train_idx)}, test={len(test_idx)})")
            continue

        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        if len(np.unique(y_tr)) < 2:
            print(f"    Iteration {i + 1}: SKIP (<2 classes in train)")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr_s, y_tr)
        preds = rf.predict(X_te_s)
        acc = accuracy_score(y_te, preds)
        session_accs.append(float(acc))
        print(f"    Iteration {i + 1}: train={len(train_idx)}, test={len(test_idx)}, "
              f"accuracy={acc:.4f} ({acc * 100:.1f}%)")

    if len(session_accs) == 0:
        print("  No valid iterations")
        results["runnable"] = False
        return results

    overall_acc = float(np.mean(session_accs))
    t_stat, p_value = stats.ttest_1samp(session_accs, chance_level)

    results["session_accuracies"] = session_accs
    results["mean_session_accuracy"] = overall_acc
    results["chance_level"] = float(chance_level)
    results["n_substances"] = len(substances)
    results["n_iterations"] = len(session_accs)
    results["t_statistic"] = float(t_stat)
    results["p_value"] = float(p_value)
    results["significant"] = bool(p_value < 0.05)

    print(f"\n  {'─' * 50}")
    print(f"  Results Summary")
    print(f"  {'─' * 50}")
    print(f"  Number of substances:      {len(substances)}")
    print(f"  Iterations completed:      {len(session_accs)}")
    print(f"  Chance level:              {chance_level:.4f} ({chance_level * 100:.1f}%)")
    print(f"  Mean accuracy:             {overall_acc:.4f} ({overall_acc * 100:.1f}%)")
    print(f"  Per-iteration accuracies:  {[f'{a:.4f}' for a in session_accs]}")
    print(f"  One-sample t-test:         t({len(session_accs) - 1}) = {t_stat:.3f}, p = {p_value:.6f}")
    result_str = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    print(f"  Result:                    Session-invariance {result_str}")

    results["runnable"] = True
    results["duration_s"] = round(time.time() - t0, 2)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    metrics["experiment_1"] = results
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  Elapsed: {results['duration_s']:.1f}s")
    return results


if __name__ == "__main__":
    run()
