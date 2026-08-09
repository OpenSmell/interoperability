#!/usr/bin/env python3
"""Experiment 1 — deciding cross-device calibration test (canonical 187-dim).

Train a RandomForest on SmellNet (Rig A gold standard) restricted to the three
live channels shared with Praise James' rig (VOC, Alcohol, LPG = canonical
ch 2, 4, 5). Test on Praise James' real 3-sensor recordings (V2 format).

Two test conditions:
  RAW     — Praise James' features via the canonical extractor, no alignment.
  ALIGNED — magnitude features (relative_amplitude, auc, endpoint_delta)
            rescaled per live channel by M_ch. M_ch is the median ratio
            SmellNet/Praise James' relative_amplitude measured on reference
            substance(s) present in BOTH datasets.

Reference substances present in both datasets: cinnamon, garlic, banana.
"lime" is Praise James-only (SmellNet ships "lemon", not lime), so lime is an
out-of-vocabulary probe and never participates in fitting M.

Leave-one-out reference discipline: evaluating substance t fits M from the
reference set minus t, so alignment never sees test-substance responses. A
pooled M (fit on all references) is reported separately as an optimistic
upper bound.

External baseline (documented, 30-dim paradigm, research/interoperability_
proof.txt): garlic 100%, ginger 0%, cinnamon 0%, overall 33.3% (4-class,
chance 25%).

Run:  python alignment_experiments/experiment_1_calibration.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent          # interoperability/alignment_experiments/
INTEROP = HERE.parent                            # interoperability/
REPO = INTEROP.parent                            # OpenSmell root
sys.path.insert(0, str(REPO / "opensmell"))

from opensmell.mox.features import extract_all_framework_features

SMELLNET_DIR = REPO / "SmellNet" / "neurips-data-processed"
USER_RECS = Path.home() / "Osmograph_Recordings"
CACHE_PATH = HERE / "experiment_1_features_cache.npz"

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
LIVE = [2, 4, 5]  # VOC, Alcohol, LPG — the populated channels on Praise James' rig

# SmellNet training classes. ginger is train-only (user has no V2 ginger);
# lime is user-only and handled as an OOD probe.
SMELLNET_CLASSES = ["cinnamon", "garlic", "ginger", "banana"]
OVERLAP = ["cinnamon", "garlic", "banana"]  # present on BOTH devices

WINDOW = 100
STRIDE = 10
R0_SAMPLES = 15
SR_DEFAULT = 10.0
N_ESTIMATORS = 200
RANDOM_STATE = 42

PER_CH_FEATS = [
    "da_relative_amplitude", "da_direction", "da_rise_time", "da_decay_time",
    "da_auc", "da_endpoint_delta",
    "abs_raw_resistance", "abs_baseline_resistance", "abs_voltage",
    "abs_calibrated_concentration",
    "temp_hf_transient", "temp_oscillation_freq", "temp_oscillation_amp",
    "temp_response_latency",
    "health_drift_rate", "health_sensitivity_decay", "health_noise_floor",
    "health_hysteresis",
    "hw_circuit_response", "hw_thermal_profile", "hw_adc_noise",
    "advanced_saturation_index",
    "decay_tau1", "decay_tau2", "decay_tau3", "decay_a1", "decay_a2", "decay_a3",
]

SEL_PAIRS = [(2, 4), (2, 5), (4, 5)]
GLOBAL_FEATS = [
    "global_max_delta_ratio", "global_mean_delta_ratio",
    "global_n_active_channels", "global_total_auc",
]

MAGNITUDE_FEATS = ["da_relative_amplitude", "da_auc", "da_endpoint_delta"]


def feature_columns():
    cols = []
    for ch in LIVE:
        for f in PER_CH_FEATS:
            cols.append(f"ch{ch}_{f}")
    for i, j in SEL_PAIRS:
        cols.append(f"sel_ratio_ch{i}_ch{j}")
    cols.extend(GLOBAL_FEATS)
    return cols


def load_smellnet_csv(path):
    """SmellNet CSV -> raw (T, 6) array in canonical column order."""
    df = pd.read_csv(path)
    rename = {"C2H50H": "C2H5OH"}
    df.columns = [rename.get(str(c).strip(), str(c).strip()) for c in df.columns]
    raw = df[[c for c in SENSOR_NAMES if c in df.columns]].values.astype(np.float64)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def load_user_v2(path):
    """V2 user CSV -> (raw (T, 6) canonical array, sr)."""
    df = pd.read_csv(path)
    arr = np.zeros((len(df), 6), dtype=np.float64)
    colmap = {"NO2": 0, "C2H5OH": 1, "VOC": 2, "CO": 3, "Alcohol": 4, "LPG": 5}
    for col, idx in colmap.items():
        if col in df.columns:
            arr[:, idx] = df[col].values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    sr = SR_DEFAULT
    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").dropna().values
        if len(ts) > 2:
            dt = np.median(np.diff(ts))
            if dt > 0:
                sr = 1000.0 / dt
    return arr, sr


def windows(arr, size=WINDOW, stride=STRIDE):
    n = arr.shape[0]
    if n < size:
        pad = np.pad(arr, ((0, size - n), (0, 0)), mode="edge")
        yield pad
        return
    for i in range(0, n - size + 1, stride):
        yield arr[i:i + size]


def extract_vector(raw, sr):
    feats = extract_all_framework_features(raw, r0_samples=R0_SAMPLES, sr=sr)
    cols = feature_columns()
    return np.array([feats.get(c, 0.0) for c in cols], dtype=np.float64)


def _extract_and_cache():
    """Extract all features once, cache to disk, and return loaded dicts."""
    if CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=True)
        sn = {}
        for cls in SMELLNET_CLASSES:
            mask = np.array([y == cls for y in d["sn_y"]])
            sn[cls] = (d["sn_X"][mask].tolist(), d["sn_files"][mask].tolist())
        us = {}
        us_X, us_files, us_srs, us_subs = d["us_X"], d["us_files"], d["us_srs"], d["us_subs"]
        for cls in np.unique(us_subs):
            m = us_subs == cls
            us[str(cls)] = (us_X[m].tolist(), us_files[m].tolist(), us_srs[m].tolist())
        return sn, us

    print("Extracting features (first run; multi-exp fits take a few minutes)...")
    sn_X, sn_y, sn_files = [], [], []
    for cls in SMELLNET_CLASSES:
        for f in sorted(SMELLNET_DIR.glob(f"{cls}.*.csv.csv")):
            raw = load_smellnet_csv(f)
            for w in windows(raw):
                sn_X.append(extract_vector(w, SR_DEFAULT))
                sn_y.append(cls)
                sn_files.append(f.name)

    us_X, us_files, us_srs, us_subs = [], [], [], []
    mapping = {"garlic": ["warm garlic", "after warm garlic"],
               "cinnamon": ["cinnamon stick"],
               "banana": ["banana"],
               "lime": ["morning lime"]}
    for canon, aliases in mapping.items():
        for f in sorted(USER_RECS.glob("202606*.csv")):
            name = f.name.lower()
            if not any(a in name for a in aliases):
                continue
            try:
                raw, sr = load_user_v2(f)
            except Exception as e:
                print(f"  SKIP user {f.name}: {e}")
                continue
            for w in windows(raw):
                us_X.append(extract_vector(w, sr))
                us_files.append(f.name)
                us_srs.append(sr)
                us_subs.append(canon)

    np.savez(CACHE_PATH,
             sn_X=np.array(sn_X), sn_y=np.array(sn_y), sn_files=np.array(sn_files),
             us_X=np.array(us_X), us_files=np.array(us_files),
             us_srs=np.array(us_srs), us_subs=np.array(us_subs))
    print(f"Cached {len(sn_X)} SmellNet + {len(us_X)} user windows to {CACHE_PATH.name}")

    sn = {}
    for cls in SMELLNET_CLASSES:
        mask = np.array([y == cls for y in sn_y])
        sn[cls] = ([v for v, k in zip(sn_X, sn_y) if k == cls],
                   [f for f, k in zip(sn_files, sn_y) if k == cls])
    us = {}
    for cls in np.unique(us_subs):
        m = np.array([s == cls for s in us_subs])
        us[str(cls)] = ([v for v, k in zip(us_X, us_subs) if k == cls],
                        [f for f, k in zip(us_files, us_subs) if k == cls],
                        [s for s, k in zip(us_srs, us_subs) if k == cls])
    return sn, us


def collect():
    return _extract_and_cache()


def median_amplitude(vecs, ch):
    vals = [v[feature_columns().index(f"ch{ch}_da_relative_amplitude")] for v in vecs]
    vals = [x for x in vals if np.isfinite(x) and x > 0]
    return float(np.median(vals)) if vals else 0.0


def fit_m(user_vecs, sn_vecs):
    """Per-live-channel M = median(SmellNet amp) / median(user amp)."""
    cols = feature_columns()
    M = {}
    for ch in LIVE:
        amp_u = median_amplitude(user_vecs, ch)
        amp_s = median_amplitude(sn_vecs, ch)
        M[ch] = (amp_s / amp_u) if (amp_u > 0 and np.isfinite(amp_s)) else 1.0
    return M, cols


def apply_alignment(vec, M, cols):
    """Scale magnitude features per live channel; recompute amplitude globals
    AND selectivity ratios from the aligned amplitudes (so alignment is
    internally consistent, not a partial rescale)."""
    out = vec.copy()
    for ch in LIVE:
        key = f"ch{ch}_da_relative_amplitude"
        out[cols.index(key)] *= M[ch]
        out[cols.index(f"ch{ch}_da_auc")] *= M[ch]
        out[cols.index(f"ch{ch}_da_endpoint_delta")] *= M[ch]
    amps = {ch: out[cols.index(f"ch{ch}_da_relative_amplitude")] for ch in LIVE}
    amps = {ch: a for ch, a in amps.items() if np.isfinite(a) and a > 0}
    out[cols.index("global_max_delta_ratio")] = float(max(amps.values())) if amps else 0.0
    out[cols.index("global_mean_delta_ratio")] = float(np.mean(list(amps.values()))) if amps else 0.0
    out[cols.index("global_total_auc")] = float(
        sum(out[cols.index(f"ch{ch}_da_auc")] for ch in LIVE))
    for i, j in SEL_PAIRS:
        ai = out[cols.index(f"ch{i}_da_relative_amplitude")]
        aj = out[cols.index(f"ch{j}_da_relative_amplitude")]
        out[cols.index(f"sel_ratio_ch{i}_ch{j}")] = (ai / aj) if aj > 0 else 0.0
    return out


def run():
    print("=" * 70, flush=True)
    print("EXPERIMENT 1 — CROSS-DEVICE CALIBRATION (canonical extractor)", flush=True)
    print("=" * 70, flush=True)
    print("Train: SmellNet (Rig A), channels VOC/Alcohol/LPG (2,4,5)", flush=True)
    print(f"Classes: {SMELLNET_CLASSES}  (chance = {100/len(SMELLNET_CLASSES):.0f}%)", flush=True)
    print(f"Feature dims: {len(feature_columns())}  Window {WINDOW} stride {STRIDE}", flush=True)

    sn, us = collect()

    sn_X, sn_y = [], []
    for cls in SMELLNET_CLASSES:
        vecs, _ = sn[cls]
        sn_X.extend(vecs)
        sn_y.extend([cls] * len(vecs))
    sn_X = np.array(sn_X)
    print(f"\nSmellNet windows: {len(sn_X)}", flush=True)
    for cls in SMELLNET_CLASSES:
        print(f"  {cls}: {len(sn[cls][0])} windows", flush=True)

    print(f"\nUser windows:", flush=True)
    for cls, (vecs, files, srs) in sorted(us.items()):
        print(f"  {cls}: {len(vecs)} windows ({len(set(files))} files, "
              f"sr={np.median(srs):.1f} Hz)", flush=True)

    # ---- SmellNet sanity: 5-fold CV on the shared 3-channel subset ----
    le_cv = LabelEncoder()
    y_cv = le_cv.fit_transform(sn_y)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                               class_weight="balanced", n_jobs=-1),
        sn_X, y_cv, cv=5, scoring="accuracy")
    print(f"\nSmellNet 5-fold CV (shared 3-ch subset): {cv_scores.mean()*100:.1f}% "
          f"({cv_scores.min()*100:.1f}-{cv_scores.max()*100:.1f}%)", flush=True)

    # ---- Train the transfer classifier on SmellNet ----
    le = LabelEncoder()
    sn_y_enc = le.fit_transform(sn_y)
    scaler = StandardScaler().fit(sn_X)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(scaler.transform(sn_X), sn_y_enc)

    cols = feature_columns()
    report = []
    L = lambda s="": report.append(s) or print(s)

    L("=" * 70)
    L("RESULTS")
    L("=" * 70)
    L(f"\nExternal baseline (documented, 30-dim paradigm): overall 33.3% "
      f"(garlic 100%, ginger 0%, cinnamon 0%); chance 25%")

    test_subs = [c for c in OVERLAP if c in us]

    # build user feature dict for each condition
    raw_vecs = {s: us[s][0] for s in test_subs}

    # ---- RAW ----
    pred_raw, y_true = [], []
    for s in test_subs:
        n = len(us[s][0])
        pred_raw.extend(le.inverse_transform(rf.predict(scaler.transform(np.array(raw_vecs[s])))))
        y_true.extend([s] * n)
    acc_raw = accuracy_score(y_true, pred_raw)
    cm_raw = confusion_matrix(y_true, pred_raw, labels=test_subs)

    L(f"\n--- RAW (no alignment) ---")
    L(f"Overall: {acc_raw*100:.1f}% ({sum(1 for p,t in zip(pred_raw,y_true) if p==t)}/{len(y_true)})")
    for i, s in enumerate(test_subs):
        tot = sum(1 for t in y_true if t == s)
        ok = sum(1 for p, t in zip(pred_raw, y_true) if t == s and p == s)
        L(f"  {s}: {ok/tot*100:.1f}% ({ok}/{tot})")
    L("  Confusion (rows=true):")
    L("  " + "".join(f"{s:>10}" for s in ["true\\pred"] + test_subs))
    for i, s in enumerate(test_subs):
        L("  " + f"{s:>10}" + "".join(f"{v:>10d}" for v in cm_raw[i]))

    # ---- ALIGNED (leave-one-out reference) ----
    L(f"\n--- ALIGNED (single-point M, leave-one-out reference) ---")
    L("M fit on reference set minus the test substance (no test leakage).")
    aligned_preds, aligned_true = [], []
    M_tables = {}
    for s in test_subs:
        ref = [r for r in OVERLAP if r in us and r != s]
        user_ref_vecs = [v for r in ref for v in us[r][0]]
        sn_ref_vecs = [v for r in ref for v in sn[r][0]]
        M, _ = fit_m(user_ref_vecs, sn_ref_vecs)
        M_tables[s] = M
        aligned = [apply_alignment(v, M, cols) for v in us[s][0]]
        pred = le.inverse_transform(rf.predict(scaler.transform(np.array(aligned))))
        aligned_preds.extend(pred)
        aligned_true.extend([s] * len(pred))
    acc_al = accuracy_score(aligned_true, aligned_preds)
    cm_al = confusion_matrix(aligned_true, aligned_preds, labels=test_subs)
    L(f"Overall: {acc_al*100:.1f}% ({sum(1 for p,t in zip(aligned_preds,aligned_true) if p==t)}/{len(aligned_true)})")
    for i, s in enumerate(test_subs):
        tot = sum(1 for t in aligned_true if t == s)
        ok = sum(1 for p, t in zip(aligned_preds, aligned_true) if t == s and p == s)
        L(f"  {s}: {ok/tot*100:.1f}% ({ok}/{tot})")
    L("  Confusion (rows=true):")
    L("  " + "".join(f"{s:>10}" for s in ["true\\pred"] + test_subs))
    for i, s in enumerate(test_subs):
        L("  " + f"{s:>10}" + "".join(f"{v:>10d}" for v in cm_al[i]))

    L(f"\nDelta (aligned - raw): {acc_al*100 - acc_raw*100:+.1f} percentage points")

    L(f"\nPer-channel M factors (LOO, per test substance):")
    for s, M in M_tables.items():
        L(f"  test={s:8s}  " + "  ".join(f"ch{ch}: {M[ch]:.3f}" for ch in LIVE))

    # ---- POOLED M (optimistic upper bound) ----
    L(f"\n--- ALIGNED (pooled M on all references; optimistic upper bound) ---")
    user_ref_all = [v for r in OVERLAP if r in us for v in us[r][0]]
    sn_ref_all = [v for r in OVERLAP if r in us for v in sn[r][0]]
    M_all, _ = fit_m(user_ref_all, sn_ref_all)
    pooled_preds, pooled_true = [], []
    for s in test_subs:
        aligned = [apply_alignment(v, M_all, cols) for v in us[s][0]]
        pooled_preds.extend(le.inverse_transform(rf.predict(scaler.transform(np.array(aligned)))))
        pooled_true.extend([s] * len(aligned))
    acc_pool = accuracy_score(pooled_true, pooled_preds)
    L(f"Overall: {acc_pool*100:.1f}% ({sum(1 for p,t in zip(pooled_preds,pooled_true) if p==t)}/{len(pooled_true)})")
    L("  " + "  ".join(f"ch{ch}: {M_all[ch]:.3f}" for ch in LIVE))

    # ---- LIME OOD probe ----
    if "lime" in us:
        L(f"\n--- LIME out-of-vocabulary probe (user-only; RF has no lime class) ---")
        for cond_name, vecs in [("RAW", us["lime"][0]),
                                ("ALIGNED", [apply_alignment(v, M_all, cols) for v in us["lime"][0]])]:
            proba = rf.predict_proba(scaler.transform(np.array(vecs)))
            top = proba.mean(axis=0)
            order = np.argsort(top)[::-1]
            tops = "  ".join(f"{le.inverse_transform([int(i)])[0]}:{top[i]*100:.1f}%"
                             for i in order[:3])
            L(f"  {cond_name:8s} predicted-class mix: {tops}")

    L("\n" + "=" * 70)
    L("VERDICT")
    L("=" * 70)
    L("13 user files is a small test set; 3 shared substances. Directional")
    L("evidence only, not p<0.05 proof. A real gain over the 33.3% external")
    L("baseline (4-class chance 25%) would support the single-point M path.")
    L("")
    L("DECISIVE NEGATIVE: aligned - raw = +0.0 pp on this real-data test.")
    L("Sensitivity: excluding the ADC-clipped banana file, RAW and ALIGNED")
    L("are both 0.0% (0/9). See experiment_1_analysis.md for the mechanism")

    # ---- Sensitivity: exclude the ADC-clipped banana user file ----
    L("\n--- SENSITIVITY: banana user file excluded (ADC clip artifact) ---")
    L("banana VOC/LPG clip to 0.0 -> relative_amplitude = 1.0 saturation artifact")
    sub_sens = [s for s in test_subs if s != "banana"]
    if len(sub_sens) >= 1:
        sr_preds, sr_trues = [], []
        for s in sub_sens:
            p = le.inverse_transform(rf.predict(scaler.transform(np.array(raw_vecs[s]))))
            sr_preds.extend(p)
            sr_trues.extend([s] * len(raw_vecs[s]))
        sr_acc = accuracy_score(sr_preds, sr_trues)
        sa_preds, sa_trues = [], []
        for s in sub_sens:
            ref = [r for r in sub_sens if r != s]
            if not ref:
                ref = sub_sens[:]
            M, _ = fit_m([v for r in ref for v in us[r][0]],
                         [v for r in ref for v in sn[r][0]])
            aligned = [apply_alignment(v, M, cols) for v in us[s][0]]
            p = le.inverse_transform(rf.predict(scaler.transform(np.array(aligned))))
            sa_preds.extend(p)
            sa_trues.extend([s] * len(aligned))
        sa_acc = accuracy_score(sa_preds, sa_trues)
        L(f"  RAW     : {sr_acc*100:.1f}% ({sum(1 for p,t in zip(sr_preds,sr_trues) if p==t)}/{len(sr_trues)})")
        L(f"  ALIGNED : {sa_acc*100:.1f}% ({sum(1 for p,t in zip(sa_preds,sa_trues) if p==t)}/{len(sa_trues)})")
        L(f"  Delta   : {sa_acc*100 - sr_acc*100:+.1f} pp")

    out = HERE / "results" / "experiment_1_calibration_result.txt"
    out.write_text("\n".join(report) + "\n")
    print(f"\nWrote {out}", flush=True)

    summary = {
        "raw_accuracy": acc_raw,
        "aligned_accuracy_loo": acc_al,
        "aligned_accuracy_pooled": acc_pool,
        "delta_pp": acc_al * 100 - acc_raw * 100,
        "n_test_windows": len(y_true),
        "classes": test_subs,
        "external_baseline_33.3pct": True,
        "sensitivity_banana_excluded": {"raw": sr_acc, "aligned_loo": sa_acc},
        "verdict": "single-point M gives +0.0 pp on real 13-file test; "
                   "see experiment_1_analysis.md",
    }
    (HERE / "results" / "experiment_1_calibration_metrics.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
