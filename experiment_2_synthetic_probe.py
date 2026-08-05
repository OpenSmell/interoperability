#!/usr/bin/env python3
"""Experiment 2 — synthetic mechanism probe: where does single-point M break?

Phase 1 (real 13-file test) showed single-point M alignment gives +0.0 pp.
That test is confounded (exposure-intensity variance, ADC-clipped banana,
per-window R0, sr mismatch). This probe removes those confounds by injecting a
KNOWN per-channel device transform into SmellNet and measuring how much a
single-point M can undo it.

Model: RandomForest trained on clean SmellNet (device A), then scored on
simulated device-B windows.

Device-B transform (per live channel, per window):
    mult_ch = g_ch * a_ch^delta_ch          (a_ch = relative_amplitude)
    rel_amp'  = a_ch * mult_ch
    auc'/epd' = same mult
    sel_ratios + amplitude globals recomputed
  * delta = 0  -> pure per-channel gain g  (M's designed regime).
  * delta > 0  -> concentration-dependent distortion (exponent/b mismatch;
                  the regime M cannot fix).
  * batch noise sigma -> per-window log-normal gain wobble (intensity
                  variance; M = median ratio is robust but per-window
                  residual remains).

M variants:
  M_oracle : fit on the test substance's OWN windows (best possible).
  M_ref    : fit on the OTHER reference substances (honest, Phase-1-style).

Two models:
  magnitude-only (15 dims) -> isolates M's mathematical adequacy.
  full 91-dim            -> shows dilution by device-bound/kinetic features.

Run:  python interoperability/experiment_2_synthetic_probe.py
"""

import sys
import json
import itertools
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "opensmell"))
sys.path.insert(0, str(ROOT / "interoperability"))

from experiment_1_calibration import (collect, feature_columns, LIVE, OVERLAP,
    SMELLNET_CLASSES, apply_alignment, fit_m, N_ESTIMATORS, RANDOM_STATE)

SEL_PAIRS = [(2, 4), (2, 5), (4, 5)]
N_CLASSES = len(SMELLNET_CLASSES)

MAG_FEATS = []
for ch in LIVE:
    MAG_FEATS += [f"ch{ch}_da_relative_amplitude", f"ch{ch}_da_auc",
                  f"ch{ch}_da_endpoint_delta"]
MAG_FEATS += [f"sel_ratio_ch{i}_ch{j}" for i, j in SEL_PAIRS]
MAG_FEATS += ["global_max_delta_ratio", "global_mean_delta_ratio", "global_total_auc"]


def transform_device_b(vec, g, delta, sigma, cols, rng):
    """Apply the known per-channel magnitude transform -> simulated device B."""
    out = vec.copy()
    idx = {c: i for i, c in enumerate(cols)}
    for ch in LIVE:
        a = out[idx[f"ch{ch}_da_relative_amplitude"]]
        wobble = float(np.exp(rng.normal(0.0, sigma))) if sigma > 0 else 1.0
        mult = g * wobble * (a ** delta) if a > 0 else g * wobble
        out[idx[f"ch{ch}_da_relative_amplitude"]] = a * mult
        out[idx[f"ch{ch}_da_auc"]] *= mult
        out[idx[f"ch{ch}_da_endpoint_delta"]] *= mult
    amps = {ch: out[idx[f"ch{ch}_da_relative_amplitude"]] for ch in LIVE}
    amps = {ch: v for ch, v in amps.items() if v > 0}
    out[idx["global_max_delta_ratio"]] = max(amps.values()) if amps else 0.0
    out[idx["global_mean_delta_ratio"]] = float(np.mean(list(amps.values()))) if amps else 0.0
    out[idx["global_total_auc"]] = sum(out[idx[f"ch{ch}_da_auc"]] for ch in LIVE)
    for i, j in SEL_PAIRS:
        ai = out[idx[f"ch{i}_da_relative_amplitude"]]
        aj = out[idx[f"ch{j}_da_relative_amplitude"]]
        out[idx[f"sel_ratio_ch{i}_ch{j}"]] = (ai / aj) if aj > 0 else 0.0
    return out


def build(matrices, mag_only, idx):
    X = matrices if not mag_only else matrices[:, [idx[c] for c in MAG_FEATS]]
    return X


def run():
    sn, _ = collect()
    cols = feature_columns()
    idx = {c: i for i, c in enumerate(cols)}

    sn_vecs = [v for cls in SMELLNET_CLASSES for v in sn[cls][0]]
    sn_y = [cls for cls in SMELLNET_CLASSES for _ in range(len(sn[cls][0]))]
    sn_X = np.array(sn_vecs)
    by_cls = {cls: np.array(sn[cls][0]) for cls in SMELLNET_CLASSES}

    test_subs = [c for c in OVERLAP]  # all in SmellNet

    # Deterministic ceilings
    print("=" * 70, flush=True)
    print("EXPERIMENT 2 — SYNTHETIC MECHANISM PROBE", flush=True)
    print("=" * 70, flush=True)
    print(f"Windows: {len(sn_X)}  classes: {SMELLNET_CLASSES}  chance={100/N_CLASSES:.0f}%",
          flush=True)
    print(f"Magnitude-only dims: {len(MAG_FEATS)}  full dims: {len(cols)}", flush=True)

    grid = [
        {"g": 1.0, "delta": 0.0, "sigma": 0.0, "label": "identity"},
        {"g": 0.5, "delta": 0.0, "sigma": 0.0, "label": "pure gain 0.5x"},
        {"g": 1.0, "delta": 0.2, "sigma": 0.0, "label": "exp-shift 0.2"},
        {"g": 1.0, "delta": 0.5, "sigma": 0.0, "label": "exp-shift 0.5"},
        {"g": 0.5, "delta": 0.5, "sigma": 0.0, "label": "gain+exp-shift"},
        {"g": 1.0, "delta": 0.0, "sigma": 0.25, "label": "gain + batch noise 25%"},
        {"g": 1.0, "delta": 0.5, "sigma": 0.25, "label": "exp-shift 0.5 + noise"},
    ]
    seeds = [0, 1, 2]

    rows = []
    for cell in grid:
        g, delta, sigma, label = cell["g"], cell["delta"], cell["sigma"], cell["label"]
        accs = {"mag_raw": [], "mag_oracle": [], "mag_ref": [],
                "full_raw": [], "full_oracle": [], "full_ref": []}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            # Train models on clean A
            for mag_only in (True, False):
                Xtr = build(sn_X, mag_only, idx)
                le = LabelEncoder(); y = le.fit_transform(sn_y)
                sc = StandardScaler().fit(Xtr)
                rf = RandomForestClassifier(
                    n_estimators=150, random_state=RANDOM_STATE,
                    class_weight="balanced", n_jobs=-1).fit(sc.transform(Xtr), y)

                for s in test_subs:
                    B_test = np.array([transform_device_b(v, g, delta, sigma, cols, rng)
                                       for v in by_cls[s]])
                    ref_subs = [r for r in test_subs if r != s]
                    B_ref = np.array([transform_device_b(v, g, delta, sigma, cols, rng)
                                      for r in ref_subs for v in by_cls[r]])
                    A_ref = np.concatenate([by_cls[r] for r in ref_subs])

                    def sel(X):
                        return X if not mag_only else X[:, [idx[c] for c in MAG_FEATS]]

                    def acc_of(XB):
                        return accuracy_score(
                            [s] * len(XB),
                            le.inverse_transform(rf.predict(sc.transform(sel(XB)))))

                    key = "mag" if mag_only else "full"
                    accs[f"{key}_raw"].append(acc_of(B_test))

                    M_or, _ = fit_m(list(B_test), list(by_cls[s]))
                    accs[f"{key}_oracle"].append(acc_of(
                        np.array([apply_alignment(v, M_or, cols) for v in B_test])))

                    M_re, _ = fit_m(list(B_ref), list(A_ref))
                    accs[f"{key}_ref"].append(acc_of(
                        np.array([apply_alignment(v, M_re, cols) for v in B_test])))

        agg = {k: (float(np.mean(v)), float(np.std(v)), len(v))
               for k, v in accs.items()}
        rows.append({"label": label, "g": g, "delta": delta, "sigma": sigma, **agg})

        L = (f"\n[{label}]  (g={g}, delta={delta}, sigma={sigma})")
        print(L, flush=True)
        for m in ("mag", "full"):
            print(f"  {m:4s} RAW {agg[m+'_raw'][0]*100:5.1f}%  |  "
                  f"ALIGNED oracle {agg[m+'_oracle'][0]*100:5.1f}%  |  "
                  f"ALIGNED ref {agg[m+'_ref'][0]*100:5.1f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("READOUT", flush=True)
    print("=" * 70, flush=True)
    print("magnitude-only model isolates M's mathematical adequacy;", flush=True)
    print("full model shows dilution by device-bound/kinetic features.", flush=True)

    out = ROOT / "interoperability" / "experiment_2_synthetic_metrics.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    run()
