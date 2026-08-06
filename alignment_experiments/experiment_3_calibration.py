#!/usr/bin/env python3
"""Experiment 3 — UCI two-point (a,b) vs single-point M calibration.

Phase 3 question: Experiment 2 showed single-point M cannot undo exponent (b)
mismatch — a single per-channel gain cannot invert a concentration-dependent
power distortion (a_B = g·a_A^delta). Does a two-point log-log power-law fit
(g, delta) per channel remove the delta penalty that M cannot, under REAL UCI
drift (10 batches = one rig measured over 36 months, so later batches are a
"different device" in the transfer sense)?

Design:
  Device A = early batches, Device B = late batches (real drift).
  6 gases. Reference gases fit the per-channel alignment; a held-out gas is
  tested (leave-one-out reference discipline — the test gas's responses never
  enter the fit). A pooled fit over all 6 gases is reported as an optimistic
  bound.
  Amplitude subspace per sensor = the two steady-state amplitude features
  (raw + normalized) at feature indices 8*s+0, 8*s+1 (32 dims total).
  Quantile pairing: A and B samples of a reference gas are paired by rank
  within that gas (monotonic response assumed, concentration unlabelled) to
  build (a_A, a_B) pairs for the log-log regression. This is histogram
  matching, not a measured concentration pair.
  Alignment models (per amplitude feature-channel):
    single-point M :  a_B' = M·a_B           M = median(A)/median(B)
    two-point      :  log a_B = log g + d·log a_A  ->  a_B' = (a_B/g)^(1/d)
  Two classifiers (RF, trained on device A):
    magnitude-only (32 dims)  -> isolates the calibration subspace.
    full 128-dim              -> shows dilution by unaligned kinetic/drift
                                 features (the Exp-2 finding, on real data).

Run:  python interoperability/experiment_3_calibration.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent          # interoperability/alignment_experiments/
INTEROP = HERE.parent                            # interoperability/
REPO = INTEROP.parent                            # OpenSmell root
sys.path.insert(0, str(INTEROP / "canonical_experiments"))

from load_uci import load_all_batches, GAS_NAMES

N_SENSORS = 16
FEATS_PER_SENSOR = 8
MAG_IDX = [s * FEATS_PER_SENSOR + f
           for s in range(N_SENSORS) for f in (0, 1)]

RANDOM_STATE = 42
N_ESTIMATORS = 200
N_QUANTILES = 50

PRIMARY_SPLIT = {"name": "early 1-3 vs late 8-10",
                 "A": [1, 2, 3], "B": [8, 9, 10]}
SENSITIVITY_SPLIT = {"name": "early 1-5 vs late 6-10",
                     "A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}


def concat_batches(batches, nums):
    Xs, ys = [], []
    for b in nums:
        X, y = batches[b]
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def quantile_pairs(a_A, a_B, n=N_QUANTILES):
    """Pair A and B amplitude samples by rank (histogram matching).

    Returns (sorted_A, sorted_B) arrays of length n at matched quantiles.
    Only positive amplitudes participate (log-space fitting).
    """
    A = np.sort(a_A[a_A > 0])
    B = np.sort(a_B[a_B > 0])
    if len(A) < 2 or len(B) < 2:
        return np.array([]), np.array([])
    q = np.linspace(0.0, 1.0, n)
    qA = np.quantile(A, q)
    qB = np.quantile(B, q)
    return qA, qB


def fit_m(A_ref, B_ref):
    """Single-point M per amplitude feature: M = median(A)/median(B)."""
    M = {}
    for i, f in enumerate(MAG_IDX):
        a = A_ref[:, i]
        b = B_ref[:, i]
        med_a = np.median(a[a > 0]) if np.any(a > 0) else 0.0
        med_b = np.median(b[b > 0]) if np.any(b > 0) else 0.0
        M[f] = (med_a / med_b) if (med_b > 0 and np.isfinite(med_a)) else 1.0
    return M


def fit_power(A_ref, B_ref, gas_of_A, gas_of_B):
    """Two-point power law per amplitude feature.

    log B = log g + d·log A  fit by least squares on quantile-paired points
    pooled over reference gases. A and B are independent sample sets, so the
    per-gas selection uses its own gas-label array for each side.
    Returns dict f -> (g, d, r2, npts). d <= 0.05 is degenerate (gains only).
    """
    out = {}
    for i, f in enumerate(MAG_IDX):
        x, y = [], []
        for g in set(gas_of_A):
            a_g = A_ref[gas_of_A == g, i]
            b_g = B_ref[gas_of_B == g, i]
            if len(a_g) < 2 or len(b_g) < 2:
                continue
            qA, qB = quantile_pairs(a_g, b_g)
            if len(qA) < 2:
                continue
            x.extend(np.log(qA))
            y.extend(np.log(qB))
        x = np.array(x)
        y = np.array(y)
        if len(x) < 10:
            out[f] = (1.0, 1.0, 0.0, len(x))
            continue
        d, logg = np.polyfit(x, y, 1)
        resid = y - (d * x + logg)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[f] = (float(np.exp(logg)), float(d), float(r2), int(len(x)))
    return out


def apply_m(XB_mag, M):
    out = XB_mag.copy()
    for i, f in enumerate(MAG_IDX):
        m = M[f]
        if m == 1.0:
            continue
        pos = out[:, i] > 0
        out[pos, i] *= m
    return out


def apply_power(XB_mag, gd, M):
    out = XB_mag.copy()
    for i, f in enumerate(MAG_IDX):
        g, d, _, _ = gd[f]
        if d <= 0.05 or g <= 0:
            m = M[f]
            if m != 1.0:
                pos = out[:, i] > 0
                out[pos, i] *= m
            continue
        pos = out[:, i] > 0
        out[pos, i] = (out[pos, i] / g) ** (1.0 / d)
    return out


def evaluate_split(split, batches, gas_names):
    XA, yA = concat_batches(batches, split["A"])
    XB, yB = concat_batches(batches, split["B"])

    le = {g: i for i, g in enumerate(gas_names)}
    yA_c = np.array([le[GAS_NAMES[g]] for g in yA])
    yB_c = np.array([le[GAS_NAMES[g]] for g in yB])

    magA = XA[:, MAG_IDX]
    magB = XB[:, MAG_IDX]

    # ---- Train the two classifiers on device A ----
    models = {}
    for tag, Xtr in (("mag", magA), ("full", XA)):
        sc = StandardScaler().fit(Xtr)
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1).fit(sc.transform(Xtr), yA_c)
        models[tag] = (rf, sc)

    # gas label of each B sample, for LOO reference filtering
    gas_of_B = np.array([GAS_NAMES[g] for g in yB])

    def acc_of(tag, X, idx):
        rf, sc = models[tag]
        return accuracy_score(yB_c[idx], rf.predict(sc.transform(X)))

    # LOO: held-out gas never contributes to the alignment fit
    gas_of_A = np.array([GAS_NAMES[g] for g in yA])

    acc = {m: {c: [] for c in ("raw", "m", "power")}
           for m in ("mag", "full")}
    for held in gas_names:
        ref_A_mask = gas_of_A != held
        ref_B_mask = gas_of_B != held

        M = fit_m(magA[ref_A_mask], magB[ref_B_mask])
        gd = fit_power(magA[ref_A_mask], magB[ref_B_mask],
                       gas_of_A[ref_A_mask], gas_of_B[ref_B_mask])

        test_idx = np.where(~ref_B_mask)[0]
        Xt_mag = magB[test_idx]
        Xt_full = XB[test_idx]

        Xt_m = apply_m(Xt_mag, M)
        Xt_p = apply_power(Xt_mag, gd, M)

        for m in ("mag", "full"):
            def X_for(tag, Xm, Xf):
                return Xm if tag == "mag" else Xf

            acc[m]["raw"].append(acc_of(m, X_for(m, Xt_mag, Xt_full), test_idx))
            acc[m]["m"].append(acc_of(m, X_for(m, Xt_m, Xt_full), test_idx))
            acc[m]["power"].append(acc_of(m, X_for(m, Xt_p, Xt_full), test_idx))

    # ---- Pooled (optimistic) fit over all 6 gases ----
    M_all = fit_m(magA, magB)
    gd_all = fit_power(magA, magB, gas_of_A, gas_of_B)
    Xt_m = apply_m(magB, M_all)
    Xt_p = apply_power(magB, gd_all, M_all)
    pooled = {}
    for m in ("mag", "full"):
        pooled[m] = {
            "raw": acc_of(m, magB if m == "mag" else XB,
                          np.arange(len(yB))),
            "m": acc_of(m, Xt_m if m == "mag" else XB, np.arange(len(yB))),
            "power": acc_of(m, Xt_p if m == "mag" else XB, np.arange(len(yB))),
        }

    # ---- Within-device ceiling: train AND test on device A ----
    ceiling = {}
    for m in ("mag", "full"):
        Xtr = magA if m == "mag" else XA
        sc = StandardScaler().fit(Xtr)
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1).fit(sc.transform(Xtr), yA_c)
        ceiling[m] = float(accuracy_score(yA_c, rf.predict(sc.transform(Xtr))))

    return acc, pooled, ceiling, XA, yA, XB, yB


def report(acc, pooled, ceiling, split, gas_names):
    lines = []
    L = lambda s="": lines.append(s)

    L("=" * 72)
    L("EXPERIMENT 3 — UCI TWO-POINT (a,b) vs SINGLE-POINT M CALIBRATION")
    L(f"Split: {split['name']}   A={split['A']}  B={split['B']}")
    L("=" * 72)
    L(f"Chance (6 classes) = {100/6:.1f}%  |  magnitude-only dims = {len(MAG_IDX)}  |  full dims = 128")
    L("")
    L("Leave-one-out (test gas excluded from the alignment fit):")
    L(f"{'model':<6} {'condition':<8} {'mean acc':>9}  per-gas accs")
    for m in ("mag", "full"):
        for c in ("raw", "m", "power"):
            vals = acc[m][c]
            mean = np.mean(vals) * 100
            per = " ".join(f"{v*100:5.1f}" for v in vals)
            L(f"{m:<6} {c:<8} {mean:6.1f}%     {per}")
    L("")
    L("Pooled fit (all 6 gases — optimistic upper bound):")
    for m in ("mag", "full"):
        for c in ("raw", "m", "power"):
            L(f"{m:<6} {c:<8} {pooled[m][c]*100:6.1f}%")
    L("")
    L(f"Within-device ceiling (train+test on A): "
      f"mag {ceiling['mag']*100:.1f}%   full {ceiling['full']*100:.1f}%")
    L("")
    L("Delta (aligned minus raw), LOO mean:")
    for m in ("mag", "full"):
        dm = (np.mean(acc[m]["m"]) - np.mean(acc[m]["raw"])) * 100
        dp = (np.mean(acc[m]["power"]) - np.mean(acc[m]["raw"])) * 100
        L(f"  {m:<6}  single-point M: {dm:+5.1f} pp   two-point power: {dp:+5.1f} pp")
    return lines


def run():
    batches = load_all_batches()
    gas_names = list(GAS_NAMES.values())

    results = {}
    all_lines = []

    for split in (PRIMARY_SPLIT, SENSITIVITY_SPLIT):
        print(f"\n=== Split: {split['name']} ===", flush=True)
        acc, pooled, ceiling, XA, yA, XB, yB = evaluate_split(
            split, batches, gas_names)
        lines = report(acc, pooled, ceiling, split, gas_names)
        all_lines.extend(lines)
        print("\n".join(lines), flush=True)

        results[split["name"]] = {
            "acc": {m: {c: [float(v) for v in acc[m][c]]
                        for c in ("raw", "m", "power")} for m in ("mag", "full")},
            "pooled": {m: {c: float(v) for c, v in pooled[m].items()}
                       for m in ("mag", "full")},
            "ceiling": {m: float(ceiling[m]) for m in ("mag", "full")},
        }

        # ---- Fitted parameters on the primary split (pooled) ----
        if split is PRIMARY_SPLIT:
            gas_of_A_all = np.array([GAS_NAMES[g] for g in yA])
            gas_of_B_all = np.array([GAS_NAMES[g] for g in yB])
            M_all = fit_m(XA[:, MAG_IDX], XB[:, MAG_IDX])
            gd_all = fit_power(XA[:, MAG_IDX], XB[:, MAG_IDX],
                               gas_of_A_all, gas_of_B_all)
            diag = []
            diag.append("Pooled fit parameters (primary split), first 8 amplitude features:")
            diag.append(f"{'feat':<6} {'sensor':<7} {'M':>9} {'g':>10} {'d':>8} {'R2':>7}")
            for i, f in enumerate(MAG_IDX[:8]):
                g, d, r2, npts = gd_all[f]
                diag.append(f"{f:<6} {f//8:<7} {M_all[f]:>9.3f} {g:>10.4f} "
                            f"{d:>8.3f} {r2:>7.3f}")
            d_med = np.median([gd_all[f][1] for f in MAG_IDX])
            n_degenerate = sum(1 for f in MAG_IDX if gd_all[f][1] <= 0.05)
            diag.append(f"\nMedian fitted exponent d = {d_med:.3f}; "
                        f"degenerate (d<=0.05) channels: {n_degenerate}/32")
            all_lines.extend(diag)
            results[split["name"]]["fit_parameters"] = {
                "median_exponent_d": float(d_med),
                "n_degenerate_channels": int(n_degenerate),
                "channels": [{"feat": f, "M": M_all[f],
                              "g": gd_all[f][0], "d": gd_all[f][1],
                              "r2": gd_all[f][2], "npts": gd_all[f][3]}
                             for f in MAG_IDX],
            }
            print("\n".join(diag), flush=True)

    all_lines.append("")
    all_lines.append("VERDICT (see experiment_3_analysis.md)")
    all_lines.append("- Two-point power law is justified only if its LOO gain "
                     "exceeds single-point M by a material margin.")
    all_lines.append("- If both fail, feature discipline (magnitude-only) "
                     "remains the binding lever, not calibration order.")

    out = HERE / "results" / "experiment_3_calibration_result.txt"
    out.write_text("\n".join(all_lines) + "\n")
    print(f"\nWrote {out}", flush=True)

    (HERE / "results" / "experiment_3_calibration_metrics.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"Wrote metrics JSON", flush=True)


if __name__ == "__main__":
    run()
