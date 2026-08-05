#!/usr/bin/env python3
"""Experiment 4 — CORAL distribution alignment on UCI drift.

Phase 4 (Qwen brief, Path A): instead of a per-feature scalar correction
(single-point M, two-point power), align the TARGET feature distribution to
the SOURCE with CORAL (Sun, Feng & Saenko, ICML 2016): a closed-form linear
map that whitens the target by its own covariance and recolors it to the
source covariance, plus a mean shift. This is the rigorous, cheap form of
"domain adaptation via feature alignment." It can correct cross-channel
correlation drift that no per-feature scalar can.

Question: does aligning B's distribution to A's (in the magnitude subspace
AND in full 128-dim space) recover transfer accuracy that RAW / M / two-point
cannot?

Conditions (same splits and classifiers as Exp 3, so numbers are directly
comparable):
  raw                - no alignment.
  coral-ref          - CORAL fit on reference A/B samples only (LOO-fair;
                       matches Exp 3's M/power discipline: the held-out gas
                       never contributes to the alignment).
  coral-all          - CORAL fit on ALL of B (transductive, standard
                       unsupervised DA; optimistic upper bound).
Each on the magnitude-only (32-dim) and full (128-dim) classifier.

Run:  python interoperability/experiment_4_coral.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "interoperability"))

from experiment_3_calibration import (
    concat_batches, load_all_batches, GAS_NAMES, MAG_IDX,
    PRIMARY_SPLIT, SENSITIVITY_SPLIT, N_ESTIMATORS, RANDOM_STATE,
)

N_DIM = 128
REG = 1e-3


def _eigh(mat):
    evals, evecs = np.linalg.eigh((mat + mat.T) / 2)
    evals = np.clip(evals, 1e-12, None)
    return evals, evecs


def _sqrt(mat):
    evals, evecs = _eigh(mat)
    return (evecs * np.sqrt(evals)) @ evecs.T


def _inv_sqrt(mat):
    evals, evecs = _eigh(mat)
    return (evecs / np.sqrt(evals)) @ evecs.T


def coral_params(Xs, Xt, reg=REG):
    """Return (W, mu_s, mu_t): (Xt - mu_t)@W + mu_s has source covariance."""
    mu_s = Xs.mean(axis=0)
    mu_t = Xt.mean(axis=0)
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False)
    Ct = np.cov(Xt, rowvar=False)
    scale_s = np.trace(Cs) / d
    scale_t = np.trace(Ct) / d
    Cs = Cs + reg * scale_s * np.eye(d)
    Ct = Ct + reg * scale_t * np.eye(d)
    W = _inv_sqrt(Ct) @ _sqrt(Cs)
    return W, mu_s, mu_t


def apply_coral(X, W, mu_s, mu_t):
    return (X - mu_t) @ W + mu_s


def run():
    batches = load_all_batches()
    gas_names = list(GAS_NAMES.values())

    results = {}
    all_lines = []

    for split in (PRIMARY_SPLIT, SENSITIVITY_SPLIT):
        XA, yA = concat_batches(batches, split["A"])
        XB, yB = concat_batches(batches, split["B"])
        le = {g: i for i, g in enumerate(gas_names)}
        yA_c = np.array([le[GAS_NAMES[g]] for g in yA])
        yB_c = np.array([le[GAS_NAMES[g]] for g in yB])
        gas_of_A = np.array([GAS_NAMES[g] for g in yA])
        gas_of_B = np.array([GAS_NAMES[g] for g in yB])

        # ---- Train classifiers on device A ----
        models = {}
        for tag, Xtr in (("mag", XA[:, MAG_IDX]), ("full", XA)):
            sc = StandardScaler().fit(Xtr)
            rf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1).fit(sc.transform(Xtr), yA_c)
            models[tag] = (rf, sc)

        def acc_of(tag, X, idx):
            rf, sc = models[tag]
            return accuracy_score(yB_c[idx], rf.predict(sc.transform(X)))

        def sub(X, tag):
            return X[:, MAG_IDX] if tag == "mag" else X

        # ---- LOO (coral-ref) and transductive (coral-all) ----
        acc = {m: {c: [] for c in ("raw", "coral_ref", "coral_all")}
               for m in ("mag", "full")}
        for held in gas_names:
            ref_A_mask = gas_of_A != held
            ref_B_mask = gas_of_B != held
            test_idx = np.where(~ref_B_mask)[0]
            for m in ("mag", "full"):
                XB_t = sub(XB, m)[test_idx]
                acc[m]["raw"].append(acc_of(m, XB_t, test_idx))
                for cond, Xs_ref, Xt_ref in (
                        ("coral_ref", sub(XA, m)[ref_A_mask],
                         sub(XB, m)[ref_B_mask]),
                        ("coral_all", sub(XA, m),
                         sub(XB, m))):
                    W, mu_s, mu_t = coral_params(Xs_ref, Xt_ref)
                    XB_al = apply_coral(XB_t, W, mu_s, mu_t)
                    acc[m][cond].append(acc_of(m, XB_al, test_idx))

        # ---- Pooled (optimistic) transductive CORAL over all B ----
        pooled = {}
        for m in ("mag", "full"):
            W, mu_s, mu_t = coral_params(sub(XA, m), sub(XB, m))
            XB_al = apply_coral(sub(XB, m), W, mu_s, mu_t)
            pooled[m] = {
                "raw": acc_of(m, sub(XB, m), np.arange(len(yB))),
                "coral_all": acc_of(m, XB_al, np.arange(len(yB))),
            }

        lines = []
        L = lambda s="": lines.append(s)
        L("=" * 72)
        L(f"EXPERIMENT 4 — CORAL ALIGNMENT  [{split['name']}]  A={split['A']} B={split['B']}")
        L("=" * 72)
        L(f"Chance = {100/6:.1f}% | LOO discipline: held-out gas excluded from coral-ref")
        L(f"{'model':<6} {'condition':<11} {'mean acc':>9}  per-gas accs")
        for m in ("mag", "full"):
            for c in ("raw", "coral_ref", "coral_all"):
                vals = acc[m][c]
                per = " ".join(f"{v*100:5.1f}" for v in vals)
                L(f"{m:<6} {c:<11} {np.mean(vals)*100:6.1f}%     {per}")
        L("")
        L("Pooled (transductive over all B — optimistic):")
        for m in ("mag", "full"):
            L(f"  {m:<6} raw {pooled[m]['raw']*100:5.1f}%  "
              f"coral_all {pooled[m]['coral_all']*100:5.1f}%  "
              f"delta {((pooled[m]['coral_all']-pooled[m]['raw'])*100):+.1f} pp")
        for m in ("mag", "full"):
            dm = (np.mean(acc[m]["coral_all"]) - np.mean(acc[m]["raw"])) * 100
            dr = (np.mean(acc[m]["coral_ref"]) - np.mean(acc[m]["raw"])) * 100
            L(f"  LOO delta: {m} coral-ref {dr:+.1f} pp   coral-all {dm:+.1f} pp")

        all_lines.extend(lines)
        print("\n".join(lines), flush=True)

        results[split["name"]] = {
            "acc": {m: {c: [float(v) for v in acc[m][c]]
                        for c in ("raw", "coral_ref", "coral_all")}
                    for m in ("mag", "full")},
            "pooled": {m: {c: float(v) for c, v in pooled[m].items()}
                       for m in ("mag", "full")},
        }

    out = ROOT / "interoperability" / "experiment_4_coral_result.txt"
    out.write_text("\n".join(all_lines) + "\n")
    print(f"\nWrote {out}", flush=True)
    (ROOT / "interoperability" / "experiment_4_coral_metrics.json").write_text(
        json.dumps(results, indent=2, default=float))
    print("Wrote metrics JSON", flush=True)


if __name__ == "__main__":
    run()
