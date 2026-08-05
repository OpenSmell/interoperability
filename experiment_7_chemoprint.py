#!/usr/bin/env python3
"""Experiment 7 — Chemoprint as a cross-device prior (UCI real drift).

Phase 4 (Qwen brief, Path B): chemoprint-as-prior. The Chemoprint README
claims a sensor array can be calibrated to output the 29-dim chemoprint with
variance-weighted R² = 0.982 (UCI, 128-dim features). That number was computed
on a random 80/20 split that MIXES the 10 drift batches — an in-sample claim.
The interoperability question is the one the README admits it never tested:
does the sensor->chemoprint map transfer ACROSS a device boundary?

Design (same splits/venue as Exps 3/4/6):
  Train = early batches A, Test = late batches B (real drift).
  Conditions (RandomForestRegressor, 200 trees):
    mixed80  - random 80/20 over ALL batches (reproduces the 0.982 claim).
    within_A - train A, test A  (source ceiling).
    within_B - train B, test B  (target ceiling).
    cross_AB - train A, test B  (THE transfer number).
    cross_BA - train B, test A  (reverse direction).
  Models: full 128-dim (the venue of the original claim) + magnitude-only
  32-dim (drift subspace, isolates amplitude drift).
  Metrics: variance-weighted R², mean per-dim R², MAE; per-dim R² for the
  most informative dims; per-gas MAE on B.

Because the chemoprint target is constant within each gas, R² measures how
well the A-learned gas->chemoprint mapping preserves gas clusters in B's
feature manifold — i.e. whether the chemoprint space is device-agnostic.

Run:  python interoperability/experiment_7_chemoprint.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "interoperability" / "canonical_experiments"))
sys.path.insert(0, str(ROOT / "Chemoprint"))

from load_uci import load_all_batches, GAS_NAMES
from experiment_3_calibration import (
    concat_batches, MAG_IDX, PRIMARY_SPLIT, SENSITIVITY_SPLIT, RANDOM_STATE,
)
from chemoprint import chemoprint_from_smiles

N_ESTIMATORS = 200
GAS_SMILES = {1: "CCO", 2: "C=C", 3: "N", 4: "CC=O", 5: "CC(=O)C", 6: "Cc1ccccc1"}


def chemoprints_by_gas():
    out = {}
    for gid, smi in GAS_SMILES.items():
        cp = chemoprint_from_smiles(smi)
        if cp is None:
            raise ValueError(f"bad SMILES for gas {gid}")
        out[gid] = np.asarray(cp, dtype=float)
    return out


def rf_reg(Xtr, Ytr, Xte, Yte, idx=None):
    Xtr = Xtr if idx is None else Xtr[:, idx]
    Xte = Xte if idx is None else Xte[:, idx]
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                  n_jobs=-1)
    model.fit(Xtr, Ytr)
    Yp = model.predict(Xte)
    r2 = r2_score(Yte, Yp, multioutput="variance_weighted")
    r2m = float(np.mean(r2_score(Yte, Yp, multioutput="raw_values")))
    mae = mean_absolute_error(Yte, Yp)
    return r2, r2m, mae


def run():
    cps = chemoprints_by_gas()
    batches = load_all_batches()

    lines = []
    L = lambda s="": lines.append(s) or print(s, flush=True)
    L("=" * 76)
    L("EXPERIMENT 7 — CHEMOPRINT AS CROSS-DEVICE PRIOR (UCI real drift)")
    L("=" * 76)
    L("Target: 29-dim chemoprint (constant per gas) | RF regressor, full 128-dim")

    # mixed 80/20 over ALL batches (reproduces the README claim)
    Xall, yall_gas = [], []
    for b in sorted(batches):
        X, y = batches[b]
        Xall.append(X)
        yall_gas.append(y)
    Xall = np.vstack(Xall)
    yall_gas = np.concatenate(yall_gas)
    Yall = np.array([cps[g] for g in yall_gas])
    Xtr, Xte, Ytr, Yte = train_test_split(Xall, Yall, test_size=0.2, random_state=42)
    r2_mixed, r2m_mixed, mae_mixed = rf_reg(Xtr, Ytr, Xte, Yte)
    L(f"mixed80/20 (all batches): variance-weighted R2 = {r2_mixed:.4f}  "
      f"mean R2 = {r2m_mixed:.4f}  MAE = {mae_mixed:.4f}")

    metrics = {"mixed80_20": {"r2_var": r2_mixed, "r2_mean": r2m_mixed, "mae": mae_mixed}}
    per_dim_info = [0, 1, 6, 7, 8, 9, 11, 12, 13, 14]
    pd_table = {}

    for split in (PRIMARY_SPLIT, SENSITIVITY_SPLIT):
        XA, yA = concat_batches(batches, split["A"])
        XB, yB = concat_batches(batches, split["B"])
        YA = np.array([cps[g] for g in yA])
        YB = np.array([cps[g] for g in yB])
        L("")
        L(f"--- {split['name']}  A={split['A']} B={split['B']}")
        row = {}
        for tag, idx in (("full", None), ("mag", MAG_IDX)):
            within_A = rf_reg(XA, YA, XA, YA, idx)
            within_B = rf_reg(XB, YB, XB, YB, idx)
            cross_AB = rf_reg(XA, YA, XB, YB, idx)
            cross_BA = rf_reg(XB, YB, XA, YA, idx)
            row[tag] = {
                "within_A_r2": within_A[0], "within_B_r2": within_B[0],
                "cross_AB_r2": cross_AB[0], "cross_BA_r2": cross_BA[0],
                "cross_AB_mae": cross_AB[2],
                "drop_AB_pp": (within_A[0] - cross_AB[0]) * 100,
            }
            L(f"  {tag:4s} within_A {within_A[0]*100:6.2f}%  within_B {within_B[0]*100:6.2f}%  "
              f"cross A->B {cross_AB[0]*100:6.2f}%  (drop {row[tag]['drop_AB_pp']:+.2f} pp)  "
              f"cross B->A {cross_BA[0]*100:6.2f}%")
            if tag == "full":
                # per-dim R2 for the informative dims + per-gas MAE
                model = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                              random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(XA, YA)
                Yp = model.predict(XB)
                rd = r2_score(YB, Yp, multioutput="raw_values")
                dim_notes = {0: "MW", 1: "heavy atoms", 6: "TPSA-ish",
                             7: "topological", 8: "rings?", 9: "aromatic?",
                             11: "hetero", 12: "Wiener", 13: "Zagreb", 14: "eccentricity"}
                line = "  per-dim R2: "
                for d in per_dim_info:
                    line += f"[{d} {dim_notes.get(d,'')}]= {rd[d]:+.3f}  "
                L(line)
                pd_table[split["name"]] = {str(d): float(rd[d]) for d in per_dim_info}
                # per-gas MAE on B (predicted chemoprint vs true)
                gmae = {}
                for gid in sorted(GAS_NAMES):
                    m = yB == gid
                    if m.sum() > 0:
                        gmae[GAS_NAMES[gid]] = float(
                            mean_absolute_error(YB[m], Yp[m]))
                L("  per-gas MAE (B): " + "  ".join(f"{k}={v:.3f}" for k, v in gmae.items()))
                row[tag]["per_gas_mae_B"] = gmae
        metrics[split["name"]] = row
        print("\n".join(lines[-6:]), flush=True)

    L("\n" + "=" * 76)
    L("NOTE: chemoprint target is constant per gas, so cross R2 measures whether")
    L("the A-learned gas->chemoprint mapping preserves gas clusters in B's")
    L("feature manifold. Variance-weighted R2 is dominated by the MW/heavy-atom")
    L("and topological dims; functional-group flags are all-zero for these 6 gases.")

    out = ROOT / "interoperability" / "experiment_7_chemoprint_result.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}", flush=True)
    metrics["per_dim_r2_full_cross_AB"] = pd_table
    (ROOT / "interoperability" / "experiment_7_chemoprint_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    print("Wrote metrics JSON", flush=True)


if __name__ == "__main__":
    run()
