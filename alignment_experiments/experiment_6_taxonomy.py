#!/usr/bin/env python3
"""Experiment 6 — Taxonomic (class-level) transfer across devices.

Phase 4 (Qwen brief, Path B): ontology/taxonomic transfer. The idea: a device
that cannot resolve fine gas/substance identity across a device boundary may
still resolve a COARSER class. If drift confusions are preferentially
within-class, then training on class labels (chemical functional groups on UCI;
perceptual grand families on SmellNet/OSMO) should transfer across devices
better than fine labels do.

Two legs:

Part A — UCI Gas Sensor Array Drift (real drift, 6 gases, 10 batches):
  Fine labels: 6 gas identities (chance 16.7%).
  Coarse labels: chemical functional classes (chance 25%):
      Ethanol     -> alcohol
      Acetaldehyde, Acetone -> carbonyl
      Ethylene, Toluene     -> hydrocarbon
      Ammonia     -> amine
  Measure, per split (A = early batches, B = late batches = "different device"):
    - fine cross-device accuracy (reproduce Exp 3 RAW)
    - coarse cross-device accuracy (direct coarse classifier trained on A)
    - coarse-collapse: collapse FINE predictions to coarse classes and score
    - within-device ceilings (train+test on A), fine and coarse
    - transfer efficiency TE = (cross - chance)/(within - chance), fine vs coarse
    - within-class error share: of fine cross-batch mispredictions, how many
      land in a same-class gas; compared to the share expected by chance.
  Falsifiable: coarse TE > fine TE  => taxonomy genuinely helps transfer.

Part B — SmellNet paradigms (30-dim, device-agnostic by R0 normalization) to
  OSMO grand families:
  - LSO classifier: per-recording paradigm vectors -> grand family, leave-one-
    substance-out. Chance = 1/8 = 12.5%.
  - Substance-level LSO (50 classes, chance 2%) for reference.
  - Cross-device: fit family RF on SmellNet paradigms, predict the USER rig's
    paradigm vectors, score against OSMO ground truth for labeled substances
    (garlic->Mineral, cinnamon->Woody, banana->Fruity, lemon->Citrus, ginger->Woody).

Run:  python interoperability/experiment_6_taxonomy.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score

HERE = Path(__file__).resolve().parent          # interoperability/alignment_experiments/
INTEROP = HERE.parent                            # interoperability/
REPO = INTEROP.parent                            # OpenSmell root
sys.path.insert(0, str(INTEROP / "canonical_experiments"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "opensmell"))

from load_uci import load_all_batches, GAS_NAMES
from experiment_3_calibration import (
    concat_batches, MAG_IDX, PRIMARY_SPLIT, SENSITIVITY_SPLIT,
    N_ESTIMATORS, RANDOM_STATE,
)
from Osmograph.viz.paradigm_features import compute_window_paradigms

N_SENSORS = 16
FEATS_PER_SENSOR = 8
N_DIM = 128

# ------------------------------------------------------------------ Part A ---
GAS_CLASS = {
    "Ethanol": "alcohol",
    "Ethylene": "hydrocarbon",
    "Ammonia": "amine",
    "Acetaldehyde": "carbonyl",
    "Acetone": "carbonyl",
    "Toluene": "hydrocarbon",
}
CLASS_ORDER = ["alcohol", "carbonyl", "hydrocarbon", "amine"]


def coarse_of_gas_ids(gas_ids):
    return np.array([CLASS_ORDER.index(GAS_CLASS[GAS_NAMES[g]]) for g in gas_ids])


def rf_acc(Xtr, ytr, Xte, yte, idx=None):
    Xtr = Xtr if idx is None else Xtr[:, idx]
    Xte = Xte if idx is None else Xte[:, idx]
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(sc.transform(Xtr), ytr)
    return rf, sc, accuracy_score(yte, rf.predict(sc.transform(Xte)))


def part_a():
    batches = load_all_batches()
    lines = []
    L = lambda s="": lines.append(s)
    L("=" * 76)
    L("PART A — UCI CHEMICAL-CLASS COARSENING (real drift A->B)")
    L("=" * 76)
    L(f"Fine: 6 gases (chance {100/6:.1f}%) | Coarse: {CLASS_ORDER} (chance {100/len(CLASS_ORDER):.1f}%)")

    results = {}
    for split in (PRIMARY_SPLIT, SENSITIVITY_SPLIT):
        XA, yA = concat_batches(batches, split["A"])
        XB, yB = concat_batches(batches, split["B"])
        fine_le = {g: i for i, g in enumerate([GAS_NAMES[k] for k in sorted(GAS_NAMES)])}
        yA_f = np.array([fine_le[GAS_NAMES[g]] for g in yA])
        yB_f = np.array([fine_le[GAS_NAMES[g]] for g in yB])
        yA_c = coarse_of_gas_ids(yA)
        yB_c = coarse_of_gas_ids(yB)

        row = {}
        L("")
        L(f"--- {split['name']}  A={split['A']} B={split['B']}")
        for tag, idx in (("mag", MAG_IDX), ("full", None)):
            # fine models
            rfF, scF, cross_fine = rf_acc(XA, yA_f, XB, yB_f, idx)
            rfFw, scFw, within_fine = rf_acc(XA, yA_f, XA, yA_f, idx)
            # coarse models
            rfC, scC, cross_coarse = rf_acc(XA, yA_c, XB, yB_c, idx)
            rfCw, scCw, within_coarse = rf_acc(XA, yA_c, XA, yA_c, idx)
            # coarse collapse of fine predictions on B
            XA_t = XA[:, idx] if idx is not None else XA
            XB_t = XB[:, idx] if idx is not None else XB
            sc_t = StandardScaler().fit(XA_t)
            rfF_fit = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                             class_weight="balanced", n_jobs=-1).fit(sc_t.transform(XA_t), yA_f)
            pred_f = rfF_fit.predict(sc_t.transform(XB_t))
            coll_coarse = accuracy_score(yB_c, coarse_of_gas_ids(pred_f + 1))  # pred_f are 0..5 gas idx

            # within-class error analysis on the fine cross-batch predictions
            err = pred_f != yB_f
            n_err = int(err.sum())
            same = np.array([GAS_NAMES[p + 1] for p in pred_f[err]])
            true = np.array([GAS_NAMES[t + 1] for t in yB_f[err]])
            within_obs = float(np.mean([GAS_CLASS[s] == GAS_CLASS[t] for s, t in zip(same, true)])) if n_err else 0.0
            exp_shares = [sum(1 for g in GAS_NAMES.values() if g != GAS_NAMES[t + 1] and
                              GAS_CLASS[g] == GAS_CLASS[GAS_NAMES[t + 1]]) / 5 for t in yB_f[err]]
            within_exp = float(np.mean(exp_shares)) if exp_shares else 0.0

            # transfer efficiency (normalized signal), guard against within=chance
            def te(cross, within, chance):
                return (cross - chance) / (within - chance) if within > chance else float("nan")

            te_f = te(cross_fine, within_fine, 1 / 6)
            te_c = te(cross_coarse, within_coarse, 1 / len(CLASS_ORDER))

            row[tag] = {
                "fine_cross": cross_fine, "fine_within": within_fine,
                "coarse_cross": cross_coarse, "coarse_within": within_coarse,
                "coarse_collapse_of_fine": coll_coarse,
                "te_fine": te_f, "te_coarse": te_c,
                "within_class_error_observed": within_obs,
                "within_class_error_expected": within_exp,
                "n_fine_errors": n_err,
            }
            L(f"  {tag:4s} fine:  cross {cross_fine*100:5.1f}%  within {within_fine*100:5.1f}%  "
              f"TE {te_f*100:5.1f}%")
            L(f"  {tag:4s} coarse: cross {cross_coarse*100:5.1f}%  within {within_coarse*100:5.1f}%  "
              f"TE {te_c*100:5.1f}%   collapse-of-fine {coll_coarse*100:5.1f}%")
            L(f"  {tag:4s} within-class error share (fine preds, B): observed {within_obs*100:5.1f}%  "
              f"chance {within_exp*100:5.1f}%  (n={n_err})")

            # per-class coarse accuracy on B
            if tag == "full":
                accs = {}
                for c, cname in enumerate(CLASS_ORDER):
                    m = yB_c == c
                    if m.sum() > 0:
                        accs[cname] = float(accuracy_score(
                            yB_c[m], coarse_of_gas_ids(pred_f[m] + 1)))
                L(f"  coarse-collapse per-class (B): " +
                  "  ".join(f"{k}={v*100:.0f}%" for k, v in accs.items()))
        results[split["name"]] = row
        print("\n".join(lines[-8:]), flush=True)

    return lines, results


# ------------------------------------------------------------------ Part B ---
TAX_DIR = REPO / "research" / "taxonomy"
LABELS_CSV = TAX_DIR / "smellnet_osmo_labels.csv"
SMELLNET_DIR = REPO / "SmellNet" / "neurips-data-processed"
USER_RECS = Path.home() / "Osmograph_Recordings"
SENSOR_COLS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
R0 = 5


def load_smellnet_6ch(path):
    df = pd.read_csv(path)
    cols = [str(c).strip() for c in df.columns]
    cols = ["C2H5OH" if c == "C2H50H" else c for c in cols]
    df.columns = cols
    arr = np.zeros((len(df), 6), dtype=np.float64)
    for i, s in enumerate(SENSOR_COLS):
        if s in cols:
            arr[:, i] = df[s].values.astype(np.float64)
    return arr


def load_user_6ch(path):
    path = str(path)
    try:
        with open(path) as f:
            first = f.readline().strip()
        has_header = not first.replace(",", "").replace(".", "").replace("-", "").lstrip()[0].isdigit()
    except Exception:
        has_header = False
    if not has_header:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] == 3:
            from opensmell.mox.preprocessing import expand_channels
            return expand_channels(data.astype(np.float32)).astype(np.float64)
        if data.shape[1] == 6:
            return data
        raise ValueError(f"unknown legacy cols {data.shape[1]}")
    df = pd.read_csv(path)
    cols = list(df.columns)
    for names in (["MQ135", "MQ3", "MQ6", "MQ7", "MQ4", "MQ8"],
                  ["VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH"],
                  ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]):
        if all(c in cols for c in names):
            arr = np.zeros((len(df), 6), dtype=np.float64)
            for i, c in enumerate(names):
                arr[:, i] = df[c].values.astype(np.float64)
            return arr
    raise ValueError(f"no known sensor columns in {cols}")


def paradigm_of(arr):
    if arr.shape[0] < 5:
        return None
    return compute_window_paradigms(arr, r0_samples=R0)


def part_b():
    labels = pd.read_csv(LABELS_CSV)
    fam_map = dict(zip(labels["substance"], labels["grand_family"]))
    fam_list = sorted(set(fam_map.values()))
    le = {f: i for i, f in enumerate(fam_list)}
    n_fam = len(fam_list)

    lines = []
    L = lambda s="": lines.append(s)
    L("=" * 76)
    L("PART B — SMELLNET PARADIGMS -> OSMO GRAND FAMILY")
    L("=" * 76)
    L(f"{len(fam_list)} grand families: {fam_list} (chance {100/n_fam:.1f}%)")
    fam_counts = labels["grand_family"].value_counts().to_dict()
    L("Label balance: " + ", ".join(f"{k}={v}" for k, v in sorted(fam_counts.items())))

    # per-recording paradigms for SmellNet
    rec = []
    csvs = sorted(SMELLNET_DIR.glob("*.csv.csv")) or sorted(SMELLNET_DIR.glob("*.csv"))
    for f in csvs:
        stem = f.stem[:-4] if f.name.endswith(".csv.csv") else f.stem
        sub = stem.split(".")[0]
        if sub not in fam_map:
            continue
        try:
            v = paradigm_of(load_smellnet_6ch(f))
        except Exception:
            v = None
        if v is None:
            continue
        rec.append({"substance": sub, "family": fam_map[sub], "feat": np.asarray(v, dtype=float)})
    X = np.array([r["feat"] for r in rec])
    y_fam = np.array([le[r["family"]] for r in rec])
    y_sub = np.array([r["substance"] for r in rec])
    subs = sorted(set(y_sub))
    sub_enc = {s: i for i, s in enumerate(subs)}
    y_sub_e = np.array([sub_enc[s] for s in y_sub])
    L(f"\n{len(rec)} recordings, {len(subs)} substances, {X.shape[1]}-dim paradigms")

    # family LSO
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pred_fam = np.zeros(len(X), dtype=int)
    pred_sub = np.zeros(len(X), dtype=int)
    for s in subs:
        te = np.array([r["substance"] == s for r in rec])
        rf_fam = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                        class_weight="balanced", n_jobs=-1).fit(Xs[~te], y_fam[~te])
        rf_sub = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                        class_weight="balanced", n_jobs=-1).fit(Xs[~te], y_sub_e[~te])
        pred_fam[te] = rf_fam.predict(Xs[te])
        pred_sub[te] = rf_sub.predict(Xs[te])
    acc_fam = accuracy_score(y_fam, pred_fam)
    bal_fam = balanced_accuracy_score(y_fam, pred_fam)
    acc_sub = accuracy_score(y_sub_e, pred_sub)
    L(f"LSO family accuracy: {acc_fam*100:.1f}% (balanced {bal_fam*100:.1f}%; chance {100/n_fam:.1f}%)")
    L(f"LSO substance accuracy: {acc_sub*100:.1f}% (chance {100/len(subs):.1f}%)")

    cm = confusion_matrix(y_fam, pred_fam, labels=range(n_fam))
    L("Family confusion (rows=true):")
    L("  " + "".join(f"{f[:9]:>10}" for f in fam_list))
    for i, f in enumerate(fam_list):
        L(f"  {f[:9]:>10}" + "".join(f"{v:>10d}" for v in cm[i]))

    # cross-device: predict user rig paradigms
    L("\n--- Cross-device: SmellNet-trained family RF -> USER rig paradigms ---")
    rf_all = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                    class_weight="balanced", n_jobs=-1).fit(Xs, y_fam)
    user_rec = []
    user_files = []
    if USER_RECS.exists():
        for f in sorted(USER_RECS.glob("2026*.csv")):
            user_files.append((f, False))
        legacy = USER_RECS / "legacy"
        if legacy.exists():
            for f in sorted(legacy.glob("*.csv")):
                user_files.append((f, True))
    for f, is_legacy in user_files:
        name = f.name.lower()
        if "warm garlic" in name or "garlic" in name:
            sub = "garlic"
        elif "ginger" in name:
            sub = "ginger"
        elif "onion" in name:
            sub = "onion"
        elif "cinnamon" in name:
            sub = "cinnamon"
        elif "lime" in name or "lemon" in name:
            sub = "lemon"
        elif "banana" in name:
            sub = "banana"
        elif "coil" in name:
            sub = "mosquito_coil"
        elif "air" in name:
            sub = "room_air"
        else:
            continue
        try:
            v = paradigm_of(load_user_6ch(f))
        except Exception as e:
            L(f"  SKIP {f.name}: {e}")
            continue
        if v is None:
            continue
        user_rec.append({"substance": sub, "feat": np.asarray(v, dtype=float)})
    if user_rec:
        Xu = np.array([r["feat"] for r in user_rec])
        preds = rf_all.predict(sc.transform(Xu))
        from collections import Counter
        per_sub = {}
        for r, p in zip(user_rec, preds):
            per_sub.setdefault(r["substance"], Counter()).update([fam_list[p]])
        for s in sorted(per_sub):
            gt = fam_map.get(s, "unlabelled")
            top = per_sub[s].most_common(1)[0][0]
            ok = "OK" if gt != "unlabelled" and top == gt else ("?" if gt == "unlabelled" else "MISMATCH")
            dist = "  ".join(f"{k}:{v}" for k, v in per_sub[s].most_common(3))
            L(f"  {s:14s} OSMO={gt:9s} predicted={top:9s} [{ok}]  {dist}")
    else:
        L("  No user recordings found.")

    return lines, {"family_lso_accuracy": float(acc_fam),
                   "family_lso_balanced": float(bal_fam),
                   "substance_lso_accuracy": float(acc_sub),
                   "n_families": n_fam, "n_substances": len(subs),
                   "n_recordings": len(rec)}


def main():
    a_lines, a_res = part_a()
    b_lines, b_res = part_b()

    all_lines = a_lines + [""] + b_lines
    out = HERE / "results" / "experiment_6_taxonomy_result.txt"
    out.write_text("\n".join(all_lines) + "\n")
    print(f"\nWrote {out}", flush=True)

    metrics = {"part_a": a_res, "part_b": b_res}
    (HERE / "results" / "experiment_6_taxonomy_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    print("Wrote metrics JSON", flush=True)


if __name__ == "__main__":
    main()
