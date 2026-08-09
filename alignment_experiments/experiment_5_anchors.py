#!/usr/bin/env python3
"""Experiment 5 — Anchor-based (supervised) alignment on Praise James' rig.

Phase 4 (Qwen brief, Path B): instead of an *unsupervised* alignment (Exp 4
CORAL), fit a SUPERVISED map using the 3 shared substances as anchors —
cinnamon, garlic, banana are measured on BOTH SmellNet (Rig A) and Praise
James' rig (Rig B). Then test transfer on a held-out anchor substance
(leave-one-anchor-out) and on lime (Praise James-only OOV probe).

Venue: the real SmellNet -> Praise James' pair, magnitude subspace only
(3 live channels VOC/Alcohol/LPG x 3 magnitude features = 9 dims), matching
the Exp 1/2/3 magnitude-isolation convention.

Two map families (the honest comparison to Exp 1's single scalar M and
Exp 4's covariance map):
  affine_loo/pool    per-feature affine y = a*x + b. With 2 anchors (LOO) the
                     line is exact through 2 points (0 degrees of freedom
                     left); with 3 anchors it is a 3-point least-squares line.
  proc_loo/pool      full Procrustes (scale + rotation + translation), Kabsch
                     superimposition of the anchor centroids. With 2 anchors
                     the rotation is constrained only on the 1-dim span of the
                     anchor segment; the 8-dim orthogonal complement is
                     unconstrained (SVD picks the closest-to-identity rotation).

Documented underdetermination: 2-3 anchor points cannot determine a 9-dim
affine/covariance map; the held-out substance's position off the anchor span
is unconstrained by the fit. This is the crisp test of "supervised anchors
beat scalars/CORAL when reference substances exist."

Run:  python interoperability/experiment_5_anchors.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent          # interoperability/alignment_experiments/
INTEROP = HERE.parent                            # interoperability/
REPO = INTEROP.parent                            # OpenSmell root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "opensmell"))

from experiment_1_calibration import (
    LIVE, PER_CH_FEATS, MAGNITUDE_FEATS, feature_columns,
    SMELLNET_CLASSES, OVERLAP, N_ESTIMATORS, RANDOM_STATE, CACHE_PATH,
)

MAG9 = [feature_columns().index(f"ch{ch}_{f}")
        for ch in LIVE for f in MAGNITUDE_FEATS]


def load_cache():
    d = np.load(CACHE_PATH, allow_pickle=True)
    return d


def mag_only(X):
    return X[:, MAG9]


def kabsch(U, S):
    """Scale+rotation+translation mapping user anchor means -> SmellNet means.
    U, S: (n_anchors, d) row vectors. Returns (scale, R, t):  map(u)=scale*R@u+t.
    """
    mu_u = U.mean(axis=0)
    mu_s = S.mean(axis=0)
    Uc = U - mu_u
    Sc = S - mu_s
    M = Sc.T @ Uc / len(U)
    Uu, sv, Vt = np.linalg.svd(M)
    R = Uu @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Uu @ Vt
    scale = sv.sum() / max(np.sum(Uc * Uc), 1e-12)
    t = mu_s - scale * R @ mu_u
    return float(scale), R, t


def fit_affine(u_vecs_by_anchor, s_vecs_by_anchor, anchors):
    """Per-feature affine y = a*x + b from anchor means. Returns (a, b) arrays."""
    U = np.array([np.mean(u_vecs_by_anchor[a], axis=0) for a in anchors])
    S = np.array([np.mean(s_vecs_by_anchor[a], axis=0) for a in anchors])
    a = np.zeros(U.shape[1])
    b = np.zeros(U.shape[1])
    for f in range(U.shape[1]):
        x, y = U[:, f], S[:, f]
        if x.std() < 1e-12:            # anchors don't vary on this dim
            a[f] = 1.0
            b[f] = y.mean() - x.mean()
        else:
            A = np.column_stack([x, np.ones_like(x)])
            a[f], b[f] = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def apply_affine(X, a, b):
    return X * a + b


def apply_procrustes(X, scale, R, t):
    return X @ (scale * R).T + t


def run():
    d = load_cache()
    sn_X, sn_y = d["sn_X"], d["sn_y"]
    us_X, us_subs = d["us_X"], d["us_subs"]

    lines = []
    L = lambda s="": lines.append(s) or print(s, flush=True)

    L("=" * 72)
    L("EXPERIMENT 5 — ANCHOR-BASED SUPERVISED ALIGNMENT (Praise James' rig)")
    L("=" * 72)
    L(f"Chance = 25% (4 SmellNet classes) | magnitude subspace = {len(MAG9)} dims")
    L(f"Anchors (shared substances): {OVERLAP} | OOV probe: lime")

    # ---- within-device sanity on the magnitude-only subspace ----
    le = LabelEncoder()
    y_enc = le.fit_transform(sn_y)
    X9 = mag_only(np.asarray(sn_X))
    cv = cross_val_score(RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                                random_state=RANDOM_STATE,
                                                class_weight="balanced", n_jobs=-1),
                         X9, y_enc, cv=5, scoring="accuracy")
    L(f"SmellNet 5-fold CV (magnitude-only {len(MAG9)}-dim): {cv.mean()*100:.1f}% "
      f"({cv.min()*100:.1f}-{cv.max()*100:.1f}%)")

    # ---- magnitude-only RF on SmellNet ----
    sc9 = StandardScaler().fit(X9)
    rf9 = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                 class_weight="balanced", n_jobs=-1).fit(sc9.transform(X9), y_enc)

    # ---- full-model RF reference (Exp 1 reproduction) ----
    sc_full = StandardScaler().fit(np.asarray(sn_X))
    rf_full = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                                     class_weight="balanced", n_jobs=-1).fit(
        sc_full.transform(np.asarray(sn_X)), y_enc)

    us_by = {}
    for s in set(us_subs):
        mask = np.array([x == s for x in us_subs])
        us_by[s] = np.asarray(us_X)[mask]
    sn_by = {}
    for s in SMELLNET_CLASSES:
        mask = np.array([y == s for y in sn_y])
        sn_by[s] = np.asarray(sn_X)[mask]

    test_subs = [s for s in OVERLAP if s in us_by]

    # ---- RAW baselines ----
    L("\n--- RAW (no alignment) ---")
    for tag, model, sc, Xtr in (("mag9", rf9, sc9, X9), ("full", rf_full, sc_full, np.asarray(sn_X))):
        preds, trues = [], []
        for s in test_subs:
            Xt = mag_only(us_by[s]) if tag == "mag9" else us_by[s]
            preds.extend(le.inverse_transform(model.predict(sc.transform(Xt))))
            trues.extend([s] * len(Xt))
        acc = accuracy_score(trues, preds)
        L(f"  {tag:4s} RAW overall: {acc*100:.1f}% "
          f"(" + "  ".join(f"{s}:{sum(1 for p,t in zip(preds,trues) if t==s and p==s)}/{sum(1 for t in trues if t==s)}"
                           for s in test_subs) + ")")

    # ---- LOO supervised alignment (fit on anchors minus test) ----
    L("\n--- ALIGNED, leave-one-anchor-out (map fit on the OTHER anchors) ---")
    results = {s: {} for s in test_subs}
    for s in test_subs:
        anchors = [a for a in OVERLAP if a in us_by and a != s]
        u_means = {a: mag_only(us_by[a]) for a in anchors}
        s_means = {a: mag_only(sn_by[a]) for a in anchors}
        U = np.array([u_means[a].mean(axis=0) for a in anchors])
        S = np.array([s_means[a].mean(axis=0) for a in anchors])
        scale, R, t = kabsch(U, S)
        a_vec, b_vec = fit_affine(u_means, s_means, anchors)
        Xt = mag_only(us_by[s])
        conds = {
            "raw": Xt,
            "affine": apply_affine(Xt, a_vec, b_vec),
            "proc": apply_procrustes(Xt, scale, R, t),
        }
        for cname, Xa in conds.items():
            preds = le.inverse_transform(rf9.predict(sc9.transform(Xa)))
            acc = accuracy_score([s] * len(preds), preds)
            results[s][cname] = acc
        L(f"  test={s:8s} anchors={anchors}  " +
          "  ".join(f"{c}:{results[s][c]*100:.1f}%" for c in ("raw", "affine", "proc")))
        L(f"    proc scale={scale:.3f}  affine slopes "
          f"min={a_vec.min():.3f} max={a_vec.max():.3f}")

    # ---- pooled supervised alignment (fit on ALL anchors; optimistic) ----
    L("\n--- ALIGNED, pooled over all anchors (optimistic upper bound) ---")
    all_a = [a for a in OVERLAP if a in us_by]
    U_all = np.array([mag_only(us_by[a]).mean(axis=0) for a in all_a])
    S_all = np.array([mag_only(sn_by[a]).mean(axis=0) for a in all_a])
    scale_all, R_all, t_all = kabsch(U_all, S_all)
    a_all, b_all = fit_affine({a: mag_only(us_by[a]) for a in all_a},
                              {a: mag_only(sn_by[a]) for a in all_a}, all_a)
    pooled = {}
    for s in test_subs:
        Xt = mag_only(us_by[s])
        for cname, Xa in (("affine", apply_affine(Xt, a_all, b_all)),
                          ("proc", apply_procrustes(Xt, scale_all, R_all, t_all))):
            preds = le.inverse_transform(rf9.predict(sc9.transform(Xa)))
            pooled.setdefault(cname, {})[s] = accuracy_score([s] * len(preds), preds)
        L(f"  test={s:8s}  " + "  ".join(
            f"{c}:{pooled[c][s]*100:.1f}%" for c in ("affine", "proc")))
    L(f"  pooled proc scale={scale_all:.3f}")

    # ---- LIME OOV probe (magnitude-only model) ----
    if "lime" in us_by:
        L("\n--- LIME out-of-vocabulary probe (user-only; class mix) ---")
        Xt = mag_only(us_by["lime"])
        for cname, Xa in (("raw", Xt),
                          ("affine(pool)", apply_affine(Xt, a_all, b_all)),
                          ("proc(pool)", apply_procrustes(Xt, scale_all, R_all, t_all))):
            proba = rf9.predict_proba(sc9.transform(Xa)).mean(axis=0)
            order = np.argsort(proba)[::-1]
            tops = "  ".join(f"{le.inverse_transform([int(i)])[0]}:{proba[i]*100:.1f}%"
                             for i in order[:3])
            L(f"  {cname:11s} {tops}")

    # ---- full model with magnitude-only alignment applied to mag dims ----
    L("\n--- FULL model + proc(pool) alignment applied to the 9 mag dims only ---")
    full_pool_preds, full_pool_trues = [], []
    for s in test_subs:
        Xf = us_by[s].copy()
        Xf[:, MAG9] = apply_procrustes(mag_only(us_by[s]), scale_all, R_all, t_all)
        full_pool_preds.extend(le.inverse_transform(rf_full.predict(sc_full.transform(Xf))))
        full_pool_trues.extend([s] * len(Xf))
    L(f"  full + proc(pool) overall: {accuracy_score(full_pool_trues, full_pool_preds)*100:.1f}%")

    # ---- summary ----
    L("\n" + "=" * 72)
    L("VERDICT")
    L("=" * 72)
    L("3 shared substances as anchors; magnitude-only 9-dim subspace; LOO discipline.")
    L("With 2-3 anchors the map is underdetermined in the 8-9 dim complement;")
    L("held-out substances are off the anchor span. See experiment_5_analysis.md.")

    out = HERE / "results" / "experiment_5_anchors_result.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}", flush=True)

    metrics = {
        "mag9_cv_within_smellnet": float(cv.mean()),
        "loo": results, "pooled": pooled,
        "proc_scale_pooled": float(scale_all),
        "n_test_windows": {s: int(len(us_by[s])) for s in test_subs},
    }
    (HERE / "results" / "experiment_5_anchors_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    run()
