"""Full 187-feature ablation study across ALL five dimensions.

Unlike the canonical Experiment 5 (which ablates only Dimension 1, the 55
device-agnostic features), this study ablates every subgroup of the full
187-feature SDK extractor, including the flagged ablation candidates from the
feature catalog.

Canonical source of truth: the SDK extractor
(`opensmell/opensmell/mox/features.py`). This script consumes it directly —
it does NOT reimplement feature extraction. This guarantees the results are
about the production feature set, and prevents the extractor drift documented
in IMPLEMENTATION_ALIGNMENT.md.

Protocol: session-invariance (held-out session accuracy), matching the
canonical Experiment 1 methodology.

Ablation groups tested:
  - Device-Agnostic (per-channel: amplitude, kinetics, integral/endpoint)
  - Absolute
  - Temporal
  - Health
  - Hardware
  - Advanced (saturation index, decay constants)
  - Selectivity ratios
  - Global
  - Plus: each individual *flagged* ablation candidate
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

MONOREPO = Path(__file__).resolve().parent.parent.parent.parent
OSMELL_PKG = MONOREPO / "opensmell"  # the repo's SDK (v3.0.0) source tree
sys.path.insert(0, str(OSMELL_PKG))

from opensmell.mox.features import (  # noqa: E402
    extract_all_framework_features,
    feature_names,
)

RANDOM_STATE = 42
N_ITERATIONS = 2
ABLATION_N_ESTIMATORS = 60
DEFAULT_LIMIT = 120  # ~20 substances; use --limit 0 for all 50 (slow)

# Features that are only meaningful when the recording follows the
# baseline -> exposure -> recovery protocol. On non-protocol data (e.g.
# SmellNet, which is a monotonic exposure ramp with NO recovery phase),
# these features are degenerate/unintelligible. Ablation verdicts about them
# on such data are NOT evidence about their true discriminative power.
#
# This is the core methodological caution of this study: you cannot judge a
# feature family on data that cannot express it.
PROTOCOL_DEPENDENT_RE = re.compile(
    r"_decay_|_da_decay_time$|_health_hysteresis$"
)  # recovery/desorption-phase features


def detect_protocol_compliance(data):
    """Heuristically check whether a recording contains a recovery phase.

    Returns "recovery" if the signal rises then falls back toward its start
    (baseline -> exposure -> recovery), or "exposure_ramp" if it monotonically
    rises/climbs without returning (SmellNet-style), or "unknown".
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n = data.shape[0]
    if n < 20:
        return "unknown"
    # Average across channels, use a robust peak
    signal = np.nanmean(data, axis=1)
    start = float(np.median(signal[: max(5, n // 20)]))
    end = float(signal[-1])
    peak_idx = int(np.argmax(signal))
    peak = float(signal[peak_idx])
    # How far does it come back down after the peak (fraction of total swing)?
    swing = peak - start
    if swing <= 1e-9:
        return "unknown"
    end_of_window = float(np.median(signal[max(peak_idx, n // 2): n]))
    recovery_frac = (peak - end_of_window) / swing
    if recovery_frac > 0.25:
        return "recovery"
    if abs(end - start) < 0.1 * swing and peak > n // 2:
        return "recovery"
    if end >= peak - 0.1 * swing:
        return "exposure_ramp"
    return "unknown"


def pull_smellnet(limit=None):
    """Return list of (substance, session_num, data_array) from SmellNet.

    Uses the canonical experiment loader rather than re-implementing data
    handling. Falls back gracefully if the dataset isn't available.
    """
    exp_root = MONOREPO / "interoperability" / "canonical_experiments"
    sys.path.insert(0, str(exp_root))
    from load_data import load_all_smellnet_data  # noqa: E402

    all_data = load_all_smellnet_data()
    records = []
    for sub, sessions in all_data.items():
        for ses_num, data_array in sessions:
            records.append((sub, ses_num, np.asarray(data_array, dtype=np.float64)))
    if limit:
        records = records[:limit]
    return records


def extract_features(data, r0_samples=15, sr=10):
    """One feature vector per recording (mean over windows)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    N = data.shape[0]
    # Whole-recording extraction for maximum fidelity to the protocol
    feats = extract_all_framework_features(data, r0_samples=min(r0_samples, max(1, N // 2)), sr=sr)
    names = feature_names()
    vec = np.array([feats.get(n, 0.0) for n in names], dtype=np.float64)
    return vec


def build_groups(all_names):
    """Map each feature name to its subgroup for ablation."""
    groups = {
        "DeviceAgnostic_Amplitude": re.compile(r"_da_(relative_amplitude|direction)$"),
        "DeviceAgnostic_Kinetics": re.compile(r"_da_(rise_time|decay_time)$"),
        "DeviceAgnostic_IntegralEndpoint": re.compile(r"_da_(auc|endpoint_delta)$"),
        "Absolute": re.compile(r"_abs_"),
        "Temporal": re.compile(r"_temp_"),
        "Health": re.compile(r"_health_"),
        "Hardware": re.compile(r"_hw_"),
        "Advanced_SaturationIndex": re.compile(r"_advanced_saturation_index$"),
        "Advanced_Decay": re.compile(r"_decay_"),
        "SelectivityRatios": re.compile(r"^sel_ratio_"),
        "Global": re.compile(r"^global_"),
    }
    # Flagged ablation candidates (individual features)
    flagged = {
        "voltage": re.compile(r"_abs_voltage$"),
        "circuit_response": re.compile(r"_hw_circuit_response$"),
        "thermal_profile": re.compile(r"_hw_thermal_profile$"),
        "sensitivity_decay": re.compile(r"_health_sensitivity_decay$"),
        "oscillation": re.compile(r"_temp_oscillation_(freq|amp)$"),
        "decay3": re.compile(r"_decay_(tau3|a3)$"),
        "hf_transient": re.compile(r"_temp_hf_transient$"),
    }
    return groups, flagged


# ---------------------------------------------------------------------------
# FEATURE ROLES
#
# Not every feature exists to discriminate substances. Two distinct jobs:
#
#   ROLE_DISCRIMINATION : helps answer "WHAT substance is present?"
#                         -> validated by classification accuracy (ablation)
#
#   ROLE_DIAGNOSTIC     : helps answer "Is the SENSOR healthy / poisoned /
#                         drifting?" -> validated by DIAGNOSTIC validity,
#                         i.e. does the feature respond correctly when a
#                         sensor is known-good vs known-bad, and does it
#                         separate those two populations.
#
# A feature may be useless for discrimination yet vital for diagnostics
# (e.g. noise_floor: tells you an ADC is degrading even though it carries no
# chemical information). Judging such a feature solely by classification
# accuracy is a category error. They must be validated against their own
# criterion.
# ---------------------------------------------------------------------------
ROLE_DISCRIMINATION = "discrimination"
ROLE_DIAGNOSTIC = "diagnostic"
ROLE_ABSOLUTE = "absolute"  # needs calibration to mean anything

# Per-role regexes over the canonical SDK feature names.
ROLE_PATTERNS = {
    ROLE_DISCRIMINATION: re.compile(
        r"_da_(relative_amplitude|direction|rise_time|decay_time|auc|endpoint_delta)$"
        r"|^sel_ratio_|^global_|_advanced_saturation_index$"
    ),
    ROLE_DIAGNOSTIC: re.compile(
        # health tracking / poisoning / drift / ADC quality
        r"_health_(drift_rate|sensitivity_decay|noise_floor|hysteresis)$"
        r"|_hw_(adc_noise|circuit_response|thermal_profile)$"
        r"|_temp_(hf_transient|oscillation_freq|oscillation_amp)$"
    ),
    ROLE_ABSOLUTE: re.compile(r"_abs_(raw_resistance|baseline_resistance|voltage|calibrated_concentration)$"),
    # decay outputs are kinetic signatures (discriminative), but they are
    # PROTOCOL-DEPENDENT; they fall under discrimination only on protocol data.
}


def classify_roles(all_names):
    """Assign each feature a role (+ sub-role for diagnostics)."""
    roles = []
    sub_specific = {
        "health_drift": re.compile(r"_health_drift_rate$"),
        "health_poisoning": re.compile(r"_health_(sensitivity_decay|hysteresis)$"),
        "health_noise": re.compile(r"_health_noise_floor$"),
        "hw_adc": re.compile(r"_hw_adc_noise$"),
        "hw_circuit": re.compile(r"_hw_circuit_response$"),
        "hw_thermal": re.compile(r"_hw_thermal_profile$"),
        "temporal_noise": re.compile(r"_temp_(hf_transient|oscillation_freq|oscillation_amp)$"),
    }
    for n in all_names:
        role = None
        sub = None
        for rname, pat in ROLE_PATTERNS.items():
            if pat.search(n):
                role = rname
                break
        if role is None:
            # decay* and anything left -> protocol-dependent kinetics
            role = ROLE_DISCRIMINATION if "_decay_" in n else "other"
        for sname, spat in sub_specific.items():
            if spat.search(n):
                sub = sname
                break
        roles.append({"name": n, "role": role, "diagnostic_subrole": sub})
    return roles


def validate_diagnostics_on_synthetic(roles, all_names):
    """Sanity-test diagnostic feature ROLE on synthetic degraded sensors.

    This is a *placeholder protocol* for validating that diagnostic features
    (health / hardware) actually track what they claim. Real validation
    requires labelled sensor-health data (known-good vs known-poisoned), which
    is a community data need. Here we demonstrate the intended test on
    synthetic data: we add sensor degradation (drift, added noise, poisoning
    via a stuck/poisoned baseline) and check that the diagnostic features
    respond in the expected direction.

    Returns: per-feature dict of how much each diagnostic feature changes when
    a synthetic fault is injected. This tells us which diagnostic features are
    *responsive* to faults (valid) vs *inert* (need reimplementation).
    """
    np.random.seed(RANDOM_STATE)
    base = 1000.0 * np.random.rand(200, 6) + 1000.0  # clean-ish baseline

    def make_series(mode):
        x = base.copy()
        if mode == "drift":
            x += np.linspace(0, 100, x.shape[0])[:, None]  # monotonic drift
        elif mode == "noise":
            x += np.random.randn(*x.shape) * 30  # elevated noise
        elif mode == "poisoned":
            x[:, 0] = 50.0 + np.random.randn(x.shape[0]) * 2  # stuck/flat poisoned channel
        return x

    modes = ["clean", "drift", "noise", "poisoned"]
    feats_by_mode = {}
    for mode in modes:
        feats = extract_all_framework_features(
            make_series(mode), r0_samples=15, sr=10
        )
        feats_by_mode[mode] = feats

    out = {"placeholder": True, "note": (
        "Diagnostic-validity test on synthetic faults (drift/noise/poisoning). "
        "Not a substitute for labelled real-sensor-health data. Real validation "
        "needs known-good vs known-bad sensor recordings."
    ), "per_feature": {}}

    clean = feats_by_mode["clean"]
    for entry in roles:
        if entry["role"] != ROLE_DIAGNOSTIC:
            continue
        n = entry["name"]
        row = {}
        for fault in ["drift", "noise", "poisoned"]:
            c = clean.get(n, 0.0) or 0.0
            f = feats_by_mode[fault].get(n, 0.0) or 0.0
            delta = (f - c)
            row[fault] = {"delta": float(delta)}
        out["per_feature"][n] = row
    return out


def run_session_invariance(X_all, y_all, sess_all, y_enc):
    """Held-out-session accuracy with balanced random train/test split."""
    sub_ses = defaultdict(lambda: defaultdict(list))
    for idx, (sub, ses) in enumerate(zip(y_all, sess_all)):
        sub_ses[sub][ses].append(idx)

    rng = np.random.RandomState(RANDOM_STATE)
    accs = []

    for _ in range(N_ITERATIONS):
        train_idx, test_idx = [], []
        for sub, ses_dict in sub_ses.items():
            sessions = sorted(ses_dict.keys())
            n_test = max(1, len(sessions) // 3)
            test_sessions = set(rng.choice(sessions, size=n_test, replace=False))
            for ses, indices in ses_dict.items():
                (test_idx if ses in test_sessions else train_idx).extend(indices)

        if len(train_idx) < 10 or len(test_idx) < 2:
            continue
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        scaler = StandardScaler().fit(X_tr)
        rf = RandomForestClassifier(
            n_estimators=ABLATION_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(scaler.transform(X_tr), y_tr)
        accs.append(float(accuracy_score(y_te, rf.predict(scaler.transform(X_te)))))

    return accs


def run():
    parser = argparse.ArgumentParser(description="Full 187-feature ablation study")
    parser.add_argument("--limit", type=int, default=None, help="Limit data size; default 120 (use 0/None for all)")
    parser.add_argument("--out", type=str, default=str(MONOREPO / "interoperability" / "research" / "results" / "full_ablation.json"))
    parser.add_argument("--no-smellnet", action="store_true", help="Skip data load (dry-run group definition)")
    args = parser.parse_args()

    t0 = time.time()
    names = feature_names()
    all_names = np.array(names)
    groups, flagged = build_groups(names)

    results = {
        "experiment": "full_187_feature_ablation",
        "n_features_total": len(names),
        "groups": {},
        "flagged_candidates": {},
        "diagnostic_validity": {},
        "runnable": True,
    }

    # ── Role classification ────────────────────────────────────────────
    roles = classify_roles(all_names)
    results["feature_roles"] = {
        e["name"]: {"role": e["role"], "diagnostic_subrole": e["diagnostic_subrole"]}
        for e in roles
    }
    from collections import Counter as _Counter
    role_counts = _Counter(e["role"] for e in roles)
    print("Feature roles:", dict(role_counts))

    # ── Verify group membership covers all features ────────────────────
    covered = set()
    for gname, pat in groups.items():
        covered.update(n for n in all_names if pat.search(n))
    uncovered = [n for n in all_names if n not in covered]
    if uncovered:
        print(f"WARNING: {len(uncovered)} features not covered by any group:")
        for n in uncovered[:20]:
            print(f"  {n}")
        results["uncovered_features"] = uncovered

    each_group = defaultdict(list)
    for i, n in enumerate(all_names):
        for gname, pat in groups.items():
            if pat.search(n):
                each_group[gname].append(i)
    results["group_membership"] = {g: sorted(v) for g, v in each_group.items()}

    if args.no_smellnet:
        print("Dry-run: group definitions only (no data).")
        results["runnable"] = False
        _write_results(args, results)
        return

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading SmellNet data...")
    records = pull_smellnet(limit=args.limit if args.limit else DEFAULT_LIMIT)
    if not records:
        print("No data. Is SmellNet available?")
        results["runnable"] = False
        _write_results(args, results)
        return

    multi_ses = {}
    for sub, ses, arr in records:
        multi_ses.setdefault(sub, []).append((ses, arr))
    substances = sorted(s for s, g in multi_ses.items() if len(g) >= 2)
    print(f"  {len(substances)} substances with >=2 sessions")

    if len(substances) < 3:
        print("Insufficient substances.")
        results["runnable"] = False
        _write_results(args, results)
        return

    # ── Feature matrix ─────────────────────────────────────────────────
    X_list, y_list, sess_list = [], [], []
    for sub in substances:
        for ses_num, arr in multi_ses[sub]:
            X_list.append(extract_features(arr))
            y_list.append(sub)
            sess_list.append(ses_num)
    X = np.vstack(X_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.array(y_list)
    sess_all = np.array(sess_list)
    le = LabelEncoder().fit(y_all)
    y_enc = le.transform(y_all)
    chance = 1.0 / len(substances)

    results.update({
        "n_substances": len(substances),
        "n_recordings": len(y_all),
        "chance_level": chance,
    })
    print(f"  Feature matrix: {X.shape}")

    # ── Baseline: all 187 features ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Baseline: ALL features")
    full_accs = run_session_invariance(X, y_all, sess_all, y_enc)
    full_mean = float(np.mean(full_accs)) if full_accs else 0.0
    print(f"  Mean session-invariance accuracy: {full_mean*100:.1f}%")
    results["baseline_accuracy"] = full_mean
    results["baseline_std"] = float(np.std(full_accs)) if full_accs else 0.0

    # ── Protocol compliance check ──────────────────────────────────────
    # The single most important caveat: verdicts about discrimination power
    # are only trustworthy for features the dataset can actually express.
    compliance = {}
    for sub, sessions in multi_ses.items():
        compliance[sub] = detect_protocol_compliance(sessions[0][1])
    from collections import Counter as _C2
    mode_counts = _C2(compliance.values())
    composite = mode_counts.most_common(1)[0][0] if mode_counts else "unknown"
    print("\n  Protocol-compliance modes:", dict(mode_counts))
    results["protocol_compliance"] = {
        "dataset": "SmellNet",
        "composite": composite,
        "per_substance_modes": dict(mode_counts),
        "note": (
            "SmellNet recordings are ~60s monotonic exposure ramps with NO "
            "recovery/desorption phase. Therefore kinetics/decay/hysteresis "
            "(recovery-phase) features are DEGENERATE here. Ablation deltas for "
            "those groups on this data are NOT evidence about their true "
            "discrimination power. Only protocol-collected data "
            "(baseline->exposure->recovery) can validate them."
        ),
    }
    if composite == "exposure_ramp":
        print("  !! WARNING: dataset's dominant mode lacks recovery phase.")
        print("     Kinetics/decay/hysteresis verdicts are INVALID on this data.")

    # ── Group ablations: remove one group at a time ────────────────────
    print("\n" + "=" * 70)
    print("Group ablations (remove group -> measure drop)")
    print("  [protocol-dependent groups marked * -> verdict INVALID on this data]")
    table = []
    for gname in sorted(groups.keys()):
        remove_idx = np.array(each_group[gname])
        keep_mask = np.ones(X.shape[1], dtype=bool)
        keep_mask[remove_idx] = False
        if keep_mask.sum() < 2:
            continue
        # Is this group protocol-dependent (contains recovery-phase features)?
        group_names = [str(all_names[i]) for i in remove_idx]
        is_protocol_dep = any(PROTOCOL_DEPENDENT_RE.search(n) for n in group_names)
        X_sub = X[:, keep_mask]
        accs = run_session_invariance(X_sub, y_all, sess_all, y_enc)
        mean = float(np.mean(accs)) if accs else 0.0
        delta = mean - full_mean
        t_stat, p_val = stats.ttest_1samp(accs, chance) if accs else (float("nan"), float("nan"))
        results["groups"][gname] = {
            "n_features_removed": int(len(remove_idx)),
            "mean_accuracy": mean,
            "delta_from_baseline": delta,
            "t_statistic": float(t_stat) if t_stat == t_stat else None,
            "p_value": float(p_val) if p_val == p_val else None,
            "protocol_dependent": bool(is_protocol_dep),
            "verdict_valid_on_this_data": not (is_protocol_dep and composite == "exposure_ramp"),
        }
        star = "*" if is_protocol_dep else " "
        table.append((gname, mean, delta))
        print(f"  Remove {gname:<35s}{star} ({len(remove_idx):>3d} feats) -> {mean*100:5.1f}%  Δ={delta*100:+5.1f}pp")

    # ── Flagged candidate ablations (single features) ─────────────────
    print("\n" + "=" * 70)
    print("Flagged ablation candidates (individual feature removal)")
    for cname, pat in flagged.items():
        remove_idx = [i for i, n in enumerate(all_names) if pat.search(n)]
        if not remove_idx:
            continue
        keep_mask = np.ones(X.shape[1], dtype=bool)
        keep_mask[remove_idx] = False
        X_sub = X[:, keep_mask]
        accs = run_session_invariance(X_sub, y_all, sess_all, y_enc)
        mean = float(np.mean(accs)) if accs else 0.0
        delta = mean - full_mean
        results["flagged_candidates"][cname] = {
            "features": [str(all_names[i]) for i in remove_idx],
            "mean_accuracy": mean,
            "delta_from_baseline": delta,
        }
        verdict = "KEEP" if delta < -0.005 else ("neutral" if abs(delta) <= 0.005 else "CONSIDER-REMOVE")
        print(f"  {'/'.join(str(all_names[i]) for i in remove_idx):<55s} Δ={delta*100:+5.1f}pp  [{verdict}]")
        # Conservative verdict: a feature is a removal candidate only if
        # removing it IMPROVES accuracy (positive delta) meaningfully.
        tags = results.setdefault("removal_candidates", {})
        if delta > 0.005:
            tags[cname] = {"features": [str(all_names[i]) for i in remove_idx], "delta": delta}

    # ── Diagnostic-role validity (separate criterion) ──────────────────
    print("\n" + "=" * 70)
    print("Diagnostic-role features: validated for DIAGNOSTIC validity,")
    print("  not discrimination accuracy (synthetic-fault responsiveness).")
    diag = validate_diagnostics_on_synthetic(roles, all_names)
    results["diagnostic_validity"] = diag

    # Simple responsiveness summary: does each health/hw feature move with a fault?
    responsive = {"drift": [], "noise": [], "poisoned": []}
    for name, row in diag["per_feature"].items():
        for fault, info in row.items():
            if abs(info["delta"]) > 1e-9:
                responsive[fault].append(name)
    results["diagnostic_responsive_features"] = responsive
    for fault, names_ in responsive.items():
        print(f"  Responsive to {fault:<9s}: {len(names_)} features")

    results["duration_s"] = round(time.time() - t0, 2)
    _write_results(args, results)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Baseline (all 187): {full_mean*100:.1f}%")
    print("\n  Largest drops (most important groups to KEEP):")
    for gname, mean, delta in sorted(table, key=lambda r: r[2]):
        print(f"    {gname:<35s} Δ={delta*100:+5.1f}pp")
    if results.get("removal_candidates"):
        print("\n  Removal candidates (removing improves accuracy):")
        for c, info in results["removal_candidates"].items():
            print(f"    {c}: Δ={info['delta']*100:+.1f}pp  {info['features']}")
    print(f"\n  Results written to {args.out}")


def _write_results(args, results):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    run()
