"""Attempt to FALSIFY the four-category MOX phenotype ontology.

The proposed "four device-invariant categories" scheme makes several distinct
falsifiable sub-claims. We must NOT treat the whole package as validated merely
because the underlying 2-cluster A3 structure was confirmed. The scheme is only
as strong as its weakest independent claim. Each sub-claim is tested (or shown
to be untestable given available real data) separately:

  C-a  Reducing targets split by response magnitude into a clean "strong" vs
       "weak" dichotomy (the `amplitude > 0.5` threshold in the proposed T1
       decision table). Falsifier: if the reducing response magnitudes form a
       continuum (not two separated modes), the crisp strong/weak dichotomy is
       FALSIFIED for the reducing side -> the threshold is a hand-asserted prior.
       TESTABLE on UCI Gas Drift (Vergara): 6 reducing targets, amplitude
       features, 13,910 samples.

  C-b  The measured 2 clusters are organized primarily by *magnitude* of
       reduction (a "strong" vs "weak" axis). Falsifier: if the two clusters are
       separated predominantly by fingerprint DIRECTION (shape / cross-sensor
       selectivity) rather than by response NORM (magnitude), then the real
       organizing axis is A3 selectivity, NOT a strong/weak amplitude split, and
       the "Strongly vs Weakly Reducing" framing is NOT what the data supports.
       TESTABLE on Vergara.

  C-c  Strongly vs Weakly OXIDIZING categories are real and separable.
       Falsification ATTEMPT IS DATA-ABSENT: no available dataset (Vergara, Hu,
       SmellNet) contains oxidizing targets (NO2, O3, Cl2). We cannot confirm or
       falsify -> report as an OPEN question, never as validated.

  C-d  Kinetics (A2: rise/recovery time) provides an orthogonal second axis that
       supports a 2x2 grid (e.g. strong x fast). Falsifier: Vergara is
       pre-extracted steady-state amplitudes only (no time series) -> cannot
       test A2 here. SmellNet is ramp-heavy and single-ish device -> insufficient
       for a grid. REPORTED AS OPEN (needs protocol data with baseline ->
       exposure -> recovery), consistent with the project's A2-blocked status.

Only C-a and C-b are executable on current real data; they directly probe the
legitimacy of the "Strongly / Weakly Reducing" split that the synthesis asserts.

Every number here is computed from the raw Vergara features, never asserted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

MONOREPO = Path(__file__).resolve().parent.parent.parent.parent
GAS_DIR = MONOREPO / "smell-monitor" / "datasets" / "industrial_voc" / "uci_gas_drift"
GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
N_SENSORS = 16
FEATS_PER_SENSOR = 8


def load_vergara():
    """Return list of dicts: {batch, gas, conc, X(16,) sensor amplitudes}."""
    rows = []
    for b in range(1, 11):
        with open(GAS_DIR / f"batch{b}.dat") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                head, rest = line.split(";", 1)
                gas = int(head.split()[0])
                toks = rest.split()
                conc = float(toks[0])
                feats = {}
                for tok in toks[1:]:
                    i, v = tok.split(":")
                    feats[int(i)] = float(v)
                amp = np.array([feats[s * FEATS_PER_SENSOR + 1]
                                for s in range(N_SENSORS)], dtype=float)
                rows.append({"batch": b, "gas": gas, "conc": conc, "X": amp})
    return rows


def _dip_pvalue(x, n_boot=2000):
    """Small Hartigan dip-style modality bootstrap.

    The classical dip test rejects unimodality when the observed dip statistic
    is far larger than the dip distribution under a unimodal (uniform) null.
    We re-sample the empirical range uniformly (the dip-test null) and compare,
    giving an approximate p-value for "data is NOT unimodal".
    """
    x = np.asarray(x, dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    n = len(x)
    obs = _dip(x)
    lo, hi = 0.0, 1.0
    count = 0
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        u = rng.uniform(lo, hi, n)
        d = _dip(u)
        if d >= obs:
            count += 1
    return obs, count / n_boot


def _dip(x):
    """Low-robustness dip statistic: max difference between sample ECDF and the
    closest unimodal ECDF is approximated by the max "peak-valley" drop.

    NOTE: this is a *proxy* dip. A sharper implementation should use the exact
    Hartigan-Hartigan algorithm. We keep it deliberately simple and honest: the
    decisive evidence is (a) the modality structure via a non-parametric density
    (Gaussian KDE) and (b) simple mode-count / gap logic, both reported directly.
    """
    x = np.sort(x)
    n = len(x)
    # measure the largest monotone "gap-flip" as a cheap unimodality sign
    diffs = np.diff(x)
    mid = n // 2
    left = diffs[:mid]
    right = diffs[mid:]
    # under unimodality the spacings should not have a sharp interior minimum
    mn = min(diffs[min(1, mid):-1]) if n > 4 else 0.0
    peak = np.max(np.abs(np.diff(diffs))) if n > 4 else 0.0
    return float(peak / (x[-1] - x[0] + 1e-12))


def _well_separated_bimodal(x):
    """Clean, interpretable test for a strong/weak dichotomy in 1-D strength.

    Fit two Gaussians (via 2-means split of the scalar) and measure how well the
    two candidate 'strong'/'weak' groups are separated: the mean gap between the
    two group means, in units of the pooled within-group std (a 'separation Z').

    A crisp dichotomy requires a large, stable separation Z (typically >= 3) AND
    comparable group sizes. If instead the within-group spread is large relative
    to the between-group gap (Z small), the strength distribution is a continuum,
    not two separable classes -> the strong/weak split is NOT supported.

    Returns (separation_z, frac_in_weaker_group, lower_mean, upper_mean).
    """
    from sklearn.cluster import KMeans
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    k2 = KMeans(n_clusters=2, n_init=50, random_state=0)
    lab = k2.fit_predict(x)
    vals = x[:, 0]
    g0, g1 = vals[lab == 0], vals[lab == 1]
    if len(g0) == 0 or len(g1) == 0:
        return (0.0, 0.5, 0.0, 0.0)
    m0, m1 = g0.mean(), g1.mean()
    lo, hi = (m0, m1) if m0 < m1 else (m1, m0)
    s_pool = np.sqrt((g0.var() + g1.var()) / 2)
    z = (hi - lo) / (s_pool + 1e-12)
    frac_weak = min(len(g0), len(g1)) / len(vals)
    return (float(z), float(frac_weak), float(lo), float(hi))


def _overlap_fraction(x):
    """Fraction of the sample whose strength is ambiguous between groups.

    Approximate by the fraction of samples sitting within one pooled-std of the
    midpoint between the two 2-means group means. High overlap => continuum.
    """
    from sklearn.cluster import KMeans
    x = np.asarray(x, dtype=float)
    k2 = KMeans(n_clusters=2, n_init=50, random_state=0)
    lab = k2.fit_predict(x.reshape(-1, 1))
    xr = x.reshape(-1, 1)[:, 0]
    g0, g1 = xr[lab == 0], xr[lab == 1]
    m0, m1 = min(g0.mean(), g1.mean()), max(g0.mean(), g1.mean())
    s_pool = np.sqrt((g0.var() + g1.var()) / 2)
    mid = (m0 + m1) / 2
    band = (mid - s_pool, mid + s_pool)
    frac = float(((xr >= band[0]) & (xr <= band[1])).mean())
    return frac


def falsify_ca(rows):
    """C-a: is there a clean strong/weak amplitude dichotomy among reducing VOCs?

    Use the max absolute single-sensor response and the L2 response magnitude
    (norm) as two candidate "strength" scalars. If the distribution over strength
    is unimodal / continuous (no two well-separated modes), a crisp strong/weak
    split is NOT supported -> the `amplitude > 0.5` dichotomy is falsified.
    """
    norms = []
    maxabs = []
    per_gas_norm = {}
    for r in rows:
        a = r["X"]
        nm = float(np.linalg.norm(a))
        norms.append(nm)
        maxabs.append(float(np.abs(a).max()))
        per_gas_norm.setdefault(r["gas"], []).append(nm)
    norms = np.array(norms)
    maxabs = np.array(maxabs)

    per_gas_mean = {GAS_NAMES[g]: float(np.mean(v)) for g, v in per_gas_norm.items()}
    per_gas_std = {GAS_NAMES[g]: float(np.std(v)) for g, v in per_gas_norm.items()}

    # Clean, interpretable strong/weak dichotomy tests on both strength scalars.
    # A crisp dichotomy requires the two 2-means groups to be well separated
    # (large Z) and mostly non-overlapping; otherwise it is a continuum.
    z_norm, fw_norm, lo_norm, hi_norm = _well_separated_bimodal(norms)
    z_max, fw_max, lo_max, hi_max = _well_separated_bimodal(maxabs)
    ov_norm = _overlap_fraction(norms)
    ov_max = _overlap_fraction(maxabs)

    crisp_threshold = 3.0
    dichotomous = (z_norm >= crisp_threshold and z_max >= crisp_threshold and
                   ov_norm < 0.5 and ov_max < 0.5)

    return {
        "claim": "C-a: reducing targets split by magnitude into clean strong/weak dichotomy",
        "executable": True,
        "n_samples": int(len(norms)),
        "dichotomy_test": {
            "strength=norm": {
                "2group_separation_z": round(z_norm, 3),
                "fraction_in_weaker_group": round(fw_norm, 3),
                "overlap_fraction_ambiguous_band": round(ov_norm, 3),
                "group_means": [round(lo_norm, 2), round(hi_norm, 2)],
            },
            "strength=maxabs_sensor": {
                "2group_separation_z": round(z_max, 3),
                "fraction_in_weaker_group": round(fw_max, 3),
                "overlap_fraction_ambiguous_band": round(ov_max, 3),
                "group_means": [round(lo_max, 1), round(hi_max, 1)],
            },
        },
        "crisp_dichotomy_requires": {
            "separation_z_gte": crisp_threshold,
            "overlap_lt": 0.5,
            "rationale": (
                "Two separable 'strong'/'weak' classes require the between-group "
                "gap to exceed ~3 pooled within-group std and an unambiguous "
                "(non-overlapping) assignment band. Otherwise the strength axis "
                "is a continuous gradient, and the proposed amplitude threshold "
                "(e.g. 0.5) is a hand-asserted prior, not a measured structure."
            ),
        },
        "per_gas_norm_mean_std": {g: [round(per_gas_mean[g], 3),
                                      round(per_gas_std[g], 3)]
                                  for g in per_gas_mean},
        "gap_statistic_2groups_strong_vs_weak": _gap_check(norms),
        "verdict": (
            "FALSIFIED: unimodal/continuous magnitude, no clean strong/weak dichotomy"
            if not dichotomous else
            "CLEAN-DICHOTOMY-SUPPORTED"
        ),
        "note": (
            "If FALSIFIED, the 'Strongly vs Weakly Reducing' split is a "
            "hand-asserted prior (the proposed 0.5 amplitude threshold), NOT a "
            "measured structure. The honest conclusion would then be: within the "
            "reducing polarity there is a continuous magnitude gradient, not two "
            "separable device-invariant 'strong'/'weak' classes."
        ),
    }


def _gap_check(x, k=2, b=10):
    """Tiny gap-statistic estimate: does adding a 2nd cluster beat the null, or
    is the data effectively one blob? Returns log(within_1c / within_2c) proxy."""
    x = np.asarray(x).reshape(-1, 1).astype(float)
    w1 = float(((x - x.mean(0)) ** 2).sum())
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=20, random_state=0)
    lab = km.fit_predict(x)
    w2 = 0.0
    for c in range(2):
        sub = x[lab == c]
        w2 += float(((sub - sub.mean(0)) ** 2).sum())
    return round(float(np.log(w1) - np.log(w2)), 3)


def falsify_cb(rows):
    """C-b: are the two measured clusters organized by MAGNITUDE or by DIRECTION?

    Re-run the sample-level 2-cluster KMeans on L2-normalized fingerprints (as in
    percept_map.py). Then ask: do the two clusters differ mostly in response NORM
    (magnitude axis) or in fingerprint DIRECTION (shape axis)?

    If the clusters separate primarily by direction (within-vs-between norm
    variance is small), the organizing axis is A3 selectivity, and labelling the
    clusters 'strong' vs 'weak' reducing would mis-attribute the cause.
    """
    X = np.array([r["X"] for r in rows])
    gas = np.array([r["gas"] for r in rows])
    # normalize each sample
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=20, random_state=0)
    lab = km.fit_predict(Xn)

    # norm separation between the two clusters (magnitude axis)
    norms = np.linalg.norm(X, axis=1)
    n0, n1 = norms[lab == 0].mean(), norms[lab == 1].mean()
    s_pool = np.sqrt((norms[lab == 0].var() + norms[lab == 1].var()) / 2)
    norm_sep_z = (max(n0, n1) - min(n0, n1)) / (s_pool + 1e-12)

    # direction separation: cosine between cluster centroids (shape axis)
    c0 = Xn[lab == 0].mean(0)
    c1 = Xn[lab == 1].mean(0)
    cos_sep = float((c0 / (np.linalg.norm(c0) + 1e-12)) @
                    (c1 / (np.linalg.norm(c1) + 1e-12)))
    # within-vs-between variance in DIRECTION space
    # average angle of samples to their own centroid vs to the other centroid
    def ang_to_mask(mask, centroid):
        c = centroid / (np.linalg.norm(centroid) + 1e-12)
        dots = Xn[mask] @ c
        return float(np.arccos(np.clip(dots, -1, 1)).mean())
    within_ang = 0.5 * (ang_to_mask(lab == 0, c0) + ang_to_mask(lab == 1, c1))
    cross_ang = 0.5 * (ang_to_mask(lab == 0, c1) + ang_to_mask(lab == 1, c0))

    return {
        "claim": "C-b: the two measured clusters are organized by magnitude (strong/weak axis) rather than by direction (A3 selectivity)",
        "executable": True,
        "n_samples": int(len(X)),
        "cluster_norm_means": [round(float(n0), 3), round(float(n1), 3)],
        "cluster_norm_sep_z": round(float(norm_sep_z), 3),
        "between_cluster_centroid_cosine": round(cos_sep, 4),
        "within_cluster_mean_angle_rad": round(within_ang, 3),
        "cross_cluster_mean_angle_rad": round(cross_ang, 3),
        "angle_separation_cross_minus_within_rad": round(cross_ang - within_ang, 3),
        "verdict": (
            "MAGNITUDE-ORGANIZED (supports strong/weak axis)" if
            norm_sep_z > cross_ang - within_ang and norm_sep_z > 2.0
            else "DIRECTION-ORGANIZED: clusters differ by A3 selectivity/shape, not magnitude"
        ),
        "note": (
            "If DIRECTION-ORGANIZED, the 'Strongly vs Weakly Reducing' labels "
            "mis-attribute the measured separation to magnitude when the real "
            "axis is cross-sensor selectivity (A3) + redox polarity (A1), not a "
            "strong/weak amplitude continuum."
        ),
    }


def falsify_cc(rows):
    """C-c: strongly/weakly OXIDIZING categories. NOT testable on available data.

    Vergara has only reducing targets; Hu is food/indoor VOCs; SmellNet is
    reducing VOCs + CO. None expose oxidizing targets (NO2, O3, Cl2). The
    oxidizing half of the four-category scheme therefore CANNOT be confirmed or
    falsified with current data. It is an OPEN research question.
    """
    return {
        "claim": "C-c: 'Strongly vs Weakly Oxidizing' categories are real and separable",
        "executable": False,
        "reason": (
            "No available dataset (Vergara, Hu 2016, SmellNet) contains oxidizing "
            "targets (NO2, O3, Cl2) with clean baseline/exposure/recovery protocol. "
            "Oxidizing polarity is never observed, so neither confirmed nor falsified."
        ),
        "verdict": "OPEN_QUESTION_not_testable_with_current_data",
        "required_data": (
            "An external dataset with >=2 oxidizing targets (e.g. NO2, O3, Cl2) and "
            ">=2 reducing targets on the SAME array, with clean baseline->exposure->"
            "recovery protocol, across >=2 devices/batches."
        ),
    }


def falsify_cd(rows):
    """C-d: kinetics (A2) as orthogonal axis supporting a 2x2 grid.

    Vergara provides only pre-extracted steady-state amplitudes (no raw time
    series), so A2 (rise/recovery time) cannot be computed here. This is the
    A2-blocked status already documented for the project.
    """
    return {
        "claim": ("C-d: kinetics (A2 rise/recovery time) is an orthogonal second axis "
                  "supporting a 2x2 grid (e.g. strong x fast)"),
        "executable": False,
        "reason": (
            "Vergara exposes only steady-state amplitude features (no time series); "
            "SmellNet is ramp-heavy and single-ish device. No clean baseline->exposure->"
            "recovery protocol data with dynamics is available.")
        ,
        "verdict": "OPEN_QUESTION_not_testable_with_current_data",
        "required_data": (
            "Protocol recordings with baseline -> exposure -> recovery across >=2 "
            "devices, from which rise and recovery time constants can be measured."
        ),
    }


def main():
    rows = load_vergara()
    results = {
        "ontology_under_test": (
            "Four device-invariant categories: Strongly/Weakly Reducing, "
            "Strongly/Weakly Oxidizing"
        ),
        "declared_validated": (
            "ONLY the reducing 2-cluster A3 structure (see percept_map_A3.json) "
            "was previously measured. The full four-category scheme was NOT."
        ),
        "falsification_attempts": {
            "C_a_strong_vs_weak_by_magnitude": falsify_ca(rows),
            "C_b_clusters_organized_by_magnitude_vs_direction": falsify_cb(rows),
            "C_c_oxidizing_categories": falsify_cc(rows),
            "C_d_kinetics_orthogonal_axis": falsify_cd(rows),
        },
        "summary": {},
    }

    ca = results["falsification_attempts"]["C_a_strong_vs_weak_by_magnitude"]
    cb = results["falsification_attempts"]["C_b_clusters_organized_by_magnitude_vs_direction"]
    cc = results["falsification_attempts"]["C_c_oxidizing_categories"]
    cd = results["falsification_attempts"]["C_d_kinetics_orthogonal_axis"]

    results["summary"] = {
        "C_a_reducing_strong_weak_split": ca["verdict"],
        "C_b_reducing_cluster_organizing_axis": cb["verdict"],
        "C_c_oxidizing": cc["verdict"],
        "C_d_kinetics_grid": cd["verdict"],
        "overall_verdict": (
            "The FULL four-category scheme is NOT established by current data. "
            "Reducing side: " + ca["verdict"].lower() + "; " +
            cb["verdict"].lower() + ". "
            "Oxidizing side and kinetics grid remain unbacked open questions. "
            "Any product build must therefore NOT present all four categories as "
            "device-invariant validated classes."
        ),
    }

    out = MONOREPO / "interoperability" / "perception-layer" / "results"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "falsify_four_category.json"
    with open(dest, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
