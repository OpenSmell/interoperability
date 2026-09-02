"""Response-type assignment: map an A3 selectivity fingerprint to a validated cluster.

This is the *scoped-down, honest* classifier. It assigns a sample to one of the
**2 device-invariant response-type clusters** that are actually measured on the
UCI Gas Drift set (Vergara et al., 2012; 13,910 samples, 16 sensors, 10 device
batches; see `results/percept_map_A3.json` and `results/falsify_four_category.json`).

Deliberately it does NOT implement:
  * a "strong vs weak" amplitude split       -> falsified (continuous magnitude)
  * oxidizing categories                     -> untested (no oxidizing data)
  * a kinetics grid                          -> untested (no protocol dynamics)

Those are open questions, documented in `docs/OPEN_QUESTIONS.md`, NOT classes
this module can report. The module is the engine->dashboard link: it CONSUMES the
A3 selectivity profile produced by `phenotype.compute_phenotype` (which itself
consumes the SDK's `extract_all_framework_features`); it does not reimplement
feature extraction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

MONOREPO = Path(__file__).resolve().parent.parent.parent.parent

# Cluster metadata decoded from the measured 2-cluster structure
# (see percept_map_A3.json). These are LABELS of measured structure, chosen by
# data (max silhouette, k=2), not hand-asserted percepts.
_CLUSTER_LABELS = {
    0: {
        "id": "small_reducing_voc",
        "measured_membership": "Acetone/Ethanol/Acetaldehyde/Toluene",
        "note": "cluster separated by A3 selectivity direction from cluster 1",
        "confidence": "medium",
    },
    1: {
        "id": "basic_reducing_gases",
        "measured_membership": "Ethylene/Ammonia",
        "note": "cluster separated by A3 selectivity direction from cluster 0",
        "confidence": "medium",
    },
}

_N_SENSORS = 16


# ---------------------------------------------------------------------------
# Learned-reference centroid model (from the validated Vergara data)
# ---------------------------------------------------------------------------

def _load_gas_features():
    """Re-load the raw Vergara sensor amplitudes (same parser as percept_map.py)."""
    gas_dir = MONOREPO / "smell-monitor" / "datasets" / "industrial_voc" / "uci_gas_drift"
    feats_per_sensor = 8
    X = []
    for b in range(1, 11):
        with open(gas_dir / f"batch{b}.dat") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                head, rest = line.split(";", 1)
                toks = rest.split()
                feats = {int(t.split(":")[0]): float(t.split(":")[1])
                         for t in toks[1:]}
                amp = np.array([feats[s * feats_per_sensor + 1]
                                for s in range(_N_SENSORS)], dtype=float)
                n = np.linalg.norm(amp)
                if n > 0:
                    amp = amp / n
                X.append(amp)
    return np.array(X)


def fit_centroids():
    """Run the same sample-level 2-cluster KMeans as percept_map.py and return
    the cluster centroids (L2-normalized) plus per-cluster metadata."""
    from sklearn.cluster import KMeans
    X = _load_gas_features()
    km = KMeans(n_clusters=2, n_init=20, random_state=0, max_iter=300,
                algorithm="lloyd", init="k-means++")
    km.fit(X)
    # centroids are already L2-normalized-ish (data on unit sphere); re-normalise
    centers = {}
    for c in range(2):
        v = km.cluster_centers_[c]
        n = np.linalg.norm(v)
        centers[c] = v / n if n else v
    return centers, km


def _save_reference(centers, dest):
    data = {
        "model": "KMeans k=2 on L2-normalized 16-sensor Vergara fingerprints",
        "k_selection": "maximum mean silhouette over k=2..8 (best k=2, silhouette 0.598)",
        "device_invariance": "confirmed on batch-1 and batches-2-10 (see percept_map_A3.json)",
        "clusters": {
            str(c): {"label": _CLUSTER_LABELS[c]["id"],
                     "measured_membership": _CLUSTER_LABELS[c]["measured_membership"],
                     "centroid": [round(x, 6) for x in centers[c].tolist()]}
            for c in centers
        },
        "caveat": (
            "These centroids are the measured reference. Assigning a NEW device's "
            "fingerprint assumes its A3 selectivity pattern sits in the same "
            "response-type space; true across-array device invariance is an open "
            "question (docs/OPEN_QUESTIONS.md Q5, docs/DEVICE_INVARIANCE_DISTINCTION.md)."
        ),
    }
    dest.write_text(json.dumps(data, indent=2) + "\n")
    return dest


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_response_type(
    selectivity_profile,
    centroids: Dict[int, np.ndarray] = None,
    *,
    a1_valence: str = "unknown",
) -> Dict:
    """Assign an A3 selectivity profile to a response-type cluster.

    Parameters
    ----------
    selectivity_profile : array-like
        The A3 unit selectivity vector from `phenotype.compute_phenotype`
        (L2-normalized over live channels).
    centroids : dict, optional
        {cluster_id: centroid vector}. Default: fitted from the validated
        Vergara data (cached reference).
    a1_valence : str
        The A1 redox valence ('reducing'/'oxidizing'/'unknown'/...) from the
        phenotype. Only 'reducing' is in the validated cluster space.

    Returns
    -------
    dict with cluster id, nearest-centroid distance, and honesty flags.
    """
    prof = np.asarray(selectivity_profile, dtype=float).ravel()
    n = np.linalg.norm(prof)
    if n < 1e-9:
        return {
            "assignable": False,
            "reason": "empty_flat_selectivity_profile",
            "response_type": None,
            "confidence": "unknown",
        }
    q = prof / n

    if centroids is None:
        centers, _ = fit_centroids()
    else:
        centers = {int(c): np.asarray(v, dtype=float) for c, v in centroids.items()}

    # cosine distance to each cluster centroid (equivalent to Euclidean on
    # unit-sphere data)
    dists = {}
    for c, v in centers.items():
        cv = np.asarray(v, dtype=float) / (np.linalg.norm(v) + 1e-12)
        cos = float(q @ cv)
        dists[c] = 1.0 - cos  # 0 = identical, 2 = opposite

    best = min(dists, key=dists.get)
    best_dist = dists[best]
    second = sorted(dists.values())[1] if len(dists) > 1 else 1.0

    # Honesty guard: only report a response-type if (a) A1 is consistent with the
    # validated (reducing) space, and (b) the assignment is reasonably confident.
    valence_ok = a1_valence in ("", "unknown", "reducing", None, "no_response")
    margin = second - best_dist

    if not valence_ok:
        return {
            "assignable": False,
            "reason": "a1_valence_outside_validated_space",
            "a1_valence": a1_valence,
            "response_type": None,
            "confidence": "unknown",
            "note": (
                "Response-type clusters are validated for reducing targets only. "
                "Oxidizing A1 is untested; refusing to classify (see "
                "docs/OPEN_QUESTIONS.md Q2)."
            ),
        }

    # coarse confidence from nearest-centroid separation and cluster size balance
    if best_dist < 0.3 and margin > 0.2:
        conf = "high"
    elif best_dist < 0.5:
        conf = "medium"
    else:
        conf = "low"

    label = _CLUSTER_LABELS[best]
    return {
        "assignable": True,
        "response_type": label["id"],
        "measured_membership": label["measured_membership"],
        "cluster_id": best,
        "nearest_centroid_cos_distance": round(best_dist, 4),
        "second_nearest_distance": round(second, 4),
        "margin": round(margin, 4),
        "confidence": conf,
        "a1_valence": a1_valence,
        "note": label["note"],
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    centers, km = fit_centroids()
    out = MONOREPO / "interoperability" / "perception-layer" / "results"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "response_type_reference.json"
    _save_reference(centers, dest)
    print(f"centroids saved to {dest}")

    # sanity: assign a couple of synthetic profiles
    import json as _json
    probe_A = np.zeros(16); probe_A[0] = 1.0
    probe_B = np.zeros(16); probe_B[8] = 1.0
    for name, p in (("probe_A", probe_A), ("probe_B", probe_B)):
        r = assign_response_type(p, centers, a1_valence="reducing")
        print(name, _json.dumps(r, indent=2))
