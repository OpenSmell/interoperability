"""Percept + use-case guidance, derived ONLY from validated response-types.

This is the scoped-down, honest end-user layer. It consumes the response-type
asset from `response_type.assign_response_type` (which is based on the 2
device-invariant clusters measured on Vergara 2012) and maps it to a
human-perceptible label plus use-case guidance.

Boundaries (what this deliberately does NOT claim):
  * does NOT identify individual molecules (gas purity 0.405 => not separable),
  * does NOT report oxidizing categories (untested),
  * does NOT report a strong/weak amplitude class (falsified),
  * does NOT decompose mixtures (untested).

Those limits are always surfaced in the returned object so the interface cannot
over-claim.
"""

from __future__ import annotations

from typing import Dict

# Percept labels are tied to the *measured membership* of each response-type
# cluster (see percept_map_A3.json). The word is a chemistry-grounded HYPOTHESIS
# attached to a measured cluster, and the measured-membership string is always
# carried alongside so the label can't float free of what was actually observed.
_PERCEPT_BY_TYPE = {
    "small_reducing_voc": {
        "percept": "reducing-VOC/solvent-like event",
        "measured_membership": "Acetone/Ethanol/Acetaldehyde/Toluene",
        "plain_language": "A strongly-reducing volatile-organic event (e.g. "
                          "solvent/alcohol-like) — one of the two response types "
                          "the array can separate from first principles.",
        "confidence_basis": "cluster membership (A3 selectivity), medium confidence",
    },
    "basic_reducing_gases": {
        "percept": "basic/reducing-gas event",
        "measured_membership": "Ethylene/Ammonia",
        "plain_language": "A basic/reducing gas event (ammoniacal / alkene-like) "
                          "— the other response type the array can separate.",
        "confidence_basis": "cluster membership (A3 selectivity), medium confidence",
    },
}

# Use-case guidance: actionable next steps per response-type. These are framed as
# hypotheses for the operator to verify, NOT as confirmed positive detections of
# a specific hazard, because the array can only separate response types.
_GUIDANCE_BY_TYPE = {
    "small_reducing_voc": [
        "Food spoilage: a reducing-VOC event is consistent with metabolic "
        "volatiles (alcohols/esters). Confirm locally before acting; the array "
        "cannot name the molecule.",
        "Fermentation: a reducing-VOC event is expected during active "
        "fermentation; trend it over time rather than trusting a single "
        "reading.",
        "Breath/ketosis screening: an elevated reducing-VOC response is "
        "consistent with acetone-type compounds, but is NOT a confirmed "
        "ketosis diagnosis — corroborate with a reference measurement.",
    ],
    "basic_reducing_gases": [
        "Leak detection: an ammonia/alkene-type reducing event is plausible for "
        "NH3 or light hydrocarbon leaks. Treat as a leak ALERT and verify with "
        "a calibrated reference — the array does not identify the specific gas.",
        "Air quality: a basic/reducing-gases response warrants ventilation "
        "if corroborated; it is one response type, not a molecule ID.",
    ],
}

_UNKNOWN_TYPE = "unknown_response_type"


def map_percept(response_type: str) -> Dict:
    """Map a validated response-type id to a percept label and plain language.

    response_type : the id from `response_type.assign_response_type(...)`
        (e.g. 'small_reducing_voc', 'basic_reducing_gases') or None.
    """
    if response_type is None:
        return {
            "percept": None,
            "measured_membership": None,
            "plain_language": (
                "No validated response-type could be assigned. The array "
                "separates only two reducing response types; anything else is "
                "outside the validated space."
            ),
            "confidence_basis": None,
            "boundaries_cannot": _boundaries(),
        }
    meta = _PERCEPT_BY_TYPE.get(response_type)
    if meta is None:
        return {
            "percept": None,
            "measured_membership": None,
            "plain_language": f"Unknown response-type id: {response_type!r}.",
            "confidence_basis": None,
            "boundaries_cannot": _boundaries(),
        }
    return {
        **meta,
        "boundaries_cannot": _boundaries(),
    }


def guidance(response_type: str, confidence: str = "medium") -> Dict:
    """Return use-case guidance for a validated response-type.

    guidance is returned as candidate NEXT STEPS to verify, keyed by use case,
    not as confirmed hazard declarations.
    """
    steps = _GUIDANCE_BY_TYPE.get(response_type, [])
    return {
        "response_type": response_type,
        "guidance": steps,
        "confidence": confidence,
        "note": (
            "Guidance is framed as 'verify this hypothesis' — the validated "
            "output is a response-type, not a molecule identification, so "
            "operational actions require local/reference confirmation."
        ),
        "unvalidated_capabilities": {
            "molecule_identification": "cannot",
            "oxidizing_detection": "cannot (untested)",
            "mixture_decomposition": "cannot (untested)",
            "strong_weak_amplitude_class": "cannot (falsified)",
        },
    }


def _boundaries():
    return [
        "absolute_ppm/concentration",
        "exact_molecule_identification",
        "oxidizing_category",
        "mixture_decomposition",
        "strong_vs_weak_amplitude_class",
    ]


if __name__ == "__main__":
    import json
    for tid in ("small_reducing_voc", "basic_reducing_gases", None):
        print("== percept ==")
        print(json.dumps(map_percept(tid), indent=2))
        if tid:
            print("== guidance ==")
            print(json.dumps(guidance(tid), indent=2))
