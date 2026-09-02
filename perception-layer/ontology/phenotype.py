"""Measured-phenotype layer: compute the physics axes A1/A2/A3 from a recording.

This is the empirical companion to the `opensmell/mox/smellability/` theory
engine. It turns a single (or multi-channel) corrected-resistance recording into
the three hard-to-vary axes defined in PHENOTYPE_ONTO.md:

    A1 redox valence     (reducing / oxidizing / inert / mixed)
    A2 kinetic binding   (fast/slow rise; fast/slow recovery)
    A3 cross-sensor selectivity profile (unit vector over channels)

It CONSUMES the SDK extractor (`opensmell.mox.features`) for the per-channel
dynamic features — it does not reimplement them. The one required input is the
*corrected* resistance series from `normalize.py`, so the redox sign is
topology-invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

MONOREPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(MONOREPO / "opensmell"))

from opensmell.mox.features import (  # noqa: E402
    extract_all_framework_features,
    feature_names,
)

from normalize import redox_sign as _redox_sign  # noqa: E402


# ---------------------------------------------------------------------------
# Axis A2 — kinetic binding
# ---------------------------------------------------------------------------

def _kinetic_class(seconds, fast_bound_s=3.0, slow_bound_s=20.0):
    """Map a rise/decay time in seconds to a qualitative binding class.

    These bounds are intentionally coarse and shape-only; they are calibrated on
    the available datasets (Vergara/Hu) in the validation harness, not here.
    A None (missing/non-finite) value -> "unknown".
    """
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    if seconds <= fast_bound_s:
        return "fast"
    if seconds <= slow_bound_s:
        return "medium"
    return "slow"


# ---------------------------------------------------------------------------
# Axis A3 — cross-sensor selectivity profile
# ---------------------------------------------------------------------------

def _selectivity_profile(relative_amplitudes: np.ndarray) -> np.ndarray:
    """Unit vector over channels from each channel's relative response.

    relative_amplitudes[i] = |(peak - R0)/R0| on channel i (device-agnostic,
    already resistance-normalized). Returns the L2-normalized selectivity vector
    (shape-comparison uses the *pattern*, not absolute magnitude).
    """
    a = np.abs(np.asarray(relative_amplitudes, dtype=float))
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.zeros_like(a)
    return a / norm


def _dead_channels(resistance: np.ndarray, amplitudes: np.ndarray,
                   saturating_value: float = 1e8) -> list:
    """Boolean dead-channel mask.

    A channel is 'dead' (excluded from the A3 profile and sign agreement) when
    it cannot carry real signal:
      * non-finite resistance anywhere in the recording (open/rail/NaN),
      * saturated at a physically implausible value (e.g. an ADC readback of
        4294967295 from a dead/no-response channel),
      * invariant / no measurable relative response (relative amplitude below
        a tiny floor), which otherwise would poison the L2 selectivity norm.
    """
    r = np.asarray(resistance, dtype=float)
    amps = np.asarray(amplitudes, dtype=float)
    nch = r.shape[1] if r.ndim == 2 else 1
    dead = []
    for i in range(nch):
        col = r[:, i]
        sat = np.nanmax(np.abs(col)) if len(col) else 0.0
        nonfinite = int(np.isfinite(col).sum()) < len(col)
        saturated = bool(sat and np.isfinite(sat) and sat >= saturating_value)
        invariant = bool(not np.isfinite(amps[i]) or abs(float(amps[i])) < 1e-9)
        dead.append(bool(nonfinite or saturated or invariant))
    return dead


# ---------------------------------------------------------------------------
# Main: compute the phenotype verdict (evidence chain)
# ---------------------------------------------------------------------------

def compute_phenotype(
    corrected_resistance: np.ndarray,
    *,
    r0_samples: int = 15,
    sr: float = 10.0,
    dataset: str = "unknown",
    session_id="?",
    device_profile: str = "default",
    has_environment: bool = False,
) -> Dict:
    """Compute A1/A2/A3 + evidence chain from a corrected resistance recording.

    Parameters
    ----------
    corrected_resistance : np.ndarray
        Multi-channel physical resistance (n_samples x n_channels), from
        `normalize.normalize()`. Pass None/0 for a channel the device lacks.
    """
    r = np.asarray(corrected_resistance, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    if r.shape[0] < 20:
        return _unknown(dataset, session_id, device_profile, reason="too_short")

    # A1: redox valence directly from corrected resistance (topology-invariant)
    a1_raw = _redox_sign(
        r,
        r0_samples=r0_samples,
        r0=None,  # SDK handles baseline; A1 sign only needs the corrected resistance
    )

    # A2/A3: run the SDK extractor on the *corrected* resistance so its
    # direction/rise/decay/amplitude agree with physics, not wiring.
    feats = extract_all_framework_features(
        r, r0_samples=min(r0_samples, max(1, r.shape[0] // 2)), sr=sr
    )
    names = feature_names()
    F = {n: feats.get(n, 0.0) for n in names}
    nch = r.shape[1]

    amplitudes = [F.get(f"ch{i}_da_relative_amplitude", 0.0) for i in range(nch)]
    directions = [F.get(f"ch{i}_da_direction", 0.0) for i in range(nch)]
    rise_times = [F.get(f"ch{i}_da_rise_time", np.nan) for i in range(nch)]
    decay_times = [F.get(f"ch{i}_da_decay_time", np.nan) for i in range(nch)]

    # A2: fastest channel's rise; recovery only meaningful if a decay value is
    # finite and non-degenerate (i.e. the recording has a recovery phase).
    finite_rises = [t for t in rise_times if np.isfinite(t) and t > 0]
    rise = min(finite_rises) if finite_rises else None
    finite_decays = [t for t in decay_times if np.isfinite(t) and t > 0]
    recovery = min(finite_decays) if finite_decays else None

    # A3: unit selectivity vector over non-dead channels
    _dead_mask = _dead_channels(r, amplitudes)
    alive = [i for i in range(nch) if not _dead_mask[i]]
    sel_full = _selectivity_profile(np.array(amplitudes, dtype=float))
    sel = np.zeros_like(sel_full)
    if alive:
        sel[alive] = _selectivity_profile(
            np.array([amplitudes[i] for i in alive], dtype=float)
        )
    else:
        sel = sel_full

    # Channel-level redox agreement -> high/low A1 confidence (live channels only)
    reducing_n = sum(1 for i in alive
                     if directions[i] not in (0, None))
    n_active = len(alive)
    a1_conf = "high"
    if a1_raw == "unknown" or a1_raw == "no_response":
        a1_conf = "unknown"
    elif n_active == 0:
        a1_conf = "low"
    elif n_active < 3:
        a1_conf = "medium"

    # A3 confidence: degrades without environment/device conditioning
    a3_conf = "high" if (has_environment and nch >= 4) else "medium"

    return {
        "axes": {
            "A1_redox_valence": a1_raw,
            "A2_kinetic_rise": _kinetic_class(rise),
            "A2_kinetic_recovery": _kinetic_class(recovery)
            if (recovery is not None and np.isfinite(recovery) and recovery > 0)
            else "unknown",
            "A3_selectivity_profile": sel.tolist(),
        },
        "confidence": {
            "A1": a1_conf,
            "A2_rise": "medium",
            "A2_recovery": "unknown" if recovery is None else "low",
            "A3": a3_conf,
        },
        "evidence": {
            "per_channel_amplitude": amplitudes,
            "per_channel_direction": directions,
            "per_channel_rise_time_s": rise_times,
            "per_channel_decay_time_s": decay_times,
        },
        "source": {
            "dataset": dataset,
            "session": session_id,
            "device_profile": device_profile,
            "has_environment": bool(has_environment),
        },
        "boundaries_cannot": [
            "absolute_ppm",
            "exact_molecule",
            "mixture_decomposition",
        ],
    }


def _unknown(dataset, session_id, device_profile, reason):
    return {
        "axes": {
            "A1_redox_valence": "unknown",
            "A2_kinetic_rise": "unknown",
            "A2_kinetic_recovery": "unknown",
            "A3_selectivity_profile": [],
        },
        "confidence": {"A1": "unknown", "A2_rise": "unknown",
                       "A2_recovery": "unknown", "A3": "unknown"},
        "evidence": {},
        "source": {"dataset": dataset, "session": session_id,
                   "device_profile": device_profile},
        "reason": reason,
        "boundaries_cannot": [
            "absolute_ppm",
            "exact_molecule",
            "mixture_decomposition",
        ],
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # self-test on a synthetic reducing pulse with baseline -> exposure -> recovery
    t = np.linspace(0, 30, 600)
    base = 12_000.0
    # baseline 0-5s; exposure 5-13s (resistance falls -> reducing); recovery after
    rc = base * np.ones_like(t)
    ex = (t >= 5) & (t <= 13)
    rec = t > 13
    rc[ex] = base - 6_000.0
    rc[rec] = base - 6_000.0 * np.exp(-((t[rec] - 13) / 1.5))
    sig = np.stack([rc, rc * 0.9, rc * 1.1, rc * 1.05, rc * 0.95, rc * 1.0], axis=1)
    v = compute_phenotype(sig, sr=20.0, dataset="selftest")
    import json
    print(json.dumps({k: v[k] for k in ("axes", "confidence")}, indent=2))
