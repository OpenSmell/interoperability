"""Topology-aware ADC -> resistance normalization.

Purpose
-------
The single most important hard-to-vary move for the measured-phenotype layer:
the *sign* of a MOX response (reducing vs oxidizing) is a statement about
**resistance**, not about the raw ADC reading. Whether a reducing gas appears as
"ADC up" or "ADC down" depends entirely on how the MOX sensor (a resistor whose
resistance FALLS under a reducing gas) is wired in the voltage divider.

This module converts raw ADC counts into physical sensor resistance, using the
voltage-divider topology from the sensor profile, so downstream features
(and especially the redox sign) are invariant to how the hardware is wired.

Nothing here reimplements OSMELL feature extraction. It produces the corrected
resistance signal that the OSMELL/feature layer and the phenotype layer consume.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class DividerConfig:
    """Voltage-divider topology needed to recover sensor resistance.

    Attributes
    ----------
    supply_voltage_vcc : float
        The divider supply voltage (V). Required.
    load_resistor_rl_ohm : float
        The fixed load resistor in series with the sensor (ohm). Required.
    sensor_on_low_side : bool
        True if the sensor is the LOW-side element (between ADC node and
        ground); False if the sensor is the HIGH-side element (between VCC and
        the ADC node). This is the topology that flips the ADC sign.
    adc_bits : Optional[int]
        ADC resolution in bits, if counts come from an N-bit ADC.
    adc_reference_voltage : Optional[float]
        ADC full-scale reference (V). If None, assumed == supply_voltage_vcc.
    adc_max_count : Optional[float]
        IEEE full-scale bin if adc_bits set (default 2**adc_bits - 1).
    """

    supply_voltage_vcc: float
    load_resistor_rl_ohm: float
    sensor_on_low_side: bool
    adc_bits: Optional[int] = None
    adc_reference_voltage: Optional[float] = None
    adc_max_count: Optional[float] = None

    @classmethod
    def from_profile(cls, channel: int, profile: Dict) -> "DividerConfig":
        circ = profile.get("circuit", {})
        if not circ.get("is_voltage_divider", True):
            raise ValueError("Only voltage-divider circuits supported by this normalizer")

        adc_bits = circ.get("adc_bits")
        adc_max = circ.get("adc_max_count")
        if adc_max is None and adc_bits is not None:
            adc_max = float((1 << adc_bits) - 1)

        # channel-level override is allowed; falls back to circuit-level
        sensor_on_low_side = circ.get("sensor_on_low_side", True)

        vcc = _req(circ.get("supply_voltage_vcc"), "circuit.supply_voltage_vcc")
        rl = _req(circ.get("load_resistor_rl_ohm"), "circuit.load_resistor_rl_ohm")

        return cls(
            supply_voltage_vcc=float(vcc),
            load_resistor_rl_ohm=float(rl),
            sensor_on_low_side=bool(sensor_on_low_side),
            adc_bits=int(adc_bits) if adc_bits is not None else None,
            adc_reference_voltage=(
                float(circ["adc_reference_voltage"])
                if circ.get("adc_reference_voltage") is not None
                else None
            ),
            adc_max_count=float(adc_max) if adc_max is not None else None,
        )


def _req(v, name):
    if v is None:
        raise ValueError(
            f"Missing required circuit parameter '{name}'. "
            "The sensor profile must describe the divider to normalize the sign."
        )
    return v


# ---------------------------------------------------------------------------
# ADC counts -> ADC voltage
# ---------------------------------------------------------------------------

def adc_count_to_voltage(
    counts: np.ndarray,
    cfg: DividerConfig,
    vref: Optional[float] = None,
) -> np.ndarray:
    """Convert raw ADC counts (int) to the voltage at the ADC node (V)."""
    counts = np.asarray(counts, dtype=float)
    adc_max = cfg.adc_max_count
    if adc_max is None:
        # assume ADC reads directly in volts
        return counts
    if vref is None:
        vref = cfg.adc_reference_voltage or cfg.supply_voltage_vcc
    if adc_max <= 0:
        raise ValueError("adc_max_count must be > 0")
    return counts * (float(vref) / adc_max)


# ---------------------------------------------------------------------------
# ADC voltage -> sensor resistance (topology-aware)
# ---------------------------------------------------------------------------

def voltage_to_resistance(
    v_adc: np.ndarray,
    cfg: DividerConfig,
) -> np.ndarray:
    """Recover the MOX sensor resistance as a function of time.

    Two topologies exist for a series divider of the load resistor RL in series
    with the sensor Rs across VCC:

    * Sensor LOW side  (common MQ-* eval boards):
          VCC -- RL -- (ADC node) -- Rs -- GND
          Rs = RL * V_adc / (VCC - V_adc)
      A reducing gas lowers Rs -> V_adc falls.

    * Sensor HIGH side:
          VCC -- Rs -- (ADC node) -- RL -- GND
          Rs = RL * (VCC - V_adc) / V_adc
      A reducing gas lowers Rs -> V_adc RISES.

    The returned series is *always* the sensor resistance, so the sign of its
    change is physically meaningful regardless of topology.
    """
    v_adc = np.asarray(v_adc, dtype=float)
    vcc = float(cfg.supply_voltage_vcc)
    rl = float(cfg.load_resistor_rl_ohm)
    with np.errstate(divide="ignore", invalid="ignore"):
        if cfg.sensor_on_low_side:
            rs = rl * v_adc / (vcc - v_adc)
        else:
            rs = rl * (vcc - v_adc) / v_adc
    # guard division-by-zero at rail and negative/clipped ADC (physically absurd)
    rs = np.where(np.isfinite(rs) & (rs > 0.0), rs, np.nan)
    return np.asarray(rs, dtype=float)


def normalize(
    adc_counts: np.ndarray,
    cfg: DividerConfig,
) -> np.ndarray:
    """Full pipeline: counts -> V_adc -> Rs (physical resistance)."""
    v = adc_count_to_voltage(adc_counts, cfg)
    return voltage_to_resistance(v, cfg)


# ---------------------------------------------------------------------------
# Redox sign from the corrected resistance
# ---------------------------------------------------------------------------

def redox_sign(
    resistance: np.ndarray,
    r0_samples: int = 15,
    rel_threshold: float = 0.01,
    r0: Optional[np.ndarray] = None,
) -> str:
    """Classify redox valence from a corrected resistance trace.

    Sign convention: on n-type MOX, a reducing analyte makes resistance FALL
    (electron donation); an oxidizing analyte makes it RISE. This is now immune
    to wiring topology because `resistance` is physical, not raw ADC.

    `r0`, if given, is the per-channel pre-exposure baseline resistance (e.g. from
    the baseline phase of the `.osmell` protocol). If omitted, the first
    `r0_samples` samples are used — only valid if the recording BEGINS at baseline
    (as Hu 2016 with negative pre-exposure time and SmellNet ramps do).

    Returns one of "reducing", "oxidizing", "inert", "mixed", "unknown".
    """
    arr = np.asarray(resistance, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < 20:
        return "unknown"
    nch = arr.shape[1]
    if r0 is not None:
        r0_arr = np.asarray(r0, dtype=float).reshape(-1)[:nch]
    else:
        r0_arr = np.array(
            [float(np.median(arr[:, c][np.isfinite(arr[:, c])][
                : min(r0_samples, max(1, np.sum(np.isfinite(arr[:, c])) // 2))
            ])) if np.any(np.isfinite(arr[:, c])) else np.nan
            for c in range(nch)]
        )
    signs = []
    for c in range(nch):
        s = arr[:, c]
        s = s[np.isfinite(s)]
        if len(s) < 20:
            continue
        r0c = r0_arr[c]
        if not np.isfinite(r0c) or abs(r0c) < 1e-12:
            continue
        # robust full-range excursion
        lo = float(np.percentile(s, 5))
        hi = float(np.percentile(s, 95))
        spread = hi - lo
        if abs(spread / abs(r0c)) < rel_threshold:
            signs.append("inert")
            continue
        # A reducing gas lowers n-type resistance -> the dominant excursion is DOWN
        if abs(lo - r0c) >= abs(hi - r0c):
            signs.append("reducing" if lo < r0c else "oxidizing")
        else:
            signs.append("oxidizing" if hi > r0c else "reducing")
    if not signs:
        return "unknown"
    # array majority; mixed channels -> mixed
    from collections import Counter
    cnt = Counter(signs)
    top, ntop = cnt.most_common(1)[0]
    if top != "inert" and ntop < len(signs) * 0.5 and len(cnt) > 1:
        return "mixed"
    return top


# ---------------------------------------------------------------------------
# Unit sanity (not a substitute for dataset validation)
# ---------------------------------------------------------------------------

def _demo():
    """Show the sign-flip: same physical signal, two topologies, same redox sign."""
    import numpy as np

    vcc, rl = 3.3, 10_000.0
    t = np.linspace(0, 5, 500)
    # physical resistance trace: 12k -> 8k -> 10k (a reducing pulse, n-type: falls)
    rs_physical = 12_000.0 - 4_000.0 * np.exp(-((t - 1.5) ** 2) / 0.4) + 2_000.0 * np.exp(
        -((t - 3.5) ** 2) / 0.5
    )

    def v_adc_for(rs, low_side):
        if low_side:
            return vcc * rs / (rl + rs)  # sensor low side
        return vcc * rl / (rl + rs)  # sensor high side (ADC across RL)

    for low_side, label in ((True, "sensor-LOW-side"), (False, "sensor-HIGH-side")):
        v = v_adc_for(rs_physical, low_side)
        cfg = DividerConfig(vcc, rl, low_side)
        recovered = voltage_to_resistance(v, cfg)
        sign = redox_sign(recovered)
        print(f"{label:18s} raw_ADC_dir={('DOWN' if v[0] > v[-1] else 'UP'):5s} "
              f"redox={sign}  ΔR_recovered={recovered[-1]-recovered[0]:+.0f} ohm")


if __name__ == "__main__":
    _demo()
