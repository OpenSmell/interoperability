"""Auto-detect the MOX response convention (wiring topology + doping type)
with a hard-to-vary, honest treatment of what the data can and cannot recover.

THE HARD FACT (verified numerically, not assumed)
-------------------------------------------------
The two divider-topology maps are algebraic inverses:
    R_low  = RL * v / (Vcc - v)
    R_high = RL * (Vcc - v) / v            with  R_low * R_high = RL^2

For a like-doped array this means the raw ADC trace CANNOT separate the
topology<->doping "mirror pair":

    (low-side,  n-type)  <-- physically identical ADC pattern -->  (high-side, p-type)
    (high-side, n-type)  <-- physically identical ADC pattern -->  (low-side,  p-type)

Every shape-based criterion is exactly tied between the two members of a pair:
  * validity   - both give finite, positive resistance for v in (0, Vcc)
  * recovery   - a monotone 1-1 map preserves "returns to baseline"
  * agreement  - the same map applies to every like-doped channel
  * log-span   - numerically identical (up to the RL^2 shift)

We additionally tested an absolute-magnitude lever (C4: does the recovered
resistance fall in a plausible 1k..10M MOX window?) and found it FRAGILE and
sometimes confidently WRONG near mid-rail (base ~ RL), so it must not be used
to fabricate auto-certainty.

CONSEQUENCE / PRODUCT DECISION
-------------------------------
1. We ALWAYS auto-recover the self-consistent convention and collapse the four
   raw candidates to the two-member mirror pair.
2. We DO NOT auto-guess the winner (no fabricated certainty). The tie between
   the two members is information-theoretic.
3. We default to the common MEMS/breakout convention (low-side, n-type) and
   expose a SINGLE binary that the user/UI can flip against something they
   understand: for a KNOWN reducing smell, does the recorded response RISE or
   FALL? (n-type low-side reducing -> resistance falls on the ADC). Not a
   divider-topology form.

This is the minimal, reliable interaction: one intuitive toggle instead of a
confusing hardware questionnaire, and never a silent wrong guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

MOX_LO_OHM = 1e3
MOX_HI_OHM = 1e7


@dataclass
class Convention:
    sensor_on_low_side: bool
    doping: str
    label: str
    sign: str = "unknown"
    validity: float = 0.0
    plausible: float = 0.0
    composite: float = 0.0


def corrected_resistance(adc: np.ndarray, vcc: float, rl: float,
                         sensor_on_low_side: bool) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if sensor_on_low_side:
            return rl * adc / (vcc - adc)
        return rl * (vcc - adc) / adc


def _channel_direction(rs: np.ndarray, rel_threshold: float = 0.01) -> str:
    s = rs[np.isfinite(rs)]
    if len(s) < 20:
        return "flat"
    r0 = float(np.median(s[: max(5, len(s) // 10)]))
    if not np.isfinite(r0) or abs(r0) < 1e-12:
        return "flat"
    lo = float(np.percentile(s, 5))
    hi = float(np.percentile(s, 95))
    if abs(hi - lo) / abs(r0) < rel_threshold:
        return "flat"
    return "down" if abs(lo - r0) >= abs(hi - r0) else "up"


def _dominant_sign(rs_list: List[np.ndarray], doping: str) -> str:
    from collections import Counter
    labels = []
    for rs in rs_list:
        d = _channel_direction(rs)  # down/up/flat on the PHYSICAL resistance
        if d == "flat":
            continue
        if doping == "n":
            # n-type: reducing gas -> resistance DOWN ; oxidizing -> resistance UP
            labels.append("reducing" if d == "down" else "oxidizing")
        else:
            # p-type inverts: reducing gas -> resistance UP
            labels.append("reducing" if d == "up" else "oxidizing")
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def detect(
    adc: np.ndarray,
    *,
    vcc: float = 3.3,
    rl: float = 10_000.0,
    doping_prior: str = "n",
    default_low_side: bool = True,
    user_sees_rise_for_reducing: Optional[bool] = None,
    mox_lo: float = MOX_LO_OHM,
    mox_hi: float = MOX_HI_OHM,
    rel_threshold: float = 0.01,
) -> dict:
    """Return the response convention.

    Parameters
    ----------
    user_sees_rise_for_reducing : bool or None
        The single human-grounded binary. True  -> the raw ADC trace RISES for
        a known reducing gas (=> high-side, n-type). False -> it FALLS
        (=> low-side, n-type). None -> not yet provided; we return the default
        low-side/n-type convention with `reliable=False` and the `mirror_pair`
        so the UI can prompt once.
    """
    adc = np.asarray(adc, dtype=float)
    if adc.ndim == 1:
        adc = adc.reshape(-1, 1)
    nch = adc.shape[1]
    if nch == 0:
        raise ValueError("empty ADC array")

    combos = [
        Convention(True, "n", "low-side/n-type"),
        Convention(False, "n", "high-side/n-type"),
        Convention(True, "p", "low-side/p-type"),
        Convention(False, "p", "high-side/p-type"),
    ]

    for conf in combos:
        rs_list, valid, intra = [], 0, 0
        for c in range(nch):
            raw = adc[:, c]
            if np.all(np.isnan(raw)):
                continue
            rs = corrected_resistance(raw, vcc, rl, conf.sensor_on_low_side)
            rs_list.append(rs)
            fin = np.isfinite(rs) & (rs > 0)
            valid += int(np.sum(fin))
            intra += int(np.sum(fin & (rs >= mox_lo) & (rs <= mox_hi)))
        conf.validity = valid / (nch * adc.shape[0]) if nch else 0.0
        conf.plausible = intra / valid if valid else 0.0
        conf.sign = _dominant_sign(rs_list, conf.doping)
        prior = 1.0 if conf.doping == doping_prior else 0.5
        conf.composite = conf.validity + 1.5 * conf.plausible * prior

    ordered = sorted(combos, key=lambda c: c.composite, reverse=True)
    n_pair = sorted([c for c in combos if c.doping == doping_prior],
                    key=lambda c: c.composite, reverse=True)

    # Choose within the mirror pair using the user binary, else default.
    if user_sees_rise_for_reducing is not None:
        # n-type: rise on ADC => high-side ; fall on ADC => low-side
        chosen_low_side = not bool(user_sees_rise_for_reducing)
    else:
        chosen_low_side = bool(default_low_side)
    chosen = next(c for c in n_pair if c.sensor_on_low_side == chosen_low_side)

    return {
        "best": {
            "label": chosen.label,
            "sensor_on_low_side": chosen.sensor_on_low_side,
            "doping": chosen.doping,
            "sign": chosen.sign,
            "validity": round(chosen.validity, 3),
            "plausible": round(chosen.plausible, 3),
        },
        "mirror_pair": [n_pair[0].label, n_pair[1].label],
        "candidates": [
            {"label": c.label, "sign": c.sign,
             "validity": round(c.validity, 3),
             "plausible": round(c.plausible, 3)}
            for c in ordered
        ],
        "reliable": bool(user_sees_rise_for_reducing is not None),
        "needs_binary": bool(user_sees_rise_for_reducing is None),
        "user_question": (
            "Pick the one option you observe for a KNOWN reducing gas (e.g. "
            "ethanol, as on the reference rig): does the raw ADC response RISE "
            "or FALL?  (rise => high-side/n-type wiring; fall => low-side/n-type)"
        ),
        "note": (
            "For a like-doped array the two members of the mirror pair are "
            "information-theoretically tied by the ADC trace; no shape-based "
            "criterion can separate them, and the absolute-magnitude lever is "
            "unreliable near mid-rail. So the choice is surfaced as ONE "
            "observation-grounded binary, defaulting to the common low-side/"
            "n-type MEMS convention, never a silent guess."
        ),
    }


if __name__ == "__main__":
    def synth(low_side, base=80_000.0, amp=50_000.0, nchan=6,
              RL=10_000.0, Vcc=3.3):
        t = np.linspace(0, 30, 600)
        Rs = np.full_like(t, base)
        ex = (t >= 5) & (t <= 13)
        Rs[ex] = base - amp
        rec = t > 13
        Rs[rec] = (base - amp) + amp * (1 - np.exp(-(t[rec] - 13) / 2.0))
        v = (Vcc * Rs / (RL + Rs)) if low_side else (Vcc * RL / (RL + Rs))
        return np.stack([v * (1 + 0.01 * i) for i in range(nchan)], axis=1)

    for ls in (True, False):
        arr = synth(ls)
        r0 = detect(arr, vcc=3.3, rl=10_000.0)
        print(f"true_low_side={ls} | no-binary -> best={r0['best']['label']:<16} "
              f"needs_binary={r0['needs_binary']}")
        # user says 'for a known reducing smell the ADC went FALL' (low-side)
        r1 = detect(arr, vcc=3.3, rl=10_000.0, user_sees_rise_for_reducing=False)
        print(f"true_low_side={ls} | user sees FALL -> best={r1['best']['label']:<16} "
              f"sign={r1['best']['sign']} reliable={r1['reliable']}")
        print()
