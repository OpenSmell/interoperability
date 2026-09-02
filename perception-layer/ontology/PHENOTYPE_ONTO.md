# The Measured-Phenotype Ontology (Physics-Axes)

> The hard-to-vary classification of what a MOX array actually measures,
> designed so that measured signals map onto the human-familiar classes users
> already trust (`opensmell/mox/smellability/ontology.py: PERCEPTS`).

Status: milestone-1 design spec (reviewable).

---

## 1. Why this exists and what it is NOT

The existing SmellAbility engine (`opensmell/mox/smellability/`) is a
**genotype → predicted-phenotype** map: it takes a molecule's chemistry
(SMILES / functional groups) and *predicts* the percepts and capability
boundaries. Its `ontology.py: MOX_BOUNDARIES` are excellent but **theoretical** —
they are asserted from chemistry, never measured.

This spec is the **measured phenotype** layer: the map in the opposite
direction, from an actual sensor recording to a human-familiar verdict. It
exists to validate/correct the theoretical predictions, and — crucially — to
work where SmellAbility cannot: **unknown mixtures, no SMILES, no calibration**,
any device.

It is NOT a new chemical taxonomy. It does NOT claim to identify exact molecules.
It is NOT a calibrated concentration measurement.

---

## 2. The single fact everything rests on (hard-to-vary anchor)

A MOX sensor measures **a change in the electrical conductance of a metal-oxide
film**, produced **only** through surface **redox reactions** with ionosorbed
oxygen. This is the irreducible physics: the sensor reads *electron transfer*,
nothing else.

Consequences (these are the hard-to-vary axes — you cannot change them and still
be describing a MOX sensor):

| Axis | What the sensor physics encodes | Hard-to-vary reason |
|---|---|---|
| **A1. Redox valence** | Reducing vs oxidizing character (n-type: reducing gas lowers resistance, oxidizing raises it) | The sign of the resistance change IS the direction of electron transfer. A MOX array cannot answer any question this sign does not encode. |
| **A2. Kinetic binding** | Rise and recovery *shape*: time constants, peak-slope ratios, phase-space trajectory | Reaction *rate* is set by adsorption enthalpy/affinity + diffusion, i.e. the target's physical chemistry — independent of the specific rig once normalized to the resistance-invariant shapes. |
| **A3. Cross-sensor selectivity** | The *relative* response pattern across chemically-different MOX materials in the array | Material-specific catalytic selectivity is a genuine second channel of information beyond any single sensor. |

Every downstream "class" a user cares about must be a **monotone function** of
these axes — i.e. measurable, not guessed.

---

## 2b. Topology auto-detection (no hardware questionnaire)

To know A1's sign we must convert raw ADC → physical resistance, which needs the
divider topology (sensor low-side vs high-side) and doping type (n vs p). Users
must NOT be asked for either — it's a dealbreaker for a desktop tool.

**The hard fact (verified numerically in `detect.py`):** the two topology maps
are algebraic inverses (`R_low · R_high = RL²`), so for a like-doped array the
raw ADC trace **cannot separate** a topology↔doping "mirror pair":

```
(low-side,  n-type)  ≡  (high-side, p-type)     # identical ADC pattern
(high-side, n-type)  ≡  (low-side,  p-type)
```

Every shape-based criterion is exactly tied between the pair's members:
validity (both finite/positive), recovery fidelity (a monotone 1-1 map preserves
"returns to baseline"), cross-channel agreement (same map on every channel), and
log-span (numerically identical up to the RL² shift). An absolute-magnitude
lever (does recovered R land in a plausible 1k–10M MOX window?) was tested and
is **fragile, even confidently wrong near mid-rail (base ≈ RL)** — so it must
never be used to fabricate auto-certainty.

**Product decision (what actually ships):**
1. We *always* auto-recover the self-consistent convention and collapse the four
   raw candidates to the two-member mirror pair.
2. We do **not** silently guess the winner — the tie is information-theoretic.
3. We default to the common MEMS/breakout convention (`low-side`, `n-type`) and
   expose **exactly one observation-grounded binary**, framed against something
   the user already understands, never a wiring form:

   > "For a known reducing smell (e.g. ethanol), does the raw ADC response
   > RISE or FALL?"   (rise ⇒ high-side/n-type ; fall ⇒ low-side/n-type)

This satisfies test §7.1 (sign invariance without user hardware knowledge) and
matches the "minimal seamless UI" requirement — one toggle, not a questionnaire,
and never a silent wrong guess.

---

## 3. The three axes, defined precisely

### Axis A1 — Redox valence `∈ {reducing, oxidizing, inert, mixed}`

Defined from the **topology-corrected** resistance direction (see
`normalize.py`): after converting ADC → resistance using the divider config, a
reducing analyte makes an n-type channel's resistance fall; an oxidizing one
makes it rise.

- `reducing:` resistance falls on exposure → electron donation (most VOCs,
  alcohols, aldehydes, ketones, H2, CO, NH3, sulfides, terpenes, alkanes).
- `oxidizing:` resistance rises on exposure → electron withdrawal
  (NO2, O3, Cl2, halogens).
- `inert:` no coherent change → non-redox-active (N2, CO2, noble gas) or below
  floor.
- `mixed:` channels disagree (array contains both n- and p-type materials, or a
  redox-active mixture whose net effect is ambiguous) → low confidence.

This is the **root** split, mirroring `MOX_BOUNDARIES.redox` capability.

### Axis A2 — Kinetic binding `∈ {fast, slow} × {rise, recovery}`

Defined from the shape of the corrected resistance curve, using
phase-space / dynamic features that are resistance-normalized and therefore
**device-agnostic**:

- **rise rate** — how quickly the signal reaches peak (large `max(dS/dt)/S0`
  → fast surface reaction; small → slow).
- **recovery rate** — how quickly it returns to baseline (requires the
  `baseline → exposure → recovery` protocol to exist; on ramp-only data this
  axis is `unknown`, never forced).
- shape complements: peak-location `a/b`, desorption slope, phase-space area.

Physical meaning: a *fast, strongly-adsorbing small reducing molecule*
(e.g. ethanol) differs from a *large, weakly adsorbing* one in these shapes.
This is the partial-resolution axis — it separates **size/affinity** even where
bare redox cannot.

### Axis A3 — Cross-sensor selectivity profile (the array fingerprint)

Defined as the **relative** response across channels, each normalized to its own
baseline (`Rs/R0` per channel), producing a unit-selectivity vector. This is
what lets the array discriminate at all beyond A1+A2. It is **temperature and
humidity dependent** unless the environmental channels and per-sensor profiles
are conditioned on — so A3 always carries a confidence that degrades when
conditioning data (temp/humidity, sensor profiles) is absent.

---

## 4. Evidence-chain verdict format

Every per-recording verdict is an **evidence chain**, not a bare label:

```
verdict:
  axes:
    A1_redox_valence: reducing        # from corrected resistance sign
    A2_kinetic:       fast_rise        # fast_rise / slow_rise / unknown
                      recovery: unknown # unknown on ramp-only data
    A3_selectivity:   [0.2, 0.9, 0.4, ...]   # unit vector over channels
  map_to_human:       "alcoholic-like"     # PERCEPT mapping (see §6)
  confidence:
    A1: high           # sign is robust where SNR above floor
    A2: medium         # needs clean rise; recovery needs protocol data
    A3: low-medium     # degrades without temp/humidity + profile
  evidence:
    - key_feature: ch2_da_relative_amplitude
      value: 0.83
      physical: "resistance fell 83% of baseline -> electron donation"
    - key_feature: ch2_da_rise_time
      value: 1.1s
      physical: "fast surface reaction -> small polar molecule"
  source:
    dataset: UCI-HomeActivity
    session: 12
    device_profile: default     # or matched profile id
  boundaries:                 # from MOX_BOUNDARIES that constrain this verdict
    - cannot: absolute_ppm
    - cannot: exact_molecule
    - cannot: mixture_decomposition
```

Confidence is a **conjunction of feature-level confidences**, never summed
blindly. A single weak axis (e.g. A1 `inert`, or A3 without conditioning)
downgrades the whole verdict.

---

## 5. Capability boundaries (measured side)

Rendered as machine-readable `{can, cannot}`, extending `MOX_BOUNDARIES` with
the *measured* logic that produces them:

| Can (measured) | Cannot (measured) |
|---|---|
| Redox valence (reducing/oxidizing) | Exact molecule / isomer / chiral identity |
| Fast vs slow kinetics (size/affinity ordering) | Absolute ppm (no calibration) |
| Relative cross-sensor signature (array fingerprint) | Decomposing arbitrary mixtures |
| "~1 ppm floor" detection-floor gate | Gases invisible to redox (CO2, N2, noble) |
| Works on unknown mixtures, no SMILES | Reliable family on ramp-only data (A2.recovery) |

The `cannot` list is what we *promise* not to claim — the honesty contract a
user relies on. It mirrors the calibrated unease the existing engine already
documents, but now grounded in measured evidence rather than theory alone.

---

## 6. Mapping to human-familiar classes (the utility)

The whole point (per §1) is a useful, interpretable verdict. The existing
`PERCEPTS` (`ontology.py:34-133`) are the target vocabulary users know:
`fruity-ester`, `citrus-terpenic`, `green-leafy`, `floral`, `minty`,
`spicy-balsamic`, `roasted-caramel`, `smoky-phenolic`, `sulfurous`,
`ammoniacal`, `solvent-industrial`, `alcoholic`, `sour-acidic`,
`neutral-gas`.

They map to the measured axes **probabilistically** (a percept is a *region* in
the 3-axis space, learned from labeled recordings, not asserted from chemistry):

| Human class | A1 | A2 (typical) | A3 selectivity notes | Typical origin chemistry |
|---|---|---|---|---|
| `alcoholic` | reducing | fast rise | strong on alcohol-selective channels | small-chain alcohols |
| `sulfurous` | reducing (strong) | — | strongest MOX reducers; high amplitude | thiols, H2S, disulfides |
| `ammoniacal` | reducing (basic) | fast | distinct NH3 channel selectivity | NH3, amines |
| `solvent-industrial` | reducing | varies | BTX/alkane selective channels | aromatics, alkanes |
| `fruity-ester` | reducing | fast | ester/ketone channels | esters, small ketones |
| `sour-acidic` | reducing (weak) | slow | low-response, slow recovery | carboxylic acids |

The mapping is learned from labeled data (e.g. Vergara's 6 pure gases, Hu's
banana/wine) — **never hard-coded from chemistry** — and each assignment carries
the evidence-chain confidence of §4. This is how a user hears "this smells
alcoholic-like (82%, fast-rise reducing, alcohol-selective channel)" and can act
on it — even for a mixture, an unknown, or a device we've never calibrated.

---

## 7. Hard-to-vary tests (falsifiability)

A claim in this ontology must survive:

1. **Sign invariance.** If you rewire the divider (topology change) and we forget
   to correct, the redox verdict flips → the feature set fails. We therefore
   *require* topology handling (§2b, `detect.py` + `normalize.py`, Milestone T1)
   and test it — but never fabricate certainty when the trace cannot separate
   the mirror pair; we surface the single observation-grounded binary instead.
2. **Cross-device invariance of A2.** The same pure gas on two devices must give
   the same kinetic *class* (fast/slow) once normalized, even if absolute
   magnitudes differ.
3. **Mixture honesty.** A 50/50 blend must produce `mixed`/low-confidence or a
   recognizable dominant percept — never a fabricated crisp identity.
4. **No-chemistry usability.** The verdict must be computable from a recording
   alone with *zero* SMILES/functional-group input (else it's not "measured").

---

## 8. What gets built next (ordered)

- **T1** `normalize.py` + `detect.py` — topology-aware ADC→resistance; auto-detect
  the self-consistent mirror pair and surface the single observation binary
  (hard-to-vary sign, no hardware questionnaire).
- **T2** `phenotype.py` — compute A1/A2/A3 from a corrected recording; evidence
  chain output.
- **T3** `percept_map.py` — learned mapping measured axes → PERCEPTS; confidence.
- **T4** piecewise validation harness across the three valid datasets
  (Vergara = chemistry/redox/batch; Hu = raw dynamics; SmellNet = no-recovery
  ramp honesty).
- **T5** report + a public `ontology/` JSON schema so contributions can
  validate against known-good vs known-bad recordings.

Each is independently useful and can be reviewed before the next starts.
