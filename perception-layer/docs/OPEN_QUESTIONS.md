# Open Research Questions & Community Data Requests

Status: **Living document.** Every claim in the phenotype ontology is scoped by
*available real data*. Each question below states exactly what data would decide
it and what each possible outcome would mean — so that contributions (new
datasets, protocol recordings) can be targeted rather than speculative.

Corresponding results: `results/percept_map_A3.json` (validated 2 clusters),
`results/falsify_four_category.json` (falsification attempt),
`results/validation_piecewise.json` (per-axis validation).

---

## Why there are open questions at all

The ontology is built from measured structure: on the UCI Gas Sensor Array Drift
set (Vergara et al., 2012; 6 reducing targets, 16 sensors, 10 device batches,
13,910 samples) the array forms **2 device-invariant response-type clusters**
(Gas Drift) with individual-gas purity 0.405. Everything beyond that — a
strong/weak amplitude split, oxidizing categories, a kinetics grid — is either
**falsified** on that data (amplitude split) or **untestable** because the
requisite data does not exist. This document turns those gaps into precise,
answerable questions with a shared standard: a claim here is only as strong as
the dataset that tests it.

---

## Question 1 — Does the array truly separate 'strong' vs 'weak' reducing response?

**Status: FALSIFIED on current data (open for richer data).**

- What current data says: the reducing-strength distribution is a **continuous,
  heavily overlapping gradient**, not two clean classes. 2-group separation
  Z ≈ 2.91 (norm) / 2.92 (max-|sensor|), both below the ~3.0 crisp-dichotomy
  threshold; ~22% of samples fall in an ambiguous band; only ~15% land in the
  'weak' group. `results/falsify_four_category.json`.
- Why it could still be real: the measured gradient may be an artefact of
  **uncontrolled concentration**. Vergara mixes concentration levels within each
  gas, so amplitude variance is dominated by dose, not by a chemical
  strong/weak identity. At equal, matched concentration, a genuine bimodal
  strong/weak structure could emerge.
- Data that would decide it: recordings with **controlled, equimolar
  concentrations** of several reducing gases (e.g. CO, H₂, NH₃, ethanol), each at
  matched dose, clean baseline→exposure→recovery, across ≥2 devices.
- What each outcome means:
  - Still continuous at matched dose → the strong/weak split is **not** a real
    axis; response-type must be reported on A3 selectivity only.
  - Bimodal at matched dose → a *concentration-controlled* strong/weak axis is
    real and should be added, with a data-derived (not hand-picked) threshold.

## Question 2 — Do oxidizing targets form distinguishable categories?

**Status: UNTESTED (no oxidizing data in the corpus).**

- What current data says: nothing — NO₂, O₃, Cl₂, etc. never appear in any
  dataset we validated against (Vergara, Hu 2016, SmellNet are all
  reducing-VOC/CO).
- Why it's worth testing: n-type MOX redox theory predicts the **opposite sign**
  of resistance change for oxidizing gases, so oxidizing targets *should* be
  cleanly separable from reducing ones on A1 polarity. Whether oxidizing gases
  then sub-split (strong/weak, or by selectivity) is unknown.
- Data that would decide it: an external dataset with **≥2 oxidizing targets and
  ≥2 reducing targets on the same array**, clean protocols, across ≥2 devices.
- What each outcome means:
  - Oxidizing cluster(s) form and split cleanly from reducing → extends the
    ontology to A1 polarity + oxidizing A3 response-types (validated).
  - Oxidizing targets overlap reducing ones → the polarity claim must be
    weakened to 'reducing-only' and the array's oxidizing sensitivity documented
    as limited.

## Question 3 — Does kinetics (A2) add a genuinely orthogonal axis?

**Status: UNTESTED (no protocol dynamics data).**

- What current data says: nothing decisive. Vergara is pre-extracted
  steady-state amplitudes (no time series); SmellNet is ramp-heavy and
  single-ish device; Hu has 3.6 s snippets with drift that swamps signal. A2
  recovery is gated *per recording* in `phenotype.py` and mostly returns
  `unknown`.
- Why it's worth testing: adsorption/desorption kinetics differ by molecule
  class and are load-pressure-independent in principle, making rise/recovery a
  plausible discriminator independent of the A3 fingerprint.
- Data that would decide it: **protocol recordings** — baseline → exposure →
  recovery with the phase boundaries marked — across ≥2 devices, several gases.
  This is the single most valuable contribution (see
  `contributor_guide/CONTRIBUTING.md`, P0 protocol-data ask).
- What each outcome means:
  - Kinetics separates classes the A3 fingerprint cannot → adds an orthogonal
    axis; a response-type grid becomes defensible.
  - Kinetics is redundant with A3 → A2 stays a confidence/quality signal, not a
    classifier.

## Question 4 — Would richer chemistry reveal more than 2 clusters?

**Status: UNTESTED (limited chemical diversity).**

- What current data says: 2 clusters on 6 reducing targets (cluster purity
  vs the 6 gas labels = 0.405). The measured clusters are separated by **A3
  selectivity direction**, not magnitude (`falsify_four_category.json`, C-b).
- Why it's worth testing: the 2-cluster split may be an artefact of the specific
  chemical set (a VOC-cluster vs an ammonia/ethylene-cluster), not a fundamental
  limit. More diverse compounds (polar, nonpolar, branched, heteroatomic) could
  split these further.
- Data that would decide it: expand the UCI-style drift set (or a new set) with
  **more pure compounds spanning diverse functional chemistry**, same array,
  multiple devices.
- What each outcome means:
  - More clusters emerge, still device-invariant → the response-type ontology
    grows with data (retrained, still silhouette/k-selected, never hand-fixed).
  - Structure stays at ~2 → the array's real information limit is coarser than
    hoped; report response-type at that coarser level.

## Question 5 — What is device-invariance *really* robust to?

**Status: PARTIALLY VALIDATED (one dataset, 10 batches); OPEN beyond that.**

- What current data says: the 2-cluster structure is stable across Vergara's 10
  device batches (per-split best k = 2 on batch-1 and batches-2-10). This is
  device-invariance of **response-type structure**, on one array family.
- Important distinction (see `docs/DEVICE_INVARIANCE_DISTINCTION.md`): this
  does **not** claim cross-device transfer of exact feature vectors — which the
  monitoring paper's Theorem 2 shows is provably impossible (Rs/R0 cancels
  circuit parameters but not sensor constants). The phenotype claims only that
  the *coarse response-type structure* is robust, which is a weaker, separate
  claim.
- Data that would decide it: recordings of the **same compounds across genuinely
  different arrays** (different sensor models / manufacturers / MEMS vs
  thick-film), not just different batches of one array.
- What each outcome means:
  - Structure survives on different arrays → device-invariance of response-type
    is broadly true; strong basis for the ecosystem.
  - Structure breaks on some array → response-type must be reported with an
    explicit array-family caveat.

---

## The data-gathering ask (for the community)

The four genuinely open questions (1–4) all resolve to the same concrete
contribution from anyone with a working MOX rig:

**Record `baseline → exposure → recovery` protocol data, mark the phase
boundaries, record across ≥2 devices, and include (a) matched-concentration
reducing gases, (b) oxidizing targets if available, (c) chemically diverse pure
compounds.**

See `contributor_guide/CONTRIBUTING.md` (P0 protocol-data + diverse-chemistry +
known-dopant asks) and the `.osmell` manifest spec for the recording contract.
A dataset does not need to be large to help; it needs to be *protocol-complete*
and *chemically targeted* so that at least one question above is decisively
answered.

---

## Validation standard (applies to every future claim)

- k for clustering is chosen by the data (max silhouette), never fixed by hand.
- Purity/confidence numbers are computed from raw features, never asserted.
- A claim is only 'validated' if a dataset actually tests it; otherwise it is
  listed here as an open question with the exact data required.
- Any proposed new category must first survive its own falsification check
  against the dataset that would reveal it.
