# Falsifying the "Four Device-Invariant Categories" Claim

Status: **Experimental falsification attempt completed.** Result decisive and
honest. Written because a synthesis draft asserted the full four-category scheme
(Strongly/Weakly Reducing + Strongly/Weakly Oxidizing) as *measured and
device-invariant*. Our own data did **not** support that package; per the
project rule that labels and derived claims must come from hard-to-vary,
data-driven criteria (never hand-asserted priors dressed as measured), the
scheme was decomposed into independently falsifiable sub-claims and tested where
real data permitted.

Source: `experiments/falsify_four_category.py` ->
`results/falsify_four_category.json` (UCI Gas Drift / Vergara 2012,
13,910 samples, 16 sensors, 10 batches).

---

## The sub-claims and their fate

| ID | Sub-claim | Data required | Executable now? | Verdict |
|----|-----------|---------------|-----------------|---------|
| C-a | Reducing targets split by magnitude into a clean **strong vs weak** dichotomy (the `amplitude > 0.5` threshold) | Vergara (reducing amplitudes) | Yes | **FALSIFIED** — continuous gradient |
| C-b | The measured 2 clusters are organized by **magnitude** (strong/weak axis) | Vergara (fingerprints + norms) | Yes | **FALSIFIED** — direction/organized by A3 selectivity |
| C-c | Strongly vs Weakly **Oxidizing** categories are real/separably measured | oxidizing targets (NO₂, O₃, Cl₂) on same array | No | **Open question** (no data) |
| C-d | Kinetics (A2) is an orthogonal axis supporting a 2×2 grid | baseline→exposure→recovery protocol data | No | **Open question** (no data) |

---

## C-a: is there a clean strong/weak amplitude dichotomy? (FALSIFIED)

A crisp dichotomy requires the two candidate "strong"/"weak" groups to be well
separated (gap ≥ ~3 pooled within-group std) and mostly non-overlapping. Measured
on two independent strength scalars (per-sample L2 response norm; per-sample
max-|sensor| response):

- 2-group separation Z: **2.91** (norm), **2.92** (maxabs) — both **below 3.0**.
- Only **15.1–15.2%** of samples fall in the "weaker" group.
- **~22%** of samples lie in an ambiguous overlap band between the two group means.
- Per-gas magnitude means overlap massively (Ethanol 129k±74k; Toluene 71k±52k;
  Acetone 254k±155k): within-gas spread is large relative to the between-gas gaps.

**Conclusion:** the reducing-strength axis is a **continuous, heavily overlapping
gradient**, not two separable device-invariant classes. The proposed
`amplitude > 0.5 = Strongly Reducing` threshold is a **hand-asserted prior, not a
measured structure**. It must not be presented as validated.

## C-b: are the clusters organized by magnitude or by direction? (FALSIFIED)

Re-ran the sample-level 2-cluster KMeans on L2-normalized fingerprints (as in
`percept_map.py`), then asked what separates the two clusters:

- Between-cluster centriod cosine: **0.658**; mean within-cluster angle **0.24 rad**,
  cross-cluster angle **0.87 rad** → directional separation **0.64 rad** (strong).
- Cluster norm separation Z: **0.97** (weak).

**Conclusion:** the two clusters differ primarily by **fingerprint shape
(cross-sensor selectivity, A3)**, not by response magnitude. Labeling them
"Strongly vs Weakly Reducing" **mis-attributes** the measured separation to an
amplitude axis that is not the real organizing signal.

## C-c: oxidizing categories — OPEN (unbacked by any data)

No available dataset (Vergara, Hu 2016, SmellNet) contains oxidizing targets
(NO₂, O₃, Cl₂) with any real protocol. Oxidizing polarity is **never observed**
anywhere in our validation corpus, so the "Strongly/Weakly Oxidizing" half of the
scheme is **neither confirmed nor falsified** — it is unbacked, and must stay an
open research question.

## C-d: kinetics grid — OPEN (unbacked by any data)

No clean baseline→exposure→recovery protocol dynamics exist, so A2
rise/recovery cannot be tested. This is the same A2-blocked status already
documented; a 2×2 (strong×fast) grid is speculative until protocol data exists.

---

## Why these claims were worth testing (David Deutsch's standard)

A claim is a good question — worthy of experiment — when there is a **physical
mechanism** for why it might be true and a **decisive test** that could refute it.
Each survival-relevant axis here was worth probing because:

1. **C-a (strong/weak):** MOX response amplitude does scale with electron-donation
   strength (surface ionosorbed oxygen model), so a magnitude split *could*
   plausibly exist — and it is directly measurable, so we tested it. It failed:
   magnitude is a continuum on the measured array. Worth testing precisely because
   it was plausible and testable; now *falsified*, it must be retracted.
2. **C-b (magnitude-vs-direction):** the synthesis's own percept-map already showed
   the array separates gases by *fingerprint*, so it was a real question whether
   "strong" was a meaningful axis at all. Tested, falsified.
3. **C-c (oxidizing):** *not yet* a good experiment — the physical mechanism
   (opposite sign redox on n-type MOX) is sound theory, but we possess **no data**,
   so there is no decisive test available. It therefore stays a well-posed
   **question** (with the theory giving the reason the *future* experiment is
   worth running), strictly NOT a validated claim.
4. **C-d (kinetics):** likewise awaits protocol data before it becomes testable.

The Deutsch-standard framing is therefore: the only *validated device-invariant
structure* we have is the **2 reducing response-type clusters separated by A3
selectivity** (gas purity 0.405). The "four categories" is a **research program**,
two of whose four pivots (reducing strong/weak) are already falsified, and two
(oxidizing, kinetics) remain open until required data exists.

---

## What this means for any build

Nearest honest product framing, given current data:

- **Validated:** redox polarity (A1, reducing-only in corpus) + A3 selectivity
  fingerprint producing ~2 device-invariant response-type clusters among reducing
  targets.
- **Falsified:** the "Strongly vs Weakly Reducing" amplitude split (a continuum).
- **Open (no data):** Oxidizing classes, kinetics grid, temperature-modulation
  separation, humidity sensitivity, mixture decomposition.

A build that presents "four device-invariant categories" as measured would be
asserting two falsified pivots as fact and two untested pivots as validated.
That is exactly the over-claim this document retracts.
