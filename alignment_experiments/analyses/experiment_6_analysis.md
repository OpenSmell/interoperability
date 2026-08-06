# Experiment 6 — Taxonomic / class-level transfer across devices

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_6_taxonomy.py`, `experiment_6_taxonomy_result.txt`,
`experiment_6_taxonomy_metrics.json`.

## Question (Phase 4, Path B)

If fine gas/substance identity does not survive a device boundary (Exps 1–4),
maybe a **coarser class** does: a device that cannot resolve *which* gas should
still resolve *what kind*. Test this "ontology / taxonomic transfer" in two
venues with different class systems:

- **Part A — UCI real drift:** coarsen the 6 gas identities to chemical
  functional classes (alcohol, carbonyl, hydrocarbon, amine) and ask whether
  class-level accuracy transfers A→B better than gas-level accuracy, and
  whether drift mispredictions are preferentially *within-class* (which would
  make coarsening a genuine transfer mechanism).
- **Part B — SmellNet/OSMO:** do 30-dim paradigm features (device-agnostic by
  R0 normalization) support OSMO grand-family prediction, and does a family
  classifier trained on SmellNet transfer to the user rig?

## Part A — UCI chemical-class coarsening

Design: same splits as Exp 3/4 (A = early batches, B = late batches = drift).
Fine = 6 gases (chance 16.7%), coarse = 4 functional classes (chance 25%).
Within-device ceiling 100% for both granularities. Transfer efficiency
`TE = (cross − chance)/(within − chance)` normalizes away the different chance
levels, so it is the honest comparison of fine vs coarse.

| Split | Model | fine cross | coarse cross | coarse TE − fine TE | within-class error share (obs / chance) |
|---|---|---|---|---|---|
| 1–3→8–10 | mag | 36.5% | 43.9% | 25.2 − 23.9 = **+1.3 pp** | 2.2% / 15.2% |
| 1–3→8–10 | full | 38.3% | 45.9% | 27.8 − 25.9 = **+1.9 pp** | 1.1% / 15.4% |
| 1–5→6–10 | mag | 44.0% | 53.5% | 38.0 − 32.8 = **+5.2 pp** | 5.3% / 15.0% |
| 1–5→6–10 | full | 49.0% | 58.7% | 45.0 − 38.8 = **+6.2 pp** | 3.3% / 16.6% |

Collapse-of-fine (coarsening the fine classifier's predictions) is always below
the direct coarse classifier (37.9 vs 43.9; 38.9 vs 45.9; 46.9 vs 53.5; 50.7 vs
58.7) — training on class labels beats post-hoc collapsing, as expected.

Per-class coarse cross accuracy (sensitivity split): **alcohol 75%, amine 64%,
carbonyl 48%, hydrocarbon 36%**. The two highest-accuracy classes are the two
*singletons* (alcohol = ethanol, amine = ammonia). The two classes that
actually merge multiple gases (carbonyl, hydrocarbon) sit at 36–48% — near or
barely above their 25% chance.

## Mechanism (Part A)

1. **Drift errors are CROSS-class, not within-class.** The within-class share
   of fine mispredictions is 1–5%, against 15–17% expected if errors were
   random, and ~20% if errors were preferentially same-class for the merged
   groups. Chemical functional structure does **not** organize drift
   confusion — the premise of the taxonomy hypothesis (errors concentrate
   inside a class, so a coarser label rescues them) is falsified on UCI.
2. **Most of the coarse gain is mechanical.** Coarse accuracy is +7–10 pp
   above fine on the raw scale, but TE (chance-adjusted) improves only
   +1.3 to +6.2 pp. A coarser label space has a higher chance level and wider
   bins; that alone buys most of the apparent gain.
3. **There is a small genuine coarse advantage in the sensitivity split**
   (+5.2/+6.2 pp TE). When A is larger (1–5), the coarse model transfers
   ~45–50% of the learnable signal vs ~39% for fine — a real but modest
   effect that does not approach the ceiling.

## Part B — SmellNet paradigms → OSMO grand families

Design: 300 SmellNet recordings (50 substances × 6), 30-dim paradigm vectors
computed per recording (`compute_window_paradigms`, r0=5). RandomForest
leave-one-substance-out for grand-family prediction (chance 12.5%, 8 families).

- **LSO family accuracy 40.7% raw / 21.1% balanced** vs chance 12.5% — above
  chance, but the confusion matrix is Woody-dominated (Woody 71/119 correct;
  the class is 19/50 = 38% of substances, and the RF predicts Woody
  everywhere). Fruity/Green/Herbal/Woody are partially resolved; Citrus,
  Floral, Mineral, Soulful (n = 1–2 each) essentially never.
- **LSO substance accuracy 0.0%** vs chance 2.0% — paradigm features carry
  **no substance identity at all** within SmellNet. Family ≫ substance: the
  device-agnostic paradigm set is a coarse-category representation by
  construction; it trades fine discrimination for exactly the structure a
  taxonomy needs.
- **Cross-device (SmellNet-trained → user rig): every user recording maps to
  Woody.** Cinnamon and ginger are true Woody (correct by dominance); garlic
  (Mineral), banana (Fruity), lemon (Citrus) all MISMATCH; unlabeled
  room_air/onion/mosquito_coil also → Woody. Decisive negative: the family
  structure that is learnable within SmellNet does not survive the device
  boundary on the real rig.

## Interpretation (honest)

- **Taxonomic transfer is not an interoperability layer — on two legs.** On UCI
  the class structure does not organize drift confusion (within-class share
  *below* chance), and on SmellNet→user the within-device family structure
  collapses to the dominant class across the device boundary. Both are clean
  negatives, and each is falsifiable and published honestly.
- **The one positive is within-device and it is exactly the expected one:** the
  30-dim paradigm set separates OSMO families (40.7% vs 12.5%) while carrying
  zero substance identity. This confirms the paradigm extractor's design claim
  — device-agnostic *category* features — and explains why the paradigm-based
  external baseline (33.3%) beat the canonical full model (8.3%) in Exp 1:
  coarse structure is all the paradigm set is built to carry.
- **Why the user-rig transfer fails:** (a) the user rig has 3 live channels
  (CO/NO2/C2H5OH are exactly 0.0), so half the paradigm dimensions are
  degenerate for every user vector; (b) sr ≈ 2.1 Hz vs 10 Hz (kinetic
  paradigm terms not comparable); (c) Woody dominance + a shifted manifold
  sends everything to the majority class. Same structural causes as Exp 1's
  collapse-to-banana.
- **Conclusion for the paper (Phase 4):** across the four feature-alignment
  families — scalar calibration (Exps 1–3), covariance map (Exp 4), anchor/
  supervised alignment (Exp 5), and ontology coarsening (Exp 6) — none
  recovers transfer accuracy on the real device pair or on real UCI drift,
  and the taxonomy-specific mechanism (within-class drift confusion) is
  directly falsified. The measured 81.78% session-invariance within one rig
  remains the only real interoperability result.

## Limitations

- Part A coarse classes are a *chemical* grouping chosen post-hoc; a
  different grouping (e.g., perceptual OSMO families for UCI gases) could in
  principle coarsen differently. The within-class-error statistic is the
  mechanism test and is grouping-invariant in spirit (it measures whether
  *any* coarse grouping captures drift confusion).
- Part B substance LSO at 0.0% (n=6 recordings per substance, 30-dim features,
  RF) is a strong but small-sample result; a different estimator might find
  weak substance signal.
- User-rig leg is confounded by the 3-live-channel/6-channel mismatch and
  sr mismatch — the collapse to Woody cannot be attributed to drift alone.
- Exp 6 fine cross numbers are pooled-over-B (overall accuracy), which differ
  by ±1–3 pp from Exp 3's LOO-mean RAW (36.5/38.3 vs 37.8/38.8 primary) — the
  two quantities are the same quantity evaluated differently; comparisons
  within Exp 6 are self-consistent.
