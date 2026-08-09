# Experiment 5 — Anchor-based supervised alignment on Praise James' rig

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_5_anchors.py`, `experiment_5_anchors_result.txt`,
`experiment_5_anchors_metrics.json`.

## Question (Phase 4, Path B)

Exps 1–4 assumed alignment without knowing which substances a reference pair
measured. Here we give the alignment **real supervised anchors**: cinnamon,
garlic and banana are recorded on *both* SmellNet (Rig A) and Praise James'
rig (Rig B), so we know the anchor-pair correspondence exactly. Can a supervised
map fitted on those anchors transfer a *held-out* substance — the one case the
impossibility theorem allows if reference substances exist?

Venue: real SmellNet→Praise James' pair, magnitude subspace only (3 live channels
VOC/Alcohol/LPG × 3 magnitude features = 9 dims), reusing the Exp 1 feature
cache. Chance = 25% (4 SmellNet classes).

Map families (vs Exp 1's scalar M and Exp 4's unsupervised CORAL):
- **affine** — per-feature `y = a·x + b`. LOO: exact 2-point line (no dof
  left); pooled: 3-point least-squares line.
- **proc (Procrustes/Kabsch)** — scale + rotation + translation. LOO: rotation
  constrained only on the 1-dim anchor-segment span, the 8-dim complement
  unconstrained.

## Headline results

| Condition | cinnamon | garlic | banana | overall |
|---|---|---|---|---|
| SmellNet 5-fold CV, mag9 (within-device ceiling) | — | — | — | **74.7%** |
| RAW mag9 | 0/3 | 0/6 | 0/3 | **0.0%** |
| RAW full 91-dim | 0/3 | 0/6 | 1/3 | **8.3%** |
| ALIGNED affine, LOO (2 anchors) | 0/3 | 0/6 | 0/3 | **0.0%** |
| ALIGNED proc, LOO (2 anchors) | 0/3 | 0/6 | 0/3 | **0.0%** |
| ALIGNED affine, pooled (3 anchors, optimistic*) | 3/3 | 0/6 | 3/3 | **66.7%** |
| ALIGNED proc, pooled (3 anchors, optimistic*) | 3/3 | 0/6 | 3/3 | **66.7%** |
| full + proc(pool) on mag dims | 0/3 | 0/6 | 1/3 | **8.3%** |

\*pooled is *not* LOO-fair: the test substance's own user windows participate in
the anchor fit (the same "optimistic bound" convention as Exp 1's pooled M).

Fitted-map diagnostics (LOO, 2 anchors): Procrustes scales 0.061 / 0.116 /
1.263; affine slopes range from **−11.5 to +14.0** — the 2-point fits are
wildly unstable. Lime OOV class mix flips between garlic / cinnamon / banana
under different conditions (no stable signal).

## Mechanism

1. **The LOO-fair result is +0.0 pp for both map families.** Supervised anchors
   do not transfer a held-out substance on this real rig pair. The affine map
   through 2 anchors has zero generalization freedom and the Procrustes map is
   unconstrained in the 8-dim complement — exactly the documented
   underdetermination, now measured: the held-out substance sits off the
   anchor segment, so any map pinned to the segment leaves it where it was.
2. **The pooled-vs-LOO gap (66.7% vs 0.0%) is the paper's number.** The map
   only works when the test substance's *own* windows are in the fit. This is
   the same reference-generalization gap Exp 4 measured in CORAL (coral_ref
   vs coral_all): alignment is a reference-point strategy; with per-substance
   reference data it helps, without it it is inert. Garlic fails even pooled
   (6 windows across 2 files, high per-recording exposure variance), so the
   method is fragile to the very variance real recordings carry.
3. **Map family is not the binding constraint.** Scalar M (Exp 1), full
   covariance (Exp 4 CORAL), per-feature affine and full Procrustes all land
   at the same LOO-fair wall: reference availability + per-recording
   variance, not the functional form of the map.
4. **Full-model invariance reproduced.** Applying the mag alignment to the
   mag dims of the 91-dim vector changes full-model accuracy by +0.0 pp
   (8.3% → 8.3%) — the model still separates on device-bound absolute
   features (Exp 1 diagnosis).

## Interpretation (honest)

- **Anchor-based alignment is calibration in disguise, and it confirms the
  impossibility theorem's corollary:** if the target device's reference
  samples for a substance exist, alignment can work (cinnamon/banana pooled);
  if they do not — the zero-shot definition — alignment is +0.0 pp. The
  supervised-anchor path does **not** escape the theorem; it restates it.
- The real-rig evidence now covers every alignment family the brief proposed:
  single-point M (Exp 1), two-point power (Exp 3), CORAL (Exp 4), supervised
  affine/Procrustes anchors (Exp 5), and ontology coarsening (Exp 6). Each is
  reference-dependent or inert LOO-fair. The measured within-rig
  session-invariance (81.78%) remains the interoperability result; the
  reference-point path needs measured calibration substances on every rig.
- Small-print positives worth keeping: within-device, the magnitude-only
  subspace is learnable (74.7%), so a *future* rig that ships with
  pre-measured reference substances could in principle use anchor alignment —
  but the requirement is per-substance reference data, not zero-shot.

## Limitations

- Tiny test set: 12 shared-class windows across 4 user files; garlic n=6 from
  2 files, banana file is ADC-clipped (VOC/LPG rails at 0.0 → amplitude
  saturation artifact).
- sr mismatch (user ≈ 2.1 Hz vs SmellNet 10 Hz) invalidates kinetic features;
  only the magnitude subspace is compared.
- Pooled condition leaks the test substance by design and is labeled
  optimistic; it must not be quoted as a zero-shot number.
- 2-anchor LOO affine is a forced 2-point line (0 dof); 3 anchors still cannot
  determine a 9-dim map, so "anchor" results are upper-confidence-inflation
  for the map family, not a complete map estimate.
