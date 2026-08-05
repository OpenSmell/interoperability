# Experiment 4 — CORAL distribution alignment on UCI drift

**Status:** COMPLETE.
**Date:** 2026-08-05
**Artifacts:** `experiment_4_coral.py`, `experiment_4_coral_result.txt`,
`experiment_4_coral_metrics.json`.

## Question (Phase 4, Path A)

Experiments 1–3 established that *per-feature scalar* calibration — single-point
M and two-point power — cannot recover cross-batch transfer on UCI: M hurts the
magnitude model, two-point power reduces but does not remove the penalty, and
the full model is +0.0 pp invariant to both. The remaining hypothesis for
**domain adaptation via feature alignment** is that real drift is not a set of
per-channel gains but a *covariance* distortion — cross-channel correlation
drift that no per-feature scalar can express. Test that with **CORAL** (Sun,
Feng & Saenko, ICML 2016): a closed-form linear map that whitens the target
device's distribution by its own covariance and recolors it to the source
device's covariance, plus a mean shift.

Question: does aligning B's feature distribution to A's — in the magnitude
subspace and in full 128-dim space — recover transfer accuracy beyond RAW /
M / two-point?

## Design

- **Devices:** same splits as Exp 3 for direct comparability — primary
  A = batches 1–3 vs B = batches 8–10 (max drift); sensitivity A = batches 1–5
  vs B = batches 6–10.
- **Alignment:** `X_B' = (X_B − μ_B) @ W + μ_A` with `W = C_B^(−1/2) C_A^(1/2)`
  (eigendecomposition, diagonal-ridge regularization `reg = 1e-3` scaled by
  `tr(C)/d` per matrix).
- **Conditions (LOO discipline, matching Exp 3):**
  - `raw` — no alignment.
  - `coral_ref` — CORAL fit on reference A/B samples *only*: the held-out gas
    never contributes to the alignment. Fair analogue of Exp 3's M / two-point.
  - `coral_all` — CORAL fit on all of B (standard transductive unsupervised DA;
    the target distribution is used, target labels are not). Optimistic bound.
- **Classifiers (RF, trained on A):** magnitude-only (32-dim) isolates the
  alignment subspace; full 128-dim includes the unaligned kinetic/derivative
  features. Chance = 16.7%.

## Headline results

Within-device ceiling stays 100% (Exp 3). CORAL is applied out-of-loop, never
retrains the classifier.

### Primary split (A = 1–3, B = 8–10)

| Model | RAW | coral_ref (LOO-fair) | coral_all (transductive) |
|---|---|---|---|
| magnitude-only (32d) | 37.8% | 31.9% (−5.9 pp) | **43.6%** (+5.8 pp) |
| full 128-dim | 38.8% | 41.6% (+2.8 pp) | **46.6%** (+7.8 pp) |

Pooled (optimistic): mag RAW 37.8% → coral_all 44.2% (+6.4 pp); full RAW 38.9%
→ coral_all 46.4% (+7.5 pp).

### Sensitivity split (A = 1–5, B = 6–10)

| Model | RAW | coral_ref (LOO-fair) | coral_all (transductive) |
|---|---|---|---|
| magnitude-only (32d) | 42.7% | 38.7% (−4.0 pp) | **49.4%** (+6.8 pp) |
| full 128-dim | 50.0% | 52.5% (+2.5 pp) | **55.3%** (+5.3 pp) |

Pooled (optimistic): mag RAW 44.5% → coral_all 52.0% (+7.5 pp); full RAW 51.3%
→ coral_all 57.0% (+5.7 pp).

## Mechanism

1. **CORAL-all is the first alignment that beats RAW on the full model** (+7.8 /
   +5.3 pp LOO), which was +0.0 pp invariant to M and two-point power. A full
   linear map corrects cross-channel covariance drift that per-feature scalars
   structurally cannot — confirming the hypothesis that part of real drift is
   *correlation* distortion, not just per-channel gain/exponent.
2. **But the LOO-fair reference fit (coral_ref) is mixed: it helps the full
   model (+2.8 / +2.5 pp) and hurts the magnitude model (−5.9 / −4.0 pp).**
   The gap between coral_ref and coral_all is the operative result: the
   alignment only pays off when the *held-out gas's own target samples* are in
   the covariance fit. Reference-only CORAL extrapolates badly to a new gas's
   amplitude regime — the same reference-manifold overfitting signature seen in
   Exp 3's two-point power (e.g., full coral_ref pushes held-out acetone to
   ~97% / ~90% while other gases collapse to ~0–13%).
3. **Magnitude vs full asymmetry:** coral_all helps mag (+5.8 / +6.8 pp) almost
   as much as full (+7.8 / +5.3 pp), so the alignment is doing real work in the
   amplitude subspace; the full model's unaligned kinetic features dilute but
   do not cancel it. The two are now roughly equal at ~44–57%, versus Exp 3
   where mag was capped ~20 pp below full. Aligning the covariance compresses
   the magnitude-only gap.
4. **Consistent cross-split facts:** ceiling 100%; toluene ≈ 0% in every
   condition (class-balance artifact: ~79 early-batch samples vs 600 in batch
   10, not an alignment effect); coral_ref per-gas variance is high (0–97%),
   coral_all per-gas variance is compressed (5–92%).

## Interpretation (honest)

- **CORAL is the best unsupervised alignment tested, and it is not a transfer
  solution.** The best LOO-fair number is full coral_ref at +2.8 pp; everything
  larger relies on `coral_all`, which requires the target device's samples for
  the test gas itself. But if target-gas target-device samples are available,
  that is **calibration data, not zero-shot** — and the user's impossibility
  result (§8.3, §8.7) says cross-device transfer without such data is
  mathematically ruled out by the unmixed (a, b) sensor constants.
- **The coral_ref→coral_all gap is the paper's number.** It quantifies exactly
  what reference-only alignment cannot do: recover a held-out gas whose
  amplitude/correlation manifold is absent from the reference fit. The gap is
  ~7–9 pp on mag, ~5 pp on full — large relative to the ~5–8 pp total gain.
- **Correlation drift is real and was the missing mechanism:** scalars (M,
  power) were +0.0 on the full model while CORAL is +2.8–7.8. A full-rank
  alignment is the correct family for cross-sensor *covariance* drift; the
  residual failure is the reference-generalization problem, not the map family.
- **Conclusion for the paper (Phase 4):** of the three feature-alignment
  families tested — per-feature scalar (Exp 1–3), full covariance map (Exp 4),
  and (up next) class-level/ontology transfer (Exp 6) — the covariance map is
  the most powerful *unsupervised* family but still cannot beat the
  within-device ceiling without target-gas reference data. The honest position:
  alignment methods are reference-point strategies in disguise (they need
  target samples), confirming rather than refuting the impossibility theorem,
  while showing that *if* reference samples exist, a full-rank map recovers
  more than scalars.

## Limitations

- `coral_all` is transductive (uses all B target samples); it is reported as an
  optimistic bound and must not be read as zero-shot. Only `coral_ref` is
  LOO-fair.
- CORAL's linear map cannot fix nonlinear drift; a nonlinear (e.g., kernel-CORAL
  or adversarial) map is a possible follow-up but shares the same
  reference-generalization problem demonstrated here.
- Toluene imbalance and quantile-pairing-free design (CORAL needs no
  concentration labels, which removes one Exp 3 limitation) still inherit the
  UCI `.dat` caveats: 128 precomputed features, no measured concentrations.
- Single regularization `reg = 1e-3` (no grid search); RF fixed as Exp 3;
  alignment applied out-of-loop.
