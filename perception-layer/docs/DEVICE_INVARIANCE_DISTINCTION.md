# Device-Invariance of Response-Type vs. Cross-Device Feature Transfer

Status: **Scope clarification.** Prevents a real self-contradiction between the
monitoring paper (`experiments/paper.md`, Theorem 2) and the phenotype ontology.

---

## The apparent contradiction

- The monitoring paper (Theorem 2: "On the Impossibility of Zero-Shot
  Cross-Device Transfer for MOX E-Noses") proves that **exact feature vectors
  cannot transfer** across devices: Rs/R0 normalization cancels the circuit
  parameters (Vcc, R_load) but not the sensor constants (a, b in the power-law
  conductance model). Two identical sensors on different circuits give different
  numeric feature vectors for the same gas.
- The phenotype ontology claims we observed **device-invariant** clusters on the
  UCI Gas Drift set (same 2-cluster structure on batch 1 and batches 2-10).

These do not contradict each other *if* we are precise about what "device
invariance" refers to. The two claims operate at different granularities.

## The distinction

| Claim | Granularity | Status |
|-------|-------------|--------|
| Exact sensor feature vectors transfer across devices | Fine (per-sensor, exact values) | **Impossible** (Theorem 2: sensor constants a,b survive Rs/R0) |
| Coarse *response-type structure* (which cluster a sample belongs to) is stable across batches | Coarse (array-level cluster assignment) | **Measured** on Vergara (2 clusters on batch-1 and batches-2-10) |

The phenotype does **not** claim the first. It claims only the second: that a
sample's *cluster identity* (one of the coarse electron-transfer response types)
survives device change, even though its exact feature vector does not.

## Why the coarse structure can survive while the fine vectors cannot

Within a single array family, the relative *ordering* of sensor selectivities
across a gas is dominated by the shared redox mechanism (which species donate
more/less electrons, driving a characteristic cross-sensor pattern). The
absolute magnitudes shift with device (a,b), but the *direction* — the A3
selectivity fingerprint — is comparatively stable, which is exactly what we
measured (C-b: clusters separate by direction, not magnitude;
`falsify_four_category.json`).

This is why we deliberately call the validated result **response-type structure**
and not "device-independent classification." It is a *weaker and safer* claim,
and it keeps the two halves of the project consistent.

## Open sub-question (see `docs/OPEN_QUESTIONS.md`, Q5)

Whether this coarse-structure invariance extends beyond batches of one array
family to genuinely different arrays (different models / manufacturers / MEMS vs
thick-film) is **untested** and listed as an open question. Until that data
exists, "device-invariant response-type" must be read as
"device-invariant *within the measured array family*", with an explicit caveat
for broader device diversity.
