# Censored QUAIDS — Cloud + MCP: Project Plan

**Goal:** Cloud infrastructure on AWS, fronted by an MCP server, that lets users run Noé's
censored AI/QUAI demand system estimator. Win condition: a *working, correct* artifact —
the Julia results must match the original R package (`censoredAIDS`) numerically.

**Method source of truth:** https://github.com/noejn2/censoredAIDS (R). We port, we do not reinvent.

---

## Architecture (strawman)

```
 user (chat / client)
        │
        ▼
   MCP server  ──submit──►  AWS job layer  ──run──►  CensoredAIDS.jl (Julia)
   (tools)     ◄─status──   (S3 + compute)  ◄────     estimation + elasticities
   ◄─results──             (run metadata)
```

Three buildable layers, bottom-up:

1. **`CensoredAIDS.jl`** — Julia port of the estimator (the core; correctness lives here).
2. **AWS job layer** — submit dataset + config → run Julia job → persist results.
3. **MCP server** — conversational front door: `submit_estimation`, `get_status`,
   `get_results`, `list_runs`, `get_elasticities`.

---

## Milestones (plan-first; each gated by your approval)

### M0 — Foundation
- `git init`, project scaffold, Julia `Project.toml`.
- Pull R golden fixtures (`qshares.rds`, `loglikes.rds`, `testing_params/*`, Mexican ENIGH data)
  and export to a language-neutral format (CSV/JSON) as the validation oracle.
- **Done when:** repo builds, fixtures loadable in Julia.

### M1 — Julia core: share equations  ⟵ *first correctness gate*
- Port `aidsCalculate` / `muaidsCalculate` (AIDS + QUAIDS, demographics, adding-up).
- **Done when:** Julia shares match R `qshares.rds` to ~1e-10 on the test params.

### M2 — Julia core: censored log-likelihood  ⟵ *the hard part*
- Port `censoredaidsLoglike`: full-purchase MVN density + partial-regime Wales–Woodland
  reduction with `pmvnorm`-style multivariate-normal CDF.
- **Key risk:** Julia equivalent of R's `mvtnorm::pmvnorm` (multivariate normal CDF).
  Candidates: `MvNormalCDF.jl`, `Distributions.jl`. De-risk this early.
- **Done when:** Julia per-obs log-likelihood matches R `loglikes.rds` to tolerance.

### M3 — Estimation engine
- Maximize the summed log-likelihood (R leaves optimization to the user; we provide it):
  optimizer (Optim.jl / NLopt) + start values + vcov from the Hessian.
- Port `censoredElasticity` (simulation-based expected shares, numeric derivatives, delta-method SEs).
- **Done when:** full estimate on the Mexican data reproduces sensible params + elasticities;
  documented tolerance vs. an R reference run.
- ✅ DONE. Elasticities reproduce R to ~2e-8 (injected-ε oracle, src/elasticities.jl); `estimate()`
  works and returns a PSD vcov (src/estimate.jl). 35/35 tests pass.
- ⚠️ **Finding (faithful, NOT a bug):** the published params are NOT the argmax of the package's
  censored log-likelihood — gradient norm ~51,793 at `params_loglike` (deterministic: identical at
  mc=2000/8000), likelihood improves ~2360 nats *in every regime*, including the deterministic ones
  (nu=3,4) we match R to 1e-14 — which proves R's likelihood behaves identically. Likely a GAUSS-vs-
  R-package modeling/regularization difference (the +1e-7/+1e-8 log floors, or the 2-demo paper-code
  issue). Flagged for **M4** (fudge-factor scrutiny) and **M10** docs. See [[published-params-not-mle]].

### M4 — Validity & improvement  ⟵ *NEW (Noé): economic-theory tests, not just R-parity*
- Test the estimator against microeconomic equalities on the estimated elasticities:
  - **Engel aggregation:** Σᵢ wᵢ ηᵢ = 1 (income elasticities, budget-weighted).
  - **Cournot aggregation:** Σᵢ wᵢ εᵢⱼ = −wⱼ.
  - **Homogeneity (degree-zero):** Σⱼ εᵢⱼ + ηᵢ = 0.
  - **Slutsky symmetry** of the compensated effects.
- "Improve it": replace the R code's numerical fudges (`+1e-7`/`1e-8`, `nearPD`) with
  principled handling; harden the likelihood where theory exposes weakness.
- Cross-check full estimate vs. the external reference (correctness standard = BOTH).
- **Done when:** theory identities hold to tolerance AND external reference matches.

### M5 — Parallelize the truncated log-likelihood  ⟵ *NEW (Noé)*
- The per-observation loop in `censoredaidsLoglike` is embarrassingly parallel
  (each household's contribution is independent). Parallelize it (Julia threads / parallel map).
- **Done when:** parallel path is faster and bit-for-bit consistent with the serial path.

### M6 — Return to testing (regression after parallelization)  ⟵ *NEW (Noé)*
- Re-run ALL gates: R golden fixtures, the M4 theory identities, the external reference.
  Parallelization must change speed only, never the numbers.
- Benchmark serial vs. parallel; record speedup.
- **Done when:** every prior gate still passes; speedup documented. *Then we move on.*

### M7 — Job entrypoint & container
- `estimate(data, config) -> results(JSON)` plus a CLI/headless entrypoint (CSV/JSON in, JSON out).
- Dockerfile (Julia + package, precompiled).
- **Done when:** `docker run` produces correct results from a mounted input file.

### M8 — AWS job layer
- S3 (input data + results), AWS Batch compute, run-metadata store.
- Minimal Terraform so it's reproducible and tear-down-able.
- **Done when:** submit a job by dropping input in S3 → results land in S3, status queryable.

### M9 — MCP server
- Python MCP server; tools wrapping the job layer; single-user first; results returned to client.
- **Done when:** a user can run a full estimation end-to-end conversationally.

### M10 — End-to-end verification & docs
- Mexican ENIGH data through the entire pipeline; compare to R + external reference; document.
- **Done when:** numbers match, README + usage docs exist.

---

## Decisions — LOCKED (Noé, 2026-05-30)

1. **AWS compute:** ✅ **AWS Batch** (Fargate-backed).
2. **IaC:** ✅ **Terraform**.
3. **MCP server language:** ✅ **Python**.
4. **Auth / multi-user:** ✅ **single-user / Noé's AWS account first**, per-user later.
5. **Optimizer/AD in Julia:** **Optim.jl + numeric/ForwardDiff**, NLopt fallback (confirm in M2/M3).

## Correctness standard — LOCKED (Noé, 2026-05-30): R parity + economic theory
- Gate on R golden fixtures (`qshares.rds`, `loglikes.rds`) to tight tolerance, **AND**
- Gate on microeconomic theory identities (M4): Engel/Cournot aggregation, homogeneity, Slutsky symmetry.
- External reference: **dropped** (per Noé) — not used.

---

## Open questions for Noé
- Your mental "steps" — does this sequence match how you pictured it? What's missing or out of order?
- Is reproducing the R numbers (golden-test parity) the right definition of "correct" for you,
  or do you have a separate reference (paper results, a known dataset) you trust more?
- Roughly how big are real user datasets (rows × goods)? Drives the AWS compute sizing.
