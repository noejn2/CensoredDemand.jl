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
- **Done when:** theory identities hold to tolerance.
- ✅ DONE (41/41 tests). Results:
  - **Homogeneity (8.1e-6), Engel (5.1e-8), Cournot (1.1e-7)** hold to FD precision (structural). ✅
  - **Slutsky symmetry FAILS (~0.105)** — concentrated at the thin Juice margin (w≈0.012); expected
    for a censored/simulated system (symmetry not enforced post-censoring). Documented, tested < 0.15.
  - **Improvement = the floor, not nearPD.** Investigation pinned the M3 finding to the additive
    `log(x+1e-7/1e-8)` floor masking 36 households (orthant p≈1e-26) → ~268 nats of spurious credit;
    `nearestPD` is a no-op (R_c always PD). Added OPT-IN `floor_mode` kwarg (`:additive_r` default =
    verbatim R/parity-preserving; `:guard` = honest `log(max(x,1e-300))`, drops sum −4511.66→−4790.71).
    Defaults unchanged → M2 R-parity intact. See [[published-params-not-mle]].

### M5 — Parallelize the truncated log-likelihood  ⟵ *NEW (Noé)*
- The per-observation loop in `censoredaidsLoglike` is embarrassingly parallel
  (each household's contribution is independent). Parallelize it (Julia threads / parallel map).
- **Done when:** parallel path is faster and bit-for-bit consistent with the serial path.
- ✅ DONE. `Threads.@threads` over the household loop; replaced the shared sequential RNG with a
  per-obs `MersenneTwister(hash((seed, i)))` (`rng` kwarg → `seed`) so results are order- and
  thread-independent. **Bit-identical across 1/4/8 threads** (sum −4511.663045501727), **~5.6× at 8
  threads** (0.110s → 0.020s). `bench/bench_loglike.jl`.

### M6 — Return to testing (regression after parallelization)  ⟵ *NEW (Noé)*
- Re-run ALL gates: R golden fixtures, the M4 theory identities.
  Parallelization must change speed only, never the numbers.
- Benchmark serial vs. parallel; record speedup.
- **Done when:** every prior gate still passes; speedup documented. *Then we move on.*
- ✅ DONE. Full suite 41/41 green post-parallelization (R-parity M0–M2, M3 estimate/elasticities,
  M4 identities/floor); cross-thread bit-identity + 5.6× speedup recorded via `bench/bench_loglike.jl`.

### M7 — Job entrypoint & container
- `estimate(data, config) -> results(JSON)` plus a CLI/headless entrypoint (CSV/JSON in, JSON out).
- Dockerfile (Julia + package, precompiled).
- **Done when:** `docker run` produces correct results from a mounted input file.
- ✅ CODE DONE + locally verified. `run_job(config)` + `bin/run.jl` (JSON config in / results JSON out),
  wrapping `estimate`/`censored_elasticity`; `sample/` demo reproduces R elasticities to 0.015 (own-draws
  MC). JSON3 added. 53/53 tests. `Dockerfile` (julia:1.11-bookworm, deps precompiled in-image,
  `JULIA_NUM_THREADS=auto`) + `.dockerignore` authored & statically sound.
- ⚠️ **`docker build`/`run` BLOCKED by host environment** (not our artifacts): Docker Desktop's internal
  proxy `http.docker.internal:3128` isn't forwarding registry traffic, so the daemon's build/pull pipeline
  hangs (host `curl` to the registry works). Fix = Docker Desktop → Settings → Resources → Proxies → "No
  proxy"/system default, then `docker build -t censored-demand:m7 .` && `docker run --rm -v <work>:/work
  censored-demand:m7 /work/config.json`. OTHERWISE the image builds in the cloud in M8 (AWS CodeBuild),
  where the local proxy is irrelevant — so this verification naturally completes there.

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
