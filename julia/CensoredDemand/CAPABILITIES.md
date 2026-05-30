# CensoredDemand.jl — Capabilities Inventory

A complete picture of what the package does, every option ("knob"), how it's validated,
the key findings, and performance. Companion to `PLAN.md` (milestones) and the git history.

`CensoredDemand.jl` estimates **censored AIDS and QUAIDS demand systems** by maximum likelihood —
a faithful Julia port **and** extension of the R `censoredAIDS` package (the method behind
Nava & Dong 2022, "Taxing sugary-sweetened beverages in México"). Zero-expenditure (censored)
households are handled via the Wales–Woodland likelihood.

---

## 1. Exported API

| Function | Purpose |
|---|---|
| `aids_shares(prices, budget, params; …)` | AIDS/QUAIDS predicted budget shares |
| `censored_loglike(shares, prices, budget, params; …)` | Wales–Woodland per-household censored log-likelihood |
| `estimate(shares, prices, budget; …)` | Maximum-likelihood estimation (→ params, vcov, se, diagnostics) |
| `censored_elasticity(prices, budget, params; …)` | Simulation-based price/income elasticities + delta-method SEs |
| `initial_values(shares, prices, budget; …)` | Principled LA-AIDS starting values |
| `check_start(start, shares, prices, budget; …)` | Starting-value appropriateness check (Σ PD, finite ll, counts) |
| `run_job(config::Dict)` | Headless JSON-config job runner (CLI + cloud entry point) |

## 2. Options (the knobs)

| Option | Values | Where | Notes |
|---|---|---|---|
| **Model** | `quaids = false\|true` | all | AIDS (linear) vs QUAIDS (quadratic) |
| **Demographics** | `demographics = nothing\|n×t` | all | translate expenditure inside the deflator |
| **Price index** | `price_index = :translog\|:stone` | shares/loglike/estimate/elasticity/job | `:translog` = full QUAIDS (R-faithful); `:stone` = LA-AIDS, limits nonlinearity |
| **Likelihood floor** | `floor_mode = :additive_r\|:guard` | loglike/estimate | `:additive_r` = verbatim R floor; `:guard` = honest `log(max(p,1e-300))` |
| **Optimizer** | `algorithm = :neldermead\|:bhhh` | estimate | `:bhhh` = gradient-based Gauss–Newton, OPG covariance |
| **Parallelism** | `parallel = true\|false` + `JULIA_NUM_THREADS`/`-t N` | loglike/estimate | per-household loop threaded; bit-identical to serial |
| **Start** | `start = nothing\|vector`, `check = true\|false` | estimate | default = `initial_values` (LA-AIDS), validated by `check_start` |

All non-default options are **opt-in**; the defaults reproduce the R package exactly.

## 3. Validation — 74 automated tests (`test/runtests.jl`, ~18 s)

| Component | Standard | Result |
|---|---|---|
| Share equations | vs R `qshares` + 12-case multi-mode oracle | **exact to ~1e-16**; 2 independent ports agreed |
| Censored log-likelihood | vs R `loglikes` (integer-sum gate; deterministic regimes) | sum **−4512** (R's own gate); det. regimes **1e-14**; 3 ports bit-identical |
| Elasticities | vs R injected-ε oracle | **~2e-8** (elasticities), **3e-16** (expected shares) |
| Theory identities | Engel / Cournot / homogeneity | hold to **~1e-6** (Slutsky symmetry ~0.105 — see findings) |
| Parallelization | serial vs 1/4/8/16 threads | **bit-identical**; 5.7× at 8 threads |
| Initial values / packing | round-trip + transposition control | packing **provably exact** (1.8e-12; broken-θ control diverges 7609) |

## 4. Key findings

- **The published estimates are NOT the likelihood maximum.** Gradient ≈ 51,793 at the published
  params; a crude LA-AIDS start already scores **−2765.5 vs −4511.7** (beats the paper's "MLE" by
  ~1746 nats). **Mechanism:** an additive `log(p + 1e-8)` floor hands ~268 nats of spurious credit
  to **36 households** whose observed regime the params deem near-impossible (orthant prob ≈ 1e-26).
  The `:guard` floor exposes it (sum → −4790.7).
- **The "issue" is in the paper's code, not the method.** The replication GAUSS script comments out
  education & sex (`@…@`), estimating a 2-demographic model while the published table reports 4.
- **Slutsky symmetry fails (~0.105)**, concentrated on the thin Juice margin (w≈0.012) — expected for
  a censored/simulated system where symmetry isn't enforced.

## 5. Performance

- Single log-likelihood evaluation: **~6 ms** (615 hh, mc=2000, 8 threads); **~32 ms** serial.
- Full MLE: **~20–25 s** at 8 threads vs **~133 s** serial (5.7×; 6.5× at 16, diminishing past 8).
- **BHHH reaches a better optimum than Nelder–Mead in ~40× fewer iterations.**

### Convergence matrix (LA-AIDS start, 615 hh, QUAIDS + demographics, 8 threads)

| algorithm / index | loglik | iters | converged | time | mean·\|g\|/n |
|---|---|---|---|---|---|
| neldermead / translog | −1982.9 | 8000 (cap) | no | 56.7 s | — (no gradient) |
| neldermead / stone | −1910.7 | 8000 (cap) | no | 55.2 s | — |
| **bhhh / translog** | −1979.5 | **110** | **yes** | 40.3 s | 2.1e-5 |
| **bhhh / stone** | −1909.5 | **76** | **yes** | 26.1 s | 1.6e-4 |

- **BHHH converges to a local optimum** (gradient ≈ 2e-5 / 1.6e-4 — far below the original GAUSS
  threshold of 0.12) in **~76–110 iterations**. Nelder–Mead reaches a similar likelihood but its
  simplex-spread criterion does **not** trigger even at the 8000-iteration cap (and it gives no
  gradient signal). → use `algorithm = :bhhh` for real fits.
- **`:translog` vs `:stone` log-likelihoods are NOT directly comparable** — they are *different
  models* (different price index), each maximized over its own likelihood. Pick the index for
  economic reasons; `:stone` is the easier-to-estimate linear approximation.

## 6. Infrastructure & status

- **Local (done, M0–M7):** the full estimator + a headless `run_job` CLI (`bin/run.jl`, JSON in/out)
  + a Docker image (`Dockerfile`; built locally is blocked by a host Docker-Desktop proxy, but builds
  in the cloud).
- **Cloud (pending, M8–M10):** AWS Batch + S3 + Terraform job layer, then a Python MCP server, then
  end-to-end docs. Paused at the AWS boundary pending credentials.
- Repo: `julia/CensoredDemand/` (package), `scripts/` (R fixture/oracle exporters), `aws/`, `mcp/`.
