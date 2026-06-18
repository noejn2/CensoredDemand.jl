# CensoredDemand.jl — Capabilities Inventory

A complete picture of what the package does, every option ("knob"), how it's validated,
the key findings, and performance. Companion to the [README](README.md) and the git history.

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
| `simulate_prices` / `simulate_data(n, params, spec; …)` | Simulate a censored dataset from known coefficients (the model's own DGP) |
| `montecarlo(n, params, spec; …)` | Estimator-performance study — bias / RMSE / SE-coverage over reps |
| `coeftable` / `elasticity_table` / `montecarlo_table` | Tidy `DataFrame` reports |

`estimate` returns an `EstimationResult`, `censored_elasticity` an `ElasticityResult`, `montecarlo` a
`MonteCarloResult` — typed structs with pretty `Base.show`. Options are `@enum`s (`PriceIndex`,
`FloorMode`) that also accept the equivalent Symbols (`:translog`, `:guard`, …) for back-compat.

## 2. Options (the knobs)

| Option | Values | Where | Notes |
|---|---|---|---|
| **Model** | `quaids = false\|true` | all | AIDS (linear) vs QUAIDS (quadratic) |
| **Demographics** | `demographics = nothing\|n×t` | all | translate expenditure inside the deflator |
| **Price index** | `price_index = :translog\|:stone` (or `TRANSLOG`/`STONE`) | shares/loglike/estimate/elasticity | `:translog` = full QUAIDS (R-faithful); `:stone` = LA-AIDS, limits nonlinearity |
| **Likelihood floor** | `floor_mode = :additive_r\|:guard` (or enum) | loglike/estimate | `:additive_r` = verbatim R floor; `:guard` = honest `log(max(p,1e-300))` |
| **Optimizer** | BHHH (sole algorithm) | estimate | gradient-based Gauss–Newton + OPG covariance; Nelder-Mead / Optim.jl removed |
| **FD score mode** | `fd_mode = :central\|:forward` | estimate | `:forward` ≈ 2× faster per iteration; final score + vcov always central |
| **Slutsky symmetry** | `symmetry = false\|true` | elasticity | min-distance symmetrization of the compensated matrix; default off (R-faithful) |
| **Parallelism** | `parallel = true\|false` + `JULIA_NUM_THREADS`/`-t N` | loglike/estimate | per-household loop threaded; bit-identical to serial |
| **Start** | `start = nothing\|vector`, `check = true\|false` | estimate | default = `initial_values` (LA-AIDS), validated by `check_start` |

All non-default options are **opt-in**; the defaults reproduce the R package exactly. Option
arguments accept either a `Symbol` or the matching `@enum` value.

## 3. Validation — 110 automated tests (`test/runtests.jl`, ~34 s)

| Component | Standard | Result |
|---|---|---|
| Share equations | vs R `qshares` + 12-case multi-mode oracle | **exact to ~1e-16**; 2 independent ports agreed |
| Censored log-likelihood | vs R `loglikes` (integer-sum gate; deterministic regimes) | sum **−4512** (R's own gate); det. regimes **1e-14**; 3 ports bit-identical |
| Elasticities | vs R injected-ε oracle | **~2e-8** (elasticities), **3e-16** (expected shares) |
| Theory identities | Engel / Cournot / homogeneity | hold to **~1e-6** (Slutsky symmetry ~0.105 by default — see findings) |
| **Slutsky symmetry option** | `symmetry=true` (translog & stone) | Slutsky residual **< 1e-10**; other identities still hold; default byte-identical |
| **Types & enums** | enum vs Symbol; struct field names | **byte-identical** results; structs are drop-in for callers |
| **Reporting** | `show` / `coeftable` / `elasticity_table` | non-empty render; correct table shapes |
| **Simulation & recovery** | DGP validity + seeded Monte-Carlo | shares add up, censoring present; estimator recovers α/β, coverage sane |
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
- **Slutsky symmetry** of the *parameters* (γᵢⱼ=γⱼᵢ) is imposed structurally; the ~0.105 residual is
  entirely in the **simulated censored elasticities** (truncation + Monte-Carlo break it), concentrated
  on the thin Juice margin (w≈0.012). Passing **`symmetry=true`** to `censored_elasticity` imposes it
  **exactly** (residual < 1e-10) via min-distance symmetrization, while preserving Engel/Cournot/homogeneity.
- **The censored estimator is a *demonstrated* benchmark, not an assumed one.** It is fit on simulated
  data and shown to recover the truth two ways: (i) **parameter-level** — simulating from known
  coefficients and re-estimating, the bias on the well-identified α/β block shrinks with n (≈0.25 at
  n=500 → ≈0.08 at n≥2000) with near-nominal SE coverage (`montecarlo` / `simulate_data`, study
  `06_montecarlo.jl`); (ii) **elasticity-level** — in the estimator-comparison study the `correct`
  cell recovers the true elasticities (low bias/RMSE, coverage → nominal as n grows). Only then is it
  used as the reference the misspecified estimators are measured against.
- **Two DGPs, both the corner-solution truth; the misspecifications are *estimation* choices.** The
  data are generated ONLY as truncated AIDS / QUAIDS (Amemiya–Tobin / Wales–Woodland corner solutions,
  translog index, demographics). The estimator-comparison study (`11_mc_estimators.jl`) then re-fits
  each dataset several ways and reads the elasticity bias against the `correct` benchmark:
  `naive` (ignores the zeros), `naive_censored` (treats the zeros as a **Tobit** censoring problem —
  latent demand unobserved at 0, per-equation, **no reallocation** — the wrong *statistical* treatment
  of genuine corners), `sy` (the **Shonkwiler–Yen 1999 two-step**: per-good probit first stage
  `sy_first_stage`, then the Gaussian system on the corrected mean Φ̂·w̄(θ)+δ·φ̂ via `sy_loglike`,
  `estimate(...; loglike=:sy)`; params gain a δ block `[θ…, δ (m−1), σ]`), plus earlier-draft
  variants (`stone`, `dropdemos`) kept in the cached results but no longer in the paper.

## 5. Performance

- Single log-likelihood evaluation: **~6 ms** (615 hh, mc=2000, 8 threads); **~37 ms** serial.
- Per-eval allocation **~23 MB**, dominated by the **external `mvnormcdf` QMC orthant routine** (called
  for the ~452 partial-censoring households/eval) — *not* our matrix algebra. So the productive speed
  levers are the **evaluation count** (`fd_mode`), **cores** (threading), and warm starts — confirmed by
  profiling (`bench/profile.jl`), not guessed. (StaticArrays on our small-matrix code was deliberately
  *not* pursued: ~3% of allocation, real risk to the 1e-14 parity.)

### Convergence / timing (LA-AIDS start, 615 hh, QUAIDS + demographics, 8 threads)

| config | loglik | iters | time | mean·\|g\|/n |
|---|---|---|---|---|
| **bhhh central / translog** (default) | −1979.5 | 84 | 28.7 s | 7.5e-5 |
| **bhhh forward / translog** | −1979.6 | 62 | **11.8 s** | 0.095 |
| **bhhh central / stone** | −1909.5 | 76 | 26.1 s | 1.6e-4 |

- **`fd_mode = :forward` ≈ 2.4× faster** for the same optimum (−1979.6 vs −1979.5) — it halves the
  per-iteration score evaluations; the final gradient + OPG covariance are still central-accurate.
  Default is `:central` (tightest gradient); use `:forward` for long runs.
- **BHHH is the sole optimizer** — gradient-based, converges to a local optimum (gradient far below the
  original GAUSS threshold of 0.12) in **~60–110 iterations**. Nelder-Mead / Optim.jl were removed.
- **`:translog` vs `:stone` log-likelihoods are NOT directly comparable** — they are *different models*
  (different price index), each maximized over its own likelihood. Pick the index for economic reasons;
  `:stone` is the easier-to-estimate linear approximation.

## 6. Status & provenance

- **Standalone Julia package.** Pure-Julia estimation library — no runtime R dependency. Layout:
  `src/` (package), `test/` (suite + committed R golden-oracle fixtures under `test/fixtures/`),
  `bench/` (profiling). Deps: `Distributions`, `MvNormalCDF`, `DataFrames`, and stdlibs.
- **Provenance.** Method ported from the R `censoredAIDS` package; correctness is defined as R
  golden-fixture parity + microeconomic theory identities (see §3). The fixtures were generated from
  R and committed as CSVs (the R generator scripts are not part of the package; they live in git history).
