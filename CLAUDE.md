# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`CensoredDemand.jl` is a pure-Julia maximum-likelihood estimator for **censored** demand
systems — both linear **AIDS** and quadratic **QUAIDS**, selected by the `quaids` flag.
Zero-expenditure (censored) households use the **Wales–Woodland** likelihood. It is a faithful
Julia port **and** extension of the R [`censoredAIDS`](https://github.com/noejn2/censoredAIDS)
package (the method behind Nava & Dong 2022, sugar-tax paper), validated to machine precision
against committed R golden fixtures. No runtime R dependency.

There are two layers in this repo:
- **the package** — root (`src/`, `test/`, `bench/`, `Project.toml`).
- **`pub/`** — a separate, reproducible publication pipeline (Quarto article + R-vs-Julia
  benchmarks + Monte-Carlo studies). It has its **own** Julia environment (`pub/env/`) and
  depends on the package, R, and Quarto. Treat `pub/` as a downstream consumer, not part of the library.

## Commands

```julia
# Run the full test suite (~117 tests). Use threads to exercise the parallel path.
using Pkg; Pkg.test()
# or, from the shell, faster + threaded:
julia --project=. -t8 test/runtests.jl
```

```bash
# Benchmarks / profiling (not part of CI)
julia --project=. -t8 bench/bench_loglike.jl
julia --project=. -t8 bench/profile.jl

# Reproduce the entire publication (provisions pub/env, builds R censoredAIDS,
# runs every study, renders article.pdf). Needs R + Quarto. Long-running.
bash pub/run_all.sh
```

There is no separate lint step. `runtests.jl` is a single `@testset` tree — to run a subset,
comment out sibling `@testset` blocks or run the file directly and rely on its labels (M0–M7).

## Data conventions (load-bearing)

These are easy to get wrong and silently produce nonsense:
- **`prices` are LOGGED** (n×m matrix), **`budget` is LOGGED** total expenditure (length n).
- **`shares`** are n×m budget shares; **zeros mark censoring** (a household that bought nothing
  of that good).
- **`demographics` (`Z`)** is an optional n×t matrix; **column order is load-bearing** (it maps
  to the packed parameter vector). See `test/runtests.jl` `load_inputs`.
- The validation dataset is 615 households, 4 goods (SSB, Juice, Milk, Water), demographics
  (age, size, educ, sex), in `test/fixtures/testing_data.csv`.

## Architecture

`src/CensoredDemand.jl` is the module file and includes everything in dependency order. Each
file is one concern; read it for the per-file commentary, which records provenance and the
judge-panel that validated each port.

- `types.jl` — typed boundary: option `@enum`s (`PriceIndex`/`TRANSLOG`/`STONE`,
  `FloorMode`/`ADDITIVE_R`/`GUARD`) and result structs (`ModelSpec`, `EstimationResult`,
  `ElasticityResult`, `SimData`, `MonteCarloResult`). **Key design rule:** the numerical core
  always compares **canonical Symbols** internally; the enums are an additive, type-stable
  boundary that *also accepts Symbols* everywhere for back-compat. Struct field names match the
  NamedTuples the functional core used to return, so callers stay drop-in.
- `shares.jl` — `aids_shares`: AIDS/QUAIDS predicted budget shares (port of `aidsCalculate`).
- `loglike.jl` — `censored_loglike`: Wales–Woodland per-household censored log-likelihood; uses
  `MvNormalCDF` orthant probabilities for partially-censored regimes. The per-household loop is
  the threaded hot path (bit-identical serial vs parallel).
- `naive.jl` — `naive_loglike`: the misspecified uncensored alternative (plain Gaussian), used
  to quantify the cost of ignoring censoring.
- `elasticities.jl` — `censored_elasticity`: simulation-based price/income elasticities with
  delta-method SEs; optional Slutsky `symmetry=true` (min-distance symmetrization). Accepts an
  `epsilons` kwarg to inject draws for near-deterministic validation.
- `start.jl` — `initial_values` (principled LA-AIDS start, the default) + `check_start`.
- `estimate.jl` — `estimate`: the MLE driver. **BHHH / Gauss–Newton is the sole optimizer**
  (per-observation scores); covariance is the OPG estimator `(SᵀS)⁻¹`. Returns
  `EstimationResult`. Nelder-Mead / Optim.jl were deliberately removed.
- `simulate.jl` — `simulate_data` / `montecarlo`: the model's own DGP + estimator recovery study.
- `report.jl` — `Base.show` + `coeftable` / `elasticity_table` / `montecarlo_table` DataFrames.

### Options (all non-default behavior is opt-in; defaults reproduce R exactly)

| Option | Values | Notes |
|---|---|---|
| `quaids` | `false`/`true` | AIDS (linear) vs QUAIDS (quadratic) |
| `demographics` | `nothing`/n×t | translates expenditure in the deflator |
| `price_index` | `:translog`/`:stone` | translog = full QUAIDS (R-faithful); stone = LA-AIDS |
| `floor_mode` | `:additive_r`/`:guard` | additive_r = verbatim R floor; guard = honest `log(max(x,1e-300))` |
| `fd_mode` | `:central`/`:forward` | forward ≈ 2× faster/iter; final score + vcov always central |
| `symmetry` | `false`/`true` | (elasticity only) impose Slutsky symmetry exactly |
| `parallel` + `-t N` | `true`/`false` | threaded per-household loop, bit-identical to serial |
| `start`, `check` | vector / bool | default start = `initial_values`, validated by `check_start` |

Every option accepts a `Symbol` *or* the matching `@enum` value.

## Correctness model (the bar for any change)

Two gates must stay green; both run in `test/runtests.jl` against committed CSV fixtures in
`test/fixtures/` (generated from R; no runtime R dep):
1. **R golden-fixture parity** — shares ~1e-16, censored log-likelihood at R's own integer
   tolerance (sum −4512), elasticity expected shares ~3e-16. Elasticity point estimates and SEs
   are deliberately **not** R-parity-checked: `0b3ab5d` fixed the finite-difference sign, which R
   carries wrong, so the old `elasticities_R.csv` / `se_R.csv` fixtures were deleted.
2. **Microeconomic theory identities** — Engel / Cournot / homogeneity; Slutsky symmetry
   exactly imposable via `symmetry=true`.

Plus a simulation/Monte-Carlo check that the estimator is consistent under its own DGP.

When editing the numerical core, **preserve byte-for-byte parity**: keep internal comparisons on
canonical Symbols, keep the threaded loop bit-identical to serial, and don't introduce type
instability in the structs. StaticArrays was deliberately *not* adopted (real risk to the 1e-14
parity for ~3% allocation gain).

## Known finding (do not "fix" as a bug)

The published Nava & Dong estimates are **NOT** the likelihood maximum of this package — a crude
LA-AIDS start scores far better. This is a faithful property of the R method (an additive
`log(p+1e-8)` floor crediting ~36 near-impossible-regime households), not a defect to be patched.
The `:guard` floor exposes it. See `CAPABILITIES.md` §4 and the `08_findings` study.

## Reference docs

- `README.md` — quick start + public API examples.
- `CAPABILITIES.md` — exhaustive API/option inventory, validation table, findings, performance.
- `pub/README.md` — the publication pipeline and what each `pub/scripts/*` produces.
