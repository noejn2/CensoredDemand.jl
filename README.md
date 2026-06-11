# CensoredDemand.jl

Maximum-likelihood estimation of **censored** demand systems — supporting **both** the linear
**AIDS** (Almost Ideal Demand System) and the quadratic **QUAIDS** (Quadratic Almost Ideal Demand
System), selected by a single `quaids` flag. Zero-expenditure (censored) households are handled via
the Wales–Woodland likelihood.

It is a faithful Julia port **and** extension of the R [`censoredAIDS`](https://github.com/noejn2/censoredAIDS)
package (the method behind Nava & Dong 2022, *Taxing sugar-sweetened beverages in México*), validated
to machine precision against the R results.

## Both AIDS and QUAIDS

One estimator, two specifications, one consistent code path:

- `quaids = false` → linear **AIDS / AI**
- `quaids = true`  → quadratic **QUAIDS / QUAI**

## Install

```julia
using Pkg
Pkg.develop(path = "/path/to/CensoredDemand")   # or Pkg.add(url = "…") once hosted
```

## Quick start

```julia
using CensoredDemand

# shares: n×m budget shares (zeros mark censoring); prices: n×m LOGGED prices;
# budget: length-n LOGGED total expenditure; Z: optional n×t demographics.

# Maximum-likelihood estimation (BHHH / Gauss–Newton; OPG covariance).
res = estimate(shares, prices, budget; quaids = true, demographics = Z)
res                       # pretty coefficient table (Base.show)
coeftable(res)            # → DataFrame (name, coef, se, t, p)

# Price/income elasticities with delta-method SEs (optionally Slutsky-symmetric).
el = censored_elasticity(prices, budget, res.params;
                         quaids = true, demographics = Z, vcov = res.vcov,
                         symmetry = true)
elasticity_table(el)      # → tidy DataFrame

# Simulate from known coefficients and verify the estimator recovers them.
spec = res.spec
mc = montecarlo(500, res.params, spec; reps = 20)   # bias / RMSE / SE-coverage
```

Speed tip for long runs: `estimate(...; fd_mode = :forward)` ≈ 2× faster per iteration (the final
gradient + covariance stay central-accurate). Threading: launch Julia with `-t N` /
`JULIA_NUM_THREADS=N` — the per-household likelihood loop is parallel and bit-identical to serial.

## Capabilities & options

The full API, every option (`price_index`, `floor_mode`, `symmetry`, `fd_mode`, parallelism, …),
validation, key findings, and performance are documented in **[CAPABILITIES.md](CAPABILITIES.md)**.
Options accept either a `Symbol` (`:translog`) or the matching `@enum` (`TRANSLOG`).

## Correctness

Two gates, both green: **R golden-fixture parity** (shares ~1e-16, censored log-likelihood at R's own
integer tolerance, elasticities ~2e-8) **and** microeconomic theory identities (Engel / Cournot /
homogeneity; Slutsky symmetry exactly imposable via `symmetry=true`). The estimator is also verified
**consistent under its own data-generating process** via the simulation/Monte-Carlo recovery study.

## Testing

```julia
using Pkg; Pkg.test()          # ~117 tests; use `julia -t8` for the threaded path
```

The R-derived golden-oracle fixtures live under `test/fixtures/` and are committed CSVs; the package
has no runtime R dependency.

## License

[MIT](LICENSE) © Noé J Nava
