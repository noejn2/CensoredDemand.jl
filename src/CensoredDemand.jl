"""
CensoredDemand.jl — Maximum-likelihood estimation of CENSORED demand systems,
supporting BOTH the linear Almost Ideal Demand System (AIDS / AI) and the
Quadratic Almost Ideal Demand System (QUAIDS / QUAI), selected via the `quaids`
flag. Handles zero-expenditure (censored) observations via the Wales–Woodland
likelihood.
"""
module CensoredDemand

using LinearAlgebra
using Statistics
using Random
using Distributions
using MvNormalCDF
using DataFrames

export aids_shares, censored_loglike, naive_loglike, naive_censored_loglike,
       sy_loglike, sy_first_stage,
       censored_elasticity, estimate,
       initial_values, check_start,
       # ---- typed surface (Change 1) ----
       PriceIndex, TRANSLOG, STONE, FloorMode, ADDITIVE_R, GUARD,
       ModelSpec, EstimationResult, ElasticityResult, SimData, MonteCarloResult,
       param_names,
       # ---- reporting (Change 2) ----
       coeftable, elasticity_table, montecarlo_table,
       # ---- simulation / Monte-Carlo (Change 5) ----
       simulate_prices, simulate_data, montecarlo

# ---- Typed surface: option enums + result structs ----
# Additive, type-stable boundary; the numerical core keeps comparing canonical Symbols.
include("types.jl")

# ---- AIDS/QUAIDS share equations ----
# Faithful port of censoredAIDS::aidsCalculate. Validated to ~1e-16 vs R across
# all four modes (AIDS/QUAIDS × demographics/none) by a judge-panel of two
# independent ports plus a 12-case oracle (see test/runtests.jl).
include("shares.jl")

# ---- Wales–Woodland censored log-likelihood ----
# Faithful port of censoredAIDS::censoredaidsLoglike (winner of a 3-way judge
# panel; all three independent ports agreed to the last bit). Deterministic
# regimes (full-purchase + all-but-one-bought) reproduce R to ~1e-14; the
# stochastic-CDF partial regimes match R to within Monte-Carlo noise. Uses
# MvNormalCDF for the censored-good orthant probabilities. See test/runtests.jl.
include("loglike.jl")

# ---- Naive (uncensored) log-likelihood ----
# The misspecified alternative that ignores Wales–Woodland censoring: the plain Gaussian
# density of the share residuals for every household. Used by estimate(...; loglike=:naive)
# and by the Monte-Carlo study that quantifies the cost of ignoring censoring.
include("naive.jl")

# ---- Naive-CENSORED (per-equation Tobit) log-likelihood ----
# The OTHER misspecification: treating the zeros as a statistical CENSORING problem (latent
# positive demand censored at 0) instead of the genuine corner solutions they are — a per-equation
# Tobit with no Wales–Woodland reallocation. Used by estimate(...; loglike=:naive_censored) and the
# Monte-Carlo study that contrasts the corner-solution vs censored-data treatments of zeros.
include("naive_censored.jl")

# ---- Shonkwiler–Yen (1999) two-step likelihood ----
# The most common practitioner treatment of zeros: a per-good probit first stage, then a Gaussian
# system on the censoring-corrected mean Φ̂·w̄(θ) + δ·φ̂ over all households. Parameter layout
# [θ…, δ (m−1), σ]; with Φ≡1, φ≡0, δ=0 it equals naive_loglike exactly (asserted in the tests).
# Used by estimate(...; loglike=:sy) and the Monte-Carlo estimator comparison.
include("sy.jl")

# ---- Simulation-based elasticities ----
# Faithful port of censoredAIDS::censoredElasticity (winner of a 2-way judge
# panel; reproduces R to ~2e-8 under injected ε draws). Amemiya–Tobin truncation
# mapping over draws, numerical price/income derivatives, delta-method SEs.
# Supports an `epsilons` kwarg to inject draws for near-deterministic validation.
include("elasticities.jl")

# ---- Starting values: principled LA-AIDS start + appropriateness check ----
include("start.jl")

# ---- Maximum-likelihood estimation driver ----
# Maximizes the summed censored log-likelihood with the BHHH / Gauss–Newton optimizer
# (per-observation scores); vcov is the OPG estimator (S'S)⁻¹. NOTE: the published params
# are NOT the argmax of this likelihood (a faithful property of the package likelihood —
# see test/runtests.jl).
include("estimate.jl")

# ---- Simulation + Monte-Carlo (Change 5) ----
# Likelihood-consistent DGP (U + ε, then Amemiya–Tobin censoring) + an estimator
# recovery study (bias / RMSE / SE-coverage). Proves the estimator on data drawn
# from known coefficients (incl. simulated prices).
include("simulate.jl")

# ---- Reporting ----
# Base.show + DataFrame tables (coef/se/t/p, labeled elasticities, MC recovery).
include("report.jl")

end # module CensoredDemand
