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
using DelimitedFiles
using Distributions
using MvNormalCDF
using Optim
using ForwardDiff
using CSV
using DataFrames
using JSON3

export aids_shares, censored_loglike, censored_elasticity, estimate, run_job,
       initial_values, check_start

# ---- M1: AIDS/QUAIDS share equations ----
# Faithful port of censoredAIDS::aidsCalculate. Validated to ~1e-16 vs R across
# all four modes (AIDS/QUAIDS × demographics/none) by a judge-panel of two
# independent ports plus a 12-case oracle (see test/runtests.jl).
include("shares.jl")

# ---- M2: Wales–Woodland censored log-likelihood ----
# Faithful port of censoredAIDS::censoredaidsLoglike (winner of a 3-way judge
# panel; all three independent ports agreed to the last bit). Deterministic
# regimes (full-purchase + all-but-one-bought) reproduce R to ~1e-14; the
# stochastic-CDF partial regimes match R to within Monte-Carlo noise. Uses
# MvNormalCDF for the censored-good orthant probabilities. See test/runtests.jl.
include("loglike.jl")

# ---- M3: simulation-based elasticities ----
# Faithful port of censoredAIDS::censoredElasticity (winner of a 2-way judge
# panel; reproduces R to ~2e-8 under injected ε draws). Amemiya–Tobin truncation
# mapping over draws, numerical price/income derivatives, delta-method SEs.
# Supports an `epsilons` kwarg to inject draws for near-deterministic validation.
include("elasticities.jl")

# ---- Starting values: principled LA-AIDS start + appropriateness check ----
include("start.jl")

# ---- M3: maximum-likelihood estimation driver ----
# Maximizes the summed censored log-likelihood (Optim.jl); vcov from a numerical
# Hessian. NOTE: the published params are NOT the argmax of this likelihood
# (a faithful property of the package likelihood — see test/runtests.jl and PLAN.md).
include("estimate.jl")

# ---- M7: headless JSON-config job entrypoint ----
# `run_job(config::AbstractDict)` drives estimate / censored_elasticity from a
# plain config dict (parsed JSON). This is the surface the AWS job layer + MCP
# server call. Wraps the body in try/catch -> {status:"ok"|"error", ...}.
include("job.jl")

end # module CensoredDemand
