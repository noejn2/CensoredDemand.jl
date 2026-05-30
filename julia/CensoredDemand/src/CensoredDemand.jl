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

export aids_shares, censored_loglike, censored_elasticity, estimate

"""
    aids_shares(...)

Compute predicted budget shares for the demand system. Supports BOTH the linear
AIDS (`quaids = false`) and the quadratic QUAIDS (`quaids = true`) specifications.

Not yet implemented.
"""
function aids_shares(args...; kwargs...)
    error("aids_shares is not yet implemented")
end

"""
    censored_loglike(...)

Wales–Woodland censored log-likelihood for the demand system, supporting BOTH
AIDS and QUAIDS via the `quaids` flag. Uses MvNormalCDF for the multivariate
normal orthant probabilities induced by zero-expenditure (censored) regimes.

Not yet implemented.
"""
function censored_loglike(args...; kwargs...)
    error("censored_loglike is not yet implemented")
end

"""
    censored_elasticity(...)

Expenditure and (compensated/uncompensated) price elasticities for the estimated
censored demand system. Supports BOTH AIDS and QUAIDS via the `quaids` flag.

Not yet implemented.
"""
function censored_elasticity(args...; kwargs...)
    error("censored_elasticity is not yet implemented")
end

"""
    estimate(...)

Maximum-likelihood estimation entry point. Set `quaids = false` for the linear
AIDS (AI) system or `quaids = true` for the quadratic QUAIDS (QUAI) system.

Not yet implemented.
"""
function estimate(args...; kwargs...)
    error("estimate is not yet implemented")
end

end # module CensoredDemand
