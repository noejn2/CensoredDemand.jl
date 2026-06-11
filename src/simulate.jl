# Simulation + Monte-Carlo for the censored AIDS / QUAIDS demand system.
#
# The data-generating process is LIKELIHOOD-CONSISTENT (matches loglike.jl): the
# latent shares are the systematic shares plus a multivariate-normal error,
# `w* = aids_shares(p,x;θ) + ε`, `ε ~ N(0, Σ)` on the m−1 free goods (m-th by
# adding-up); the observed censored shares apply the Wales–Woodland / Amemiya–Tobin
# map (negatives → 0, renormalize) = `_treat_truncations`. So the MLE in estimate.jl
# is correctly specified for data produced here, which is what makes the recovery
# study a true test of the estimator.
#
# Simulation is defined for the STRUCTURAL `:translog` index. The `:stone` index
# consumes observed shares in its deflator, so it is not a generative model — we
# error on it rather than fake a fixed point.
#
# Includable into the module: assumes `aids_shares`, `_treat_truncations`,
# `estimate`, `ModelSpec`, `param_names`, the result structs, and the deps are in scope.

using Random, Statistics, LinearAlgebra

# Build Σ ((m-1)×(m-1)) from the trailing j = (m-1)m/2 params (column-major upper-tri
# of the Cholesky-like factor R; Σ = R'R) — same convention as loglike.jl / elasticities.jl.
function _sigma_from_params(params::AbstractVector, m::Integer)
    j = Int(0.5 * (m - 1) * m)
    sig = params[(length(params) - j + 1):end]
    R = zeros(Float64, m - 1, m - 1)
    k = 1
    for jj in 1:(m - 1), ii in 1:jj
        R[ii, jj] = sig[k]; k += 1
    end
    return Symmetric(R' * R)
end

"""
    simulate_prices(n, m; logmean=zeros(m), logcov=0.01*I, rng) -> n×m

Draw `n` rows of LOGGED prices from a multivariate normal with the given log-mean
and log-covariance (mild positive spread by default). Pass `logmean`/`logcov`
estimated from a reference dataset to calibrate realistic prices.
"""
function simulate_prices(n::Integer, m::Integer; logmean = nothing, logcov = nothing,
                         rng::AbstractRNG = Random.MersenneTwister(20240530))
    mu = logmean === nothing ? zeros(m) : Vector{Float64}(logmean)
    C  = logcov  === nothing ? Matrix(0.01 * I, m, m) : Matrix{Float64}(logcov)
    L  = cholesky(Symmetric(C)).L
    return (randn(rng, n, m) * Matrix(L)') .+ mu'      # row_i ~ N(mu, C)
end

"""
    simulate_data(n, params, spec::ModelSpec; prices=nothing, budget=nothing,
                  demographics=nothing, price_logmean, price_logcov,
                  budget_logmean=0.0, budget_logsd=0.5, rng) -> SimData

Generate a synthetic censored dataset of `n` households from the true coefficients
`params` and model `spec`. Prices / budget / demographics are simulated unless a
fixed design is supplied (pass `prices`, `budget`, and/or `demographics` to hold
them constant — a fixed-design Monte-Carlo). Returns a [`SimData`](@ref).

Defined for `spec.price_index == TRANSLOG` only (the structural index).
"""
function simulate_data(n::Integer, params::AbstractVector, spec::ModelSpec;
                       prices = nothing, budget = nothing, demographics = nothing,
                       price_logmean = nothing, price_logcov = nothing,
                       budget_logmean::Real = 0.0, budget_logsd::Real = 0.5,
                       rng::AbstractRNG = Random.MersenneTwister(20240530))
    Symbol(spec.price_index) === :translog ||
        error("simulate_data: simulation is defined for the structural :translog index; " *
              ":stone consumes observed shares and is not a generative DGP")
    m, t = spec.m, spec.t

    P = prices === nothing ?
        simulate_prices(n, m; logmean = price_logmean, logcov = price_logcov, rng = rng) :
        Matrix{Float64}(prices)
    bvec = budget === nothing ?
        (budget_logmean .+ budget_logsd .* randn(rng, n)) :
        Vector{Float64}(budget)
    Z = nothing
    if spec.has_dems
        Z = demographics === nothing ? randn(rng, n, t) : Matrix{Float64}(demographics)
    end

    j = Int(0.5 * (m - 1) * m)
    params_model = params[1:(length(params) - j)]
    U = aids_shares(P, bvec, params_model; quaids = spec.quaids, demographics = Z,
                    price_index = :translog)                       # n×m systematic shares

    # ε ~ N(0, Σ) on the m-1 free goods, append -rowSums for the m-th (adding-up).
    Sigma = _sigma_from_params(params, m)
    Rchol = cholesky(Sigma).U                                     # R'R = Σ
    eps_mm1 = randn(rng, n, m - 1) * Matrix(Rchol)                # n×(m-1)
    eps_full = hcat(eps_mm1, -vec(sum(eps_mm1, dims = 2)))         # n×m, rows sum to 0

    latent = U .+ eps_full                                        # rows sum to 1
    shares = _treat_truncations(latent)                          # censor: negatives→0, renormalize

    return SimData(shares, P, bvec, Z, Vector{Float64}(params), spec)
end

# Convenience: simulate from a fitted result.
simulate_data(n::Integer, r::EstimationResult; kwargs...) =
    simulate_data(n, r.params, r.spec; kwargs...)

"""
    montecarlo(n, params, spec::ModelSpec; reps=100, start=:truth, seed=20240530,
               mc_points=500, maxiters=200, g_tol=1e-5, prices, budget, demographics,
               price_logmean, price_logcov, budget_logmean, budget_logsd,
               est_kwargs=(;)) -> MonteCarloResult

Estimator-performance study: repeat `simulate_data → estimate` `reps` times and
summarize the sampling distribution of θ̂ — per-parameter **bias**, **RMSE**, and
**coverage** (share of reps with truth ∈ θ̂ ± 1.96·se), plus the converged fraction.

`start` controls the optimizer warm start each rep: `:truth` (default — isolates the
estimator's sampling properties), `:la_aids` (cold LA-AIDS start — also tests global
convergence), or an explicit vector. Reps are deterministic via per-rep RNGs seeded
from `seed`. Pass a fixed `prices`/`budget`/`demographics` design to hold it constant.
"""
function montecarlo(n::Integer, params::AbstractVector, spec::ModelSpec;
                    reps::Integer = 100, start = :truth, seed::Integer = 20240530,
                    mc_points::Integer = 500, maxiters::Integer = 200, g_tol::Real = 1e-5,
                    prices = nothing, budget = nothing, demographics = nothing,
                    price_logmean = nothing, price_logcov = nothing,
                    budget_logmean::Real = 0.0, budget_logsd::Real = 0.5,
                    est_kwargs = (;))
    p = length(params)
    truth = Vector{Float64}(params)
    start_vec = start === :truth ? truth : (start === :la_aids ? nothing : Vector{Float64}(start))

    ests = Matrix{Float64}(undef, reps, p)
    ses  = Matrix{Float64}(undef, reps, p)
    conv = falses(reps)
    for r in 1:reps
        rrng = Random.MersenneTwister(hash((seed, r)))            # deterministic per rep
        sd = simulate_data(n, truth, spec; prices = prices, budget = budget,
                           demographics = demographics,
                           price_logmean = price_logmean, price_logcov = price_logcov,
                           budget_logmean = budget_logmean, budget_logsd = budget_logsd,
                           rng = rrng)
        res = estimate(sd.shares, sd.prices, sd.budget;
                       quaids = spec.quaids, demographics = sd.demographics,
                       start = start_vec, mc_points = mc_points, maxiters = maxiters,
                       g_tol = g_tol, price_index = :translog, check = false,
                       share_names = spec.share_names, demographic_names = spec.demographic_names,
                       est_kwargs...)
        ests[r, :] = res.params
        ses[r, :]  = res.se
        conv[r]    = res.converged
    end

    bias = vec(mean(ests, dims = 1)) .- truth
    rmse = vec(sqrt.(mean((ests .- truth').^2, dims = 1)))
    coverage = [mean(abs.(ests[:, k] .- truth[k]) .<= 1.96 .* ses[:, k]) for k in 1:p]
    return MonteCarloResult(truth, ests, bias, rmse, coverage, mean(conv),
                            param_names(spec), spec)
end

# Convenience: study around a fitted result.
montecarlo(n::Integer, r::EstimationResult; kwargs...) =
    montecarlo(n, r.params, r.spec; kwargs...)
