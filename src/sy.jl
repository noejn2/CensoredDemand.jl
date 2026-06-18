# Shonkwiler–Yen (1999) two-step estimator for the censored AIDS / QUAIDS demand system.
#
# The most common practitioner treatment of zero expenditures (AJAE 1999). Step 1 fits, for each
# good, a PROBIT of the purchase indicator 1{w_ig > 0} on x = [1, logged prices, logged budget,
# demographics], yielding Φ̂_ig = Φ(x_i'γ̂_g) and φ̂_ig = φ(x_i'γ̂_g). Step 2 replaces the demand
# system's mean with the censoring-corrected mean and fits ALL households (zeros included) by a
# Gaussian likelihood:
#
#     w_ig = Φ̂_ig · w̄_ig(θ) + δ_g · φ̂_ig + ξ_ig ,   g = 1, …, m−1 .
#
# Parameter layout: [θ (the usual share parameters)…, δ (m−1)…, σ (upper-tri of R, Σ = R'R)] —
# the δ block is spliced between the share and Σ parameters, so the trailing-σ convention of
# loglike.jl / naive.jl is preserved.
#
# Known properties, stated honestly where this estimator is reported:
#  - the corrected means do not add up across goods (the SY system violates adding-up);
#  - the two-step δ̂/θ̂ are estimated conditional on the first-stage γ̂ (no first-stage
#    uncertainty is propagated — exactly how the estimator is used in practice).
#
# DEGENERATE CENSORING: a good with no zeros (or no buyers) in the sample has an unidentified
# probit; its columns fall back to Φ̂ = 1, φ̂ = 0, collapsing that equation's mean to the
# uncensored w̄_ig(θ). With Φ ≡ 1, φ ≡ 0 and δ = 0 the whole objective equals `naive_loglike`
# to machine precision (the test suite asserts this identity).
#
# Includable INTO the CensoredDemand module: assumes `aids_shares` is defined and the package
# `using`s (Distributions, LinearAlgebra) are loaded.

"""
    _probit_fit(y, X; maxiter=50, ridge=1e-6, tol=1e-8) -> Vector{Float64}

Probit MLE by Newton steps on the expected information (Fisher scoring), with a small ridge for
numerical safety (guards near-separation at small n) and an iteration cap. `y` ∈ {0,1}.
"""
function _probit_fit(y::AbstractVector, X::AbstractMatrix;
                     maxiter::Integer = 50, ridge::Real = 1e-6, tol::Real = 1e-8)
    k = size(X, 2)
    beta = zeros(k)
    dist = Normal()
    for _ in 1:maxiter
        z  = X * beta
        Ph = clamp.(cdf.(dist, z), 1e-10, 1 - 1e-10)
        ph = pdf.(dist, z)
        g  = X' * (y .* (ph ./ Ph) .- (1 .- y) .* (ph ./ (1 .- Ph)))   # score
        w  = ph .^ 2 ./ (Ph .* (1 .- Ph))                              # expected information weights
        H  = X' * (X .* w) + ridge * I
        d  = H \ g
        beta += d
        norm(d) < tol && break
    end
    return beta
end

"""
    sy_first_stage(shares, prices, budget, demographics) -> (Phi, phi, B)

Step 1 of Shonkwiler–Yen: per-good probit of the purchase indicator on
x = [1, prices, budget, demographics] for the first `m−1` goods. Returns the n×(m−1) matrices of
fitted Φ̂ and φ̂, plus the k×(m−1) probit coefficient matrix `B` (needed to evaluate the corrected
mean away from the sample, e.g. for SY-convention elasticities). Goods with no zeros (or no
buyers) fall back to Φ̂ = 1, φ̂ = 0 and a zero coefficient column.
"""
function sy_first_stage(shares::AbstractMatrix, prices::AbstractMatrix,
                        budget::AbstractVector, demographics)
    S = Matrix{Float64}(shares)
    P = Matrix{Float64}(prices)
    b = Vector{Float64}(budget)
    n, m = size(S)
    X = demographics === nothing ? hcat(ones(n), P, b) :
                                   hcat(ones(n), P, b, Matrix{Float64}(demographics))
    Phi = ones(n, m - 1)
    phi = zeros(n, m - 1)
    B   = zeros(size(X, 2), m - 1)
    dist = Normal()
    for g in 1:(m - 1)
        y = Float64.(S[:, g] .!= 0.0)
        if any(==(0.0), y) && any(==(1.0), y)          # both regimes present → probit identified
            beta = _probit_fit(y, X)
            z = X * beta
            Phi[:, g] = cdf.(dist, z)
            phi[:, g] = pdf.(dist, z)
            B[:, g]   = beta
        end
    end
    return Phi, phi, B
end

"""
    sy_loglike(shares, prices, budget, params; quaids=false, demographics=nothing,
               price_index=:translog, Phi, phi, parallel=true) -> Vector{Float64}

Step 2 of Shonkwiler–Yen: per-household Gaussian log-likelihood of the censoring-corrected system
`w_g = Φ̂_g·w̄_g(θ) + δ_g·φ̂_g + ξ_g` over the first `m−1` goods, fit on **every** household.

- `params` : `[θ…, δ (m−1), σ (upper-tri of R, Σ = R'R)]` — δ spliced between the share and Σ blocks.
- `Phi, phi` : the n×(m−1) first-stage matrices from [`sy_first_stage`](@ref).

Other arguments mirror [`naive_loglike`](@ref). With `Phi .== 1`, `phi .== 0` and `δ .== 0` this
equals `naive_loglike` exactly.
"""
function sy_loglike(shares::AbstractMatrix, prices::AbstractMatrix,
                    budget::AbstractVector, params::AbstractVector;
                    quaids::Bool = false, demographics = nothing,
                    price_index = :translog,
                    Phi::AbstractMatrix, phi::AbstractMatrix,
                    parallel::Bool = true)::Vector{Float64}

    price_index = _price_index_sym(price_index)
    P = Matrix{Float64}(prices)
    n, m = size(P)
    S = Matrix{Float64}(shares)
    p = Vector{Float64}(params)
    j  = Int(0.5 * (m - 1) * m)                     # number of sigma params
    nd = m - 1                                      # number of δ params
    delta = p[(end - j - nd + 1):(end - j)]
    theta = p[1:(end - j - nd)]

    # ----: predicted shares (same call the other likelihoods use) :----
    U = aids_shares(P, budget, theta;
                    quaids = quaids, demographics = demographics,
                    price_index = price_index, shares = S)          # n × m

    # ----: Sigma (covariance of the m-1 errors) — identical unpack to loglike.jl / naive.jl :----
    sig_params = p[(end - j + 1):end]
    R = zeros(Float64, m - 1, m - 1)
    k = 1
    for jj in 1:(m - 1)
        for ii in 1:jj
            R[ii, jj] = sig_params[k]
            k += 1
        end
    end
    Sigma = R' * R
    mvn = MvNormal(zeros(m - 1), Symmetric(Sigma))

    lf = zeros(Float64, n)
    # Gaussian density of the SY residual w − (Φ̂·w̄(θ) + δ·φ̂) for EVERY household.
    function lf_of(i)
        e = [S[i, g] - (Phi[i, g] * U[i, g] + delta[g] * phi[i, g]) for g in 1:(m - 1)]
        return logpdf(mvn, e)
    end
    if parallel
        Threads.@threads for i in 1:n
            lf[i] = lf_of(i)
        end
    else
        for i in 1:n
            lf[i] = lf_of(i)
        end
    end
    return lf
end
