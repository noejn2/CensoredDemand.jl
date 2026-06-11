# Naive (UNCENSORED) log-likelihood for the AIDS / QUAIDS demand system.
#
# This is the estimator a practitioner reaches for when they IGNORE censoring: every
# household is scored with the plain multivariate-normal density of the residual
# e = w_obs[1:m-1] - w_pred[1:m-1], exactly as the censored likelihood scores its
# FULL-purchase regime (loglike.jl) — but applied to EVERY household, including the
# censored ones whose observed zero shares are treated as if they were latent shares.
#
# It deliberately drops the Wales–Woodland correction: no micro-regimes, no orthant
# probabilities, no log-floor. The DGP is still the truncated reality (shares in [0,1]);
# the naive estimator simply uses the wrong likelihood to estimate it. That mismatch is
# what the Monte-Carlo study measures.
#
# Same parameter layout, Σ-unpacking convention, and threading as `censored_loglike`, so
# `estimate(...; loglike = :naive)` can swap one for the other with no other change.
#
# Includable INTO the CensoredDemand module: assumes `aids_shares` is defined and the
# package `using`s (Distributions, LinearAlgebra, Random) are loaded.

"""
    naive_loglike(shares, prices, budget, params; quaids=false, demographics=nothing,
                  price_index=:translog, parallel=true) -> Vector{Float64}

Uncensored ("naive") per-household log-likelihood: the multivariate-normal density of the
`m-1` share residuals for **every** household, ignoring the censoring of zero shares.

Arguments mirror [`censored_loglike`](@ref):
- `shares`        : n×m raw budget shares (zeros are treated as ordinary values here).
- `prices`        : n×m LOGGED prices.
- `budget`        : length-n LOGGED total expenditure.
- `params`        : FULL parameter vector `[share-params..., sigma-params]`; the last
                    `j = 0.5*(m-1)*m` entries are Σ (column-major upper-tri of R, Σ = R'R).
- `quaids`        : include the quadratic (QUAIDS) term.
- `demographics`  : `nothing` or an n×t demographic matrix.
- `price_index`   : `:translog` (default) or `:stone` (needs observed `shares`, passed through).
- `parallel`      : thread the per-household loop (default true); result is order-independent.

Returns a length-n vector of per-household log-likelihood contributions. On a dataset with
**no censoring** (all shares strictly positive) this equals `censored_loglike` to machine
precision, since every household then falls in the full-purchase regime.
"""
function naive_loglike(shares::AbstractMatrix, prices::AbstractMatrix,
                       budget::AbstractVector, params::AbstractVector;
                       quaids::Bool = false, demographics = nothing,
                       price_index = :translog,
                       parallel::Bool = true)::Vector{Float64}

    price_index = _price_index_sym(price_index)    # accept Symbol or PriceIndex enum
    P = Matrix{Float64}(prices)
    n, m = size(P)
    S = Matrix{Float64}(shares)
    p = Vector{Float64}(params)
    j = Int(0.5 * (m - 1) * m)                      # number of sigma params

    # ----: predicted shares (same call the censored likelihood uses) :----
    U = aids_shares(P, budget, p[1:(end - j)];
                    quaids = quaids, demographics = demographics,
                    price_index = price_index, shares = S)          # n × m

    # ----: Sigma (covariance of the m-1 errors) — identical unpack to loglike.jl :----
    sig_params = p[(end - j + 1):end]
    R = zeros(Float64, m - 1, m - 1)
    k = 1
    for jj in 1:(m - 1)
        for ii in 1:jj
            R[ii, jj] = sig_params[k]
            k += 1
        end
    end
    Sigma = R' * R                                  # (m-1)×(m-1)
    mvn = MvNormal(zeros(m - 1), Symmetric(Sigma))

    lf = zeros(Float64, n)
    # No regimes, no orthant probabilities: the plain Gaussian density for EVERY household.
    if parallel
        Threads.@threads for i in 1:n
            e = @views S[i, 1:(m - 1)] .- U[i, 1:(m - 1)]
            lf[i] = logpdf(mvn, e)
        end
    else
        for i in 1:n
            e = @views S[i, 1:(m - 1)] .- U[i, 1:(m - 1)]
            lf[i] = logpdf(mvn, e)
        end
    end
    return lf
end
