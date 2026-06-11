# Naive-CENSORED (per-equation Tobit) log-likelihood for the AIDS / QUAIDS demand system.
#
# This is the estimator a practitioner reaches for when they recognise the zeros but treat them
# as a STATISTICAL CENSORING problem rather than as the genuine CORNER SOLUTIONS they are. The
# Tobit "we do not observe the data" story: each share has a latent value `w*_ig = U_ig + ε_ig`,
# and a zero is read as a latent POSITIVE demand that was censored at 0 — so a purchased good
# contributes a normal density and a zero contributes the censored mass `Φ(-U_ig/σ_g)`.
#
# It is WRONG here because households do not have hidden positive demand for the goods they skip:
# they deliberately choose the corner and REALLOCATE that budget across the goods they do buy.
# The correct (Wales–Woodland / Amemiya–Tobin) likelihood in `loglike.jl` encodes exactly that;
# this one does not. The three concrete departures from `censored_loglike`:
#   1. zeros get a univariate-normal CDF "censored mass" term instead of the corner-solution
#      orthant probability;
#   2. NO reallocation — the predicted shares `U_ig` are used as-is, never renormalised after
#      censoring (so the implied shares need not re-sum to 1);
#   3. goods are scored INDEPENDENTLY using only the marginal variances `σ_g² = Σ_gg` — the
#      off-diagonal error correlations in Σ are ignored (the "naive" part — per-equation Tobit).
#
# Same parameter layout, Σ-unpacking convention, and threading as `naive_loglike` /
# `censored_loglike`, so `estimate(...; loglike = :naive_censored)` swaps it in with no other change.
#
# Includable INTO the CensoredDemand module: assumes `aids_shares` is defined and the package
# `using`s (Distributions, LinearAlgebra, Random) are loaded.

using LinearAlgebra, Distributions

"""
    naive_censored_loglike(shares, prices, budget, params; quaids=false, demographics=nothing,
                           price_index=:translog, parallel=true) -> Vector{Float64}

Per-equation Tobit ("naive censored") log-likelihood: each of the `m-1` free goods is scored with
a univariate censored-normal likelihood — the normal density of the residual when the good is
bought, and the censored mass `Φ(-U_ig/σ_g)` when its share is zero — summed over goods within a
household. The zeros are treated as latent positive demand censored at 0 (the Tobit "unobserved
data" interpretation), **with no Wales–Woodland reallocation and ignoring the off-diagonal error
correlations**. This is the deliberately misspecified alternative to [`censored_loglike`] whose
bias the Monte-Carlo study measures.

Arguments mirror [`naive_loglike`](@ref) / [`censored_loglike`](@ref):
- `shares`        : n×m raw budget shares (zeros mark non-purchase / censoring).
- `prices`        : n×m LOGGED prices.
- `budget`        : length-n LOGGED total expenditure.
- `params`        : FULL parameter vector `[share-params..., sigma-params]`; the last
                    `j = 0.5*(m-1)*m` entries are Σ (column-major upper-tri of R, Σ = R'R).
- `quaids`        : include the quadratic (QUAIDS) term.
- `demographics`  : `nothing` or an n×t demographic matrix.
- `price_index`   : `:translog` (default) or `:stone` (needs observed `shares`, passed through).
- `parallel`      : thread the per-household loop (default true); result is order-independent.

Returns a length-n vector of per-household log-likelihood contributions.
"""
function naive_censored_loglike(shares::AbstractMatrix, prices::AbstractMatrix,
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

    # ----: predicted shares (same call the censored / naive likelihoods use) :----
    U = aids_shares(P, budget, p[1:(end - j)];
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
    Sigma = R' * R                                  # (m-1)×(m-1)
    sg = sqrt.(clamp.(diag(Sigma), 0.0, Inf))       # marginal sd per free good (off-diagonals dropped)

    lf = zeros(Float64, n)
    # Per-equation Tobit: density for goods bought, censored mass for zeros, summed over m-1 goods.
    # `logpdf`/`logcdf` of a univariate Normal are numerically stable (no ad-hoc floor needed).
    function lf_of(i)
        acc = 0.0
        for g in 1:(m - 1)
            dist = Normal(U[i, g], sg[g])
            acc += S[i, g] != 0.0 ? logpdf(dist, S[i, g]) : logcdf(dist, 0.0)
        end
        return acc
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
