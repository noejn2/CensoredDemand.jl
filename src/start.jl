# Starting values for the censored AIDS/QUAIDS MLE, and an appropriateness check.
#
# Good starts are the practical key to convergence of the censored likelihood (an
# off-the-shelf optimizer from an arbitrary start does NOT converge — see PLAN.md /
# bench). `initial_values` builds a principled Linear-Approximate AIDS (LA-AIDS)
# start; `check_start` verifies a start is viable before burning thousands of
# iterations on it.

using LinearAlgebra, Statistics

"""
    initial_values(shares, prices, budget; quaids=false, demographics=nothing) -> Vector{Float64}

Principled starting values via a **Stone-index Linear-Approximate AIDS (LA-AIDS)** regression.

For each of the first `m-1` share equations it runs an OLS of the observed budget share on
homogeneity-imposed log-price ratios `(ln pⱼ − ln pₘ)`, the real-expenditure term
`(ln x − ln P*)` with the Stone index `ln P* = Σ wₖ ln pₖ`, and (if present) the demographic
interactions `Zₖ·(ln x − ln P*)`. Γ is then symmetrized (Slutsky), Σ is taken from the residual
covariance (as its Cholesky factor), and the QUAIDS λ is initialized at 0. The result is packed
in the exact order `aids_shares`/`censored_loglike` expect.

This is far better than a flat/arbitrary start and is the standard way AIDS/QUAIDS is initialized
in practice; the censored MLE then corrects the censoring bias from this neighbourhood.
"""
function initial_values(shares::AbstractMatrix, prices::AbstractMatrix, budget::AbstractVector;
                        quaids::Bool = false, demographics = nothing)
    W   = Matrix{Float64}(shares)
    Lnp = Matrix{Float64}(prices)
    Lnw = Vector{Float64}(budget)
    n, m = size(Lnp)
    has_dems = demographics !== nothing
    Z = has_dems ? Matrix{Float64}(demographics) : zeros(n, 0)
    t = size(Z, 2)

    # Stone price index (observed shares) and the real-expenditure term.
    lnPstar = vec(sum(W .* Lnp, dims = 2))            # length n
    expterm = Lnw .- lnPstar                          # length n

    # Homogeneity-imposed price regressors (ln pⱼ − ln pₘ), j = 1..m-1.
    prdiff = Lnp[:, 1:(m - 1)] .- Lnp[:, m]           # n × (m-1)

    # Per-equation design matrix: [1 | price-diffs | expterm | Z.*expterm].
    Xcols = Any[ones(n), prdiff, expterm]
    has_dems && push!(Xcols, Z .* expterm)            # n × t
    X = hcat(Xcols...)
    βcol = 1 + (m - 1) + 1                             # column index of the expterm (β) coefficient

    alpha = zeros(m - 1)
    beta  = zeros(m - 1)
    Gam   = zeros(m - 1, m - 1)                        # Gam[i, j] = γᵢⱼ
    Theta = zeros(t, m - 1)                            # Theta[k, i] = θ for demo k, good i
    resid = zeros(n, m - 1)
    for i in 1:(m - 1)
        coef = X \ W[:, i]                             # OLS
        alpha[i]   = coef[1]
        Gam[i, :]  = coef[2:(1 + (m - 1))]
        beta[i]    = coef[βcol]
        has_dems && (Theta[:, i] = coef[(βcol + 1):(βcol + t)])
        resid[:, i] = W[:, i] .- X * coef
    end

    # Slutsky symmetry imposed on the start.
    Gam = (Gam .+ Gam') ./ 2

    # Σ from residual covariance, as its Cholesky upper factor R (Σ = R'R), PD-guarded.
    Sig = cov(resid)
    Sig = (Sig .+ Sig') ./ 2
    mineig = minimum(eigen(Symmetric(Sig)).values)
    mineig <= 0 && (Sig += (abs(mineig) + 1e-6) * I)
    R = Matrix(cholesky(Symmetric(Sig)).U)            # R'R = Σ, upper-triangular

    # ---- pack in model order: α, β, γ(uptri), θ(if dems), λ(if quaids), σ(uptri of R) ----
    params = Float64[]
    append!(params, alpha)                            # m-1
    append!(params, beta)                             # m-1
    for j in 1:(m - 1), i in 1:j                       # γ upper-tri incl diag, COLUMN-MAJOR
        push!(params, Gam[i, j])
    end
    if has_dems                                        # θ: column-major over t×(m-1) (good outer, demo inner)
        for i in 1:(m - 1), k in 1:t
            push!(params, Theta[k, i])
        end
    end
    quaids && append!(params, zeros(m - 1))            # λ ≈ 0
    for j in 1:(m - 1), i in 1:j                       # σ = upper-tri of R, COLUMN-MAJOR
        push!(params, R[i, j])
    end
    return params
end

"""
    check_start(start, shares, prices, budget; quaids=false, demographics=nothing) -> NamedTuple

Decide whether `start` is an appropriate starting vector. Returns
`(ok::Bool, issues::Vector{String}, diagnostics::NamedTuple)` checking: parameter count matches
the model `(m, t, quaids)`; the implied Σ is positive-definite and not ill-conditioned; the predicted
shares are finite; and the censored log-likelihood is finite at every household (none pinned at −∞).
"""
function check_start(start::AbstractVector, shares::AbstractMatrix, prices::AbstractMatrix,
                     budget::AbstractVector; quaids::Bool = false, demographics = nothing)
    issues = String[]
    P = Matrix{Float64}(prices)
    n, m = size(P)
    t = demographics === nothing ? 0 : size(demographics, 2)
    j = Int(0.5 * (m - 1) * m)

    nshare = (m - 1) + (m - 1) + j +
             (demographics === nothing ? 0 : (m - 1) * t) +
             (quaids ? (m - 1) : 0)
    expected = nshare + j
    count_ok = length(start) == expected
    count_ok || push!(issues, "parameter count $(length(start)) ≠ expected $expected for (m=$m,t=$t,quaids=$quaids)")

    sigma_min_eig = NaN; sigma_cond = NaN; sigma_pd = false
    shares_finite = false; loglike = NaN; n_nonfinite = -1
    if count_ok
        # Σ = R'R from the trailing j params.
        sp = start[(end - j + 1):end]
        R = zeros(m - 1, m - 1); k = 1
        for jj in 1:(m - 1), ii in 1:jj
            R[ii, jj] = sp[k]; k += 1
        end
        Sig = R'R
        ev = eigen(Symmetric(Sig)).values
        sigma_min_eig = minimum(ev)
        sigma_cond = maximum(ev) / max(minimum(ev), eps())
        sigma_pd = sigma_min_eig > 0
        sigma_pd || push!(issues, "Σ is not positive-definite (min eigenvalue $sigma_min_eig)")
        (sigma_pd && sigma_cond > 1e10) && push!(issues, "Σ is ill-conditioned (cond ≈ $(round(sigma_cond, sigdigits=3)))")

        U = aids_shares(P, budget, start[1:(end - j)]; quaids = quaids, demographics = demographics)
        shares_finite = all(isfinite, U)
        shares_finite || push!(issues, "predicted shares contain NaN/Inf")

        ll = censored_loglike(Matrix{Float64}(shares), P, budget, start;
                              quaids = quaids, demographics = demographics, mc_points = 2000)
        n_nonfinite = count(!isfinite, ll)
        n_nonfinite == 0 || push!(issues, "$n_nonfinite household(s) have non-finite log-likelihood")
        loglike = sum(x -> isfinite(x) ? x : 0.0, ll)
    end

    ok = count_ok && sigma_pd && shares_finite && (n_nonfinite == 0)
    return (ok = ok, issues = issues,
            diagnostics = (param_count = length(start), expected_count = expected,
                           sigma_min_eig = sigma_min_eig, sigma_cond = sigma_cond,
                           shares_finite = shares_finite, n_loglike_nonfinite = n_nonfinite,
                           loglike = loglike))
end
