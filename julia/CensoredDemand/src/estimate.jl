# Implementation A: MAXIMUM-LIKELIHOOD ESTIMATION entry point for the censored
# AIDS / QUAIDS demand system.
#
# Minimizes the summed NEGATIVE Wales–Woodland censored log-likelihood
# (`censored_loglike`, defined in loglike.jl) over the full parameter vector
# `theta = [share-params..., sigma-params]`.
#
# Why a derivative-free / finite-difference optimizer? `censored_loglike` calls
# `MvNormalCDF.mvnormcdf` for the partial-censoring regimes. With a FIXED rng
# seed (the loglike default) the same QMC draws are reused every evaluation, so
# the objective is DETERMINISTIC — but it still carries tiny QMC-induced
# roughness that breaks exact analytic/AD gradients. We therefore default to
# NelderMead (derivative-free) and compute the variance–covariance matrix from a
# FINITE-DIFFERENCE numerical Hessian of the negative log-likelihood at the
# optimum (ForwardDiff cannot differentiate cleanly through mvnormcdf).
#
# Includable standalone (no module wrapper). Assumes `aids_shares` and
# `censored_loglike` are already defined and in scope; does NOT redefine them.

using LinearAlgebra, Optim, Random

# ----------------------------------------------------------------------------
# Default starting values
# ----------------------------------------------------------------------------

"""
    _default_start(m, t; quaids) -> Vector{Float64}

Build a sensible (but generic) starting vector for the full parameter
`theta = [alpha(m-1), beta(m-1), gamma(0.5*(m-1)*m), theta(t*(m-1) if dems),
          lambda(m-1) if quaids, sigma(0.5*(m-1)*m)]`.

The defaults are deliberately mild:
  - `alpha` : equal split `1/m` for each of the first m-1 goods (a neutral guess
              consistent with the adding-up reconstruction in `aids_shares`).
  - `beta`  : small zeros (no expenditure response a priori).
  - `gamma` : near-zero (no a-priori price response).
  - `theta` : zeros (no a-priori demographic response).
  - `lambda`: zeros (start from the linear-AIDS nest).
  - `sigma` : a small positive diagonal Cholesky factor (0.1 on the diagonal,
              giving an error covariance ≈ 0.01·I for the m-1 free shares).

NOTE: good starting values matter a great deal for censored MLE — the
likelihood surface is non-convex and the QMC orthant probabilities are only
mildly smooth. Whenever an external estimate or a warm start is available,
pass it via `start=`; the generic defaults below are only a fallback.
"""
function _default_start(m::Integer, t::Integer; quaids::Bool)
    nshare = 2 * (m - 1) + Int(0.5 * (m - 1) * m)        # alpha + beta + gamma
    nshare += t * (m - 1)                                # theta
    if quaids
        nshare += (m - 1)                               # lambda
    end
    nsig = Int(0.5 * (m - 1) * m)

    theta = zeros(Float64, nshare + nsig)

    # alpha = 1/m for the first m-1 goods
    theta[1:(m - 1)] .= 1.0 / m
    # beta, gamma, theta, lambda left at (near) zero

    # sigma = small diagonal Cholesky factor (column-major upper-tri incl. diag)
    sig = @view theta[(nshare + 1):(nshare + nsig)]
    k = 1
    for jj in 1:(m - 1)
        for ii in 1:jj
            sig[k] = (ii == jj) ? 0.1 : 0.0             # diagonal = 0.1
            k += 1
        end
    end
    return theta
end

# ----------------------------------------------------------------------------
# Finite-difference symmetric Hessian
# ----------------------------------------------------------------------------

"""
    _fd_hessian(f, x; h_rel=1e-4, h_abs=1e-5) -> Matrix

Central-difference numerical Hessian of a scalar function `f` at `x`, computed
symmetrically. Step per coordinate is `max(h_rel*|x_i|, h_abs)`. Used for the
negative-log-likelihood Hessian when AD is unavailable (mvnormcdf). The result
is explicitly re-symmetrized.
"""
function _fd_hessian(f, x::AbstractVector; h_rel::Real = 1e-4, h_abs::Real = 1e-5)
    n = length(x)
    x = collect(float.(x))
    h = [max(h_rel * abs(x[i]), h_abs) for i in 1:n]
    H = zeros(Float64, n, n)

    f0 = f(x)

    # diagonal: central second difference
    @inbounds for i in 1:n
        xp = copy(x); xp[i] += h[i]
        xm = copy(x); xm[i] -= h[i]
        H[i, i] = (f(xp) - 2.0 * f0 + f(xm)) / (h[i]^2)
    end

    # off-diagonal: standard 4-point central scheme
    @inbounds for i in 1:n
        for jcol in (i + 1):n
            xpp = copy(x); xpp[i] += h[i]; xpp[jcol] += h[jcol]
            xpm = copy(x); xpm[i] += h[i]; xpm[jcol] -= h[jcol]
            xmp = copy(x); xmp[i] -= h[i]; xmp[jcol] += h[jcol]
            xmm = copy(x); xmm[i] -= h[i]; xmm[jcol] -= h[jcol]
            val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[jcol])
            H[i, jcol] = val
            H[jcol, i] = val
        end
    end

    # symmetrize
    H .= 0.5 .* (H .+ H')
    return H
end

"""
    _project_psd(M; floor=1e-10) -> (Matrix, Bool)

Symmetrize `M` and, if it is not positive semidefinite, project it onto the
nearest PSD matrix by clipping eigenvalues to `floor`. Returns the (possibly
projected) matrix and a flag indicating whether projection was needed.
"""
function _project_psd(M::AbstractMatrix; floor::Real = 1e-10)
    S = Symmetric((M .+ M') ./ 2)
    if isposdef(S)
        return Matrix(S), false
    end
    F = eigen(S)
    vals = max.(F.values, floor)
    P = F.vectors * Diagonal(vals) * F.vectors'
    P = (P .+ P') ./ 2
    return Matrix(P), true
end

# ----------------------------------------------------------------------------
# Estimation entry point
# ----------------------------------------------------------------------------

"""
    estimate(shares, prices, budget; quaids=false, demographics=nothing,
             start=nothing, mc_points=2000, optimizer=NelderMead(),
             maxiters=2000, g_tol=1e-6, x_tol=1e-8, f_tol=1e-10,
             show_trace=false, hess_h_rel=1e-4, hess_h_abs=1e-5) -> NamedTuple

Maximum-likelihood estimation of the censored AIDS / QUAIDS demand system.

Minimizes the summed NEGATIVE censored log-likelihood
`theta -> -sum(censored_loglike(shares, prices, budget, theta; quaids,
demographics, mc_points))` with `Optim.jl`. The objective is deterministic
(fixed QMC seed inside `censored_loglike`).

# Arguments
- `shares`       : n×m matrix of raw budget shares (zeros mark censoring).
- `prices`       : n×m matrix of LOGGED prices.
- `budget`       : length-n vector of LOGGED total expenditure.
- `quaids`       : include the quadratic term if true.
- `demographics` : `nothing` or an n×t demographic matrix.
- `start`        : full parameter vector to warm-start from. If `nothing`,
                   `_default_start` builds a generic guess (good starts matter!).
- `mc_points`    : QMC sample budget passed to `censored_loglike`.
- `optimizer`    : an `Optim.jl` optimizer (default `NelderMead()`; `LBFGS()`
                   with finite-difference gradients is also fine).
- `maxiters`     : maximum optimizer iterations.
- `g_tol`,`x_tol`,`f_tol` : Optim convergence tolerances (modest by default to
                   tolerate QMC roughness).

# Returns
A `NamedTuple` with at least:
- `params`     : the estimated full parameter vector.
- `loglike`    : the maximized log-likelihood (`+sum`, i.e. `-minimum`).
- `vcov`       : variance–covariance matrix = `inv(Hessian)` of the NEGATIVE
                 log-likelihood at the optimum (symmetrized; PSD-projected if
                 needed — see `vcov_projected`).
- `converged`  : `Optim.converged(result)`.
- `iterations` : iterations taken.
Plus `se` (standard errors), `hessian`, `vcov_projected`, `nll`, `optimizer`,
and the raw `result`.
"""
function estimate(shares::AbstractMatrix, prices::AbstractMatrix,
                  budget::AbstractVector;
                  quaids::Bool = false, demographics = nothing,
                  start = nothing, mc_points::Integer = 2000,
                  optimizer = NelderMead(),
                  maxiters::Integer = 2000,
                  g_tol::Real = 1e-6, x_tol::Real = 1e-8, f_tol::Real = 1e-10,
                  show_trace::Bool = false,
                  hess_h_rel::Real = 1e-4, hess_h_abs::Real = 1e-5,
                  floor_mode::Symbol = :additive_r)

    P = Matrix{Float64}(prices)
    n, m = size(P)
    t = demographics === nothing ? 0 : size(demographics, 2)

    # --- deterministic negative log-likelihood objective ---
    # censored_loglike's default rng is a FIXED seed => same QMC draws each call.
    nll(theta) = -sum(censored_loglike(shares, P, budget, theta;
                                       quaids = quaids,
                                       demographics = demographics,
                                       mc_points = mc_points,
                                       floor_mode = floor_mode))

    # --- starting values ---
    theta0 = start === nothing ? _default_start(m, t; quaids = quaids) :
                                 collect(float.(start))

    # --- optimize ---
    # Use the current (non-deprecated) Optim.Options keyword names.
    opts = Optim.Options(iterations = Int(maxiters),
                         g_abstol = g_tol, x_abstol = x_tol, f_reltol = f_tol,
                         show_trace = show_trace)

    result = Optim.optimize(nll, theta0, optimizer, opts)

    theta_hat = Optim.minimizer(result)
    nll_min   = Optim.minimum(result)
    loglike   = -nll_min

    # --- variance–covariance: numerical Hessian of the NEGATIVE log-lik ---
    H = _fd_hessian(nll, theta_hat; h_rel = hess_h_rel, h_abs = hess_h_abs)
    H = Symmetric((H .+ H') ./ 2)

    local vcov::Matrix{Float64}
    vcov_projected = false
    try
        vcov = inv(Matrix(H))
    catch
        # singular Hessian -> pseudo-inverse
        vcov = pinv(Matrix(H))
        vcov_projected = true
    end

    # vcov must itself be a valid covariance: symmetrize, project to PSD if needed
    vcov, projected = _project_psd(vcov)
    vcov_projected = vcov_projected || projected

    se = sqrt.(clamp.(diag(vcov), 0.0, Inf))

    return (params      = theta_hat,
            loglike     = loglike,
            vcov        = vcov,
            se          = se,
            hessian     = Matrix(H),
            vcov_projected = vcov_projected,
            converged   = Optim.converged(result),
            iterations  = Optim.iterations(result),
            nll         = nll_min,
            optimizer   = optimizer,
            result      = result)
end
