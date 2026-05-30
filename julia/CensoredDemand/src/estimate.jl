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

# Gauss–Newton / BHHH optimizer driven by the per-observation score matrix — the gradient-based
# routine the original GAUSS code used. `ll_vec(theta)` returns the length-n vector of per-household
# log-likelihoods. Direction is (S'S)⁻¹ g, where g = Σᵢ sᵢ and S is the n×p score matrix (central
# finite differences of the per-obs likelihoods — clean because the per-obs RNG is fixed). A
# step-halving line search guarantees the summed log-likelihood increases. Convergence: mean
# |gradient| per observation < gtol. Covariance is the OPG estimator (S'S)⁻¹ (no separate Hessian).
function _bhhh(ll_vec, theta0::AbstractVector; maxiters::Integer, gtol::Real,
               fdstep::Real, line_halvings::Integer = 40, show_trace::Bool = false)
    theta = collect(float.(theta0))
    p = length(theta)
    n = length(ll_vec(theta))
    Smat = zeros(n, p)
    score!(th) = begin
        for k in 1:p
            h = fdstep * max(abs(th[k]), 1.0)
            tp = copy(th); tp[k] += h
            tm = copy(th); tm[k] -= h
            Smat[:, k] = (ll_vec(tp) .- ll_vec(tm)) ./ (2h)
        end
        return vec(sum(Smat, dims = 1))                 # gradient g = Σᵢ sᵢ
    end

    g = zeros(p)
    iters = 0
    converged = false
    for it in 1:maxiters
        iters = it
        f0 = sum(ll_vec(theta))
        g = score!(theta)
        gnorm = sum(abs, g) / n
        show_trace && println("bhhh it=$it  loglik=$(round(f0, digits = 3))  mean|g|/n=$(round(gnorm, sigdigits = 3))")
        if gnorm < gtol
            converged = true
            break
        end
        OPGm = Smat' * Smat
        d = try
            OPGm \ g
        catch
            pinv(OPGm) * g
        end
        step = 1.0
        improved = false
        for _ in 1:line_halvings                         # step-halving line search
            cand = theta .+ step .* d
            fc = sum(ll_vec(cand))
            if isfinite(fc) && fc > f0
                theta = cand
                improved = true
                break
            end
            step /= 2
        end
        improved || break                                # no improving step → stop
    end

    g = score!(theta)                                    # final score at the estimate
    OPG = Smat' * Smat
    vcov = try
        inv(OPG)
    catch
        pinv(OPG)
    end
    return (params = theta, loglike = sum(ll_vec(theta)), opg = OPG, vcov = vcov,
            gradient = g, gradnorm = sum(abs, g) / n,
            converged = converged, iterations = iters)
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
- `start`        : full parameter vector to warm-start from. If `nothing`, a principled
                   LA-AIDS start (`initial_values`) is built and validated (`check_start`).
- `mc_points`    : QMC sample budget passed to `censored_loglike`.
- `algorithm`    : `:neldermead` (default, derivative-free, via Optim) or `:bhhh` — the
                   gradient-based Gauss–Newton/BHHH step on the per-observation scores, with
                   the OPG covariance `(S'S)⁻¹` (converges in far fewer iterations).
- `price_index`  : `:translog` (default, full QUAIDS index) or `:stone` (LA-AIDS Stone index —
                   predetermined, so it linearizes the share equations and eases estimation).
- `optimizer`    : the `Optim.jl` optimizer used when `algorithm = :neldermead` (default
                   `NelderMead()`; `LBFGS()` etc. also fine).
- `maxiters`     : maximum optimizer iterations.
- `g_tol`,`x_tol`,`f_tol` : convergence tolerances. For `:bhhh`, `g_tol` is the mean
                   |gradient|-per-observation threshold.
- `parallel`     : thread the per-observation likelihood loop (default true).
- `check`        : run `check_start` on the start and warn if inappropriate (default true).

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
                  floor_mode::Symbol = :additive_r,
                  parallel::Bool = true,
                  check::Bool = true,
                  algorithm::Symbol = :neldermead,
                  price_index::Symbol = :translog)

    P = Matrix{Float64}(prices)
    n, m = size(P)
    t = demographics === nothing ? 0 : size(demographics, 2)

    # --- deterministic objectives (per-obs vector + summed) ---
    # censored_loglike's default rng is a FIXED seed => same QMC draws each call.
    ll_vec(theta) = censored_loglike(shares, P, budget, theta;
                                     quaids = quaids, demographics = demographics,
                                     mc_points = mc_points, floor_mode = floor_mode,
                                     parallel = parallel, price_index = price_index)
    nll(theta) = -sum(ll_vec(theta))

    # --- starting values ---
    # Default to a principled LA-AIDS start; verify any start is viable before optimizing.
    theta0 = start === nothing ?
             initial_values(shares, P, budget; quaids = quaids, demographics = demographics) :
             collect(float.(start))
    start_check = check_start(theta0, shares, P, budget;
                              quaids = quaids, demographics = demographics)
    if check && !start_check.ok
        @warn "estimate: starting values may be inappropriate" issues = start_check.issues
    end

    # --- optimize ---
    algorithm in (:neldermead, :bhhh) ||
        error("estimate: algorithm must be :neldermead or :bhhh " *
              "(use the `optimizer` kwarg for other Optim methods)")

    if algorithm === :bhhh
        # Gradient-based Gauss–Newton/BHHH (the original GAUSS routine): vcov is the OPG
        # covariance (S'S)⁻¹, built from the per-observation scores — no separate Hessian.
        bh = _bhhh(ll_vec, theta0; maxiters = Int(maxiters), gtol = g_tol,
                   fdstep = hess_h_rel, show_trace = show_trace)
        vcov, projected = _project_psd(bh.vcov)
        return (params      = bh.params,
                loglike     = bh.loglike,
                vcov        = vcov,
                se          = sqrt.(clamp.(diag(vcov), 0.0, Inf)),
                hessian     = bh.opg,
                vcov_projected = projected,
                converged   = bh.converged,
                iterations  = bh.iterations,
                nll         = -bh.loglike,
                optimizer   = :bhhh,
                start_check = start_check,
                result      = bh)
    end

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
            start_check = start_check,
            result      = result)
end
