# Implementation A: MAXIMUM-LIKELIHOOD ESTIMATION entry point for the censored
# AIDS / QUAIDS demand system.
#
# Maximizes the summed Wales–Woodland censored log-likelihood (`censored_loglike`,
# defined in loglike.jl) over the full parameter vector
# `theta = [share-params..., sigma-params]`.
#
# Optimizer: BHHH / Gauss–Newton on the per-observation scores — the gradient-based
# routine the original GAUSS code used. `censored_loglike` calls `MvNormalCDF.mvnormcdf`
# for the partial-censoring regimes; with a FIXED rng seed (the loglike default) the same
# QMC draws are reused every evaluation, so the objective is DETERMINISTIC and a
# finite-difference score is clean. The step is (S'S)⁻¹g; the variance–covariance is the
# OPG estimator (S'S)⁻¹ — no separate Hessian, and no derivative-free fallback.
#
# Includable standalone (no module wrapper). Assumes `aids_shares` and
# `censored_loglike` are already defined and in scope; does NOT redefine them.

using LinearAlgebra, Random

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
# PSD projection for the covariance estimate
# ----------------------------------------------------------------------------

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
# log-likelihoods. Direction is (S'S)⁻¹ g, where g = Σᵢ sᵢ and S is the n×p score matrix (finite
# differences of the per-obs likelihoods — clean because the per-obs RNG is fixed). A step-halving
# line search guarantees the summed log-likelihood increases. Convergence: mean |gradient| per
# observation < gtol. Covariance is the OPG estimator (S'S)⁻¹ (no separate Hessian).
#
# `fd_mode` controls the per-iteration score: `:central` (default; 2p evals/iter, the most accurate)
# or `:forward` (p+1 evals/iter, ~2× fewer — uses the cached base likelihoods). The FINAL score and
# OPG covariance are ALWAYS computed with central differences for an accurate gradient/vcov.
function _bhhh(ll_vec, theta0::AbstractVector; maxiters::Integer, gtol::Real,
               fdstep::Real, line_halvings::Integer = 40, show_trace::Bool = false,
               fd_mode::Symbol = :central, free = nothing)
    theta = collect(float.(theta0))
    p = length(theta)
    # `free` is an optional length-p Bool mask of parameters to OPTIMIZE; the rest are held fixed at
    # `theta0`. `fi` = free indices. Default (free=nothing) optimizes all p and is byte-identical to
    # the unmasked path. The mask is what lets the per-equation Tobit (naive_censored) hold its
    # unidentified off-diagonal covariance entries at 0 (diagonal/independent covariance).
    fi = free === nothing ? collect(1:p) : findall(free)
    n = length(ll_vec(theta))
    Smat = zeros(n, p)
    # Build the score matrix at `th` over the FREE columns only (fixed columns stay 0); `lf_base =
    # ll_vec(th)` is reused by :forward. Returns the full-length gradient (0 in the fixed slots).
    score!(th, lf_base, mode) = begin
        for k in fi
            h = fdstep * max(abs(th[k]), 1.0)
            tp = copy(th); tp[k] += h
            if mode === :forward
                Smat[:, k] = (ll_vec(tp) .- lf_base) ./ h
            else
                tm = copy(th); tm[k] -= h
                Smat[:, k] = (ll_vec(tp) .- ll_vec(tm)) ./ (2h)
            end
        end
        return vec(sum(Smat, dims = 1))                 # gradient g = Σᵢ sᵢ (0 in fixed slots)
    end

    g = zeros(p)
    iters = 0
    converged = false
    stalled = false
    for it in 1:maxiters
        iters = it
        lf0 = ll_vec(theta)
        f0 = sum(lf0)
        g = score!(theta, lf0, fd_mode)
        gnorm = sum(abs, @view g[fi]) / n                # gradient norm over the FREE params only
        show_trace && println("bhhh it=$it  loglik=$(round(f0, digits = 3))  mean|g|/n=$(round(gnorm, sigdigits = 3))")
        if gnorm < gtol
            converged = true
            break
        end
        Sf   = @view Smat[:, fi]
        OPGf = Sf' * Sf                                  # OPG restricted to the free subspace
        df = try
            OPGf \ g[fi]
        catch
            pinv(OPGf) * g[fi]
        end
        d = zeros(p); d[fi] = df                         # full-length step (0 in fixed slots)
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
        if !improved
            stalled = true                               # no improving step → at a local optimum
            break
        end
    end

    g = score!(theta, ll_vec(theta), :central)           # final score: always central (accurate vcov)
    Sf    = @view Smat[:, fi]
    OPGf  = Sf' * Sf
    vcovf = try
        inv(OPGf)
    catch
        pinv(OPGf)
    end
    OPG  = Smat' * Smat                                   # full p×p OPG (fixed rows/cols are 0)
    vcov = zeros(p, p); vcov[fi, fi] = vcovf              # sampling variance only for estimated params
    # A line-search stall = no direction improves the likelihood = a local optimum, so report
    # convergence even when the (strict) gradient tolerance wasn't reached (e.g. the +1e-8 floor
    # makes the objective slightly non-smooth around floored households).
    converged = converged || stalled
    return (params = theta, loglike = sum(ll_vec(theta)), opg = OPG, vcov = vcov,
            gradient = g, gradnorm = sum(abs, @view g[fi]) / n,
            converged = converged, iterations = iters)
end

# ----------------------------------------------------------------------------
# Estimation entry point
# ----------------------------------------------------------------------------

"""
    estimate(shares, prices, budget; quaids=false, demographics=nothing,
             start=nothing, mc_points=2000, maxiters=2000, g_tol=1e-6,
             show_trace=false, fd_step=1e-4, floor_mode=:additive_r,
             parallel=true, check=true, price_index=:translog,
             share_names=nothing, demographic_names=nothing) -> EstimationResult

Maximum-likelihood estimation of the censored AIDS / QUAIDS demand system.

Maximizes the summed censored log-likelihood
`theta -> sum(censored_loglike(shares, prices, budget, theta; quaids, demographics,
mc_points))` with the **BHHH / Gauss–Newton** optimizer (the sole algorithm): a gradient
step `(S'S)⁻¹g` on the per-observation scores, with the OPG covariance `(S'S)⁻¹`. The
objective is deterministic (fixed QMC seed inside `censored_loglike`), so the
finite-difference score is clean.

# Arguments
- `shares`       : n×m matrix of raw budget shares (zeros mark censoring).
- `prices`       : n×m matrix of LOGGED prices.
- `budget`       : length-n vector of LOGGED total expenditure.
- `quaids`       : include the quadratic (QUAIDS) term if true.
- `demographics` : `nothing` or an n×t demographic matrix.
- `start`        : full parameter vector to warm-start from. If `nothing`, a principled
                   LA-AIDS start (`initial_values`) is built and validated (`check_start`).
- `mc_points`    : QMC sample budget passed to `censored_loglike`.
- `price_index`  : `:translog`/`TRANSLOG` (default, full QUAIDS index) or `:stone`/`STONE`
                   (LA-AIDS Stone index — predetermined, linearizes the share equations).
- `floor_mode`   : `:additive_r`/`ADDITIVE_R` (default, R-faithful) or `:guard`/`GUARD`.
- `loglike`      : `:censored` (default, Wales–Woodland corner-solution likelihood), `:naive`
                   (uncensored Gaussian — ignores truncation entirely), `:sy` (Shonkwiler–Yen
                   two-step: probit first stage, then the Gaussian system on the corrected mean
                   Φ̂·w̄(θ)+δ·φ̂; params gain a δ block — `[θ…, δ (m−1), σ]`; pass the first
                   stage via `sy_stage1` or let it be computed internally), or `:naive_censored`
                   (per-equation Tobit — treats zeros as censored latent demand, no reallocation).
                   The two `naive*` options are the misspecified estimators the MC study compares.
- `maxiters`     : maximum BHHH iterations.
- `g_tol`        : convergence tolerance — the mean |gradient|-per-observation threshold.
- `fd_step`      : relative finite-difference step for the per-observation scores.
- `fd_mode`      : `:central` (default, 2p evals/iter) or `:forward` (≈p+1 evals/iter, ~2× faster
                   per iteration; the final score + OPG vcov are always central). Use `:forward` to
                   speed up long runs.
- `parallel`     : thread the per-observation likelihood loop (default true).
- `free`         : optional length-`p` `Bool` mask of parameters to OPTIMIZE; the rest are held
                   fixed at `start`. `nothing` (default) estimates all parameters. Used to fit a
                   restricted model (e.g. the per-equation Tobit `:naive_censored`, which holds its
                   unidentified off-diagonal covariance entries at 0 — a diagonal/independent Σ).
                   Fixed parameters get zero rows/cols in `vcov` (zero SE).
- `check`        : run `check_start` on the start and warn if inappropriate (default true).
- `share_names`, `demographic_names` : optional labels for reporting (default `good1…`, `demo1…`).

# Returns
An `EstimationResult`: `params`, `loglike`, `vcov`, `se`, `opg`, `converged`, `iterations`,
`gradnorm`, `nll`, `optimizer` (`:bhhh`), `param_names`, `spec`, and `start_check`.
"""
function estimate(shares::AbstractMatrix, prices::AbstractMatrix,
                  budget::AbstractVector;
                  quaids::Bool = false, demographics = nothing,
                  start = nothing, mc_points::Integer = 2000,
                  maxiters::Integer = 2000,
                  g_tol::Real = 1e-6,
                  show_trace::Bool = false,
                  fd_step::Real = 1e-4,
                  fd_mode::Symbol = :central,
                  floor_mode = :additive_r,
                  parallel::Bool = true,
                  check::Bool = true,
                  price_index = :translog,
                  loglike::Symbol = :censored,
                  free = nothing,
                  sy_stage1 = nothing,
                  share_names = nothing, demographic_names = nothing)

    floor_mode  = _floor_mode_sym(floor_mode)      # accept Symbol or FloorMode enum
    price_index = _price_index_sym(price_index)    # accept Symbol or PriceIndex enum
    loglike in (:censored, :naive, :naive_censored, :sy) ||
        error("loglike must be :censored, :naive, :naive_censored, or :sy, got :$loglike")

    P = Matrix{Float64}(prices)
    n, m = size(P)
    t = demographics === nothing ? 0 : size(demographics, 2)

    # --- deterministic objective (per-obs vector); fixed QMC seed => same draws each call ---
    # `:censored` (default) = Wales–Woodland censored likelihood; `:naive` = the misspecified
    # uncensored Gaussian likelihood (ignores truncation); `:naive_censored` = the per-equation
    # Tobit likelihood (treats zeros as censored latent demand, no reallocation). All three return
    # a length-n per-obs vector, so the BHHH/OPG machinery below is identical either way.
    ll_vec = if loglike === :naive
        theta -> naive_loglike(shares, P, budget, theta;
                               quaids = quaids, demographics = demographics,
                               parallel = parallel, price_index = price_index)
    elseif loglike === :naive_censored
        theta -> naive_censored_loglike(shares, P, budget, theta;
                                        quaids = quaids, demographics = demographics,
                                        parallel = parallel, price_index = price_index)
    elseif loglike === :sy
        # Shonkwiler–Yen: the probit first stage is data, not parameters — computed once (or
        # supplied via `sy_stage1`) and held fixed through the step-2 optimization, exactly as
        # the two-step estimator is used in practice.
        Phi, phi = sy_stage1 === nothing ?
                   sy_first_stage(shares, P, budget, demographics) : sy_stage1
        theta -> sy_loglike(shares, P, budget, theta;
                            quaids = quaids, demographics = demographics,
                            Phi = Phi, phi = phi,
                            parallel = parallel, price_index = price_index)
    else
        theta -> censored_loglike(shares, P, budget, theta;
                                  quaids = quaids, demographics = demographics,
                                  mc_points = mc_points, floor_mode = floor_mode,
                                  parallel = parallel, price_index = price_index)
    end

    # --- starting values: principled LA-AIDS start; verify viability before optimizing ---
    # (:sy carries an extra δ block, so a default start splices δ = 0 into the LA-AIDS start and
    #  the standard-layout `check_start` is skipped.)
    theta0 = if start !== nothing
        collect(float.(start))
    elseif loglike === :sy
        iv = initial_values(shares, P, budget; quaids = quaids, demographics = demographics)
        j = Int(0.5 * (m - 1) * m)
        vcat(iv[1:(end - j)], zeros(m - 1), iv[(end - j + 1):end])
    else
        initial_values(shares, P, budget; quaids = quaids, demographics = demographics)
    end
    start_check = loglike === :sy ? (ok = true, issues = String[]) :
                  check_start(theta0, shares, P, budget;
                              quaids = quaids, demographics = demographics)
    if check && !start_check.ok
        @warn "estimate: starting values may be inappropriate" issues = start_check.issues
    end

    # --- optimize: BHHH / Gauss–Newton (sole algorithm); vcov is the OPG (S'S)⁻¹ ---
    bh = _bhhh(ll_vec, theta0; maxiters = Int(maxiters), gtol = g_tol,
               fdstep = fd_step, fd_mode = fd_mode, show_trace = show_trace, free = free)
    vcov, projected = _project_psd(bh.vcov)
    if free !== nothing
        # Held-fixed parameters have no sampling variance; the PSD projection smears a tiny floor
        # into their (zero) rows/cols, so restore the exact zeros.
        fixed = .!collect(free)
        vcov[fixed, :] .= 0.0
        vcov[:, fixed] .= 0.0
    end

    spec = ModelSpec(m, t; quaids = quaids, price_index = price_index,
                     share_names = share_names, demographic_names = demographic_names)

    return EstimationResult(bh.params, bh.loglike, vcov,
                            sqrt.(clamp.(diag(vcov), 0.0, Inf)),
                            bh.opg, projected, bh.converged, bh.iterations,
                            bh.gradnorm, -bh.loglike, :bhhh,
                            param_names(spec), spec, start_check)
end
