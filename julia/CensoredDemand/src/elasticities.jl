# Implementation A: simulation-based censored AIDS / QUAIDS demand elasticities,
# with delta-method standard errors.
#
# Faithful port of censoredAIDS::censoredElasticity (R). Mirrors the R algorithm
# exactly, including:
#   - reps Monte-Carlo draws of latent errors, delta = 1e-5 finite-difference step;
#   - evaluation point = column means (point = mean) of prices/budget/demographics;
#   - Sigma built from the LAST j = 0.5*(m-1)*m params; epsilons ~ N(0, Sigma) of
#     shape reps x (m-1), then cbind(epsilons, -rowSums) -> reps x m;
#   - Amemiya-Tobin truncation map treatTruncations: negatives -> 0, then each row
#     divided by its row sum;
#   - LEVEL perturbations for variables (exp, +delta, log) of the m prices then
#     the budget; per-parameter +delta perturbation for the SE Jacobian;
#   - income & price elasticity finite-difference formulas with the p1/p2 terms and
#     the diagonal -1 / -I corrections EXACTLY as R (including R's `/delta` term in
#     the SE p2 denominator);
#   - SE via delta method: J = (dy - f)/delta; etavcov = J' (vcov[1:vd,1:vd]/n) J.
#
# REUSES aids_shares (assumed in scope) for the point-share math: muaidsCalculate is
# just aids_shares evaluated at a single point (a 1 x m price matrix, length-1 budget,
# 1 x t demographics).
#
# Includable standalone (no module wrapper).

using LinearAlgebra
using Statistics
using Random

# Point-share helper: muaidsCalculate(muPrices, muBudget, muDemographics, Params, ...)
# evaluated at a single observation. Returns a length-m vector of shares.
#
# `params_model` excludes the trailing sigma block (R passes Params[-sigma]).
function _mu_shares(muPrices::AbstractVector, muBudget::Real,
                    muDemogs, params_model::AbstractVector;
                    quaids::Bool, has_dems::Bool)
    m = length(muPrices)
    P1 = reshape(Vector{Float64}(muPrices), 1, m)     # 1 x m price matrix
    b1 = [Float64(muBudget)]                          # length-1 budget
    if has_dems
        t = length(muDemogs)
        Z1 = reshape(Vector{Float64}(muDemogs), 1, t) # 1 x t demographics
        W = aids_shares(P1, b1, params_model; quaids = quaids, demographics = Z1)
    else
        W = aids_shares(P1, b1, params_model; quaids = quaids, demographics = nothing)
    end
    return vec(W)                                     # length m
end

# Amemiya-Tobin mapping (Wales & Woodland 1983; Amemiya & Tobin 1987).
# x is reps x m. Negatives -> 0, then divide each row by its row sum.
function _treat_truncations(x::AbstractMatrix)
    out = copy(Matrix{Float64}(x))
    @inbounds for idx in eachindex(out)
        if out[idx] < 0
            out[idx] = 0.0
        end
    end
    rsums = vec(sum(out, dims = 2))                   # length reps
    @inbounds for j in 1:size(out, 2)
        for i in 1:size(out, 1)
            out[i, j] /= rsums[i]
        end
    end
    return out
end

# Expected observed shares given a constant latent-share vector `u` (length m),
# adding the simulated errors `eps_full` (reps x m), treating truncations, and
# taking column means -> length-m vector. Mirrors:
#   Ulat = t(apply(epsilons, 1, function(e) u + e)); colMeans(treatTruncations(Ulat))
function _expected_obs(u::AbstractVector, eps_full::AbstractMatrix)
    reps, m = size(eps_full)
    Ulat = Matrix{Float64}(undef, reps, m)
    @inbounds for j in 1:m
        uj = u[j]
        for i in 1:reps
            Ulat[i, j] = uj + eps_full[i, j]
        end
    end
    Uobs = _treat_truncations(Ulat)
    return vec(mean(Uobs, dims = 1))                  # length m
end

"""
    censored_elasticity(prices, budget, params; quaids=false, demographics=nothing,
                        vcov, point=mean, reps=100000, epsilons=nothing,
                        delta=1e-5, rng=Random.MersenneTwister(20240530))
        -> NamedTuple(elasticities, se, e_uobs)

Simulation-based censored AIDS/QUAIDS demand elasticities with delta-method SEs.

- `prices`        : n x m matrix of LOGGED prices.
- `budget`        : length-n vector of LOGGED total expenditure.
- `params`        : full parameter vector. The trailing j = 0.5*(m-1)*m entries are
                    the upper-triangular (incl. diag, column-major) of an (m-1)x(m-1)
                    Cholesky-like factor S; Sigma = S' S. The leading entries are the
                    model params passed to the share equations.
- `quaids`        : include the QUAIDS quadratic term.
- `demographics`  : `nothing` for no-demographics mode, otherwise an n x t matrix.
- `vcov`          : variance-covariance matrix of `params` (only its leading
                    vd x vd block is used, vd = #model params).
- `point`         : reduction applied column-wise to get the evaluation point
                    (default `mean`).
- `reps`          : number of Monte-Carlo replications.
- `epsilons`      : optional reps x (m-1) matrix of error draws. If provided it is
                    USED VERBATIM (the -rowSums m-th column is appended internally);
                    otherwise draws N(0, Sigma) with `rng`.
- `delta`         : finite-difference disturbance.
- `rng`           : RNG used when `epsilons === nothing`.

Returns a NamedTuple with
  - `elasticities` : m x (m+1) matrix; price cols 1..m then income (own-price on the
                     diagonal with the identity correction applied).
  - `se`           : m x (m+1) matrix of delta-method standard errors.
  - `e_uobs`       : length-m vector of expected Amemiya-Tobin observed shares.
"""
function censored_elasticity(prices::AbstractMatrix, budget::AbstractVector,
                             params::AbstractVector;
                             quaids::Bool = false,
                             demographics = nothing,
                             vcov::AbstractMatrix,
                             point = mean,
                             reps::Integer = 100_000,
                             epsilons = nothing,
                             delta::Real = 1e-5,
                             rng = Random.MersenneTwister(20240530))

    P = Matrix{Float64}(prices)
    n, m = size(P)
    b = Vector{Float64}(budget)
    pars = Vector{Float64}(params)

    has_dems = demographics !== nothing
    t = has_dems ? size(demographics, 2) : 0

    j = Int(0.5 * (m - 1) * m)                       # number of sigma params

    # ----: Evaluation point (column means via `point`) :----
    muPrices = [Float64(point(@view P[:, k])) for k in 1:m]
    muBudget = Float64(point(b))
    muDemogs = has_dems ? [Float64(point(@view demographics[:, k])) for k in 1:t] :
                          Float64[]

    # ----: Model params (drop the trailing sigma block) :----
    nparam = length(pars)
    sigma_block = pars[(nparam - j + 1):nparam]
    params_model = pars[1:(nparam - j)]
    vd = length(params_model)                        # vcov dims

    # ----: Sigma & error simulations :----
    # S is (m-1)x(m-1), upper.tri(diag=TRUE) filled COLUMN-MAJOR with sigma_block.
    S = zeros(Float64, m - 1, m - 1)
    let k = 1
        for jj in 1:(m - 1)
            for ii in 1:jj
                S[ii, jj] = sigma_block[k]
                k += 1
            end
        end
    end
    Sigma = transpose(S) * S                         # (m-1) x (m-1)

    # epsilons: reps x (m-1). Injected verbatim if provided, else drawn N(0, Sigma).
    if epsilons === nothing
        # rmvnorm with method="chol": draws %*% chol(Sigma), where R's chol returns
        # an UPPER-triangular factor R with R'R = Sigma. So eps = Z * R, Z ~ N(0,I).
        Rchol = cholesky(Symmetric(Sigma)).U          # upper factor, R'R = Sigma
        Z = randn(rng, reps, m - 1)
        eps_mm1 = Z * Rchol
    else
        eps_mm1 = Matrix{Float64}(epsilons)
        @assert size(eps_mm1) == (reps, m - 1) "epsilons must be reps x (m-1)"
    end

    # cbind(epsilons, -rowSums(epsilons)) -> reps x m
    eps_full = hcat(eps_mm1, -vec(sum(eps_mm1, dims = 2)))

    # ----: E(X, b) -- expected share without disturbance :----
    U = _mu_shares(muPrices, muBudget, muDemogs, params_model;
                   quaids = quaids, has_dems = has_dems)
    E_Uobs = _expected_obs(U, eps_full)              # length m

    # ----: E(X + delta, b) -- expected share with disturbance in each variable :----
    # Variables in order: the m prices (LEVEL perturbation) then budget (LEVEL).
    nvar = m + 1
    # m_EUobs_dx: rows = good i (1..m), cols = variable x (1..m prices, m+1 budget)
    m_EUobs_dx = Matrix{Float64}(undef, m, nvar)
    for x in 1:nvar
        if x > m
            # Budget perturbation in levels.
            muBudget_delta = log(exp(muBudget) + delta)
            u = _mu_shares(muPrices, muBudget_delta, muDemogs, params_model;
                           quaids = quaids, has_dems = has_dems)
        else
            # Price x perturbation in levels.
            mp = exp.(muPrices)
            mp[x] += delta
            muPrices_delta = log.(mp)
            u = _mu_shares(muPrices_delta, muBudget, muDemogs, params_model;
                           quaids = quaids, has_dems = has_dems)
        end
        m_EUobs_dx[:, x] = _expected_obs(u, eps_full)
    end

    # ----: E(X, b + delta) -- expected share with disturbance in each parameter :----
    # EUobs_db: rows = parameter p (1..vd), cols = good i (1..m)
    EUobs_db = Matrix{Float64}(undef, vd, m)
    for p in 1:vd
        b_delta = copy(params_model)
        b_delta[p] += delta
        u = _mu_shares(muPrices, muBudget, muDemogs, b_delta;
                       quaids = quaids, has_dems = has_dems)
        EUobs_db[p, :] = _expected_obs(u, eps_full)
    end

    # ----: E(X + delta, b + delta) -- disturbance in variable AND parameter :----
    # For each variable x, a vd x m matrix: rows = parameter p, cols = good i.
    EUobs_dxb = Vector{Matrix{Float64}}(undef, nvar)
    for x in 1:nvar
        mat = Matrix{Float64}(undef, vd, m)
        for p in 1:vd
            b_delta = copy(params_model)
            b_delta[p] += delta
            if x > m
                muBudget_delta = log(exp(muBudget) + delta)
                u = _mu_shares(muPrices, muBudget_delta, muDemogs, b_delta;
                               quaids = quaids, has_dems = has_dems)
            else
                mp = exp.(muPrices)
                mp[x] += delta
                muPrices_delta = log.(mp)
                u = _mu_shares(muPrices_delta, muBudget, muDemogs, b_delta;
                               quaids = quaids, has_dems = has_dems)
            end
            mat[p, :] = _expected_obs(u, eps_full)
        end
        EUobs_dxb[x] = mat
    end

    # ----: Elasticity calculations :----
    # etas built as in R: a (m+1) x m matrix `etas[var, good]`, transposed at the end.
    # Column i of R's `etas` = c(price_eta (length m), income_eta) for good i.
    etas = Matrix{Float64}(undef, m + 1, m)
    for i in 1:m
        # Income elasticity (uses budget column = m+1 of m_EUobs_dx).
        p1 = (E_Uobs[i] - m_EUobs_dx[i, m + 1]) / delta
        p2_num = exp(muBudget) + 0.5 * delta
        p2_den = E_Uobs[i] + 0.5 * (E_Uobs[i] - m_EUobs_dx[i, m + 1])
        income_eta = p1 * (p2_num / p2_den) + 1

        # Price elasticities.
        for jj in 1:m
            p1j = (E_Uobs[i] - m_EUobs_dx[i, jj]) / delta
            p2_numj = exp(muPrices[jj]) + 0.5 * delta
            p2_denj = E_Uobs[i] + 0.5 * (E_Uobs[i] - m_EUobs_dx[i, jj])
            etas[jj, i] = p1j * (p2_numj / p2_denj)
        end
        etas[m + 1, i] = income_eta
    end

    # Diagonal correction: etas[1:m, 1:m] -= I(m) (price block only), then transpose.
    for d in 1:m
        etas[d, d] -= 1.0
    end
    elasticities = permutedims(etas)                 # m x (m+1): rows=quantity, cols=price..income

    # ----: Standard Errors (delta method) :----
    # For each good i, build dy (vd x (m+1)): per-parameter elasticity, then
    #   J = (dy - f)/delta, etavcov = J' (vcov[1:vd,1:vd]/n) J, se_i = sqrt(diag).
    V = Matrix{Float64}(vcov)[1:vd, 1:vd] ./ n
    se = Matrix{Float64}(undef, m, m + 1)
    for i in 1:m
        dy = Matrix{Float64}(undef, vd, m + 1)

        # Income elasticity per parameter (variable = budget = EUobs_dxb[m+1]).
        m_EUobs_dxb_inc = EUobs_dxb[m + 1]            # vd x m
        for p in 1:vd
            p1 = (EUobs_db[p, i] - m_EUobs_dxb_inc[p, i]) / delta
            p2_num = exp(muBudget) + 0.5 * delta
            p2_den = EUobs_db[p, i] + 0.5 * (EUobs_db[p, i] - m_EUobs_dxb_inc[p, i])
            dy[p, m + 1] = p1 * (p2_num / p2_den) + 1
        end

        # Price elasticities per parameter. NOTE: R's p2_den divides the second term
        # by delta (a quirk we reproduce verbatim), and applies a -1 only when i==j.
        for jj in 1:m
            m_EUobs_dxb_j = EUobs_dxb[jj]             # vd x m
            for p in 1:vd
                p1 = (EUobs_db[p, i] - m_EUobs_dxb_j[p, i]) / delta
                p2_num = exp(muPrices[jj]) + 0.5 * delta
                p2_den = EUobs_db[p, i] +
                         0.5 * (EUobs_db[p, i] - m_EUobs_dxb_j[p, i]) / delta
                val = p1 * (p2_num / p2_den)
                dy[p, jj] = (i == jj) ? (val - 1) : val
            end
        end

        # f = expand_rVector(etas[i, ], dy): each column c filled with elasticities[i, c].
        # J = (dy - f)/delta.
        J = Matrix{Float64}(undef, vd, m + 1)
        for c in 1:(m + 1)
            fc = elasticities[i, c]
            for p in 1:vd
                J[p, c] = (dy[p, c] - fc) / delta
            end
        end

        etavcov = transpose(J) * V * J               # (m+1) x (m+1)
        for c in 1:(m + 1)
            se[i, c] = sqrt(etavcov[c, c])
        end
    end

    return (elasticities = elasticities, se = se, e_uobs = E_Uobs)
end
