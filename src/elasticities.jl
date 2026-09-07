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
                    quaids::Bool, has_dems::Bool,
                    price_index::Symbol = :translog, mu_shares = nothing)
    m = length(muPrices)
    P1 = reshape(Vector{Float64}(muPrices), 1, m)     # 1 x m price matrix
    b1 = [Float64(muBudget)]                          # length-1 budget
    sh1 = mu_shares === nothing ? nothing : reshape(Vector{Float64}(mu_shares), 1, m)
    if has_dems
        t = length(muDemogs)
        Z1 = reshape(Vector{Float64}(muDemogs), 1, t) # 1 x t demographics
        W = aids_shares(P1, b1, params_model; quaids = quaids, demographics = Z1,
                        price_index = price_index, shares = sh1)
    else
        W = aids_shares(P1, b1, params_model; quaids = quaids, demographics = nothing,
                        price_index = price_index, shares = sh1)
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

# Minimum-distance Slutsky symmetrization of an m x (m+1) Marshallian elasticity matrix
# `E` (price cols 1..m, income col m+1) at budget shares `w` (length m). Builds the
# compensated εᶜ_ij = ε_ij + w_j·η_i, forms the substitution matrix S_ij = w_i·εᶜ_ij,
# averages S ← (S+S')/2, and maps back to Marshallian. Income elasticities (col m+1) are
# untouched. After this, w_i·εᶜ_ij is symmetric to machine precision. Returns a NEW matrix.
function _symmetrize_marshallian(E::AbstractMatrix, w::AbstractVector)
    m = length(w)
    Es = Matrix{Float64}(E)                           # copy; rows = goods, cols = price..income
    eta = Es[:, m + 1]
    S = Matrix{Float64}(undef, m, m)                  # Slutsky substitution matrix
    @inbounds for i in 1:m, jc in 1:m
        epc = Es[i, jc] + w[jc] * eta[i]              # compensated price elasticity
        S[i, jc] = w[i] * epc
    end
    Ssym = (S .+ S') ./ 2
    @inbounds for i in 1:m, jc in 1:m
        epc_sym = Ssym[i, jc] / w[i]                  # symmetric compensated
        Es[i, jc] = epc_sym - w[jc] * eta[i]          # back to Marshallian
    end
    return Es
end

"""
    censored_elasticity(prices, budget, params; quaids=false, demographics=nothing,
                        vcov, point=mean, reps=100000, epsilons=nothing,
                        delta=1e-5, rng=Random.MersenneTwister(20240530),
                        method=:finite_difference)
        -> ElasticityResult(elasticities, se, e_uobs, symmetric, share_names)

Simulation-based censored AIDS/QUAIDS demand elasticities with delta-method SEs.

- `method`        : `:finite_difference` (default; the R algorithm below) or `:closed_form`.
                    The closed form evaluates the exact derivative of the expected observed
                    share on the same draws, in one pass, with no step size, and its
                    delta-method SEs use the exact parameter gradient (pathwise plus the
                    regime-boundary term) with `vcov[1:vd,1:vd]` as the sampling variance
                    of the model parameters (NOT divided by n). See the section
                    "Closed-form derivatives" at the end of this file.

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
                             rng = Random.MersenneTwister(20240530),
                             price_index = :translog,
                             shares = nothing,
                             symmetry::Bool = false,
                             share_names = nothing,
                             method::Symbol = :finite_difference)

    method in (:finite_difference, :closed_form) ||
        error("censored_elasticity: method must be :finite_difference or :closed_form, got :$method")
    st = _elasticity_setup(prices, budget, params; demographics = demographics, point = point,
                           reps = reps, epsilons = epsilons, rng = rng,
                           price_index = price_index, shares = shares)
    (; n, m, has_dems, muPrices, muBudget, muDemogs, muShares, params_model, vd, Sigma, eps_full) = st
    price_index = st.price_index
    sn = share_names === nothing ? ["good$(i)" for i in 1:m] : String.(share_names)

    if method === :closed_form
        E, se_cf, s, _ = _closed_form_elasticity(st, vcov; quaids = quaids, symmetry = symmetry)
        return ElasticityResult(E, se_cf, s, symmetry, sn)
    end

    # ----: E(X, b) -- expected share without disturbance :----
    U = _mu_shares(muPrices, muBudget, muDemogs, params_model;
                   quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
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
                           quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
        else
            # Price x perturbation in levels.
            mp = exp.(muPrices)
            mp[x] += delta
            muPrices_delta = log.(mp)
            u = _mu_shares(muPrices_delta, muBudget, muDemogs, params_model;
                           quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
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
                       quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
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
                               quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
            else
                mp = exp.(muPrices)
                mp[x] += delta
                muPrices_delta = log.(mp)
                u = _mu_shares(muPrices_delta, muBudget, muDemogs, b_delta;
                               quaids = quaids, has_dems = has_dems,
                   price_index = price_index, mu_shares = muShares)
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
        p1 = (m_EUobs_dx[i, m + 1] - E_Uobs[i]) / delta
        p2_num = exp(muBudget) + 0.5 * delta
        p2_den = E_Uobs[i] + 0.5 * (m_EUobs_dx[i, m + 1] - E_Uobs[i])
        income_eta = p1 * (p2_num / p2_den) + 1

        # Price elasticities.
        for jj in 1:m
            p1j = (m_EUobs_dx[i, jj] - E_Uobs[i]) / delta
            p2_numj = exp(muPrices[jj]) + 0.5 * delta
            p2_denj = E_Uobs[i] + 0.5 * (m_EUobs_dx[i, jj] - E_Uobs[i])
            etas[jj, i] = p1j * (p2_numj / p2_denj)
        end
        etas[m + 1, i] = income_eta
    end

    # Diagonal correction: etas[1:m, 1:m] -= I(m) (price block only), then transpose.
    for d in 1:m
        etas[d, d] -= 1.0
    end
    elasticities = permutedims(etas)                 # m x (m+1): rows=quantity, cols=price..income

    # ----: Optional Slutsky symmetry (opt-in) :----
    # Min-distance symmetrization of the compensated substitution matrix at the model's own
    # expected shares E_Uobs. Off by default (R-faithful); see PLAN.md Change 3.
    if symmetry
        elasticities = _symmetrize_marshallian(elasticities, E_Uobs)
    end

    # ----: Standard Errors (delta method) :----
    # For each good i, build dy (vd x (m+1)): per-parameter elasticity, then
    #   J = (dy - f)/delta, etavcov = J' (vcov[1:vd,1:vd]/n) J, se_i = sqrt(diag).
    V = Matrix{Float64}(vcov)[1:vd, 1:vd] ./ n
    se = Matrix{Float64}(undef, m, m + 1)

    if !symmetry
        # Default path — byte-identical to the R-faithful SE block.
        for i in 1:m
            dy = Matrix{Float64}(undef, vd, m + 1)

            # Income elasticity per parameter (variable = budget = EUobs_dxb[m+1]).
            m_EUobs_dxb_inc = EUobs_dxb[m + 1]            # vd x m
            for p in 1:vd
                p1 = (m_EUobs_dxb_inc[p, i] - EUobs_db[p, i]) / delta
                p2_num = exp(muBudget) + 0.5 * delta
                p2_den = EUobs_db[p, i] + 0.5 * (m_EUobs_dxb_inc[p, i] - EUobs_db[p, i])
                dy[p, m + 1] = p1 * (p2_num / p2_den) + 1
            end

            # Price elasticities per parameter. NOTE: R's p2_den divides the second term
            # by delta (a quirk we reproduce verbatim), and applies a -1 only when i==j.
            for jj in 1:m
                m_EUobs_dxb_j = EUobs_dxb[jj]             # vd x m
                for p in 1:vd
                    p1 = (m_EUobs_dxb_j[p, i] - EUobs_db[p, i]) / delta
                    p2_num = exp(muPrices[jj]) + 0.5 * delta
                    p2_den = EUobs_db[p, i] +
                             0.5 * (m_EUobs_dxb_j[p, i] - EUobs_db[p, i]) / delta
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
    else
        # Symmetry-constrained path: build the FULL perturbed Marshallian matrix per
        # parameter (same R-faithful formulas), symmetrize each with its perturbed shares,
        # then the delta method on the symmetrized elasticities.
        Ep_all = Vector{Matrix{Float64}}(undef, vd)
        for p in 1:vd
            Ep = Matrix{Float64}(undef, m, m + 1)
            wp = EUobs_db[p, :]                            # perturbed expected shares (length m)
            for i in 1:m
                p1 = (EUobs_dxb[m + 1][p, i] - EUobs_db[p, i]) / delta
                p2_num = exp(muBudget) + 0.5 * delta
                p2_den = EUobs_db[p, i] + 0.5 * (EUobs_dxb[m + 1][p, i] - EUobs_db[p, i])
                Ep[i, m + 1] = p1 * (p2_num / p2_den) + 1
                for jj in 1:m
                    p1j = (EUobs_dxb[jj][p, i] - EUobs_db[p, i]) / delta
                    p2_numj = exp(muPrices[jj]) + 0.5 * delta
                    p2_denj = EUobs_db[p, i] +
                              0.5 * (EUobs_dxb[jj][p, i] - EUobs_db[p, i]) / delta
                    val = p1j * (p2_numj / p2_denj)
                    Ep[i, jj] = (i == jj) ? (val - 1) : val
                end
            end
            Ep_all[p] = _symmetrize_marshallian(Ep, wp)
        end
        for i in 1:m
            J = Matrix{Float64}(undef, vd, m + 1)
            for c in 1:(m + 1)
                fc = elasticities[i, c]                   # symmetrized base
                for p in 1:vd
                    J[p, c] = (Ep_all[p][i, c] - fc) / delta
                end
            end
            etavcov = transpose(J) * V * J
            for c in 1:(m + 1)
                se[i, c] = sqrt(etavcov[c, c])
            end
        end
    end

    return ElasticityResult(elasticities, se, E_Uobs, symmetry, sn)
end

# Evaluation point, model/sigma split, Sigma and the reps x m error draws shared by both methods
# (moved verbatim from censored_elasticity; the arithmetic is unchanged).
function _elasticity_setup(prices::AbstractMatrix, budget::AbstractVector, params::AbstractVector;
                           demographics, point, reps, epsilons, rng, price_index, shares)
    price_index = _price_index_sym(price_index)   # accept Symbol or PriceIndex enum
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

    # Stone price index (if requested) uses the point (mean) of the OBSERVED shares.
    muShares = nothing
    if price_index === :stone
        shares === nothing && error("censored_elasticity: price_index=:stone requires observed `shares`")
        Sh = Matrix{Float64}(shares)
        muShares = [Float64(point(@view Sh[:, k])) for k in 1:m]
    end

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

    return (n = n, m = m, t = t, has_dems = has_dems, price_index = price_index,
            muPrices = muPrices, muBudget = muBudget, muDemogs = muDemogs, muShares = muShares,
            params_model = params_model, vd = vd, Sigma = Sigma, eps_full = eps_full)
end

# ============================================================================
# Closed-form derivatives of the expected observed shares (method = :closed_form).
#
# Latent shares S* = U(p, w, z) + ε, observed S_i = S*_i 1{S*_i > 0} / T with T = Σ_{j∈B} S*_j
# and B = {j : S*_j > 0}. For s_i = E[S_i] and v ∈ {ln p_k, ln w} with d_j = ∂U_j/∂v,
#     ∂s_i/∂v = E[ 1{i∈B} (d_i − S_i Σ_{j∈B} d_j) / T ],
# evaluated on the draws used for s_i itself: one pass, no perturbation, no step size
# (Nava 2026, "Closed-form elasticities for the censored QUAIDS", Proposition 1). Then
#     e_ik = (∂s_i/∂ln p_k)/s_i − δ_ik,   η_i = (∂s_i/∂ln w)/s_i + 1,
# the limit Δ → 0 of the finite-difference scheme above. Engel and Cournot aggregation and
# homogeneity hold exactly, draw by draw.
#
# Delta-method SEs use the exact gradient of (s, ∂s/∂ln p, ∂s/∂ln w) in the model parameters θ
# (draws held fixed): a pathwise part (B fixed) plus a boundary part from draws that switch
# purchase regime, Σ_j f_j E[Δ_j g | S*_j = 0] ∂U_j/∂θ, with f_j the density of S*_j at zero and
# the conditional expectation taken on the same draws projected onto {S*_j = 0} (ibid.,
# Proposition 2). The θ-Jacobians of the smooth latent functions (U, D, Dw) are central
# differences with relative step 1e-6; everything else is exact.
# ============================================================================

# Latent mean shares U (m), D[i,k] = ∂U_i/∂ln p_k (m x m), Dw[i] = ∂U_i/∂ln w (m) at one point,
# for the package's share equations (demographics in the expenditure slope):
#   U_i = α_i + Σ_l γ_il ln p_l + (β_i + θ_i'z) x + λ_i x²/b(p),  x = ln w − ln a(p),  b(p) = Π p_l^β_l.
# :translog  ∂ln a/∂ln p_k = α_k + Σ_l γ_kl ln p_l;  :stone  ∂ln P*/∂ln p_k = w̄_k (predetermined).
function _latent_derivs(lnp::Vector{Float64}, lnw::Float64, z::Vector{Float64},
                        pm::Vector{Float64}; quaids::Bool, has_dems::Bool,
                        price_index::Symbol, mu_shares)
    m = length(lnp); t = has_dems ? length(z) : 0
    α, β, Γ, Θ, λ = _unpack_params(pm, m, t; quaids = quaids, has_dems = has_dems)
    bz = has_dems ? β .+ Θ * z : β                       # β_i + θ_i'z
    Gp = Γ * lnp
    if price_index === :stone
        a = Vector{Float64}(mu_shares)
        lna = dot(a, lnp)
    else
        a = α .+ Gp
        lna = dot(α, lnp) + 0.5 * dot(lnp, Gp)
    end
    x   = lnw - lna
    bp  = exp(dot(β, lnp))
    lam = quaids ? λ : zeros(m)
    U  = α .+ Gp .+ bz .* x .+ lam .* (x * x / bp)
    Dw = bz .+ lam .* (2x / bp)
    D  = Γ .- Dw * a' .- (x * x / bp) .* (lam * β')
    return U, D, Dw
end

# Central-difference Jacobians of (U, D, Dw) in the vd model parameters: JU (m x vd),
# JD (m x m x vd), JDw (m x vd). The latent functions are smooth and cheap, so this is exact
# to ~1e-10; vd = 0 (no SEs wanted) returns empty arrays.
function _latent_jacobians(lnp, lnw, z, pm; vd::Int = length(pm), kw...)
    m = length(lnp)
    JU = zeros(m, vd); JD = zeros(m, m, vd); JDw = zeros(m, vd)
    for q in 1:vd
        h = 1e-6 * max(1.0, abs(pm[q]))
        pp = copy(pm)
        pp[q] += h;  Up, Dp, Dwp = _latent_derivs(lnp, lnw, z, pp; kw...)
        pp[q] -= 2h; Um, Dm, Dwm = _latent_derivs(lnp, lnw, z, pp; kw...)
        JU[:, q] = (Up .- Um) ./ (2h); JD[:, :, q] = (Dp .- Dm) ./ (2h); JDw[:, q] = (Dwp .- Dwm) ./ (2h)
    end
    return JU, JD, JDw
end

# One pass over the draws: s = E[S], dp = ∂s/∂ln p (m x m), dw = ∂s/∂ln w (m), and their
# PATHWISE θ-Jacobians Js (m x vd), Jdp (m x m x vd), Jdw (m x vd) (B held fixed per draw).
function _closed_form_pass(U, D, Dw, eps_full, JU, JD, JDw)
    reps, m = size(eps_full); vd = size(JU, 2)
    s = zeros(m); dp = zeros(m, m); dw = zeros(m)
    Js = zeros(m, vd); Jdp = zeros(m, m, vd); Jdw = zeros(m, vd)
    Sst = zeros(m); inB = falses(m); sumD = zeros(m); sumJD = zeros(m)
    @inbounds for r in 1:reps
        T = 0.0; sumDw = 0.0; fill!(sumD, 0.0)
        for j in 1:m
            Sst[j] = U[j] + eps_full[r, j]
            inB[j] = Sst[j] > 0
            if inB[j]
                T += Sst[j]; sumDw += Dw[j]
                for k in 1:m; sumD[k] += D[j, k]; end
            end
        end
        for i in 1:m
            inB[i] || continue
            Si = Sst[i] / T
            s[i] += Si
            for k in 1:m; dp[i, k] += (D[i, k] - Si * sumD[k]) / T; end
            dw[i] += (Dw[i] - Si * sumDw) / T
        end
        for q in 1:vd
            dT = 0.0; sumJDw = 0.0; fill!(sumJD, 0.0)
            for j in 1:m
                inB[j] || continue
                dT += JU[j, q]; sumJDw += JDw[j, q]
                for k in 1:m; sumJD[k] += JD[j, k, q]; end
            end
            for i in 1:m
                inB[i] || continue
                Si = Sst[i] / T
                dSi = (JU[i, q] - Si * dT) / T                     # ∂S_i/∂θ_q
                Js[i, q] += dSi
                for k in 1:m
                    g = (D[i, k] - Si * sumD[k]) / T
                    Jdp[i, k, q] += (JD[i, k, q] - dSi * sumD[k] - Si * sumJD[k]) / T - g * dT / T
                end
                gw = (Dw[i] - Si * sumDw) / T
                Jdw[i, q] += (JDw[i, q] - dSi * sumDw - Si * sumJDw) / T - gw * dT / T
            end
        end
    end
    return s ./ reps, dp ./ reps, dw ./ reps, Js ./ reps, Jdp ./ reps, Jdw ./ reps
end

# Boundary part of ∂dp/∂θ (m x m x vd) and ∂dw/∂θ (m x vd): for each good j, the density f_j of
# S*_j at zero times the mean over the draws projected onto {S*_j = 0} of the jump of the integrand
# when j enters B (D_jk/T for i = j, −S_i D_jk/T for i ∈ B), times ∂U_j/∂θ.
function _boundary_terms(U, D, Dw, eps_full, Sigma, JU)
    reps, m = size(eps_full); vd = size(JU, 2)
    # Covariance of the full ε (ε_m = −Σ_{j<m} ε_j): C = [Σ  −Σ1; −1'Σ  1'Σ1]
    C = zeros(m, m)
    C[1:m-1, 1:m-1] = Sigma
    C[1:m-1, m] = -vec(sum(Sigma, dims = 2)); C[m, 1:m-1] = C[1:m-1, m]; C[m, m] = sum(Sigma)
    BTp = zeros(m, m, vd); BTw = zeros(m, vd)
    A = zeros(m, m); Aw = zeros(m); Sst = zeros(m)
    for j in 1:m
        ω2 = C[j, j]
        fj = exp(-0.5 * U[j]^2 / ω2) / sqrt(2π * ω2)          # density of S*_j at 0
        fill!(A, 0.0); fill!(Aw, 0.0)
        @inbounds for r in 1:reps
            shift = (eps_full[r, j] + U[j]) / ω2              # Gaussian projection onto S*_j = 0
            T = 0.0
            for l in 1:m
                Sst[l] = U[l] + eps_full[r, l] - C[l, j] * shift
                (l != j && Sst[l] > 0) && (T += Sst[l])
            end
            for i in 1:m
                fac = i == j ? 1.0 / T : (Sst[i] > 0 ? -(Sst[i] / T) / T : 0.0)
                fac == 0.0 && continue
                for k in 1:m; A[i, k] += fac * D[j, k]; end
                Aw[i] += fac * Dw[j]
            end
        end
        for q in 1:vd, i in 1:m
            c = fj * JU[j, q] / reps
            for k in 1:m; BTp[i, k, q] += c * A[i, k]; end
            BTw[i, q] += c * Aw[i]
        end
    end
    return BTp, BTw
end

# Closed-form elasticities at the setup `st` (see _elasticity_setup): returns
# (E, se, s, G) with E the m x (m+1) Marshallian price + income matrix, se its delta-method SEs
# under V = vcov[1:vd,1:vd], s = E[S], and G the m x (m+1) x vd gradient of E in θ.
# `se = false` skips the gradient (G empty, se = NaN).
function _closed_form_elasticity(st, vcov; quaids::Bool, symmetry::Bool, se::Bool = true)
    (; m, has_dems, price_index, muPrices, muBudget, muDemogs, muShares, params_model, vd, Sigma, eps_full) = st
    kw = (quaids = quaids, has_dems = has_dems, price_index = price_index, mu_shares = muShares)
    nq = se ? vd : 0
    U, D, Dw = _latent_derivs(muPrices, muBudget, muDemogs, params_model; kw...)
    JU, JD, JDw = _latent_jacobians(muPrices, muBudget, muDemogs, params_model; vd = nq, kw...)
    s, dp, dw, Js, Jdp, Jdw = _closed_form_pass(U, D, Dw, eps_full, JU, JD, JDw)
    E = hcat(dp ./ s .- I(m), dw ./ s .+ 1)               # m x (m+1)
    se || return E, fill(NaN, m, m + 1), s, zeros(m, m + 1, 0)
    BTp, BTw = _boundary_terms(U, D, Dw, eps_full, Sigma, JU)
    G = zeros(m, m + 1, vd)
    for q in 1:vd, i in 1:m
        for k in 1:m
            G[i, k, q] = (Jdp[i, k, q] + BTp[i, k, q]) / s[i] - dp[i, k] * Js[i, q] / s[i]^2
        end
        G[i, m + 1, q] = (Jdw[i, q] + BTw[i, q]) / s[i] - dw[i] * Js[i, q] / s[i]^2
    end
    if symmetry
        # Chain rule through the (smooth, deterministic) symmetrization map by a central
        # directional difference along each parameter's (∂E/∂θ_q, ∂s/∂θ_q).
        h = 1e-6
        for q in 1:vd
            G[:, :, q] = (_symmetrize_marshallian(E .+ h .* G[:, :, q], s .+ h .* Js[:, q]) .-
                          _symmetrize_marshallian(E .- h .* G[:, :, q], s .- h .* Js[:, q])) ./ (2h)
        end
        E = _symmetrize_marshallian(E, s)
    end
    V = Matrix{Float64}(vcov)[1:vd, 1:vd]
    sem = Matrix{Float64}(undef, m, m + 1)
    for i in 1:m, c in 1:(m + 1)
        g = vec(G[i, c, :])
        sem[i, c] = sqrt(max(dot(g, V * g), 0.0))
    end
    return E, sem, s, G
end
