# Implementation A: CENSORED log-likelihood for the AIDS / QUAIDS demand system.
#
# Faithful port of censoredAIDS::censoredaidsLoglike (R) — the Wales–Woodland
# censored MLE. Returns an n-vector of per-household log-likelihood contributions.
#
# The +1e-7 / +1e-8 fudge factors are kept EXACTLY as in the R source (M2 must
# replicate R, including these; they are revisited later).
#
# Includable INTO the CensoredDemand module: it assumes `aids_shares` is already
# defined and in scope, and that the package `using`s are already loaded. The
# explicit `using` below is harmless (re-importing) and makes the file usable as
# a standalone include for the scratch self-test harness.

using LinearAlgebra, Distributions, MvNormalCDF, Random

# ---- helpers ----

"""
    cov2cor(M) -> Matrix

Convert a covariance matrix to a correlation matrix (mirrors `stats::cov2cor`).
"""
function cov2cor(M::AbstractMatrix)
    s = sqrt.(diag(M))
    return M ./ (s * s')
end

"""
    nearestPD(A) -> Matrix

Nearest positive-(semi)definite matrix via symmetric eigen-decomposition, clipping
eigenvalues to a small positive floor and rebuilding. Mirrors the role of
`Matrix::nearPD` in the R source (used to sanitize the MVN correlation matrix
before the orthant-probability call). The result is re-symmetrized.
"""
function nearestPD(A::AbstractMatrix; floor::Real = 1e-8)
    B = Symmetric((A .+ A') ./ 2)
    F = eigen(B)
    vals = max.(F.values, floor)
    Apd = F.vectors * Diagonal(vals) * F.vectors'
    Apd = (Apd .+ Apd') ./ 2          # re-symmetrize
    return Matrix(Apd)
end

"""
    censored_loglike(shares, prices, budget, params; quaids=false, demographics=nothing,
                     seed=20240530, mc_points=2000, floor_mode=:additive_r) -> Vector{Float64}

Wales–Woodland censored log-likelihood. Faithful port of
`censoredAIDS::censoredaidsLoglike`.

- `shares`        : n x m matrix of raw budget shares (zeros mark non-purchase / censoring).
- `prices`        : n x m matrix of LOGGED prices.
- `budget`        : length-n vector of LOGGED total expenditure.
- `params`        : FULL parameter vector `[share-params..., sigma-params]`. The last
                    `j = 0.5*(m-1)*m` entries are the Σ (column-major upper-tri) params;
                    everything before is the `aids_shares` parameter vector.
- `quaids`        : if true, include the quadratic (QUAIDS) term.
- `demographics`  : `nothing`, or an n x t demographic matrix.
- `seed`          : base RNG seed. Observation `i` draws its orthant-probability Monte-Carlo
                    sample from `MersenneTwister(hash((seed, i)))`, so the result is BOTH
                    reproducible AND independent of thread scheduling — the per-household loop is
                    threaded (`Threads.@threads`) and serial vs parallel are bit-for-bit identical.
- `mc_points`     : sample-point budget for `MvNormalCDF.mvnormcdf` (>= 2000).
- `floor_mode`    : how the partial-regime log terms are floored.
                    `:additive_r` (default) reproduces the R source's additive `log(x + 1e-7/1e-8)`
                    fudge EXACTLY (use this for R-parity). `:guard` uses the principled
                    `log(max(x, 1e-300))` instead — the additive floor hands ~268 nats of spurious
                    credit to households whose observed regime the params deem near-impossible
                    (orthant prob ≈ 0), masking misfit and creating a likelihood plateau (see M4 / PLAN.md).

- `jacobian`      : if true (default) the partial-purchase regimes (1 ≤ k < m goods bought) include the
                    Jacobian T^(k−1) of the Wales–Woodland share renormalisation, making the likelihood a
                    proper density of the observed shares. `false` reproduces the original GAUSS/R
                    objective (Appendix D of Nava & Dong 2022), which omits that factor.
- `parallel`      : if true (default) the per-household loop is threaded with `Threads.@threads`
                    (set the core count via `JULIA_NUM_THREADS` / `julia -t N`); if false it runs
                    serially. Results are bit-for-bit identical either way.

Returns a length-n vector of per-household log-likelihood contributions.
"""
# Floored log for the partial-purchase regimes. `:additive_r` == R's verbatim `log(x + F)`;
# `:guard` == principled `log(max(x, 1e-300))` (honest scoring of near-impossible regimes).
_flog(x::Real, F::Real, mode::Symbol) = mode === :guard ? log(max(x, 1e-300)) : log(x + F)


# ============================================================================
# Partial regimes WITH the share-renormalisation Jacobian (default since the fix)
# ============================================================================
#
# A household buying k of the m goods reports S_i = S*_i / T, T = Σ_{bought} S*_j
# = 1 − Σ_{unbought} S*_j. The map (latent shares) → (observed shares, unbought latent
# shares y) has Jacobian T^(k−1), so the likelihood of the observed data is
#
#     L = ∫_{y ≤ 0} N_{m−1}( x(y); U, Σ ) · T(y)^(k−1) dy,      x(y) = A·y + c0,
#
# x being the m−1 retained latent shares implied by (S, y). Completing the square,
#
#     L = C · E[ (1 − 1'y)^(k−1) · 1{y ≤ 0} ],   y ~ N_d(μ, Ω),   d = m − k,
#
# with C, μ, Ω in closed form. Appendix D of Nava & Dong (2022), the GAUSS script and the
# R port integrate the density along the ray WITHOUT T^(k−1): their partial-regime terms
# are not densities (for the paper's (U, Σ) they integrate to ≈0.33 instead of 1) and the
# resulting estimator is not maximum likelihood. That legacy path is kept under
# `jacobian = false`; k = 1 (pure probability) and k = m (plain density) are unaffected.

_Φ(x) = cdf(Normal(), x)
_φ(x) = pdf(Normal(), x)
function _gausslegendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* abs2.(E.vectors[1, :])
end
const _GL6  = _gausslegendre(6)
const _GL12 = _gausslegendre(12)
const _GL20 = _gausslegendre(20)
const _KRONECKER = [sqrt(p) for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)]

"Genz (2004) BVND: P(X > h, Y > k) for a standard bivariate normal with correlation r."
function _bvnu(dh::Real, dk::Real, r::Real)
    if abs(r) < 0.3
        x, w = _GL6
    elseif abs(r) < 0.75
        x, w = _GL12
    else
        x, w = _GL20
    end
    h = dh; k = dk; hk = h * k
    bvn = 0.0
    if abs(r) < 0.925
        hs = (h * h + k * k) / 2
        asr = asin(r)
        for j in eachindex(x)
            sn = sin(asr * (x[j] + 1) / 2)
            bvn += w[j] * exp((sn * hk - hs) / (1 - sn * sn))
        end
        bvn = bvn * asr / (4π) + _Φ(-h) * _Φ(-k)
    else
        if r < 0
            k = -k; hk = -hk
        end
        if abs(r) < 1
            as = (1 - r) * (1 + r); a = sqrt(as); bs = (h - k)^2
            c = (4 - hk) / 8; d = (12 - hk) / 16
            asr = -(bs / as + hk) / 2
            if asr > -100
                bvn = a * exp(asr) * (1 - c * (bs - as) * (1 - d * bs / 5) / 3 + c * d * as * as / 5)
            end
            if -hk < 100
                b = abs(h - k)
                bvn -= exp(-hk / 2) * sqrt(2π) * _Φ(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
            end
            a = a / 2
            for j in eachindex(x)
                xs = (a * (x[j] + 1))^2
                rs = sqrt(1 - xs)
                asr = -(bs / xs + hk) / 2
                if asr > -100
                    bvn += a * w[j] * exp(asr) * (exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs - (1 + c * xs * (1 + d * xs)))
                end
            end
            bvn = -bvn / (2π)
        end
        if r > 0
            bvn += _Φ(-max(h, k))
        else
            bvn = -bvn
            if k > h
                bvn += (k < 0) ? _Φ(k) - _Φ(h) : _Φ(-h) - _Φ(-k)
            end
        end
    end
    return bvn
end

"""
    _trunc_moment(μ, Ω, p, mc_points, rng) -> E[(1 − 1'y)^p · 1{y ≤ 0}],  y ~ N_d(μ, Ω)

Exact for d = 1 (univariate truncated-moment recursion, any p) and for d = 2 with p ≤ 1
(Genz bivariate CDF, bivariate first moment). Otherwise the orthant integral is evaluated
by `mvnormcdf` for p = 0 and, for p ≥ 1, by quasi-Monte-Carlo with the Genz
separation-of-variables transform (Kronecker sequence, one random shift from `rng`).
"""
function _trunc_moment(mu::AbstractVector, Om::AbstractMatrix, p::Int, mc_points::Int, rng)
    d = length(mu)
    if d == 1
        m1 = mu[1]; v = Om[1, 1]; s = sqrt(v)
        I0 = _Φ(-m1 / s)
        p == 0 && return I0
        I = zeros(p + 1)
        I[1] = I0
        I[2] = m1 * I0 - v * _φ(m1 / s) / s              # ∫_{-∞}^0 y f(y) dy
        for n in 2:p
            I[n + 1] = m1 * I[n] + (n - 1) * v * I[n - 1]
        end
        return sum(binomial(p, j) * (-1)^j * I[j + 1] for j in 0:p)
    elseif d == 2 && p <= 1
        s1 = sqrt(Om[1, 1]); s2 = sqrt(Om[2, 2]); ρ = Om[2, 1] / (s1 * s2)
        P = _bvnu(mu[1] / s1, mu[2] / s2, ρ)              # P(y1 ≤ 0, y2 ≤ 0)
        p == 0 && return P
        # E[y_i 1{y≤0}] = μ_i P − Σ_j Ω_ij f_j(0) F_{−j|j}(0 | y_j = 0)
        f1 = _φ(mu[1] / s1) / s1
        f2 = _φ(mu[2] / s2) / s2
        m21 = mu[2] - Om[2, 1] / Om[1, 1] * mu[1]; s21 = sqrt(Om[2, 2] - Om[2, 1]^2 / Om[1, 1])
        m12 = mu[1] - Om[2, 1] / Om[2, 2] * mu[2]; s12 = sqrt(Om[1, 1] - Om[2, 1]^2 / Om[2, 2])
        return (1 - mu[1] - mu[2]) * P +
               (Om[1, 1] + Om[2, 1]) * f1 * _Φ(-m21 / s21) +
               (Om[2, 1] + Om[2, 2]) * f2 * _Φ(-m12 / s12)
    elseif p == 0
        return mvnormcdf(Vector{Float64}(mu), Matrix{Float64}(Om), fill(-Inf, d), zeros(d);
                         m = mc_points, rng = rng)[1]
    else
        L = cholesky(Symmetric(Matrix{Float64}(Om))).L
        α = _KRONECKER[1:d]
        shift = rand(rng, d)
        z = zeros(d)
        acc = 0.0
        for jpt in 1:mc_points
            w = 1.0; ysum = 0.0
            for i in 1:d
                u = mod(jpt * α[i] + shift[i], 1.0)
                bi = -mu[i]
                for j in 1:i-1
                    bi -= L[i, j] * z[j]
                end
                bi /= L[i, i]
                Pb = _Φ(bi)
                w *= Pb
                z[i] = quantile(Normal(), clamp(u * Pb, 1e-300, 1 - 1e-16))
                yi = mu[i]
                for j in 1:i
                    yi += L[i, j] * z[j]
                end
                ysum += yi
            end
            acc += w * (1 - ysum)^p
        end
        return acc / mc_points
    end
end

"""
    _partial_ll_jac(Si, Ui, Sinv, logdetS, m, mc_points, rng, floor_mode)

Log-likelihood contribution of a household with 1 ≤ k < m purchased goods, including the
Jacobian T^(k−1). `Si`/`Ui` are the household's observed shares and predicted latent shares
(length m; good m is the dropped equation); `Sinv`, `logdetS` refer to the (m−1)×(m−1) Σ.
"""
function _partial_ll_jac(Si::AbstractVector, Ui::AbstractVector, Sinv::AbstractMatrix, logdetS::Real,
                         m::Int, mc_points::Int, rng, floor_mode::Symbol)
    bought = [Si[i] != 0.0 for i in 1:m]
    k = count(bought); d = m - k
    zero_idx = findall(!, bought)                         # unbought goods, index order
    A = zeros(m - 1, d); c0 = zeros(m - 1)
    for i in 1:m-1
        if bought[i]
            A[i, :] .= -Si[i]; c0[i] = Si[i]
        else
            A[i, findfirst(==(i), zero_idx)] = 1.0
        end
    end
    Ut = [Ui[i] for i in 1:m-1] .- c0
    SU = Sinv * Ut
    quad = dot(Ut, SU)
    Qm = A' * Sinv * A
    Q = Symmetric((Qm .+ Qm') ./ 2)
    r = A' * SU
    Om = Symmetric(inv(Q))
    mu = Om * r
    K = quad - dot(mu, r)
    logC = -0.5 * (m - 1 - d) * log(2π) - 0.5 * logdetS - 0.5 * logdet(Q) - 0.5 * K
    M = _trunc_moment(mu, Matrix(Om), k - 1, mc_points, rng)
    return logC + _flog(M, 1e-8, floor_mode)
end

function censored_loglike(shares::AbstractMatrix, prices::AbstractMatrix,
                          budget::AbstractVector, params::AbstractVector;
                          quaids::Bool = false, demographics = nothing,
                          seed::Integer = 20240530,
                          mc_points::Integer = 2000,
                          floor_mode = :additive_r,
                          parallel::Bool = true,
                          price_index = :translog,
                          jacobian::Bool = true)::Vector{Float64}

    floor_mode  = _floor_mode_sym(floor_mode)      # accept Symbol or FloorMode enum
    price_index = _price_index_sym(price_index)    # accept Symbol or PriceIndex enum
    P = Matrix{Float64}(prices)
    n, m = size(P)
    S = Matrix{Float64}(shares)
    p = Vector{Float64}(params)
    j = Int(0.5 * (m - 1) * m)                     # number of sigma params

    # ----: predicted shares :----
    U = aids_shares(P, budget, p[1:(end - j)];
                    quaids = quaids, demographics = demographics,
                    price_index = price_index, shares = S)          # n x m

    # ----: micro-regime dummies :----
    d = S .!= 0.0                                  # n x m Bool
    nu = vec(sum(d, dims = 2))                      # goods bought per household

    lf = zeros(Float64, n)

    # ----: Sigma (covariance of the m-1 errors) :----
    # last j params are the COLUMN-MAJOR upper-tri (incl diag) of (m-1)x(m-1) R; Sigma = R'R.
    sig_params = p[(end - j + 1):end]
    R = zeros(Float64, m - 1, m - 1)
    k = 1
    for jj in 1:(m - 1)
        for ii in 1:jj
            R[ii, jj] = sig_params[k]
            k += 1
        end
    end
    Sigma = R' * R                                 # (m-1)x(m-1)

    # full_sigma = cbind(Sigma, -rowSums(Sigma)) then rbind(., -colSums(.)) -> m x m
    rowsum = vec(sum(Sigma, dims = 2))             # length m-1
    fs_aug = hcat(Sigma, -rowsum)                  # (m-1) x m
    colsum = vec(sum(fs_aug, dims = 1))            # length m
    full_sigma = vcat(fs_aug, -reshape(colsum, 1, m))   # m x m

    # FULL regime distribution (deterministic).
    mvn_full = MvNormal(zeros(m - 1), Symmetric(Sigma))
    Sinv = inv(Symmetric(Sigma)); logdetS = logdet(Symmetric(Sigma))   # for the Jacobian path

    # Per-household contribution. Each is independent, so the loop below runs threaded or
    # serial (see `parallel`). Determinism + thread-independence: obs i uses its OWN
    # MersenneTwister(hash((seed, i))) for the orthant-probability Monte-Carlo, so serial
    # and parallel runs are bit-for-bit identical.
    function lf_of(i)
        nui = nu[i]

        if nui == m
            # ----: FULL regime — all goods purchased :----
            e = S[i, 1:(m - 1)] .- U[i, 1:(m - 1)]
            return logpdf(mvn_full, e)
        end

        if jacobian
            # ----: PARTIAL regimes WITH the share-renormalisation Jacobian (see _partial_ll_jac) :----
            rng_i = MersenneTwister(hash((seed, i)))
            return _partial_ll_jac(view(S, i, :), view(U, i, :), Sinv, logdetS, m, Int(mc_points),
                                   rng_i, floor_mode)
        end

        # ----: PARTIAL regimes, legacy (no Jacobian) — rearrange bought goods first :----
        # (views + findall(!,·) avoid the row-copy / negated-array allocations; numerically identical.)
        bght_index = findall(@view d[i, :])
        zero_index = findall(!, @view d[i, :])
        index_arrn = vcat(bght_index, zero_index)

        sigma = full_sigma[index_arrn, index_arrn]
        sigma = sigma[1:(m - 1), 1:(m - 1)]        # sorted small sigma

        S_a = S[i, index_arrn]                      # length m
        U_a = U[i, index_arrn]                      # length m (col vector)

        a = S_a[1:nui] ./ S_a[1]                    # length nui

        if nui == (m - 1)
            # ----: regime where all goods except one are bought :----
            AA1   = diagm(a)
            omega = AA1 * inv(sigma) * AA1'
            s11   = omega[1:nui, 1:nui]

            U_bar = U_a[1]
            II    = ones(nui)
            JJ    = U_a[1:nui] ./ (a .* U_a[1])

            omega11 = II' * s11 * II                # scalar (1x1)
            omega10 = II' * s11 * JJ                # scalar
            omega00 = JJ' * s11 * JJ                # scalar

            inv_omega11 = inv(omega11)
            U_sta       = (inv_omega11 * omega10) * U_bar     # scalar
            omg_final   = inv_omega11                          # scalar

            part1 = U_bar' * omega00 * U_bar - U_sta' * omega11 * U_sta   # scalar
            part2 = exp(-0.5 * part1) * (2 * pi)^(0.5 * (1 - nui)) *
                    (det(sigma)^(-0.5)) / (omg_final^(-0.5))

            # 1-D MVN orthant: pmvnorm(upper=-BB, sigma=diag(1)) == Phi(-BB).
            D_2 = sqrt(omg_final)                   # size m-nui == 1
            # RR = cov2cor(omg_final) == 1 in 1-D
            AA  = 1 / S_a[1]
            CC  = AA * D_2
            PP0 = 1.0
            PP  = PP0 - AA * U_sta
            diagD = (CC * 1.0 * CC)^(-0.5)
            DD = diagD
            BB = DD * PP
            # R_c = DD*CC*RR*CC'*DD (== 1) — not needed for the 1-D CDF.

            return _flog(part2, 1e-7, floor_mode) +
                   _flog(cdf(Normal(), -BB), 1e-7, floor_mode)

        else
            # ----: all other partial regimes (nui < m-1) :----
            onesvec = ones(m - nui - 1)
            AA1 = diagm(vcat(a, onesvec))
            omega = AA1 * inv(sigma) * AA1'

            s11 = omega[1:nui, 1:nui]
            s10 = omega[1:nui, (nui + 1):(m - 1)]
            s00 = omega[(nui + 1):(m - 1), (nui + 1):(m - 1)]

            Ubar = U_a[vcat(1, (nui + 1):(m - 1))]
            II   = ones(nui)
            JJ   = U_a[1:nui] ./ (a .* U_a[1])

            # NOTE: these are assembled exactly as the R source wrote them.
            omega11 = [II' * s11 * II    (II' * s10);
                       (s10' * II)        s00]
            omega10 = [II' * s11 * JJ    (II' * s10);
                       (s10' * JJ)        s00]
            omega00 = [JJ' * s11 * JJ    (JJ' * s10);
                       (s10' * JJ)        s00]

            inv_omega11 = inv(omega11)
            U_sta     = inv_omega11 * omega10 * Ubar
            omg_final = inv_omega11

            part1 = Ubar' * omega00 * Ubar - U_sta' * omega11 * U_sta   # 1x1
            part1 = part1[1]
            part2 = exp(-0.5 * part1) * (2 * pi)^(0.5 * (1 - nui)) *
                    (det(sigma)^(-0.5)) / (det(omg_final)^(-0.5))

            D_2 = diagm(sqrt.(diag(omg_final)))     # (m-nui) x (m-nui)
            RR  = cov2cor(omg_final)

            AA = Matrix(-1.0I, m - nui, m - nui)
            AA[1, :] = vcat(1 / S_a[1], ones(m - nui - 1))
            CC  = AA * D_2
            PP0 = vcat(1.0, zeros(m - nui - 1))
            PP  = PP0 - AA * U_sta

            diagD = zeros(m - nui)
            for kk in 1:(m - nui)
                row = CC[kk, :]
                diagD[kk] = (row' * RR * row)^(-0.5)
            end
            DD  = diagm(diagD)
            BB  = DD * PP
            R_c = DD * CC * RR * CC' * DD
            R_c = nearestPD(R_c)

            upper = -vec(BB)
            lower = fill(-Inf, length(upper))
            rng_i = MersenneTwister(hash((seed, i)))   # per-obs, thread-independent
            pr = mvnormcdf(R_c, lower, upper; m = Int(mc_points), rng = rng_i)[1]

            return _flog(part2, 1e-8, floor_mode) + _flog(pr, 1e-8, floor_mode)
        end
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
