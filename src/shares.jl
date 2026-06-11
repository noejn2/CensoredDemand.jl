# Implementation A: AIDS / QUAIDS demand SHARE equations.
#
# Faithful port of censoredAIDS::aidsCalculate (R). Computes the n x m matrix of
# estimated budget shares for an AI or QUAI demand system, with optional
# demographic variables.
#
# Param vector order (m = #goods, t = #demographic columns):
#   [ alpha (m-1),
#     beta  (m-1),
#     gamma upper-triangular incl. diag, COLUMN-MAJOR fill, length 0.5*(m-1)*m,
#     theta (m-1)*t        (only if demographics, COLUMN-MAJOR),
#     lambda (m-1)         (only if quaids) ]
#
# Includable standalone (no module wrapper).

using LinearAlgebra

"""
    aids_shares(prices, budget, params; quaids=false, demographics=nothing) -> Matrix{Float64}

Compute the AIDS/QUAIDS demand share equations.

- `prices`        : n x m matrix of LOGGED prices.
- `budget`        : length-n vector of LOGGED total expenditure.
- `params`        : parameter vector (see module header for ordering).
- `quaids`        : if true, include the quadratic (QUAIDS) term.
- `demographics`  : `nothing` for no-demographics mode, otherwise an n x t matrix.

Returns an n x m matrix of estimated shares (column order = good order).
"""
function aids_shares(prices::AbstractMatrix, budget::AbstractVector, params::AbstractVector;
                     quaids::Bool=false, demographics=nothing,
                     price_index=:translog, shares=nothing)::Matrix{Float64}

    price_index = _price_index_sym(price_index)   # accept Symbol or PriceIndex enum
    Lnp = Matrix{Float64}(prices)              # n x m
    n, m = size(Lnp)
    Lnw = Vector{Float64}(budget)              # length n
    p   = Vector{Float64}(params)

    has_dems = demographics !== nothing
    t = has_dems ? size(demographics, 2) : 0

    # ----: Reconstruct full parameter blocks :----

    # Alpha: length m-1 -> append (1 - sum)
    Alpha = p[1:(m - 1)]
    full_alpha = vcat(Alpha, 1.0 - sum(Alpha))             # length m

    # Beta: length m-1 -> append (-sum)
    Beta = p[m:(2 * (m - 1))]
    full_beta = vcat(Beta, -sum(Beta))                     # length m

    # Gamma: upper-triangular (incl diag) of an (m-1)x(m-1) matrix, COLUMN-MAJOR.
    ngamma = Int(0.5 * (m - 1) * m)
    gstart = 2 * (m - 1) + 1
    Gamma = p[gstart:(gstart + ngamma - 1)]

    g = zeros(Float64, m - 1, m - 1)
    # R: full_gamma[upper.tri(full_gamma, diag = TRUE)] <- Gamma
    # upper.tri index assignment fills in COLUMN-MAJOR order: for each column j,
    # rows i = 1..j (i <= j).
    k = 1
    for j in 1:(m - 1)
        for i in 1:j
            g[i, j] = Gamma[k]
            k += 1
        end
    end
    # Mirror to lower triangle to make it symmetric.
    for j in 1:(m - 1)
        for i in (j + 1):(m - 1)
            g[i, j] = g[j, i]
        end
    end
    # cbind(g, -rowSums(g)) then rbind(., -colSums(.)) -> m x m.
    rowsum = vec(sum(g, dims = 2))                         # length m-1
    g_aug = hcat(g, -rowsum)                               # (m-1) x m
    colsum = vec(sum(g_aug, dims = 1))                     # length m
    full_gamma = vcat(g_aug, -reshape(colsum, 1, m))       # m x m

    nn1 = (m - 1) * (2 + 0.5 * m)
    nn1 = Int(nn1)

    # Theta (only if demographics). COLUMN-MAJOR fill into a t x (m-1) block.
    full_theta = nothing  # will be m x t after transpose
    if has_dems
        Theta = p[(nn1 + 1):(nn1 + t * (m - 1))]
        ft = zeros(Float64, t, m)                          # t x m
        # R: full_theta[1:t, 1:(m-1)] <- Theta  (column-major over t x (m-1))
        kk = 1
        for j in 1:(m - 1)
            for i in 1:t
                ft[i, j] = Theta[kk]
                kk += 1
            end
        end
        ft[:, m] = -vec(sum(ft, dims = 2))                 # last col = -rowSums
        full_theta = permutedims(ft)                       # m x t  (R: t(full_theta))
    end

    # Lambda (only if quaids).
    full_lambda = nothing
    if quaids
        nn2 = has_dems ? (nn1 + t * (m - 1)) : nn1
        Lambda = p[(nn2 + 1):(nn2 + m - 1)]
        full_lambda = vcat(Lambda, -sum(Lambda))           # length m
    end

    # ----: Price index (per row) :----
    # `:translog` (default) = full QUAIDS index ln a(p) = p'α + 0.5 p'Γp (nonlinear in α,Γ).
    # `:stone`   = LA-AIDS Stone index ln P* = Σ wₖ ln pₖ — uses observed shares, so it is
    #             predetermined and linearizes the share equations ("limits the nonlinearity").
    Gp = Lnp * full_gamma                                  # n x m (Γ·lnp term; used in the shares below)
    if price_index === :stone
        shares === nothing && error("aids_shares: price_index=:stone requires observed `shares`")
        Wobs = Matrix{Float64}(shares)
        size(Wobs) == (n, m) || error("aids_shares: `shares` must be $(n)×$(m) for the Stone index")
        lnpindex = vec(sum(Wobs .* Lnp, dims = 2))
    elseif price_index === :translog
        lnpindex = Lnp * full_alpha                        # length n
        @inbounds for i in 1:n
            acc = 0.0
            for jj in 1:m
                acc += Lnp[i, jj] * Gp[i, jj]
            end
            lnpindex[i] += 0.5 * acc
        end
    else
        error("aids_shares: price_index must be :translog or :stone, got :$price_index")
    end

    # ----: Shares :----
    # qshare = ones_n * full_alpha' + Lnp*full_gamma
    W = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m
        a = full_alpha[j]
        for i in 1:n
            W[i, j] = a + Gp[i, j]
        end
    end

    # + (ones_n*full_beta' + Z*full_theta') .* (Lnw - lnpindex)
    if has_dems
        Z = Matrix{Float64}(demographics)                  # n x t
        ZT = Z * permutedims(full_theta)                   # n x m  (full_theta is m x t -> theta' is t x m)
        @inbounds for j in 1:m
            bj = full_beta[j]
            for i in 1:n
                W[i, j] += (bj + ZT[i, j]) * (Lnw[i] - lnpindex[i])
            end
        end
    else
        @inbounds for j in 1:m
            bj = full_beta[j]
            for i in 1:n
                W[i, j] += bj * (Lnw[i] - lnpindex[i])
            end
        end
    end

    # + (quaids) (ones_n*full_lambda' ./ exp(Lnp*full_beta)) .* (Lnw - lnpindex).^2
    if quaids
        bofp = exp.(Lnp * full_beta)                       # length n
        @inbounds for j in 1:m
            lj = full_lambda[j]
            for i in 1:n
                d = Lnw[i] - lnpindex[i]
                W[i, j] += (lj / bofp[i]) * d * d
            end
        end
    end

    return W
end
