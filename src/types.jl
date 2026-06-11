# Typed surface for CensoredDemand: option ENUMS + result STRUCTS.
#
# Design: this is an additive, typed *boundary*. The numerical core keeps
# comparing canonical Symbols (`=== :stone`, `=== :guard`), so internals stay
# byte-identical; the enums exist for type-safety / clarity and accept Symbols
# everywhere for back-compat. The result structs carry the SAME field names the
# functional core used to return as NamedTuples, so every existing caller and
# test keeps working unchanged.
#
# Concrete field types throughout: these structs must NOT introduce type
# instability (see PLAN.md Change 7 — enums/structs prevent pessimization).

# ----------------------------------------------------------------------------
# Option enums (with Symbol back-compat)
# ----------------------------------------------------------------------------

"""
    PriceIndex

Demand-system price deflator: `TRANSLOG` (full QUAIDS translog index, R-faithful)
or `STONE` (LA-AIDS Stone index — predetermined, linearizes the share equations).
Accepts/returns the Symbols `:translog` / `:stone` for back-compat.
"""
@enum PriceIndex TRANSLOG STONE

"""
    FloorMode

Partial-regime log floor: `ADDITIVE_R` (R's verbatim `log(x + 1e-7/1e-8)`) or
`GUARD` (principled `log(max(x, 1e-300))`). Accepts/returns `:additive_r`/`:guard`.
"""
@enum FloorMode ADDITIVE_R GUARD

function PriceIndex(s::Symbol)
    s === :translog && return TRANSLOG
    s === :stone    && return STONE
    error("price_index must be :translog or :stone (or a PriceIndex), got :$s")
end
Base.Symbol(p::PriceIndex) = p === TRANSLOG ? :translog : :stone

function FloorMode(s::Symbol)
    s === :additive_r && return ADDITIVE_R
    s === :guard      && return GUARD
    error("floor_mode must be :additive_r or :guard (or a FloorMode), got :$s")
end
Base.Symbol(f::FloorMode) = f === ADDITIVE_R ? :additive_r : :guard

# Normalize a Symbol|enum option to the canonical Symbol used internally
# (also validates the input). Defined for both arms so call sites can pass either.
_price_index_sym(x::PriceIndex) = Symbol(x)
_price_index_sym(x::Symbol)     = Symbol(PriceIndex(x))     # validate then canonicalize
_floor_mode_sym(x::FloorMode)   = Symbol(x)
_floor_mode_sym(x::Symbol)      = Symbol(FloorMode(x))

# ----------------------------------------------------------------------------
# Model specification (dimensions + labels)
# ----------------------------------------------------------------------------

"""
    ModelSpec

The shape of a censored demand model: number of goods `m`, demographic count `t`,
`quaids` flag, `price_index`, whether demographics are present, and human-readable
`share_names` / `demographic_names` (used by the reporting layer).
"""
struct ModelSpec
    m::Int
    t::Int
    quaids::Bool
    price_index::PriceIndex
    has_dems::Bool
    share_names::Vector{String}
    demographic_names::Vector{String}
end

# Convenience builder with sensible default labels.
function ModelSpec(m::Integer, t::Integer; quaids::Bool, price_index = :translog,
                   share_names = nothing, demographic_names = nothing)
    sn = share_names === nothing ? ["good$(i)" for i in 1:m] : String.(share_names)
    dn = demographic_names === nothing ? ["demo$(i)" for i in 1:t] : String.(demographic_names)
    length(sn) == m || error("share_names must have length m=$m")
    length(dn) == t || error("demographic_names must have length t=$t")
    return ModelSpec(Int(m), Int(t), quaids, PriceIndex(Symbol(price_index)), t > 0, sn, dn)
end

# ----------------------------------------------------------------------------
# Result structs (field names preserved for drop-in compatibility)
# ----------------------------------------------------------------------------

"""
    EstimationResult

Maximum-likelihood fit of a censored AIDS/QUAIDS system (BHHH optimizer). Carries
the same field names the previous NamedTuple exposed (`params, loglike, vcov, se,
converged, iterations, optimizer`) plus `opg`, `vcov_projected`, `gradnorm`, `nll`,
`param_names`, `spec`, and the `start_check` diagnostics.
"""
struct EstimationResult
    params::Vector{Float64}
    loglike::Float64
    vcov::Matrix{Float64}
    se::Vector{Float64}
    opg::Matrix{Float64}
    vcov_projected::Bool
    converged::Bool
    iterations::Int
    gradnorm::Float64
    nll::Float64
    optimizer::Symbol
    param_names::Vector{String}
    spec::ModelSpec
    start_check::Any
end

"""
    ElasticityResult

Simulation-based censored elasticities. `elasticities` is m×(m+1) (price cols 1..m
then income), `se` the matching delta-method SEs, `e_uobs` the expected
Amemiya–Tobin observed shares. `symmetric` records whether Slutsky symmetry was
imposed; `share_names` labels the rows/price-columns.
"""
struct ElasticityResult
    elasticities::Matrix{Float64}
    se::Matrix{Float64}
    e_uobs::Vector{Float64}
    symmetric::Bool
    share_names::Vector{String}
end

"""
    SimData

A synthetic dataset drawn from known coefficients: censored `shares` (exact zeros
mark non-purchase), the simulated `prices`/`budget`/`demographics`, and the true
`params`/`spec` that generated them.
"""
struct SimData
    shares::Matrix{Float64}
    prices::Matrix{Float64}
    budget::Vector{Float64}
    demographics::Union{Nothing,Matrix{Float64}}
    params::Vector{Float64}
    spec::ModelSpec
end

"""
    MonteCarloResult

Sampling-distribution summary of the estimator over `reps` simulate→estimate
replications: the `truth`, the `estimates` matrix (reps×p), per-parameter `bias`,
`rmse`, and `coverage` (share of reps with truth ∈ θ̂ ± 1.96·se), the
`converged_frac`, and `param_names`/`spec`.
"""
struct MonteCarloResult
    truth::Vector{Float64}
    estimates::Matrix{Float64}
    bias::Vector{Float64}
    rmse::Vector{Float64}
    coverage::Vector{Float64}
    converged_frac::Float64
    param_names::Vector{String}
    spec::ModelSpec
end

# ----------------------------------------------------------------------------
# Parameter names (packing order = start.jl / shares.jl)
# ----------------------------------------------------------------------------

"""
    param_names(spec::ModelSpec) -> Vector{String}

Human-readable names for the FULL parameter vector, in the exact packing order
used everywhere (α, β, γ upper-tri column-major, θ good-outer/demo-inner, λ, σ
upper-tri column-major). Length equals the full parameter vector (incl. Σ).
"""
function param_names(spec::ModelSpec)
    m, t = spec.m, spec.t
    g = spec.share_names
    names = String[]
    for i in 1:(m - 1); push!(names, "α_$(g[i])"); end
    for i in 1:(m - 1); push!(names, "β_$(g[i])"); end
    for j in 1:(m - 1), i in 1:j                       # γ upper-tri, column-major
        push!(names, "γ_$(g[i]),$(g[j])")
    end
    if spec.has_dems
        for i in 1:(m - 1), k in 1:t                   # θ good-outer, demo-inner
            push!(names, "θ_$(g[i])×$(spec.demographic_names[k])")
        end
    end
    if spec.quaids
        for i in 1:(m - 1); push!(names, "λ_$(g[i])"); end
    end
    for j in 1:(m - 1), i in 1:j                       # σ upper-tri, column-major
        push!(names, "σ_$(g[i]),$(g[j])")
    end
    return names
end
