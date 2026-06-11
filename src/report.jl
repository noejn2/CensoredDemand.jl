# Reporting layer for CensoredDemand: human-readable `Base.show` for the result
# structs + tidy `DataFrame` tables (coef/se/t/p, labeled elasticities, MC recovery).
#
# Uses only Base string utilities for formatting (no Printf dependency) and the
# already-present DataFrames / Distributions deps. Includable into the module.

using DataFrames, Distributions

# Compact numeric formatter for table cells (returns "NA" for non-finite).
function _fmtnum(x; sig::Int = 4)
    (x isa Real && !isfinite(x)) && return "NA"
    return string(round(float(x); sigdigits = sig))
end

# ----------------------------------------------------------------------------
# EstimationResult
# ----------------------------------------------------------------------------

"""
    coeftable(r::EstimationResult) -> DataFrame

Coefficient table: `name`, `coef`, `se`, `t` (= coef/se), `p` (two-sided Normal).
One row per FULL parameter (incl. Σ), in packing order.
"""
function coeftable(r::EstimationResult)
    t = r.params ./ r.se
    p = [isfinite(ti) ? 2 * ccdf(Normal(), abs(ti)) : NaN for ti in t]
    return DataFrame(name = r.param_names, coef = r.params, se = r.se, t = t, p = p)
end

function Base.show(io::IO, ::MIME"text/plain", r::EstimationResult)
    sp = r.spec
    model = sp.quaids ? "QUAIDS" : "AIDS"
    println(io, "Censored $model demand system  ",
            "(m=$(sp.m) goods, t=$(sp.t) demographics, price_index=$(Symbol(sp.price_index)))")
    println(io, "  loglike = ", round(r.loglike; digits = 3),
            "   converged = ", r.converged,
            "   iters = ", r.iterations,
            "   mean|g|/n = ", round(r.gradnorm; sigdigits = 3),
            "   optimizer = ", r.optimizer)
    r.vcov_projected && println(io, "  (note: vcov was PSD-projected)")
    println(io)
    names = r.param_names
    wn = max(5, maximum(length, names))
    hdr = rpad("param", wn) * "  " * lpad("coef", 12) * "  " * lpad("se", 12) * "  " * lpad("t", 10)
    println(io, hdr)
    println(io, "-"^length(hdr))
    for i in eachindex(names)
        ti = r.params[i] / r.se[i]
        println(io, rpad(names[i], wn), "  ",
                lpad(_fmtnum(r.params[i]), 12), "  ",
                lpad(_fmtnum(r.se[i]), 12), "  ",
                lpad(_fmtnum(ti), 10))
    end
end

Base.show(io::IO, r::EstimationResult) =
    print(io, "EstimationResult($(r.spec.quaids ? "QUAIDS" : "AIDS"), ",
          "loglike=", round(r.loglike; digits = 2), ", converged=", r.converged, ")")

# ----------------------------------------------------------------------------
# ElasticityResult
# ----------------------------------------------------------------------------

# Column labels for an m×(m+1) elasticity matrix: price_<good> … then income.
_elast_colnames(sn::Vector{String}) = vcat(["price_$(g)" for g in sn], ["income"])

"""
    elasticity_table(r::ElasticityResult) -> DataFrame

Tidy long form: `good`, `response` (price_<good> or income), `elasticity`, `se`.
"""
function elasticity_table(r::ElasticityResult)
    sn = r.share_names
    cols = _elast_colnames(sn)
    m = length(sn)
    good = String[]; response = String[]; el = Float64[]; se = Float64[]
    for i in 1:m, c in 1:(m + 1)
        push!(good, sn[i]); push!(response, cols[c])
        push!(el, r.elasticities[i, c]); push!(se, r.se[i, c])
    end
    return DataFrame(good = good, response = response, elasticity = el, se = se)
end

function Base.show(io::IO, ::MIME"text/plain", r::ElasticityResult)
    sn = r.share_names; cols = _elast_colnames(sn); m = length(sn)
    println(io, "Censored demand elasticities  (",
            r.symmetric ? "Slutsky-symmetric" : "unconstrained", ", SE in parentheses)")
    wrow = max(8, maximum(length, sn))
    wcol = 18
    print(io, rpad("", wrow))
    for c in cols; print(io, lpad(c, wcol)); end
    println(io)
    for i in 1:m
        print(io, rpad(sn[i], wrow))
        for c in 1:(m + 1)
            cell = _fmtnum(r.elasticities[i, c]) * " (" * _fmtnum(r.se[i, c]) * ")"
            print(io, lpad(cell, wcol))
        end
        println(io)
    end
end

Base.show(io::IO, r::ElasticityResult) =
    print(io, "ElasticityResult($(length(r.share_names)) goods, ",
          r.symmetric ? "symmetric" : "unconstrained", ")")

# ----------------------------------------------------------------------------
# MonteCarloResult
# ----------------------------------------------------------------------------

"""
    montecarlo_table(r::MonteCarloResult) -> DataFrame

Per-parameter recovery: `name`, `truth`, `mean_est`, `bias`, `rmse`, `coverage`.
"""
function montecarlo_table(r::MonteCarloResult)
    mean_est = r.truth .+ r.bias
    return DataFrame(name = r.param_names, truth = r.truth, mean_est = mean_est,
                     bias = r.bias, rmse = r.rmse, coverage = r.coverage)
end

function Base.show(io::IO, ::MIME"text/plain", r::MonteCarloResult)
    reps = size(r.estimates, 1)
    sp = r.spec
    model = sp.quaids ? "QUAIDS" : "AIDS"
    println(io, "Monte-Carlo recovery — censored $model  (reps=$reps, ",
            "converged ", round(100 * r.converged_frac; digits = 1), "%)")
    names = r.param_names
    wn = max(5, maximum(length, names))
    hdr = rpad("param", wn) * "  " * lpad("truth", 11) * "  " * lpad("bias", 11) *
          "  " * lpad("rmse", 11) * "  " * lpad("cover", 8)
    println(io, hdr)
    println(io, "-"^length(hdr))
    for i in eachindex(names)
        println(io, rpad(names[i], wn), "  ",
                lpad(_fmtnum(r.truth[i]), 11), "  ",
                lpad(_fmtnum(r.bias[i]), 11), "  ",
                lpad(_fmtnum(r.rmse[i]), 11), "  ",
                lpad(_fmtnum(r.coverage[i]), 8))
    end
end

Base.show(io::IO, r::MonteCarloResult) =
    print(io, "MonteCarloResult(reps=", size(r.estimates, 1),
          ", converged=", round(100 * r.converged_frac; digits = 1), "%)")
