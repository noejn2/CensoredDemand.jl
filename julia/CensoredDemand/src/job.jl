# HEADLESS job entrypoint for the CensoredDemand package.
#
# `run_job(config::AbstractDict)::Dict` drives the package APIs (estimate,
# censored_elasticity) from a plain config dictionary — the same shape the AWS
# job layer + MCP server will hand over as parsed JSON. Everything is wrapped in
# a try/catch so a bad config or numerical failure surfaces as
# `status="error"` rather than throwing across the FFI/process boundary.
#
# Includable standalone (no module wrapper). Assumes `estimate`,
# `censored_elasticity`, `aids_shares`, `censored_loglike` and the deps
# CSV / DataFrames / JSON3 are already in scope.

using CSV, DataFrames, JSON3

# ----------------------------------------------------------------------------
# Small config helpers
# ----------------------------------------------------------------------------

# Fetch `key` from a config dict tolerating String / Symbol keys (JSON3 parses
# to Symbol keys by default; a hand-built Dict may use String keys).
function _cfg_get(cfg::AbstractDict, key::AbstractString, default)
    haskey(cfg, key)          && return cfg[key]
    haskey(cfg, Symbol(key))  && return cfg[Symbol(key)]
    return default
end

_has_key(cfg::AbstractDict, key::AbstractString) =
    haskey(cfg, key) || haskey(cfg, Symbol(key))

# Coerce a config value that should be a vector of column-name strings.
function _as_str_vec(x)
    x === nothing && return String[]
    return String[String(s) for s in x]
end

# Coerce a config value that should be a vector of Float64 (params / inlined nums).
function _as_float_vec(x)
    x === nothing && return nothing
    return Float64[Float64(v) for v in x]
end

# Serialize a matrix as a row-major array-of-arrays (JSON-friendly).
_rows(M::AbstractMatrix) = [collect(Float64.(M[i, :])) for i in 1:size(M, 1)]

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

"""
    run_job(config::AbstractDict) -> Dict

Run a censored AIDS / QUAIDS job described by `config` (parsed JSON / Dict).

Recognised keys (snake_case):
- `data_csv`         : path to the input CSV.
- `share_cols`       : array of share column names (n×m).
- `logprice_cols`    : array of logged-price column names (n×m).
- `budget_col`       : logged-expenditure column name.
- `demographic_cols` : optional array of demographic column names (order is
                       load-bearing); omit/empty => no demographics.
- `quaids`           : include the quadratic term (default `false`).
- `tasks`            : subset of `["estimate", "elasticities"]`.
- `params`           : optional full parameter vector (used for elasticities
                       when not estimating, or as the estimate warm start).
- `vcov_csv`         : optional path to a headerless vcov CSV (for elasticities
                       when not estimating).
- `mc_points`        : QMC budget for estimation (default 2000).
- `floor_mode`       : likelihood floor mode string (default "additive_r").
- `maxiters`         : optimizer iteration cap (default 500).
- `reps`             : Monte-Carlo reps for elasticities (default 100000).
- `share_names`      : optional pretty good labels echoed in the elasticities block.

Returns a `Dict` with `status`, observation/good counts, per-task sub-dicts,
and a `meta` block. On failure returns `Dict("status" => "error", ...)`.
"""
function run_job(config::AbstractDict)::Dict
    t_start = time()
    try
        # ---- config ----
        data_csv   = _cfg_get(config, "data_csv", nothing)
        data_csv === nothing && error("config: `data_csv` is required")

        share_cols = _as_str_vec(_cfg_get(config, "share_cols", nothing))
        price_cols = _as_str_vec(_cfg_get(config, "logprice_cols", nothing))
        budget_col = _cfg_get(config, "budget_col", nothing)
        budget_col === nothing && error("config: `budget_col` is required")
        budget_col = String(budget_col)

        demo_cols  = _as_str_vec(_cfg_get(config, "demographic_cols", nothing))
        quaids     = Bool(_cfg_get(config, "quaids", false))
        tasks      = _as_str_vec(_cfg_get(config, "tasks", String[]))
        cfg_params = _as_float_vec(_cfg_get(config, "params", nothing))
        vcov_csv   = _cfg_get(config, "vcov_csv", nothing)
        mc_points  = Int(_cfg_get(config, "mc_points", 2000))
        floor_mode = Symbol(String(_cfg_get(config, "floor_mode", "additive_r")))
        maxiters   = Int(_cfg_get(config, "maxiters", 500))
        reps       = Int(_cfg_get(config, "reps", 100_000))
        share_names = _as_str_vec(_cfg_get(config, "share_names", nothing))

        isempty(share_cols) && error("config: `share_cols` must be non-empty")
        isempty(price_cols) && error("config: `logprice_cols` must be non-empty")
        isempty(tasks)      && error("config: `tasks` must be non-empty")

        # ---- load data ----
        isfile(String(data_csv)) || error("data_csv not found: $(data_csv)")
        df = CSV.read(String(data_csv), DataFrame)

        getcol(c) = begin
            hasproperty(df, Symbol(c)) || error("column `$(c)` not found in data_csv")
            Float64.(df[!, Symbol(c)])
        end

        S = hcat((getcol(c) for c in share_cols)...)   # n×m raw shares
        P = hcat((getcol(c) for c in price_cols)...)   # n×m logged prices
        b = getcol(budget_col)                          # length-n logged budget
        Z = isempty(demo_cols) ? nothing :
            hcat((getcol(c) for c in demo_cols)...)     # n×t demographics or nothing

        n, m = size(S)
        size(P, 2) == m || error("share_cols ($(m)) and logprice_cols ($(size(P,2))) imply different #goods")
        ndemo = Z === nothing ? 0 : size(Z, 2)

        result = Dict{String,Any}(
            "status"   => "ok",
            "n_obs"    => n,
            "n_goods"  => m,
            "quaids"   => quaids,
        )

        # ---- estimate ----
        params_hat = nothing
        vcov_hat   = nothing
        if "estimate" in tasks
            res = estimate(S, P, b;
                           quaids = quaids, demographics = Z,
                           start = cfg_params,
                           mc_points = mc_points, maxiters = maxiters,
                           floor_mode = floor_mode)
            params_hat = collect(Float64.(res.params))
            vcov_hat   = Matrix{Float64}(res.vcov)
            result["estimate"] = Dict{String,Any}(
                "params"     => params_hat,
                "loglike"    => Float64(res.loglike),
                "converged"  => Bool(res.converged),
                "iterations" => Int(res.iterations),
                "vcov"       => _rows(vcov_hat),
                "se"         => collect(Float64.(res.se)),
            )
        end

        # ---- elasticities ----
        if "elasticities" in tasks
            params_e = params_hat !== nothing ? params_hat : cfg_params
            params_e === nothing &&
                error("elasticities: need estimated params or config `params`")

            vcov_e = vcov_hat
            if vcov_e === nothing
                vcov_csv === nothing &&
                    error("elasticities: need estimated vcov or config `vcov_csv`")
                isfile(String(vcov_csv)) || error("vcov_csv not found: $(vcov_csv)")
                vcov_e = Matrix{Float64}(CSV.read(String(vcov_csv), DataFrame;
                                                  header = false) |> Matrix)
            end

            el = censored_elasticity(P, b, params_e;
                                     quaids = quaids, demographics = Z,
                                     vcov = vcov_e, reps = reps)

            ela = Dict{String,Any}(
                "elasticities" => _rows(el.elasticities),
                "se"           => _rows(el.se),
                "e_uobs"       => collect(Float64.(el.e_uobs)),
            )
            if !isempty(share_names)
                ela["share_names"] = share_names
            end
            result["elasticities"] = ela
        end

        result["meta"] = Dict{String,Any}(
            "julia_version" => string(VERSION),
            "n_demographics" => ndemo,
            "mc_points"      => mc_points,
            "floor_mode"     => String(floor_mode),
            "reps"           => reps,
            "elapsed_s"      => round(time() - t_start; digits = 3),
        )
        return result

    catch err
        io = IOBuffer()
        showerror(io, err)
        return Dict{String,Any}(
            "status"  => "error",
            "message" => String(take!(io)),
            "meta"    => Dict{String,Any}(
                "julia_version" => string(VERSION),
                "elapsed_s"     => round(time() - t_start; digits = 3),
            ),
        )
    end
end
