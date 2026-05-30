#!/usr/bin/env julia
#
# Thin headless runner for the CensoredDemand job entrypoint.
#
#   julia --project=<pkg> bin/run.jl <config.json> [<output.json>]
#
# Reads a JSON config, calls `CensoredDemand.run_job`, and writes the result as
# pretty JSON to `config["output_json"]` (or ARGS[2] if given). Prints a
# one-line summary. This is the surface the AWS job layer + MCP server invoke.

using CensoredDemand
using JSON3

function main(args)
    isempty(args) && error("usage: julia --project=<pkg> bin/run.jl <config.json> [<output.json>]")

    config_path = args[1]
    isfile(config_path) || error("config not found: $(config_path)")

    # Parse the config JSON into a Dict (run_job tolerates Symbol/String keys).
    config = JSON3.read(read(config_path, String), Dict{String,Any})

    result = CensoredDemand.run_job(config)

    # Resolve output path: ARGS[2] wins, else config["output_json"], else default.
    out_path = length(args) >= 2 ? args[2] :
               get(config, "output_json", joinpath(dirname(abspath(config_path)), "results.json"))
    mkpath(dirname(abspath(out_path)))

    open(out_path, "w") do io
        JSON3.pretty(io, result)
    end

    status = get(result, "status", "?")
    if status == "ok"
        tasks = String[]
        haskey(result, "estimate")     && push!(tasks, "estimate")
        haskey(result, "elasticities") && push!(tasks, "elasticities")
        meta = get(result, "meta", Dict())
        elapsed = get(meta, "elapsed_s", "?")
        println("run_job: status=ok  n_obs=$(get(result, "n_obs", "?")) " *
                "n_goods=$(get(result, "n_goods", "?")) quaids=$(get(result, "quaids", "?")) " *
                "tasks=[$(join(tasks, ","))] elapsed_s=$(elapsed) -> $(out_path)")
    else
        println("run_job: status=error  message=$(get(result, "message", "?")) -> $(out_path)")
    end

    return status == "ok" ? 0 : 1
end

exit(main(ARGS))
