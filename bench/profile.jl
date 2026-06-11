# Performance baseline for CensoredDemand (Change 7, profile-first).
#
# Measures: per-evaluation time + allocations of `censored_loglike` (the hot path that
# dominates estimation, evaluated O(2p) times per BHHH iteration), the censoring-regime
# mix, and a full BHHH MLE timing. Run with:  julia --project -t8 bench/profile.jl
# stdlib only (@time/@allocated/Base.summarysize) — no BenchmarkTools dependency.

using CensoredDemand, DelimitedFiles, Statistics, Printf

FIX = joinpath(@__DIR__, "..", "test", "fixtures")
td, h = readdlm(joinpath(FIX, "testing_data.csv"), ',', header = true)
hh = [strip(string(x), ['"']) for x in vec(h)]
col(n) = Float64.(td[:, findfirst(==(n), hh)])
S = hcat(col("s1"), col("s2"), col("s3"), col("s4"))
P = hcat(col("lnp1"), col("lnp2"), col("lnp3"), col("lnp4"))
b = col("lnw"); Z = hcat(col("age"), col("size"), col("educ"), col("sex"))
params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

n = size(S, 1)
nu = vec(sum(S .!= 0.0, dims = 2))
println("threads = ", Threads.nthreads(), "   n = ", n)
println("regime mix (goods bought → count): ",
        [(k, count(==(k), nu)) for k in sort(unique(nu))])

# warmup (compile)
censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 2000)

evals = 10
t = @elapsed for _ in 1:evals
    censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 2000)
end
alloc = @allocated censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 2000)
@printf("loglike eval: %.2f ms/eval (mc=2000, %d threads) | %.2f MB alloc/eval\n",
        1000 * t / evals, Threads.nthreads(), alloc / 2^20)

# serial single eval allocations (allocation count is thread-independent but clearer here)
ts = @elapsed for _ in 1:evals
    censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 2000, parallel = false)
end
@printf("loglike eval (serial): %.2f ms/eval\n", 1000 * ts / evals)

# full MLE timing from the principled start — central (default, most accurate) vs forward (~2× evals)
iv = initial_values(S, P, b; quaids = true, demographics = Z)
estimate(S, P, b; quaids = true, demographics = Z, start = iv, mc_points = 500, maxiters = 2)  # warmup
for mode in (:central, :forward)
    tm = @elapsed res = estimate(S, P, b; quaids = true, demographics = Z, start = iv,
                                 mc_points = 2000, maxiters = 300, g_tol = 1e-4, fd_mode = mode)
    @printf("full BHHH MLE (%-8s): %5.1f s | iters=%d converged=%s loglike=%.2f mean|g|/n=%.2g\n",
            mode, tm, res.iterations, res.converged, res.loglike, res.gradnorm)
end
