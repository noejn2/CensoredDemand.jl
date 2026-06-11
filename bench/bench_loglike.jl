# Benchmark + cross-thread bit-identity check for the censored log-likelihood.
#
# Run with varying thread counts and confirm the SUM is identical across them
# (the per-observation RNG makes the result thread-independent), while wall-time drops:
#
#   julia -t1 --project=. bench/bench_loglike.jl
#   julia -t4 --project=. bench/bench_loglike.jl
#   julia -t8 --project=. bench/bench_loglike.jl
#
# Expected: identical `sum` on every run (e.g. -4511.663045501727 at mc_points=8000),
# wall-time ~5-6x lower at 8 threads. See M5/M6 in PLAN.md.

using CensoredDemand, DelimitedFiles, Printf

const FIX = joinpath(@__DIR__, "..", "test", "fixtures")

td, h = readdlm(joinpath(FIX, "testing_data.csv"), ',', header = true)
hh = [strip(string(x), ['"']) for x in vec(h)]
col(name) = Float64.(td[:, findfirst(==(name), hh)])
S = hcat(col("s1"), col("s2"), col("s3"), col("s4"))
P = hcat(col("lnp1"), col("lnp2"), col("lnp3"), col("lnp4"))
b = col("lnw")
Z = hcat(col("age"), col("size"), col("educ"), col("sex"))
params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

f() = censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 8000)

ll = f()                                   # warmup / compile
t  = minimum(@elapsed(f()) for _ in 1:4)   # best of 4
@printf("threads=%d  sum=%.12f  time=%.4fs\n", Threads.nthreads(), sum(ll), t)
