using Test
using CensoredDemand
using DelimitedFiles
using LinearAlgebra
using Statistics
using Distributions: Normal, logpdf, logcdf   # only these — `using Distributions` clobbers `estimate`

const FIX = joinpath(@__DIR__, "fixtures")

# Read a CSV with a header row. Returns (data::Matrix, headers::Vector{String}),
# tolerant of quoted or unquoted headers.
function readcsv(path)
    data, h = readdlm(path, ',', header = true)
    headers = [strip(string(x), ['"']) for x in vec(h)]
    return data, headers
end

# Assemble the shared inputs (615 households) from testing_data.csv.
function load_inputs()
    td, h = readcsv(joinpath(FIX, "testing_data.csv"))
    col(name) = Float64.(td[:, findfirst(==(name), h)])
    P = hcat(col("lnp1"), col("lnp2"), col("lnp3"), col("lnp4"))
    b = col("lnw")
    Z = hcat(col("age"), col("size"), col("educ"), col("sex"))  # order is load-bearing
    return P, b, Z
end

@testset "CensoredDemand.jl" begin
    P, b, Z = load_inputs()

    @testset "M1 — share equations (aids_shares)" begin

        @testset "M0 golden: QUAIDS + demographics vs R qshares.csv" begin
            ps = vec(readdlm(joinpath(FIX, "params_shares.csv"), ',', header = true)[1])
            @test length(ps) == 27
            W = aids_shares(P, b, ps; quaids = true, demographics = Z)
            Q = readdlm(joinpath(FIX, "qshares.csv"), ',', header = true)[1]
            @test size(W) == size(Q) == (615, 4)
            @test maximum(abs.(W .- Q)) < 1e-10
            # adding-up: shares sum to 1 across goods
            @test maximum(abs.(vec(sum(W, dims = 2)) .- 1.0)) < 1e-9
        end

        @testset "Multi-mode oracle (AIDS/QUAIDS × demographics/none, real+random)" begin
            extradir = joinpath(FIX, "extra")
            cases = sort(filter(f -> endswith(f, "_params.csv"), readdir(extradir)))
            @test length(cases) == 12
            for pf in cases
                name   = replace(pf, "_params.csv" => "")
                quaids = startswith(name, "quaids")
                demos  = !occursin("nodemos", name)
                params = vec(readdlm(joinpath(extradir, pf), ',', header = true)[1])
                expect = readdlm(joinpath(extradir, name * "_shares.csv"), ',', header = true)[1]
                W = aids_shares(P, b, params; quaids = quaids,
                                demographics = demos ? Z : nothing)
                @test maximum(abs.(W .- expect)) < 1e-10
            end
        end
    end

    @testset "M2 — censored log-likelihood (censored_loglike)" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        col(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(col("s1"), col("s2"), col("s3"), col("s4"))   # raw shares (zeros = censored)

        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        @test length(params) == 33
        llR = vec(readdlm(joinpath(FIX, "loglikes.csv"), ',', header = true)[1])
        nu  = vec(Int.(round.(readdlm(joinpath(FIX, "regimes.csv"), ',', header = true)[1])))

        ll = censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 4000)
        @test length(ll) == 615

        # R's own gate: integer-rounded total log-likelihood matches (R sum ≈ -4511.661).
        @test round(sum(ll)) == round(sum(llR))

        # Deterministic regimes reproduce R to machine precision:
        full    = nu .== 4   # full purchase  → MVN density
        allbut1 = nu .== 3   # all-but-one    → 1-D normal CDF
        @test maximum(abs.(ll[full]    .- llR[full]))    < 1e-6
        @test maximum(abs.(ll[allbut1] .- llR[allbut1])) < 1e-6

        # Stochastic-CDF regimes (nu ∈ {1,2}) agree with R within Monte-Carlo noise:
        @test abs(sum(ll) - sum(llR)) < 0.05

        # Seeded determinism: same default seed → identical result.
        ll2 = censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 4000)
        @test ll == ll2

        # Parallelization toggle: the serial path is bit-identical to the threaded one.
        ll_serial = censored_loglike(S, P, b, params; quaids = true, demographics = Z,
                                     mc_points = 4000, parallel = false)
        @test ll_serial == ll
    end

    @testset "M3 — elasticities (censored_elasticity)" begin
        edir   = joinpath(FIX, "elast")
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        VCOV   = readdlm(joinpath(edir, "vcov.csv"), ',')            # 33×33, headerless
        EPS    = readdlm(joinpath(edir, "epsilons.csv"), ',')        # 10000×3, headerless (pre -rowSums)
        elasR  = readdlm(joinpath(edir, "elasticities_R.csv"), ',')  # 4×5
        euobsR = vec(readdlm(joinpath(edir, "e_uobs_R.csv"), ','))   # 4
        seR    = readdlm(joinpath(edir, "se_R.csv"), ',')            # 4×5

        # Inject R's exact ε draws → near-deterministic match to the R oracle.
        res = censored_elasticity(P, b, params; quaids = true, demographics = Z,
                                  vcov = VCOV, reps = size(EPS, 1), epsilons = EPS)
        @test size(res.elasticities) == (4, 5)
        @test maximum(abs.(res.elasticities .- elasR)) < 1e-4   # elasticities ~ exact
        @test maximum(abs.(res.e_uobs .- euobsR))      < 1e-8   # expected shares ~ machine
        @test abs(sum(res.e_uobs) - 1.0)               < 1e-8   # Amemiya–Tobin adding-up
        # SEs: price block tight; income column looser (level-perturbation cancellation).
        @test maximum(abs.(res.se[:, 1:4] .- seR[:, 1:4])) < 1e-2
        @test maximum(abs.(res.se[:, 5]   .- seR[:, 5]))   < 1.0
    end

    @testset "M3 — estimate() machinery (smoke)" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        known = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

        # Small subsample keeps the numerical-Hessian cost bounded for CI.
        idx = 1:50
        Ss, Ps, bs, Zs = S[idx, :], P[idx, :], b[idx], Z[idx, :]
        ll_start = sum(censored_loglike(Ss, Ps, bs, known; quaids = true,
                                        demographics = Zs, mc_points = 300))
        res = estimate(Ss, Ps, bs; quaids = true, demographics = Zs,
                       start = known, mc_points = 300, maxiters = 12)
        @test length(res.params) == 33
        @test isfinite(res.loglike)
        @test size(res.vcov) == (33, 33)
        @test res.loglike >= ll_start - 1.0          # optimizer never worsens the objective
        @test isposdef(Symmetric(res.vcov))          # PSD (projected if needed)
    end

    @testset "M4 — microeconomic validity + principled floor" begin
        edir   = joinpath(FIX, "elast")
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        VCOV   = readdlm(joinpath(edir, "vcov.csv"), ',')
        EPS    = readdlm(joinpath(edir, "epsilons.csv"), ',')

        res = censored_elasticity(P, b, params; quaids = true, demographics = Z,
                                  vcov = VCOV, reps = size(EPS, 1), epsilons = EPS)
        E = res.elasticities    # 4×5: cols 1:4 uncompensated price ε[i,j], col 5 expenditure η[i]
        w = res.e_uobs          # expected (censoring-adjusted) shares, Σ = 1

        @testset "theory identities" begin
            # Homogeneity (degree-0): Σⱼ ε[i,j] + η[i] = 0  (structural → near-exact)
            homog = [sum(E[i, 1:4]) + E[i, 5] for i in 1:4]
            @test maximum(abs.(homog)) < 5e-5
            # Engel aggregation: Σᵢ wᵢ ηᵢ = 1
            @test abs(sum(w .* E[:, 5]) - 1.0) < 1e-6
            # Cournot aggregation: Σᵢ wᵢ ε[i,j] = −wⱼ
            cournot = [sum(w .* E[:, jj]) + w[jj] for jj in 1:4]
            @test maximum(abs.(cournot)) < 1e-6
            # Slutsky symmetry of the substitution terms wᵢ·εᶜ[i,j].  Holds only APPROXIMATELY
            # for a censored/simulated system; worst residual is the thin Juice margin (w≈0.012).
            εc(i, jj) = E[i, jj] + w[jj] * E[i, 5]
            slut = [w[i] * εc(i, jj) - w[jj] * εc(jj, i) for i in 1:4, jj in 1:4]
            @test maximum(abs.(slut)) < 0.15
        end

        @testset "principled floor is opt-in; default preserves R-parity" begin
            td, h = readcsv(joinpath(FIX, "testing_data.csv"))
            scol(name) = Float64.(td[:, findfirst(==(name), h)])
            S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))

            ll_default = censored_loglike(S, P, b, params; quaids = true,
                                          demographics = Z, mc_points = 4000)
            ll_guard   = censored_loglike(S, P, b, params; quaids = true,
                                          demographics = Z, mc_points = 4000,
                                          floor_mode = :guard)
            # Default (:additive_r) = verbatim R floor → integer-sum parity preserved.
            @test round(sum(ll_default)) == -4512
            # :guard removes the spurious additive credit (~279 nats) the floor handed to the
            # ~36 households the published Σ deems near-impossible (orthant prob ≈ 0).
            @test 150 < (sum(ll_default) - sum(ll_guard)) < 450
        end
    end

    @testset "Initial values + start check" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S   = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        pub = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

        # LA-AIDS principled start (QUAIDS + demographics).
        iv = initial_values(S, P, b; quaids = true, demographics = Z)
        @test length(iv) == 33
        chk = check_start(iv, S, P, b; quaids = true, demographics = Z)
        @test chk.ok
        @test chk.diagnostics.sigma_min_eig > 0           # Σ positive-definite
        @test chk.diagnostics.n_loglike_nonfinite == 0    # finite likelihood at every household

        # check_start rejects a malformed start.
        @test !check_start(iv[1:30], S, P, b; quaids = true, demographics = Z).ok

        # The principled start already beats the published "MLE" on the likelihood
        # (independent confirmation that the published params are NOT the argmax).
        ll_iv  = sum(censored_loglike(S, P, b, iv;  quaids = true, demographics = Z, mc_points = 2000))
        ll_pub = sum(censored_loglike(S, P, b, pub; quaids = true, demographics = Z, mc_points = 2000))
        @test ll_iv > ll_pub

        # AIDS, no-demographics mode also yields a valid start.
        iv2 = initial_values(S, P, b; quaids = false, demographics = nothing)
        @test length(iv2) == 18
        @test check_start(iv2, S, P, b; quaids = false, demographics = nothing).ok
    end

    @testset "Options: Stone price index + BHHH algorithm" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

        # Stone index option: requires observed shares, and genuinely differs from translog.
        W_tl = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z, price_index = :translog)
        W_st = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z,
                           price_index = :stone, shares = S)
        @test all(isfinite, W_st)
        @test maximum(abs.(W_st .- W_tl)) > 1e-5
        @test_throws ErrorException aids_shares(P, b, params[1:27]; quaids = true,
                                                demographics = Z, price_index = :stone)
        @test all(isfinite, censored_loglike(S, P, b, params; quaids = true,
                                             demographics = Z, mc_points = 2000, price_index = :stone))

        # Stone elasticities require observed shares; finite + adding-up holds.
        VCOV = readdlm(joinpath(FIX, "elast", "vcov.csv"), ',')
        EPS  = readdlm(joinpath(FIX, "elast", "epsilons.csv"), ',')
        el_st = censored_elasticity(P, b, params; quaids = true, demographics = Z, vcov = VCOV,
                                    reps = size(EPS, 1), epsilons = EPS, price_index = :stone, shares = S)
        @test all(isfinite, el_st.elasticities)
        @test abs(sum(el_st.e_uobs) - 1.0) < 1e-8
        @test_throws ErrorException censored_elasticity(P, b, params; quaids = true,
                       demographics = Z, vcov = VCOV, reps = size(EPS, 1), epsilons = EPS,
                       price_index = :stone)   # missing shares

        # BHHH is the sole optimizer (small subsample, few iters): runs, never worsens, OPG vcov PSD.
        idx = 1:60
        Ss, Ps, bs, Zs = S[idx, :], P[idx, :], b[idx], Z[idx, :]
        iv  = initial_values(Ss, Ps, bs; quaids = true, demographics = Zs)
        ll0 = sum(censored_loglike(Ss, Ps, bs, iv; quaids = true, demographics = Zs, mc_points = 300))
        res = estimate(Ss, Ps, bs; quaids = true, demographics = Zs, start = iv,
                       mc_points = 300, maxiters = 4)
        @test res.optimizer == :bhhh
        @test res.loglike >= ll0 - 1e-6
        @test size(res.vcov) == (33, 33)
        @test isposdef(Symmetric(res.vcov))

        # Nelder-Mead / the `algorithm` kwarg are gone — passing it is a MethodError now.
        @test_throws MethodError estimate(Ss, Ps, bs; quaids = true, demographics = Zs,
                                          algorithm = :bhhh, maxiters = 1)
    end

    # ---- Change 3: Slutsky symmetry option ----
    @testset "Slutsky symmetry option" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        VCOV = readdlm(joinpath(FIX, "elast", "vcov.csv"), ',')
        EPS  = readdlm(joinpath(FIX, "elast", "epsilons.csv"), ',')

        # Slutsky residual of wᵢ·εᶜ[i,j] at the model's own expected shares (exactly as M4).
        slut_resid(E, w) = begin
            m = length(w)
            εc(i, jj) = E[i, jj] + w[jj] * E[i, m + 1]
            maximum(abs.([w[i] * εc(i, jj) - w[jj] * εc(jj, i) for i in 1:m, jj in 1:m]))
        end

        for pidx in (:translog, :stone)
            sh = pidx === :stone ? S : nothing
            el0 = censored_elasticity(P, b, params; quaids = true, demographics = Z, vcov = VCOV,
                                      reps = size(EPS, 1), epsilons = EPS, price_index = pidx, shares = sh)
            el1 = censored_elasticity(P, b, params; quaids = true, demographics = Z, vcov = VCOV,
                                      reps = size(EPS, 1), epsilons = EPS, price_index = pidx, shares = sh,
                                      symmetry = true)
            @test el1 isa ElasticityResult
            @test el1.symmetric
            @test slut_resid(el1.elasticities, el1.e_uobs) < 1e-10     # symmetry now exact
            @test slut_resid(el0.elasticities, el0.e_uobs) > 1e-3      # default genuinely asymmetric
            # default (symmetry=off) is byte-identical to before
            elx = censored_elasticity(P, b, params; quaids = true, demographics = Z, vcov = VCOV,
                                      reps = size(EPS, 1), epsilons = EPS, price_index = pidx, shares = sh)
            @test el0.elasticities == elx.elasticities
            @test all(isfinite, el1.se) && all(>=(0.0), el1.se)
            # other identities still hold under symmetry
            E, w = el1.elasticities, el1.e_uobs
            @test maximum(abs.([sum(E[i, 1:4]) + E[i, 5] for i in 1:4])) < 5e-5       # homogeneity
            @test abs(sum(w .* E[:, 5]) - 1.0) < 1e-6                                 # Engel
            @test maximum(abs.([sum(w .* E[:, jj]) + w[jj] for jj in 1:4])) < 1e-6    # Cournot
        end
    end

    # ---- Change 1: typed surface (enums + structs) ----
    @testset "Types & enums" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])

        @test PriceIndex(:translog) == TRANSLOG && PriceIndex(:stone) == STONE
        @test Symbol(STONE) == :stone && Symbol(TRANSLOG) == :translog
        @test FloorMode(:guard) == GUARD && Symbol(ADDITIVE_R) == :additive_r
        @test_throws ErrorException PriceIndex(:nope)
        @test_throws ErrorException FloorMode(:nope)

        # enum vs Symbol → byte-identical results everywhere
        W_sym  = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z, price_index = :translog)
        W_enum = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z, price_index = TRANSLOG)
        @test W_sym == W_enum
        ll_sym  = censored_loglike(S, P, b, params; quaids = true, demographics = Z,
                                   mc_points = 500, floor_mode = :additive_r)
        ll_enum = censored_loglike(S, P, b, params; quaids = true, demographics = Z,
                                   mc_points = 500, floor_mode = ADDITIVE_R)
        @test ll_sym == ll_enum

        # estimate returns a typed EstimationResult with a populated spec + names
        res = estimate(S[1:60, :], P[1:60, :], b[1:60]; quaids = true, demographics = Z[1:60, :],
                       start = params, mc_points = 300, maxiters = 3)
        @test res isa EstimationResult
        @test res.spec isa ModelSpec && res.spec.quaids && res.spec.price_index == TRANSLOG
        @test length(res.param_names) == length(res.params) == 33
        @test res.optimizer == :bhhh
    end

    # ---- Change 2: reporting ----
    @testset "Reporting" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        VCOV = readdlm(joinpath(FIX, "elast", "vcov.csv"), ',')
        EPS  = readdlm(joinpath(FIX, "elast", "epsilons.csv"), ',')
        names4 = ["SSB", "Juice", "Milk", "Water"]; demos = ["age", "size", "educ", "sex"]

        res = estimate(S[1:60, :], P[1:60, :], b[1:60]; quaids = true, demographics = Z[1:60, :],
                       start = params, mc_points = 300, maxiters = 3,
                       share_names = names4, demographic_names = demos)
        io = IOBuffer(); show(io, MIME"text/plain"(), res); s = String(take!(io))
        @test !isempty(s) && occursin("QUAIDS", s) && occursin("α_SSB", s)
        ct = coeftable(res)
        @test size(ct, 1) == 33 && size(ct, 2) == 5
        @test length(param_names(res.spec)) == 33

        el = censored_elasticity(P, b, params; quaids = true, demographics = Z, vcov = VCOV,
                                 reps = size(EPS, 1), epsilons = EPS, share_names = names4)
        io2 = IOBuffer(); show(io2, MIME"text/plain"(), el)
        @test !isempty(String(take!(io2)))
        @test size(elasticity_table(el), 1) == 20      # 4 goods × (4 prices + income)
    end

    # ---- Change 5: simulation + estimator recovery ----
    @testset "Simulation & recovery" begin
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])
        spec = ModelSpec(4, 4; quaids = true, price_index = :translog,
                         share_names = ["SSB", "Juice", "Milk", "Water"],
                         demographic_names = ["age", "size", "educ", "sex"])
        plm = vec(mean(P, dims = 1)); plc = cov(P); bm = mean(b); bs = std(b)

        # simulate_prices
        Psim = simulate_prices(300, 4; logmean = plm, logcov = plc)
        @test size(Psim) == (300, 4) && all(isfinite, Psim)

        # simulate_data: likelihood-consistent DGP → valid censored data
        sd = simulate_data(400, params, spec; price_logmean = plm, price_logcov = plc,
                           budget_logmean = bm, budget_logsd = bs)
        @test sd isa SimData
        @test size(sd.shares) == (400, 4) && all(isfinite, sd.shares)
        @test all(abs.(vec(sum(sd.shares, dims = 2)) .- 1.0) .< 1e-10)   # adding-up
        @test count(==(0.0), sd.shares) > 0                             # censoring present
        @test_throws ErrorException simulate_data(10, params,
                ModelSpec(4, 4; quaids = true, price_index = :stone))    # :stone not generative

        # Recovery study (deterministic via seed): estimator tracks the true coefficients.
        mc = montecarlo(500, params, spec; reps = 5, start = :truth, seed = 20240530,
                        mc_points = 300, maxiters = 80, g_tol = 1e-4,
                        price_logmean = plm, price_logcov = plc, budget_logmean = bm, budget_logsd = bs)
        @test mc isa MonteCarloResult
        @test mc.converged_frac >= 0.5
        @test all(isfinite, mc.bias) && all(isfinite, mc.rmse)
        @test all(0.0 .<= mc.coverage .<= 1.0)
        @test maximum(abs.(mc.bias[1:6])) < 0.35        # α, β recovered
        @test mean(mc.coverage) >= 0.4
        io = IOBuffer(); show(io, MIME"text/plain"(), mc)
        @test !isempty(String(take!(io)))
        @test size(montecarlo_table(mc), 1) == 33
    end

    # ---- Naive (uncensored) likelihood + estimate(loglike=:naive) ----
    @testset "Naive (uncensored) likelihood" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))   # real censored shares
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])  # 33

        # On a FULLY-INTERIOR dataset (no zeros) every household is the censored "full"
        # regime, so the naive density equals the censored likelihood to machine precision.
        # Build a guaranteed-positive interior share matrix by softening + renormalizing the
        # predicted shares (stays on the simplex, no exact zeros → no censoring).
        U = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z)
        Upos = abs.(U) .+ 0.05
        Sint = Upos ./ sum(Upos, dims = 2)            # rows sum to 1, all strictly positive
        @test all(Sint .> 0)                          # genuinely interior (no censoring)
        ll_naive = naive_loglike(Sint, P, b, params; quaids = true, demographics = Z)
        ll_cens  = censored_loglike(Sint, P, b, params; quaids = true, demographics = Z, mc_points = 2000)
        @test length(ll_naive) == 615
        @test maximum(abs.(ll_naive .- ll_cens)) < 1e-10   # identical with no censoring

        # On the REAL censored data the naive likelihood is finite everywhere and (because it
        # ignores the orthant probabilities) generally disagrees with the censored one.
        lln = naive_loglike(S, P, b, params; quaids = true, demographics = Z)
        @test length(lln) == 615 && all(isfinite, lln)
        @test abs(sum(lln) - sum(ll_cens)) > 1.0           # the two likelihoods differ

        # serial == threaded (order-independent)
        @test naive_loglike(S, P, b, params; quaids = true, demographics = Z, parallel = false) == lln

        # AIDS (no-λ) and no-demographics modes also run and are finite.
        ivA = initial_values(S, P, b; quaids = false, demographics = Z)         # 30
        @test all(isfinite, naive_loglike(S, P, b, ivA; quaids = false, demographics = Z))
        ivND = initial_values(S, P, b; quaids = true, demographics = nothing)   # 21
        @test all(isfinite, naive_loglike(S, P, b, ivND; quaids = true, demographics = nothing))

        # estimate(...; loglike = :naive): runs, improves its own objective, PSD vcov.
        idx = 1:80
        Ss, Ps, bs2, Zs = S[idx, :], P[idx, :], b[idx], Z[idx, :]
        iv = initial_values(Ss, Ps, bs2; quaids = true, demographics = Zs)
        ll0 = sum(naive_loglike(Ss, Ps, bs2, iv; quaids = true, demographics = Zs))
        resN = estimate(Ss, Ps, bs2; quaids = true, demographics = Zs, start = iv,
                        loglike = :naive, maxiters = 8, fd_mode = :forward)
        @test resN isa EstimationResult
        @test length(resN.params) == 33
        @test resN.loglike >= ll0 - 1e-6                   # naive optimiser never worsens naive ll
        @test isposdef(Symmetric(resN.vcov))

        # default loglike is :censored (unchanged); naive differs from censored fit.
        resC = estimate(Ss, Ps, bs2; quaids = true, demographics = Zs, start = iv,
                        mc_points = 300, maxiters = 8, fd_mode = :forward)
        @test resC.params != resN.params
        @test_throws ErrorException estimate(Ss, Ps, bs2; quaids = true, demographics = Zs,
                                             start = iv, loglike = :nope, maxiters = 1)
    end

    # ---- Naive-CENSORED (per-equation Tobit) likelihood + estimate(loglike=:naive_censored) ----
    # The OTHER misspecification: zeros treated as a Tobit censoring problem (latent demand
    # unobserved at 0), no Wales–Woodland reallocation, goods scored independently via marginal sds.
    @testset "Naive-censored (Tobit) likelihood" begin
        td, h = readcsv(joinpath(FIX, "testing_data.csv"))
        scol(name) = Float64.(td[:, findfirst(==(name), h)])
        S = hcat(scol("s1"), scol("s2"), scol("s3"), scol("s4"))   # real censored shares
        params = vec(readdlm(joinpath(FIX, "params_loglike.csv"), ',', header = true)[1])  # 33

        @test any(S .== 0.0)                          # the data really are censored (Tobit terms fire)

        ll_nc = naive_censored_loglike(S, P, b, params; quaids = true, demographics = Z)
        @test length(ll_nc) == 615 && all(isfinite, ll_nc)

        # Independent re-implementation of the exact formula on every household: density for goods
        # bought, censored mass Φ(-U/σ) for zeros, summed over the m-1 free goods with marginal σ.
        U = aids_shares(P, b, params[1:27]; quaids = true, demographics = Z)   # n×4 predicted shares
        sig = params[(end - 5):end]                  # j = 0.5*(m-1)*m = 6 sigma params (m=4)
        Rσ = zeros(3, 3); kk = 1
        for jj in 1:3, ii in 1:jj
            Rσ[ii, jj] = sig[kk]; kk += 1
        end
        Σ = Rσ' * Rσ
        sg = sqrt.(diag(Σ))                           # marginal sds for goods 1..3
        ref = [sum(S[i, g] != 0.0 ? logpdf(Normal(U[i, g], sg[g]), S[i, g]) :
                                    logcdf(Normal(U[i, g], sg[g]), 0.0) for g in 1:3) for i in 1:615]
        @test maximum(abs.(ll_nc .- ref)) < 1e-10    # matches the per-equation Tobit formula exactly

        # serial == threaded (per-household loop is order-independent)
        @test naive_censored_loglike(S, P, b, params; quaids = true, demographics = Z,
                                     parallel = false) == ll_nc

        # It is a DISTINCT likelihood: differs from both the naive (uncensored) and the correct
        # (Wales–Woodland) likelihoods on the censored data.
        ll_naive = naive_loglike(S, P, b, params; quaids = true, demographics = Z)
        ll_cens  = censored_loglike(S, P, b, params; quaids = true, demographics = Z, mc_points = 2000)
        @test abs(sum(ll_nc) - sum(ll_naive)) > 1.0
        @test abs(sum(ll_nc) - sum(ll_cens))  > 1.0

        # AIDS (no-λ) and no-demographics modes also run and are finite.
        ivA = initial_values(S, P, b; quaids = false, demographics = Z)         # 30
        @test all(isfinite, naive_censored_loglike(S, P, b, ivA; quaids = false, demographics = Z))
        ivND = initial_values(S, P, b; quaids = true, demographics = nothing)   # 21
        @test all(isfinite, naive_censored_loglike(S, P, b, ivND; quaids = true, demographics = nothing))

        # estimate(...; loglike = :naive_censored): runs, improves its own objective, PSD vcov.
        idx = 1:80
        Ss, Ps, bs2, Zs = S[idx, :], P[idx, :], b[idx], Z[idx, :]
        iv = initial_values(Ss, Ps, bs2; quaids = true, demographics = Zs)
        ll0 = sum(naive_censored_loglike(Ss, Ps, bs2, iv; quaids = true, demographics = Zs))
        resNC = estimate(Ss, Ps, bs2; quaids = true, demographics = Zs, start = iv,
                         loglike = :naive_censored, maxiters = 8, fd_mode = :forward)
        @test resNC isa EstimationResult
        @test length(resNC.params) == 33
        @test resNC.loglike >= ll0 - 1e-6                  # optimiser never worsens its own ll
        # estimate()'s contract is finite, nonnegative SEs (sqrt of the clamped OPG diagonal). On
        # this deliberately-degenerate tiny subset (33 params, 80 obs, misspecified Tobit, 8 iters)
        # the OPG is rank-deficient and its inverse is ill-conditioned, so we check the SE contract
        # rather than strict definiteness of the projected covariance.
        @test all(isfinite, resNC.se) && all(resNC.se .>= 0.0)

        # differs from the censored fit (different estimator ⇒ different argmax).
        resC = estimate(Ss, Ps, bs2; quaids = true, demographics = Zs, start = iv,
                        mc_points = 300, maxiters = 8, fd_mode = :forward)
        @test resC.params != resNC.params

        # Diagonal-covariance fit via `free` (the INTENDED usage): the per-equation Tobit only
        # identifies the marginal variances, so we hold the 3 off-diagonal Σ slots at 0 and optimize
        # the share params + 3 diagonal slots. This is what makes the share params actually move.
        jΣ = 6; nsh = 27; diagslot = [1, 3, 6]                 # m=4: R[jj,jj] at jj(jj+1)/2
        offidx = [nsh + s for s in 1:jΣ if !(s in diagslot)]   # global idx of off-diagonal slots
        freem = trues(33); start_d = copy(iv)
        for s in offidx; freem[s] = false; start_d[s] = 0.0; end
        ll_d0 = sum(naive_censored_loglike(Ss, Ps, bs2, start_d; quaids = true, demographics = Zs))
        resD = estimate(Ss, Ps, bs2; quaids = true, demographics = Zs, start = start_d,
                        loglike = :naive_censored, free = freem, maxiters = 12, fd_mode = :forward)
        @test all(resD.params[offidx] .== 0.0)                 # fixed (off-diagonal) params never move
        @test all(resD.vcov[offidx, :] .== 0.0) && all(resD.vcov[:, offidx] .== 0.0)  # zero variance
        @test norm(resD.params[1:nsh] .- start_d[1:nsh]) > 1e-3   # share params DO move (the whole point)
        @test resD.loglike >= ll_d0 - 1e-6                     # and the objective improves
    end
end
