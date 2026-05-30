using Test
using CensoredDemand
using DelimitedFiles
using LinearAlgebra

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

        @testset "M7 — run_job headless entrypoint (elasticities)" begin
            edir = joinpath(FIX, "elast")
            cfg = Dict{String,Any}(
                "data_csv"         => joinpath(FIX, "testing_data.csv"),
                "share_cols"       => ["s1", "s2", "s3", "s4"],
                "logprice_cols"    => ["lnp1", "lnp2", "lnp3", "lnp4"],
                "budget_col"       => "lnw",
                "demographic_cols" => ["age", "size", "educ", "sex"],
                "quaids"           => true,
                "tasks"            => ["elasticities"],
                "params"           => vec(readdlm(joinpath(FIX, "params_loglike.csv"),
                                                  ',', header = true)[1]),
                "vcov_csv"         => joinpath(edir, "vcov.csv"),
                "reps"             => 20000,
                "share_names"      => ["SSB", "Juice", "Milk", "Water"],
            )

            out = run_job(cfg)
            @test out["status"] == "ok"
            @test out["n_obs"] == 615
            @test out["n_goods"] == 4
            @test haskey(out, "elasticities")

            ela = out["elasticities"]
            # Matrices serialize as row-major arrays-of-arrays.
            @test length(ela["elasticities"]) == 4
            @test all(length(r) == 5 for r in ela["elasticities"])
            @test length(ela["se"]) == 4
            @test all(length(r) == 5 for r in ela["se"])
            @test length(ela["e_uobs"]) == 4
            @test ela["share_names"] == ["SSB", "Juice", "Milk", "Water"]

            # Amemiya–Tobin adding-up of expected observed shares.
            @test abs(sum(ela["e_uobs"]) - 1.0) < 1e-8
            @test out["meta"]["n_demographics"] == 4
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
end
