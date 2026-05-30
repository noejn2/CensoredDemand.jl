using Test
using CensoredDemand
using DelimitedFiles

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
end
