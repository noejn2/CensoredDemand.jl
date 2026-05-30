using JSON3, DelimitedFiles

res = JSON3.read(read(joinpath(@__DIR__, "out", "results.json"), String))
el = res.elasticities.elasticities                        # array-of-arrays (4x5)
E = permutedims(hcat([collect(Float64.(r)) for r in el]...))   # 4x5 matrix
elasR = readdlm(joinpath(@__DIR__, "..", "test", "fixtures", "elast", "elasticities_R.csv"), ',')
maxerr = maximum(abs.(E .- elasR))
euobs = collect(Float64.(res.elasticities.e_uobs))
addup = abs(sum(euobs) - 1.0)

println("status=", res.status)
println("size(E)=", size(E), "  size(elasR)=", size(elasR))
println("sample_elasticity_max_err_vs_R=", maxerr)
println("euobs sum=", sum(euobs), "  |sum-1|=", addup)
println("euobs_adding_up(<1e-8)=", addup < 1e-8)
println("share_names=", res.elasticities.share_names)
