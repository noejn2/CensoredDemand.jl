#!/usr/bin/env julia
# verify_roundtrip.jl  DEC_CSV  HEX_CSV
#
# Both CSVs have an identical header row and identical shape. DEC_CSV holds the
# decimal (%.17g) strings actually written into a fixture; HEX_CSV holds the
# exact C99 hex-float (%a) representation of the same originating doubles.
# We parse both with Julia's correctly-rounded parse(Float64, .) and assert that
# every cell matches bit-for-bit (=== handles NaN/sign). Prints "OK" on success.

function readnum(path)
    rows = Vector{Vector{Float64}}()
    for (i, ln) in enumerate(eachline(path))
        i == 1 && continue                      # skip header
        isempty(strip(ln)) && continue
        push!(rows, [parse(Float64, strip(t)) for t in split(strip(ln), ",")])
    end
    return rows
end

function main()
    dec_path, hex_path = ARGS[1], ARGS[2]
    a = readnum(dec_path)
    b = readnum(hex_path)
    if length(a) != length(b)
        error("row count mismatch: $(length(a)) vs $(length(b))")
    end
    bad = 0
    firstbad = ""
    for r in eachindex(a)
        length(a[r]) == length(b[r]) || error("col count mismatch at row $r")
        for c in eachindex(a[r])
            if !(a[r][c] === b[r][c])
                bad += 1
                if isempty(firstbad)
                    firstbad = "row $r col $c: dec=$(a[r][c]) hex=$(b[r][c])"
                end
            end
        end
    end
    if bad == 0
        print("OK")
    else
        error("Julia round-trip FAILED: $bad cells differ. First: $firstbad")
    end
end

main()
