# Polynomial stair key switch (Section 4.3 and Appendix B of the paper): noise and time, one thread, d = 16, keys at σ_128(x).
# The chains are those of the cost model: for each configuration of the comparison at equal p̂,
# the cheapest chain whose noise fits the budget left by the bootstrap. NS samples for the noise (default 200),
# REPS repetitions for the time (default 20). Run from the julia folder of the repository:  julia benchmarks/run_stair_ks.jl
# (CH=1,2: some chains only).
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf, Statistics

const CHAINS = [
    ("error bootstrap, p̂ = 256, refresh", 2048, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16)]),
    ("error bootstrap, p̂ = 256, before a product", 2048, [(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (830, 1, 17)]),
    ("error bootstrap, p̂ = 64", 2048, [(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (780, 2, 8)]),
    ("TFHE PBS, 4-bit blocks", 2048, [(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14)]),
    ("TFHE single ring, N = 2^15", 32768, [(2048, 16, 2), (1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18)]),
    ("TFHE single ring, N = 2^16", 65536, [(2048, 16, 2), (1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7), (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19)]),
]

const NS = parse(Int, get(ENV, "NS", "200"))
const REPS = parse(Int, get(ENV, "REPS", "20"))
const CH = parse.(Int, split(get(ENV, "CH", join(1:length(CHAINS), ",")), ","))

println("Polynomial stair key switch, d = 16, one thread (log2 σ: measured / predicted)")
for c in CH
    lab, T, chain = CHAINS[c]
    rng = Xoshiro(40 + c)
    sp = Int64.(rand(rng, 0:1, T))
    K = stair_keygen(sp, T, chain, rng)
    n = chain[end][1]
    e = Float64[]
    for _ in 1:NS
        μ = rand(rng, UInt64)
        o = stair_ks(K, lwe_encrypt(sp, μ, 0.0, rng))
        push!(e, torus_to_float(lwe_phase(o, sp[1:n]) - μ))
    end
    x = lwe_encrypt(sp, rand(rng, UInt64), 0.0, rng)
    stair_ks(K, x)
    t = minimum(@elapsed(stair_ks(K, x)) for _ in 1:REPS)
    @printf("  %-44s %6d -> %4d (%2d steps): log2 σ %7.2f / %7.2f   %7.3f ms   keys %6.1f MB\n", lab, T, n, length(chain),
            0.5log2(mean(abs2, e)), 0.5log2(stair_var(T, chain)), 1e3t, Base.summarysize(K) / 2.0^20)
    flush(stdout)
end
