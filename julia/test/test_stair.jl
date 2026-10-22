# Tests of the polynomial stair key switch (src/stair_ks.jl). Run from the julia folder of the repository: julia test/test_stair.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random, Statistics

const rng = Xoshiro(31)

@testset "stair key switch" begin
    T = 128
    sp = Int64.(rand(rng, 0:1, T))
    for chain in ([(96, 8, 4), (64, 6, 5), (60, 4, 8)], [(80, 8, 4), (48, 6, 5)], [(77, 10, 4)])
        K = stair_keygen(sp, T, chain, rng; σf = x -> 2.0^-40)
        n = chain[end][1]
        for trial in 1:20
            μ = rand(rng, UInt64)
            c = lwe_encrypt(sp, μ, 0.0, rng)
            o = stair_ks(K, c)
            @test length(o.a) == n
            @test abs(torus_to_float(lwe_phase(o, sp[1:n]) - μ)) < 1e-6
        end
    end
    # input of dimension < T (padding with zero coordinates)
    K = stair_keygen(sp, T, [(64, 8, 4)], rng; σf = x -> 2.0^-40)
    c = lwe_encrypt(sp[1:100], UInt64(12345) << 40, 0.0, rng)
    @test abs(torus_to_float(lwe_phase(stair_ks(K, c), sp[1:64]) - (UInt64(12345) << 40))) < 1e-6
    # noise against the prediction
    for chain in ([(96, 8, 2), (64, 6, 3), (60, 4, 4)], [(80, 3, 5)])
        σf = x -> 2.0^-22
        K = stair_keygen(sp, T, chain, rng; σf)
        n = chain[end][1]
        e = [begin
                 μ = rand(rng, UInt64); o = stair_ks(K, lwe_encrypt(sp, μ, 0.0, rng))
                 torus_to_float(lwe_phase(o, sp[1:n]) - μ)
             end for _ in 1:400]
        r = mean(abs2, e) / stair_var(T, chain; σf)
        @test 0.75 < r < 1.3
    end
end
