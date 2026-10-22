# Tests of the grouping by m (src/grouped.jl): comb bootstrap (h = 1 and h = 2) and separate rotations (multi-bit form),
# m = 1..4. Run from the julia folder of the repository:  julia --project test/test_grouped.jl   (julia --project -t 4: also the multithreaded versions)
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random

const rng = Xoshiro(17)

@testset "grouping by m" begin
    N = 64; r = Ring(N); fp = FPlan(N)
    for h in (1, 2), m in 1:4
        P = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(6, 4), σ = 2.0^-40, h = h)
        K = fast_keygen_grouped(P, r, fp, rng; m)
        @test length(K.bsk) == (P.n ÷ m) * (2^m - 1)
        wg = WorkG(P, fp, K; m)
        for trial in 1:3
            aints, bints, E = random_inputs(P, K.s, rng)
            o = combboot_grouped(P, fp, K, aints, bints; m, wg)
            @test maximum(abs.(output_errors(P, o, K.sk.s, E))) < 1e-4
            s = separate_rotations_grouped(P, fp, K, aints, bints; m, wg)
            @test maximum(abs.(output_errors(P, s, K.sk.s, E))) < 1e-4
            if Threads.nthreads() >= 2              # julia --project -t 2 (or more): multithreaded versions, same results
                T = Threads.nthreads()
                oT = combboot_grouped(P, fp, K, aints, bints; m, wg, nthreads = T)
                @test all(o[k].a == oT[k].a && o[k].b == oT[k].b for k in eachindex(o))
                sT = separate_rotations_grouped(P, fp, K, aints, bints; m, wg, nthreads = T)
                @test all(s[k].a == sT[k].a && s[k].b == sT[k].b for k in eachindex(s))
            end
        end
    end
    # m = 1, h = 2: same as the optimized comb bootstrap of fast_opt.jl (same keys)
    P = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(6, 4), σ = 2.0^-40)
    K = fast_keygen_grouped(P, r, fp, rng; m = 1)
    aints, bints, E = random_inputs(P, K.s, rng)
    o1 = combboot_fast(P, fp, K, aints, bints)
    o2 = combboot_grouped(P, fp, K, aints, bints; m = 1)
    @test maximum(abs(Int64(o1[k].b - o2[k].b)) for k in eachindex(o1)) < 2^20
    @test maximum(abs.(output_errors(P, o2, K.sk.s, E))) < 1e-4
end
