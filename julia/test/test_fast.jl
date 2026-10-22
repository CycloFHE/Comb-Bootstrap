# Tests of the FFT backend (needs FFTW.jl). Run from the julia folder of the repository:  julia --project test/test_fast.jl
# (julia --project -t 4 test/test_fast.jl also tests the multithreaded versions)
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random

const rng = Xoshiro(7)
dist(u, v) = maximum(abs.(torus_to_float.(u .- v)))

@testset "FFT backend" begin
    N = 64
    r = Ring(N); fp = FPlan(N)
    t = rand(rng, UInt64, N)
    @test dist(Fast.finv(Fast.ffwd(t, fp), fp), t) < 2.0^-50
    d = Int64.(rand(rng, -512:512, N))
    @test dist(Fast.finv(Fast.ffwd(d, fp) .* Fast.ffwd(t, fp), fp), mul_schoolbook(d, t)) < 2.0^-35
    for a in (3, 5, 17, 2N - 1)
        src, cj = Fast.faut_map(N, a)
        Y = Fast.ffwd(d, fp)
        Yτ = [cj[k] ? conj(Y[src[k]]) : Y[src[k]] for k in 1:N÷2]
        @test maximum(abs.(Yτ .- Fast.ffwd(aut_poly(d, a), fp))) < 1e-8
    end

    P = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(6, 4), σ = 2.0^-40)
    acol = rand(rng, 0:P.p-1, P.B)
    T = Fast.fba_table(P, fp, acol)
    for a in Ugroup(P), e in (1, 3, 2N - 1)
        @test maximum(abs.(Fast.fba_hat(P, T, a, e) .- Fast.ffwd(aut_poly(BA_poly(P, acol, a), e), fp))) < 1e-8
    end

    K = fast_keygen(P, r, fp, rng)
    c = glwe_encrypt(r, K.sk, t, P.σ, rng)
    @test dist(glwe_phase(r, K.sk, fextprod(fp, c, K.bsk[findfirst(==(1), K.s)])), t) < 1e-5
    for Kτ in vcat(K.baby, K.giant)
        @test dist(glwe_phase(r, K.sk, fks(fp, c, Kτ)), aut_poly(t, Kτ.a)) < 1e-5
    end

    aints, bints, E = random_inputs(P, K.s, rng)
    ref = Fast.combboot_fast_ref(P, fp, K, aints, bints)
    @test maximum(abs.(output_errors(P, ref, K.sk.s, E))) < 1e-4
    ref2 = Fast.separate_rotations_fast_ref(P, fp, K, aints, bints)
    @test maximum(abs.(output_errors(P, ref2, K.sk.s, E))) < 1e-4
    outs = combboot_fast(P, fp, K, aints, bints)
    @test maximum(abs.(output_errors(P, outs, K.sk.s, E))) < 1e-4
    outs2 = separate_rotations_fast(P, fp, K, aints, bints)
    @test maximum(abs.(output_errors(P, outs2, K.sk.s, E))) < 1e-4
    outs3 = separate_rotations_fast(P, fp, K, aints, bints; batched = false)
    @test maximum(abs.(output_errors(P, outs3, K.sk.s, E))) < 1e-4
    if Threads.nthreads() >= 2                     # julia --project -t 2 (or more): multithreaded versions, same results
        T = Threads.nthreads()
        ws = Fast.Work(P, fp, K)
        o1 = combboot_fast(P, fp, K, aints, bints; ws)
        oT = combboot_fast(P, fp, K, aints, bints; ws, nthreads = T)
        @test all(o1[k].a == oT[k].a && o1[k].b == oT[k].b for k in eachindex(o1))
        s1 = separate_rotations_fast(P, fp, K, aints, bints; ws)
        sT = separate_rotations_fast(P, fp, K, aints, bints; ws, nthreads = T)
        @test all(s1[k].a == sT[k].a && s1[k].b == sT[k].b for k in eachindex(s1))
    end
end
