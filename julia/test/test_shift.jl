# Tests of the shifted gadget of the automorphism keys, Gadget(logB, ℓ, off) with off ≤ log2 B and h = 1: digits and
# representative, exact comb bootstrap (Algorithm 1), FFT comb bootstraps (fast_opt.jl, grouped.jl) against the reference,
# and the noise of the key switchings, divided by about 2^(2 off).
# Run from the julia folder of the repository:  julia --project test/test_shift.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random

const rng = Xoshiro(29)
meas(e) = sum(abs2, e) / length(e)
l2(v) = 0.5 * log2(v)

@testset "shifted gadget" begin
    @testset "digits and representative" begin
        for (b, ℓ, off) in ((6, 3, 4), (7, 5, 4), (11, 3, 3), (8, 4, 0))
            G = Gadget(b, ℓ, off)
            for t in (rand(rng, UInt64, 512) .>> off, rand(rng, UInt64, 512))     # top off bits zero, or any
                D = decompose(G, t); L = gadget_lift(G, t)
                @test all(all(abs.(d) .<= 1 << (b - 1)) for d in D)
                rec = sum(reinterpret.(UInt64, D[j]) .* gvalue(G, j) for j in 1:ℓ)
                @test all(abs.(reinterpret.(Int64, rec .- L)) .<= Int64(1) << (63 - off - b * ℓ))
                @test off == 0 ? L == t : all(((L .- t) .<< (64 - off)) .== 0)    # L ≡ t modulo 2^-off
            end
        end
        N = 64; fp = FPlan(N)
        for G in (Gadget(6, 4, 3), Gadget(5, 3, 0))                                 # FFT digits = exact digits
            t = rand(rng, UInt64, N) .>> G.off
            D = decompose(G, t)
            for j in 1:G.ell
                Dj = Vector{ComplexF64}(undef, fp.H)
                Fast.digit_fwd!(Dj, t, nothing, G, j, fp)
                @test maximum(abs.(Dj .- Fast.ffwd(D[j], fp))) < 1e-9
            end
        end
        @test_throws ErrorException CBParams(N = 64, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3),
                                             Gks = Gadget(6, 4, 3), σ = 2.0^-40, h = 2)
        @test_throws ErrorException CBParams(N = 64, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3),
                                             Gks = Gadget(6, 4, 4), σ = 2.0^-40, h = 1)    # off > log2 B = 3
    end

    @testset "exact comb bootstrap, noise of the key switchings" begin
        N = 64; r = Ring(N)
        lv = Dict{Int,Float64}()
        for off in (0, 3)
            P = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(4, 3, off), σ = 2.0^-40, h = 1)
            K = cb_keygen(P, r, rng)
            e = Float64[]
            for trial in 1:40
                aints, bints, E = random_inputs(P, K.s, rng)
                append!(e, output_errors(P, combboot(P, r, K, aints, bints), K.sk.s, E))
            end
            pred = var_combboot(P, sum(K.s))
            @test abs(l2(meas(e)) - l2(pred)) < 0.4
            lv[off] = l2(meas(e))
        end
        @test lv[0] - lv[3] > 2.0                    # the rounding term of the key switchings is divided by 2^6
    end

    @testset "FFT comb bootstraps" begin
        N = 64; r = Ring(N); fp = FPlan(N)
        P = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(5, 4, 3), σ = 2.0^-40, h = 1)
        K = fast_keygen(P, r, fp, rng); ws = Fast.Work(P, fp, K)
        for trial in 1:3
            aints, bints, E = random_inputs(P, K.s, rng)
            ref = Fast.combboot_fast_ref(P, fp, K, aints, bints)
            o = combboot_fast(P, fp, K, aints, bints; ws)
            @test maximum(abs.(output_errors(P, ref, K.sk.s, E))) < 1e-4
            @test maximum(abs.(output_errors(P, o, K.sk.s, E))) < 1e-4
            @test maximum(abs(reinterpret(Int64, o[k].b - ref[k].b)) for k in eachindex(o)) < 2^28
        end
        for m in 1:3
            Kg = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, Kg; m)
            aints, bints, E = random_inputs(P, Kg.s, rng)
            o = combboot_grouped(P, fp, Kg, aints, bints; m, wg)
            @test maximum(abs.(output_errors(P, o, Kg.sk.s, E))) < 1e-4
            if m == 1
                o1 = combboot_fast(P, fp, Kg, aints, bints)
                @test maximum(abs(reinterpret(Int64, o1[k].b - o[k].b)) for k in eachindex(o)) < 2^24
            end
        end
    end

    @testset "N = 2048, p = 256: noise of the FFT comb bootstrap" begin
        N = 2048; r = Ring(N); fp = FPlan(N)
        lv = Dict{Int,Float64}(); pv = Dict{Int,Float64}()
        for off in (0, 4)
            P = CBParams(N = N, p = 256, phat = 4, n = 16, Gbr = Gadget(14, 2), Gks = Gadget(6, 3, off),
                         σ = 2.0^-50.3, h = 1)
            K = fast_keygen(P, r, fp, rng); ws = Fast.Work(P, fp, K)
            e = Float64[]
            for trial in 1:8
                aints, bints, E = random_inputs(P, K.s, rng)
                append!(e, output_errors(P, combboot_fast(P, fp, K, aints, bints; ws), K.sk.s, E))
            end
            pv[off] = l2(var_combboot(P, sum(K.s))); lv[off] = l2(meas(e))
            @test abs(lv[off] - pv[off]) < 0.3
        end
        @test pv[0] - pv[4] > 3.5                      # rounding term divided by 2^8 (it dominates here) ...
        @test abs((lv[0] - lv[4]) - (pv[0] - pv[4])) < 0.4                       # ... as measured
    end
end
