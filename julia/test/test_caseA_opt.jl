# Tests of the optimized engine of case (A) (src/caseA_opt.jl): transforms (round trip, products against the exact
# convolution, automorphisms, public polynomials), comb bootstrap and error bootstrap (both transforms of B = 1) over the
# whole window, whole refresh.
# Run:  julia --project=. test/test_caseA_opt.jl
using Test, Random, FFTW
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
include("../src/caseA_fast.jl"); using .CaseAFast
import .CombBoot: Gadget, LWE, lwe_phase, torus_frac, torus_to_float, lwe_encrypt
const CA = CaseAFast

sdiff(x::Vector{UInt64}, y::Vector{UInt64}) = maximum(abs.(reinterpret.(Int64, x .- y)))

@testset "case (A), optimized engine: transforms (B ≥ 2)" begin
    rng = MersenneTwister(11)
    for (p, M0) in ((3, 4), (5, 4), (13, 8), (17, 4), (257, 16), (13, 16), (7, 32))
        Rg = ARing(p, M0); P = AOPlan(Rg); B = Rg.B
        for trial in 1:3
            x = rand_aug(Rg, rng); r = normR!(Rg, rand(rng, -2^10:2^10, B * p))
            @test sdiff(ao_inv(ao_fwd(x, P), P), x) < 2^16
            @test sdiff(ao_inv(ao_fwd(r, P) .* ao_fwd(x, P), P), conv_generic(Rg, r, x)) < 2^30
            for c in Ugroup_A(Rg)
                @test sdiff(ao_inv(ao_tau(ao_fwd(x, P), c, P), P), aut_A(Rg, x, c)) < 2^16
            end
        end
        # public polynomials: Σ_c B·A_c τ_c(y_c) against the monomials of BA_poly_A
        acol = rand(rng, 0:p-1, B); T = zeros(ComplexF64, p, B)
        CA.ao_table!(T, acol, P, plan_bfft!(copy(T), 2))
        ref = zeros(UInt64, B * p); acc = zeros(ComplexF64, p, P.B2)
        for c in Ugroup_A(Rg)
            y = rand_aug(Rg, rng)
            for (s, ku, kv) in BA_poly_A(Rg, acol, c)
                ref .+= mulUV(Rg, aut_A(Rg, y, c), s, ku, kv)
            end
            tix = CA._otix(c, B); Fy = ao_tau(ao_fwd(y, P), c, P)
            for j in 1:P.B2
                acc[:, j] .+= T[:, tix[j]] .* Fy[:, j]
            end
        end
        @test sdiff(ao_inv(acc, P), ref) < 2^20
    end
end

@testset "case (A), optimized engine: transforms (B = 1)" begin
    rng = MersenneTwister(12)
    for p in (13, 17, 31, 2017, 2039), mode in (:rader, :pad)
        Rg = ARing(p, 1); P1 = CA.AOPlan1(Rg, 2; mode)
        x = rand_aug(Rg, rng); y = rand_aug(Rg, rng)
        r = normR!(Rg, rand(rng, -2^10:2^10, p)); r2 = normR!(Rg, rand(rng, -2^10:2^10, p))
        a, b = CA.eb_inv2(CA.eb_fwd2(x, y, P1), P1)
        @test sdiff(a, x) < 2^16 && sdiff(b, y) < 2^16
        Dr = CA.eb_fwd2(r, r2, P1); Dx = CA.eb_fwd2(x, y, P1)
        pa, pb = CA.eb_inv2(hcat(Dr[:, 1] .* Dx[:, 1], Dr[:, 2] .* Dx[:, 2]), P1)
        @test sdiff(pa, conv_generic(Rg, r, x)) < 2^31 && sdiff(pb, conv_generic(Rg, r2, y)) < 2^31
    end
end

@testset "case (A), optimized engine: comb bootstrap and error bootstrap" begin
    rng = MersenneTwister(13)
    for (p, M0, n, m) in ((13, 8, 16, 1), (17, 4, 20, 1), (3, 4, 4, 1), (13, 16, 16, 1), (13, 8, 16, 2), (17, 4, 18, 3))
        Rg = ARing(p, M0)
        Pa = AParams(Rg; phat = 4, n, Gbr = Gadget(8, 3), Gks = Gadget(10, 3), σ = 2.0^-40)
        P = AOPlan(Rg); K = aokeygen(Rg, Pa, P, rng; m); ws = AOWork(Rg, Pa, P; m)
        R = Pa.phat * p; emax = (p - 1) ÷ 2
        for trial in 1:3
            aints = rand(rng, 0:p-1, Rg.B, n)
            E = rand(rng, -emax:emax, Rg.B); E[1] = isodd(trial) ? emax : -emax
            bints = [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + E[k], p) for k in 1:Rg.B]
            o = combboot_A_opt(Rg, Pa, K, aints, bints; ws)
            for k in 1:Rg.B
                @test abs(torus_to_float(lwe_phase(o[k], K.sk.svec) - torus_frac(E[k], R))) < 2.0^-14
            end
        end
        mus = rand(rng, 0:Pa.phat-1, Rg.B)
        cts = [lwe_encrypt(K.s, torus_frac(μ, Pa.phat), 2.0^-20, rng) for μ in mus]
        outs = refresh_A_opt(Rg, Pa, K, cts; ws)
        for k in 1:Rg.B
            @test abs(torus_to_float(lwe_phase(outs[k], K.sk.svec) - torus_frac(mus[k], Pa.phat))) < 2.0^-14
        end
    end
    for (p, n, m) in ((17, 12, 1), (31, 24, 1), (17, 12, 2), (31, 24, 4), (13, 12, 3)), mode in (:rader, :pad)
        Rg = ARing(p, 1); Pa = AParams(Rg; phat = 4, n, Gbr = Gadget(8, 3), σ = 2.0^-40)
        ws = AOWork1(Rg, Pa; mode); K = aokeygen1(Rg, Pa, ws.P1, rng; m)
        R = Pa.phat * p; emax = (p - 1) ÷ 2
        for trial in 1:4
            arow = rand(rng, 0:p-1, n); E = isodd(trial) ? emax : rand(rng, -emax:emax)
            bint = mod(sum(arow[u] * K.s[u] for u in 1:n) + E, p)
            o = blind_rotation_A_opt(Rg, Pa, K, arow, bint; ws)
            @test abs(torus_to_float(lwe_phase(o, K.sk.svec) - torus_frac(E, R))) < 2.0^-14
        end
        μ = rand(rng, 0:Pa.phat-1)
        out = refresh_A_opt(Rg, Pa, K, [lwe_encrypt(K.s, torus_frac(μ, Pa.phat), 2.0^-20, rng)]; ws)[1]
        @test abs(torus_to_float(lwe_phase(out, K.sk.svec) - torus_frac(μ, Pa.phat))) < 2.0^-14
    end
end
