# Tests of the FFT backend of case (A) (src/caseA_fast.jl) against the exact backend (src/caseA.jl): products,
# automorphisms, public products, external product, key switching, error bootstrap and comb bootstrap on small rings
# and on the rings of the paper.
# Run:  julia --project=. test/test_caseA_fft.jl
using Test, Random
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
include("../src/caseA_fast.jl"); using .CaseAFast
import .CombBoot: Gadget, LWE, lwe_phase, torus_frac, torus_to_float, lwe_encrypt

sdiff(x::Vector{UInt64}, y::Vector{UInt64}) = maximum(abs.(reinterpret.(Int64, x .- y)))
"Modes of the transform: all for the products (:pad without chained products), the comb needs :rader or :fftw."
modes(B; chained = false) = B == 1 ? (:rader, :pad, :fftw) : chained ? (:rader, :fftw) : (:rader, :pad, :fftw)

@testset "case (A), FFT: products and automorphisms" begin
    rng = MersenneTwister(5)
    for (p, M0) in ((3, 1), (17, 1), (5, 4), (13, 8), (3, 4), (257, 16), (2039, 1), (2017, 1)), mode in modes(M0 == 1 ? 1 : M0 ÷ 2)
        Rg = ARing(p, M0); fp = AFPlan(Rg; mode); B = Rg.B
        for trial in 1:3
            x = rand_aug(Rg, rng)
            r = normR!(Rg, rand(rng, -2^10:2^10, B * p))
            ref = conv_generic(Rg, r, x)
            out = afinv(affwd(r, fp) .* affwd(x, fp), fp)
            @test sdiff(out, ref) < 2^30                         # FFT error, 2^-34 of the torus
            @test all(sum(out[k*p+1:(k+1)*p]) == 0 for k in 0:B-1)   # gauge
            # any lift of r: r + (constant) Σ_v V^v per column
            r2 = copy(r); for k in 0:B-1; r2[k*p+1:(k+1)*p] .+= rand(rng, -5:5); end
            mode == :pad || @test sdiff(afinv(affwd(r2, fp) .* affwd(x, fp), fp), ref) < 2^30   # :pad: Ω form only
            # round trip, and two transforms at once (one Rader transform for B = 1)
            @test sdiff(afinv(affwd(x, fp), fp), x) <= 2^16
            # (paired inputs of comparable sizes: the error of the FFT is relative to the larger one)
            y = rand_aug(Rg, rng); r3 = normR!(Rg, rand(rng, -2^10:2^10, B * p))
            Z1 = similar(affwd(x, fp)); Z2 = similar(Z1)
            affwd2!(Z1, Z2, r, r3, fp)
            @test Z1 ≈ affwd(r, fp) && Z2 ≈ affwd(r3, fp)
            o1 = similar(x); o2 = similar(x); afinv2!(o1, o2, Z1 .* affwd(x, fp), affwd(y, fp), fp)
            @test sdiff(o1, ref) < 2^30 && sdiff(o2, y) < 2^30            # o2: error of the larger of the pair
            # automorphisms c ≡ 1 (mod p): permutation of the columns of the Fourier values
            for c in Ugroup_A(Rg)
                pm = afperm(Rg, c)
                @test sdiff(afinv(affwd(x, fp)[:, pm], fp), aut_A(Rg, x, c)) <= 2^16
            end
        end
        if B > 1 && mode != :pad
            # public products: Σ_c B·A_c·Cs[c] against the monomials of BA_poly_A
            cb = AFComb(Rg, fp); h = fp.h
            acol = rand(rng, 0:p-1, B)
            xs = [rand_aug(Rg, rng) for _ in cb.U]
            T = aftable!(zeros(ComplexF64, h, B), cb, acol)
            GA = zeros(ComplexF64, h, B); GB = similar(GA)
            afpublic!(GA, GB, cb, T, [(affwd(x, fp), affwd(x, fp)) for x in xs])
            ref = zeros(UInt64, B * p)
            for (i, c) in enumerate(cb.U), (s, ku, kv) in BA_poly_A(Rg, acol, c)
                ref .+= mulUV(Rg, xs[i], s, ku, kv)
            end
            @test sdiff(afinv(GA, fp), ref) <= 2^20
        end
    end
end

@testset "case (A), FFT: external product, key switching" begin
    rng = MersenneTwister(6)
    for (p, M0) in ((17, 1), (13, 8), (17, 4), (257, 16)), mode in modes(M0 == 1 ? 1 : M0 ÷ 2)
        Rg = ARing(p, M0); fp = AFPlan(Rg; mode); B = Rg.B
        sk = akeygen_sk(Rg, rng); σ = 2.0^-45
        G = Gadget(10, 3)
        for μ in (0, 1)
            K = afggsw(Rg, fp, sk, μ, G, σ, rng)
            m = rand_aug(Rg, rng)
            c = aglwe_encrypt(Rg, sk, m, σ, rng)
            d = afextprod(Rg, fp, c, K)
            @test sdiff(aphase(Rg, sk, d), μ == 1 ? aphase(Rg, sk, c) : zeros(UInt64, B * p)) < 2^44
        end
        for c in Ugroup_A(Rg)[2:end]
            K = afks_keygen(Rg, fp, sk, c, G, σ, rng)
            C = aglwe_encrypt(Rg, sk, rand_aug(Rg, rng), σ, rng)
            @test sdiff(aphase(Rg, sk, afks(Rg, fp, C, K)), aut_A(Rg, aphase(Rg, sk, C), c)) < 2^44
        end
    end
end

@testset "case (A), FFT: error bootstrap and comb bootstrap" begin
    rng = MersenneTwister(7)
    for (p, M0, n, m) in ((17, 1, 12, 1), (13, 8, 16, 1), (17, 4, 20, 1), (3, 4, 4, 1), (31, 1, 24, 1),
                          (17, 1, 12, 2), (13, 8, 16, 2), (17, 4, 18, 3), (31, 1, 24, 4)),
        mode in modes(M0 == 1 ? 1 : M0 ÷ 2; chained = true)
        Rg = ARing(p, M0); fp = AFPlan(Rg; mode)
        P = AParams(Rg; phat = 4, n, Gbr = Gadget(8, 3), Gks = Gadget(10, 3), σ = 2.0^-40)
        K = afkeygen(Rg, P, fp, rng; m)
        R = P.phat * p; emax = (p - 1) ÷ 2
        for trial in 1:3
            aints = rand(rng, 0:p-1, Rg.B, n)
            E = rand(rng, -emax:emax, Rg.B); E[1] = isodd(trial) ? emax : -emax       # the edges of the window
            bints = [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + E[k], p) for k in 1:Rg.B]
            outs = Rg.B == 1 ? [blind_rotation_A_fft(Rg, P, fp, K, view(aints, 1, :), bints[1])] :
                               combboot_A_fft(Rg, P, fp, K, aints, bints)
            for k in 1:Rg.B
                @test abs(torus_to_float(lwe_phase(outs[k], K.sk.svec) - torus_frac(E[k], R))) < 2.0^-14
            end
        end
        mus = rand(rng, 0:P.phat-1, Rg.B)
        cts = [lwe_encrypt(K.s, torus_frac(μ, P.phat), 2.0^-20, rng) for μ in mus]
        outs = refresh_A_fft(Rg, P, fp, K, cts)
        for k in 1:Rg.B
            @test abs(torus_to_float(lwe_phase(outs[k], K.sk.svec) - torus_frac(mus[k], P.phat))) < 2.0^-14
        end
    end
end
