# Unit tests on small parameters. Run from the julia folder of the repository:  julia --project test/runtests.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
using Test, Random

const rng = Xoshiro(2025)

@testset "CombBoot (case B)" begin
    N = 64
    r = Ring(N)
    P = r.plan.m.P

    @testset "NTT and exact products" begin
        x = rand(rng, UInt64(0):P-1, N)
        y = ntt!(similar(x), x, r.plan)
        @test intt!(similar(x), y, r.plan) == x
        s = Int64.(rand(rng, -3:3, N))
        t = rand(rng, UInt64, N)
        @test mul_small_torus(r, ntt_small(r, s), t) == mul_schoolbook(s, t)
        d = Int64.(rand(rng, -100:100, N))
        for a in (3, 5, 2N - 1, 37)
            @test ntt_small(r, aut_poly(d, a)) == ntt_small(r, d)[ntt_aut_perm(N, a)]
        end
        @test mulX(mulX(t, 5), 2N - 5) == t
        @test mulX(t, N) == .-t
    end

    @testset "gadget decomposition" begin
        G = Gadget(8, 3)
        t = rand(rng, UInt64, N)
        ds = decompose(G, t)
        rec = sum(reinterpret.(UInt64, ds[j]) .* gvalue(G, j) for j in 1:G.ell)
        err = maximum(abs.(torus_to_float.(t .- rec)))
        @test err <= 2.0^(-G.ell * G.logB - 1) + 1e-30
        @test all(all(abs.(dj) .<= 2^(G.logB - 1)) for dj in ds)
    end

    σ = 2.0^-40
    sk = SecretKey(r, Int64.(rand(rng, 0:1, N)))
    small(v) = maximum(abs.(torus_to_float.(v)))

    @testset "GLWE, external product, key switching" begin
        m = rand(rng, UInt64, N)
        c = glwe_encrypt(r, sk, m, σ, rng)
        @test small(glwe_phase(r, sk, c) .- m) < 1e-9
        G = Gadget(8, 3)
        K1 = ggsw_encrypt(r, sk, 1, G, σ, rng)
        K0 = ggsw_encrypt(r, sk, 0, G, σ, rng)
        @test small(glwe_phase(r, sk, extprod(r, c, K1)) .- m) < 1e-5
        @test small(glwe_phase(r, sk, extprod(r, c, K0))) < 1e-5
        c2 = glwe_encrypt(r, sk, mulX(m, 7), σ, rng)
        @test small(glwe_phase(r, sk, cmux(r, c, c2, K1)) .- mulX(m, 7)) < 1e-5
        @test small(glwe_phase(r, sk, cmux(r, c, c2, K0)) .- m) < 1e-5
        Gk = Gadget(6, 4)
        keys = [ks_keygen(r, sk, a, Gk, σ, rng) for a in (17, 33, 2N - 1)]
        outs = ks_hoisted(r, c, keys)
        for (K, o) in zip(keys, outs)
            @test small(glwe_phase(r, sk, o) .- aut_poly(m, K.a)) < 1e-5
        end
        l = extract(c, 5)
        @test small([lwe_phase(l, sk.s) - m[6]]) < 1e-9
    end

    Pm = CBParams(N = N, p = 16, phat = 2, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(6, 4), σ = σ)

    @testset "clear-text accumulator (Prop. 10, Thm. 9)" begin
        @test Pm.B == 8 && Pm.M0 == 8
        sprime = Int64.(rand(rng, 0:1, N))
        s = sprime[1:Pm.n]
        aints, bints, E = random_inputs(Pm, s, rng)
        acc = acc0(Pm, bints)
        for u in 1:Pm.n
            acol = aints[:, u]
            # Proposition 3: B·L(P) = Σ_a (B A_a) τ_a(P)
            lhs = UInt64(Pm.B) .* L_clear(Pm, acc, acol)
            rhs = zeros(UInt64, N)
            for a in Ugroup(Pm)
                Q = BA_poly(Pm, acol, a)
                τacc = aut_poly(acc, a)
                for e in 0:N-1
                    Q[e+1] == 0 || CombBoot.addmulX!(rhs, τacc, reinterpret(UInt64, Q[e+1]), e)
                end
            end
            @test lhs == rhs
            s[u] == 1 && (acc = L_clear(Pm, acc, acol))
        end
        R = R_of(Pm)
        @test all(acc[k+1] == torus_frac(E[k+1], R) for k in 0:Pm.B-1)
    end

    @testset "comb bootstrap vs separate blind rotations" begin
        K = cb_keygen(Pm, r, rng)
        aints, bints, E = random_inputs(Pm, K.s, rng)
        # clear accumulator, to follow the decryption at each iteration
        acc = Ref(acc0(Pm, bints))
        drift = Ref(0.0)
        trace = (u, C) -> begin
            K.s[u] == 1 && (acc[] = L_clear(Pm, acc[], aints[:, u]))
            drift[] = max(drift[], small(glwe_phase(r, K.sk, C) .- acc[]))
        end
        # NTT values of the public polynomials from the table (c_3)
        acol = aints[:, 1]
        T = ba_table(Pm, r, acol)
        for a in Ugroup(Pm), e in (1, 3, 2N - 1, 1 + Pm.p)
            @test ba_hat(Pm, T, a, e) == ntt_small(r, aut_poly(BA_poly(Pm, acol, a), e))
        end
        @test ntt_products_ok(Pm, r)
        outs = combboot(Pm, r, K, aints, bints; trace = trace)
        @test drift[] < 1e-4
        errs = output_errors(Pm, outs, K.sk.s, E)
        @test maximum(abs.(errs)) < 1e-4
        outs_c = combboot(Pm, r, K, aints, bints; ntt_products = false)
        @test output_errors(Pm, outs_c, K.sk.s, E) ≈ errs atol = 1e-6
        outs2 = separate_rotations(Pm, r, K, aints, bints)
        errs2 = output_errors(Pm, outs2, K.sk.s, E)
        @test maximum(abs.(errs2)) < 1e-4
    end

    @testset "refresh (error bootstrap of B ciphertexts)" begin
        P2 = CBParams(N = N, p = 32, phat = 4, n = 12, Gbr = Gadget(8, 3), Gks = Gadget(6, 4), σ = σ)
        K = cb_keygen(P2, r, rng)
        msgs = rand(rng, 0:P2.phat-1, P2.B)
        cts = [lwe_encrypt(K.s, torus_frac(m, P2.phat), 2.0^-20, rng) for m in msgs]
        fresh = refresh(P2, r, K, cts)
        @test length(fresh[1].a) == N
        for k in 1:P2.B
            err = torus_to_float(lwe_phase(fresh[k], K.sk.s) - torus_frac(msgs[k], P2.phat))
            @test abs(err) < 1e-4
        end
    end
end
