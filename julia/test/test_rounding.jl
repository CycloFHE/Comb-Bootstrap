# Tests of the rounding with recentering and selection (src/rounding.jl) and of the whole refresh: refresh (exact
# backend) and eb_refresh (FFT backend, src/fast_refresh.jl, key switch included), for the comb bootstrap and for the
# error bootstrap alone (B = 1, p = 2N).
#  - the rounding errors X_u, the public mean c and variance V of T = Σ s_u X_u (against exact arithmetic and a Monte
#    Carlo over the keys);
#  - selection: V ≤ V* after about 100 trials, the message unchanged; recentering: the rounding term E - R e has mean 0
#    and variance E[V] + 1/6, against about n/24 without both;
#  - refreshed ciphertexts decrypt to their messages, with the errors E_k inside the window p/4;
#  - N = 2048, configuration of 8 bits of Table 3 (p = 256, n = 896): end to end, inputs at the noise of the budget.
# Run from the julia folder of the repository:  julia --project test/test_rounding.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random, Statistics

const rng = Xoshiro(41)
Δ(i, phat) = UInt64(i) << (64 - trailing_zeros(phat))
err(c, s, μ) = torus_to_float(lwe_phase(c, s) - μ)

@testset "rounding with recentering and selection" begin
    @testset "rounding errors and their statistics" begin
        R = 1 << 12; n = 64
        a = rand(rng, UInt64, n)
        Rx = [big(x) * R / big(2)^64 for x in a]                       # R a_u, exactly
        X = [Float64(round(y, RoundNearestTiesUp) - y) for y in Rx]
        @test all(abs.(X) .<= 0.5)
        @test [CombBoot.round_to(x, R) for x in a] == [Int(mod(round(y, RoundNearestTiesUp), R)) for y in Rx]
        c, V = rounding_stats(a, R)
        @test c ≈ sum(X) / 2 atol = 1e-12
        @test V ≈ sum(abs2, X) / 4 atol = 1e-12
        T = [sum(X[u] for u in 1:n if s[u] == 1; init = 0.0) for s in (rand(rng, 0:1, n) for _ in 1:40000)]
        @test abs(mean(T) - c) < 4 * sqrt(V / 40000)
        @test abs(var(T) / V - 1) < 0.05
    end

    @testset "selection and recentering (n = 896, R = 2^16)" begin
        n = 896; R = 1 << 16; s = Int64.(rand(rng, 0:1, n))
        pool = zero_pool(s, 128, 0.0, rng)                              # noise-free: E is the rounding term alone
        Vs = vstar_threshold(n)
        trials = Int[]; V = Float64[]; Es = Int[]; Ep = Int[]
        for _ in 1:600
            μ = rand(rng, 0:255); ct = lwe_encrypt(s, Δ(μ, 256), 0.0, rng)
            r = round_select(ct, R, pool; Vstar = Vs, rng)
            @test lwe_phase(r.ct, s) == lwe_phase(ct, s)                 # the message is unchanged
            push!(trials, r.trials); push!(V, r.V)
            push!(Es, rounded_errors([r], s, [256μ], R)[1])
            r0 = round_select(ct, R, nothing; recenter = false)          # neither selection nor recentering
            push!(Ep, rounded_errors([r0], s, [256μ], R)[1])
        end
        @test all(V .<= Vs)
        @test 70 <= mean(trials) <= 140                                  # geometric law of mean 100
        @test abs(mean(Es)) < 0.1
        @test abs(var(Es) / (mean(V) + 1 / 6) - 1) < 0.15
        @test var(Ep) > 1.6 * var(Es)                                    # about n/24 + 1/6 against n/48 + 1/6
    end

    @testset "exact refresh, B = 8 and B = 1" begin
        N = 64; r = Ring(N)
        for (p, phat, Gks) in ((16, 4, Gadget(6, 4)), (2N, 8, Gadget(6, 4)))
            P = CBParams(N = N, p = p, phat = phat, n = 12, Gbr = Gadget(8, 3), Gks = Gks, σ = 2.0^-40)
            K = cb_keygen(P, r, rng); pool = zero_pool(K.s, 32, 2.0^-35, rng)
            for _ in 1:3
                is = rand(rng, 0:phat-1, P.B)
                cts = [lwe_encrypt(K.s, Δ(i, phat), 2.0^-30, rng) for i in is]
                Vs = vstar_threshold(P.n; z = -1.28)                    # 10% quantile: n = 12 is small
                out, rs = refresh(P, r, K, cts; pool, Vstar = Vs, rng, info = true)
                @test all(x.V <= Vs for x in rs)
                @test all(abs.(rounded_errors(rs, K.s, [P.p * i for i in is], R_of(P))) .<= P.p ÷ 4)
                @test maximum(abs(err(out[k], K.sk.s, Δ(is[k], phat))) for k in 1:P.B) < 1 / (16phat)
            end
        end
    end

    @testset "FFT refresh with key switch, B = 8 and B = 1" begin
        N = 64; r = Ring(N); fp = FPlan(N)
        for (p, phat, Gks, m) in ((16, 4, Gadget(6, 4, 3), 1), (16, 4, Gadget(6, 4), 2), (2N, 8, Gadget(6, 4), 2))
            P = CBParams(N = N, p = p, phat = phat, n = 12, Gbr = Gadget(8, 3), Gks = Gks, σ = 2.0^-40, h = 1)
            K = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, K; m, nthreads = 1)
            ksk = stair_keygen(K.sk.s, N, [(12, 1, 20)], rng; σf = x -> 2.0^-35)
            pool = zero_pool(K.s, 32, 2.0^-35, rng)
            for _ in 1:3
                is = rand(rng, 0:phat-1, P.B)
                cts = [lwe_encrypt(K.sk.s, Δ(i, phat), 2.0^-30, rng) for i in is]
                Vs = vstar_threshold(P.n; z = -1.28)
                out, rs = eb_refresh(P, fp, K, cts; ksk, m, wg, pool, Vstar = Vs, rng, info = true)
                @test all(x.V <= Vs for x in rs)
                @test all(abs.(rounded_errors(rs, K.s, [P.p * i for i in is], R_of(P))) .<= P.p ÷ 4)
                @test maximum(abs(err(out[k], K.sk.s, Δ(is[k], phat))) for k in 1:P.B) < 1 / (16phat)
            end
        end
    end

    @testset "N = 2048, 8 bits (p = 256, n = 896): end to end" begin
        N = 2048; r = Ring(N); fp = FPlan(N); phat = 256
        P = CBParams(N = N, p = 256, phat = phat, n = 896, Gbr = Gadget(11, 3), Gks = Gadget(11, 3, 4),
                     σ = 2.0^-50.3, h = 1)
        K = fast_keygen_grouped(P, r, fp, rng; m = 1); wg = WorkG(P, fp, K; m = 1, nthreads = 1)
        chain = [(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7),
                 (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20),
                 (912, 1, 19), (896, 1, 19)]
        ksk = stair_keygen(K.sk.s, N, chain, rng); pool = zero_pool(K.s, 128, σ128(P.n), rng)
        Vs = 4.2407e-03 * P.p^2 / 16                                     # V* of this configuration (16 V*/p^2 = 4.2407e-3)
        @test abs(Vs - vstar_threshold(P.n)) < 0.01
        σin = 2.0^-15                                                    # within the budget of this configuration
        E = Int[]; o = Float64[]; tr = Int[]
        for _ in 1:4
            is = rand(rng, 0:phat-1, P.B)
            cts = [lwe_encrypt(K.sk.s, Δ(i, phat), σin, rng) for i in is]
            out, rs = eb_refresh(P, fp, K, cts; ksk, wg, pool, Vstar = Vs, rng, info = true)
            append!(E, rounded_errors(rs, K.s, [P.p * i for i in is], R_of(P))); append!(tr, [x.trials for x in rs])
            append!(o, [err(out[k], K.sk.s, Δ(is[k], phat)) for k in 1:P.B])
            @test all(x.V <= Vs for x in rs)
        end
        @test maximum(abs, E) <= P.p ÷ 4
        @test maximum(abs, o) < 1 / (64phat)
    end
end
