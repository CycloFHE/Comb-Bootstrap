# Tests of the product of two LWE ciphertexts (src/product.jl, src/fast_product.jl). Run from the julia folder of the repository:
#     julia test/test_product.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Test, Random

const rng = Xoshiro(23)

@testset "product" begin
    # arithmetic modulo q' = 2^64 - 1
    x = [CombBoot.qnorm(v) for v in rand(rng, UInt64, 1000)]
    @test all(qfrom(Int128(qlift(qdivN(v, 6))) * 64) == v for v in x)
    @test all(qadd(v, qneg(v)) == 0 for v in x)
    @test all(qfrom(Int128(qlift(v))) == v for v in x)
    # 16-bit limbs and scaling
    t = rand(rng, UInt64, 100)
    X = limbs16(t)
    @test all(sum(Int128(X[u][i]) << (16(u-1)) for u in 1:4) == Int128(reinterpret(Int64, t[i])) for i in 1:100)
    @test all(abs(X[u][i]) <= 2^15 + 1 for u in 1:4, i in 1:100)
    for N in (32, 64)
        r = Ring(N); fp = FPlan(N)
        sp = Int64.(rand(rng, 0:1, N)); sk = SecretKey(r, sp)
        raw = product_keys_raw(r, sk, Gadget(8, 6), Gadget(10, 4), 2.0^-45, rng)
        K = ProductKeys(r, raw); FK = FProductKeys(fp, raw); ws = PWork(fp, 6)
        for phat in (4, 16, 256), trial in 1:4
            Δ(i) = UInt64(round(BigInt, big(2)^64 * i // phat) % big(2)^64)
            i1, i2 = rand(rng, 0:phat-1), rand(rng, 0:phat-1)
            c1 = lwe_encrypt(sp, Δ(i1), 2.0^-40, rng); c2 = lwe_encrypt(sp, Δ(i2), 2.0^-40, rng)
            # trace: constant coefficient μ + e, the others ~ 0 (exact and FFT)
            q = lwe_to_qrlwe(c1)
            te = qtrace(r, q, K); tf = ftrace(ws, q, FK, fp)
            @test te.a == tf.a && te.b == tf.b                 # split keys: the FFT trace is exact
            for tq in (te, tf)
                ph = q_phase(r, sk, tq)
                @test abs(q_to_float(qsub(ph[1], q_phase(r, sk, q)[1]))) < 1e-6
                @test maximum(abs.(q_to_float.(ph[2:end]))) < 1e-6
            end
            # tensor product: FFT = exact
            t1 = lwe_to_rlwe_clean(r, K, c1); t2 = lwe_to_rlwe_clean(r, K, c2)
            @test all(tensor(r, t1, t2, phat) .== ftensor(ws, t1, t2, phat, fp))
            # product: LWE_{s'}((i1 i2 mod p̂)/p̂)
            for o in (lwe_product(r, K, c1, c2, phat), flwe_product(ws, FK, c1, c2, phat, fp))
                @test abs(torus_to_float(lwe_phase(o, sp) - Δ(mod(i1 * i2, phat)))) < 1e-4
            end
        end
        # predicted trace profile: constant coefficient (4^L - 1)/3
        @test trace_profile(N)[1] == (4^trailing_zeros(N) - 1) / 3
    end
end
