# Product of two LWE ciphertexts (Section 6 of the paper) at N = 2048: noise and times, one thread.
#  PART=N: noise. The inputs are refreshed by the error bootstrap (B = 1, p = 2N, mask grouped by m = 2 in the rotation,
#          rounding with recentering and selection (src/rounding.jl), FFT backend), the products are computed with the exact backend (and the FFT backend for comparison):
#          input noise, noise of the trace (constant coefficient and total), noise of the product, against the
#          predictions. CFG = 1..7 selects (p̂, n, bootstrapping gadget, trace gadget); NPROD products (default 150).
#  PART=T: times of the FFT backend (conversion and trace of one input, tensor product, relinearization, total).
# Run from the julia folder of the repository:   PART=N CFG=1 julia benchmarks/run_product.jl        PART=T julia benchmarks/run_product.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf, Statistics

const PART = get(ENV, "PART", "NT")
const NPROD = parse(Int, get(ENV, "NPROD", "150"))
const REPS = parse(Int, get(ENV, "REPS", "20"))
const N = 2048; const σ = 2.0^-50.3; const U128 = 13.1086
const CFGS = [(256, 830, Gadget(9, 4), Gadget(8, 6)), (256, 830, Gadget(9, 4), Gadget(5, 10)),
              (256, 830, Gadget(9, 4), Gadget(4, 12)), (64, 780, Gadget(16, 2), Gadget(8, 6)),
              (256, 830, Gadget(12, 3), Gadget(5, 10)), (64, 780, Gadget(12, 3), Gadget(8, 6)),
              (64, 780, Gadget(13, 3), Gadget(8, 6))]
const GRL = Gadget(10, 4)

l2(x) = 0.5 * log2(x)                                   # log2 of a standard deviation from a variance
Δ(i, phat) = UInt64(i) << (64 - trailing_zeros(phat))

"Predicted variance of the output of the rotation (separate rotations, multi-bit form for m ≥ 2)."
function var_rot(n, m, G::Gadget)
    g = n / m; b = G.logB; ℓ = G.ell
    key = 2ℓ * N * (2.0^(2b) + 2) / 12 * σ^2; rnd = (1 + N / 2) * 2.0^(-2ℓ * b) / 12
    return ((2^m - 1) * g * key + g * (1 - 2.0^-m) * rnd) * (m > 1 ? 2 : 1)
end

"""Refreshed ciphertexts LWE_{s'}(i/p̂), dimension N, for i in is: error bootstrap (B = 1, eb_refresh of
src/fast_refresh.jl, rounding with recentering and selection below V* = vstar_threshold(n)) of fresh LWE_s(i/p̂)."""
function refreshed(P, fp, K, wg, is, phat, rng; m = 2, pool = nothing)
    out = LWE[]
    for i in is
        append!(out, eb_refresh(P, fp, K, [lwe_encrypt(K.s, Δ(i, phat), 2.0^-20, rng)]; m, wg, pool,
                                Vstar = vstar_threshold(P.n), rng, comb = false))
    end
    return out
end

if occursin('N', PART)
    for c in parse.(Int, split(get(ENV, "CFG", "1,2,3,4"), ","))
        phat, n, Gbr, Gtr = CFGS[c]; m = 2; tstart = time()
        rng = Xoshiro(100 + c); r = Ring(N); fp = FPlan(N)
        P = CBParams(N = N, p = 2N, phat = phat, n = n, Gbr = Gbr, Gks = Gadget(14, 2), σ = σ, h = 1)     # B = 1
        K = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, K; m, nthreads = 1)
        pool = zero_pool(K.s, 128, σ128(n), rng)
        raw = product_keys_raw(r, K.sk, Gtr, GRL, σ, rng); PK = ProductKeys(r, raw)
        FK = FProductKeys(fp, raw); ws = PWork(fp, max(Gtr.ell, GRL.ell))
        sp = K.sk.s; sk = K.sk
        ein = Float64[]; t0 = Float64[]; trest = Float64[]; eprod = Float64[]; eprodf = Float64[]
        for t in 1:NPROD
            is = rand(rng, 0:phat-1, 2)
            c1, c2 = refreshed(P, fp, K, wg, is, phat, rng; m, pool)
            for (cc, i) in ((c1, is[1]), (c2, is[2]))
                push!(ein, torus_to_float(lwe_phase(cc, sp) - Δ(i, phat)))
            end
            # noise of the trace (exact), input 1: constant coefficient minus the phase of the input, other coefficients
            q = lwe_to_qrlwe(c1); ph = q_phase(r, sk, qtrace(r, q, PK))
            push!(t0, q_to_float(qsub(ph[1], q_phase(r, sk, q)[1])))
            push!(trest, sum(abs2, q_to_float.(ph[2:end])))
            o = lwe_product(r, PK, c1, c2, phat); of = flwe_product(ws, FK, c1, c2, phat, fp)
            want = Δ(mod(is[1] * is[2], phat), phat)
            push!(eprod, torus_to_float(lwe_phase(o, sp) - want)); push!(eprodf, torus_to_float(lwe_phase(of, sp) - want))
        end
        vks = var_ks_gadget(N, Gtr, σ); prof = trace_profile(N)
        ve = mean(abs2, ein); V0 = mean(abs2, t0); T = mean(trest) + V0
        win = 1 / (4 * U128 * phat)
        @printf("p̂ = %d, bootstrap n = %d, m = %d, (2^%d,%d); trace (2^%d,%d); relinearization (2^%d,%d); %d products\n",
                phat, n, m, Gbr.logB, Gbr.ell, Gtr.logB, Gtr.ell, GRL.logB, GRL.ell, NPROD)
        @printf("  inputs (refreshed):          log2 σ_e   = %7.2f   predicted %7.2f\n", l2(ve), l2(var_rot(n, m, Gbr)))
        @printf("  trace, constant coefficient: log2 σ_t0  = %7.2f   predicted %7.2f  ((4^L-1)/3 V_KS)\n", l2(V0), l2(prof[1] * vks))
        @printf("  trace, all coefficients:     log2 √T    = %7.2f   predicted %7.2f\n", l2(T), l2(sum(prof) * vks))
        @printf("  product (exact):             log2 σ     = %7.2f   predicted %7.2f   without the trace %7.2f   p̂√(N/12)σ_e %7.2f\n",
                l2(mean(abs2, eprod)), l2(var_product(N, phat, ve, T, V0, GRL, σ)), l2(var_product(N, phat, ve, 0.0, 0.0, GRL, σ)),
                log2(phat * sqrt(N / 12) * sqrt(ve)))
        @printf("  product (FFT):               log2 σ     = %7.2f\n", l2(mean(abs2, eprodf)))
        @printf("  window 1/(4 u p̂): log2 = %.2f, margin %.2f bits; max |error| = %.3f of 1/(4p̂)\n",
                log2(win), log2(win) - l2(mean(abs2, eprod)), maximum(abs.(eprod)) * 4phat)
        @printf("  (%.0f s)\n", time() - tstart); flush(stdout)
    end
end

if occursin('T', PART)
    println("Times (FFT backend), N = 2048, one thread, ms")
    rng = Xoshiro(7); r = Ring(N); fp = FPlan(N)
    sp = Int64.(rand(rng, 0:1, N)); sk = SecretKey(r, sp)
    for Gtr in (Gadget(8, 6), Gadget(5, 10), Gadget(4, 12))
        raw = product_keys_raw(r, sk, Gtr, GRL, σ, rng); FK = FProductKeys(fp, raw); ws = PWork(fp, max(Gtr.ell, GRL.ell))
        c1 = lwe_encrypt(sp, Δ(3, 256), 2.0^-27, rng); c2 = lwe_encrypt(sp, Δ(5, 256), 2.0^-27, rng)
        tm(f) = (f(); minimum(@elapsed(f()) for _ in 1:REPS) * 1e3)
        t1 = flwe_to_rlwe_clean(ws, FK, c1, fp); t2 = flwe_to_rlwe_clean(ws, FK, c2, fp)
        D0, D1, D2 = ftensor(ws, t1, t2, 256, fp)
        ttr = tm(() -> flwe_to_rlwe_clean(ws, FK, c1, fp))
        tten = tm(() -> ftensor(ws, t1, t2, 256, fp))
        trl = tm(() -> frelin(ws, D0, D1, D2, FK, fp))
        ttot = tm(() -> flwe_product(ws, FK, c1, c2, 256, fp))
        @printf("  trace (2^%d,%d): conversion and trace of one input %.3f, tensor product %.3f, relinearization (2^%d,%d) %.3f, product %.3f\n",
                Gtr.logB, Gtr.ell, ttr, tten, GRL.logB, GRL.ell, trl, ttot)
    end
end
