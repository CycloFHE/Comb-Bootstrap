# Comparison curves: the error bootstrap at N = 2048 for p̂ = 4^k, k = 1..8 (k digits of 2 bits),
# against TFHE-rs on FheUint2..FheUint16 (tfhe-rs-bench, binary `curves`). One thread.
#  For each k, two configurations of the error bootstrap (B = 1, p = 2N, mask grouped by m = 2 in the rotation and
#  rounded coordinate by coordinate with recentering and selection (src/rounding.jl), polynomial key switch d = 16):
#   R: refresh after ν = 1000 additions: (4p̂)^2 (ν σ_out^2 + σ_KS^2) + 16 V*/(2N)^2 ≤ 1/U^2, U = 13.11 (2^-128);
#   P: refresh before a product (k ≤ 5): fresh inputs for the product of Section 6 (trace gadget Gtr, relinearization
#      (2^10,4)), whose output must satisfy the same condition with σ_prod instead of √ν σ_out.
#  The parameters are the cheapest ones of the cost model (noise of the paper + error of the FFT, 2^-85 B^2 ℓ per
#  external product, margin of 0.2 bit on the standard deviations), with the chains of the key switch of the cost model.
#  Beyond k = 5, the product is not possible at N = 2048 in double precision (noise of the keys of the trace, error of
#  the FFT).
#  PART=N: measured noise of each configuration (FFT backend for the rotations, exact backend for the product) and
#          margin of the condition, in bits (2 NREF refreshes, NPROD products, 400 key switches; default 250, 200);
#  PART=T: native times (rotation of the whole mask, key switch, product), in ms, minimum over REPS runs (default 5).
#  PCFG=exact: configurations P chosen with the exact law of the noise of the product (the noise model: its main
#          term is a Laplace law; P(fail) <= 2^-128 with standard deviations multiplied by 2^0.2), instead of the
#          Gaussian condition above (PCFG=gauss, default).
# Run from the julia folder of the repository:   PART=T julia --project -t 1 benchmarks/run_curves.jl      (KS=1,2,...: some k only)
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf, Statistics

const PART = get(ENV, "PART", "T")
const KS = parse.(Int, split(get(ENV, "KS", "1,2,3,4,5,6,7,8"), ","))
const REPS = parse(Int, get(ENV, "REPS", "5"))
const NREF = parse(Int, get(ENV, "NREF", "250"))
const NPROD = parse(Int, get(ENV, "NPROD", "200"))
const N = 2048; const σ = 2.0^-50.3; const U128 = 13.1086; const NU = 1000; const GRL = Gadget(10, 4); const M = 2

# k => (R = (n, gadget, 16 V*/(2N)^2, chain), P = (n, gadget, 16 V*/(2N)^2, chain, trace gadget) or nothing)
const CFG = Dict(
    1 => (R = (570, Gadget(23, 1), 1.1606e-05, [(1280, 15, 1), (960, 7, 2), (864, 5, 3), (816, 4, 4), (768, 3, 5), (736, 2, 7), (720, 2, 7), (704, 2, 7), (688, 2, 7), (672, 1, 13), (656, 1, 12), (640, 1, 12), (624, 1, 12), (608, 1, 11), (592, 1, 11), (576, 1, 11), (570, 1, 10)]),
          P = (576, Gadget(22, 1), 1.1735e-05, [(1280, 15, 1), (960, 7, 2), (864, 5, 3), (816, 4, 4), (768, 3, 5), (736, 2, 7), (720, 2, 7), (704, 2, 7), (688, 2, 7), (672, 1, 13), (656, 1, 12), (640, 1, 12), (624, 1, 12), (608, 1, 11), (592, 1, 11), (576, 1, 11)], Gadget(8, 5))),
    2 => (R = (650, Gadget(13, 2), 1.3319e-05, [(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13), (656, 1, 13), (650, 1, 13)]),
          P = (650, Gadget(15, 2), 1.3319e-05, [(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13), (656, 1, 13), (650, 1, 13)], Gadget(8, 5))),
    3 => (R = (732, Gadget(15, 2), 1.5080e-05, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (732, 1, 15)]),
          P = (768, Gadget(15, 2), 1.5854e-05, [(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16)], Gadget(8, 5))),
    4 => (R = (816, Gadget(14, 2), 1.6887e-05, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17)]),
          P = (816, Gadget(9, 4), 1.6887e-05, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17)], Gadget(7, 6))),
    5 => (R = (896, Gadget(15, 2), 1.8612e-05, [(1600, 13, 2), (1520, 12, 2), (1376, 8, 3), (1296, 6, 4), (1248, 5, 5), (1200, 4, 6), (1168, 3, 8), (1136, 3, 8), (1104, 2, 12), (1088, 2, 12), (1072, 2, 12), (1056, 1, 23), (1040, 1, 22), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (976, 1, 21), (960, 1, 21), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19)]),
          P = (906, Gadget(6, 7), 1.8828e-05, [(1600, 13, 2), (1520, 12, 2), (1376, 8, 3), (1296, 6, 4), (1248, 5, 5), (1200, 4, 6), (1168, 3, 8), (1136, 3, 8), (1104, 2, 12), (1088, 2, 12), (1072, 2, 12), (1056, 1, 23), (1040, 1, 22), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (976, 1, 21), (960, 1, 21), (944, 1, 20), (928, 1, 20), (912, 1, 19), (906, 1, 19)], Gadget(2, 23))),
    6 => (R = (976, Gadget(11, 3), 2.0340e-05, [(1600, 13, 2), (1520, 12, 2), (1376, 8, 3), (1296, 6, 4), (1248, 5, 5), (1200, 4, 6), (1168, 3, 8), (1136, 3, 8), (1104, 2, 12), (1088, 2, 12), (1072, 2, 12), (1056, 1, 23), (1040, 1, 22), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (976, 1, 21)]),
          P = nothing),
    7 => (R = (1088, Gadget(12, 3), 2.2763e-05, [(1776, 14, 2), (1616, 10, 3), (1488, 7, 4), (1424, 6, 5), (1360, 4, 7), (1328, 4, 7), (1296, 3, 9), (1264, 2, 14), (1248, 2, 14), (1232, 2, 14), (1216, 2, 13), (1200, 1, 26), (1184, 1, 26), (1168, 1, 26), (1152, 1, 25), (1136, 1, 25), (1120, 1, 25), (1104, 1, 24), (1088, 1, 24)]),
          P = nothing),
    8 => (R = (1152, Gadget(9, 4), 2.4150e-05, [(1776, 14, 2), (1616, 10, 3), (1488, 7, 4), (1424, 6, 5), (1360, 4, 7), (1328, 4, 7), (1296, 3, 9), (1264, 2, 14), (1248, 2, 14), (1232, 2, 14), (1216, 2, 13), (1200, 1, 26), (1184, 1, 26), (1168, 1, 26), (1152, 1, 25)]),
          P = nothing))

# k => P = (n, gadget, 16 V*/(2N)^2, chain, trace gadget), exact law of the product noise.
# k = 5: proposed by the cost model, but its measured output noise is 2.75 bits above the model of the error of the FFT
# (benchmarks/run_fft_floor.jl), and the refresh after the product then fails with probability 2^-72: no product at k = 5.
const CFG_EXACT = Dict(
    1 => (564, Gadget(13, 2), 1.1478e-05, [(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13), (672, 1, 13), (656, 1, 13), (640, 1, 12), (624, 1, 12), (608, 1, 11), (592, 1, 11), (576, 1, 11), (564, 1, 10)], Gadget(8, 5)),
    2 => (650, Gadget(15, 2), 1.3319e-05, [(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13), (656, 1, 13), (650, 1, 13)], Gadget(8, 5)),
    3 => (734, Gadget(11, 3), 1.5123e-05, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (734, 1, 15)], Gadget(8, 5)),
    4 => (816, Gadget(8, 5), 1.6887e-05, [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17)], Gadget(6, 8)),
    5 => (996, Gadget(2, 22), 2.0772e-05, [(1664, 13, 2), (1488, 9, 3), (1408, 7, 4), (1328, 5, 5), (1280, 4, 7), (1248, 3, 9), (1216, 3, 9), (1184, 2, 13), (1168, 2, 13), (1152, 2, 13), (1136, 2, 12), (1120, 1, 24), (1104, 1, 24), (1088, 1, 24), (1072, 1, 23), (1056, 1, 23), (1040, 1, 23), (1024, 1, 22), (1008, 1, 22), (996, 1, 22)], Gadget(2, 24)))
const PCFG = get(ENV, "PCFG", "gauss")
cfgP(k) = PCFG == "exact" ? get(CFG_EXACT, k, nothing) : CFG[k].P

l2(x) = 0.5 * log2(x)                                   # log2 of a standard deviation from a variance
Δ(i, phat) = UInt64(i) << (64 - trailing_zeros(phat))
gstr(G) = "(2^$(G.logB),$(G.ell))"
params(phat, n, G) = CBParams(N = N, p = 2N, phat = phat, n = n, Gbr = G, Gks = Gadget(14, 2), σ = σ, h = 1)  # B = 1

"Predicted variance of the output of the rotation (separate rotations, multi-bit form), with the error of the FFT."
function var_rot(n, m, G::Gadget; fft = true)
    g = n / m; b = G.logB; ℓ = G.ell
    key = 2ℓ * N * (2.0^(2b) + 2) / 12 * σ^2; rnd = (1 + N / 2) * 2.0^(-2ℓ * b) / 12
    return ((2^m - 1) * g * key + g * (1 - 2.0^-m) * rnd) * (m > 1 ? 2 : 1) + (fft ? g * 2.0^-85 * 2.0^(2b) * ℓ : 0.0)
end

"""Refreshed ciphertexts LWE_{s'}(i/p̂), dimension N, for i in is: error bootstrap (B = 1, FFT backend, eb_refresh of
src/fast_refresh.jl: rounding with recentering and selection below Vstar with the pool, rotation, subtraction) of
fresh LWE_s(i/p̂)."""
function refreshed(P, fp, K, wg, is, phat, rng; m = M, pool = nothing, Vstar = Inf)
    out = LWE[]
    for i in is
        append!(out, eb_refresh(P, fp, K, [lwe_encrypt(K.s, Δ(i, phat), 2.0^-20, rng)]; m, wg, pool, Vstar, rng, comb = false))
    end
    return out
end

"Measured variance of the key switch N -> n along `chain` (keys of the bootstrap: s = first n coefficients of s')."
function ks_var(sp, chain, rng; ns = 400)
    K = stair_keygen(sp, N, chain, rng); n = chain[end][1]
    e = Float64[]
    for _ in 1:ns
        μ = rand(rng, UInt64)
        push!(e, torus_to_float(lwe_phase(stair_ks(K, lwe_encrypt(sp, μ, 0.0, rng)), sp[1:n]) - μ))
    end
    return mean(abs2, e), stair_var(N, chain)
end

margin(phat, vin, vks, vterm) = l2((1 / U128^2) / ((4phat)^2 * (vin + vks) + vterm))

if occursin('N', PART)
    println("Noise (log2 σ, measured / predicted; margin of the condition in bits, measured noises), PCFG = $PCFG"); flush(stdout)
    for k in KS
        phat = 4^k; c = CFG[k]; tstart = time()
        for (lab, cfg) in (("refresh", c.R), ("before a product", cfgP(k)))
            cfg === nothing && continue
            n, G, vterm, chain = cfg[1], cfg[2], cfg[3], cfg[4]
            rng = Xoshiro(1000k + n); r = Ring(N); fp = FPlan(N)
            P = params(phat, n, G)
            K = fast_keygen_grouped(P, r, fp, rng; m = M); wg = WorkG(P, fp, K; m = M, nthreads = 1)
            sp = K.sk.s; pool = zero_pool(K.s, 128, σ128(n), rng); Vs = vterm * (2N)^2 / 16      # selection
            vks, vksp = ks_var(sp, chain, rng)
            if lab == "refresh"
                e = Float64[]
                for _ in 1:NREF
                    is = rand(rng, 0:phat-1, 2)
                    for (cc, i) in zip(refreshed(P, fp, K, wg, is, phat, rng; pool, Vstar = Vs), is)
                        push!(e, torus_to_float(lwe_phase(cc, sp) - Δ(i, phat)))
                    end
                end
                vo = mean(abs2, e)
                @printf("k = %d (p̂ = 2^%2d), refresh, n = %4d, %-9s out %7.2f / %7.2f (%7.2f without FFT)   KS %7.2f / %7.2f   margin %5.2f bits   (%.0f s)\n",
                        k, 2k, n, gstr(G), l2(vo), l2(var_rot(n, M, G)), l2(var_rot(n, M, G; fft = false)), l2(vks), l2(vksp),
                        margin(phat, NU * vo, vks, vterm), time() - tstart)
            else
                Gtr = cfg[5]
                raw = product_keys_raw(r, K.sk, Gtr, GRL, σ, rng); PK = ProductKeys(r, raw)
                FK = FProductKeys(fp, raw); ws = PWork(fp, max(Gtr.ell, GRL.ell))
                ein = Float64[]; trest = Float64[]; t0 = Float64[]; ep = Float64[]; epf = Float64[]
                for _ in 1:NPROD
                    is = rand(rng, 0:phat-1, 2)
                    c1, c2 = refreshed(P, fp, K, wg, is, phat, rng; pool, Vstar = Vs)
                    for (cc, i) in ((c1, is[1]), (c2, is[2]))
                        push!(ein, torus_to_float(lwe_phase(cc, sp) - Δ(i, phat)))
                    end
                    q = lwe_to_qrlwe(c1); ph = q_phase(r, K.sk, qtrace(r, q, PK))
                    push!(t0, q_to_float(qsub(ph[1], q_phase(r, K.sk, q)[1]))); push!(trest, sum(abs2, q_to_float.(ph[2:end])))
                    want = Δ(mod(is[1] * is[2], phat), phat)
                    push!(ep, torus_to_float(lwe_phase(lwe_product(r, PK, c1, c2, phat), sp) - want))
                    push!(epf, torus_to_float(lwe_phase(flwe_product(ws, FK, c1, c2, phat, fp), sp) - want))
                end
                ve = mean(abs2, ein); V0 = mean(abs2, t0); T = mean(trest) + V0; vp = mean(abs2, ep); vpf = mean(abs2, epf)
                @printf("k = %d (p̂ = 2^%2d), product, n = %4d, %-9s in %7.2f / %7.2f   trace %-9s √T %7.2f / %7.2f  √V0 %7.2f   product %7.2f / %7.2f (FFT %7.2f)   KS %7.2f / %7.2f   margin %5.2f bits   (%.0f s)\n",
                        k, 2k, n, gstr(G), l2(ve), l2(var_rot(n, M, G)), gstr(Gtr), l2(T), l2(sum(trace_profile(N)) * var_ks_gadget(N, Gtr, σ)), l2(V0),
                        l2(vp), l2(var_product(N, phat, ve, T, V0, GRL, σ)), l2(vpf), l2(vks), l2(vksp),
                        margin(phat, max(vp, vpf), vks, vterm), time() - tstart)
            end
            flush(stdout); K = nothing; wg = nothing; GC.gc()
        end
    end
end

"Workspace with the bootstrapping keys of the grouping by m, generated directly in split form (one key at a time)."
function lean_workspace(P::CBParams, r::Ring, fp::FPlan, m::Int, rng)
    sprime = Int64.(rand(rng, 0:1, P.N)); sk = SecretKey(r, sprime)
    Mg = 2 * P.N; γ = 1 + P.p
    baby = [Fast.fkskey(r, fp, sk, powermod(γ, j, Mg), P.Gks, P.σ, rng) for j in 1:P.g-1]
    K0 = Fast.FKeys(sk, sprime[1:P.n], Fast.FGGSWf[], baby, Fast.FKSf[])
    wg = WorkG(P, fp, K0; m, nthreads = 1)
    for t in 1:P.n÷m, π in 1:(1 << m) - 1
        bit = 1
        for i in 1:m
            si = Int(sprime[(t-1)*m+i])
            bit *= ((π >> (i - 1)) & 1) == 1 ? si : 1 - si
        end
        key = Fast.fggsw(r, fp, sk, bit, P.Gbr, P.σ, rng)
        push!(wg.ws.bsk, [(Fast.SV(x[1]), Fast.SV(x[2])) for x in key.rows])
    end
    return wg
end

"Time of one rotation of one ciphertext (all n coordinates, grouped by m = 2), in ms."
function rotation_time(n, G)
    rng = Xoshiro(7); r = Ring(N); fp = FPlan(N)
    P = params(4, n, G)
    wg = lean_workspace(P, r, fp, M, rng)
    mask = 2N - 1; v = Fast.vstar(P); a = rand(rng, 0:2N-1, n)
    function run1()
        C = Fast.trivial(Fast.mulX(v, 0))
        for t in 1:n÷M
            es = ntuple(i -> i <= M ? a[(t-1)*M+i] & mask : 0, 4)
            Fast.combine_key!(wg, es, t, P, fp); Fast._mbit!(C, P, wg, fp)
        end
        return C
    end
    run1()
    return 1e3 * minimum(@elapsed(run1()) for _ in 1:REPS)
end

"Time of the key switch N -> n along `chain`, in ms."
function ks_time(chain)
    rng = Xoshiro(8); sp = Int64.(rand(rng, 0:1, N))
    K = stair_keygen(sp, N, chain, rng); x = lwe_encrypt(sp, rand(rng, UInt64), 0.0, rng)
    stair_ks(K, x)
    return 1e3 * minimum(@elapsed(stair_ks(K, x)) for _ in 1:20REPS)
end

"Time of the product of two ciphertexts (FFT backend: two traces, tensor product, relinearization), in ms."
function product_time(phat, Gtr)
    rng = Xoshiro(9); r = Ring(N); fp = FPlan(N)
    sp = Int64.(rand(rng, 0:1, N)); sk = SecretKey(r, sp)
    raw = product_keys_raw(r, sk, Gtr, GRL, σ, rng)
    FK = FProductKeys(fp, raw); ws = PWork(fp, max(Gtr.ell, GRL.ell))
    c1 = lwe_encrypt(sp, Δ(3 % phat, phat), 2.0^-30, rng); c2 = lwe_encrypt(sp, Δ(phat - 1, phat), 2.0^-30, rng)
    o = flwe_product(ws, FK, c1, c2, phat, fp)
    abs(torus_to_float(lwe_phase(o, sp) - Δ(mod(3 * (phat - 1), phat), phat))) < 1 / (4phat) || error("product: wrong result")
    return 1e3 * minimum(@elapsed(flwe_product(ws, FK, c1, c2, phat, fp)) for _ in 1:20REPS)
end

"Time of the addition of two LWE ciphertexts of dimension N, in ms."
function add_time()
    rng = Xoshiro(10); a = LWE(rand(rng, UInt64, N), rand(rng, UInt64)); b = LWE(rand(rng, UInt64, N), rand(rng, UInt64))
    f() = LWE(a.a .+ b.a, a.b + b.b)
    f()
    return 1e3 * minimum(@elapsed(f()) for _ in 1:1000)
end

if occursin('T', PART)
    println("Times (native, one thread, ms): error bootstrap at N = 2048, m = 2, d = 16, PCFG = $PCFG"); flush(stdout)
    csv = String[]
    tadd = add_time()
    @printf("  addition of two ciphertexts (dimension N): %.5f ms\n", tadd)
    for k in KS
        phat = 4^k; c = CFG[k]
        n, G, _, chain = c.R
        tr = rotation_time(n, G); tk = ks_time(chain); GC.gc()
        @printf("  k = %d (%2d bits) refresh:           n = %4d, %-9s rotation %7.2f + key switch %6.3f = %7.2f\n",
                k, 2k, n, gstr(G), tr, tk, tr + tk)
        push!(csv, @sprintf("eb,%d,refresh,%.4f", k, tr + tk), @sprintf("eb,%d,addition,%.5f", k, tadd + (tr + tk) / NU))
        if cfgP(k) !== nothing
            n, G, _, chain, Gtr = cfgP(k)
            tr = rotation_time(n, G); tk = ks_time(chain); tp = product_time(phat, Gtr); GC.gc()
            @printf("  k = %d (%2d bits) before a product:  n = %4d, %-9s rotation %7.2f + key switch %6.3f = %7.2f;  product, trace %-9s %6.3f;  total %7.2f\n",
                    k, 2k, n, gstr(G), tr, tk, tr + tk, gstr(Gtr), tp, tr + tk + tp)
            push!(csv, @sprintf("eb,%d,product,%.4f", k, tr + tk + tp))
        end
        flush(stdout)
    end
    println("CSV (scheme,k,operation,ms)"); foreach(println, csv)
end
