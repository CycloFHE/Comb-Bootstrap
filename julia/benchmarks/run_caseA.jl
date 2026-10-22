# Case (A), exact backend (src/caseA.jl): the error bootstrap in the ring of conductor p = 2039 (N = 2038) and the comb
# bootstrap in the ring of conductor 16·257 (B = 8, N = 2048), with the whole refresh (polynomial stair key switch,
# rounding at R = p̂ p with recentering and selection, rotation, subtraction). Measured noises against the predictions,
# errors E within the full window (p - 1)/2, margins at failure probability 2^-128 after ν = 1000 additions, and the
# message modulus allowed by the measured noise, against case (B) with the same keys (p = 2N, window p/4).
# Keys at σ_128(N) (the security of the ring of conductor p is assumed to be that of a power-of-two ring of the same
# degree, Appendix I). Times are those of the exact backend (NTT products), not comparable with Section 7.
# BACKEND=fft: the FFT backend (src/caseA_fast.jl; zero-padded FFT for p = 2039, Rader for p = 257), whose rounding errors
# add to the noise; its times are those of a reference implementation (see benchmarks/timing_caseA_fft.jl).
# BACKEND=opt: the optimized FFT engine (src/caseA_opt.jl), times comparable with fast_opt.jl (benchmarks/timing_caseA_opt.jl).
# Run:  julia -t 2 --project=. benchmarks/run_caseA.jl [eb|comb|all] [count]
#       BACKEND=fft (or opt) julia -t 2 --project=. benchmarks/run_caseA.jl [eb|comb|all] [count]
using Random, Printf, Statistics
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
include("../src/caseA_fast.jl"); using .CaseAFast
import .CombBoot: Gadget, LWE, lwe_phase, torus_frac, torus_to_float, lwe_encrypt, zero_pool, vstar_threshold,
                  rounded_errors
import .Fast: stair_keygen, stair_ks, stair_var, σ128, StairKey, FPlan

const U128 = 13.1086; const NU = 1000
const BACKEND = get(ENV, "BACKEND", "exact")
const FFTB = BACKEND == "fft"
const OPTB = BACKEND == "opt"
what = length(ARGS) >= 1 ? ARGS[1] : "all"
count = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

"Margin (bits) of (w p̂)^2 (ν σ_out^2 + σ_KS^2) + w^2 V*/p^2 ≤ u^-2, w = 2 (case A) or 4 (case B)."
margin(w, phat, vout, vks, Vs, p) = 0.5 * log2(U128^-2 / ((w * phat)^2 * (NU * vout + vks) + w^2 * Vs / p^2))
"Largest p̂ for the noise: (w p̂)^2 (ν σ_out^2 + σ_KS^2) + w^2 V*/p^2 = u^-2."
phatmax(w, vout, vks, Vs, p) = sqrt(U128^-2 - w^2 * Vs / p^2) / (w * sqrt(NU * vout + vks))

function run(label, p, M0, phat, n, Gbr, Gks, chain; m = 1, count = 8, seed = 1)
    Rg = ARing(p, M0); N = Rg.N; B = Rg.B
    σ = σ128(N)
    P = AParams(Rg; phat, n, Gbr, Gks, σ)
    rng = MersenneTwister(seed)
    t0 = time()
    # FFT backend: the plans hold buffers, one per task (keys and tables are shared)
    fps = FFTB ? [AFPlan(Rg) for _ in 1:max(2, Threads.nthreads())] : nothing
    cbs = FFTB && B > 1 ? [AFComb(Rg, f) for f in fps] : nothing
    # optimized engine: one workspace per task (keys shared)
    ows = OPTB ? [B == 1 ? AOWork1(Rg, P) : AOWork(Rg, P, AOPlan(Rg); m) for _ in 1:max(2, Threads.nthreads())] : nothing
    K = FFTB ? afkeygen(Rg, P, fps[1], rng; m) :
        OPTB ? (B == 1 ? aokeygen1(Rg, P, ows[1].P1, rng; m) : aokeygen(Rg, P, ows[1].P, rng; m)) : akeygen(Rg, P, rng; m)
    ksk = stair_keygen(K.sk.svec, 16 * cld(N, 16), chain, rng)
    pool = zero_pool(K.s, 128, σ128(n), rng)
    Vs = vstar_threshold(n)
    @printf("%s, %s backend: p = %d, M0 = %d, B = %d, N = %d, p̂ = 2^%d, n = %d, m = %d, BR (2^%d,%d), aut (2^%d,%d), σ = 2^%.2f, keys %.0f s\n",
            label, FFTB ? "FFT" : OPTB ? "optimized FFT" : "exact", p, M0, B, N, round(Int, log2(phat)), n, m, Gbr.logB, Gbr.ell, Gks.logB, Gks.ell,
            log2(σ), time() - t0)
    R = phat * p
    outs = Vector{Vector{Float64}}(undef, count); Es = Vector{Vector{Int}}(undef, count)
    kse = Vector{Vector{Float64}}(undef, count); times = zeros(count)
    # the FFT plans of the stair key switch hold buffers: one copy of the key per thread
    ksks = [StairKey(ksk.d, ksk.T, ksk.n, ksk.steps, FPlan(ksk.d)) for _ in 1:Threads.nthreads()]
    # one task per copy (threadid() is not an index: with interactive threads it may exceed nthreads())
    nt = length(ksks)
    @sync for c in 1:nt
        Threads.@spawn begin
            kt = ksks[c]
            for i in c:nt:count
                r = MersenneTwister(seed * 1000 + i)
                mus = rand(r, 0:phat-1, B)
                cts = [lwe_encrypt(K.sk.svec, torus_frac(μ, phat), 2.0^-30, r) for μ in mus]
                kse[i] = [torus_to_float(lwe_phase(stair_ks(kt, ct), K.s) - lwe_phase(ct, K.sk.svec)) for ct in cts]
                t = time()
                o, rs, _ = FFTB ?
                    refresh_A_fft(Rg, P, fps[c], K, cts; ksk = kt, pool, Vstar = Vs, rng = r, info = true,
                                  cb = B > 1 ? cbs[c] : nothing) :
                    OPTB ? refresh_A_opt(Rg, P, K, cts; ws = ows[c], ksk = kt, pool, Vstar = Vs, rng = r, info = true) :
                    refresh_A(Rg, P, K, cts; ksk = kt, pool, Vstar = Vs, rng = r, info = true)
                times[i] = time() - t
                outs[i] = [torus_to_float(lwe_phase(o[k], K.sk.svec) - torus_frac(mus[k], phat)) for k in 1:B]
                Es[i] = rounded_errors(rs, K.s, [mus[k] * p for k in 1:B], R)
            end
        end
    end
    e = reduce(vcat, outs); E = reduce(vcat, Es); ek = reduce(vcat, kse)
    nstar = sum(K.s[(t-1)*m+1:t*m] != zeros(Int, m) for t in 1:n÷m)
    vpred = B == 1 ? var_out_A(P, nstar; m) : var_comb_A(P, nstar; m)
    vks_pred = stair_var(16 * cld(N, 16), chain)
    vout = mean(e .^ 2); vks = mean(ek .^ 2)
    @printf("  %d outputs: σ_out 2^%.2f measured / 2^%.2f predicted, max |e| 2^%.2f; σ_KS 2^%.2f / 2^%.2f; |E| ≤ %d (window %d); %.1f s per refresh of %d\n",
            length(e), 0.5 * log2(vout), 0.5 * log2(vpred), log2(maximum(abs.(e))), 0.5 * log2(vks), 0.5 * log2(vks_pred),
            maximum(abs.(E)), (p - 1) ÷ 2, mean(times), B)
    mA = margin(2, phat, vout, vks, Vs, p)
    @printf("  margin at 2^-128 after ν = %d additions: %.2f bits; largest p̂ for this noise: 2^%.2f\n",
            NU, mA, log2(phatmax(2, vout, vks, Vs, p)))
    # case (B) with the same keys: p = 2N (B = 1) or p = 2N/B, window p/4, output noise without the doubling
    pB = B == 1 ? 2 * 2^ceil(Int, log2(N)) : 2 * 2^ceil(Int, log2(N)) ÷ B
    @printf("  case (B), same keys (p = %d, predicted noise of case (B) = half): margin %.2f bits, largest p̂ 2^%.2f\n",
            pB, margin(4, phat, vpred / 2, vks, Vs, pB), log2(phatmax(4, vpred / 2, vks, Vs, pB)))
    # the full window: noise-free inputs with errors at the edges ±(p - 1)/2 and inside, rotated directly
    emax = (p - 1) ÷ 2; nw = max(2, Threads.nthreads()); ws = zeros(nw)
    @sync for i in 1:nw
        Threads.@spawn begin
        r = MersenneTwister(seed * 7000 + i)
        aints = rand(r, 0:p-1, B, n)
        Ew = [isodd(i + k) ? emax : -emax for k in 1:B]; B > 2 && (Ew[3:end] .= rand(r, -emax:emax, B - 2))
        bints = [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + Ew[k], p) for k in 1:B]
        o = FFTB ? (B == 1 ? [blind_rotation_A_fft(Rg, P, fps[i], K, view(aints, 1, :), bints[1])] :
                             combboot_A_fft(Rg, P, fps[i], K, aints, bints; cb = cbs[i])) :
            OPTB ? (B == 1 ? [blind_rotation_A_opt(Rg, P, K, view(aints, 1, :), bints[1]; ws = ows[i])] :
                             combboot_A_opt(Rg, P, K, aints, bints; ws = ows[i])) :
                   (B == 1 ? [blind_rotation_A(Rg, P, K, view(aints, 1, :), bints[1])] : combboot_A(Rg, P, K, aints, bints))
        w = maximum(abs(torus_to_float(lwe_phase(o[k], K.sk.svec) - torus_frac(Ew[k], R))) for k in 1:B)
        ws[i] = w
        end
    end
    worst = maximum(ws)
    @printf("  errors at the edges of the window ±%d: largest output error 2^%.2f (exact correction E/R)\n", emax, log2(worst))
    return (; vout, vpred, vks, E, mA)
end

const CHAIN816 = [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10),
                  (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18),
                  (832, 1, 17), (816, 1, 17)]
const CHAIN896 = [(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7),
                  (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20),
                  (912, 1, 19), (896, 1, 19)]

if what in ("eb", "all")
    run("error bootstrap (A)", 2039, 1, 2^8, 816, Gadget(14, 2), Gadget(1, 1), CHAIN816; count = count > 0 ? count : 64)
end
if what in ("comb", "all")
    run("comb bootstrap (A)", 257, 16, 2^8, 896, Gadget(11, 3), Gadget(11, 3), CHAIN896; count = count > 0 ? count : 8)
end
