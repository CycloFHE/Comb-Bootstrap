# Case (A) with the FFT backend (src/caseA_fast.jl) against case (B) with the FFT backend of fast.jl, single thread, same
# LWE dimension and gadgets, ring degree about 2048: the error bootstrap (case (A): p = 2039, N = 2038, zero-padded FFT,
# and p = 2017, N = 2016, Rader (p - 1 = 2^5·3^2·7); case (B): p = 4096, N = 2048; n = 816, BR (2^14,2)) and the comb
# bootstrap with B = 8 (case (A): p = 257, M0 = 16, N = 2048, Rader (p - 1 = 256); case (B): p = 512, N = 2048; n = 896,
# BR and aut (2^11,3), h = 1).
# The reference implementations of case (B) (blind_rotation_fast, combboot_fast_ref: allocations, rotations in the
# coefficient domain) have the structure of caseA_fast.jl; the optimized ones (separate_rotations_fast, combboot_fast)
# give the scale: case (A) optimized in the same way should cost about t_B,opt · t_A / t_B,ref.
# Noise: errors at the output of the rotation (E/R subtracted), against the predictions of the exact backend; and the
# error of the FFT itself: one external product and one key switching against the exact backend with the same keys
# (same encryptions) and the same input (same digits). Its variance, times about n for the external products and
# n* B (B - 1) for the key switchings, stays far below the output variance.
# Run (single thread):  julia --project=. benchmarks/timing_caseA_fft.jl [eb|comb|err|all] [count]
using Random, Printf, Statistics
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
include("../src/caseA_fast.jl"); using .CaseAFast
import .CombBoot: Gadget, lwe_phase, torus_frac, torus_to_float, CBParams, Ring, random_inputs, output_errors,
                  var_combboot, var_separate
import .Fast: σ128, FPlan, fast_keygen, blind_rotation_fast, combboot_fast_ref, combboot_fast, separate_rotations_fast,
              Work

what = length(ARGS) >= 1 ? ARGS[1] : "all"
count = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
Threads.nthreads() == 1 || @warn "single-thread comparison: start julia without -t"
ms(t) = 1e3 * t
std2(e) = 0.5 * log2(mean(abs2, e))

"Rotations of count random batches (one warm-up); times (s) per call and output errors."
function measure(rot, inputs, count)
    times = Float64[]; errs = Float64[]
    for i in 0:count
        aints, bints, E = inputs()
        t = @elapsed e = rot(aints, bints, E)
        i == 0 && continue
        push!(times, t); append!(errs, e)
    end
    return times, errs
end

function caseA(p, M0, n, Gbr, Gks, count; seed = 1, mode = nothing)
    Rg = ARing(p, M0); fp = mode === nothing ? AFPlan(Rg) : AFPlan(Rg; mode); B = Rg.B; N = Rg.N
    P = AParams(Rg; phat = 2^8, n, Gbr, Gks, σ = σ128(N))
    rng = MersenneTwister(seed)
    tk = @elapsed K = afkeygen(Rg, P, fp, rng)
    cb = B > 1 ? AFComb(Rg, fp) : nothing
    R = P.phat * p; emax = (p - 1) ÷ 2
    inputs() = begin
        aints = rand(rng, 0:p-1, B, n); E = rand(rng, -emax:emax, B)
        aints, [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + E[k], p) for k in 1:B], E
    end
    rot(aints, bints, E) = begin
        o = B == 1 ? [blind_rotation_A_fft(Rg, P, fp, K, view(aints, 1, :), bints[1])] :
                     combboot_A_fft(Rg, P, fp, K, aints, bints; cb)
        [torus_to_float(lwe_phase(o[k], K.sk.svec) - torus_frac(E[k], R)) for k in 1:B]
    end
    times, errs = measure(rot, inputs, count)
    vpred = B == 1 ? var_out_A(P, sum(K.s)) : var_comb_A(P, sum(K.s))
    @printf("  case (A), p = %d, M0 = %d, B = %d, N = %d, transform %s (keys %.0f s)\n", p, M0, B, N,
            fp isa CaseAFast.AFPlanR ? "Rader" : fp isa CaseAFast.AFPlanZ ? "zero padding" : "FFTW, size p", tk)
    @printf("    FFT          : %7.2f ms per call (min %.2f), %6.2f ms per error; σ_out 2^%.2f measured / 2^%.2f predicted (exact backend), max |e| 2^%.2f, %d outputs\n",
            ms(median(times)), ms(minimum(times)), ms(median(times)) / B, std2(errs), 0.5 * log2(vpred),
            log2(maximum(abs.(errs))), length(errs))
    return median(times)
end

function caseB(p, n, Gbr, Gks, count; seed = 2)
    N = 2048; r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = p, phat = 2^8, n = n, Gbr = Gbr, Gks = Gks, σ = σ128(N), h = 1)
    rng = MersenneTwister(seed)
    tk = @elapsed K = fast_keygen(P, r, fp, rng)
    B = P.B
    inputs() = random_inputs(P, K.s, rng)
    ref(aints, bints, E) = output_errors(P, B == 1 ? [blind_rotation_fast(P, fp, K, view(aints, 1, :), bints[1])] :
                                                     combboot_fast_ref(P, fp, K, aints, bints), K.sk.s, E)
    times, errs = measure(ref, inputs, count)
    vpred = B == 1 ? var_separate(P, sum(K.s)) : var_combboot(P, sum(K.s))
    @printf("  case (B), p = %d, B = %d, N = %d (keys %.0f s)\n", p, B, N, tk)
    @printf("    FFT, reference: %7.2f ms per call (min %.2f), %6.2f ms per error; σ_out 2^%.2f measured / 2^%.2f predicted (exact backend), %d outputs\n",
            ms(median(times)), ms(minimum(times)), ms(median(times)) / B, std2(errs), 0.5 * log2(vpred), length(errs))
    topt = NaN
    try
        ws = Work(P, fp, K)
        opt(aints, bints, E) = output_errors(P, B == 1 ? separate_rotations_fast(P, fp, K, aints, bints; ws) :
                                                         combboot_fast(P, fp, K, aints, bints; ws), K.sk.s, E)
        t2, e2 = measure(opt, inputs, count)
        topt = median(t2)
        @printf("    FFT, optimized: %7.2f ms per call (min %.2f), %6.2f ms per error; σ_out 2^%.2f\n",
                ms(topt), ms(minimum(t2)), ms(topt) / B, std2(e2))
    catch err
        println("    FFT, optimized: not available for B = $B ($(typeof(err)))")
    end
    return median(times), topt
end

function compare(label, a, b, B)
    tA = a; tB, tBopt = b
    @printf("  %s: case (A) / case (B), reference implementations: %.2f", label, tA / tB)
    isnan(tBopt) || @printf("; case (A) optimized as case (B): about %.2f ms per error", ms(tBopt * tA / tB) / B)
    println()
end

"FFT error of one external product (and one key switching) against the exact backend, same keys and inputs."
function fft_error(p, M0, G, Gk; trials = 20)
    Rg = ARing(p, M0); σ = σ128(Rg.N); fp = AFPlan(Rg)
    sk = akeygen_sk(Rg, MersenneTwister(1))
    Ke = aggsw_encrypt(Rg, sk, 1, G, σ, MersenneTwister(2)); Kf = afggsw(Rg, fp, sk, 1, G, σ, MersenneTwister(2))
    v(x) = mean(abs2, torus_to_float.(x))
    rng = MersenneTwister(3); de = Float64[]; dk = Float64[]
    for _ in 1:trials
        C = AGLWE(rand_aug(Rg, rng), rand_aug(Rg, rng))
        push!(de, v(aphase(Rg, sk, afextprod(Rg, fp, C, Kf)) .- aphase(Rg, sk, aextprod(Rg, C, Ke))))
        if Gk !== nothing
            c = Ugroup_A(Rg)[2]
            ke = aks_keygen(Rg, sk, c, Gk, σ, MersenneTwister(4)); kf = afks_keygen(Rg, fp, sk, c, Gk, σ, MersenneTwister(4))
            push!(dk, v(aphase(Rg, sk, afks(Rg, fp, C, kf)) .- aphase(Rg, sk, aks_hoisted(Rg, C, [ke])[1])))
        end
    end
    @printf("  p = %d, B = %d, transform %s: one external product (2^%d,%d) 2^%.2f", p, Rg.B,
            fp isa CaseAFast.AFPlanR ? "Rader" : "zero padding", G.logB, G.ell, 0.5log2(mean(de)))
    isempty(dk) || @printf(", one key switching (2^%d,%d) 2^%.2f", Gk.logB, Gk.ell, 0.5log2(mean(dk)))
    println(" (standard deviation per coordinate of the phase)")
end

if what in ("err", "all")
    println("Error of the FFT against the exact backend (same keys, same inputs)")
    fft_error(2039, 1, Gadget(14, 2), nothing)
    fft_error(2017, 1, Gadget(14, 2), nothing)
    fft_error(257, 16, Gadget(11, 3), Gadget(11, 3))
end
if what in ("eb", "all")
    println("Error bootstrap, n = 816, BR (2^14,2)")
    c = count > 0 ? count : 16
    a1 = caseA(2039, 1, 816, Gadget(14, 2), Gadget(1, 1), c)
    a2 = caseA(2017, 1, 816, Gadget(14, 2), Gadget(1, 1), c)
    b = caseB(4096, 816, Gadget(14, 2), Gadget(1, 1), c)
    compare("error bootstrap, p = 2039", a1, b, 1)
    compare("error bootstrap, p = 2017", a2, b, 1)
end
if what in ("comb", "all")
    println("Comb bootstrap, B = 8, n = 896, BR and aut (2^11,3), h = 1")
    c = count > 0 ? count : 4
    a = caseA(257, 16, 896, Gadget(11, 3), Gadget(11, 3), c)
    b = caseB(512, 896, Gadget(11, 3), Gadget(11, 3), c)
    compare("comb bootstrap", a, b, 8)
end
