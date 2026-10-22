# Case (A) with the optimized FFT engine (src/caseA_opt.jl) against case (B) with the optimized engine of fast_opt.jl
# (grouped.jl for m ≥ 2), single thread, same LWE dimension and gadgets, ring degree about 2048:
#   comb bootstrap, B = 8: case (A) p = 257, M0 = 16, N = 2048 (Rader, FFTs of size 256); case (B) p = 512, N = 2048;
#     n = 896, BR and aut (2^11,3), h = 1, mask grouped by m (environment variable M, default 1);
#   error bootstrap (m = 1): case (A) p = 2039, N = 2038 (zero padding to 4096) and p = 2017, N = 2016 (Rader, FFTs of
#     size 2016); case (B) p = 4096, N = 2048 (separate_rotations_fast); n = 816, BR (2^14,2).
# The two cases run alternately, REPS times (default 6, the first one discarded): minimum and median of the times.
# Output noise of case (A) against the prediction of the exact backend (sanity check).
# Run (single thread):  julia --project=. benchmarks/timing_caseA_opt.jl [comb|eb|all]     (M=2 for m = 2, comb only)
using Random, Printf, Statistics, FFTW
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
include("../src/caseA_fast.jl"); using .CaseAFast
import .CombBoot: Gadget, lwe_phase, torus_frac, torus_to_float, CBParams, Ring, random_inputs
import .Fast: σ128, FPlan, fast_keygen, fast_keygen_grouped, combboot_fast, combboot_grouped, separate_rotations_fast,
              Work, WorkG

what = length(ARGS) >= 1 ? ARGS[1] : "all"
const M = parse(Int, get(ENV, "M", "1"))
const REPS = parse(Int, get(ENV, "REPS", "6"))
Threads.nthreads() == 1 || @warn "single-thread comparison: start julia without -t"
ms(t) = 1e3 * t
rng = MersenneTwister(7)

function bench_comb()
    n = 896; Gbr = Gadget(11, 3); Gks = Gadget(11, 3)
    Rg = ARing(257, 16); Pa = AParams(Rg; phat = 256, n, Gbr, Gks, σ = σ128(2048))
    P = AOPlan(Rg); K = aokeygen(Rg, Pa, P, MersenneTwister(1); m = M); ws = AOWork(Rg, Pa, P; m = M)
    PB = CBParams(N = 2048, p = 512, phat = 256, n = n, Gbr = Gbr, Gks = Gks, σ = σ128(2048), h = 1)
    fp = FPlan(2048)
    KB = M == 1 ? fast_keygen(PB, Ring(2048), fp, MersenneTwister(2)) : fast_keygen_grouped(PB, Ring(2048), fp, MersenneTwister(2); m = M)
    wB = M == 1 ? Work(PB, fp, KB) : WorkG(PB, fp, KB; m = M)
    runB(a, b) = M == 1 ? combboot_fast(PB, fp, KB, a, b; ws = wB) : combboot_grouped(PB, fp, KB, a, b; m = M, wg = wB)
    tA = Float64[]; tB = Float64[]; errs = Float64[]
    for rep in 1:REPS
        aints = rand(rng, 0:256, 8, n); E = rand(rng, -128:128, 8)
        bints = [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + E[k], 257) for k in 1:8]
        t = @elapsed o = combboot_A_opt(Rg, Pa, K, aints, bints; ws)
        append!(errs, [torus_to_float(lwe_phase(o[k], K.sk.svec) - torus_frac(E[k], 256 * 257)) for k in 1:8])
        aB, bB, _ = random_inputs(PB, KB.s, rng)
        tb = @elapsed runB(aB, bB)
        rep > 1 && (push!(tA, t); push!(tB, tb))
    end
    @printf("Comb bootstrap, B = 8, n = %d, BR and aut (2^11,3), h = 1, m = %d\n", n, M)
    @printf("  case (A), p = 257, optimized: min %7.2f ms, median %7.2f ms per call; %5.2f ms per error (min); σ_out 2^%.2f (exact prediction 2^%.2f, %d outputs)\n",
            ms(minimum(tA)), ms(median(tA)), ms(minimum(tA)) / 8, 0.5log2(mean(abs2, errs)), 0.5log2(var_comb_A(Pa, sum(K.s); m = M)), length(errs))
    @printf("  case (B), p = 512, optimized: min %7.2f ms, median %7.2f ms per call; %5.2f ms per error (min)\n",
            ms(minimum(tB)), ms(median(tB)), ms(minimum(tB)) / 8)
    @printf("  case (A) / case (B): %.2f (minima), %.2f (medians)\n", minimum(tA) / minimum(tB), median(tA) / median(tB))
end

function bench_eb()
    n = 816; Gbr = Gadget(14, 2)
    cfg = [(2039, :pad), (2017, :rader)]
    As = map(cfg) do (p, mode)
        Rg = ARing(p, 1); Pa = AParams(Rg; phat = 256, n, Gbr, σ = σ128(p - 1))
        ws = AOWork1(Rg, Pa; mode); K = aokeygen1(Rg, Pa, ws.P1, MersenneTwister(1))
        (Rg, Pa, ws, K)
    end
    PB = CBParams(N = 2048, p = 4096, phat = 256, n = n, Gbr = Gbr, Gks = Gadget(1, 1), σ = σ128(2048), h = 1)
    fp = FPlan(2048); KB = fast_keygen(PB, Ring(2048), fp, MersenneTwister(2)); wB = Work(PB, fp, KB)
    tA = [Float64[] for _ in cfg]; eA = [Float64[] for _ in cfg]; tB = Float64[]
    for rep in 1:REPS
        for (i, (Rg, Pa, ws, K)) in enumerate(As)
            p = Rg.p; arow = rand(rng, 0:p-1, n); E = rand(rng, -(p - 1) ÷ 2:(p - 1) ÷ 2)
            bint = mod(sum(arow[u] * K.s[u] for u in 1:n) + E, p)
            t = @elapsed o = blind_rotation_A_opt(Rg, Pa, K, arow, bint; ws)
            push!(eA[i], torus_to_float(lwe_phase(o, K.sk.svec) - torus_frac(E, 256p)))
            rep > 1 && push!(tA[i], t)
        end
        aB, bB, _ = random_inputs(PB, KB.s, rng)
        tb = @elapsed separate_rotations_fast(PB, fp, KB, aB, bB; ws = wB)
        rep > 1 && push!(tB, tb)
    end
    @printf("Error bootstrap, n = %d, BR (2^14,2), m = 1\n", n)
    for (i, (p, mode)) in enumerate(cfg)
        Rg, Pa, ws, K = As[i]
        @printf("  case (A), p = %d, %-12s: min %6.2f ms, median %6.2f ms; σ_out 2^%.2f (exact prediction 2^%.2f, %d outputs); case (A) / case (B): %.2f (minima)\n",
                p, mode == :pad ? "zero padding" : "Rader", ms(minimum(tA[i])), ms(median(tA[i])), 0.5log2(mean(abs2, eA[i])),
                0.5log2(var_out_A(Pa, sum(K.s))), length(eA[i]), minimum(tA[i]) / minimum(tB))
    end
    @printf("  case (B), p = 4096, optimized  : min %6.2f ms, median %6.2f ms\n", ms(minimum(tB)), ms(median(tB)))
end

what in ("comb", "all") && bench_comb()
what in ("eb", "all") && M == 1 && bench_eb()
