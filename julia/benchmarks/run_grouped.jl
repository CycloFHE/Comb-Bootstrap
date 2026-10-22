# Grouping of the mask by m = 1..4 and number of giant steps h = 1, 2 (src/grouped.jl), FFT backend:
# time per error of the comb bootstrap and of the separate rotations (batched; multi-bit form for m ≥ 2), measured noise,
# and time of each part of a group of the comb bootstrap.
# Run from the julia folder of the repository:   julia benchmarks/run_grouped.jl      (QUICK=1: n = 48; P=128 or P=512: other p;
# L=3: gadgets (2^10,3) / (2^6,6) with σ = 2^-40 instead of (2^23,1) / (2^14,2) with σ = 2^-50.3;
# H=1 or H=2: one value of h only; M=1,2: some values of m only; REPS=5: number of runs, the best is kept)
# julia --project -t 8 benchmarks/run_grouped.jl: also 8 threads for the B errors (threads inside each group of the comb bootstrap /
# the B rotations shared between the threads); TH=4,8: 4 then 8 threads (at most the number given to julia -t).
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf

const QUICK = get(ENV, "QUICK", "0") == "1"
rng = Xoshiro(23)
N = 2048; phat = 4
p = parse(Int, get(ENV, "P", "256"))
n = QUICK ? 48 : 624                     # multiple of 1, 2, 3, 4
l3 = get(ENV, "L", "1") == "3"
Gbr, Gks, σ = l3 ? (Gadget(10, 3), (2N ÷ p <= 16 ? Gadget(6, 6) : Gadget(5, 8)), 2.0^-40) :
                   (Gadget(23, 1), Gadget(14, 2), 2.0^-50.3)
r = Ring(N); fp = FPlan(N)
meas(e) = sum(abs2, e) / length(e)
const REPS = parse(Int, get(ENV, "REPS", "5"))
best(f) = minimum(@elapsed(f()) for _ in 1:REPS)
const PH = ["division", "FFTs, digits, tables", "baby + public", "giant steps", "inv FFT, digits C_L-C",
            "ext. product", "final inv FFT"]
const T = Threads.nthreads()
const TH = filter(t -> 1 < t <= T, parse.(Int, split(get(ENV, "TH", string(T)), ",")))
function phases(f)                                             # phase times (µs per coordinate), minimum over REPS
    tim = fill(Inf, 7)
    for _ in 1:REPS
        t7 = zeros(7); f(t7); tim .= min.(tim, t7)
    end
    for (nm, x) in zip(PH, tim)
        @printf(" %s %.1f;", nm, x / (1e3 * n))
    end
    @printf(" total %.1f\n", sum(tim) / (1e3 * n))
end

println("N = $N, p = $p, n = $n, RGSW (2^$(Gbr.logB),$(Gbr.ell)), aut (2^$(Gks.logB),$(Gks.ell)), σ = 2^$(round(log2(σ), digits=1))")
HS = parse.(Int, split(get(ENV, "H", "2,1"), ","))
MS = parse.(Int, split(get(ENV, "M", "1,2,3,4"), ","))
for h in HS
    P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = h)
    println("\nh = $h, (g, h) = ($(P.g), $(P.h)), B = $(P.B)")
    for m in MS
        K = fast_keygen_grouped(P, r, fp, rng; m)
        wg = WorkG(P, fp, K; m)
        aints, bints, E = random_inputs(P, K.s, rng)
        a2 = aints[:, 1:m]
        combboot_grouped(P, fp, K, a2, bints; m, wg)            # warm-up
        separate_rotations_grouped(P, fp, K, a2, bints; m, wg)
        t_mb = best(() -> combboot_grouped(P, fp, K, aints, bints; m, wg))
        t_sep = best(() -> separate_rotations_grouped(P, fp, K, aints, bints; m, wg))
        e1 = output_errors(P, combboot_grouped(P, fp, K, aints, bints; m, wg), K.sk.s, E)
        e2 = output_errors(P, separate_rotations_grouped(P, fp, K, aints, bints; m, wg), K.sk.s, E)
        @printf("  m = %d: comb bootstrap %5.2f ms/error, separate %5.2f ms/error; ratio %.2f; log2 std %.2f / %.2f\n",
                m, 1e3t_mb / P.B, 1e3t_sep / P.B, t_sep / t_mb, 0.5log2(meas(e1)), 0.5log2(meas(e2)))
        print("        µs per coordinate:")
        phases(t7 -> combboot_grouped(P, fp, K, aints, bints; m, wg, tim = t7))
        for t in TH
            combboot_grouped(P, fp, K, a2, bints; m, wg, nthreads = t)          # warm-up
            separate_rotations_grouped(P, fp, K, a2, bints; m, wg, nthreads = t)
            t_mbT = best(() -> combboot_grouped(P, fp, K, aints, bints; m, wg, nthreads = t))
            t_sepT = best(() -> separate_rotations_grouped(P, fp, K, aints, bints; m, wg, nthreads = t))
            @printf("     %d threads: comb bootstrap %5.2f ms/error (×%.1f), separate %5.2f ms/error (×%.1f); ratio %.2f\n",
                    t, 1e3t_mbT / P.B, t_mb / t_mbT, 1e3t_sepT / P.B, t_sep / t_sepT, t_sepT / t_mbT)
            print("        µs per coordinate:")
            phases(t7 -> combboot_grouped(P, fp, K, aints, bints; m, wg, nthreads = t, tim = t7))
        end
        if m == 1 && h == 2                                    # the engine of fast_opt.jl, for reference
            ws = Fast.Work(P, fp, K)
            combboot_fast(P, fp, K, a2, bints; ws)
            t0 = best(() -> combboot_fast(P, fp, K, aints, bints; ws))
            @printf("        (fast_opt.jl: comb bootstrap %5.2f ms/error", 1e3t0 / P.B)
            for t in TH
                combboot_fast(P, fp, K, a2, bints; ws, nthreads = t)
                t0T = best(() -> combboot_fast(P, fp, K, aints, bints; ws, nthreads = t))
                @printf("; %d threads %5.2f", t, 1e3t0T / P.B)
            end
            println(")")
        end
    end
end
