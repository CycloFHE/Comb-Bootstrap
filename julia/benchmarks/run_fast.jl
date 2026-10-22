# Comb bootstrap, case (B), FFT backend (FFTW.jl): timings and noise for two gadget choices.
# Separate rotations: "batched" (the B rotations advance together, each bootstrapping key read once per
# iteration, as the comb bootstrap does) and "one by one" (each rotation reads all the keys).
# Run from the julia folder of the repository:   julia benchmarks/run_fast.jl     (QUICK=1 for n = 64, P=128 for p = 128,
# N=4096 for the ring of degree 4096, with σ = 2^-62)
# julia --project -t 4 benchmarks/run_fast.jl: also 4 threads, for one batch of B errors (latency: threads inside each
# iteration of the comb bootstrap / B rotations shared).
# Multithreaded FFTs (not useful at this size):  FFT_THREADS=4 julia --project -t 4 benchmarks/run_fast.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf

const QUICK = get(ENV, "QUICK", "0") == "1"
rng = Xoshiro(11)
N = parse(Int, get(ENV, "N", "2048")); phat = 4
p = parse(Int, get(ENV, "P", "256"))     # B = 2N/p columns
n = QUICK ? 64 : 630
r = Ring(N); fp = FPlan(N)
meas(e) = sum(abs2, e) / length(e)

# automorphism gadget: precision and base adapted to B (Section 5.5)
Gks1 = 2N ÷ p <= 16 ? Gadget(6, 6) : Gadget(5, 8)
configs = N == 2048 ? [
    ("RGSW (2^10,3), aut (2^$(Gks1.logB),$(Gks1.ell)), σ = 2^-40  ", Gadget(10, 3), Gks1, 2.0^-40),
    ("RGSW (2^23,1), aut (2^14,2), σ = 2^-50.3", Gadget(23, 1), Gadget(14, 2), 2.0^-50.3),
] : [         # N = 4096: σ ≈ 2^-62 at 128 bits (to be checked with the estimator); one 28-bit digit for the
              # automorphism keys would suffice for the noise of the keys, but not for the precision of the FFT
    ("RGSW (2^23,1), aut (2^14,2), σ = 2^-62", Gadget(23, 1), Gadget(14, 2), 2.0^-62),
]

for (name, Gbr, Gks, σ) in configs
    P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ)
    K = fast_keygen(P, r, fp, rng)
    ws = Fast.Work(P, fp, K)                                  # workspace and split keys, built once
    aints, bints, E = random_inputs(P, K.s, rng)
    n1 = sum(K.s)
    combboot_fast(P, fp, K, aints[:, 1:2], bints; ws)            # warm-up
    separate_rotations_fast(P, fp, K, aints[:, 1:2], bints; ws)
    separate_rotations_fast(P, fp, K, aints[:, 1:2], bints; ws, batched = false)
    best(f) = minimum(@elapsed(f()) for _ in 1:3)                  # best of 3 runs
    t_mb = best(() -> combboot_fast(P, fp, K, aints, bints; ws))
    t_bat = best(() -> separate_rotations_fast(P, fp, K, aints, bints; ws))
    t_seq = best(() -> separate_rotations_fast(P, fp, K, aints, bints; ws, batched = false))
    outs = combboot_fast(P, fp, K, aints, bints; ws); outs2 = separate_rotations_fast(P, fp, K, aints, bints; ws)
    e1 = output_errors(P, outs, K.sk.s, E); e2 = output_errors(P, outs2, K.sk.s, E)
    println("\n", name, "   (N = $N, p = $p, B = $(P.B), n = $n, (g,h) = ($(P.g),$(P.h)), FFT threads = $(fp.nthreads))")
    @printf("  comb bootstrap           : %6.3f s for %d errors = %5.2f ms each; log2 std %.2f (exact model %.2f)\n",
            t_mb, P.B, 1e3t_mb / P.B, 0.5log2(meas(e1)), 0.5log2(var_combboot(P, n1)))
    @printf("  separate, batched    : %6.3f s for %d errors = %5.2f ms each; log2 std %.2f (exact model %.2f)\n",
            t_bat, P.B, 1e3t_bat / P.B, 0.5log2(meas(e2)), 0.5log2(var_separate(P, n1)))
    @printf("  separate, one by one : %6.3f s for %d errors = %5.2f ms each\n", t_seq, P.B, 1e3t_seq / P.B)
    @printf("  ratio %.2f (batched), %.2f (one by one); max |err|·R = %.1e\n", t_bat / t_mb, t_seq / t_mb,
            max(maximum(abs.(e1)), maximum(abs.(e2))) * R_of(P))
    T = Threads.nthreads()
    T == 1 && continue
    # T threads, one batch of B errors (latency): threads inside each iteration / B rotations shared
    t_mbT = best(() -> combboot_fast(P, fp, K, aints, bints; ws, nthreads = T))
    t_batT = best(() -> separate_rotations_fast(P, fp, K, aints, bints; ws, nthreads = T))
    @printf("  %d threads, 1 batch   : comb bootstrap %5.2f ms/error (×%.1f), separate %5.2f ms/error (×%.1f); ratio %.2f\n",
            T, 1e3t_mbT / P.B, t_mb / t_mbT, 1e3t_batT / P.B, t_bat / t_batT, t_batT / t_mbT)
end
