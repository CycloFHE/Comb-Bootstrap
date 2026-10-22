# Time of each phase of one comb bootstrap iteration, against B CMux of the separate rotations (FFT backend).
# Run from the julia folder of the repository:  julia benchmarks/profile_fast.jl   (P=128 for p = 128, L=3 for the ℓ = 3 gadgets;
# julia --project -t 4 ...: also the iteration with 4 threads)
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf

const PHASES = ["division", "FFT A0 B0, digits, c3", "baby + public", "giant step(s)", "inv C_L", "CMux digits",
                "external product", "inv + add"]

"µs per iteration (fixed costs removed) and phase times seen by thread 1."
function per_iteration(P, fp, K, ws, rng; nthreads = 1, reps = 10)
    aints = rand(rng, 0:P.p-1, P.B, P.n); a0 = zeros(Int, P.B, 0); bints = rand(rng, 0:P.p-1, P.B)
    tim = zeros(8)
    run(a, t) = combboot_fast(P, fp, K, a, bints; ws, nthreads, tim = t)
    run(aints, tim); run(a0, tim); run(a0, nothing)
    t = minimum(@elapsed(run(aints, tim)) for _ in 1:reps)
    t0 = minimum(@elapsed(run(a0, nothing)) for _ in 1:reps)
    fill!(tim, 0); run(aints, tim)
    return 1e6 * (t - t0) / P.n, tim ./ (1e3 * P.n)
end

function main()
    rng = Xoshiro(3)
    N = 2048
    p = parse(Int, get(ENV, "P", "256"))
    l3 = get(ENV, "L", "1") == "3"
    Gbr = l3 ? Gadget(10, 3) : Gadget(23, 1)
    Gks = l3 ? (2N ÷ p <= 16 ? Gadget(6, 6) : Gadget(5, 8)) : Gadget(14, 2)
    P = CBParams(N = N, p = p, phat = 4, n = 64, Gbr = Gbr, Gks = Gks, σ = 2.0^-45)
    r = Ring(N); fp = FPlan(N)
    K = fast_keygen(P, r, fp, rng)
    T = Threads.nthreads()
    ws = Fast.Work(P, fp, K; nthreads = max(T, 4))
    Y = zeros(ComplexF64, fp.H); R = 20000
    fp.pf * Y; t_fft = 1e6 * (@elapsed for _ in 1:R; fp.pf * Y; end) / R
    C = GLWE(rand(rng, UInt64, N), rand(rng, UInt64, N))
    Fast.cmux_rot!(C, 77, 1, P, ws, fp)
    t_rot = 1e6 * (@elapsed for _ in 1:R÷10; Fast.cmux_rot!(C, 77, 1, P, ws, fp); end) / (R ÷ 10)
    @printf("p = %d, B = %d, (g,h) = (%d,%d), ℓ = %d, ℓ_aut = %d; one FFT of size N/2: %.2f µs\n",
            p, P.B, P.g, P.h, Gbr.ell, Gks.ell, t_fft)
    t1, tim = per_iteration(P, fp, K, ws, rng)
    @printf("  1 thread: %.1f µs per iteration\n", t1)
    for (nm, x) in zip(PHASES, tim)
        @printf("    %-20s %6.1f µs\n", nm, x)
    end
    @printf("  one CMux with rotation: %.1f µs; B of them: %.1f µs; ratio %.2f\n", t_rot, P.B * t_rot, P.B * t_rot / t1)
    if T > 1
        tT, timT = per_iteration(P, fp, K, ws, rng; nthreads = T)
        @printf("  %d threads, 1 batch: %.1f µs per iteration (×%.1f); phases seen by thread 1:", T, tT, t1 / tT)
        for (nm, x) in zip(PHASES, timT)
            @printf(" %s %.1f;", nm, x)
        end
        println()
    end
end
main()
