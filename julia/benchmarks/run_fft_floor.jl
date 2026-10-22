# Floor of the error of the FFT (Section 6 and Appendix J of the paper): output noise of the error bootstrap (FFT backend,
# eb_refresh, B = 1) at N = 2048, p̂ = 2^10, n = NN (default 996), m = 2, for the gadgets GADS of the rotation, measured
# on NS refreshes (default 40), against the model of the cost model (2^-85 B^2 l per external product for the FFT).
# Run from the julia folder:   GADS=6:7,5:9,4:11,3:15,2:22 julia --project -t 1 benchmarks/run_fft_floor.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl")); using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl")); using .Fast
using Random, Printf, Statistics
const N = 2048; const σ = 2.0^-50.3; const M = 2
Δ(i, phat) = UInt64(i) << (64 - trailing_zeros(phat))
l2(x) = 0.5 * log2(x)
function var_rot(n, m, b, ℓ; fft = true)
    g = n / m; key = 2ℓ * N * (2.0^(2b) + 2) / 12 * σ^2; rnd = (1 + N / 2) * 2.0^(-2ℓ * b) / 12
    return ((2^m - 1) * g * key + g * (1 - 2.0^-m) * rnd) * 2 + (fft ? g * 2.0^-85 * 2.0^(2b) * ℓ : 0.0)
end
phat = 1024; n = parse(Int, get(ENV, "NN", "996")); NS = parse(Int, get(ENV, "NS", "40"))
for (b, ℓ) in [parse.(Int, split(x, ":")) for x in split(get(ENV, "GADS", "6:7,5:9,4:11,3:15,2:22"), ",")]
    t0 = time(); rng = Xoshiro(17 + b); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = 2N, phat = phat, n = n, Gbr = Gadget(b, ℓ), Gks = Gadget(14, 2), σ = σ, h = 1)
    K = fast_keygen_grouped(P, r, fp, rng; m = M); wg = WorkG(P, fp, K; m = M, nthreads = 1)
    sp = K.sk.s; pool = zero_pool(K.s, 128, σ128(n), rng)
    e = Float64[]
    for _ in 1:NS
        i = rand(rng, 0:phat-1)
        c = eb_refresh(P, fp, K, [lwe_encrypt(K.s, Δ(i, phat), 2.0^-20, rng)]; m = M, wg, pool, Vstar = Inf, rng, comb = false)[1]
        push!(e, torus_to_float(lwe_phase(c, sp) - Δ(i, phat)))
    end
    @printf("(2^%d,%d) n=%d: out %7.2f measured, %7.2f predicted (%7.2f without FFT)  [%d samples, %.0f s]\n", b, ℓ, n, l2(mean(abs2, e)),
            l2(var_rot(n, M, b, ℓ)), l2(var_rot(n, M, b, ℓ; fft = false)), NS, time() - t0); flush(stdout)
    K = nothing; wg = nothing; GC.gc()
end
