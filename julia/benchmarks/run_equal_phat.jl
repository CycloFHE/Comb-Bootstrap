# Comparison at equal message modulus: measured times, one thread, full rotations (all n coordinates).
#  part A: one blind rotation of one ciphertext at N = 2048 (mask grouped by m, bootstrapping gadget (2^b, l)): error
#          bootstrap with B = 1 (p = 2N), and PBS of TFHE on blocks of 4 bits;
#  part B: one blind rotation of TFHE in a single ring, N = 2^15 and 2^16 (keys generated directly in the split form of
#          the workspace: about 2.8 GB at N = 2^15 and 5.6 GB at N = 2^16);
#  part P: product of two ciphertexts at N = 2048 (conversion and trace of both inputs, tensor product, relinearization).
# The size of the bootstrapping keys (Fourier domain) is measured with Base.summarysize.
# Run from the julia folder of the repository:   julia benchmarks/run_equal_phat.jl      (PART=A, B or P: one part only; REPS=5; NB=16:
# part B at N = 2^16 only)
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
include(joinpath(@__DIR__, "..", "src", "fast.jl"))
using .Fast
using Random, Printf

const REPS = parse(Int, get(ENV, "REPS", "5"))
const PART = get(ENV, "PART", "ABP")
const NB = parse.(Int, split(get(ENV, "NB", "15,16"), ","))
const σ128 = Dict(2048 => 2.0^-50.3, 2^15 => 2.0^-62, 2^16 => 2.0^-62)

"Workspace with the bootstrapping keys of the grouping by m, generated directly in split form (one key at a time)."
function lean_workspace(P::CBParams, r::Ring, fp::FPlan, m::Int, rng)
    sprime = Int64.(rand(rng, 0:1, P.N)); sk = SecretKey(r, sprime)
    M = 2 * P.N; γ = 1 + P.p
    baby = [Fast.fkskey(r, fp, sk, powermod(γ, j, M), P.Gks, P.σ, rng) for j in 1:P.g-1]
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

"Time of one rotation of one ciphertext (all n coordinates, grouped by m) and size of the keys in GB."
function rotation_time(N, n, m, b, l)
    rng = Xoshiro(7); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = 2N, phat = 4, n = n, Gbr = Gadget(b, l), Gks = Gadget(14, 2), σ = σ128[N], h = 1)  # B = 1
    wg = lean_workspace(P, r, fp, m, rng)
    mask = 2N - 1; v = Fast.vstar(P); a = rand(rng, 0:2N-1, n)
    function run1()
        C = Fast.trivial(Fast.mulX(v, 0))
        if m == 1
            for u in 1:n
                Fast.cmux_rot!(C, a[u] & mask, u, P, wg.ws, fp)
            end
        else
            for t in 1:n÷m
                es = ntuple(i -> i <= m ? a[(t-1)*m+i] & mask : 0, 4)
                Fast.combine_key!(wg, es, t, P, fp); Fast._mbit!(C, P, wg, fp)
            end
        end
        return C
    end
    run1()
    t = minimum(@elapsed(run1()) for _ in 1:REPS)
    return t, Base.summarysize(wg.ws.bsk) / 2.0^30
end

if occursin('A', PART)
    println("A. one rotation of one ciphertext, N = 2048")
    for (lab, n, m, b, l) in (("error bootstrap, p̂ = 256, refresh", 800, 2, 15, 2),
                              ("error bootstrap, p̂ = 256, before a product", 830, 2, 9, 4),
                              ("error bootstrap, p̂ = 64, refresh and product", 780, 2, 12, 3),
                              ("TFHE PBS, 4-bit blocks (p̂ = 16)", 704, 1, 23, 1))
        t, gb = rotation_time(2048, n, m, b, l)
        @printf("  %-48s n = %4d, m = %d, (2^%d,%d): %7.2f ms   keys %.3f GB\n", lab, n, m, b, l, 1e3t, gb)
        GC.gc()
    end
end
if occursin('B', PART)
    println("B. TFHE in a single ring, one rotation (all n coordinates): p̂ = 256 (N = 2^15), 512 (N = 2^16)")
    for (N, n) in ((2^15, 848), (2^16, 896))      # n with the stair key switch of benchmarks/run_stair_ks.jl
        trailing_zeros(N) in NB || continue
        t, gb = rotation_time(N, n, 2, 16, 2)
        @printf("  N = %5d, n = %d, m = 2, (2^16,2): %7.1f ms   keys %.2f GB\n", N, n, 1e3t, gb)
        GC.gc()
    end
end
if occursin('P', PART)
    println("P. product of two ciphertexts, N = 2048 (FFT backend; relinearization (2^10,4))")
    rng = Xoshiro(9); N = 2048; r = Ring(N); fp = FPlan(N)
    sp = Int64.(rand(rng, 0:1, N)); sk = SecretKey(r, sp)
    for (phat, Gtr) in ((256, Gadget(5, 10)), (64, Gadget(8, 6)))
        raw = product_keys_raw(r, sk, Gtr, Gadget(10, 4), σ128[N], rng)
        FK = FProductKeys(fp, raw); ws = PWork(fp, max(Gtr.ell, 4))
        Δ(i) = UInt64(i) << (64 - trailing_zeros(phat))
        c1 = lwe_encrypt(sp, Δ(3), 2.0^-27, rng); c2 = lwe_encrypt(sp, Δ(5), 2.0^-27, rng)
        tm(f) = (f(); minimum(@elapsed(f()) for _ in 1:20REPS) * 1e3)
        t1 = flwe_to_rlwe_clean(ws, FK, c1, fp); t2 = flwe_to_rlwe_clean(ws, FK, c2, fp)
        D0, D1, D2 = ftensor(ws, t1, t2, phat, fp)
        @printf("  p̂ = %3d, trace (2^%d,%d): trace of one input %.3f ms, tensor product %.3f ms, relinearization %.3f ms, product %.3f ms   keys %.3f GB\n",
                phat, Gtr.logB, Gtr.ell, tm(() -> flwe_to_rlwe_clean(ws, FK, c1, fp)), tm(() -> ftensor(ws, t1, t2, phat, fp)),
                tm(() -> frelin(ws, D0, D1, D2, FK, fp)), tm(() -> flwe_product(ws, FK, c1, c2, phat, fp)), Base.summarysize(FK) / 2.0^30)
    end
end
