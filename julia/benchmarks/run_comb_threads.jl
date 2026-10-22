# CombBoot with several threads at 2, 4, 8, 12 and 16 bits (configurations of Table 3, benchmarks/comb_sizes.jl):
# latency of one call and time per ciphertext. N = 2048, h = 1, mask grouped by m (M, default 1) in the rotation,
# engine of src/grouped.jl.
#  One call refreshes B LWE ciphertexts under s (dimension n, at the noise of ν = 1000 fresh outputs), as in Table 3:
#  rounding with recentering and selection (src/rounding.jl, pool of 128 encryptions of zero, threshold V*), comb
#  bootstrap, and the B key switches N -> n (polynomial, d = 16; in a refresh they precede the rounding, the cost is
#  the same).
#  For t threads (TH, default 1,4,8), 1 batch (latency): B ciphertexts; the t threads share the B roundings, cooperate
#  inside each iteration of the blind rotation (SPMD, spin barriers, src/grouped.jl), then share the B key switches.
#  Latency: time of the call; time per ciphertext: latency / number of ciphertexts. Minimum over REPS runs (default 5);
#  the generators of the selection are reseeded at each run, so that all the runs do the same trials.
#  Check: largest |E_k| given to the rotation against the window p/4, largest error of the outputs of the rotation and
#  largest error added by the key switch, in units of the margin 1/(4p̂).
#  PROF=1: also the times of the 7 phases of an iteration of the rotation (µs per iteration, thread 1, waiting
#  included), 1 batch: 1 FFTs of A0, B0 and digits | 2 baby steps | 3 giant steps | 4 inverse FFTs of C_L |
#  5 digits of C_L - C | 6 external product | 7 C += inverse FFT, division, tables of the next iteration.
#  M=2: mask grouped by m = 2 in the rotation: one iteration per two coordinates, with the 2^m - 1 = 3 patterns, hence
#  twice the work between two barriers (keys of the rotation 1.5 times larger, noise not larger); n odd (16 bits: 1157)
#  is raised to the next multiple of m (last step of the key switch and V* recomputed), as in run_comb_sizes.jl.
#  The threads of one batch synchronize with spin barriers: take at most the number of performance cores (M3 Max: 10 or 12).
# Run from the julia folder of the repository (julia started with at least max(TH) threads):
#     julia --project -t 8 benchmarks/run_comb_threads.jl | tee ../results/julia/comb_threads_mac2.txt
#   (TH=1,4,6,8: other numbers of threads; BITS=8: some sizes only; PROF=1: phases; M=2: grouping by 2;
#    QUICK=1: n = 64, to test the script)
ENV["PART"] = "-"                                   # run_curves.jl: definitions only (N, σ, NU, REPS, gstr, timers)
include(joinpath(@__DIR__, "run_curves.jl"))
include(joinpath(@__DIR__, "comb_sizes.jl"))         # SIZES, gs, var_mbs

const QUICK = get(ENV, "QUICK", "0") == "1"
const PROF = get(ENV, "PROF", "0") == "1"
const MG = parse(Int, get(ENV, "M", "1"))
const BITS = parse.(Int, split(get(ENV, "BITS", "2,4,8,12,16"), ","))
const TH = sort(unique(vcat(1, filter(t -> t <= Threads.nthreads(), parse.(Int, split(get(ENV, "TH", "1,4,8"), ","))))))
const TMAX = maximum(TH)

"Copy of the key of the key switch with its own FFT plan (the plan holds a buffer): one per thread."
kcopy(K::StairKey) = StairKey(K.d, K.T, K.n, K.steps, FPlan(K.d))

"Key switches of cts into outs, shared between t threads (Ks[tid]: key of thread tid)."
function ks_par!(outs, Ks, cts, t)
    if t == 1
        for i in eachindex(cts)
            outs[i] = stair_ks(Ks[1], cts[i])
        end
    else
        Threads.@threads :static for tid in 1:t
            for i in tid:t:length(cts)
                outs[i] = stair_ks(Ks[tid], cts[i])
            end
        end
    end
    return outs
end

"Rounding with selection of the ciphertexts ks of a batch (generator g): Rounded into rs, inputs of the rotation
(mask and body modulo p) into A and b."
function round_batch!(rs, A, b, cts, R, pool, Vs, p, ks, g)
    for k in ks
        r = round_select(cts[k], R, pool; Vstar = Vs, rng = g)
        rs[k] = r
        @inbounds for u in eachindex(r.abar)
            A[k, u] = mod(r.abar[u], p)
        end
        b[k] = mod(r.bbar, p)
    end
    return nothing
end

"The same, shared between t threads (generators gs reseeded: the same trials at each run)."
function round_par!(rs, A, b, cts, R, pool, Vs, p, gs, t)
    if t == 1
        Random.seed!(gs[1], 1); round_batch!(rs, A, b, cts, R, pool, Vs, p, eachindex(cts), gs[1])
    else
        Threads.@threads :static for tid in 1:t
            Random.seed!(gs[tid], tid); round_batch!(rs, A, b, cts, R, pool, Vs, p, tid:t:length(cts), gs[tid])
        end
    end
    return nothing
end

"Largest error added by the key switch, in units of the margin 1/(4p̂) (phase under the first coordinates of s' against
the phase under s')."
ks_err(P, ko, o, sp) = maximum(abs(torus_to_float(lwe_phase(ko[i], sp[1:length(ko[i].a)]) - lwe_phase(o[i], sp))) for i in eachindex(o)) * 4P.phat

bestof(f) = minimum(@elapsed(f()) for _ in 1:REPS)

println("CombBoot with t threads, N = 2048, m = $MG, h = 1, configurations of Table 3, rounding with selection included, threads ",
        join(TH, ", "), " (julia -t $(Threads.nthreads()))", QUICK ? "   (QUICK: n = 64, test only)" : ""); flush(stdout)
csv = String[]
const tag = MG == 1 ? "cbtr" : "cbtr_m$MG"
for bits in BITS
    phat = 2^bits; p, Gbr, Gks, n0, vterm, chain = SIZES[bits]
    m = MG
    if n0 % m != 0                                    # n raised to a multiple of m: last step of the chain and V*
        n0 += m - n0 % m; chain = [chain[1:end-1]; (n0, chain[end][2], chain[end][3])]
        vterm = 16 * vstar_threshold(n0) / p^2
    end
    n = QUICK ? 64 : n0
    rng = Xoshiro(500bits + n); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = 1)
    K = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, K; m, nthreads = TMAX)
    K1 = stair_keygen(K.sk.s, N, chain, rng); Ks = [K1; [kcopy(K1) for _ in 2:TMAX]]
    B = P.B; R = R_of(P); Vs = vterm * p^2 / 16; pool = zero_pool(K.s, 128, σ128(n0), rng)
    vin = NU * var_mbs(P, nactive(K.s, m), m)                         # ν fresh outputs (Proposition 1, grouping m)
    @printf("\n%2d bits: p = %d, B = %d, n = %d, RGSW %s, aut %s, V* = %.2f\n", bits, p, B, n, gstr(Gbr), gs(Gks), Vs)
    # B ciphertexts LWE_s(μ), their roundings and the inputs of the rotation
    mus = [rand(rng, 0:phat-1, B) for _ in 1:1]
    cts = [[lwe_encrypt(K.s, Δ(i, phat), sqrt(vin), rng) for i in mus[j]] for j in 1:1]
    rs = [Vector{Rounded}(undef, B) for _ in 1:1]
    As = [zeros(Int, B, n) for _ in 1:1]; bs = [zeros(Int, B) for _ in 1:1]
    gens = [Xoshiro(0) for _ in 1:TMAX]
    Eof(j) = rounded_errors(rs[j], K.s, [p * i for i in mus[j]], R)
    t1 = 0.0
    for t in TH
        # one batch of B ciphertexts: t threads share the roundings, cooperate inside the rotation, share the key switches
        rnd1() = round_par!(rs[1], As[1], bs[1], cts[1], R, pool, Vs, p, gens, t)
        rot1() = t == 1 ? combboot_grouped(P, fp, K, As[1], bs[1]; m, wg) :
                          combboot_grouped(P, fp, K, As[1], bs[1]; m, wg, nthreads = t)
        ko = Vector{LWE}(undef, B)
        call1() = (rnd1(); ks_par!(ko, Ks, rot1(), t))
        rnd1(); o = rot1(); ks_par!(ko, Ks, o, t)                                       # warm-up
        E1 = Eof(1)
        ee = maximum(abs, E1) / (p ÷ 4); er = maximum(abs, output_errors(P, o, K.sk.s, E1)) * 4phat; ek = ks_err(P, ko, o, K.sk.s)
        tn = bestof(rnd1); tr = bestof(rot1); tk = bestof(() -> ks_par!(ko, Ks, o, t)); tl = bestof(call1)
        t == 1 && (t1 = tl / B)
        @printf("  %2d thread%s, 1 batch  : %4d ciphertexts in %8.2f ms (rounding %5.2f + rotation %8.2f + key switches %6.2f): %6.3f ms per ciphertext (×%.1f)   [|E|/window %.2f, error/margin: rotation %.1e, KS %.1e]\n",
                t, t == 1 ? " " : "s", B, 1e3tl, 1e3tn, 1e3tr, 1e3tk, 1e3tl / B, t1 / (tl / B), ee, er, ek)
        push!(csv, @sprintf("%s,%d,%d,1batch,%d,%.3f,%.4f", tag, bits, t, B, 1e3tl, 1e3tl / B))
        if PROF
            tim = zeros(UInt64, 7)
            t == 1 ? combboot_grouped(P, fp, K, As[1], bs[1]; m, wg, tim) :
                     combboot_grouped(P, fp, K, As[1], bs[1]; m, wg, nthreads = t, tim)
            @printf("      phases (µs per iteration of m coordinates): %s;  total %.1f\n",
                    join([@sprintf("%.2f", x / 1e3 / (n ÷ m)) for x in tim], " | "), sum(tim) / 1e3 / (n ÷ m))
        end
    end
    flush(stdout); K = nothing; wg = nothing; Ks = nothing; K1 = nothing; GC.gc()
end
println("CSV (scheme,bits,threads,mode,ciphertexts,latency in ms,ms per ciphertext; cbtr: rounding with selection included, m = 1; cbtr_m2: m = 2)")
foreach(println, csv)
