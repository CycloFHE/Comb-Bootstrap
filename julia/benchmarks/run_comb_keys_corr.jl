# Table 3, keys column, and the independence of the B outputs of one comb bootstrap (Appendix J).
# Configurations of Table 3 (benchmarks/comb_sizes.jl), N = 2048, h = 1, mask grouped by m (M, default 1) in the rotation.
#  - size in MB (Base.summarysize, Fourier domain) of the bootstrapping keys, of the automorphism keys (baby and
#    giant steps), of the stair key-switching key and of the pool of 128 public encryptions of zero;
#  - NR comb bootstraps (default 300) on random inputs: pairwise correlations of the B output errors (mean, largest
#    |.|, adjacent columns) and var(e_k + e_l) / (var e_k + var e_l) over the pairs k < l.
#  M=2: n odd (16 bits: 1157) is raised to the next multiple of m, as in run_comb_sizes.jl.
# Run from the julia folder of the repository:
#     julia --project -t 1 benchmarks/run_comb_keys_corr.jl | tee ../results/julia/comb_keys_corr_vm.txt
#   (BITS=8: some sizes only; NR=1000: more runs; WHAT=K: key sizes only; M=2: grouping by 2)
ENV["PART"] = "-"                                   # run_curves.jl: definitions only (N, σ, ...)
include(joinpath(@__DIR__, "run_curves.jl"))
include(joinpath(@__DIR__, "comb_sizes.jl"))         # SIZES
const NR = parse(Int, get(ENV, "NR", "300"))
const MG = parse(Int, get(ENV, "M", "1"))
MB(x) = Base.summarysize(x) / 2^20
for bits in parse.(Int, split(get(ENV, "BITS", "2,4,8,12,16"), ","))
    phat = 2^bits; p, Gbr, Gks, n, vterm, chain = SIZES[bits]
    m = MG
    if n % m != 0
        n += m - n % m; chain = [chain[1:end-1]; (n, chain[end][2], chain[end][3])]
    end
    rng = Xoshiro(900 + bits); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = 1)
    K = fast_keygen_grouped(P, r, fp, rng; m)
    ksk = stair_keygen(K.sk.s, N, chain, rng); pool = zero_pool(K.s, 128, σ128(n), rng)
    @printf("%2d bits (n = %d, m = %d): keys (MB): bootstrapping %.1f, automorphisms %.2f, key switch %.2f, pool %.2f, total %.1f\n",
            bits, n, m, MB(K.bsk), MB(K.baby) + MB(K.giant), MB(ksk), MB(pool), MB(K.bsk) + MB(K.baby) + MB(K.giant) + MB(ksk) + MB(pool))
    flush(stdout)
    occursin("C", get(ENV, "WHAT", "KC")) || continue
    wg = WorkG(P, fp, K; m, nthreads = 1)
    E = zeros(NR, P.B)
    for t in 1:NR
        aints, bints, Ein = random_inputs(P, K.s, rng)
        E[t, :] = output_errors(P, combboot_grouped(P, fp, K, aints, bints; m, wg), K.sk.s, Ein)
    end
    C = cor(E); off = [C[k, l] for k in 1:P.B for l in 1:P.B if k < l]; adj = [C[k, k+1] for k in 1:P.B-1]
    v = vec(var(E; dims = 1)); rat = [var(E[:, k] .+ E[:, l]) / (v[k] + v[l]) for k in 1:P.B for l in 1:P.B if k < l]
    @printf("   outputs of one comb bootstrap (%d runs): correlation mean %.3f, largest |.| %.3f (adjacent columns: mean %.3f); var(e_k + e_l)/(var e_k + var e_l): mean %.3f, range %.3f to %.3f\n",
            NR, mean(off), maximum(abs, off), mean(adj), mean(rat), minimum(rat), maximum(rat))
    flush(stdout); K = nothing; wg = nothing; GC.gc()
end
