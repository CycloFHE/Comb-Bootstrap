# Comb bootstrap at p̂ = 256: refresh of 8-bit integers by the comb bootstrap, against the error bootstrap
# one ciphertext at a time (configuration k = 4, refresh, of run_curves.jl), in the same session. One thread.
#  N = 2048, binary keys, coordinate-wise rounding (m = 1), h = 1 (no giant steps), polynomial key switch (d = 16),
#  ν = 1000 additions, failure 2^-128:  (4p̂)^2 (ν σ_out^2 + σ_KS^2) + 16 V*/p^2 ≤ 1/U^2,  U = 13.11.
#  Configurations: the cheapest of the cost model (noise of Proposition 1 of the paper, h = 1,
#  error of the FFT 2^-85 B^2 ℓ per external product, margin of 0.2 bit on the standard deviation, chain of the key
#  switch of the cost model), with the gadget of the automorphism keys as a parameter:
#    p = 512 (B = 8):  RGSW (2^14,2), aut (2^8,4), n = 826        (model margin 0.20 bit)
#    p = 512 (B = 8):  RGSW (2^11,3), aut (2^8,4), n = 816        (model margin 0.22 bit)
#    p = 256 (B = 16): RGSW (2^11,3), aut (2^7,5), n = 896        (model margin 0.19 bit: at p = 256, the rounding
#                      term 16 V*/p^2 takes most of the budget, and no n reaches 0.2 bit)
#  With the automorphism gadget (2^14,2) of run_fast.jl, the noise of the 15 to 56 key switchings per coordinate
#  exceeds the budget at p̂ = 256 for every p.
#  PART=N: measured noise of the comb bootstrap (NE errors, default 2000) and of the key switch (400), margin in bits;
#  PART=T: native times per error, minimum over REPS runs (default 5): comb bootstrap of B errors / B + key switch
#          N -> n (one per ciphertext), and the error bootstrap alone (k = 4 of run_curves.jl: n = 816, (2^14,2), m = 2).
# Run from the julia folder of the repository:   PART=NT julia --project -t 1 benchmarks/run_comb_phat256.jl
#   (CFGS=1,3: some configurations only; QUICK=1: n = 64, to test the script, numbers meaningless)
const MYPART = get(ENV, "PART", "NT")
ENV["PART"] = "-"                                   # run_curves.jl: definitions only (N, σ, U128, NU, CFG, timers)
include(joinpath(@__DIR__, "run_curves.jl"))
ENV["PART"] = MYPART

const QUICK = get(ENV, "QUICK", "0") == "1"
const NE = parse(Int, get(ENV, "NE", "2000"))
const PH = 256
const STAIR_816 = [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10),
                   (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18),
                   (832, 1, 17)]
# (p, RGSW gadget, automorphism gadget, n, 16 V*/p^2, chain of the key switch N -> n)
const MBCFG = [
    (512, Gadget(14, 2), Gadget(8, 4), 826, 9.7432e-04, vcat(STAIR_816, [(826, 1, 17)])),
    (512, Gadget(11, 3), Gadget(8, 4), 816, 9.6207e-04, vcat(STAIR_816, [(816, 1, 17)])),
    (256, Gadget(11, 3), Gadget(7, 5), 896, 4.2407e-03,
     [(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7), (1024, 2, 11),
      (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19)]),
]
const CFGS = parse.(Int, split(get(ENV, "CFGS", join(1:length(MBCFG), ",")), ","))

cbparams(p, n, Gbr, Gks) = CBParams(N = N, p = p, phat = PH, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = 1)
"Predicted output variance of the comb bootstrap (Proposition 1, h = 1) with the error of the FFT of the external products."
var_mb(P, n1) = var_combboot(P, n1) + P.n * 2.0^-85 * 2.0^(2P.Gbr.logB) * P.Gbr.ell

println("Comb bootstrap at p̂ = 256, N = 2048, m = 1, h = 1", QUICK ? "   (QUICK: n = 64, test only)" : ""); flush(stdout)
csv = String[]
for c in CFGS
    p, Gbr, Gks, n0, vterm, chain = MBCFG[c]
    n = QUICK ? 64 : n0
    rng = Xoshiro(100c + n); r = Ring(N); fp = FPlan(N)
    P = cbparams(p, n, Gbr, Gks)
    K = fast_keygen(P, r, fp, rng); ws = Fast.Work(P, fp, K; nthreads = 1)
    @printf("\nconfiguration %d: p = %d, B = %d, n = %d, RGSW %s, aut %s, (g,h) = (%d,%d)\n",
            c, p, P.B, n, gstr(Gbr), gstr(Gks), P.g, P.h)
    if occursin('N', MYPART)
        tstart = time(); e = Float64[]; emax = 0.0
        for _ in 1:cld(NE, P.B)
            aints, bints, E = random_inputs(P, K.s, rng)
            ek = output_errors(P, combboot_fast(P, fp, K, aints, bints; ws), K.sk.s, E)
            append!(e, ek); emax = max(emax, maximum(abs, ek))
        end
        vo = mean(abs2, e); vks, vksp = ks_var(K.sk.s, chain, rng)
        @printf("  noise (log2 σ, measured / predicted): out %7.2f / %7.2f   KS %7.2f / %7.2f   max |err|·R %.1e   (%d errors)\n",
                l2(vo), l2(var_mb(P, sum(K.s))), l2(vks), l2(vksp), emax * R_of(P), length(e))
        @printf("  margin of the condition: %5.2f bits measured, %5.2f bits predicted   (%.0f s)\n",
                margin(PH, NU * vo, vks, vterm), margin(PH, NU * var_mb(P, n ÷ 2), vksp, vterm), time() - tstart)
    end
    if occursin('T', MYPART)
        aints, bints, _ = random_inputs(P, K.s, rng)
        combboot_fast(P, fp, K, aints, bints; ws)                                        # warm-up
        tr = 1e3 * minimum(@elapsed(combboot_fast(P, fp, K, aints, bints; ws)) for _ in 1:REPS) / P.B
        tk = ks_time(chain)
        @printf("  time per error: rotation %6.2f (%d errors in %.1f) + key switch %6.3f = %6.2f ms\n",
                tr, P.B, tr * P.B, tk, tr + tk)
        push!(csv, @sprintf("cb,%d,p%d_B%d_n%d,%.4f", c, p, P.B, n, tr + tk))
    end
    flush(stdout); K = nothing; ws = nothing; GC.gc()
end

if occursin('T', MYPART)
    n, G, _, chain = CFG[4].R
    n = QUICK ? 64 : n
    tr = rotation_time(n, G); tk = ks_time(chain)
    @printf("\nerror bootstrap alone (run_curves.jl, k = 4): n = %d, %s, m = 2: rotation %6.2f + key switch %6.3f = %6.2f ms\n",
            n, gstr(G), tr, tk, tr + tk)
    push!(csv, @sprintf("eb,4,refresh,%.4f", tr + tk))
    println("CSV (scheme,configuration,parameters,ms per error)"); foreach(println, csv)
end
