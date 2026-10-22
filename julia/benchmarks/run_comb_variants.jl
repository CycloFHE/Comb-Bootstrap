# Comb bootstrap at p̂ = 256: variants of run_comb_phat256.jl with h = 2 (giant steps) and m = 2 (mask grouped by 2),
# against the error bootstrap alone (k = 4 of run_curves.jl), in the same session. One thread.
#  Every variant runs on the engine of src/grouped.jl, which also gives the time of each phase of an iteration
#  (µs per coordinate for the whole batch); for m = 1, the time of the engine of src/fast_opt.jl is given too.
#  Rounding coordinate by coordinate (Section 4.3): the rounding term 16 V*/p^2 is that of m = 1 for every m.
#  Configurations (cost model, margin of 0.2 bit on the standard deviation):
#    1-2: p = 256 (B = 16), RGSW (2^11,3), aut (2^7,5), n = 896, m = 1, h = 1 and 2   (configuration 3 of
#         run_comb_phat256.jl; with m = 2, the keys of the 3 patterns leave no n with a margin of 0.2 bit at p = 256)
#    3-4: p = 512 (B = 8),  RGSW (2^14,2), aut (2^8,4),  n = 826, m = 1, h = 1 and 2   (its configuration 1)
#    5:   p = 512 (B = 8),  RGSW (2^14,2), aut (2^8,4),  n = 822, m = 2, h = 1
#    6:   p = 512 (B = 8),  RGSW (2^14,2), aut (2^10,3), n = 836, m = 2, h = 1   (m = 2: 3n/8 active groups instead
#         of n/2 non-zero coordinates, so that three digits suffice for the automorphism keys)
#    7:   p = 256 (B = 16), RGSW (2^11,3), aut (2^9,4),  n = 896, m = 1, h = 1   (four digits, margin 0.19 bit)
#    8:   p = 256 (B = 16), RGSW (2^11,3), aut (2^11,3) shifted by 4 bits, n = 896, m = 1, h = 1
#    9:   p = 512 (B = 8),  RGSW (2^14,2), aut (2^11,3), n = 820, m = 2, h = 1
#    10:  p = 512 (B = 8),  RGSW (2^14,2), aut (2^14,2) shifted by 3 bits, n = 830, m = 2, h = 1
#  Shifted gadget Gadget(b, ℓ, off) (src/glwe.jl): the mask divided by B has its top log2 B bits equal to zero, and the
#  digits start at bit off + 1 = log2 B + 1; the rounding term of the automorphism key switchings is divided by 2^(2 off).
#  PART=N: measured noise (NE errors, default 2000; key switch: 400) against the prediction, and margin in bits;
#  PART=T: time per error, minimum over REPS runs (default 5): comb bootstrap of B errors / B + key switch N -> n,
#          phases of an iteration, and the error bootstrap alone.
# Run from the julia folder of the repository:   PART=NT julia --project -t 1 benchmarks/run_comb_variants.jl | tee mb_variants_mac.txt
#   (CFGS=1,6,7,8,9,10: some configurations only; QUICK=1: n = 64, to test the script, numbers meaningless)
const MYPART = get(ENV, "PART", "NT")
ENV["PART"] = "-"                                   # run_curves.jl: definitions only (N, σ, U128, NU, CFG, timers)
include(joinpath(@__DIR__, "run_curves.jl"))
ENV["PART"] = MYPART

const QUICK = get(ENV, "QUICK", "0") == "1"
const NE = parse(Int, get(ENV, "NE", "2000"))
const PH = 256
const STAIR_832 = [(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10),
                   (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18),
                   (832, 1, 17)]
const STAIR_896 = [(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7),
                   (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20),
                   (912, 1, 19), (896, 1, 19)]
const STAIR_836 = vcat(STAIR_896, [(880, 1, 19), (864, 1, 18), (848, 1, 18), (836, 1, 17)])
# (p, RGSW gadget, automorphism gadget, n, m, h, 16 V*/p^2, chain of the key switch N -> n)
const VAR = [
    (256, Gadget(11, 3), Gadget(7, 5), 896, 1, 1, 4.2407e-03, STAIR_896),
    (256, Gadget(11, 3), Gadget(7, 5), 896, 1, 2, 4.2407e-03, STAIR_896),
    (512, Gadget(14, 2), Gadget(8, 4), 826, 1, 1, 9.7432e-04, vcat(STAIR_832, [(826, 1, 17)])),
    (512, Gadget(14, 2), Gadget(8, 4), 826, 1, 2, 9.7432e-04, vcat(STAIR_832, [(826, 1, 17)])),
    (512, Gadget(14, 2), Gadget(8, 4), 822, 2, 1, 9.6942e-04, vcat(STAIR_832, [(822, 1, 17)])),
    (512, Gadget(14, 2), Gadget(10, 3), 836, 2, 1, 9.8658e-04, STAIR_836),
    (256, Gadget(11, 3), Gadget(9, 4), 896, 1, 1, 4.2407e-03, STAIR_896),
    (256, Gadget(11, 3), Gadget(11, 3, 4), 896, 1, 1, 4.2407e-03, STAIR_896),
    (512, Gadget(14, 2), Gadget(11, 3), 820, 2, 1, 9.6697e-04, vcat(STAIR_832, [(820, 1, 17)])),
    (512, Gadget(14, 2), Gadget(14, 2, 3), 830, 2, 1, 9.7922e-04, vcat(STAIR_832, [(830, 1, 17)])),
]
const CFGS = parse.(Int, split(get(ENV, "CFGS", join(1:length(VAR), ",")), ","))
const PHASES = ["division", "FFTs, digits, tables", "baby + public", "giant steps", "inv FFT, digits C_L",
                "ext. product", "final inv FFT"]

npatterns(m) = (1 << m) - 1
gs(G) = G.off == 0 ? gstr(G) : "(2^$(G.logB),$(G.ell)) shifted by $(G.off) bits"
"Groups of the key s (first n coordinates of s') with a non-zero pattern."
nactive(s, m) = count(t -> any(==(1), s[(t-1)*m+1:t*m]), 1:length(s)÷m)
"Predicted output variance (Proposition 1 grouped by m): keys of the 2^m - 1 patterns of each group, rounding and
key switchings of the automorphisms for the active groups, error of the FFT of the external products."
function var_mbg(P, m, nact)
    key, rnd = var_ext(P); G = P.n ÷ m
    return npatterns(m) * G * (key + 2.0^-85 * 2.0^(2P.Gbr.logB) * P.Gbr.ell) +
           nact * (rnd + ((P.B - P.h) * P.B + P.h - 1) * var_ks(P))
end

println("Comb bootstrap at p̂ = 256, N = 2048, variants in h and m", QUICK ? "   (QUICK: n = 64, test only)" : ""); flush(stdout)
csv = String[]
for c in CFGS
    p, Gbr, Gks, n0, m, h, vterm, chain = VAR[c]
    n = QUICK ? 64 : n0
    rng = Xoshiro(200c + n); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = p, phat = PH, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = h)
    K = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, K; m, nthreads = 1)
    @printf("\nvariant %d: p = %d, B = %d, n = %d, m = %d, (g,h) = (%d,%d), RGSW %s, aut %s\n",
            c, p, P.B, n, m, P.g, P.h, gstr(Gbr), gs(Gks))
    if occursin('N', MYPART)
        tstart = time(); e = Float64[]
        for _ in 1:cld(NE, P.B)
            aints, bints, E = random_inputs(P, K.s, rng)
            append!(e, output_errors(P, combboot_grouped(P, fp, K, aints, bints; m, wg), K.sk.s, E))
        end
        vo = mean(abs2, e); vks, vksp = ks_var(K.sk.s, chain, rng); vp = var_mbg(P, m, nactive(K.s, m))
        @printf("  noise (log2 σ, measured / predicted): out %7.2f / %7.2f   KS %7.2f / %7.2f   (%d errors)\n",
                l2(vo), l2(vp), l2(vks), l2(vksp), length(e))
        @printf("  margin of the condition: %5.2f bits measured, %5.2f bits predicted   (%.0f s)\n",
                margin(PH, NU * vo, vks, vterm), margin(PH, NU * vp, vksp, vterm), time() - tstart)
    end
    if occursin('T', MYPART)
        aints, bints, _ = random_inputs(P, K.s, rng)
        combboot_grouped(P, fp, K, aints, bints; m, wg)                                   # warm-up
        tg = 1e3 * minimum(@elapsed(combboot_grouped(P, fp, K, aints, bints; m, wg)) for _ in 1:REPS) / P.B
        tk = ks_time(chain)
        @printf("  time per error: rotation %6.2f (%d errors in %.1f) + key switch %6.3f = %6.2f ms\n",
                tg, P.B, tg * P.B, tk, tg + tk)
        tim = fill(Inf, 7)
        for _ in 1:REPS
            t7 = zeros(7); combboot_grouped(P, fp, K, aints, bints; m, wg, tim = t7); tim .= min.(tim, t7)
        end
        print("  µs per coordinate (batch):")
        for (nm, x) in zip(PHASES, tim)
            @printf(" %s %.1f;", nm, x / (1e3 * n))
        end
        @printf(" total %.1f\n", sum(tim) / (1e3 * n))
        if m == 1                                                                          # engine of fast_opt.jl
            ws = Fast.Work(P, fp, K; nthreads = 1)
            combboot_fast(P, fp, K, aints, bints; ws)
            tf = 1e3 * minimum(@elapsed(combboot_fast(P, fp, K, aints, bints; ws)) for _ in 1:REPS) / P.B
            @printf("  (fast_opt.jl: rotation %6.2f + key switch %6.3f = %6.2f ms per error)\n", tf, tk, tf + tk)
            ws = nothing
        end
        push!(csv, @sprintf("cbv,%d,p%d_B%d_n%d_m%d_h%d_aut%d-%d-%d,%.4f", c, p, P.B, n, m, h, Gks.logB, Gks.ell, Gks.off, tg + tk))
    end
    flush(stdout); K = nothing; wg = nothing; GC.gc()
end

if occursin('T', MYPART)
    n, G, _, chain = CFG[4].R
    n = QUICK ? 64 : n
    tr = rotation_time(n, G); tk = ks_time(chain)
    @printf("\nerror bootstrap alone (run_curves.jl, k = 4): n = %d, %s, m = 2: rotation %6.2f + key switch %6.3f = %6.2f ms\n",
            n, gstr(G), tr, tk, tr + tk)
    push!(csv, @sprintf("eb,4,refresh,%.4f", tr + tk))
    println("CSV (scheme,variant,parameters,ms per error)"); foreach(println, csv)
end
