# Comb bootstrap at 2, 4, 8, 12 and 16 bits (p̂ = 4^k, k = 1, 2, 4, 6, 8), against the error bootstrap alone (refresh
# configurations k of run_curves.jl), in the same session. One thread; N = 2048, binary keys, m = 1, h = 1, rounding
# coordinate by coordinate, polynomial key switch (d = 16), ν = 1000 additions, failure 2^-128, engine of src/grouped.jl.
#  Configurations: cheapest of the cost model (margin of 0.2 bit, time per coordinate fitted
#  on run_comb_variants.jl), with the shifted gadget of the automorphism keys (src/glwe.jl) where it pays:
#     2 bits: p = 256 (B = 16), RGSW (2^23,1), aut (2^14,2),                 n = 608
#     4 bits: p = 256 (B = 16), RGSW (2^14,2), aut (2^14,2) shifted by 4,    n = 688
#     8 bits: p = 256 (B = 16), RGSW (2^11,3), aut (2^11,3) shifted by 4,    n = 896   (variant 8 of run_comb_variants.jl)
#    12 bits: p = 512 (B = 8),  RGSW (2^11,3), aut (2^11,3) shifted by 3,    n = 986
#    16 bits: p = 512 (B = 8),  RGSW (2^8,5),  aut (2^6,6) shifted by 3,     n = 1157
#  The prediction of the noise adds to Proposition 1 the error of the FFT of the external products (2^-85 B^2 ℓ) and
#  of the automorphism key switchings (2^-88.5 B^2 ℓ, fitted on variant 8 of run_comb_variants.jl).
#  Rounding (src/rounding.jl): coordinate by coordinate, recentered, and selected below the threshold V* of the
#  configuration (16 V*/p^2 in SIZES) with a pool of 128 public encryptions of zero under s (noise σ_128(n)).
#  PART=N: measured noise of the rotation (NE errors, default 2000, inputs E_k uniform in the window; key switch: 400)
#          against the prediction, and margin in bits;
#  PART=E: end to end (src/fast_refresh.jl): NE ciphertexts LWE_{s'}(μ) at the noise of ν = 1000 fresh outputs (predicted),
#          key switch, rounding with recentering and selection, comb bootstrap, subtraction: the errors E_k given to the
#          rotation (variance against the prediction, largest |E_k| against the window p/4), the margin of the condition
#          computed with their measured variance, the trials of the selection, and the noise of the refreshed outputs;
#  PART=T: time per error, minimum over REPS runs (default 5): comb bootstrap of B errors / B + key switch N -> n + rounding
#          with selection (mean over 20 B ciphertexts), and the error bootstrap alone of the same size.
#  M=2: mask grouped by m = 2 in the rotation (keys of the rotation 1.5 times larger); n odd (16 bits: 1157) is raised to
#  the next multiple of m (last step of the key switch and V* recomputed); PART=T also times the rotation with m = 1 in the
#  same session (other keys), for comparison.
# Run from the julia folder of the repository:   PART=NET julia --project -t 1 benchmarks/run_comb_sizes.jl | tee mb_sizes_mac.txt
#   (BITS=2,4: some sizes only; M=2: grouping by 2; QUICK=1: n = 64, to test the script, numbers meaningless)
const MYPART = get(ENV, "PART", "NET")
ENV["PART"] = "-"                                   # run_curves.jl: definitions only (N, σ, U128, NU, CFG, timers)
include(joinpath(@__DIR__, "run_curves.jl"))
ENV["PART"] = MYPART

const QUICK = get(ENV, "QUICK", "0") == "1"
const NE = parse(Int, get(ENV, "NE", "2000"))
include(joinpath(@__DIR__, "comb_sizes.jl"))          # SIZES, gs, CAUT, var_mbs
const BITS = parse.(Int, split(get(ENV, "BITS", "2,4,8,12,16"), ","))
const MG = parse(Int, get(ENV, "M", "1"))

println("Comb bootstrap at 2 to 16 bits, N = 2048, m = $MG, h = 1", QUICK ? "   (QUICK: n = 64, test only)" : ""); flush(stdout)
csv = String[]
for bits in BITS
    phat = 2^bits; p, Gbr, Gks, n0, vterm, chain = SIZES[bits]
    m = MG
    if n0 % m != 0                                    # n raised to a multiple of m: last step of the chain and V*
        n0 += m - n0 % m; chain = [chain[1:end-1]; (n0, chain[end][2], chain[end][3])]
        vterm = 16 * vstar_threshold(n0) / p^2
    end
    n = QUICK ? 64 : n0
    rng = Xoshiro(300bits + n); r = Ring(N); fp = FPlan(N)
    P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ, h = 1)
    K = fast_keygen_grouped(P, r, fp, rng; m); wg = WorkG(P, fp, K; m, nthreads = 1)
    R = R_of(P); Vs = vterm * p^2 / 16; pool = zero_pool(K.s, 128, σ128(n0), rng)       # selection
    vin = NU * var_mbs(P, nactive(K.s, m), m)                         # ν fresh outputs (Proposition 1, grouping m)
    @printf("\n%2d bits: p = %d, B = %d, n = %d, RGSW %s, aut %s, V* = %.2f, m = %d\n", bits, p, P.B, n, gstr(Gbr), gs(Gks), Vs, m)
    if occursin('N', MYPART)
        tstart = time(); e = Float64[]
        for _ in 1:cld(NE, P.B)
            aints, bints, E = random_inputs(P, K.s, rng)
            append!(e, output_errors(P, combboot_grouped(P, fp, K, aints, bints; m, wg), K.sk.s, E))
        end
        vo = mean(abs2, e); vks, vksp = ks_var(K.sk.s, chain, rng); vp = var_mbs(P, nactive(K.s, m), m)
        @printf("  noise (log2 σ, measured / predicted): out %7.2f / %7.2f   KS %7.2f / %7.2f   (%d errors)\n",
                l2(vo), l2(vp), l2(vks), l2(vksp), length(e))
        @printf("  margin of the condition: %5.2f bits measured, %5.2f bits predicted   (%.0f s)\n",
                margin(phat, NU * vo, vks, vterm), margin(phat, NU * vp, vksp, vterm), time() - tstart)
    end
    if occursin('E', MYPART)
        tstart = time()
        ksk = QUICK ? stair_keygen(K.sk.s, N, [(n, 1, 20)], rng; σf = x -> 2.0^-40) : stair_keygen(K.sk.s, N, chain, rng)
        E = Int[]; o = Float64[]; ntr = Int[]; Vsel = Float64[]
        for _ in 1:cld(NE, P.B)
            is = rand(rng, 0:phat-1, P.B)
            cts = [lwe_encrypt(K.sk.s, Δ(i, phat), sqrt(vin), rng) for i in is]
            out, rs = eb_refresh(P, fp, K, cts; ksk, m, wg, pool, Vstar = Vs, rng, info = true)
            append!(E, rounded_errors(rs, K.s, [p * i for i in is], R))
            append!(o, [torus_to_float(lwe_phase(out[k], K.sk.s) - Δ(is[k], phat)) for k in 1:P.B])
            append!(ntr, [x.trials for x in rs]); append!(Vsel, [x.V for x in rs])
        end
        vE = mean(abs2, E); vEp = R^2 * (vin + stair_var(N, chain) + 2σ128(n0)^2) + mean(Vsel) + 1 / 6
        @printf("  end to end: E_k (rotation inputs) σ %6.2f measured / %6.2f predicted, max |E_k| %d (window %d), mean %.3f;  selection: V ≤ V* in %d/%d, mean V %.2f, %.0f trials on average;  output noise %7.2f   (%d ciphertexts)\n",
                0.5log2(vE), 0.5log2(vEp), maximum(abs, E), p ÷ 4, mean(E), count(<=(Vs), Vsel), length(Vsel), mean(Vsel),
                mean(ntr), l2(mean(abs2, o)), length(E))
        @printf("  margin of the condition, end to end: %5.2f bits (measured variance of E_k), %5.2f bits predicted   (%.0f s)\n",
                l2((1 / U128^2) / ((4phat)^2 * vE / R^2)), margin(phat, vin, stair_var(N, chain), vterm), time() - tstart)
        ksk = nothing
    end
    if occursin('T', MYPART)
        cts = [lwe_encrypt(K.s, rand(rng, UInt64), sqrt(vin), rng) for _ in 1:20P.B]          # under s, after the key switch
        foreach(c -> round_select(c, R, pool; Vstar = Vs, rng), cts[1:P.B])                     # warm-up
        ts = 1e3 * mean(@elapsed(round_select(c, R, pool; Vstar = Vs, rng)) for c in cts)
        aints, bints, _ = random_inputs(P, K.s, rng)
        combboot_grouped(P, fp, K, aints, bints; m, wg)                               # warm-up
        tg = 1e3 * minimum(@elapsed(combboot_grouped(P, fp, K, aints, bints; m, wg)) for _ in 1:REPS) / P.B
        tk = ks_time(chain)
        K = nothing; wg = nothing; GC.gc()
        if m > 1                                      # the rotation with m = 1, same session
            K1 = fast_keygen_grouped(P, r, fp, rng; m = 1); wg1 = WorkG(P, fp, K1; m = 1, nthreads = 1)
            a1, b1, _ = random_inputs(P, K1.s, rng)
            combboot_grouped(P, fp, K1, a1, b1; m = 1, wg = wg1)
            tg1 = 1e3 * minimum(@elapsed(combboot_grouped(P, fp, K1, a1, b1; m = 1, wg = wg1)) for _ in 1:REPS) / P.B
            @printf("  rotation per error, same session: m = %d %6.2f ms, m = 1 %6.2f ms (ratio %.3f)\n", m, tg, tg1, tg / tg1)
            push!(csv, @sprintf("rot,%d,m1,%.4f", bits, tg1), @sprintf("rot,%d,m%d,%.4f", bits, m, tg))
            K1 = nothing; wg1 = nothing; GC.gc()
        end
        ne, Ge, _, che = CFG[bits ÷ 2].R
        ne = QUICK ? 64 : ne
        tr = rotation_time(ne, Ge); tke = ks_time(che)
        @printf("  time per error: rotation %6.2f (%d errors in %.1f) + key switch %6.3f + rounding with selection %6.3f = %6.2f ms;  error bootstrap alone (n = %d, %s, m = 2): %6.2f + %6.3f = %6.2f ms;  ratio %.2f\n",
                tg, P.B, tg * P.B, tk, ts, tg + tk + ts, ne, gstr(Ge), tr, tke, tr + tke, (tr + tke) / (tg + tk + ts))
        push!(csv, @sprintf("cbs,%d,p%d_B%d_n%d%s,%.4f", bits, p, P.B, n, m == 1 ? "" : "_m$m", tg + tk + ts),
              @sprintf("eb,%d,refresh,%.4f", bits ÷ 2, tr + tke))
    end
    flush(stdout); K = nothing; wg = nothing; GC.gc()
end
occursin('T', MYPART) && (println("CSV (scheme,bits or k,parameters,ms per error)"); foreach(println, csv))
