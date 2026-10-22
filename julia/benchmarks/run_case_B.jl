# Comb bootstrap, case (B), realistic sizes: correctness, noise (measured vs Proposition 1), timings.
# Run from the julia folder of the repository:   julia benchmarks/run_case_B.jl
# Quick run (n = 64):              QUICK=1 julia benchmarks/run_case_B.jl
include(joinpath(@__DIR__, "..", "src", "CombBoot.jl"))
using .CombBoot
using Random, Printf

const QUICK = get(ENV, "QUICK", "0") == "1"
rng = Xoshiro(1)

# ---------------------------------------------------------------- parameters (edit here)
N    = 2048            # ring X^N + 1
p    = 256             # p = 2^β: M0 = B = 2N/p = 16 columns, window |E| ≤ p/4
phat = 4               # message modulus
n    = QUICK ? 64 : 630
σ    = 2.0^-40         # GLWE noise (2^-50.3 is the 128-bit value of Section 7.1 for N = 2048)
Gbr  = Gadget(10, 3)   # RGSW(s_u): B_g = 2^10, ℓ = 3
Gks  = Gadget(6, 6)    # automorphism keys: B_aut = 2^6, ℓ_aut = 6 (Section 5.5; 36 bits)

P = CBParams(N = N, p = p, phat = phat, n = n, Gbr = Gbr, Gks = Gks, σ = σ)
@printf("N = %d, p = %d, M0 = B = %d, R = %d, n = %d, (g,h) = (%d,%d), σ = 2^%.1f\n",
        P.N, P.p, P.B, R_of(P), P.n, P.g, P.h, log2(σ))
@printf("gadgets: RGSW (2^%d, %d), automorphisms (2^%d, %d)\n", Gbr.logB, Gbr.ell, Gks.logB, Gks.ell)

r = Ring(N)
t_key = @elapsed K = cb_keygen(P, r, rng)
@printf("key generation: %.1f s\n", t_key)

# ---------------------------------------------------------------- comb bootstrap vs separate rotations
aints, bints, E = random_inputs(P, K.s, rng)
n1 = sum(K.s)
combboot(P, r, K, aints[:, 1:2], bints)                       # warm-up (compilation)
separate_rotations(P, r, K, aints[:, 1:2], bints)
t_mb = @elapsed outs = combboot(P, r, K, aints, bints)
errs = output_errors(P, outs, K.sk.s, E)

t_sep = @elapsed outs2 = separate_rotations(P, r, K, aints, bints)
errs2 = output_errors(P, outs2, K.sk.s, E)

meas(e) = sum(abs2, e) / length(e)
@printf("\ncomb bootstrap          : %.2f s, max |err|·R = %.2e, log2 std measured %.2f, predicted %.2f\n",
        t_mb, maximum(abs.(errs)) * R_of(P), 0.5log2(meas(errs)), 0.5log2(var_combboot(P, n1)))
@printf("separate rotations  : %.2f s, max |err|·R = %.2e, log2 std measured %.2f, predicted %.2f\n",
        t_sep, maximum(abs.(errs2)) * R_of(P), 0.5log2(meas(errs2)), 0.5log2(var_separate(P, n1)))
@printf("time ratio separate/comb bootstrap = %.2f  (B = %d errors per rotation)\n", t_sep / t_mb, P.B)
println("(std estimated on ", P.B, " outputs; run several seeds for a precise comparison)")

# ---------------------------------------------------------------- end-to-end refresh
msgs = rand(rng, 0:phat-1, P.B)
cts = [lwe_encrypt(K.s, torus_frac(m, phat), 2.0^-30, rng) for m in msgs]
t_ref = @elapsed fresh = refresh(P, r, K, cts)
ok = all(abs(torus_to_float(lwe_phase(fresh[k], K.sk.s) - torus_frac(msgs[k], phat))) < 1 / (4phat)
         for k in 1:P.B)
@printf("\nrefresh of %d ciphertexts: %.2f s, all correct: %s\n", P.B, t_ref, ok)
