# The whole refresh of Figures 2 and 4 of the paper (included in module Fast, after grouped.jl and stair_ks.jl):
# key switch N -> n (polynomial stair key switch), rounding at R = p̂ p with recentering and selection
# (round_select, src/rounding.jl), rotation of the errors E_k = E_R + c̄ (comb bootstrap of the B ciphertexts, or
# separate rotations; B = 1, p = 2N: the error bootstrap alone), subtraction from the noise-free encryption of
# μ_k + E_k/R.

using Random
export eb_refresh

"""
    eb_refresh(P, fp, K, cts; ksk = nothing, m = 1, wg = WorkG(P, fp, K; m), pool = nothing, Vstar = Inf,
               recenter = true, rng = Random.default_rng(), comb = P.B > 1, info = false)

Refresh of the B = length(cts) LWE ciphertexts cts: under s' (dimension N) if a stair key ksk (N -> n) is given,
under s (dimension n) otherwise. comb = true: one comb bootstrap (combboot_grouped, keys of fast_keygen_grouped with
the same m); comb = false: separate rotations (separate_rotations_grouped). Returns the B refreshed ciphertexts
LWE_{s'}(μ_k), of dimension N, and with info = true also the Rounded of each ciphertext (selection, c̄) and the
times (ms) of the key switches, of the rounding with selection and of the rotation.
"""
function eb_refresh(P::CBParams, fp::FPlan, K::FKeys, cts::AbstractVector{LWE}; ksk::Union{StairKey,Nothing} = nothing,
                    m::Int = 1, wg::WorkG = WorkG(P, fp, K; m, nthreads = 1), pool = nothing, Vstar::Real = Inf,
                    recenter::Bool = true, rng::AbstractRNG = Random.default_rng(), comb::Bool = P.B > 1,
                    info::Bool = false)
    length(cts) == P.B || error("B = $(P.B) ciphertexts expected")
    R = R_of(P); n = P.n
    t0 = time_ns()
    lw = ksk === nothing ? cts : [stair_ks(ksk, ct) for ct in cts]
    t1 = time_ns()
    rs = [round_select(ct, R, pool; Vstar, recenter, rng) for ct in lw]
    t2 = time_ns()
    aints = [mod(rs[k].abar[u], P.p) for k in 1:P.B, u in 1:n]
    bints = [mod(rs[k].bbar, P.p) for k in 1:P.B]
    errs = comb ? combboot_grouped(P, fp, K, aints, bints; m, wg) :
                  separate_rotations_grouped(P, fp, K, aints, bints; m, wg)
    t3 = time_ns()
    out = LWE[]
    for k in 1:P.B
        a = zeros(UInt64, P.N)
        @inbounds for u in 1:n
            a[u] = torus_frac(rs[k].abar[u], R)
        end
        push!(out, LWE(a .- errs[k].a, torus_frac(rs[k].bbar, R) - errs[k].b))
    end
    info || return out
    return out, rs, (ks = (t1 - t0) / 1e6, rounding = (t2 - t1) / 1e6, rotation = (t3 - t2) / 1e6)
end
