# Predicted variances (torus², per extracted LWE coefficient), Section 5.5 of the paper.
# The stored coefficients are the coordinates on Ω*_i = X^i/N, so N·E‖ε‖² = coefficient variance.

"External product with a binary GGSW: key term (always) and rounding term (only if s_u = 1)."
function var_ext(P::CBParams)
    b = P.Gbr.logB; ℓ = P.Gbr.ell; N = P.N
    key = 2ℓ * N * (2.0^(2b) + 2) / 12 * P.σ^2
    rnd = (1 + N / 2) * 2.0^(-2ℓ * b) / 12
    return key, rnd
end

"Automorphism key switching (ρ = 1/2 for a binary key)."
function var_ks(P::CBParams)
    b = P.Gks.logB; ℓ = P.Gks.ell; N = P.N
    return ℓ * N * (2.0^(2b) + 2) / 12 * P.σ^2 + (N / 2) * 2.0^(-2(ℓ * b + P.Gks.off)) / 12
end

"Proposition 1: n·V_ext(key) + n1·[V_ext(rnd) + ((B-h)B + h-1)·V_ks]."
function var_combboot(P::CBParams, n1::Int)
    key, rnd = var_ext(P)
    return P.n * key + n1 * (rnd + ((P.B - P.h) * P.B + P.h - 1) * var_ks(P))
end

"B separate blind rotations in CMux form: n·V_ext(key) + n1·V_ext(rnd)."
function var_separate(P::CBParams, n1::Int)
    key, rnd = var_ext(P)
    return P.n * key + n1 * rnd
end

"Errors (in torus units) between the phases of the outputs and E_k/R."
function output_errors(P::CBParams, outs::Vector{LWE}, sprime::Vector{Int64}, E::Vector{Int})
    R = R_of(P)
    return [torus_to_float(lwe_phase(outs[k], sprime) - torus_frac(E[k], R)) for k in eachindex(outs)]
end
