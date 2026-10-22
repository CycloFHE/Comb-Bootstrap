# Rounding of an LWE ciphertext at the resolution R (Sections 4.1 and 4.3, Appendix B of the paper), for a binary
# key s, coordinate by coordinate (the rounded ciphertext is under s itself: the transfer to s' is free for every m):
#
#   ā_u = round(R a_u),  X_u = ā_u - R a_u ∈ [-1/2, 1/2] (public),   b̄ = round(R b),  β = b̄ - R b,
#   E_R = R e + β - T,   T = Σ_u s_u X_u.
#
# Over a uniform binary key, T has the public mean c = Σ_u X_u / 2 and the public variance V = Σ_u X_u^2 / 4.
#   Recentering: c̄ = round(c) is added to b̄ (and to the error E = E_R + c̄, which the rotation refreshes), so that the
#                rounding term T - c̄ has the mean c - c̄ ∈ [-1/2, 1/2].
#   Selection:   while V > V*, the ciphertext is re-randomized by public encryptions of zero under s (a pool of encryptions
#                of zero; each trial adds a pair {i, j} of the pool drawn at random, so that the fixed errors of the pool
#                average out over the ciphertexts), which changes the mask but not μ, at the cost of the noise of two
#                encryptions of zero. With V* at the 1% quantile of V, about 100 trials.

export ZeroPool, zero_pool, vstar_threshold, rounding_stats, Rounded, round_select, rounded_errors

"Threshold V* of the selection, binary key, coordinate-wise rounding: quantile z (default 1%) of V for uniform masks
(E[X^2/4] = 1/48, Var[X^2/4] = 1/2880 per coordinate)."
vstar_threshold(n::Integer; z::Real = -2.3263) = n / 48 + z * sqrt(n / 2880)

"Public encryptions of zero under s (dimension n), noise σ."
struct ZeroPool
    cts::Vector{LWE}
    σ::Float64
end
zero_pool(s::Vector{Int64}, count::Int, σ::Float64, rng) =
    ZeroPool([lwe_encrypt(s, UInt64(0), σ, rng) for _ in 1:count], σ)

"Rounding error X = round(R x) - R x of x ∈ T at the resolution R (Float64)."
@inline rounding_error(x::UInt64, R::UInt64) = -Float64(reinterpret(Int64, x * R)) * 2.0^-64

"Mean c and variance V of T = Σ s_u X_u over a uniform binary key, for the mask a rounded at the resolution R."
function rounding_stats(a::AbstractVector{UInt64}, R::Integer)
    Ru = UInt64(R); c = 0.0; V = 0.0
    @inbounds @simd for u in eachindex(a)
        x = rounding_error(a[u], Ru)
        c += x; V += x * x
    end
    return c / 2, V / 4
end

"rounding_stats of the mask a + z1 + z2 (without forming it)."
function rounding_stats(a::AbstractVector{UInt64}, z1::AbstractVector{UInt64}, z2::AbstractVector{UInt64}, R::Integer)
    Ru = UInt64(R); c = 0.0; V = 0.0
    @inbounds @simd for u in eachindex(a)
        x = rounding_error(a[u] + z1[u] + z2[u], Ru)
        c += x; V += x * x
    end
    return c / 2, V / 4
end

"A ciphertext rounded at the resolution R: ā (in [0, R)), b̄ + c̄ (in [0, R)), c̄, the public c and V of the rounding,
the number of trials of the selection, and the selected ciphertext (input plus the public encryptions of zero)."
struct Rounded
    abar::Vector{Int}
    bbar::Int
    cbar::Int
    c::Float64
    V::Float64
    trials::Int
    ct::LWE
end

"""
    round_select(ct, R, pool; Vstar = Inf, recenter = true, rng = Random.default_rng(), maxtrials = 100_000)

Rounding of the LWE ciphertext ct = (a, b) (binary key) at the resolution R, coordinate by coordinate, with selection
(while V > Vstar, each trial adds a random pair of distinct encryptions of zero of the pool) and recentering
(c̄ = round(c) is added to the rounded body). If no pair reaches Vstar within maxtrials, the pair of smallest V is kept.
"""
function round_select(ct::LWE, R::Integer, pool::Union{ZeroPool,Nothing}; Vstar::Real = Inf, recenter::Bool = true,
                      rng::AbstractRNG = Random.default_rng(), maxtrials::Int = 100_000)
    a = ct.a; b = ct.b
    c, V = rounding_stats(a, R); trials = 0
    if V > Vstar && pool !== nothing
        Z = pool.cts; P = length(Z); best = (V, 0, 0, c)
        while trials < maxtrials
            trials += 1
            i = rand(rng, 1:P); j = rand(rng, 1:P-1); j += (j >= i)
            c2, V2 = rounding_stats(a, Z[i].a, Z[j].a, R)
            V2 < best[1] && (best = (V2, i, j, c2))
            V2 <= Vstar && break
        end
        if best[2] > 0
            V, i, j, c = best
            a = a .+ Z[i].a .+ Z[j].a; b = b + Z[i].b + Z[j].b
        end
    end
    abar = [round_to(x, R) for x in a]
    cbar = recenter ? round(Int, c) : 0
    return Rounded(abar, mod(round_to(b, R) + cbar, R), cbar, c, V, trials, LWE(a, b))
end

"""
    rounded_errors(rs, s, mus, R)

The errors E = E_R + c̄ of the rounded ciphertexts rs (Rounded) of the messages mus ∈ Z_R (R μ), in (-R/2, R/2]:
E ≡ b̄ + c̄ - ā·s - R μ (mod R). They are the inputs of the rotation (|E| ≤ p/4 in case (B)).
"""
function rounded_errors(rs::AbstractVector{Rounded}, s::AbstractVector{Int64}, mus::AbstractVector{<:Integer}, R::Integer)
    E = Int[]
    for (r, μ) in zip(rs, mus)
        x = mod(r.bbar - sum(r.abar[u] * s[u] for u in eachindex(r.abar)) - μ, R)
        push!(E, x > R ÷ 2 ? x - R : x)
    end
    return E
end
