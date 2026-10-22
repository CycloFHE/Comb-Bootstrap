# Case (A) of the paper (Appendix C): p an odd prime, ring of conductor M = M0·p.
#   M0 = 1      : the error bootstrap alone, ring of conductor p, N = p - 1 (Figure 2);
#   M0 ≥ 4 (2^k): the comb bootstrap, B = M0/2 columns, N = B (p - 1) (Algorithm 1).
# Exact reference implementation: products modulo 2^64 are exact (three signed limbs of 22, 21 and 21 bits, negacyclic
# NTT modulo the 62-bit prime of ntt.jl on the Kronecker substitution U = Z^L), so that the measured noise is the noise of
# the scheme.
#
# Representation (Appendix C.3). R = Z[U]/(U^B + 1) ⊗ Z[V]/Φ_p(V), with U = X^p (order M0; U^B = -1) and V = X^{M0}
# (order p), X = U^α V^β with p α + M0 β ≡ 1 (mod M); ϖ_k = U^k, ω_j = V^j, Ω_{j,k} = U^k V^j (k < B, j ≤ p - 2).
#   * An element of R is stored by its coordinates on Ω_{j,k}: a vector of length B p, entry k p + j + 1, whose entries
#     j = p - 1 are zero ("Ω form"; normR! brings any lift to this form: V^{p-1} = -Σ_{j<p-1} V^j).
#   * An element P of T^∨ = K/R^∨ is stored by its augmented coordinates x[k][v], v ∈ Z_p (same indexing):
#         P = Σ_k (U^k/B) (1/p) Σ_v x[k][v] V^v,   Σ_v x[k][v] = 0,
#     so that x[k][j] = ⟨P, Ω_{j,k}⟩ (j ≤ p - 2) are its coordinates on Ω*_{j,k} = U^k (V^j - V^{p-1})/(p B) (Prop. 3),
#     and ⟨P, U^k V^v⟩ = x[k][v] for every v ∈ Z_p. The product r·P (r ∈ R) has the augmented coordinates r ⋆ x
#     (negacyclic convolution in k, cyclic in v), X^e and the automorphisms act by signed permutations.
#   * Decomposition. R^∨ = Ω*_{0,0} R, Ω*_{0,0} = (1 - V^{-1})/(p B): a ∈ T^∨ is decomposed through
#     a' = a/Ω*_{0,0} ∈ T = K/R (prefix sums along v), whose Ω-coordinates are split into digits in R; the GGSW and
#     key-switching rows carry the gadget Ω*_{0,0} g_l. The automorphisms of the comb (a ≡ 1 mod p) fix Ω*_{0,0}, so that
#     they commute with this decomposition (hoisting).
#
# Usage:  include("src/CombBoot.jl"); using .CombBoot; include("src/fast.jl"); using .Fast;
#         include("src/caseA.jl"); using .CaseA

module CaseA

using Random, Printf
using ..CombBoot
import ..CombBoot: NTTPlan, Modulus, find_ntt_prime, ntt!, intt!, to_modP, from_modP, mulmod, addmod, mac!,
                    Gadget, gvalue, decompose, LWE, lwe_phase, lwe_encrypt, torus_frac, torus_to_float, gauss_torus,
                    round_select, ZeroPool, zero_pool, vstar_threshold, Rounded, round_to
import ..Fast: StairKey, stair_keygen, stair_ks, stair_var, σ128

export pattern, pattern_exp, ARing, ASecretKey, AGLWE, AGGSW, AKSKey, akeygen_sk, aglwe_encrypt, aphase, aggsw_encrypt, aextprod, acmux,
       aks_keygen, aks_hoisted, aextract, mulUV, mulX_A, aut_A, normR!, fixaug!, rand_aug, noise_aug, omega00_times,
       div_omega00, adecompose, mulRT, kint, ktorus, from_kron, mods, avstar, aacc0, Ugroup_A, BA_poly_A, divB_A,
       AParams, AKeys, akeygen, blind_rotation_A, combboot_A, refresh_A, var_out_A, var_ks_A, var_comb_A, conv_generic,
       rotate_v

# ---------------------------------------------------------------- ring

"Ring of conductor M = M0 p (M0 = 1, or a power of 2 ≥ 4), p an odd prime."
struct ARing
    p::Int; M0::Int; B::Int; N::Int; M::Int
    L::Int          # Kronecker stride: power of 2 ≥ 2p - 2 (the linear product along v has degree ≤ 2p - 3)
    NK::Int         # NTT size B L (negacyclic in U = Z^L)
    α::Int; β::Int  # X = U^α V^β
    plan::NTTPlan
end

isprime_small(n::Integer) = n >= 2 && all(n % d != 0 for d in 2:isqrt(n))

function ARing(p::Int, M0::Int)
    (p > 2 && isprime_small(p)) || error("p must be an odd prime")
    (M0 == 1 || (ispow2(M0) && M0 >= 4)) || error("M0 must be 1 or a power of 2 ≥ 4")
    B = M0 == 1 ? 1 : M0 ÷ 2
    N = B * (p - 1); M = M0 * p
    L = nextpow(2, max(2p - 2, 2)); NK = B * L
    α, β = M0 == 1 ? (0, 1) : (invmod(p, M0), invmod(M0, p))
    CombBoot._PRIME[] == 0 && (CombBoot._PRIME[] = find_ntt_prime())
    return ARing(p, M0, B, N, M, L, NK, α, β, NTTPlan(NK, Modulus(CombBoot._PRIME[])))
end

@inline idx(Rg::ARing, k::Int, v::Int) = k * Rg.p + v + 1
mods(v::Integer, p::Integer) = (w = mod(v, p); w > p ÷ 2 ? w - p : w)

"X^e = sign · U^ku V^kv."
function xmono(Rg::ARing, e::Integer)
    kv = mod(e * Rg.β, Rg.p)
    Rg.M0 == 1 && return 1, 0, kv
    ku = mod(e * Rg.α, Rg.M0); s = 1
    if ku >= Rg.B
        ku -= Rg.B; s = -1
    end
    return s, ku, kv
end

"(± U^ku V^kv)·x for augmented torus coordinates (or R coordinates, then followed by normR!)."
function mulUV(Rg::ARing, x::AbstractVector{T}, s::Int, ku::Int, kv::Int) where {T}
    B, p = Rg.B, Rg.p
    out = Vector{T}(undef, B * p)
    @inbounds for k in 0:B-1
        k2 = k + ku; sg = s
        while k2 >= B
            k2 -= B; sg = -sg
        end
        for v in 0:p-1
            v2 = v + kv; v2 >= p && (v2 -= p)
            val = x[k*p+v+1]
            out[k2*p+v2+1] = sg > 0 ? val : -val
        end
    end
    return out
end
mulX_A(Rg::ARing, x::AbstractVector, e::Integer) = mulUV(Rg, x, xmono(Rg, e)...)
"Rotation of every column by V^c (X^{M0 c})."
rotate_v(Rg::ARing, x::AbstractVector, c::Integer) = mulUV(Rg, x, 1, 0, mod(c, Rg.p))

"Automorphism X ↦ X^c (c prime to M) on augmented or R coordinates: U ↦ U^c, V ↦ V^c."
function aut_A(Rg::ARing, x::AbstractVector{T}, c::Integer) where {T}
    B, p = Rg.B, Rg.p
    out = zeros(T, B * p)
    cu = Rg.M0 == 1 ? 0 : mod(c, Rg.M0); cv = mod(c, p)
    @inbounds for k in 0:B-1
        e = Rg.M0 == 1 ? 0 : mod(cu * k, Rg.M0); sg = 1
        if e >= B
            e -= B; sg = -1
        end
        for v in 0:p-1
            v2 = mod(cv * v, p)
            val = x[k*p+v+1]
            out[e*p+v2+1] += sg > 0 ? val : -val
        end
    end
    return out
end

"Bring an R element to the Ω form (entries v = p - 1 equal to zero)."
function normR!(Rg::ARing, r::AbstractVector)
    p = Rg.p
    @inbounds for k in 0:Rg.B-1
        c = r[k*p+p]
        iszero(c) && continue
        for v in 0:p-2
            r[k*p+v+1] -= c
        end
        r[k*p+p] = zero(c)
    end
    return r
end

"Restore the augmented invariant: x[k][p-1] = -Σ_{v<p-1} x[k][v]."
function fixaug!(Rg::ARing, x::AbstractVector{T}) where {T}
    p = Rg.p
    @inbounds for k in 0:Rg.B-1
        s = zero(T)
        for v in 0:p-2
            s += x[k*p+v+1]
        end
        x[k*p+p] = -s
    end
    return x
end

rand_aug(Rg::ARing, rng) = fixaug!(Rg, rand(rng, UInt64, Rg.B * Rg.p))
"Error with i.i.d. coordinates of standard deviation σ on the basis Ω* (the noise model of the paper)."
noise_aug(Rg::ARing, σ::Float64, rng) = fixaug!(Rg, [gauss_torus(rng, σ) for _ in 1:Rg.B*Rg.p])

"Ω*_{0,0}·r (r ∈ R or T = K/R in Ω coordinates) in augmented coordinates: x[k][v] = r[k][v] - r[k][v+1]."
function omega00_times(Rg::ARing, r::AbstractVector{T}) where {T}
    p = Rg.p
    out = Vector{T}(undef, Rg.B * p)
    @inbounds for k in 0:Rg.B-1, v in 0:p-1
        w = v + 1 == p ? 0 : v + 1
        out[k*p+v+1] = r[k*p+v+1] - r[k*p+w+1]
    end
    return out
end

"a' = a/Ω*_{0,0} ∈ T = K/R in Ω coordinates (a ∈ T^∨ augmented): prefix sums y_{v+1} = y_v - a_v, a'_v = y_v - y_{p-1}."
function div_omega00(Rg::ARing, a::AbstractVector{T}) where {T}
    p = Rg.p
    out = zeros(T, Rg.B * p)
    y = Vector{T}(undef, p)
    @inbounds for k in 0:Rg.B-1
        y[1] = zero(T)
        for v in 0:p-2
            y[v+2] = y[v+1] - a[k*p+v+1]
        end
        for v in 0:p-2
            out[k*p+v+1] = y[v+1] - y[p]
        end
    end
    return out
end

"Digits in R (Ω form) of a ∈ T^∨: decomposition of a/Ω*_{0,0}."
adecompose(Rg::ARing, G::Gadget, a::Vector{UInt64}) = decompose(G, div_omega00(Rg, a))

# ---------------------------------------------------------------- exact products

"Signed limbs (22, 21, 21 bits): x ≡ l0 + 2^22 l1 + 2^43 l2 (mod 2^64), |l| ≤ 2^21."
@inline function limbs3(x::UInt64)
    l0 = Int64(x & 0x3fffff); l0 >= (Int64(1) << 21) && (l0 -= Int64(1) << 22)
    y = Int64((x - reinterpret(UInt64, l0)) >> 22)                      # < 2^42
    l1 = y & 0x1fffff; l1 >= (Int64(1) << 20) && (l1 -= Int64(1) << 21)
    z = (y - l1) >> 21
    l2 = z & 0x1fffff; l2 >= (Int64(1) << 20) && (l2 -= Int64(1) << 21)
    return l0, l1, l2
end

"Kronecker NTT of an R element in Ω form (entries v ≤ p - 2)."
function kint(Rg::ARing, r::AbstractVector{<:Integer})
    P = Rg.plan.m.P; p = Rg.p
    x = zeros(UInt64, Rg.NK)
    @inbounds for k in 0:Rg.B-1
        r[k*p+p] == 0 || error("kint: R element not in Ω form")
        for v in 0:p-2
            x[k*Rg.L+v+1] = to_modP(Int64(r[k*p+v+1]), P)
        end
    end
    return ntt!(similar(x), x, Rg.plan)
end

"Kronecker NTTs of the three limbs of an augmented torus vector."
function ktorus(Rg::ARing, t::AbstractVector{UInt64})
    P = Rg.plan.m.P; p = Rg.p
    xs = ntuple(_ -> zeros(UInt64, Rg.NK), 3)
    @inbounds for k in 0:Rg.B-1, v in 0:p-1
        l = limbs3(t[k*p+v+1])
        for i in 1:3
            xs[i][k*Rg.L+v+1] = to_modP(l[i], P)
        end
    end
    return ntuple(i -> ntt!(similar(xs[i]), xs[i], Rg.plan), 3)
end

"Augmented torus vector from the accumulated NTTs of the three limbs (inverse NTT, fold v mod p, recombination)."
function from_kron(Rg::ARing, accs)
    P = Rg.plan.m.P; p = Rg.p; L = Rg.L
    out = zeros(UInt64, Rg.B * p)
    sh = (0, 22, 43)
    for i in 1:3
        x = intt!(Vector{UInt64}(undef, Rg.NK), accs[i], Rg.plan)
        @inbounds for k in 0:Rg.B-1, w in 0:min(L, 2p - 2)-1
            v = w >= p ? w - p : w
            out[k*p+v+1] += reinterpret(UInt64, from_modP(x[k*L+w+1], P)) << sh[i]
        end
    end
    return out
end

"Exact product r·t mod 2^64, r ∈ R given by kint, t ∈ T^∨ augmented."
function mulRT(Rg::ARing, rk::Vector{UInt64}, t::AbstractVector{UInt64})
    ft = ktorus(Rg, t); m = Rg.plan.m
    accs = ntuple(_ -> zeros(UInt64, Rg.NK), 3)
    for i in 1:3
        mac!(accs[i], rk, ft[i], m)
    end
    return from_kron(Rg, accs)
end

"Reference product (any element type): negacyclic convolution in k, cyclic in v (r in Ω form or any lift)."
function conv_generic(Rg::ARing, r::AbstractVector, x::AbstractVector{T}) where {T}
    B, p = Rg.B, Rg.p
    out = zeros(T, B * p)
    for k1 in 0:B-1, v1 in 0:p-1
        c = r[k1*p+v1+1]
        iszero(c) && continue
        for k2 in 0:B-1, v2 in 0:p-1
            k = k1 + k2; sg = 1
            k >= B && (k -= B; sg = -1)
            v = mod(v1 + v2, p)
            cc = T <: Unsigned ? reinterpret(T, Int64(c)) : T(c)
            out[k*p+v+1] += sg > 0 ? cc * x[k2*p+v2+1] : -(cc * x[k2*p+v2+1])
        end
    end
    return out
end

# ---------------------------------------------------------------- keys, GLWE, GGSW, key switching, extraction

struct ASecretKey
    s::Vector{Int64}        # Ω coordinates of s' (binary)
    svec::Vector{Int64}     # the LWE key s' = (s'_{j,k}), index k (p - 1) + j + 1
    sk::Vector{UInt64}      # kint(s)
end

function akeygen_sk(Rg::ARing, rng)
    p = Rg.p
    svec = Int64.(rand(rng, 0:1, Rg.N))
    s = zeros(Int64, Rg.B * p)
    for k in 0:Rg.B-1, j in 0:p-2
        s[k*p+j+1] = svec[k*(p-1)+j+1]
    end
    return ASecretKey(s, svec, kint(Rg, s))
end

struct AGLWE
    a::Vector{UInt64}
    b::Vector{UInt64}
end
Base.:+(x::AGLWE, y::AGLWE) = AGLWE(x.a .+ y.a, x.b .+ y.b)
Base.:-(x::AGLWE, y::AGLWE) = AGLWE(x.a .- y.a, x.b .- y.b)
azero(Rg::ARing) = AGLWE(zeros(UInt64, Rg.B * Rg.p), zeros(UInt64, Rg.B * Rg.p))
mulUV(Rg::ARing, c::AGLWE, s::Int, ku::Int, kv::Int) = AGLWE(mulUV(Rg, c.a, s, ku, kv), mulUV(Rg, c.b, s, ku, kv))
rotate_v(Rg::ARing, c::AGLWE, e::Integer) = AGLWE(rotate_v(Rg, c.a, e), rotate_v(Rg, c.b, e))
aut_A(Rg::ARing, c::AGLWE, e::Integer) = AGLWE(aut_A(Rg, c.a, e), aut_A(Rg, c.b, e))

function aglwe_encrypt(Rg::ARing, sk::ASecretKey, m::AbstractVector{UInt64}, σ::Float64, rng)
    a = rand_aug(Rg, rng)
    return AGLWE(a, mulRT(Rg, sk.sk, a) .+ m .+ noise_aug(Rg, σ, rng))
end
aphase(Rg::ARing, sk::ASecretKey, c::AGLWE) = c.b .- mulRT(Rg, sk.sk, c.a)

const KRow = Tuple{NTuple{3,Vector{UInt64}},NTuple{3,Vector{UInt64}}}

"GGSW(μ) over T^∨: rows 1..ℓ encrypt 0 with μ Ω*_{0,0} g_l added to the mask, rows ℓ+1..2ℓ with it added to the body."
struct AGGSW
    rows::Vector{KRow}
    G::Gadget
end

function aggsw_encrypt(Rg::ARing, sk::ASecretKey, μ::Integer, G::Gadget, σ::Float64, rng)
    rows = KRow[]; z = zeros(UInt64, Rg.B * Rg.p)
    for part in 1:2, l in 1:G.ell
        c = aglwe_encrypt(Rg, sk, z, σ, rng)
        g = UInt64(μ) * gvalue(G, l)
        t = part == 1 ? c.a : c.b
        t[idx(Rg, 0, 0)] += g; t[idx(Rg, 0, Rg.p - 1)] -= g
        push!(rows, (ktorus(Rg, c.a), ktorus(Rg, c.b)))
    end
    return AGGSW(rows, G)
end

"Dec(c)·GGSW(μ): an encryption of μ·(phase of c)."
function aextprod(Rg::ARing, c::AGLWE, K::AGGSW)
    G = K.G; ℓ = G.ell; m = Rg.plan.m
    accs = ntuple(_ -> zeros(UInt64, Rg.NK), 6)
    for (part, comp) in enumerate((c.a, c.b))
        digits = adecompose(Rg, G, comp)
        for l in 1:ℓ
            D = kint(Rg, digits[l])
            ra, rb = K.rows[(part-1)*ℓ+l]
            for i in 1:3
                mac!(accs[i], D, ra[i], m); mac!(accs[3+i], D, rb[i], m)
            end
        end
    end
    return AGLWE(from_kron(Rg, accs[1:3]), from_kron(Rg, accs[4:6]))
end

acmux(Rg::ARing, c0::AGLWE, c1::AGLWE, K::AGGSW) = c0 + aextprod(Rg, c1 - c0, K)

"Key-switching key from τ_c(s') to s' (c ≡ 1 mod p): rows encrypt τ_c(s') Ω*_{0,0} g_l under s'."
struct AKSKey
    c::Int
    rows::Vector{KRow}
    G::Gadget
end

function aks_keygen(Rg::ARing, sk::ASecretKey, c::Integer, G::Gadget, σ::Float64, rng)
    mod(c, Rg.p) == 1 || error("aks_keygen: c ≢ 1 (mod p) does not commute with the decomposition")
    τs = normR!(Rg, aut_A(Rg, sk.s, c))
    base = omega00_times(Rg, τs)
    rows = KRow[]
    for l in 1:G.ell
        msg = [reinterpret(UInt64, x) * gvalue(G, l) for x in base]
        ct = aglwe_encrypt(Rg, sk, msg, σ, rng)
        push!(rows, (ktorus(Rg, ct.a), ktorus(Rg, ct.b)))
    end
    return AKSKey(Int(c), rows, G)
end

"Hoisted key switching: KS_c(C) = (0, τ_c(b)) - τ_c(Dec(a))·K_c for every key, with one decomposition of a."
function aks_hoisted(Rg::ARing, C::AGLWE, keys::Vector{AKSKey})
    isempty(keys) && return AGLWE[]
    G = keys[1].G; ℓ = G.ell; m = Rg.plan.m
    digits = adecompose(Rg, G, C.a)
    out = AGLWE[]
    for K in keys
        accs = ntuple(_ -> zeros(UInt64, Rg.NK), 6)
        for l in 1:ℓ
            D = kint(Rg, aut_A(Rg, digits[l], K.c))
            ra, rb = K.rows[l]
            for i in 1:3
                mac!(accs[i], D, ra[i], m); mac!(accs[3+i], D, rb[i], m)
            end
        end
        α = from_kron(Rg, accs[1:3]); β = from_kron(Rg, accs[4:6])
        push!(out, AGLWE(.-α, aut_A(Rg, C.b, K.c) .- β))
    end
    return out
end

"LWE encryption of ⟨phase, ϖ_k⟩ = ⟨phase, U^k⟩ under the vector s' (svec): mask ⟨a, U^{k-k'} V^{-j}⟩."
function aextract(Rg::ARing, C::AGLWE, k::Int)
    B, p = Rg.B, Rg.p
    a = Vector{UInt64}(undef, Rg.N)
    @inbounds for kp in 0:B-1, j in 0:p-2
        d = k - kp; sg = 1
        if d < 0
            d += B; sg = -1
        end
        v = j == 0 ? 0 : p - j
        x = C.a[d*p+v+1]
        a[kp*(p-1)+j+1] = sg > 0 ? x : -x
    end
    return LWE(a, C.b[k*p+1])
end

# ---------------------------------------------------------------- the error bootstrap and the comb bootstrap

struct AParams
    p::Int; M0::Int; B::Int; N::Int
    phat::Int
    n::Int
    Gbr::Gadget
    Gks::Gadget
    σ::Float64        # noise of the GLWE keys (coordinates on Ω*)
end
AParams(Rg::ARing; phat, n, Gbr, Gks = Gadget(1, 1), σ) = AParams(Rg.p, Rg.M0, Rg.B, Rg.N, phat, n, Gbr, Gks, σ)

R_of(P::AParams) = P.phat * P.p

struct AKeys
    m::Int                    # grouping of the mask (multi-bit form)
    sk::ASecretKey
    s::Vector{Int64}          # LWE key s = first n coordinates of s'
    bsk::Vector{Vector{AGGSW}}  # bsk[t][j] = RGSW(δ_{j, s_[t]}), group t = 1..n/m, pattern j = 1..2^m - 1 (bit i: coordinate i)
    ksk::Vector{AKSKey}       # τ_c, c ∈ U \ {1} (comb)
end

"The group U = {c ∈ (Z/M)^× : c ≡ 1 mod p}, c = 1 first (c mod M0 runs over the odd residues)."
function Ugroup_A(Rg::ARing)
    Rg.M0 == 1 && return [1]
    return [mod(1 + Rg.p * mod((r - 1) * invmod(Rg.p, Rg.M0), Rg.M0), Rg.M) for r in 1:2:Rg.M0-1]
end

"Pattern of the group t (bits of s_{(t-1)m+1..tm}, coordinate i on bit i - 1)."
pattern(s::AbstractVector{<:Integer}, t::Int, m::Int) = sum(Int(s[(t-1)*m+i]) << (i - 1) for i in 1:m)

"Exponent of the pattern j in the group t: sum of the rounded coordinates it selects (coordinate-wise rounding)."
pattern_exp(a::AbstractVector{<:Integer}, t::Int, j::Int, m::Int, p::Int) =
    mod(sum((j >> (i - 1)) & 1 == 1 ? Int(a[(t-1)*m+i]) : 0 for i in 1:m), p)

"Keys: RGSW(δ_{j,s_[t]}) for every group and non-zero pattern ((2^m - 1) n/m keys), and the automorphism keys."
function akeygen(Rg::ARing, P::AParams, rng; m::Int = 1)
    P.n % m == 0 || error("n must be a multiple of m")
    sk = akeygen_sk(Rg, rng)
    s = sk.svec[1:P.n]
    bsk = [[aggsw_encrypt(Rg, sk, pattern(s, t, m) == j ? 1 : 0, P.Gbr, P.σ, rng) for j in 1:(1<<m)-1] for t in 1:P.n÷m]
    ksk = [aks_keygen(Rg, sk, c, P.Gks, P.σ, rng) for c in Ugroup_A(Rg)[2:end]]
    return AKeys(m, sk, s, bsk, ksk)
end

"v*: augmented coordinates x[k][v] = (v mods p)/R in every column (Appendices C.1 and C.5)."
avstar(Rg::ARing, R::Integer) = [torus_frac(mods(v, Rg.p), R) for k in 0:Rg.B-1 for v in 0:Rg.p-1]

"acc_0 = Σ_k X^{-M b_k} π_k(v*): column k of v* rotated by V^{-bints[k+1]}."
function aacc0(Rg::ARing, R::Integer, bints::AbstractVector{<:Integer})
    v = avstar(Rg, R); p = Rg.p
    out = similar(v)
    for k in 0:Rg.B-1, w in 0:p-1
        out[k*p+w+1] = v[k*p+mod(w + bints[k+1], p)+1]
    end
    return out
end

"B·A_c = Σ_k X^{ã_k} ϖ_k^{1-c} = Σ_k V^{a_k} U^{k(1-c)}: list of its B monomials (sign, ku, kv)."
function BA_poly_A(Rg::ARing, acol::AbstractVector{<:Integer}, c::Integer)
    mons = Tuple{Int,Int,Int}[]
    for k in 0:Rg.B-1
        e = mod(k * (1 - c), Rg.M0); s = 1
        if e >= Rg.B
            e -= Rg.B; s = -1
        end
        push!(mons, (s, e, mod(acol[k+1], Rg.p)))
    end
    return mons
end

"Division by B of the coordinates on Ω* (rounded shift; any representative works, Lemma 4)."
function divB_A(Rg::ARing, c::AGLWE)
    s = trailing_zeros(Rg.B)
    s == 0 && return c
    f(x::UInt64) = (x >> s) + ((x >> (s - 1)) & 0x1)
    return AGLWE(fixaug!(Rg, f.(c.a)), fixaug!(Rg, f.(c.b)))
end

"Error bootstrap (B = 1): blind rotation in CMux form, read at ⟨·, 1⟩."
function blind_rotation_A(Rg::ARing, P::AParams, K::AKeys, arow::AbstractVector{<:Integer}, bint::Integer)
    m = K.m
    C = AGLWE(zeros(UInt64, Rg.B * Rg.p), aacc0(Rg, R_of(P), [bint]))
    for t in 1:length(arow)÷m
        D = azero(Rg)
        for j in 1:(1<<m)-1                       # acc + Σ_j δ_j (X^{ã_{t,j}} acc - acc)
            D = D + aextprod(Rg, rotate_v(Rg, C, pattern_exp(arow, t, j, m, Rg.p)) - C, K.bsk[t][j])
        end
        C = C + D
    end
    return aextract(Rg, C, 0)
end

"""
    combboot_A(Rg, P, K, aints, bints; trace)

Algorithm 1 in case (A), mask grouped by K.m, all the B - 1 automorphisms from one decomposition (h = 1): aints[k+1, u] = p a_{k,u},
bints[k+1] = p b_k (noise-free LWE ciphertexts modulo p). Returns LWE encryptions of E_k/R under s'.
"""
function combboot_A(Rg::ARing, P::AParams, K::AKeys, aints::AbstractMatrix{<:Integer}, bints::AbstractVector{<:Integer};
                    trace = nothing)
    U = Ugroup_A(Rg); m = K.m
    C = AGLWE(zeros(UInt64, Rg.B * Rg.p), aacc0(Rg, R_of(P), bints))     # line 1
    for t in 1:size(aints, 2)÷m                                             # line 2
        C0 = divB_A(Rg, C)                                                  # line 4
        Cs = vcat([C0], aks_hoisted(Rg, C0, K.ksk))                         # lines 5-7, shared by the patterns
        D = azero(Rg)
        for j in 1:(1<<m)-1                                                 # line 8
            acol = [pattern_exp(view(aints, k, :), t, j, m, Rg.p) for k in 1:Rg.B]
            CL = azero(Rg)
            for (i, c) in enumerate(U)                                      # line 9 (public products)
                for (s, ku, kv) in BA_poly_A(Rg, acol, c)
                    CL = CL + mulUV(Rg, Cs[i], s, ku, kv)
                end
            end
            D = D + aextprod(Rg, CL - C, K.bsk[t][j])                       # line 10
        end
        C = C + D
        trace === nothing || trace(t, C)
    end
    return [aextract(Rg, C, k) for k in 0:Rg.B-1]                           # lines 11-13
end

"""
    refresh_A(Rg, P, K, cts; ksk = nothing, pool = nothing, Vstar = Inf, recenter = true, rng, info = false)

The whole refresh in case (A): key switch N -> n (polynomial stair key switch, if ksk is given), rounding at R = p̂ p
with recentering and selection, rotation of the errors (one blind rotation for B = 1, Algorithm 1 for B ≥ 2),
subtraction from the noise-free encryption of μ_k + E_k/R. Returns LWE_{s'}(μ_k) (dimension N).
"""
function refresh_A(Rg::ARing, P::AParams, K::AKeys, cts::AbstractVector{LWE}; ksk::Union{StairKey,Nothing} = nothing,
                   pool = nothing, Vstar::Real = Inf, recenter::Bool = true, rng::AbstractRNG = Random.default_rng(),
                   info::Bool = false)
    length(cts) == Rg.B || error("B = $(Rg.B) ciphertexts expected")
    R = R_of(P); n = P.n
    t0 = time_ns()
    lw = ksk === nothing ? cts : [stair_ks(ksk, ct) for ct in cts]
    t1 = time_ns()
    rs = [round_select(ct, R, pool; Vstar, recenter, rng) for ct in lw]
    t2 = time_ns()
    aints = [mod(rs[k].abar[u], P.p) for k in 1:Rg.B, u in 1:n]
    bints = [mod(rs[k].bbar, P.p) for k in 1:Rg.B]
    errs = Rg.B == 1 ? [blind_rotation_A(Rg, P, K, view(aints, 1, :), bints[1])] : combboot_A(Rg, P, K, aints, bints)
    t3 = time_ns()
    out = LWE[]
    for k in 1:Rg.B
        a = zeros(UInt64, Rg.N)
        @inbounds for u in 1:n
            a[u] = torus_frac(rs[k].abar[u], R)
        end
        push!(out, LWE(a .- errs[k].a, torus_frac(rs[k].bbar, R) - errs[k].b))
    end
    info || return out
    return out, rs, (ks = (t1 - t0) / 1e6, rounding = (t2 - t1) / 1e6, rotation = (t3 - t2) / 1e6)
end

# ---------------------------------------------------------------- predicted variances (torus², per output coordinate)

"""Output variance of the error bootstrap in case (A), mask grouped by m (n* active groups), binary keys (Appendix G, doubled in case (A)): the digits of
a/Ω*_{0,0} meet the noise through Ω̄-coordinates that are differences of two digits."""
function var_out_A(P::AParams, nstar::Integer; m::Int = 1)
    b = P.Gbr.logB; ℓ = P.Gbr.ell; N = P.N
    key = 2ℓ * N * (2.0^(2b) + 2) / 12 * P.σ^2
    rnd = (1 + N / 2) * 2.0^(-2ℓ * b) / 12
    return 2 * (P.n ÷ m * ((1 << m) - 1) * key + nstar * rnd)
end

"Automorphism key switching in case (A) (same doubling)."
function var_ks_A(P::AParams)
    b = P.Gks.logB; ℓ = P.Gks.ell; N = P.N
    return 2 * (ℓ * N * (2.0^(2b) + 2) / 12 * P.σ^2 + (N / 2) * 2.0^(-2ℓ * b) / 12)
end

"Comb bootstrap (h = 1): n V_ext(key) + n1 [V_ext(rnd) + (B - 1) B V_ks] (Proposition 4, case (A))."
var_comb_A(P::AParams, nstar::Integer; m::Int = 1) = var_out_A(P, nstar; m) + nstar * (P.B - 1) * P.B * var_ks_A(P)

end # module
