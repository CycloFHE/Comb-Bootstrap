# GLWE (k = 1, i.e. RLWE), GGSW, external product, CMux, automorphism key switching, extraction.
# Convention: C = (a, b) encrypts μ under s' if b - a·s' = μ + e.

# ---------------------------------------------------------------- keys and gadget

struct SecretKey
    s::Vector{Int64}         # coefficients of s'(X)
    sntt::Vector{UInt64}     # its NTT
end
SecretKey(r::Ring, s::Vector{Int64}) = SecretKey(s, ntt_small(r, s))

"""Gadget g_j = 2^-(off + j·logB), j = 1..ℓ. off = 0: the usual gadget. off > 0 (shifted gadget): the digits start at
bit off + 1 and represent t modulo 2^-off; for the automorphism keys after the division by B of Algorithm 1
(off ≤ log2 B), whose masks have their top log2 B bits equal to zero, the off bits gained go to the precision."""
struct Gadget
    logB::Int
    ell::Int
    off::Int
end
Gadget(logB::Int, ell::Int) = Gadget(logB, ell, 0)
gvalue(G::Gadget, j::Int) = UInt64(1) << (64 - G.off - j * G.logB)

"Balanced decomposition: t ≈ Σ_j d_j 2^(64 - off - j logB) modulo 2^(64 - off), |d_j| ≤ B/2 (rounding to ℓ logB bits)."
function decompose(G::Gadget, t::Vector{UInt64})
    b = G.logB; ℓ = G.ell
    shift = 64 - G.off - b * ℓ
    shift >= 1 || error("ℓ·logB must be < 64")
    N = length(t)
    out = [Vector{Int64}(undef, N) for _ in 1:ℓ]
    mask = (UInt64(1) << b) - 1
    half = Int64(1) << (b - 1)
    full = Int64(1) << b
    @inbounds for i in 1:N
        x = t[i]
        y = (x >> shift) + ((x >> (shift - 1)) & 0x1)
        carry = Int64(0)
        for j in ℓ:-1:1
            d = Int64(y & mask) + carry
            y >>= b
            if d >= half
                d -= full
                carry = Int64(1)
            else
                carry = Int64(0)
            end
            out[j][i] = d
        end
    end
    return out
end

"""Representative of t that the digits of decompose(G, t) reconstruct up to the rounding: Σ_j d_j g_j = lift + ε,
|ε| ≤ 2^(63 - off - ℓ logB). It differs from t by a multiple of 2^-off (t itself for off = 0); after the division by B
of Algorithm 1, with off ≤ log2 B, it is another representative of the division (Lemma 4)."""
function gadget_lift(G::Gadget, t::Vector{UInt64})
    G.off == 0 && return t
    D = decompose(G, t); shift = 64 - G.off - G.logB * G.ell
    out = similar(t)
    @inbounds for i in eachindex(t)
        x = t[i]; y = (x >> shift) + ((x >> (shift - 1)) & 0x1)
        rec = UInt64(0)
        for j in 1:G.ell
            rec += reinterpret(UInt64, D[j][i]) * gvalue(G, j)
        end
        out[i] = x - (y << shift) + rec
    end
    return out
end

# ---------------------------------------------------------------- GLWE

struct GLWE
    a::Vector{UInt64}
    b::Vector{UInt64}
end
GLWE(N::Int) = GLWE(zeros(UInt64, N), zeros(UInt64, N))
Base.:+(x::GLWE, y::GLWE) = GLWE(x.a .+ y.a, x.b .+ y.b)
Base.:-(x::GLWE, y::GLWE) = GLWE(x.a .- y.a, x.b .- y.b)
Base.copy(x::GLWE) = GLWE(copy(x.a), copy(x.b))
trivial(m::Vector{UInt64}) = GLWE(zeros(UInt64, length(m)), copy(m))

function glwe_encrypt(r::Ring, sk::SecretKey, m::Vector{UInt64}, σ::Float64, rng)
    a = rand(rng, UInt64, r.N)
    b = mul_small_torus(r, sk.sntt, a)
    @inbounds for i in 1:r.N
        b[i] += m[i] + gauss_torus(rng, σ)
    end
    return GLWE(a, b)
end

glwe_phase(r::Ring, sk::SecretKey, c::GLWE) = c.b .- mul_small_torus(r, sk.sntt, c.a)

aut(c::GLWE, a::Integer) = GLWE(aut_poly(c.a, a), aut_poly(c.b, a))
mulX(c::GLWE, e::Integer) = GLWE(mulX(c.a, e), mulX(c.b, e))

"acc += Q · c for a sparse integer polynomial Q (exact, coefficient domain)."
function sparse_mul_add!(acc::GLWE, Q::Vector{Int64}, c::GLWE)
    @inbounds for e in 0:length(Q)-1
        q = Q[e+1]
        q == 0 && continue
        u = reinterpret(UInt64, q)
        addmulX!(acc.a, c.a, u, e)
        addmulX!(acc.b, c.b, u, e)
    end
    return acc
end

# ---------------------------------------------------------------- GGSW and external product

"GGSW(μ) in NTT form: rows 1..ℓ encrypt -μ s' g_j (mask rows), rows ℓ+1..2ℓ encrypt μ g_j."
struct FGGSW
    rows::Vector{Tuple{FTorus,FTorus}}
    G::Gadget
end

function ggsw_encrypt(r::Ring, sk::SecretKey, μ::Integer, G::Gadget, σ::Float64, rng)
    check_exact(r, 2 * G.ell, 2.0^(G.logB - 1))
    rows = Tuple{FTorus,FTorus}[]
    z = zeros(UInt64, r.N)
    for part in 1:2, j in 1:G.ell
        c = glwe_encrypt(r, sk, z, σ, rng)
        if part == 1
            c.a[1] += UInt64(μ) * gvalue(G, j)
        else
            c.b[1] += UInt64(μ) * gvalue(G, j)
        end
        push!(rows, (ftorus_key(r, c.a), ftorus_key(r, c.b)))
    end
    return FGGSW(rows, G)
end

"Dec(c) · GGSW(μ): an encryption of μ·(phase of c)."
function extprod(r::Ring, c::GLWE, K::FGGSW)
    G = K.G; ℓ = G.ell; N = r.N; P = r.plan.m.P
    acc = [zeros(UInt64, N) for _ in 1:4]
    for (part, comp) in enumerate((c.a, c.b))
        digits = decompose(G, comp)
        for j in 1:ℓ
            D = ntt_small(r, digits[j])
            ra, rb = K.rows[(part-1)*ℓ+j]
            mac_key!(acc[1], D, ra.lo, ra.lo_sh, P); mac_key!(acc[2], D, ra.hi, ra.hi_sh, P)
            mac_key!(acc[3], D, rb.lo, rb.lo_sh, P); mac_key!(acc[4], D, rb.hi, rb.hi_sh, P)
        end
    end
    return GLWE(torus_from_limbs(r, acc[1], acc[2]), torus_from_limbs(r, acc[3], acc[4]))
end

"CMux: c0 + Dec(c1 - c0)·GGSW(s) encrypts c1 if s = 1 and c0 if s = 0."
cmux(r::Ring, c0::GLWE, c1::GLWE, K::FGGSW) = c0 + extprod(r, c1 - c0, K)

# ---------------------------------------------------------------- automorphism key switching

"Key-switching key from τ_a(s') to s': rows encrypt τ_a(s') g_j under s'."
struct FKSKey
    a::Int
    perm::Vector{Int}
    rows::Vector{Tuple{FTorus,FTorus}}
    G::Gadget
end

function ks_keygen(r::Ring, sk::SecretKey, a::Integer, G::Gadget, σ::Float64, rng)
    check_exact(r, G.ell, 2.0^(G.logB - 1))
    τs = aut_poly(sk.s, a)
    rows = Tuple{FTorus,FTorus}[]
    for j in 1:G.ell
        msg = [reinterpret(UInt64, τs[i]) * gvalue(G, j) for i in 1:r.N]
        c = glwe_encrypt(r, sk, msg, σ, rng)
        push!(rows, (ftorus_key(r, c.a), ftorus_key(r, c.b)))
    end
    return FKSKey(Int(a), ntt_aut_perm(r.N, a), rows, G)
end

"Hoisted key switching: KS_τ(c) = (0, τ(b)) - Dec(τ(a))·K_τ for every key, with one decomposition of a."
function ks_hoisted(r::Ring, c::GLWE, keys::Vector{FKSKey})
    isempty(keys) && return GLWE[]
    G = keys[1].G; ℓ = G.ell; N = r.N; P = r.plan.m.P
    D = [ntt_small(r, d) for d in decompose(G, c.a)]
    out = GLWE[]
    for K in keys
        acc = [zeros(UInt64, N) for _ in 1:4]
        for j in 1:ℓ
            ra, rb = K.rows[j]
            mac_key!(acc[1], D[j], K.perm, ra.lo, ra.lo_sh, P); mac_key!(acc[2], D[j], K.perm, ra.hi, ra.hi_sh, P)
            mac_key!(acc[3], D[j], K.perm, rb.lo, rb.lo_sh, P); mac_key!(acc[4], D[j], K.perm, rb.hi, rb.hi_sh, P)
        end
        α = torus_from_limbs(r, acc[1], acc[2])
        β = torus_from_limbs(r, acc[3], acc[4])
        push!(out, GLWE(.-α, aut_poly(c.b, K.a) .- β))
    end
    return out
end
ks(r::Ring, c::GLWE, K::FKSKey) = ks_hoisted(r, c, [K])[1]

# ---------------------------------------------------------------- LWE

struct LWE
    a::Vector{UInt64}
    b::UInt64
end

"LWE encryption of the coefficient k (0-based) of the phase, under the key (s'_i)."
function extract(c::GLWE, k::Int)
    N = length(c.a)
    a = Vector{UInt64}(undef, N)
    @inbounds for i in 0:N-1
        a[i+1] = i <= k ? c.a[k-i+1] : -c.a[N+k-i+1]
    end
    return LWE(a, c.b[k+1])
end

function lwe_phase(c::LWE, s::AbstractVector{Int64})
    acc = c.b
    @inbounds for i in eachindex(s)
        acc -= c.a[i] * reinterpret(UInt64, s[i])
    end
    return acc
end

# ---------------------------------------------------------------- GLWE in exact NTT form

"GLWE in NTT form: the NTTs of its four limb polynomials (a_lo, a_hi, b_lo, b_hi), exact over Z."
struct NGLWE
    v::NTuple{4,Vector{UInt64}}
end
NGLWE(N::Int) = NGLWE(ntuple(_ -> zeros(UInt64, N), 4))

function nglwe(r::Ring, c::GLWE)
    fa = ftorus(r, c.a); fb = ftorus(r, c.b)
    return NGLWE((fa.lo, fa.hi, fb.lo, fb.hi))
end
to_glwe(r::Ring, x::NGLWE) =
    GLWE(torus_from_limbs(r, x.v[1], x.v[2]), torus_from_limbs(r, x.v[3], x.v[4]))

"acc += Q̂ .* x (pointwise in the NTT domain)."
function mac!(acc::NGLWE, Qh::Vector{UInt64}, x::NGLWE, m::Modulus)
    for i in 1:4
        mac!(acc.v[i], Qh, x.v[i], m)
    end
    return acc
end

"Hoisted key switching with outputs left in NTT form (nc = nglwe(r, c) provides the limbs of b)."
function ks_hoisted_ntt(r::Ring, c::GLWE, nc::NGLWE, keys::Vector{FKSKey})
    isempty(keys) && return NGLWE[]
    G = keys[1].G; ℓ = G.ell; N = r.N; P = r.plan.m.P
    D = [ntt_small(r, d) for d in decompose(G, c.a)]
    out = NGLWE[]
    for K in keys
        acc = [zeros(UInt64, N) for _ in 1:4]
        for j in 1:ℓ
            ra, rb = K.rows[j]
            mac_key!(acc[1], D[j], K.perm, ra.lo, ra.lo_sh, P); mac_key!(acc[2], D[j], K.perm, ra.hi, ra.hi_sh, P)
            mac_key!(acc[3], D[j], K.perm, rb.lo, rb.lo_sh, P); mac_key!(acc[4], D[j], K.perm, rb.hi, rb.hi_sh, P)
        end
        alo = [submod(UInt64(0), x, P) for x in acc[1]]
        ahi = [submod(UInt64(0), x, P) for x in acc[2]]
        blo = [submod(nc.v[3][K.perm[i]], acc[3][i], P) for i in 1:N]
        bhi = [submod(nc.v[4][K.perm[i]], acc[4][i], P) for i in 1:N]
        push!(out, NGLWE((alo, ahi, blo, bhi)))
    end
    return out
end
