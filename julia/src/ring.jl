# Torus T = R/Z represented by UInt64 (q = 2^64), polynomials of Z[X]/(X^N+1) and T[X]/(X^N+1).
# A ciphertext component a ∈ T^∨ = K/R^∨ is stored by its coordinates on Ω*_i = X^i/N, i.e. by the
# torus polynomial N·a (case (B): R^∨ = R/N). Products by integer polynomials are computed exactly
# modulo 2^64: the torus polynomial is split into two signed 32-bit limbs, each multiplied exactly over Z
# with the NTT (|result| < P/2 is checked when keys are built).

const TWO64 = 18446744073709551616.0

torus_to_float(t::UInt64) = reinterpret(Int64, t) / TWO64

"Torus element num/den mod 1, rounded to 2^-64."
function torus_frac(num::Integer, den::Integer)
    v = round(BigInt, (big(num) << 64) // big(den))
    return UInt64(mod(v, big(2)^64))
end

gauss_torus(rng, σ::Float64) = reinterpret(UInt64, round(Int64, randn(rng) * σ * TWO64))

# ---------------------------------------------------------------- ring context

struct Ring
    N::Int
    plan::NTTPlan
end

const _PRIME = Ref{UInt64}(0)
function Ring(N::Int)
    _PRIME[] == 0 && (_PRIME[] = find_ntt_prime())
    return Ring(N, NTTPlan(N, Modulus(_PRIME[])))
end

# ---------------------------------------------------------------- plain polynomial operations

"X^e · P in Z[X]/(X^N+1) or T[X]/(X^N+1)."
function mulX(P::Vector{T}, e::Integer) where {T}
    N = length(P)
    e = mod(Int(e), 2N)
    out = Vector{T}(undef, N)
    @inbounds for i in 0:N-1
        j = i + e
        if j < N
            out[j+1] = P[i+1]
        elseif j < 2N
            out[j-N+1] = -P[i+1]
        else
            out[j-2N+1] = P[i+1]
        end
    end
    return out
end

"dst += c · X^e · src (torus polynomials, c a wrapping UInt64 scalar, 0 ≤ e < 2N)."
function addmulX!(dst::Vector{UInt64}, src::Vector{UInt64}, c::UInt64, e::Int)
    N = length(src)
    if e >= N
        e -= N
        c = -c
    end
    @inbounds for i in 0:N-1
        j = i + e
        if j < N
            dst[j+1] += c * src[i+1]
        else
            dst[j-N+1] -= c * src[i+1]
        end
    end
    return dst
end

"Automorphism τ_a: P(X) ↦ P(X^a) in Z[X]/(X^N+1) (a odd)."
function aut_poly(P::Vector{T}, a::Integer) where {T}
    N = length(P)
    twoN = 2N
    a = mod(Int(a), twoN)
    out = zeros(T, N)
    @inbounds for i in 0:N-1
        e = mod(i * a, twoN)
        if e < N
            out[e+1] += P[i+1]
        else
            out[e-N+1] -= P[i+1]
        end
    end
    return out
end

# ---------------------------------------------------------------- exact products

"NTT of the two signed 32-bit limbs of a torus polynomial: t ≡ lo + 2^32 hi, |lo|, |hi| ≤ 2^31."
struct FTorus
    lo::Vector{UInt64}
    hi::Vector{UInt64}
    lo_sh::Vector{UInt64}   # Shoup companions (keys only; empty otherwise)
    hi_sh::Vector{UInt64}
end
FTorus(lo, hi) = FTorus(lo, hi, UInt64[], UInt64[])

"Key polynomial: limbs in NTT form with Shoup companions (fast products by constants)."
function ftorus_key(r::Ring, t::Vector{UInt64})
    f = ftorus(r, t); P = r.plan.m.P
    return FTorus(f.lo, f.hi, [shoup(x, P) for x in f.lo], [shoup(x, P) for x in f.hi])
end

"acc += D .* K for a key limb K with Shoup companion Ksh (optionally with D permuted)."
function mac_key!(acc::Vector{UInt64}, D::Vector{UInt64}, K::Vector{UInt64}, Ksh::Vector{UInt64}, P::UInt64)
    @inbounds for i in eachindex(acc)
        acc[i] = addmod(acc[i], mulmod_shoup(D[i], K[i], Ksh[i], P), P)
    end
    return acc
end
function mac_key!(acc::Vector{UInt64}, D::Vector{UInt64}, perm::Vector{Int}, K::Vector{UInt64},
                  Ksh::Vector{UInt64}, P::UInt64)
    @inbounds for i in eachindex(acc)
        acc[i] = addmod(acc[i], mulmod_shoup(D[perm[i]], K[i], Ksh[i], P), P)
    end
    return acc
end

function ftorus(r::Ring, t::Vector{UInt64})
    N = r.N; P = r.plan.m.P
    lo = Vector{UInt64}(undef, N); hi = Vector{UInt64}(undef, N)
    @inbounds for i in 1:N
        l = Int64(t[i] & 0xffffffff)
        l >= (Int64(1) << 31) && (l -= Int64(1) << 32)
        h = Int64((t[i] - reinterpret(UInt64, l)) >> 32)
        h >= (Int64(1) << 31) && (h -= Int64(1) << 32)
        lo[i] = to_modP(l, P)
        hi[i] = to_modP(h, P)
    end
    return FTorus(ntt!(similar(lo), lo, r.plan), ntt!(similar(hi), hi, r.plan))
end

"NTT of a small integer polynomial."
function ntt_small(r::Ring, d::Vector{Int64})
    P = r.plan.m.P
    x = Vector{UInt64}(undef, r.N)
    @inbounds for i in 1:r.N
        x[i] = to_modP(d[i], P)
    end
    return ntt!(similar(x), x, r.plan)
end

"acc += D .* K (mod P); with perm: acc[i] += D[perm[i]] * K[i]."
function mac!(acc::Vector{UInt64}, D::Vector{UInt64}, K::Vector{UInt64}, m::Modulus)
    P = m.P
    @inbounds for i in eachindex(acc)
        acc[i] = addmod(acc[i], mulmod(D[i], K[i], m), P)
    end
    return acc
end
function mac!(acc::Vector{UInt64}, D::Vector{UInt64}, perm::Vector{Int}, K::Vector{UInt64}, m::Modulus)
    P = m.P
    @inbounds for i in eachindex(acc)
        acc[i] = addmod(acc[i], mulmod(D[perm[i]], K[i], m), P)
    end
    return acc
end

"Torus polynomial from the NTT of the exact products with the two limbs."
function torus_from_limbs(r::Ring, acc_lo::Vector{UInt64}, acc_hi::Vector{UInt64})
    pl = r.plan; P = pl.m.P; N = r.N
    xlo = intt!(Vector{UInt64}(undef, N), acc_lo, pl)
    xhi = intt!(Vector{UInt64}(undef, N), acc_hi, pl)
    out = Vector{UInt64}(undef, N)
    @inbounds for i in 1:N
        out[i] = reinterpret(UInt64, from_modP(xlo[i], P)) +
                 (reinterpret(UInt64, from_modP(xhi[i], P)) << 32)
    end
    return out
end

"Exact product s·t mod 2^64, s small integer polynomial (given by its NTT), t torus polynomial."
function mul_small_torus(r::Ring, sntt::Vector{UInt64}, t::Vector{UInt64})
    ft = ftorus(r, t)
    m = r.plan.m
    lo = zeros(UInt64, r.N); hi = zeros(UInt64, r.N)
    mac!(lo, sntt, ft.lo, m); mac!(hi, sntt, ft.hi, m)
    return torus_from_limbs(r, lo, hi)
end

"Schoolbook negacyclic product (tests only)."
function mul_schoolbook(s::Vector{Int64}, t::Vector{UInt64})
    N = length(t)
    out = zeros(UInt64, N)
    for i in 0:N-1, j in 0:N-1
        c = reinterpret(UInt64, s[i+1]) * t[j+1]
        k = i + j
        if k < N
            out[k+1] += c
        else
            out[k-N+1] -= c
        end
    end
    return out
end

"Bound check: rows × N × (digit bound) × 2^31 must stay below P/2."
function check_exact(r::Ring, rows::Int, digit_bound::Real)
    lhs = log2(rows) + log2(r.N) + log2(digit_bound) + 31
    lhs < log2(Float64(r.plan.m.P)) - 1.5 ||
        error("exact NTT product may overflow: 2^$(round(lhs, digits=1)) ≥ P/2")
    return nothing
end
