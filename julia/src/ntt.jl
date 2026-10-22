# Exact negacyclic NTT over Z_P[X]/(X^N+1), with P a 62-bit prime, P ≡ 1 (mod 2^18).
# Evaluation order: Y[k] = P(ψ^(2k+1)), k = 0..N-1, with ψ a primitive 2N-th root of unity mod P.
# In this order, the automorphism X ↦ X^a is a permutation of the evaluation points.

const NTT_TWO_ADICITY = 18          # supports N ≤ 2^17

# ---------------------------------------------------------------- modular arithmetic

struct Modulus
    P::UInt64
    mu::UInt64          # floor(2^124 / P), P ∈ (2^61, 2^62)
end
Modulus(P::UInt64) = Modulus(P, UInt64(div(UInt128(1) << 124, UInt128(P))))

# Barrett reduction of a*b, a, b < P < 2^62.
@inline function mulmod(a::UInt64, b::UInt64, m::Modulus)
    x = widemul(a, b)
    q = UInt64((UInt128(UInt64(x >> 60)) * m.mu) >> 64)
    r = UInt64(x - widemul(q, m.P))
    while r >= m.P
        r -= m.P
    end
    return r
end
"Shoup multiplication by a constant w < P with wsh = floor(w 2^64 / P); x < 2^64."
@inline function mulmod_shoup(x::UInt64, w::UInt64, wsh::UInt64, P::UInt64)
    q = UInt64(widemul(x, wsh) >> 64)
    r = x * w - q * P
    return r >= P ? r - P : r
end
shoup(w::UInt64, P::UInt64) = UInt64(div(UInt128(w) << 64, UInt128(P)))

@inline addmod(a::UInt64, b::UInt64, P::UInt64) = (s = a + b; s >= P ? s - P : s)
@inline submod(a::UInt64, b::UInt64, P::UInt64) = a >= b ? a - b : a + (P - b)

function powmod(b::UInt64, e::Integer, m::Modulus)
    r = UInt64(1)
    b %= m.P
    e = UInt64(e)
    while e > 0
        if e & 0x1 == 0x1
            r = mulmod(r, b, m)
        end
        b = mulmod(b, b, m)
        e >>= 1
    end
    return r
end

@inline to_modP(d::Int64, P::UInt64) = d >= 0 ? UInt64(d) : P - UInt64(-d)
@inline from_modP(v::UInt64, P::UInt64) = v > (P >> 1) ? -Int64(P - v) : Int64(v)

# ---------------------------------------------------------------- prime search

function _mr_pow(b::UInt64, e::UInt64, n::UInt64)
    r = UInt64(1); b %= n
    while e > 0
        if e & 0x1 == 0x1
            r = UInt64(widemul(r, b) % n)
        end
        b = UInt64(widemul(b, b) % n)
        e >>= 1
    end
    return r
end

function is_prime_u64(n::UInt64)
    n < 2 && return false
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        n % UInt64(q) == 0 && return n == UInt64(q)
    end
    d = n - 1; s = 0
    while iseven(d)
        d >>= 1; s += 1
    end
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        x = _mr_pow(UInt64(a), d, n)
        (x == 1 || x == n - 1) && continue
        composite = true
        for _ in 1:s-1
            x = UInt64(widemul(x, x) % n)
            if x == n - 1
                composite = false
                break
            end
        end
        composite && return false
    end
    return true
end

"Largest prime P < 2^62 with P ≡ 1 (mod 2^NTT_TWO_ADICITY)."
function find_ntt_prime()
    k = (UInt64(1) << 62 - 1) >> NTT_TWO_ADICITY
    while true
        P = (k << NTT_TWO_ADICITY) + 1
        P > (UInt64(1) << 61) || error("no NTT prime found")
        is_prime_u64(P) && return P
        k -= 1
    end
end

# ---------------------------------------------------------------- NTT plan

struct NTTPlan
    N::Int
    m::Modulus
    psi::Vector{UInt64}     # ψ^i, i = 0..N-1
    ipsi::Vector{UInt64}    # N^{-1} ψ^{-i}
    w::Vector{UInt64}       # ω^j, j = 0..N/2-1, ω = ψ^2
    iw::Vector{UInt64}      # ω^{-j}
    rev::Vector{Int}        # bit reversal (1-based)
    psi_sh::Vector{UInt64}  # Shoup companions
    ipsi_sh::Vector{UInt64}
    w_sh::Vector{UInt64}
    iw_sh::Vector{UInt64}
end

function NTTPlan(N::Int, m::Modulus)
    ispow2(N) && N >= 2 || error("N must be a power of 2")
    2N <= (1 << NTT_TWO_ADICITY) || error("N too large for this prime")
    P = m.P
    g = UInt64(2); psi = UInt64(0)
    while true
        x = powmod(g, (P - 1) ÷ UInt64(2N), m)
        if powmod(x, N, m) == P - 1
            psi = x
            break
        end
        g += 1
    end
    ipsi1 = powmod(psi, 2N - 1, m)
    invN = powmod(UInt64(N), P - 2, m)
    psis = Vector{UInt64}(undef, N); ipsis = Vector{UInt64}(undef, N)
    a = UInt64(1); b = invN
    for i in 1:N
        psis[i] = a; ipsis[i] = b
        a = mulmod(a, psi, m); b = mulmod(b, ipsi1, m)
    end
    om = mulmod(psi, psi, m); iom = mulmod(ipsi1, ipsi1, m)
    w = Vector{UInt64}(undef, N ÷ 2); iw = Vector{UInt64}(undef, N ÷ 2)
    a = UInt64(1); b = UInt64(1)
    for j in 1:N÷2
        w[j] = a; iw[j] = b
        a = mulmod(a, om, m); b = mulmod(b, iom, m)
    end
    L = trailing_zeros(N)
    rev = [bitreverse(UInt64(i)) >> (64 - L) + 1 for i in 0:N-1]
    sh(v) = [shoup(x, P) for x in v]
    return NTTPlan(N, m, psis, ipsis, w, iw, Int.(rev), sh(psis), sh(ipsis), sh(w), sh(iw))
end

function _butterflies!(y::Vector{UInt64}, tw::Vector{UInt64}, twsh::Vector{UInt64}, pl::NTTPlan)
    N = pl.N; P = pl.m.P
    len = 2
    @inbounds while len <= N
        half = len >> 1
        stride = N ÷ len
        for start in 1:len:N
            for j in 0:half-1
                idx = j * stride + 1
                u = y[start+j]
                v = mulmod_shoup(y[start+j+half], tw[idx], twsh[idx], P)
                y[start+j] = addmod(u, v, P)
                y[start+j+half] = submod(u, v, P)
            end
        end
        len <<= 1
    end
    return y
end

"Forward negacyclic NTT: y[k+1] = x(ψ^(2k+1)); x has entries < P; y ≠ x."
function ntt!(y::Vector{UInt64}, x::AbstractVector{UInt64}, pl::NTTPlan)
    P = pl.m.P
    @inbounds for i in 1:pl.N
        y[pl.rev[i]] = mulmod_shoup(x[i], pl.psi[i], pl.psi_sh[i], P)
    end
    return _butterflies!(y, pl.w, pl.w_sh, pl)
end

"Inverse negacyclic NTT; x ≠ y."
function intt!(x::Vector{UInt64}, y::AbstractVector{UInt64}, pl::NTTPlan)
    P = pl.m.P
    @inbounds for i in 1:pl.N
        x[pl.rev[i]] = y[i]
    end
    _butterflies!(x, pl.iw, pl.iw_sh, pl)
    @inbounds for i in 1:pl.N
        x[i] = mulmod_shoup(x[i], pl.ipsi[i], pl.ipsi_sh[i], P)
    end
    return x
end

"Permutation of the evaluation points induced by X ↦ X^a (a odd): (τ_a Y)[k] = Y[perm[k]]."
function ntt_aut_perm(N::Int, a::Integer)
    twoN = 2N
    perm = Vector{Int}(undef, N)
    for k in 0:N-1
        c = mod(Int(a) * (2k + 1), twoN)
        perm[k+1] = (c - 1) ÷ 2 + 1
    end
    return perm
end
