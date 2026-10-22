# Product of two LWE ciphertexts of dimension N by a tensor product and a relinearization (Section 6 of the paper).
#
#  1. LWE -> RLWE: A(X) = a_0 - Σ_{j≥1} a_j X^{N-j}, so that [A s']_0 = a·s'; (A, b) has a phase Φ with Φ_0 = μ + e.
#  2. Trace: switch to the odd modulus q' = 2^64 - 1, product by N^{-1} = 2^{64 - log2 N} (a rotation of the 64 bits),
#     then c ← c + KS_d(τ_d(c)) for d = 2^j + 1, j = log2 N, ..., 1 [CDKS21]: Tr(N^{-1} Φ) = Φ_0 (mod q'); switch back.
#  3. Tensor product of the lifts in [-1/2, 1/2), times p̂: D0 - D1 s' + D2 s'^2 = p̂ Φ̃_1 Φ̃_2 (mod 1), relinearization
#     of D2 with RLWE_{s'}(s'^2 g_j), extraction of the constant coefficient: LWE_{s'}(p̂ μ_1 μ_2).
# All the products are exact: modulo q' through the 32-bit limbs of the centered lifts, modulo 2^64 as in ring.jl,
# and the tensor product through 16-bit limbs (NTT modulo P, then Σ_w 2^{16w} P_w modulo 2^128).

# ---------------------------------------------------------------- arithmetic modulo q' = 2^64 - 1

const QM = typemax(UInt64)          # q' = 2^64 - 1; residues in [0, q')

@inline qnorm(x::UInt64) = x == QM ? UInt64(0) : x
@inline qadd(a::UInt64, b::UInt64) = (s = a + b; s < a ? qnorm(s + 1) : qnorm(s))   # 2^64 ≡ 1
@inline qneg(x::UInt64) = x == 0 ? x : QM - x
@inline qsub(a::UInt64, b::UInt64) = qadd(a, qneg(b))
"Centered lift of a residue, in (-q'/2, q'/2)."
@inline qlift(x::UInt64) = x < (UInt64(1) << 63) ? reinterpret(Int64, x) : reinterpret(Int64, x) + 1
"Residue of an integer."
@inline function qfrom(v::Int128)
    r = v % Int128(QM)
    return r < 0 ? UInt64(r + Int128(QM)) : UInt64(r)
end
"Modulus switches 2^64 -> q' and q' -> 2^64 (rounding: at most half a unit)."
@inline to_q(x::UInt64) = x < (UInt64(1) << 63) ? x : x - 1
@inline from_q(x::UInt64) = x < (UInt64(1) << 63) ? x : x + 1
"Product by N^{-1} = 2^{64 - L} modulo 2^64 - 1 (N = 2^L): a rotation of the bits."
@inline qdivN(x::UInt64, L::Int) = bitrotate(x, 64 - L)
"Torus value of a residue."
q_to_float(x::UInt64) = qlift(x) / 18446744073709551615.0

"Automorphism X ↦ X^d on residues modulo q' (d odd)."
function aut_q(P::Vector{UInt64}, d::Integer)
    N = length(P); twoN = 2N
    out = Vector{UInt64}(undef, N)
    @inbounds for i in 0:N-1
        e = mod(i * Int(d), twoN)
        if e < N
            out[e+1] = P[i+1]
        else
            out[e-N+1] = qneg(P[i+1])
        end
    end
    return out
end

"NTT of the 32-bit limbs of the centered lifts (as ftorus: lo + 2^32 hi = lift, exactly)."
fq(r::Ring, t::Vector{UInt64}) = ftorus(r, [reinterpret(UInt64, qlift(x)) for x in t])
fq_key(r::Ring, t::Vector{UInt64}) = ftorus_key(r, [reinterpret(UInt64, qlift(x)) for x in t])

"Residues of the exact products, from the NTTs of the limb accumulators."
function q_from_limbs(r::Ring, acc_lo::Vector{UInt64}, acc_hi::Vector{UInt64})
    pl = r.plan; P = pl.m.P; N = r.N
    xlo = intt!(Vector{UInt64}(undef, N), acc_lo, pl)
    xhi = intt!(Vector{UInt64}(undef, N), acc_hi, pl)
    return [qfrom(Int128(from_modP(xlo[i], P)) + (Int128(from_modP(xhi[i], P)) << 32)) for i in 1:N]
end

"Exact product s·t modulo q', s small integer polynomial (given by its NTT)."
function mul_small_q(r::Ring, sntt::Vector{UInt64}, t::Vector{UInt64})
    ft = fq(r, t); m = r.plan.m
    lo = zeros(UInt64, r.N); hi = zeros(UInt64, r.N)
    mac!(lo, sntt, ft.lo, m); mac!(hi, sntt, ft.hi, m)
    return q_from_limbs(r, lo, hi)
end

# ---------------------------------------------------------------- RLWE modulo q'

struct QRLWE
    a::Vector{UInt64}
    b::Vector{UInt64}
end
Base.:+(x::QRLWE, y::QRLWE) = QRLWE(qadd.(x.a, y.a), qadd.(x.b, y.b))

gauss_q(rng, σ::Float64) = qfrom(Int128(round(Int64, randn(rng) * σ * 2.0^64)))

function qrlwe_encrypt(r::Ring, sk::SecretKey, m::Vector{UInt64}, σ::Float64, rng)
    a = [qnorm(x) for x in rand(rng, UInt64, r.N)]
    as = mul_small_q(r, sk.sntt, a)
    return QRLWE(a, [qadd(qadd(as[i], m[i]), gauss_q(rng, σ)) for i in 1:r.N])
end
q_phase(r::Ring, sk::SecretKey, c::QRLWE) = qsub.(c.b, mul_small_q(r, sk.sntt, c.a))

"LWE (modulus 2^64, dimension N) -> RLWE modulo q': A(X) = a_0 - Σ_{j≥1} a_j X^{N-j}, b constant."
function lwe_to_qrlwe(c::LWE)
    N = length(c.a)
    A = zeros(UInt64, N); A[1] = to_q(c.a[1])
    @inbounds for j in 1:N-1
        A[N-j+1] = qneg(to_q(c.a[j+1]))
    end
    b = zeros(UInt64, N); b[1] = to_q(c.b)
    return QRLWE(A, b)
end
to_glwe_64(c::QRLWE) = GLWE(from_q.(c.a), from_q.(c.b))

# ---------------------------------------------------------------- keys

"Raw keys (residues and torus polynomials), shared by the exact and FFT backends."
struct ProductKeysRaw
    tr::Vector{Vector{QRLWE}}       # tr[j][l]: RLWE_{s'}(τ_d(s') g_l) mod q', d = 2^j + 1
    rlk::Vector{GLWE}               # rlk[l]: RLWE_{s'}(s'^2 g_l) mod 2^64
    Gtr::Gadget
    Grl::Gadget
end

"s'^2 in Z[X]/(X^N+1) (exact)."
function square_key(r::Ring, sk::SecretKey)
    m = r.plan.m; P = m.P
    y = [mulmod(x, x, m) for x in sk.sntt]
    return [from_modP(v, P) for v in intt!(similar(y), y, r.plan)]
end

function product_keys_raw(r::Ring, sk::SecretKey, Gtr::Gadget, Grl::Gadget, σ::Float64, rng)
    L = trailing_zeros(r.N)
    tr = Vector{Vector{QRLWE}}()
    for j in 1:L
        τs = aut_poly(sk.s, (1 << j) + 1)
        push!(tr, [qrlwe_encrypt(r, sk, [qfrom(Int128(τs[i]) * gvalue(Gtr, l)) for i in 1:r.N], σ, rng)
                   for l in 1:Gtr.ell])
    end
    s2 = square_key(r, sk)
    rlk = [glwe_encrypt(r, sk, [reinterpret(UInt64, s2[i]) * gvalue(Grl, l) for i in 1:r.N], σ, rng) for l in 1:Grl.ell]
    return ProductKeysRaw(tr, rlk, Gtr, Grl)
end

"Keys of the exact backend: limbs in NTT form."
struct ProductKeys
    tr::Vector{Vector{Tuple{FTorus,FTorus}}}
    rlk::Vector{Tuple{FTorus,FTorus}}
    Gtr::Gadget
    Grl::Gadget
end
function ProductKeys(r::Ring, K::ProductKeysRaw)
    check_exact(r, K.Gtr.ell, 2.0^(K.Gtr.logB - 1)); check_exact(r, K.Grl.ell, 2.0^(K.Grl.logB - 1))
    tr = [[(fq_key(r, c.a), fq_key(r, c.b)) for c in rows] for rows in K.tr]
    rlk = [(ftorus_key(r, c.a), ftorus_key(r, c.b)) for c in K.rlk]
    return ProductKeys(tr, rlk, K.Gtr, K.Grl)
end

# ---------------------------------------------------------------- trace

"KS_d(τ_d(c)) modulo q': from τ_d(s') back to s' (d = 2^j + 1, key rows rows)."
function qks_aut(r::Ring, c::QRLWE, d::Int, rows::Vector{Tuple{FTorus,FTorus}}, G::Gadget)
    N = r.N; P = r.plan.m.P
    τa = aut_q(c.a, d); τb = aut_q(c.b, d)
    digits = decompose(G, τa)            # Σ_l d_l g_l ≡ τa (mod q') up to rounding (2^64 ≡ 1)
    acc = [zeros(UInt64, N) for _ in 1:4]
    for l in 1:G.ell
        D = ntt_small(r, digits[l]); ra, rb = rows[l]
        mac_key!(acc[1], D, ra.lo, ra.lo_sh, P); mac_key!(acc[2], D, ra.hi, ra.hi_sh, P)
        mac_key!(acc[3], D, rb.lo, rb.lo_sh, P); mac_key!(acc[4], D, rb.hi, rb.hi_sh, P)
    end
    α = q_from_limbs(r, acc[1], acc[2]); β = q_from_limbs(r, acc[3], acc[4])
    return QRLWE(qneg.(α), qsub.(τb, β))
end

"Tr(N^{-1} c) modulo q' (tower order d = N+1, N/2+1, ..., 3)."
function qtrace(r::Ring, c::QRLWE, K::ProductKeys)
    L = trailing_zeros(r.N)
    c = QRLWE(qdivN.(c.a, L), qdivN.(c.b, L))
    for j in L:-1:1
        c = c + qks_aut(r, c, (1 << j) + 1, K.tr[j], K.Gtr)
    end
    return c
end

"LWE_{s'}(μ) of dimension N -> RLWE_{s'} modulo 2^64 whose phase is μ + e + (noise of the trace)."
lwe_to_rlwe_clean(r::Ring, K::ProductKeys, c::LWE) = to_glwe_64(qtrace(r, lwe_to_qrlwe(c), K))

# ---------------------------------------------------------------- tensor product and relinearization

"Signed 16-bit limbs of the lifts in [-2^63, 2^63): t = Σ_{u=0}^{3} 2^{16u} x_u, |x_u| ≤ 2^15 + 1."
function limbs16(t::Vector{UInt64})
    N = length(t)
    X = [Vector{Int64}(undef, N) for _ in 1:4]
    @inbounds for i in 1:N
        v = Int128(reinterpret(Int64, t[i]))
        for u in 1:3
            l = Int64(v & 0xffff)
            l >= 0x8000 && (l -= 0x10000)
            X[u][i] = l
            v = (v - l) >> 16                 # exact: v - l ≡ 0 (mod 2^16)
        end
        X[4][i] = Int64(v)
    end
    return X
end

"round(p̂ V / 2^64) mod 2^64 from V = Σ_w 2^{16w} P_w (only V mod 2^128 matters)."
@inline function scale_phat(Pw::NTuple{7,Int64}, phat::UInt128)
    V = zero(UInt128)
    for w in 0:6
        V += reinterpret(UInt128, Int128(Pw[w+1])) << (16w)
    end
    return UInt64((phat * V + (UInt128(1) << 63)) >> 64)
end

"""Tensor product of two RLWE ciphertexts modulo 2^64 (lifts in [-1/2, 1/2)): D0 = p̂ b̃1 b̃2, D1 = p̂ (b̃1 Ã2 + b̃2 Ã1),
D2 = p̂ Ã1 Ã2, so that D0 - D1 s' + D2 s'^2 = p̂ Φ̃1 Φ̃2 (mod 1)."""
function tensor(r::Ring, c1::GLWE, c2::GLWE, phat::Integer)
    N = r.N; m = r.plan.m; P = m.P
    log2(8N) + 2log2(2.0^15 + 1) < log2(Float64(P)) - 1.5 || error("tensor product: NTT overflow")   # ≤ 8 terms of N·2^30
    F(t) = [ntt_small(r, x) for x in limbs16(t)]
    A1, B1, A2, B2 = F(c1.a), F(c1.b), F(c2.a), F(c2.b)
    acc = [[zeros(UInt64, N) for _ in 0:6] for _ in 1:3]
    for u in 1:4, v in 1:4
        w = u + v - 1
        mac!(acc[1][w], B1[u], B2[v], m)
        mac!(acc[2][w], B1[u], A2[v], m); mac!(acc[2][w], B2[u], A1[v], m)
        mac!(acc[3][w], A1[u], A2[v], m)
    end
    ph = UInt128(phat)
    out = Vector{Vector{UInt64}}()
    for o in 1:3
        xs = [intt!(Vector{UInt64}(undef, N), acc[o][w], r.plan) for w in 1:7]
        push!(out, [scale_phat(ntuple(w -> from_modP(xs[w][i], P), 7), ph) for i in 1:N])
    end
    return out[1], out[2], out[3]
end

"Relinearization: (D1 + Σ_l d_l rlk_l.a, D0 + Σ_l d_l rlk_l.b), d_l the digits of D2; phase D0 - D1 s' + D2 s'^2."
function relin(r::Ring, D0::Vector{UInt64}, D1::Vector{UInt64}, D2::Vector{UInt64}, K::ProductKeys)
    N = r.N; P = r.plan.m.P; G = K.Grl
    digits = decompose(G, D2)
    acc = [zeros(UInt64, N) for _ in 1:4]
    for l in 1:G.ell
        D = ntt_small(r, digits[l]); ra, rb = K.rlk[l]
        mac_key!(acc[1], D, ra.lo, ra.lo_sh, P); mac_key!(acc[2], D, ra.hi, ra.hi_sh, P)
        mac_key!(acc[3], D, rb.lo, rb.lo_sh, P); mac_key!(acc[4], D, rb.hi, rb.hi_sh, P)
    end
    return GLWE(D1 .+ torus_from_limbs(r, acc[1], acc[2]), D0 .+ torus_from_limbs(r, acc[3], acc[4]))
end

"""
    lwe_product(r, K, c1, c2, phat) -> LWE

Product of LWE_{s'}(μ_1) and LWE_{s'}(μ_2) (dimension N, μ_i ∈ (1/p̂)Z): an LWE_{s'} encryption of p̂ μ_1 μ_2.
"""
function lwe_product(r::Ring, K::ProductKeys, c1::LWE, c2::LWE, phat::Integer)
    t1 = lwe_to_rlwe_clean(r, K, c1); t2 = lwe_to_rlwe_clean(r, K, c2)
    D0, D1, D2 = tensor(r, t1, t2, phat)
    return extract(relin(r, D0, D1, D2, K), 0)
end

# ---------------------------------------------------------------- noise model

"Variance of one automorphism key switching (binary key, ρ = 1/2): key term and rounding term."
var_ks_gadget(N::Int, G::Gadget, σ::Float64) =
    G.ell * N * (2.0^(2G.logB) + 2) / 12 * σ^2 + (N / 2) * 2.0^(-2G.ell * G.logB) / 12

"""Multipliers of the key-switching variance in each coefficient after the trace (tower order, independent
noises): the noise added at step j goes through the partial trace of the steps j-1, ..., 1."""
function trace_profile(N::Int)
    L = trailing_zeros(N); twoN = 2N
    prof = zeros(Float64, N)
    cnt = zeros(Int, N)
    for j in L:-1:1
        S = [1]
        for i in 1:j-1
            S = vcat(S, [mod(a * ((1 << i) + 1), twoN) for a in S])
        end
        for k in 0:N-1
            for a in S
                e = mod(a * k, twoN)
                e < N ? (cnt[e+1] += 1) : (cnt[e-N+1] -= 1)
            end
            for a in S
                e = mod(a * k, twoN); t = e < N ? e : e - N
                if cnt[t+1] != 0
                    prof[t+1] += cnt[t+1]^2; cnt[t+1] = 0
                end
            end
        end
    end
    return prof
end

"""Predicted variance of the product: p̂^2 (N/24 + 1/12) Σ_i (σ_i^2 + T_i) + (p̂^2/12) Σ_i (σ_i^2 + V_{i,0}) + relinearization,
T_i the total variance of the trace noise of input i (all coefficients), V_{i,0} its constant coefficient."""
function var_product(N::Int, phat::Integer, σe2::Float64, Ttr::Float64, V0::Float64, Grl::Gadget, σ::Float64)
    rl = Grl.ell * N * (2.0^(2Grl.logB) + 2) / 12 * σ^2 + N * 2.0^(-2Grl.ell * Grl.logB) / 12 * N^2 / 48
    return phat^2 * (N / 24 + 1 / 12) * 2 * (σe2 + Ttr) + phat^2 / 12 * 2 * (σe2 + V0) + rl
end
