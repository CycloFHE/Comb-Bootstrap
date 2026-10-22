# Product of two LWE ciphertexts (Section 6 of the paper), FFT backend (included in module Fast): the steps of src/product.jl,
# with the products in the Fourier domain. The key switchings of the trace use keys split in two 32-bit limbs (lifts
# lo + 2^32 hi), so that the Fourier products of the digits by each limb are rounded exactly: the trace, which multiplies
# the error of its key switchings by up to N/√3, is then exact, as in the exact backend. The tensor product uses 16-bit
# limbs, rounded exactly as well. Only the relinearization, whose error is not amplified, uses 64-bit keys.

import ..CombBoot: QM, QRLWE, qadd, qsub, qneg, qlift, qdivN, lwe_to_qrlwe, limbs16, scale_phat, ProductKeysRaw
export FProductKeys, PWork, flwe_product, ftrace, ftensor, frelin, flwe_to_rlwe_clean

struct FProductKeys
    tr::Vector{Vector{NTuple{4,Vector{ComplexF64}}}}    # tr[j][l], d = 2^j + 1: Fourier values of the limbs (a_lo, a_hi, b_lo, b_hi)
    rlk::Vector{Row}                # RLWE_{s'}(s'^2 g_l) mod 2^64
    Gtr::Gadget
    Grl::Gadget
end
"Signed 32-bit limbs of the centered lifts: lift = lo + 2^32 hi."
function qlimbs32(t::Vector{UInt64})
    lo = Vector{Int64}(undef, length(t)); hi = similar(lo)
    @inbounds for i in eachindex(t)
        v = qlift(t[i]); l = v & Int64(0xffffffff)
        l >= Int64(1) << 31 && (l -= Int64(1) << 32)
        lo[i] = l; hi[i] = (v - l) >> 32
    end
    return lo, hi
end
function _frow4(c::QRLWE, fp::FPlan)
    alo, ahi = qlimbs32(c.a); blo, bhi = qlimbs32(c.b)
    return (ffwd(alo, fp), ffwd(ahi, fp), ffwd(blo, fp), ffwd(bhi, fp))
end
FProductKeys(fp::FPlan, K::ProductKeysRaw) =
    FProductKeys([[_frow4(c, fp) for c in rows] for rows in K.tr], [frow(c, fp) for c in K.rlk], K.Gtr, K.Grl)

"Workspace of the product (FFT backend)."
struct PWork
    a::Vector{UInt64}; b::Vector{UInt64}; τa::Vector{UInt64}; τb::Vector{UInt64}
    α::Vector{UInt64}; β::Vector{UInt64}
    dig::Vector{Vector{Int64}}
    D::Vector{ComplexF64}; accA::Vector{ComplexF64}; accB::Vector{ComplexF64}
    acc4::NTuple{4,Vector{ComplexF64}}; x4::NTuple{4,Vector{UInt64}}
    F::Vector{Vector{ComplexF64}}               # 16 transforms of the limbs (A1, B1, A2, B2)
    acc::Vector{Vector{ComplexF64}}             # 21 accumulators: output o, weight w
    Pw::Vector{Vector{UInt64}}                  # 7 inverse transforms
end
function PWork(fp::FPlan, ℓmax::Int)
    N = fp.N; H = fp.H
    z() = zeros(UInt64, N); c() = zeros(ComplexF64, H)
    return PWork(z(), z(), z(), z(), z(), z(), [zeros(Int64, N) for _ in 1:ℓmax], c(), c(), c(),
                 (c(), c(), c(), c()), (z(), z(), z(), z()),
                 [c() for _ in 1:16], [c() for _ in 1:21], [z() for _ in 1:7])
end

"Balanced digits in place (as CombBoot.decompose)."
function decomp!(out::Vector{Vector{Int64}}, t::Vector{UInt64}, G::Gadget)
    b = G.logB; ℓ = G.ell; shift = 64 - b * ℓ
    mask = (UInt64(1) << b) - 1; half = Int64(1) << (b - 1); full = Int64(1) << b
    @inbounds for i in eachindex(t)
        x = t[i]
        y = (x >> shift) + ((x >> (shift - 1)) & 0x1)
        carry = Int64(0)
        for j in ℓ:-1:1
            d = Int64(y & mask) + carry
            y >>= b
            if d >= half
                d -= full; carry = Int64(1)
            else
                carry = Int64(0)
            end
            out[j][i] = d
        end
    end
    return out
end

"Residue modulo q' of a rounded real v = k 2^64 + r (2^64 ≡ 1 mod q')."
@inline function f2q(v::Float64)
    k = round(v / TWO64)
    x = Int128(round(v - TWO64 * k)) + Int128(unsafe_trunc(Int64, k))
    x < 0 && (x += Int128(QM))
    x >= Int128(QM) && (x -= Int128(QM))
    return UInt64(x)
end
function finv_q!(out::Vector{UInt64}, Y::Vector{ComplexF64}, fp::FPlan)
    H = fp.H; z = fp.buf
    copyto!(z, Y); fp.pi * z
    @inbounds for j in 1:H
        w = z[j] * fp.itw[j]
        out[j] = f2q(real(w)); out[j+H] = f2q(imag(w))
    end
    return out
end

"Residue modulo q' of lo + 2^32 hi (exact integers, |lo|, |hi| < 2^63)."
@inline function q_combine(lo::Int64, hi::Int64)
    v = Int128(lo) + (Int128(hi) << 32)
    x = (v >> 64) + (v & Int128(typemax(UInt64)))         # 2^64 ≡ 1
    while x < 0
        x += Int128(QM)
    end
    while x >= Int128(QM)
        x -= Int128(QM)
    end
    return UInt64(x)
end

function aut_q!(out::Vector{UInt64}, P::Vector{UInt64}, d::Int)
    N = length(P); mask = 2N - 1
    @inbounds for i in 0:N-1
        e = (i * d) & mask
        if e < N
            out[e+1] = P[i+1]
        else
            out[e-N+1] = qneg(P[i+1])
        end
    end
    return out
end

"Tr(N^{-1} c) modulo q' (tower order), in place in ws.a, ws.b."
function ftrace!(ws::PWork, c::QRLWE, K::FProductKeys, fp::FPlan)
    N = fp.N; H = fp.H; L = trailing_zeros(N); G = K.Gtr; ℓ = G.ell
    a = ws.a; b = ws.b
    @inbounds for i in 1:N
        a[i] = qdivN(c.a[i], L); b[i] = qdivN(c.b[i], L)
    end
    for j in L:-1:1
        d = (1 << j) + 1
        aut_q!(ws.τa, a, d); aut_q!(ws.τb, b, d)
        decomp!(ws.dig, ws.τa, G)
        A1, A2, B1, B2 = ws.acc4
        for x in ws.acc4
            fill!(x, 0)
        end
        for l in 1:ℓ
            ffwd!(ws.D, ws.dig[l], fp)
            k1, k2, k3, k4 = K.tr[j][l]; D = ws.D
            @inbounds @simd for k in 1:H
                A1[k] = muladd(D[k], k1[k], A1[k]); A2[k] = muladd(D[k], k2[k], A2[k])
                B1[k] = muladd(D[k], k3[k], B1[k]); B2[k] = muladd(D[k], k4[k], B2[k])
            end
        end
        for q in 1:4
            finv!(ws.x4[q], ws.acc4[q], fp)                # exact integers (< 2^53), read as Int64
        end
        alo, ahi, blo, bhi = ws.x4
        @inbounds for i in 1:N
            α = q_combine(reinterpret(Int64, alo[i]), reinterpret(Int64, ahi[i]))
            β = q_combine(reinterpret(Int64, blo[i]), reinterpret(Int64, bhi[i]))
            a[i] = qsub(a[i], α)
            b[i] = qadd(b[i], qsub(ws.τb[i], β))
        end
    end
    return ws
end
ftrace(ws::PWork, c::QRLWE, K::FProductKeys, fp::FPlan) = (ftrace!(ws, c, K, fp); QRLWE(copy(ws.a), copy(ws.b)))

"LWE_{s'}(μ), dimension N -> RLWE modulo 2^64 with phase μ + e + noise of the trace."
function flwe_to_rlwe_clean(ws::PWork, K::FProductKeys, c::LWE, fp::FPlan)
    ftrace!(ws, lwe_to_qrlwe(c), K, fp)
    return GLWE([x < (UInt64(1) << 63) ? x : x + 1 for x in ws.a], [x < (UInt64(1) << 63) ? x : x + 1 for x in ws.b])
end

"Tensor product (D0, D1, D2) of src/product.jl, with 16-bit limbs in the Fourier domain."
function ftensor(ws::PWork, c1::GLWE, c2::GLWE, phat::Integer, fp::FPlan)
    N = fp.N; H = fp.H; F = ws.F; acc = ws.acc
    for (q, t) in enumerate((c1.a, c1.b, c2.a, c2.b))
        X = limbs16(t)
        for u in 1:4
            ffwd!(F[4(q-1)+u], X[u], fp)
        end
    end
    for x in acc
        fill!(x, 0)
    end
    for u in 1:4, v in 1:4
        w = u + v - 1
        A1 = F[u]; B1 = F[4+u]; A2 = F[8+v]; B2 = F[12+v]
        A2u = F[8+u]; B2u = F[12+u]; A1v = F[v]
        o0 = acc[w]; o1 = acc[7+w]; o2 = acc[14+w]
        @inbounds @simd for k in 1:H
            o0[k] = muladd(B1[k], B2[k], o0[k])
            o1[k] = muladd(B1[k], A2[k], o1[k])
            o2[k] = muladd(A1[k], A2[k], o2[k])
        end
    end
    # D1 also needs b2·A1 = Σ B2_u A1_v: add the symmetric products
    for u in 1:4, v in 1:4
        w = u + v - 1
        B2 = F[12+u]; A1 = F[v]; o1 = acc[7+w]
        @inbounds @simd for k in 1:H
            o1[k] = muladd(B2[k], A1[k], o1[k])
        end
    end
    ph = UInt128(phat)
    outs = (Vector{UInt64}(undef, N), Vector{UInt64}(undef, N), Vector{UInt64}(undef, N))
    for o in 1:3
        for w in 1:7
            finv!(ws.Pw[w], acc[7(o-1)+w], fp)
        end
        out = outs[o]; Pw = ws.Pw
        @inbounds for i in 1:N
            out[i] = scale_phat(ntuple(w -> reinterpret(Int64, Pw[w][i]), 7), ph)
        end
    end
    return outs
end

"Relinearization (FFT): (D1 + Σ_l d_l rlk_l.a, D0 + Σ_l d_l rlk_l.b), d_l the digits of D2."
function frelin(ws::PWork, D0::Vector{UInt64}, D1::Vector{UInt64}, D2::Vector{UInt64}, K::FProductKeys, fp::FPlan)
    H = fp.H; G = K.Grl
    decomp!(ws.dig, D2, G)
    fill!(ws.accA, 0); fill!(ws.accB, 0)
    for l in 1:G.ell
        ffwd!(ws.D, ws.dig[l], fp)
        ra, rb = K.rlk[l]; D = ws.D; A = ws.accA; B = ws.accB
        @inbounds @simd for k in 1:H
            A[k] = muladd(D[k], ra[k], A[k]); B[k] = muladd(D[k], rb[k], B[k])
        end
    end
    finv!(ws.α, ws.accA, fp); finv!(ws.β, ws.accB, fp)
    return GLWE(D1 .+ ws.α, D0 .+ ws.β)
end

"Product of LWE_{s'}(μ_1) and LWE_{s'}(μ_2) (FFT backend): LWE_{s'}(p̂ μ_1 μ_2)."
function flwe_product(ws::PWork, K::FProductKeys, c1::LWE, c2::LWE, phat::Integer, fp::FPlan)
    t1 = flwe_to_rlwe_clean(ws, K, c1, fp); t2 = flwe_to_rlwe_clean(ws, K, c2, fp)
    D0, D1, D2 = ftensor(ws, t1, t2, phat, fp)
    return extract(frelin(ws, D0, D1, D2, K, fp), 0)
end
