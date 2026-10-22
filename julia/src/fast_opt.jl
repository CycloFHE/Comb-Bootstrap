# Optimized FFT engine (included in module Fast).
#  - preallocated workspace `Work` (built once from the keys);
#  - balanced digits by the offset trick (no carries, vectorized), written as twisted FFT inputs;
#  - external products on split (real / imaginary) keys;
#  - "T-layout" of the Fourier values for the baby and giant steps (see below): the automorphisms become
#    rotations of contiguous rows and the public polynomials of c_3 (which lie in Z[X^M0]) row scalings;
#  - key switching with pre-permuted keys, τ(D)·K = τ(D·τ⁻¹(K)); baby steps and public products computed row
#    by row, keys packed in reading order; table of c_3 by B-point FFTs;
#  - threads can cooperate inside each iteration.

# ---------------------------------------------------------------- split complex vectors

struct SV
    r::Vector{Float64}
    i::Vector{Float64}
end
SV(H::Int) = SV(zeros(H), zeros(H))

function split!(s::SV, Y::Vector{ComplexF64})
    r = s.r; i = s.i
    @inbounds @simd for k in eachindex(Y)
        r[k] = real(Y[k]); i[k] = imag(Y[k])
    end
    return s
end
SV(Y::Vector{ComplexF64}) = split!(SV(length(Y)), Y)
const SRow = NTuple{2,SV}

# ---------------------------------------------------------------- packing, digits

@inline sval(t::UInt64) = Float64(reinterpret(Int64, t))

"Y = Fourier values of the torus polynomial t."
function fwd_torus!(Y::Vector{ComplexF64}, t::Vector{UInt64}, fp::FPlan)
    H = fp.H; tw = fp.tw
    @inbounds @simd for k in 1:H
        x1 = sval(t[k]); x2 = sval(t[k+H]); w = tw[k]
        Y[k] = complex(x1 * real(w) - x2 * imag(w), x1 * imag(w) + x2 * real(w))
    end
    fp.pf * Y
    return Y
end

"out (=|+=) torus polynomial with Fourier values Y; Y is destroyed."
function inv_torus!(out::Vector{UInt64}, Y::Vector{ComplexF64}, fp::FPlan, add::Bool)
    H = fp.H; itw = fp.itw
    fp.pi * Y
    if add
        @inbounds @simd for k in 1:H
            w = Y[k] * itw[k]
            out[k] += f2t(real(w)); out[k+H] += f2t(imag(w))
        end
    else
        @inbounds @simd for k in 1:H
            w = Y[k] * itw[k]
            out[k] = f2t(real(w)); out[k+H] = f2t(imag(w))
        end
    end
    return out
end

"Constants of the balanced digits (offset trick): x + C, then digit j = ((x + C) >> (64 - off - j·logB)) & mask - half."
function _dconst(G::Gadget)
    b = G.logB; ℓ = G.ell; shift = 64 - G.off - b * ℓ
    h = UInt64(0)
    for j in 0:ℓ-1
        h += (UInt64(1) << (b - 1)) << (b * j)
    end
    return (UInt64(1) << (shift - 1)) + (h << shift), (UInt64(1) << b) - 1, Float64(1 << (b - 1))
end

"Dj = Fourier values of the j-th balanced digit of x - x0 (x0 = nothing: of x), rounded to the top ℓ·logB bits.
Offset trick: the digits of x + Σ_j (B/2)·B^j, minus B/2, lie in [-B/2, B/2) and need no carry."
function digit_fwd!(Dj::Vector{ComplexF64}, x::Vector{UInt64}, x0, G::Gadget, j::Int, fp::FPlan)
    H = fp.H; tw = fp.tw
    C, mask, half = _dconst(G); sh = (64 - G.off - G.logB * j) & 63
    if x0 === nothing
        @inbounds @simd for k in 1:H
            d1 = Float64(reinterpret(Int64, ((x[k] + C) >> sh) & mask)) - half
            d2 = Float64(reinterpret(Int64, ((x[k+H] + C) >> sh) & mask)) - half
            w = tw[k]
            Dj[k] = complex(d1 * real(w) - d2 * imag(w), d1 * imag(w) + d2 * real(w))
        end
    else
        @inbounds @simd for k in 1:H
            d1 = Float64(reinterpret(Int64, ((x[k] - x0[k] + C) >> sh) & mask)) - half
            d2 = Float64(reinterpret(Int64, ((x[k+H] - x0[k+H] + C) >> sh) & mask)) - half
            w = tw[k]
            Dj[k] = complex(d1 * real(w) - d2 * imag(w), d1 * imag(w) + d2 * real(w))
        end
    end
    fp.pf * Dj
    return Dj
end

"Ds[off+j] (j = 1..ℓ) = Fourier values of the balanced digits of x."
function digits_fwd!(Ds::Vector{Vector{ComplexF64}}, off::Int, x::Vector{UInt64}, G::Gadget, fp::FPlan)
    for j in 1:G.ell
        digit_fwd!(Ds[off+j], x, nothing, G, j, fp)
    end
    return Ds
end

"out = t1 - t0."
function diff!(out::Vector{UInt64}, t1::Vector{UInt64}, t0::Vector{UInt64})
    @inbounds @simd for i in eachindex(out)
        out[i] = t1[i] - t0[i]
    end
    return out
end

"out = X^e·t - t, 0 ≤ e < 2N."
function rotdiff!(out::Vector{UInt64}, t::Vector{UInt64}, e::Int)
    N = length(t); neg = e >= N; e2 = neg ? e - N : e
    if neg
        @inbounds @simd for i in 1:e2
            out[i] = t[i-e2+N] - t[i]
        end
        @inbounds @simd for i in e2+1:N
            out[i] = -t[i-e2] - t[i]
        end
    else
        @inbounds @simd for i in 1:e2
            out[i] = -t[i-e2+N] - t[i]
        end
        @inbounds @simd for i in e2+1:N
            out[i] = t[i-e2] - t[i]
        end
    end
    return out
end

# ---------------------------------------------------------------- pointwise kernels

"accA (=|+=) D·KA, accB (=|+=) D·KB   (accumulators interleaved, for the inverse FFT)."
function kmacc!(accA::Vector{ComplexF64}, accB::Vector{ComplexF64}, D::Vector{ComplexF64}, KA::SV, KB::SV,
                add::Bool, rng::UnitRange{Int} = 1:length(D))
    ar = KA.r; ai = KA.i; br = KB.r; bi = KB.i
    if add
        @inbounds @simd for k in rng
            x = D[k]; dr = real(x); di = imag(x)
            accA[k] += complex(dr * ar[k] - di * ai[k], dr * ai[k] + di * ar[k])
            accB[k] += complex(dr * br[k] - di * bi[k], dr * bi[k] + di * br[k])
        end
    else
        @inbounds @simd for k in rng
            x = D[k]; dr = real(x); di = imag(x)
            accA[k] = complex(dr * ar[k] - di * ai[k], dr * ai[k] + di * ar[k])
            accB[k] = complex(dr * br[k] - di * bi[k], dr * bi[k] + di * br[k])
        end
    end
    return nothing
end

# T-layout. Natural Fourier index k = κ + L·m (0-based, L = p/4, m < M0): the automorphisms X ↦ X^a with
# a ≡ 1 mod p (all baby and giant steps of case (B)) fix κ and rotate m by s_κ, and the public polynomials
# of c_3 depend on κ only. Stored split, with m fast (index M0·κ + m): τ is a rotation of contiguous rows of
# length M0 and the public products are row-wise scalings, all vectorizable.

"X (T-layout, split) = Y (natural, interleaved)."
function toT!(X::SV, Y::Vector{ComplexF64}, M0::Int)
    L = length(Y) ÷ M0; Xr = X.r; Xi = X.i
    @inbounds for κ in 0:L-1
        base = M0 * κ
        @simd for m in 0:M0-1
            y = Y[κ+L*m+1]; Xr[base+m+1] = real(y); Xi[base+m+1] = imag(y)
        end
    end
    return X
end

"Y (natural, interleaved) = X (T-layout, split)."
function fromT!(Y::Vector{ComplexF64}, X::SV, M0::Int)
    L = length(Y) ÷ M0; Xr = X.r; Xi = X.i
    @inbounds for m in 0:M0-1
        @simd for κ in 0:L-1
            Y[κ+L*m+1] = complex(Xr[M0*κ+m+1], Xi[M0*κ+m+1])
        end
    end
    return Y
end

"Rotation amounts of τ in the T-layout (checked against its permutation src, no conjugation)."
function tshift(src::Vector{Int}, sg::Vector{Float64}, M0::Int)
    H = length(src); L = H ÷ M0
    sh = [(src[κ+1] - 1 - κ) ÷ L for κ in 0:L-1]
    for κ in 0:L-1, m in 0:M0-1
        @assert src[κ+L*m+1] - 1 == κ + L * mod(m + sh[κ+1], M0) && sg[κ+L*m+1] == 1.0
    end
    return sh
end

"Key K' = τ⁻¹(K) in the T-layout (so that τ(D)·K = τ(D·K'))."
function prepermT(Kv::Vector{ComplexF64}, src::Vector{Int}, M0::Int)
    Kp = similar(Kv)
    for k in eachindex(Kv)
        Kp[src[k]] = Kv[k]
    end
    return toT!(SV(length(Kv)), Kp, M0)
end

# Key switching before τ, on all points (any layout): (oA, oB) = (0, Bv) - Σ_l D_l·K'_l, two digits per pass.

function ks_init!(oA::SV, oB::SV, D::SV, KA::SV, KB::SV, Bv::SV, rng::UnitRange{Int})
    oAr = oA.r; oAi = oA.i; oBr = oB.r; oBi = oB.i; dr_ = D.r; di_ = D.i
    ar = KA.r; ai = KA.i; br = KB.r; bi = KB.i; vr = Bv.r; vi = Bv.i
    @inbounds @simd ivdep for k in rng
        dr = dr_[k]; di = di_[k]
        oAr[k] = di * ai[k] - dr * ar[k]; oAi[k] = -(dr * ai[k] + di * ar[k])
        oBr[k] = vr[k] - dr * br[k] + di * bi[k]; oBi[k] = vi[k] - dr * bi[k] - di * br[k]
    end
    return nothing
end

function ks_sub!(oA::SV, oB::SV, D::SV, KA::SV, KB::SV, rng::UnitRange{Int})
    oAr = oA.r; oAi = oA.i; oBr = oB.r; oBi = oB.i; dr_ = D.r; di_ = D.i
    ar = KA.r; ai = KA.i; br = KB.r; bi = KB.i
    @inbounds @simd ivdep for k in rng
        dr = dr_[k]; di = di_[k]
        oAr[k] += di * ai[k] - dr * ar[k]; oAi[k] -= dr * ai[k] + di * ar[k]
        oBr[k] += di * bi[k] - dr * br[k]; oBi[k] -= dr * bi[k] + di * br[k]
    end
    return nothing
end

function ks_init2!(oA::SV, oB::SV, D1::SV, K1A::SV, K1B::SV, D2::SV, K2A::SV, K2B::SV, Bv::SV, rng::UnitRange{Int})
    oAr = oA.r; oAi = oA.i; oBr = oB.r; oBi = oB.i; vr = Bv.r; vi = Bv.i
    a1r = K1A.r; a1i = K1A.i; b1r = K1B.r; b1i = K1B.i; a2r = K2A.r; a2i = K2A.i; b2r = K2B.r; b2i = K2B.i
    D1r = D1.r; D1i = D1.i; D2r = D2.r; D2i = D2.i
    @inbounds @simd ivdep for k in rng
        d1r = D1r[k]; d1i = D1i[k]; d2r = D2r[k]; d2i = D2i[k]
        oAr[k] = d1i * a1i[k] - d1r * a1r[k] + d2i * a2i[k] - d2r * a2r[k]
        oAi[k] = -(d1r * a1i[k] + d1i * a1r[k] + d2r * a2i[k] + d2i * a2r[k])
        oBr[k] = vr[k] - d1r * b1r[k] + d1i * b1i[k] - d2r * b2r[k] + d2i * b2i[k]
        oBi[k] = vi[k] - d1r * b1i[k] - d1i * b1r[k] - d2r * b2i[k] - d2i * b2r[k]
    end
    return nothing
end

function ks_sub2!(oA::SV, oB::SV, D1::SV, K1A::SV, K1B::SV, D2::SV, K2A::SV, K2B::SV, rng::UnitRange{Int})
    oAr = oA.r; oAi = oA.i; oBr = oB.r; oBi = oB.i
    a1r = K1A.r; a1i = K1A.i; b1r = K1B.r; b1i = K1B.i; a2r = K2A.r; a2i = K2A.i; b2r = K2B.r; b2i = K2B.i
    D1r = D1.r; D1i = D1.i; D2r = D2.r; D2i = D2.i
    @inbounds @simd ivdep for k in rng
        d1r = D1r[k]; d1i = D1i[k]; d2r = D2r[k]; d2i = D2i[k]
        oAr[k] += d1i * a1i[k] - d1r * a1r[k] + d2i * a2i[k] - d2r * a2r[k]
        oAi[k] -= d1r * a1i[k] + d1i * a1r[k] + d2r * a2i[k] + d2i * a2r[k]
        oBr[k] += d1i * b1i[k] - d1r * b1r[k] + d2i * b2i[k] - d2r * b2r[k]
        oBi[k] -= d1r * b1i[k] + d1i * b1r[k] + d2r * b2i[k] + d2i * b2r[k]
    end
    return nothing
end

"(oA, oB) = (0, Bv) - Σ_l D_l·K_l on the points rng."
function keyswitch!(oA::SV, oB::SV, D::Vector{SV}, K::Vector{SRow}, Bv::SV, rng::UnitRange{Int} = 1:length(oA.r))
    ℓ = length(D)
    if ℓ >= 2
        ks_init2!(oA, oB, D[1], K[1][1], K[1][2], D[2], K[2][1], K[2][2], Bv, rng); l = 3
    else
        ks_init!(oA, oB, D[1], K[1][1], K[1][2], Bv, rng); l = 2
    end
    while l + 1 <= ℓ
        ks_sub2!(oA, oB, D[l], K[l][1], K[l][2], D[l+1], K[l+1][1], K[l+1][2], rng); l += 2
    end
    l <= ℓ && ks_sub!(oA, oB, D[l], K[l][1], K[l][2], rng)
    return nothing
end

# τ in the T-layout: row κ rotated by sh[κ]; points base+m, m in [m1, m2], read at base+m+off.
@inline function _segadd!(ga::SV, gb::SV, oA::SV, oB::SV, base, m1, m2, off)
    gar = ga.r; gai = ga.i; gbr = gb.r; gbi = gb.i; oar = oA.r; oai = oA.i; obr = oB.r; obi = oB.i
    @inbounds @simd ivdep for m in m1:m2
        x = base + m; y = x + off
        gar[x] += oar[y]; gai[x] += oai[y]; gbr[x] += obr[y]; gbi[x] += obi[y]
    end
end

"(GA, GB) += τ(O) on the rows κs (T-layout)."
function rotadd!(GA::SV, GB::SV, oA::SV, oB::SV, sh::Vector{Int}, M0::Int, κs::UnitRange{Int} = 0:length(sh)-1)
    @inbounds for κ in κs
        base = M0 * κ; s = sh[κ+1]
        _segadd!(GA, GB, oA, oB, base, 1, M0 - s, s)
        _segadd!(GA, GB, oA, oB, base, M0 - s + 1, M0, s - M0)
    end
    return nothing
end

# Baby steps row by row (T-layout, M0 known at compile time): for each row κ, the j = 0 term, then for each
# baby step j the key switched row (0, B0) - Σ_l D_l·K'_(j,l) (doubled in a small buffer), rotated by s_κ and
# multiplied by the public values of the giant accumulators. The keys are packed in reading order
# [κ][j][l][re A, im A, re B, im B][m]: one sequential stream; the rows of D, B0 and G stay in L1.

"Packed baby keys (T-layout rows, reading order of baby_rows!)."
function packkeys(K::Vector{Vector{SRow}}, M0::Int)
    isempty(K) && return Float64[]                   # B = 1: no baby step
    g1 = length(K); ℓ = length(K[1]); H = length(K[1][1][1].r); L = H ÷ M0
    out = Vector{Float64}(undef, H * g1 * ℓ * 4)
    o = 0
    for κ in 0:L-1, j in 1:g1, l in 1:ℓ, v in (K[j][l][1].r, K[j][l][1].i, K[j][l][2].r, K[j][l][2].i)
        copyto!(out, o + 1, v, M0 * κ + 1, M0); o += M0
    end
    return out
end

"tmp = (0, Bv) - Σ_l D_l·K_l on one row, doubled (tmp[m+M0] = tmp[m]) by the last pass; K packed from offset ko."
@inline function _row_ks!(tmp::NTuple{4,Vector{Float64}}, base::Int, D::Vector{SV}, Kp::Vector{Float64}, ko::Int,
                          Bv::SV, ::Val{M0}) where {M0}
    tar, tai, tbr, tbi = tmp
    ℓ = length(D)
    vr = Bv.r; vi = Bv.i
    Dr = D[1].r; Di = D[1].i
    if ℓ == 1
        @inbounds @simd ivdep for m in 1:M0
            k = base + m; c = ko + m
            dr = Dr[k]; di = Di[k]
            ar = Kp[c]; ai = Kp[c+M0]; br = Kp[c+2M0]; bi = Kp[c+3M0]
            xr = di * ai - dr * ar; xi = -(dr * ai + di * ar)
            yr = vr[k] - dr * br + di * bi; yi = vi[k] - dr * bi - di * br
            tar[m] = xr; tai[m] = xi; tbr[m] = yr; tbi[m] = yi
            tar[m+M0] = xr; tai[m+M0] = xi; tbr[m+M0] = yr; tbi[m+M0] = yi
        end
        return nothing
    end
    @inbounds @simd ivdep for m in 1:M0
        k = base + m; c = ko + m
        dr = Dr[k]; di = Di[k]
        ar = Kp[c]; ai = Kp[c+M0]; br = Kp[c+2M0]; bi = Kp[c+3M0]
        tar[m] = di * ai - dr * ar; tai[m] = -(dr * ai + di * ar)
        tbr[m] = vr[k] - dr * br + di * bi; tbi[m] = vi[k] - dr * bi - di * br
    end
    for l in 2:ℓ-1
        Dr = D[l].r; Di = D[l].i; kl = ko + 4M0 * (l - 1)
        @inbounds @simd ivdep for m in 1:M0
            k = base + m; c = kl + m
            dr = Dr[k]; di = Di[k]
            ar = Kp[c]; ai = Kp[c+M0]; br = Kp[c+2M0]; bi = Kp[c+3M0]
            tar[m] += di * ai - dr * ar; tai[m] -= dr * ai + di * ar
            tbr[m] += di * bi - dr * br; tbi[m] -= dr * bi + di * br
        end
    end
    Dr = D[ℓ].r; Di = D[ℓ].i; kl = ko + 4M0 * (ℓ - 1)
    @inbounds @simd ivdep for m in 1:M0
        k = base + m; c = kl + m
        dr = Dr[k]; di = Di[k]
        ar = Kp[c]; ai = Kp[c+M0]; br = Kp[c+2M0]; bi = Kp[c+3M0]
        xr = tar[m] + di * ai - dr * ar; xi = tai[m] - (dr * ai + di * ar)
        yr = tbr[m] + di * bi - dr * br; yi = tbi[m] - (dr * bi + di * br)
        tar[m] = xr; tai[m] = xi; tbr[m] = yr; tbi[m] = yi
        tar[m+M0] = xr; tai[m+M0] = xi; tbr[m+M0] = yr; tbi[m+M0] = yi
    end
    return nothing
end

@inline function _row_pub!(ga::SV, gb::SV, tmp::NTuple{4,Vector{Float64}}, base::Int, s::Int, qr::Float64,
                           qi::Float64, ::Val{M0}) where {M0}
    tar, tai, tbr, tbi = tmp
    gar = ga.r; gai = ga.i; gbr = gb.r; gbi = gb.i
    @inbounds @simd ivdep for m in 1:M0
        x = base + m; y = m + s
        xr = tar[y]; xi = tai[y]; yr = tbr[y]; yi = tbi[y]
        gar[x] += qr * xr - qi * xi; gai[x] += qr * xi + qi * xr
        gbr[x] += qr * yr - qi * yi; gbi[x] += qr * yi + qi * yr
    end
    return nothing
end

"G += q1·τ1(row1) + q2·τ2(row2) on one row (two baby steps at once: G read and written once)."
@inline function _row_pub2!(ga::SV, gb::SV, t1::NTuple{4,Vector{Float64}}, s1::Int, q1r::Float64, q1i::Float64,
                            t2::NTuple{4,Vector{Float64}}, s2::Int, q2r::Float64, q2i::Float64, base::Int,
                            ::Val{M0}) where {M0}
    a1r, a1i, b1r, b1i = t1; a2r, a2i, b2r, b2i = t2
    gar = ga.r; gai = ga.i; gbr = gb.r; gbi = gb.i
    @inbounds @simd ivdep for m in 1:M0
        x = base + m; y1 = m + s1; y2 = m + s2
        xr = a1r[y1]; xi = a1i[y1]; yr = b1r[y1]; yi = b1i[y1]
        ur = a2r[y2]; ui = a2i[y2]; vr = b2r[y2]; vi = b2i[y2]
        gar[x] += q1r * xr - q1i * xi + q2r * ur - q2i * ui; gai[x] += q1r * xi + q1i * xr + q2r * ui + q2i * ur
        gbr[x] += q1r * yr - q1i * yi + q2r * vr - q2i * vi; gbi[x] += q1r * yi + q1i * yr + q2r * vi + q2i * vr
    end
    return nothing
end

@inline function _row_set!(ga::SV, gb::SV, A::SV, B::SV, base::Int, qr::Float64, qi::Float64, ::Val{M0}) where {M0}
    gar = ga.r; gai = ga.i; gbr = gb.r; gbi = gb.i; ar = A.r; ai = A.i; br = B.r; bi = B.i
    @inbounds @simd ivdep for m in 1:M0
        x = base + m
        gar[x] = qr * ar[x] - qi * ai[x]; gai[x] = qr * ai[x] + qi * ar[x]
        gbr[x] = qr * br[x] - qi * bi[x]; gbi[x] = qr * bi[x] + qi * br[x]
    end
    return nothing
end

"""G_kk = Σ_j Q_(kk,j)·τ_j((0, B0) - Σ_l D_l·K'_(j,l))  (j = 0: Q_(kk,0)·(A0, B0)), on the rows κs, for each
workspace of wss (one batch: wss = (ws,)). Baby steps by pairs: two key switched rows (buffers t1, t2), then
the giant accumulators read and written once per pair."""
function baby_rows!(::Val{M0}, κs::UnitRange{Int}, wss, t1::NTuple{4,Vector{Float64}},
                    t2::NTuple{4,Vector{Float64}}) where {M0}
    w1 = first(wss); h, g = size(w1.tix); ℓ = length(w1.DksT); Kp = w1.bpack; st = 4M0 * ℓ
    @inbounds for κ in κs
        base = M0 * κ
        for w in wss
            ko = κ * (g - 1) * st
            for t in 1:h
                q = w.Q[w.tix[t, 1]]
                _row_set!(w.GA[t], w.GB[t], w.A0T, w.B0T, base, q.r[κ+1], q.i[κ+1], Val(M0))
            end
            j = 1
            while j <= g - 1
                s1 = w1.bsh[j][κ+1]
                _row_ks!(t1, base, w.DksT, Kp, ko, w.B0T, Val(M0)); ko += st
                if j + 1 <= g - 1
                    s2 = w1.bsh[j+1][κ+1]
                    _row_ks!(t2, base, w.DksT, Kp, ko, w.B0T, Val(M0)); ko += st
                    for t in 1:h
                        u1 = w.tix[t, j+1]; u2 = w.tix[t, j+2]
                        q1r = u1 == 0 ? 0.0 : w.Q[u1].r[κ+1]; q1i = u1 == 0 ? 0.0 : w.Q[u1].i[κ+1]
                        q2r = u2 == 0 ? 0.0 : w.Q[u2].r[κ+1]; q2i = u2 == 0 ? 0.0 : w.Q[u2].i[κ+1]
                        _row_pub2!(w.GA[t], w.GB[t], t1, s1, q1r, q1i, t2, s2, q2r, q2i, base, Val(M0))
                    end
                    j += 2
                else
                    for t in 1:h
                        u1 = w.tix[t, j+1]
                        u1 == 0 && continue
                        _row_pub!(w.GA[t], w.GB[t], t1, base, s1, w.Q[u1].r[κ+1], w.Q[u1].i[κ+1], Val(M0))
                    end
                    j += 1
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------- workspace

const CV = Vector{ComplexF64}

struct Work
    bsk::Vector{Vector{SRow}}           # bootstrapping keys, split
    bpack::Vector{Float64}; bsh::Vector{Vector{Int}}    # baby keys τ_j⁻¹(K) (T-layout, packed), rotations
    giant::Vector{Vector{SRow}}; gsh::Vector{Vector{Int}}
    Dbr::Vector{CV}                     # 2ℓ digits of the external product
    Dks::Vector{CV}; DksT::Vector{SV}   # ℓ_aut digits (baby steps)
    Dg::Vector{CV}; DgT::Vector{SV}     # ℓ_aut digits (giant steps)
    accA::CV; accB::CV; Y::CV; YA::CV; YB::CV
    A0T::SV; B0T::SV
    GA::Vector{SV}; GB::Vector{SV}      # giant accumulators (T-layout), kk = 0..h-1 (kk = 0 is C_L)
    OA::SV; OB::SV                      # one giant step before τ
    rows::Vector{NTuple{4,Vector{Float64}}}   # thread t: rows[2t-1], rows[2t], key switched rows, doubled
    ta::Vector{UInt64}; tb::Vector{UInt64}; ua::Vector{UInt64}; ub::Vector{UInt64}; tmp::Vector{UInt64}
    S::Matrix{ComplexF64}               # B × p/2, then table of c_3
    pS::Any
    idx::Vector{Vector{Int}}            # for each t = (kk, j): linear indices into the table, one period
    Q::Vector{SV}                       # public polynomials, one value per row κ
    tix::Matrix{Int}                    # t of (kk, j), 0 if j + g·kk ≥ B
    pool::Vector{Work}                  # workspaces of threads 2, 3, ... (same keys, own buffers)
end

function Work(P::CBParams, fp::FPlan, K::FKeys; nthreads::Int = Threads.nthreads())
    H = fp.H; N = P.N; M0 = P.M0
    @assert P.p >= M0 "T-layout needs p ≥ M0"
    bsk = [[(SV(r[1]), SV(r[2])) for r in Kg.rows] for Kg in K.bsk]
    prow(r, Kτ) = (prepermT(r[1], Kτ.src, M0), prepermT(r[2], Kτ.src, M0))
    baby = [[prow(r, Kτ) for r in Kτ.rows] for Kτ in K.baby]
    giant = [[prow(r, Kτ) for r in Kτ.rows] for Kτ in K.giant]
    bsh = [tshift(Kτ.src, Kτ.sg, M0) for Kτ in K.baby]
    gsh = [tshift(Kτ.src, Kτ.sg, M0) for Kτ in K.giant]
    # table indices: T[t+1, (c-1)/2+1], c = (4k+1)e mod 2N, t = -c r mod B, for a = c_k γ^j, e = c_k^{-1};
    # they depend on c mod p (p ≥ B), hence on κ = k mod L only, L = p/4
    M = 2N; γ = 1 + P.p; mask = M - 1; pm = P.p - 1; bm = P.B - 1
    L = H ÷ M0
    idx = Vector{Int}[]; tix = zeros(Int, P.h, P.g)
    for kk in 0:P.h-1, j in 0:P.g-1
        j + P.g * kk < P.B || continue
        ck = powermod(γ, P.g * kk, M)
        a = mod(ck * powermod(γ, j, M), M); e = invmod(ck, M)
        rr = mod((a - 1) ÷ P.p, P.M0)
        v = Vector{Int}(undef, H)
        c = e & mask; step = (4e) & mask
        for k in 1:H
            col = ((c & pm) >> 1) + 1
            row = ((-c * rr) & bm) + 1
            v[k] = row + P.B * (col - 1)
            c = (c + step) & mask
        end
        @assert all(v[k] == v[mod1(k, L)] for k in 1:H)
        push!(idx, v[1:L]); tix[kk+1, j+1] = length(idx)
    end
    ws = _work(P, fp, bsk, packkeys(baby, M0), bsh, giant, gsh, idx, tix, nthreads)
    for _ in 2:nthreads
        push!(ws.pool, Work(ws, P, fp))
    end
    return ws
end

"Buffers around the given keys and tables."
function _work(P::CBParams, fp::FPlan, bsk, bpack, bsh, giant, gsh, idx, tix, nthreads::Int)
    H = fp.H; N = P.N; M0 = P.M0; L = H ÷ M0
    z() = zeros(ComplexF64, H); u() = zeros(UInt64, N)
    S = zeros(ComplexF64, P.B, P.p ÷ 2)
    pS = plan_bfft!(S, 1; flags = FFTW.MEASURE, num_threads = fp.nthreads)
    return Work(bsk, bpack, bsh, giant, gsh,
                [z() for _ in 1:2P.Gbr.ell], [z() for _ in 1:P.Gks.ell], [SV(H) for _ in 1:P.Gks.ell],
                [z() for _ in 1:P.Gks.ell], [SV(H) for _ in 1:P.Gks.ell],
                z(), z(), z(), z(), z(), SV(H), SV(H),
                [SV(H) for _ in 1:P.h], [SV(H) for _ in 1:P.h],
                SV(H), SV(H), [(zeros(2M0), zeros(2M0), zeros(2M0), zeros(2M0)) for _ in 1:2nthreads],
                u(), u(), u(), u(), u(), S, pS, idx, [SV(L) for _ in idx], tix, Work[])
end

"Workspace sharing the keys of ws, with its own buffers (for another thread)."
Work(ws::Work, P::CBParams, fp::FPlan) =
    _work(P, fp, ws.bsk, ws.bpack, ws.bsh, ws.giant, ws.gsh, ws.idx, ws.tix, length(ws.rows) ÷ 2)

"Workspace of thread t (t = 1: ws itself)."
wsof(ws::Work, t::Int) = t == 1 ? ws : ws.pool[t-1]

# ---------------------------------------------------------------- CMux in place

"C ← C + Dec(C1 - C)·GGSW(s_u), C1 = (c1a, c1b)  (rot: C1 = X^e·C, c1a, c1b ignored)."
function _cmux!(C::GLWE, c1a, c1b, e::Int, rows::Vector{SRow}, G::Gadget, ws::Work, fp::FPlan)
    ℓ = G.ell
    e >= 0 ? rotdiff!(ws.tmp, C.a, e) : diff!(ws.tmp, c1a, C.a)
    digits_fwd!(ws.Dbr, 0, ws.tmp, G, fp)
    e >= 0 ? rotdiff!(ws.tmp, C.b, e) : diff!(ws.tmp, c1b, C.b)
    digits_fwd!(ws.Dbr, ℓ, ws.tmp, G, fp)
    for r in 1:2ℓ
        kmacc!(ws.accA, ws.accB, ws.Dbr[r], rows[r][1], rows[r][2], r > 1)
    end
    inv_torus!(C.a, ws.accA, fp, true); inv_torus!(C.b, ws.accB, fp, true)
    return C
end
cmux!(C::GLWE, c1a, c1b, u::Int, P::CBParams, ws::Work, fp::FPlan) = _cmux!(C, c1a, c1b, -1, ws.bsk[u], P.Gbr, ws, fp)
cmux_rot!(C::GLWE, e::Int, u::Int, P::CBParams, ws::Work, fp::FPlan) = _cmux!(C, nothing, nothing, e, ws.bsk[u], P.Gbr, ws, fp)

# ---------------------------------------------------------------- Algorithm 1 (optimized, multithreaded)

# Sense-reversing spin barrier: a few hundred ns, against several µs for a task spawn. The counter and the flag are
# 128 bytes apart (cache lines of 128 bytes on Apple cores): the threads that wait spin on the flag without being
# disturbed by the increments of the counter.
mutable struct BarrierState
    @atomic count::Int
    pad1::NTuple{15,Int}
    @atomic sense::Int
    pad2::NTuple{15,Int}
end
struct Barrier
    n::Int
    st::BarrierState
end
Barrier(n::Int) = Barrier(n, BarrierState(0, ntuple(_ -> 0, 15), 0, ntuple(_ -> 0, 15)))

@inline function sync!(b::Barrier, ls::Int)
    b.n == 1 && return ls
    ls = 1 - ls
    st = b.st
    if (@atomic st.count += 1) == b.n
        @atomic st.count = 0
        @atomic st.sense = ls
    else
        while (@atomic :acquire st.sense) != ls
            ccall(:jl_cpu_pause, Cvoid, ())
            GC.safepoint()
        end
    end
    return ls
end

"Line 3: table of c_3 (B-point FFTs) for the errors aints[:, u]; public polynomials, one value per row κ."
function table!(ws::Work, P::CBParams, fp::FPlan, aints::AbstractMatrix{Int}, u::Int)
    S = ws.S; Z = fp.Z; mask = 2P.N - 1; B = P.B
    @inbounds for ci in 1:(P.p÷2), k in 1:B
        S[k, ci] = Z[((P.M0 * (2ci - 1) * aints[k, u]) & mask) + 1]
    end
    ws.pS * S
    @inbounds for t in eachindex(ws.idx)
        id = ws.idx[t]; Q = ws.Q[t]
        for k in eachindex(id)
            x = S[id[k]]; Q.r[k] = real(x); Q.i[k] = imag(x)
        end
    end
    return nothing
end

@inline _part(n::Int, T::Int, t::Int) = ((t - 1) * n) ÷ T + 1 : (t * n) ÷ T     # t-th of T slices of 1:n

# Phases of one iteration, for thread tid of T (T = 1: the whole phase). Between two phases: a barrier.

"""Line 4: division by B, points rN. Shifted gadget G of the automorphism keys (off > 0): the mask is replaced by the
representative that its digits reconstruct, t - k·2^(64 - off) with k = (t + C) >> (64 - off) (offset trick), another
representative of the division (Lemma 4, off ≤ log2 B)."""
function _ph_div!(ws::Work, C::GLWE, rN::UnitRange{Int}, s::Int, G::Gadget)
    if G.off == 0
        @inbounds @simd for i in rN
            x = C.a[i]; ws.ta[i] = (x >> s) + ((x >> (s - 1)) & 0x1)
            y = C.b[i]; ws.tb[i] = (y >> s) + ((y >> (s - 1)) & 0x1)
        end
    else
        Cd = _dconst(G)[1]; sh = 64 - G.off
        @inbounds @simd for i in rN
            x = C.a[i]; t = (x >> s) + ((x >> (s - 1)) & 0x1)
            ws.ta[i] = t - (((t + Cd) >> sh) << sh)
            y = C.b[i]; ws.tb[i] = (y >> s) + ((y >> (s - 1)) & 0x1)
        end
    end
    return nothing
end

"""Line 4, one part (1: mask, 2: body), points rN: as _ph_div!, for the fused phase of the grouped comb bootstrap
(the thread that adds the inverse FFT of a part to C divides it at once)."""
function _ph_div_part!(ws::Work, C::GLWE, part::Int, rN::UnitRange{Int}, s::Int, G::Gadget)
    if part == 2
        @inbounds @simd for i in rN
            y = C.b[i]; ws.tb[i] = (y >> s) + ((y >> (s - 1)) & 0x1)
        end
    elseif G.off == 0
        @inbounds @simd for i in rN
            x = C.a[i]; ws.ta[i] = (x >> s) + ((x >> (s - 1)) & 0x1)
        end
    else
        Cd = _dconst(G)[1]; sh = 64 - G.off
        @inbounds @simd for i in rN
            x = C.a[i]; t = (x >> s) + ((x >> (s - 1)) & 0x1)
            ws.ta[i] = t - (((t + Cd) >> sh) << sh)
        end
    end
    return nothing
end

"FFTs of A0 and B0, hoisted digits of A0, table of c_3: one task each."
function _ph_fft!(ws::Work, P::CBParams, fp::FPlan, aints::AbstractMatrix{Int}, u::Int, tid::Int, T::Int, M0::Int)
    for it in tid:T:3+P.Gks.ell
        if it == 1
            fwd_torus!(ws.YA, ws.ta, fp); toT!(ws.A0T, ws.YA, M0)
        elseif it == 2
            fwd_torus!(ws.YB, ws.tb, fp); toT!(ws.B0T, ws.YB, M0)
        elseif it == 3
            table!(ws, P, fp, aints, u)
        else
            l = it - 3
            digit_fwd!(ws.Dks[l], ws.ta, nothing, P.Gks, l, fp); toT!(ws.DksT[l], ws.Dks[l], M0)
        end
    end
    return nothing
end

"Giant step kk, 1: inverse FFT of the mask of G_kk (one task)."
function _ph_ginv!(ws::Work, kk::Int, fp::FPlan, M0::Int)
    fromT!(ws.Y, ws.GA[kk+1], M0); inv_torus!(ws.ua, ws.Y, fp, false)
    return nothing
end

"Giant step, 2: its digits (one task each)."
function _ph_gdig!(ws::Work, P::CBParams, fp::FPlan, tid::Int, T::Int, M0::Int)
    for l in tid:T:P.Gks.ell
        digit_fwd!(ws.Dg[l], ws.ua, nothing, P.Gks, l, fp); toT!(ws.DgT[l], ws.Dg[l], M0)
    end
    return nothing
end

"Giant step kk, 3: key switching and τ_kk, added to C_L (points pts = rows κs)."
function _ph_gks!(ws::Work, kk::Int, pts::UnitRange{Int}, κs::UnitRange{Int}, M0::Int)
    keyswitch!(ws.OA, ws.OB, ws.DgT, ws.giant[kk], ws.GB[kk+1], pts)
    rotadd!(ws.GA[1], ws.GB[1], ws.OA, ws.OB, ws.gsh[kk], M0, κs)
    return nothing
end

"Line 11, 1: inverse FFTs of C_L (two tasks)."
function _ph_cinv!(ws::Work, fp::FPlan, tid::Int, T::Int, M0::Int)
    for it in tid:T:2
        if it == 1
            fromT!(ws.YA, ws.GA[1], M0); inv_torus!(ws.ua, ws.YA, fp, false)
        else
            fromT!(ws.YB, ws.GB[1], M0); inv_torus!(ws.ub, ws.YB, fp, false)
        end
    end
    return nothing
end

"Line 11, 2: digits of C_L - C (2ℓ tasks)."
function _ph_cdig!(ws::Work, C::GLWE, P::CBParams, fp::FPlan, tid::Int, T::Int)
    ℓ = P.Gbr.ell
    for it in tid:T:2ℓ
        j = (it - 1) % ℓ + 1
        it <= ℓ ? digit_fwd!(ws.Dbr[it], ws.ua, C.a, P.Gbr, j, fp) : digit_fwd!(ws.Dbr[it], ws.ub, C.b, P.Gbr, j, fp)
    end
    return nothing
end

"Line 11, 3: external product with GGSW(s_u), points rH."
function _ph_ext!(ws::Work, u::Int, rH::UnitRange{Int})
    rows = ws.bsk[u]
    for r in eachindex(ws.Dbr)
        kmacc!(ws.accA, ws.accB, ws.Dbr[r], rows[r][1], rows[r][2], r > 1, rH)
    end
    return nothing
end

"Line 11, 4: C += inverse FFTs (two tasks)."
function _ph_cadd!(ws::Work, C::GLWE, fp::FPlan, tid::Int, T::Int)
    for it in tid:T:2
        it == 1 ? inv_torus!(C.a, ws.accA, fp, true) : inv_torus!(C.b, ws.accB, fp, true)
    end
    return nothing
end

"""Iterations us of Algorithm 1, run by thread tid of T (SPMD, all T threads call it; T = 1: sequential).
Synchronization by spin barriers only. tim (optional, thread 1): times of the 8 phases, waiting included."""
function cb_spmd!(::Val{M0}, C::GLWE, aints::AbstractMatrix{Int}, us::UnitRange{Int}, P::CBParams, fp::FPlan,
                  ws::Work, bar::Barrier, tid::Int, tim = nothing) where {M0}
    T = bar.n; H = fp.H; L = H ÷ M0; s = trailing_zeros(P.B)
    rN = _part(P.N, T, tid); rH = _part(H, T, tid)
    κs = _part(L, T, tid) .- 1; pts = M0 * first(κs) + 1 : M0 * (last(κs) + 1)
    t1buf = ws.rows[2tid-1]; t2buf = ws.rows[2tid]
    ls = 0
    clk = tim !== nothing && tid == 1
    t0 = clk ? time_ns() : UInt64(0)
    for u in us
        _ph_div!(ws, C, rN, s, P.Gks)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[1] += t1 - t0; t0 = t1)
        _ph_fft!(ws, P, fp, aints, u, tid, T, M0)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[2] += t1 - t0; t0 = t1)
        baby_rows!(Val(M0), κs, (ws,), t1buf, t2buf)                # lines 5-10
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[3] += t1 - t0; t0 = t1)
        for kk in 1:P.h-1
            tid == 1 && _ph_ginv!(ws, kk, fp, M0)
            ls = sync!(bar, ls)
            _ph_gdig!(ws, P, fp, tid, T, M0)
            ls = sync!(bar, ls)
            _ph_gks!(ws, kk, pts, κs, M0)
            ls = sync!(bar, ls)
        end
        clk && (t1 = time_ns(); tim[4] += t1 - t0; t0 = t1)
        _ph_cinv!(ws, fp, tid, T, M0)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[5] += t1 - t0; t0 = t1)
        _ph_cdig!(ws, C, P, fp, tid, T)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[6] += t1 - t0; t0 = t1)
        _ph_ext!(ws, u, rH)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[7] += t1 - t0; t0 = t1)
        _ph_cadd!(ws, C, fp, tid, T)
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[8] += t1 - t0; t0 = t1)
    end
    return nothing
end

"One iteration (u-th column of aints), sequential."
cb_step!(C::GLWE, aints::AbstractMatrix{Int}, u::Int, P::CBParams, fp::FPlan, ws::Work) =
    cb_spmd!(Val(P.M0), C, aints, u:u, P, fp, ws, Barrier(1), 1)

"Algorithm 1 for the B errors (columns of aints); nthreads threads inside each iteration (latency)."
function combboot_fast(P::CBParams, fp::FPlan, K::FKeys, aints::AbstractMatrix{Int}, bints::Vector{Int};
                        ws::Work = Work(P, fp, K), nthreads::Int = 1, tim = nothing)
    C = trivial(acc0(P, bints))
    us = 1:size(aints, 2)
    if nthreads == 1
        cb_spmd!(Val(P.M0), C, aints, us, P, fp, ws, Barrier(1), 1, tim)
    else
        @assert nthreads <= min(Threads.nthreads(), length(ws.rows) ÷ 2) "start julia with -t $nthreads"
        bar = Barrier(nthreads)
        Threads.@threads :static for tid in 1:nthreads
            cb_spmd!(Val(P.M0), C, aints, us, P, fp, ws, bar, tid, tim)
        end
    end
    return [extract(C, k) for k in 0:P.B-1]
end

"""B separate blind rotations. batched = false: one after the other (each one reads all the bootstrapping keys);
batched = true: the B rotations advance together, each key GGSW(s_u) is read once for the B CMux.
nthreads > 1: the B rotations are shared between the threads (each with its own workspace)."""
function separate_rotations_fast(P::CBParams, fp::FPlan, K::FKeys, aints::AbstractMatrix{Int},
                                 bints::Vector{Int}; ws::Work = Work(P, fp, K), batched::Bool = true,
                                 nthreads::Int = 1)
    N = P.N
    v = vstar(P)
    Cs = [trivial(mulX(v, -P.M0 * bints[k])) for k in 1:P.B]
    go(t) = begin
        w = wsof(ws, t); ks = t:nthreads:P.B
        if batched
            for u in 1:size(aints, 2), k in ks
                cmux_rot!(Cs[k], mod(P.M0 * aints[k, u], 2N), u, P, w, fp)
            end
        else
            for k in ks, u in 1:size(aints, 2)
                cmux_rot!(Cs[k], mod(P.M0 * aints[k, u], 2N), u, P, w, fp)
            end
        end
    end
    if nthreads == 1
        go(1)
    else
        @assert nthreads <= min(Threads.nthreads(), length(ws.pool) + 1) "start julia with -t $nthreads"
        Threads.@threads :static for t in 1:nthreads
            go(t)
        end
    end
    return [extract(C, 0) for C in Cs]
end
