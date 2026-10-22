# Optimized FFT engine of case (A) (included in module CaseAFast), on the model of fast_opt.jl.
#
# Comb bootstrap (B ≥ 2). Rader's algorithm along V on the B/2 complex columns w_k = x_k + i x_{k+B/2} (two real
# columns of the same polynomial, no unpacking), then the negacyclic transform along U on the rows, at the roots
# u_j = ζ^{4j+1}, ζ = e^{iπ/B}, j < B/2 (one in each pair of conjugate roots, as fast.jl): X(u_j, V) = Σ_k w_k(V) ζ^k
# e^{2iπ jk/(B/2)}. The twist ζ^k and the terms of Rader's algorithm in f = 0 and v = 0 are folded into the kernels
# of the convolutions, so that a transform is: one pass (gather in Rader order), FFTs of size p - 1, a pointwise
# product, inverse FFTs, B/2-point FFTs of the rows.
#   Fourier values: p × B/2 matrix, row q + 1 (q < p - 1) for V_f, f = g^{-q} (g a primitive root mod p), row p for
#   f = 0; column j + 1 for u_j.
#   τ_c (c ≡ 1 mod p): u_j ↦ u_j^c = ζ^{(4j+1)c}; if (4j+1)c ≡ 1 (mod 4), column j' = ((4j+1)c mod 2B - 1)/4; otherwise
#   the conjugate of column j'' = (2B - (4j+1)c mod 2B - 1)/4 at -f, i.e. rotated by (p - 1)/2 in Rader order.
#   Key switching with pre-permuted keys, τ(D)·K = τ(D·τ⁻¹(K)); τ is applied while adding to C_L, with the public
#   polynomial B·A_c, whose value at (u_j, V_f) is T[f, t], t = (4j+1)(1-c)/2 mod B, T[f, ·] the B-point (inverse)
#   FFT of (V_f^{a_k})_k: one table per pattern.
#   Digits by the offset trick (as fast_opt.jl), read from the suffix sums a/Ω*_{0,0} (one pass), written in Rader
#   order; keys split (real / imaginary parts); preallocated workspace; the accumulator stays in the coefficient
#   domain (the inverse transforms add into it).
#
# Error bootstrap (B = 1): pairs of real polynomials (the digits of the external product two by two, mask and body of
# the accumulator) in one complex Rader transform, unpacked into half spectra (rows q < (p-1)/2 and f = 0); or, when
# p - 1 is not smooth (p = 2039), zero padding to 2^⌈log2(2p - 2)⌉ (real FFTs, folded by the inverse).

export AOPlan, AOKeys, aokeygen, AOWork, combboot_A_opt, ao_fwd, ao_inv, ao_tau, AOPlan1, AOKeys1, aokeygen1,
       AOWork1, blind_rotation_A_opt, refresh_A_opt

# ---------------------------------------------------------------- transform, B ≥ 2
#
# A batch of np polynomials is a p × (np B/2) matrix, polynomial s in the columns (s-1)B/2+1 .. s B/2 ("slot" s).
# Forward: fill (rows 1..p-1: w_k at v = g^r, row p: w_k(0)); FFTs of size p - 1 of all the columns; one pass that
# multiplies by the kernel (common to the columns) and mixes the columns of each polynomial (twist ζ^k and B/2-point
# transform along U, which commute with the FFTs along V), with the terms in f = 0 and v = 0; inverse FFTs.
# Inverse: FFTs; the mixing pass (inverse transform along U, kernel, terms in f = 0 and v = 0); inverse FFTs.

struct AOPlan
    p::Int; B::Int; B2::Int; n1::Int; h1::Int
    gp::Vector{Int}                # gp[r+1] = g^r mod p
    dl::Vector{Int}                # dl[v+1] = r with g^r ≡ v (v ≠ 0)
    pos::Vector{Int}               # Rader order of the coefficients: pos[v+1] = r + 1 for v = g^r, pos[1] = p
    kf::Matrix{ComplexF64}         # (p-1) × B/2: FFT(b) ζ^k / (p - 1), b_s = ω^{g^{-s}}, ω = e^{-2iπ/p}
    ki::Matrix{ComplexF64}         # (p-1) × B/2: FFT(β) ζ^{-k} / ((p - 1) p B/2), β_s = ω^{-g^s}
    tw::Vector{ComplexF64}         # ζ^k
    itw::Vector{ComplexF64}        # ζ^{-k} / (p B/2)
    Wn::Vector{ComplexF64}         # Wn[m+1] = ω^{g^{-m}}, m < 2(p - 1) (tables of the public polynomials)
    plans::Dict{Int,Any}           # number of slots => FFTs in place: along V (rows 1..p-1), along U (B/2 points)
end

AOPlan(Rg::ARing) = AOPlan(Rg.p, Rg.B)

function AOPlan(p::Int, B::Int)
    B >= 2 || error("AOPlan: B ≥ 2 (error bootstrap: AOPlan1)")
    B2 = B ÷ 2; n1 = p - 1; h1 = n1 ÷ 2
    g = primroot(p); gi = invmod(g, p)
    gp = [powermod(g, r, p) for r in 0:n1-1]
    dl = fill(-1, p)
    for r in 0:n1-1
        dl[gp[r+1]+1] = r
    end
    ζ = [cispi(k / B) for k in 0:B2-1]
    bh = fft([cispi(-2 * powermod(gi, s, p) / p) for s in 0:n1-1])
    βh = fft([cispi(2 * powermod(g, s, p) / p) for s in 0:n1-1])
    kf = [bh[r] * ζ[k] / n1 for r in 1:n1, k in 1:B2]
    ki = [βh[r] * conj(ζ[k]) / (n1 * p * B2) for r in 1:n1, k in 1:B2]
    Wn = [cispi(-2 * powermod(gi, mod(m, n1), p) / p) for m in 0:2n1-1]
    pos = [v == 0 ? p : dl[v+1] + 1 for v in 0:p-1]
    return AOPlan(p, B, B2, n1, h1, gp, dl, pos, kf, ki, ζ, conj.(ζ) ./ (p * B2), Wn, Dict{Int,Any}())
end

function _plans(P::AOPlan, ns::Int)
    get!(P.plans, ns) do
        F = zeros(ComplexF64, P.p, ns * P.B2); Fv = view(F, 1:P.n1, :); F3 = reshape(F, P.p, P.B2, ns)
        fl = FFTW.MEASURE
        (plan_fft!(Fv, 1; flags = fl), plan_bfft!(Fv, 1; flags = fl),
         P.B2 > 1 ? plan_bfft!(F3, 2; flags = fl) : nothing, P.B2 > 1 ? plan_fft!(F3, 2; flags = fl) : nothing)
    end
end

"Forward transform of the ns = size(F, 2)/(B/2) filled slots of F, in place."
function ao_fwd!(F::Matrix{ComplexF64}, P::AOPlan)
    p = P.p; n1 = P.n1; B2 = P.B2; ns = size(F, 2) ÷ B2
    pa, pai, pu, _ = _plans(P, ns)
    Fv = view(F, 1:n1, :)
    pa * Fv
    @inbounds for c in 1:size(F, 2)
        k = mod1(c, B2); w = P.tw[k]
        z0 = F[p, c]
        F[p, c] = (z0 + F[1, c]) * w                     # f = 0: Σ_v w_k(v), twisted
        @simd for r in 1:n1
            F[r, c] *= P.kf[r, k]
        end
        F[1, c] += z0 * w                                # + w_k(0) at every f ≠ 0
    end
    B2 > 1 && (pu * reshape(F, p, B2, ns))               # transform along U (commutes with the FFTs along V)
    pai * Fv
    return F
end

"Inverse transform of the slots of F, in place (then ao_out!)."
function ao_invt!(F::Matrix{ComplexF64}, P::AOPlan)
    p = P.p; n1 = P.n1; B2 = P.B2; ns = size(F, 2) ÷ B2
    pa, pai, _, pui = _plans(P, ns)
    Fv = view(F, 1:n1, :)
    pa * Fv
    B2 > 1 && (pui * reshape(F, p, B2, ns))
    @inbounds for c in 1:size(F, 2)
        k = mod1(c, B2); w = P.itw[k]
        W0 = F[p, c]
        F[p, c] = (W0 + F[1, c]) * w                     # v = 0: (1/p) Σ_f W(f)
        @simd for r in 1:n1
            F[r, c] *= P.ki[r, k]
        end
        F[1, c] += W0 * w                                # + W(0)/p at every v ≠ 0
    end
    pai * Fv
    return F
end

"Augmented vector in Rader order (each column: position r for v = g^r, r = 1..p-1, position p for v = 0)."
function ao_toR(x::AbstractVector{T}, P::AOPlan) where {T}
    p = P.p; out = similar(x)
    for k in 0:P.B-1, v in 0:p-1
        out[k*p+P.pos[v+1]] = x[k*p+v+1]
    end
    return out
end
"Back to the natural order."
function ao_fromR(xR::AbstractVector{T}, P::AOPlan) where {T}
    p = P.p; out = similar(xR)
    for k in 0:P.B-1, v in 0:p-1
        out[k*p+v+1] = xR[k*p+P.pos[v+1]]
    end
    return out
end

"Slot s of F = the complex columns x_k + i x_{k+B/2} of x (Rader order; torus read as signed, or integers)."
function ao_fill!(F::Matrix{ComplexF64}, s::Int, xR::AbstractVector, P::AOPlan)
    p = P.p; B2 = P.B2; c0 = (s - 1) * B2
    @inbounds for k in 0:B2-1
        o0 = k * p; o1 = (k + B2) * p; oc = (c0 + k) * p
        @simd for r in 1:p
            F[oc+r] = complex(tofloat(xR[o0+r]), tofloat(xR[o1+r]))
        end
    end
    return F
end

"Constants of the balanced digits (offset trick, as fast_opt.jl)."
function _adconst(G::Gadget)
    b = G.logB; ℓ = G.ell; shift = 64 - G.off - b * ℓ
    h = UInt64(0)
    for j in 0:ℓ-1
        h += (UInt64(1) << (b - 1)) << (b * j)
    end
    return (UInt64(1) << (shift - 1)) + (h << shift), (UInt64(1) << b) - 1, Float64(1 << (b - 1))
end

@inline _dig(x::UInt64, C::UInt64, sh::Int, mask::UInt64, half::Float64) =
    Float64(reinterpret(Int64, ((x + C) >> sh) & mask)) - half

"Slots s0+1..s0+ℓ of F = the balanced digits of apR (Ω coordinates, in Rader order: ao_suffix!), before the transform."
function ao_fill_digits!(F::Matrix{ComplexF64}, s0::Int, apR::Vector{UInt64}, G::Gadget, P::AOPlan)
    p = P.p; B2 = P.B2
    C, mask, half = _adconst(G)
    @inbounds for j in 1:G.ell
        sh = (64 - G.off - G.logB * j) & 63; c0 = (s0 + j - 1) * B2
        for k in 0:B2-1
            o0 = k * p; o1 = (k + B2) * p; oc = (c0 + k) * p
            @simd for r in 1:p
                F[oc+r] = complex(_dig(apR[o0+r], C, sh, mask, half), _dig(apR[o1+r], C, sh, mask, half))
            end
        end
    end
    return F
end

"""apR = (x - x0)/Ω*_{0,0} in Ω coordinates (x0 = nothing: x), x, x0 and apR in Rader order: suffix sums
ap_v = Σ_{v ≤ w ≤ p-2} (x - x0)_w, ap_{p-1} = 0 (div_omega00 in one pass)."""
function ao_suffix!(apR::Vector{UInt64}, xR::Vector{UInt64}, x0R, B::Int, P::AOPlan)
    p = P.p; pos = P.pos
    @inbounds for k in 0:B-1
        o = k * p; s = zero(UInt64); apR[o+pos[p]] = 0                  # v = p - 1
        if x0R === nothing
            for v in p-2:-1:0
                i = o + pos[v+1]; s += xR[i]; apR[i] = s
            end
        else
            for v in p-2:-1:0
                i = o + pos[v+1]; s += xR[i] - x0R[i]; apR[i] = s
            end
        end
    end
    return apR
end

"out (=|+=) the torus vector (Rader order) of slot s of F (after ao_invt!); rounding errors spread per column."
function ao_out!(out::AbstractVector{UInt64}, F::Matrix{ComplexF64}, s::Int, P::AOPlan, add::Bool)
    c0 = (s - 1) * P.B2; p = P.p
    for k in 0:P.B2-1
        _ao_col!(out, k * p, F, (c0 + k) * p, p, Val(false), add)
        _ao_col!(out, (k + P.B2) * p, F, (c0 + k) * p, p, Val(true), add)
    end
    return out
end

@inline function _ao_col!(out, o::Int, F::Matrix{ComplexF64}, oc::Int, p::Int, ::Val{IM}, add::Bool) where {IM}
    s = zero(UInt64)
    @inbounds begin
        if add
            @simd for r in 1:p
                z = F[oc+r]; t = f2t(IM ? imag(z) : real(z)); out[o+r] += t; s += t
            end
        else
            @simd for r in 1:p
                z = F[oc+r]; t = f2t(IM ? imag(z) : real(z)); out[o+r] = t; s += t
            end
        end
        q, rr = fldmod(reinterpret(Int64, s), p)
        uq = reinterpret(UInt64, q)
        @simd for v in 1:p
            out[o+v] -= uq + (v <= rr)
        end
    end
    return nothing
end

"Fourier values of x (natural order)."
ao_fwd(x::AbstractVector, P::AOPlan) = ao_fwd!(ao_fill!(zeros(ComplexF64, P.p, P.B2), 1, ao_toR(x, P), P), P)
"Torus vector (natural order) of the Fourier values F."
ao_inv(F::Matrix{ComplexF64}, P::AOPlan) = ao_fromR(ao_out!(zeros(UInt64, P.p * P.B), ao_invt!(copy(F), P), 1, P, false), P)

"Columns of τ_c: target j ← source src[j], conjugated at -f where cj[j]."
function ao_taumap(c::Integer, B::Int)
    B2 = B ÷ 2; src = zeros(Int, B2); cj = falses(B2)
    for j in 0:B2-1
        e = mod((4j + 1) * c, 2B)
        if e % 4 == 1
            src[j+1] = (e - 1) ÷ 4 + 1
        else
            src[j+1] = (2B - e - 1) ÷ 4 + 1; cj[j+1] = true
        end
    end
    return src, cj
end

"Fourier values of τ_c(x) from those of x."
function ao_tau(F::Matrix{ComplexF64}, c::Integer, P::AOPlan)
    src, cj = ao_taumap(c, P.B)
    out = similar(F); n1 = P.n1; h1 = P.h1; p = P.p
    for j in 1:P.B2
        s = src[j]
        if !cj[j]
            out[:, j] .= F[:, s]
        else
            for i in 1:n1
                out[i, j] = conj(F[mod(i - 1 + h1, n1) + 1, s])
            end
            out[p, j] = conj(F[p, s])
        end
    end
    return out
end

# ---------------------------------------------------------------- keys (B ≥ 2)

"A key row (mask, body), split into real and imaginary parts."
struct ORow
    ar::Vector{Float64}; ai::Vector{Float64}; br::Vector{Float64}; bi::Vector{Float64}
end
ORow(A::Matrix{ComplexF64}, Bm::Matrix{ComplexF64}) = ORow(vec(real.(A)), vec(imag.(A)), vec(real.(Bm)), vec(imag.(Bm)))

struct OKS
    c::Int
    src::Vector{Int}; cj::Vector{Bool}   # τ_c on the columns
    tix::Vector{Int}                     # column of the table: 1 + (4j+1)(1-c)/2 mod B
    rows::Vector{ORow}                   # τ_c⁻¹(K_l)
end

struct AOKeys
    m::Int
    sk::ASecretKey
    s::Vector{Int64}
    bsk::Vector{Vector{Vector{ORow}}}    # bsk[t][π][r], r = 1..2ℓ
    ksk::Vector{OKS}                     # c ∈ U \ {1}
end

_otix(c::Integer, B::Int) = [1 + mod((4j + 1) * ((1 - c) ÷ 2), B) for j in 0:B÷2-1]

"""Keys of CaseA.akeygen (same random sequence: same secret key and same encryptions), for the optimized engine."""
function aokeygen(Rg::ARing, Pa::AParams, P::AOPlan, rng; m::Int = 1)
    Pa.n % m == 0 || error("n must be a multiple of m")
    sk = akeygen_sk(Rg, rng)
    s = sk.svec[1:Pa.n]
    z = zeros(UInt64, Rg.B * Rg.p)
    function ggsw(μ)
        rows = ORow[]
        for part in 1:2, l in 1:Pa.Gbr.ell
            c = aglwe_encrypt(Rg, sk, z, Pa.σ, rng)
            g = UInt64(μ) * gvalue(Pa.Gbr, l)
            t = part == 1 ? c.a : c.b
            t[1] += g; t[Rg.p] -= g
            push!(rows, ORow(ao_fwd(c.a, P), ao_fwd(c.b, P)))
        end
        return rows
    end
    bsk = [[ggsw(pattern(s, t, m) == j ? 1 : 0) for j in 1:(1<<m)-1] for t in 1:Pa.n÷m]
    ksk = OKS[]
    for c in Ugroup_A(Rg)[2:end]
        base = omega00_times(Rg, normR!(Rg, aut_A(Rg, sk.s, c)))
        ci = invmod(c, Rg.M)
        rows = ORow[]
        for l in 1:Pa.Gks.ell
            ct = aglwe_encrypt(Rg, sk, [reinterpret(UInt64, x) * gvalue(Pa.Gks, l) for x in base], Pa.σ, rng)
            push!(rows, ORow(ao_tau(ao_fwd(ct.a, P), ci, P), ao_tau(ao_fwd(ct.b, P), ci, P)))
        end
        src, cj = ao_taumap(c, Rg.B)
        push!(ksk, OKS(Int(c), src, cj, _otix(c, Rg.B), rows))
    end
    return AOKeys(m, sk, s, bsk, ksk)
end

# ---------------------------------------------------------------- pointwise kernels

"ACC slots (1, 2) (=|+=) D·(K_A, K_B), D = slot sd of FD (one polynomial: p B/2 values)."
function ao_mac!(ACC::Matrix{ComplexF64}, FD::Matrix{ComplexF64}, sd::Int, K::ORow, add::Bool)
    n = length(K.ar); od = (sd - 1) * n
    ar = K.ar; ai = K.ai; br = K.br; bi = K.bi
    if add
        @inbounds @simd for i in 1:n
            x = FD[od+i]; dr = real(x); di = imag(x)
            ACC[i] += complex(dr * ar[i] - di * ai[i], dr * ai[i] + di * ar[i])
            ACC[n+i] += complex(dr * br[i] - di * bi[i], dr * bi[i] + di * br[i])
        end
    else
        @inbounds @simd for i in 1:n
            x = FD[od+i]; dr = real(x); di = imag(x)
            ACC[i] = complex(dr * ar[i] - di * ai[i], dr * ai[i] + di * ar[i])
            ACC[n+i] = complex(dr * br[i] - di * bi[i], dr * bi[i] + di * br[i])
        end
    end
    return nothing
end

"""Key switching of the source column s, before τ: (oa, ob) = (0, B0) - Σ_l D_l·K'_l on that column (FK: slot 2 = B0,
slots 3.. = the digits), two digits per pass."""
function ao_kscol!(oa::Vector{ComplexF64}, ob::Vector{ComplexF64}, FK::Matrix{ComplexF64}, s::Int, rows::Vector{ORow},
                   P::AOPlan)
    p = P.p; B2 = P.B2; n = p * B2; ok = (s - 1) * p; ℓ = length(rows)
    ob0 = n + ok                                                          # B0, column s
    l = 1
    while l <= ℓ
        two = l + 1 <= ℓ
        o1 = (l + 1) * n + ok; K1 = rows[l]
        o2 = two ? (l + 2) * n + ok : o1; K2 = rows[two ? l + 1 : l]
        a1r = K1.ar; a1i = K1.ai; b1r = K1.br; b1i = K1.bi; a2r = K2.ar; a2i = K2.ai; b2r = K2.br; b2i = K2.bi
        w2 = two ? 1.0 : 0.0
        if l == 1
            @inbounds @simd for i in 1:p
                x = FK[o1+i]; d1r = real(x); d1i = imag(x); y = FK[o2+i]; d2r = w2 * real(y); d2i = w2 * imag(y)
                k = ok + i
                oa[i] = complex(d1i * a1i[k] - d1r * a1r[k] + d2i * a2i[k] - d2r * a2r[k],
                                -(d1r * a1i[k] + d1i * a1r[k] + d2r * a2i[k] + d2i * a2r[k]))
                ob[i] = FK[ob0+i] + complex(d1i * b1i[k] - d1r * b1r[k] + d2i * b2i[k] - d2r * b2r[k],
                                            -(d1r * b1i[k] + d1i * b1r[k] + d2r * b2i[k] + d2i * b2r[k]))
            end
        else
            @inbounds @simd for i in 1:p
                x = FK[o1+i]; d1r = real(x); d1i = imag(x); y = FK[o2+i]; d2r = w2 * real(y); d2i = w2 * imag(y)
                k = ok + i
                oa[i] += complex(d1i * a1i[k] - d1r * a1r[k] + d2i * a2i[k] - d2r * a2r[k],
                                 -(d1r * a1i[k] + d1i * a1r[k] + d2r * a2i[k] + d2i * a2r[k]))
                ob[i] += complex(d1i * b1i[k] - d1r * b1r[k] + d2i * b2i[k] - d2r * b2r[k],
                                 -(d1r * b1i[k] + d1i * b1r[k] + d2r * b2i[k] + d2i * b2r[k]))
            end
        end
        l += 2
    end
    return nothing
end

@inline _cmul_conj(q::ComplexF64, x::ComplexF64) = complex(real(q) * real(x) + imag(q) * imag(x),
                                                            imag(q) * real(x) - real(q) * imag(x))

"G slots (1, 2), column j += T[:, t] ⊙ (oa, ob), conjugated at -f if cj (τ applied to a source column)."
function ao_pubcol!(G::Matrix{ComplexF64}, j::Int, T::Matrix{ComplexF64}, t::Int, oa::Vector{ComplexF64},
                    ob::Vector{ComplexF64}, cj::Bool, P::AOPlan)
    p = P.p; n1 = P.n1; h1 = P.h1; n = p * P.B2; oA = (j - 1) * p; oB = n + oA; ot = (t - 1) * p
    @inbounds if !cj
        @simd for i in 1:p
            q = T[ot+i]
            G[oA+i] += q * oa[i]; G[oB+i] += q * ob[i]
        end
    else
        @simd for i in 1:h1
            q = T[ot+i]
            G[oA+i] += _cmul_conj(q, oa[i+h1]); G[oB+i] += _cmul_conj(q, ob[i+h1])
        end
        @simd for i in h1+1:n1
            q = T[ot+i]
            G[oA+i] += _cmul_conj(q, oa[i-h1]); G[oB+i] += _cmul_conj(q, ob[i-h1])
        end
        q = T[ot+p]
        G[oA+p] += _cmul_conj(q, oa[p]); G[oB+p] += _cmul_conj(q, ob[p])
    end
    return nothing
end

"G slots (1, 2) = T[:, 1] ⊙ (A0, B0) (c = 1; FK slots 1, 2)."
function ao_pubset!(G::Matrix{ComplexF64}, T::Matrix{ComplexF64}, FK::Matrix{ComplexF64}, P::AOPlan)
    p = P.p; n = p * P.B2
    @inbounds for j in 0:P.B2-1
        o = j * p
        @simd for i in 1:p
            q = T[i]
            G[o+i] = q * FK[o+i]; G[n+o+i] = q * FK[n+o+i]
        end
    end
    return nothing
end

"T[i, t+1] = Σ_k V_{f_i}^{a_k} e^{2iπ kt/B}: values of the public polynomials (rows of the Fourier values)."
function ao_table!(T::Matrix{ComplexF64}, acol::Vector{Int}, P::AOPlan, ptb)
    p = P.p; n1 = P.n1; dl = P.dl; Wn = P.Wn
    @inbounds for k in 1:P.B
        a = acol[k]
        if a == 0
            @simd for i in 1:p
                T[i, k] = 1.0
            end
        else
            d = dl[a+1]                                   # V_f^a = ω^{g^{d-q}} = Wn[q - d + n1 + 1] (f = g^{-q})
            @simd for i in 1:n1
                T[i, k] = Wn[i-d+n1]
            end
            T[p, k] = 1.0
        end
    end
    ptb * T
    return T
end

# ---------------------------------------------------------------- workspace and comb bootstrap

struct AOWork
    P::AOPlan
    m::Int
    FK::Matrix{ComplexF64}               # slots: A0, B0, digits of A0 (key switching)
    FD::Matrix{ComplexF64}               # slots: the 2ℓ digits of C_L - C
    G::Vector{Matrix{ComplexF64}}        # per pattern: slots C_L (mask, body)
    T::Vector{Matrix{ComplexF64}}        # per pattern: table of the public polynomials
    ACC::Matrix{ComplexF64}              # slots: accumulators of the external products
    oa::Vector{ComplexF64}; ob::Vector{ComplexF64}
    ta::Vector{UInt64}; tb::Vector{UInt64}; ap::Vector{UInt64}; cla::Vector{UInt64}; clb::Vector{UInt64}
    acol::Vector{Int}
    ptb::Any
end

function AOWork(Rg::ARing, Pa::AParams, P::AOPlan; m::Int = 1)
    p, B2 = P.p, P.B2; np = (1 << m) - 1
    z(ns) = zeros(ComplexF64, p, ns * B2); u() = zeros(UInt64, p * P.B)
    T = [zeros(ComplexF64, p, P.B) for _ in 1:np]
    ptb = plan_bfft!(T[1], 2; flags = FFTW.MEASURE)
    ws = AOWork(P, m, z(2 + Pa.Gks.ell), z(2Pa.Gbr.ell), [z(2) for _ in 1:np], T, z(2), zeros(ComplexF64, p),
                zeros(ComplexF64, p), u(), u(), u(), u(), u(), zeros(Int, P.B), ptb)
    _plans(P, 2 + Pa.Gks.ell); _plans(P, 2Pa.Gbr.ell); _plans(P, 2)
    return ws
end

"Line 4: (ta, tb) = C/B (rounded shift, augmented coordinates restored as divB_A; Rader order: v = p - 1 at pos[p])."
function ao_divB!(ws::AOWork, C::AGLWE, B::Int, P::AOPlan)
    s = trailing_zeros(B); p = P.p; f = P.pos[p]
    for (x, y) in ((C.a, ws.ta), (C.b, ws.tb))
        @inbounds for k in 0:B-1
            o = k * p; acc = zero(UInt64)
            @simd for v in 1:p
                t = (x[o+v] >> s) + ((x[o+v] >> (s - 1)) & 0x1); y[o+v] = t; acc += t
            end
            acc -= y[o+f]
            y[o+f] = -acc
        end
    end
    return nothing
end

"""
    combboot_A_opt(Rg, Pa, K, aints, bints; ws = AOWork(Rg, Pa, AOPlan(Rg); m = K.m))

Algorithm 1 in case (A) with the optimized engine (keys of aokeygen, mask grouped by K.m, h = 1): as combboot_A_fft.
"""
function combboot_A_opt(Rg::ARing, Pa::AParams, K::AOKeys, aints::AbstractMatrix{<:Integer},
                        bints::AbstractVector{<:Integer}; ws::AOWork = AOWork(Rg, Pa, AOPlan(Rg); m = K.m))
    P = ws.P; m = K.m; np = (1 << m) - 1; B = Rg.B; p = Rg.p
    C = AGLWE(zeros(UInt64, B * p), ao_toR(aacc0(Rg, R_of(Pa), bints), P))          # Rader order
    Gks = Pa.Gks; Gbr = Pa.Gbr; ℓ = Gbr.ell
    for t in 1:size(aints, 2)÷m
        ao_divB!(ws, C, B, P)                                                      # line 4
        ao_fill!(ws.FK, 1, ws.ta, P); ao_fill!(ws.FK, 2, ws.tb, P)
        ao_suffix!(ws.ap, ws.ta, nothing, B, P)
        ao_fill_digits!(ws.FK, 2, ws.ap, Gks, P)
        ao_fwd!(ws.FK, P)
        for π in 1:np                                                              # tables, c = 1
            for k in 1:B
                ws.acol[k] = pattern_exp(view(aints, k, :), t, π, m, p)
            end
            ao_table!(ws.T[π], ws.acol, P, ws.ptb)
            ao_pubset!(ws.G[π], ws.T[π], ws.FK, P)
        end
        for Kc in K.ksk, j in 1:P.B2                                               # lines 5-9
            ao_kscol!(ws.oa, ws.ob, ws.FK, Kc.src[j], Kc.rows, P)
            for π in 1:np
                ao_pubcol!(ws.G[π], j, ws.T[π], Kc.tix[j], ws.oa, ws.ob, Kc.cj[j], P)
            end
        end
        for π in 1:np                                                              # line 10
            G = ao_invt!(ws.G[π], P)
            ao_out!(ws.cla, G, 1, P, false); ao_out!(ws.clb, G, 2, P, false)
            ao_suffix!(ws.ap, ws.cla, C.a, B, P); ao_fill_digits!(ws.FD, 0, ws.ap, Gbr, P)
            ao_suffix!(ws.ap, ws.clb, C.b, B, P); ao_fill_digits!(ws.FD, ℓ, ws.ap, Gbr, P)
            ao_fwd!(ws.FD, P)
            rows = K.bsk[t][π]
            for r in 1:2ℓ
                ao_mac!(ws.ACC, ws.FD, r, rows[r], π > 1 || r > 1)
            end
        end
        ao_invt!(ws.ACC, P)
        ao_out!(C.a, ws.ACC, 1, P, true); ao_out!(C.b, ws.ACC, 2, P, true)
    end
    C = AGLWE(ao_fromR(C.a, P), ao_fromR(C.b, P))
    return [aextract(Rg, C, k) for k in 0:B-1]
end

# ---------------------------------------------------------------- error bootstrap (B = 1)
#
# :rader  two real polynomials x, y in one complex column x + i y (the transform of AOPlan(p, 2): no twist, no
#         transform along U), unpacked into the half spectra X, Y: rows q = 1..(p-1)/2 for f = g^{-(q-1)}, row (p+1)/2
#         for f = 0 (the values at -f are conjugate). The digits of the external product go by pairs (digit l of the
#         mask, digit l of the body), the inverse transform of the accumulator by pair (mask, body).
# :pad    zero padding to L = 2^⌈log2(2p - 2)⌉: real FFTs of the digits (batched), L/2 + 1 values, inverse real FFT of
#         the accumulator (mask and body batched), folded modulo V^p - 1.
# The accumulator stays in the coefficient domain (natural order); V^e acc - acc and the suffix sums of the division
# by Ω*_{0,0} are computed in one pass.

struct AOPlan1
    mode::Symbol
    p::Int; h::Int; L::Int
    R::Union{AOPlan,Nothing}       # :rader, the transform of pairs
    W::Matrix{ComplexF64}          # :rader, p × ℓ (pairs of digits)
    W1::Matrix{ComplexF64}         # :rader, p × 1 (accumulator)
    Xr::Matrix{Float64}            # :pad, L × 2ℓ (digits, rows p+1..L zero)
    Xo::Matrix{Float64}            # :pad, L × 2
    pr::Any; pbr::Any              # :pad, real FFT of the digits, inverse real FFT of the accumulator
    acc::Vector{Float64}           # :pad, fold
end

function AOPlan1(Rg::ARing, ℓ::Int; mode::Symbol = lpf(Rg.p - 1) <= 7 ? :rader : :pad)
    Rg.B == 1 || error("AOPlan1: B = 1")
    p = Rg.p
    if mode == :rader
        R = AOPlan(p, 2); _plans(R, ℓ); _plans(R, 1)
        return AOPlan1(:rader, p, (p + 1) ÷ 2, 0, R, zeros(ComplexF64, p, ℓ), zeros(ComplexF64, p, 1), zeros(0, 0),
                       zeros(0, 0), nothing, nothing, Float64[])
    else
        L = nextpow(2, 2p - 2); h = L ÷ 2 + 1
        Xr = zeros(L, 2ℓ); Xo = zeros(L, 2); Y = zeros(ComplexF64, h, 2ℓ); Y2 = zeros(ComplexF64, h, 2)
        pr = plan_rfft(Xr, 1; flags = FFTW.MEASURE); pbr = plan_brfft(Y2, L, 1; flags = FFTW.MEASURE)
        return AOPlan1(:pad, p, h, L, nothing, zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0), Xr, Xo, pr, pbr, zeros(p))
    end
end

"Half spectra (rows 1..h) of the slot of W into the columns cx, cy of D."
@inline function _unpack!(D::Matrix{ComplexF64}, cx::Int, cy::Int, W::Matrix{ComplexF64}, c::Int, p::Int)
    h1 = (p - 1) ÷ 2
    @inbounds begin
        @simd for q in 1:h1
            Zq = W[q, c]; Zc = conj(W[q+h1, c])
            D[q, cx] = 0.5 * (Zq + Zc)
            d = Zq - Zc
            D[q, cy] = complex(0.5 * imag(d), -0.5 * real(d))
        end
        z = W[p, c]; D[h1+1, cx] = real(z); D[h1+1, cy] = imag(z)
    end
    return nothing
end

"Slot c of W = the full spectrum of x + i y from the half spectra X = D[:, cx], Y = D[:, cy]."
@inline function _pack!(W::Matrix{ComplexF64}, c::Int, D::Matrix{ComplexF64}, cx::Int, cy::Int, p::Int)
    h1 = (p - 1) ÷ 2
    @inbounds begin
        @simd for q in 1:h1
            X = D[q, cx]; Y = D[q, cy]
            W[q, c] = complex(real(X) - imag(Y), imag(X) + real(Y))
            W[q+h1, c] = complex(real(X) + imag(Y), real(Y) - imag(X))
        end
        W[p, c] = complex(real(D[h1+1, cx]), real(D[h1+1, cy]))
    end
    return nothing
end

"Half spectra of two real polynomials x, y (natural order) into D[:, 1], D[:, 2] (keys, tests)."
function eb_fwd2(x::AbstractVector, y::AbstractVector, P1::AOPlan1)
    p = P1.p; D = zeros(ComplexF64, P1.h, 2)
    if P1.mode == :rader
        R = P1.R; W = zeros(ComplexF64, p, 1)
        for v in 0:p-1
            W[R.pos[v+1], 1] = complex(tofloat(x[v+1]), tofloat(y[v+1]))
        end
        ao_fwd!(W, R); _unpack!(D, 1, 2, W, 1, p)
    else
        Xr = zeros(P1.L, 2)
        for v in 1:p
            Xr[v, 1] = tofloat(x[v]); Xr[v, 2] = tofloat(y[v])
        end
        D .= rfft(Xr, 1)
    end
    return D
end

"Torus vectors (natural order) of the half spectra D[:, 1], D[:, 2] (tests)."
function eb_inv2(D::Matrix{ComplexF64}, P1::AOPlan1)
    a = zeros(UInt64, P1.p); b = zeros(UInt64, P1.p)
    eb_inv_add!(a, b, copy(D), P1)
    return a, b
end

"""ap = (V^e x - x)/Ω*_{0,0} (B = 1, x natural order): suffix sums of d_v = x_{v-e} - x_v, v ≤ p - 2; written in
Rader order (rader = true: position pos[v+1]) or in natural order."""
function eb_rotsuffix!(ap::Vector{UInt64}, x::Vector{UInt64}, e::Int, p::Int, pos::Vector{Int}, rader::Bool)
    s = zero(UInt64)
    @inbounds if rader
        ap[pos[p]] = 0
        for v in p-2:-1:0
            w = v - e; w < 0 && (w += p)
            s += x[w+1] - x[v+1]; ap[pos[v+1]] = s
        end
    else
        ap[p] = 0
        for v in p-2:-1:0
            w = v - e; w < 0 && (w += p)
            s += x[w+1] - x[v+1]; ap[v+1] = s
        end
    end
    return ap
end

"D[:, r] = half spectra of the 2ℓ balanced digits of (apa, apb) (r ≤ ℓ: mask, r > ℓ: body)."
function eb_fwd_digits!(D::Matrix{ComplexF64}, apa::Vector{UInt64}, apb::Vector{UInt64}, G::Gadget, P1::AOPlan1)
    p = P1.p; ℓ = G.ell
    C, mask, half = _adconst(G)
    if P1.mode == :rader
        W = P1.W
        @inbounds for l in 1:ℓ
            sh = (64 - G.off - G.logB * l) & 63
            @simd for r in 1:p
                W[r, l] = complex(_dig(apa[r], C, sh, mask, half), _dig(apb[r], C, sh, mask, half))
            end
        end
        ao_fwd!(W, P1.R)
        for l in 1:ℓ
            _unpack!(D, l, ℓ + l, W, l, p)
        end
    else
        Xr = P1.Xr
        @inbounds for l in 1:ℓ
            sh = (64 - G.off - G.logB * l) & 63
            @simd for v in 1:p
                Xr[v, l] = _dig(apa[v], C, sh, mask, half); Xr[v, ℓ+l] = _dig(apb[v], C, sh, mask, half)
            end
        end
        mul!(D, P1.pr, Xr)
    end
    return D
end

"(a, b) += the torus vectors (natural order) of the half spectra ACC[:, 1], ACC[:, 2] (ACC destroyed)."
function eb_inv_add!(a::Vector{UInt64}, b::Vector{UInt64}, ACC::Matrix{ComplexF64}, P1::AOPlan1)
    p = P1.p
    if P1.mode == :rader
        R = P1.R; W1 = P1.W1
        _pack!(W1, 1, ACC, 1, 2, p)
        ao_invt!(W1, R)
        _eb_col!(a, W1, R.gp, p, Val(false)); _eb_col!(b, W1, R.gp, p, Val(true))
    else
        mul!(P1.Xo, P1.pbr, ACC)
        L = P1.L; acc = P1.acc; iL = 1 / L
        for (c, out) in ((1, a), (2, b))
            fill!(acc, 0.0)
            v = 0
            @inbounds for w in 1:L
                v += 1; v > p && (v = 1)
                acc[v] += P1.Xo[w, c]
            end
            s = zero(UInt64)
            @inbounds for v in 1:p
                t = f2t(acc[v] * iL); out[v] += t; s += t
            end
            _spread!(out, s, p)
        end
    end
    return nothing
end

@inline function _eb_col!(out::Vector{UInt64}, W::Matrix{ComplexF64}, gp::Vector{Int}, p::Int, ::Val{IM}) where {IM}
    s = zero(UInt64)
    @inbounds begin
        z = W[p, 1]; t = f2t(IM ? imag(z) : real(z)); out[1] += t; s += t
        for r in 1:p-1
            z = W[r, 1]; t = f2t(IM ? imag(z) : real(z)); out[gp[r]+1] += t; s += t
        end
    end
    _spread!(out, s, p)
    return nothing
end

@inline function _spread!(out::Vector{UInt64}, s::UInt64, p::Int)
    q, rr = fldmod(reinterpret(Int64, s), p)
    uq = reinterpret(UInt64, q)
    @inbounds @simd for v in 1:p
        out[v] -= uq + (v <= rr)
    end
    return nothing
end

"Bootstrapping keys of the error bootstrap: rows (half spectra of mask and body), per group and pattern."
struct AOKeys1
    m::Int
    sk::ASecretKey
    s::Vector{Int64}
    bsk::Vector{Vector{Vector{ORow}}}    # bsk[t][π][r]: half spectra of mask and body, split
end

"Keys of CaseA.akeygen (same random sequence) for the optimized error bootstrap."
function aokeygen1(Rg::ARing, Pa::AParams, P1::AOPlan1, rng; m::Int = 1)
    Pa.n % m == 0 || error("n must be a multiple of m")
    sk = akeygen_sk(Rg, rng)
    s = sk.svec[1:Pa.n]
    z = zeros(UInt64, Rg.p)
    function ggsw(μ)
        rows = ORow[]
        for part in 1:2, l in 1:Pa.Gbr.ell
            c = aglwe_encrypt(Rg, sk, z, Pa.σ, rng)
            g = UInt64(μ) * gvalue(Pa.Gbr, l)
            t = part == 1 ? c.a : c.b
            t[1] += g; t[Rg.p] -= g
            D = eb_fwd2(c.a, c.b, P1)
            push!(rows, ORow(D[:, 1:1], D[:, 2:2]))
        end
        return rows
    end
    bsk = [[ggsw(pattern(s, t, m) == j ? 1 : 0) for j in 1:(1<<m)-1] for t in 1:Pa.n÷m]
    return AOKeys1(m, sk, s, bsk)
end

struct AOWork1
    P1::AOPlan1
    D::Matrix{ComplexF64}          # h × 2ℓ
    ACC::Matrix{ComplexF64}        # h × 2
    apa::Vector{UInt64}; apb::Vector{UInt64}
end
AOWork1(Rg::ARing, Pa::AParams; mode::Symbol = lpf(Rg.p - 1) <= 7 ? :rader : :pad) =
    (P1 = AOPlan1(Rg, Pa.Gbr.ell; mode); AOWork1(P1, zeros(ComplexF64, P1.h, 2Pa.Gbr.ell), zeros(ComplexF64, P1.h, 2),
                                                 zeros(UInt64, Rg.p), zeros(UInt64, Rg.p)))

"ACC (=|+=) Σ_r D[:, r]·K_r (keys split: real and imaginary parts)."
function eb_mac!(ACC::Matrix{ComplexF64}, D::Matrix{ComplexF64}, rows::Vector{ORow}, add::Bool)
    h = size(D, 1)
    for (r, K) in enumerate(rows)
        ar = K.ar; ai = K.ai; br = K.br; bi = K.bi; od = (r - 1) * h
        if r == 1 && !add
            @inbounds @simd for q in 1:h
                x = D[od+q]; dr = real(x); di = imag(x)
                ACC[q] = complex(dr * ar[q] - di * ai[q], dr * ai[q] + di * ar[q])
                ACC[h+q] = complex(dr * br[q] - di * bi[q], dr * bi[q] + di * br[q])
            end
        else
            @inbounds @simd for q in 1:h
                x = D[od+q]; dr = real(x); di = imag(x)
                ACC[q] += complex(dr * ar[q] - di * ai[q], dr * ai[q] + di * ar[q])
                ACC[h+q] += complex(dr * br[q] - di * bi[q], dr * bi[q] + di * br[q])
            end
        end
    end
    return nothing
end

"""
    blind_rotation_A_opt(Rg, Pa, K, arow, bint; ws = AOWork1(Rg, Pa))

The error bootstrap (B = 1) with the optimized engine (keys of aokeygen1, mask grouped by K.m), as blind_rotation_A.
"""
function blind_rotation_A_opt(Rg::ARing, Pa::AParams, K::AOKeys1, arow::AbstractVector{<:Integer}, bint::Integer;
                              ws::AOWork1 = AOWork1(Rg, Pa))
    P1 = ws.P1; m = K.m; p = Rg.p; G = Pa.Gbr
    rader = P1.mode == :rader; pos = rader ? P1.R.pos : Int[]
    C = AGLWE(zeros(UInt64, p), aacc0(Rg, R_of(Pa), [bint]))
    for t in 1:length(arow)÷m
        for j in 1:(1<<m)-1
            e = pattern_exp(arow, t, j, m, p)
            eb_rotsuffix!(ws.apa, C.a, e, p, pos, rader); eb_rotsuffix!(ws.apb, C.b, e, p, pos, rader)
            eb_fwd_digits!(ws.D, ws.apa, ws.apb, G, P1)
            eb_mac!(ws.ACC, ws.D, K.bsk[t][j], j > 1)
        end
        eb_inv_add!(C.a, C.b, ws.ACC, P1)
    end
    return aextract(Rg, C, 0)
end


# ---------------------------------------------------------------- whole refresh

"""
    refresh_A_opt(Rg, Pa, K, cts; ws, ksk = nothing, pool = nothing, Vstar = Inf, recenter = true, rng, info = false)

CaseA.refresh_A with the optimized engine: key switch N -> n, rounding at R = p̂ p with recentering and selection,
rotation of the errors (blind_rotation_A_opt with the keys of aokeygen1 for B = 1, combboot_A_opt with those of aokeygen
for B ≥ 2; ws: AOWork1 or AOWork), subtraction.
"""
function refresh_A_opt(Rg::ARing, Pa::AParams, K::Union{AOKeys,AOKeys1}, cts::AbstractVector{LWE};
                       ws::Union{AOWork,AOWork1}, ksk::Union{StairKey,Nothing} = nothing, pool = nothing,
                       Vstar::Real = Inf, recenter::Bool = true, rng::AbstractRNG = Random.default_rng(),
                       info::Bool = false)
    length(cts) == Rg.B || error("B = $(Rg.B) ciphertexts expected")
    R = R_of(Pa); n = Pa.n
    t0 = time_ns()
    lw = ksk === nothing ? cts : [stair_ks(ksk, ct) for ct in cts]
    t1 = time_ns()
    rs = [round_select(ct, R, pool; Vstar, recenter, rng) for ct in lw]
    t2 = time_ns()
    aints = [mod(rs[k].abar[u], Pa.p) for k in 1:Rg.B, u in 1:n]
    bints = [mod(rs[k].bbar, Pa.p) for k in 1:Rg.B]
    errs = Rg.B == 1 ? [blind_rotation_A_opt(Rg, Pa, K, view(aints, 1, :), bints[1]; ws)] :
                       combboot_A_opt(Rg, Pa, K, aints, bints; ws)
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
