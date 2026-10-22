# FFT backend of case (A) (src/caseA.jl), as fast.jl for case (B): products in Float64 through FFTW, the error of the
# FFT being an extra noise. The encryptions of the keys are exact (caseA.jl); their Fourier values are stored.
#
# Fourier representation. Z[U]/(U^B + 1) ⊗ Z[V]/(V^p - 1): an augmented vector x (entry k p + v + 1, i.e. the p × B
# matrix x[v, k]) is represented by its values at U_j = e^{-iπ(2j+1)/B} (j < B) and V_f = e^{-2iπ f/p} for h = (p + 1)/2
# values of f, one in each pair {f, -f} (the values at -f are conjugate, x being real): a DFT of size p per column, a
# twist by e^{-iπk/B} and a complex FFT of size B per row; h B values (N/2 + B/2). The products in R
# (negacyclic in k, cyclic in v) are pointwise, for any lift of the element of R: a multiple of Φ_p(V) = Σ_v V^v adds
# Σ_v x[k][v] ≡ 0 (mod 2^64) to each coordinate of the column k. The inverse transform ends by restoring
# Σ_v x[k][v] ≡ 0 (mod 2^64), which the rounding errors of the FFT break: their sum over a column (a small integer e) is
# spread over its p coordinates (e/p each) instead of being put on one (fixaug!). (Subtracting the constant c with
# p c ≡ e (mod 2^64) would not do: the constant columns j/p are not zero in T^∨.)
#
# The DFT of prime size p, three implementations (AFPlan(Rg; mode)):
#   :rader  Rader's algorithm: two real columns form one complex column, whose DFT at f = g^{-q} (g a primitive root
#           mod p) is a cyclic convolution of size p - 1 (two complex FFTs of size p - 1). B ≥ 2: the columns 2κ, 2κ+1
#           of one polynomial; B = 1: two polynomials (affwd2!, afinv2!: the digits of the external product by pairs,
#           mask and body). Fast when p - 1 is smooth (p = 257: 256; p = 2017: 2^5·3^2·7). Default for B ≥ 2, and for
#           B = 1 when p - 1 is 7-smooth;
#   :pad    zero padding to L = 2^⌈log2(2p - 2)⌉ (real FFT of size L, folded modulo V^p - 1 by the inverse): exact for a
#           product of an element of R in Ω form (degree ≤ p - 2) by a torus element (degree ≤ p - 1), and for sums of
#           such products, but not for chained products (the comb bootstrap multiplies the key switchings, kept in the
#           Fourier domain, by the public polynomials); L/2 + 1 values per column (2× the others). Default for B = 1
#           when p - 1 is not smooth (p = 2039: p - 1 = 2·1019);
#   :fftw   the real FFT of size p of FFTW (slow for a prime size: about 15 times a power of 2 of the same size).
#
# τ_c, c ≡ 1 (mod p) (the automorphisms of the comb), fixes V and maps U_j to U_j', 2j' + 1 ≡ c (2j + 1) (mod 2B): a
# permutation of the columns of the Fourier values, without conjugation.
# The public polynomial B·A_c = Σ_k V^{a_k} U^{k(1-c)} takes at (U_j, V_f) the value T[f, t] = Σ_k V_f^{a_k} e^{-2iπ tk/B},
# t = (2j + 1)(1 - c)/2 mod B: one table of h × B values per pattern, shared by the B automorphisms.
#
# Usage:  include("src/CombBoot.jl"); using .CombBoot; include("src/fast.jl"); using .Fast
#         include("src/caseA.jl"); using .CaseA; include("src/caseA_fast.jl"); using .CaseAFast

module CaseAFast

using FFTW, Random, LinearAlgebra
using ..CombBoot
import ..CombBoot: Gadget, gvalue, LWE, torus_frac, round_select
import ..Fast: StairKey, stair_ks, f2t
using ..CaseA
import ..CaseA: ARing, AGLWE, ASecretKey, AParams, adecompose, aglwe_encrypt, akeygen_sk, aut_A, normR!, omega00_times,
                aextract, aacc0, divB_A, rotate_v, Ugroup_A, pattern, pattern_exp, R_of

export AFPlan, affwd, affwd!, afinv, afinv!, AFGGSW, AFKS, AFKeys, afggsw, afks_keygen, afkeygen, afextprod,
       afextprod_acc!, afks_hoisted, afks, AFComb, aftable!, afpublic!, blind_rotation_A_fft, combboot_A_fft,
       refresh_A_fft, afperm, fexps, affwd2!, afinv2!, afglwe

# ---------------------------------------------------------------- transforms

abstract type AFP end

@inline tofloat(t::UInt64) = Float64(reinterpret(Int64, t))
@inline tofloat(t::Integer) = Float64(t)

"The negacyclic part along k: twist and B-point FFT of the rows (forward), or the inverse (normalized by 1/B)."
function _ufwd!(Z::Matrix{ComplexF64}, fp::AFP)
    fp.B == 1 && return Z
    @inbounds for k in 1:fp.B
        w = fp.tw[k]
        @simd for f in 1:fp.h
            Z[f, k] *= w
        end
    end
    fp.pf * Z
    return Z
end
function _uinv!(Y::Matrix{ComplexF64}, fp::AFP)
    fp.B > 1 && (fp.pb * Y)
    @inbounds for k in 1:fp.B
        w = fp.itw[k]
        @simd for f in 1:fp.h
            Y[f, k] *= w
        end
    end
    return Y
end
_utw(B) = ([cispi(-k / B) for k in 0:B-1], [cispi(k / B) / B for k in 0:B-1])

"Rounding modulo 2^64 of the column k (values c(v), v = 1..p) and restoration of Σ_v ≡ 0 (rounding errors spread)."
@inline function _round_column!(out::AbstractVector{UInt64}, k::Int, p::Int, c)
    s = zero(UInt64)
    @inbounds for v in 1:p
        t = f2t(c(v)); out[k*p+v] = t; s += t
    end
    q, r = fldmod(reinterpret(Int64, s), p)
    uq = reinterpret(UInt64, q)
    @inbounds for v in 1:p
        out[k*p+v] -= uq + (v <= r)
    end
    return nothing
end

# -------- :fftw, real FFT of size p

struct AFPlanF{PR,PB,PF,PI} <: AFP
    p::Int; B::Int; h::Int
    xr::Matrix{Float64}           # p × B
    Y::Matrix{ComplexF64}         # h × B
    tw::Vector{ComplexF64}        # e^{-iπk/B}
    itw::Vector{ComplexF64}       # e^{iπk/B}/B
    prf::PR                       # real FFT along v
    pbr::PB                       # unnormalized inverse real FFT along v (destroys its input)
    pf::PF                        # in-place FFT along k (B > 1)
    pb::PI                        # in-place unnormalized inverse FFT along k (B > 1)
end

function AFPlanF(Rg::ARing; flags = FFTW.MEASURE)
    p, B = Rg.p, Rg.B; h = (p + 1) ÷ 2
    xr = zeros(Float64, p, B); Y = zeros(ComplexF64, h, B)
    prf = plan_rfft(xr, 1; flags); pbr = plan_brfft(Y, p, 1; flags)
    pf = B > 1 ? plan_fft!(Y, 2; flags) : nothing
    pb = B > 1 ? plan_bfft!(Y, 2; flags) : nothing
    return AFPlanF(p, B, h, xr, Y, _utw(B)..., prf, pbr, pf, pb)
end

fexps(fp::AFPlanF) = collect(0:fp.h-1)

function affwd!(Z::Matrix{ComplexF64}, x::AbstractVector, fp::AFPlanF)
    xr = fp.xr
    @inbounds for i in eachindex(xr)
        xr[i] = tofloat(x[i])
    end
    mul!(Z, fp.prf, xr)
    return _ufwd!(Z, fp)
end

function afinv!(out::AbstractVector{UInt64}, Z::AbstractMatrix{ComplexF64}, fp::AFPlanF)
    p = fp.p
    Y = fp.Y; copyto!(Y, Z); _uinv!(Y, fp)
    mul!(fp.xr, fp.pbr, Y)
    xr = fp.xr; ip = 1 / p
    for k in 0:fp.B-1
        _round_column!(out, k, p, v -> xr[v, k+1] * ip)
    end
    return out
end

# -------- :rader, Rader's algorithm on pairs of real columns

struct AFPlanR{PA,PI,PF,PB} <: AFP
    p::Int; B::Int; h::Int
    np::Int                       # pairs per transform: B/2, or 1 for B = 1 (two polynomials: affwd2!, afinv2!)
    gp::Vector{Int}               # gp[r+1] = g^r mod p, r = 0..p-2
    fx::Vector{Int}               # row i: f = fx[i] (fx[1] = 0, fx[q+2] = g^{-q}, q < (p - 1)/2)
    bh::Vector{ComplexF64}        # FFT(b)/(p - 1), b_s = ω^{g^{-s}}, ω = e^{-2iπ/p}
    βh::Vector{ComplexF64}        # FFT(β)/((p - 1) p), β_s = ω^{-g^s}
    A::Matrix{ComplexF64}         # (p - 1) × np
    s0::Vector{ComplexF64}        # np
    s1::Vector{ComplexF64}        # np
    Y::Matrix{ComplexF64}         # h × B
    Y2::Matrix{ComplexF64}        # h × B (B = 1: second polynomial; zero for a single inverse)
    Zs::Matrix{ComplexF64}        # h × B scratch (B = 1: single forward)
    zx::Vector{Int64}             # p zeros (B = 1: single forward)
    os::Vector{UInt64}            # p scratch (B = 1: single inverse)
    tw::Vector{ComplexF64}
    itw::Vector{ComplexF64}
    pa::PA                        # in-place FFT of size p - 1 of the columns of A
    pai::PI                       # in-place unnormalized inverse
    pf::PF
    pb::PB
end

function primroot(p::Int)
    fs = [q for q in 2:p-1 if (p - 1) % q == 0 && all(q % d != 0 for d in 2:isqrt(q))]
    return first(g for g in 2:p-1 if all(powermod(g, (p - 1) ÷ q, p) != 1 for q in fs))
end

function AFPlanR(Rg::ARing; flags = FFTW.MEASURE)
    p, B = Rg.p, Rg.B
    h = (p + 1) ÷ 2; n1 = p - 1; g = primroot(p); gi = invmod(g, p); np = max(1, B ÷ 2)
    gp = [powermod(g, r, p) for r in 0:n1-1]
    fx = vcat(0, [powermod(gi, q, p) for q in 0:n1÷2-1])
    A = zeros(ComplexF64, n1, np); Y = zeros(ComplexF64, h, B)
    pa = plan_fft!(A, 1; flags); pai = plan_bfft!(A, 1; flags)
    b = [cispi(-2 * powermod(gi, s, p) / p) for s in 0:n1-1]
    β = [cispi(2 * powermod(g, s, p) / p) for s in 0:n1-1]
    bh = fft(b) ./ n1; βh = fft(β) ./ (n1 * p)
    pf = B > 1 ? plan_fft!(Y, 2; flags) : nothing
    pb = B > 1 ? plan_bfft!(Y, 2; flags) : nothing
    return AFPlanR(p, B, h, np, gp, fx, bh, βh, A, zeros(ComplexF64, np), zeros(ComplexF64, np), Y, zeros(ComplexF64, h, B),
                   zeros(ComplexF64, h, B), zeros(Int64, p), zeros(UInt64, p), _utw(B)..., pa, pai, pf, pb)
end

fexps(fp::AFPlanR) = fp.fx

"Pair κ (forward): A[:, κ] = (x_a + i x_b)(g^r), s0[κ] = Σ_v (x_a + i x_b)(v); x_a, x_b read at the offsets oa, ob."
@inline function _rgather!(fp::AFPlanR, κ::Int, xa, oa::Int, xb, ob::Int)
    p = fp.p; A = fp.A; gp = fp.gp
    S = zero(ComplexF64)
    @inbounds for v in 1:p
        S += complex(tofloat(xa[oa+v]), tofloat(xb[ob+v]))
    end
    fp.s0[κ] = S
    @inbounds for r in 1:p-1
        v = gp[r] + 1
        A[r, κ] = complex(tofloat(xa[oa+v]), tofloat(xb[ob+v]))
    end
    return nothing
end

"Cyclic convolutions of size p - 1 of the columns of A with the kernel of Fourier values kh."
function _rconv!(fp::AFPlanR, kh::Vector{ComplexF64})
    A = fp.A
    fp.pa * A
    @inbounds for κ in 1:fp.np
        @simd for r in 1:fp.p-1
            A[r, κ] *= kh[r]
        end
    end
    fp.pai * A
    return nothing
end

"Pair κ (forward): the values at f = 0 and f = g^{-q} (q < (p-1)/2) of x_a and x_b, to Za[:, ka] and Zb[:, kb]."
@inline function _runpack!(fp::AFPlanR, κ::Int, xa, oa::Int, xb, ob::Int, Za, ka::Int, Zb, kb::Int)
    h1 = (fp.p - 1) ÷ 2; A = fp.A
    @inbounds begin
        z0 = complex(tofloat(xa[oa+1]), tofloat(xb[ob+1]))
        S = fp.s0[κ]
        Za[1, ka] = real(S); Zb[1, kb] = imag(S)
        for q in 1:h1
            Zq = z0 + A[q, κ]; Zc = conj(z0 + A[q+h1, κ])
            Za[q+1, ka] = 0.5 * (Zq + Zc)
            d = Zq - Zc
            Zb[q+1, kb] = complex(0.5 * imag(d), -0.5 * real(d))      # (Zq - conj(Zc))/(2i)
        end
    end
    return nothing
end

"Pair κ (inverse): the full spectrum of x_a + i x_b from the values Ya[:, ka], Yb[:, kb] (conjugates at -f), in A."
@inline function _rpack!(fp::AFPlanR, κ::Int, Ya, ka::Int, Yb, kb::Int)
    p = fp.p; h1 = (p - 1) ÷ 2; A = fp.A
    @inbounds begin
        Z0 = complex(real(Ya[1, ka]), real(Yb[1, kb]))
        S = Z0
        for q in 1:h1
            X0 = Ya[q+1, ka]; X1 = Yb[q+1, kb]
            Zq = complex(real(X0) - imag(X1), imag(X0) + real(X1))        # X0 + i X1 at f = g^{-q}
            Zc = complex(real(X0) + imag(X1), real(X1) - imag(X0))        # conj(X0) + i conj(X1) at -f
            A[q, κ] = Zq; A[q+h1, κ] = Zc
            S += Zq + Zc
        end
        fp.s0[κ] = Z0 / p; fp.s1[κ] = S / p
    end
    return nothing
end

"Pair κ (inverse): z_0 = S/p, z_{g^r} = Z0/p + A[r]; real (IM = false) or imaginary part, rounded, to out[o + 1..o + p]."
@inline function _rscatter!(out, o::Int, fp::AFPlanR, κ::Int, ::Val{IM}) where {IM}
    p = fp.p; A = fp.A; gp = fp.gp
    z00 = IM ? imag(fp.s1[κ]) : real(fp.s1[κ]); c0 = IM ? imag(fp.s0[κ]) : real(fp.s0[κ])
    s = zero(UInt64)
    @inbounds begin
        t = f2t(z00); out[o+1] = t; s += t
        for r in 1:p-1
            a = A[r, κ]
            t = f2t(c0 + (IM ? imag(a) : real(a))); out[o+gp[r]+1] = t; s += t
        end
        q, rr = fldmod(reinterpret(Int64, s), p)
        uq = reinterpret(UInt64, q)
        for v in 1:p
            out[o+v] -= uq + (v <= rr)
        end
    end
    return nothing
end

function affwd!(Z::Matrix{ComplexF64}, x::AbstractVector, fp::AFPlanR)
    fp.B == 1 && (affwd2!(Z, fp.Zs, x, fp.zx, fp); return Z)
    p = fp.p
    for κ in 1:fp.np
        _rgather!(fp, κ, x, (2κ - 2) * p, x, (2κ - 1) * p)
    end
    _rconv!(fp, fp.bh)
    for κ in 1:fp.np
        _runpack!(fp, κ, x, (2κ - 2) * p, x, (2κ - 1) * p, Z, 2κ - 1, Z, 2κ)
    end
    return _ufwd!(Z, fp)
end

function afinv!(out::AbstractVector{UInt64}, Z::AbstractMatrix{ComplexF64}, fp::AFPlanR)
    if fp.B == 1
        fill!(fp.Zs, 0)
        afinv2!(out, fp.os, Z, fp.Zs, fp)
        return out
    end
    p = fp.p
    Y = fp.Y; copyto!(Y, Z); _uinv!(Y, fp)
    for κ in 1:fp.np
        _rpack!(fp, κ, Y, 2κ - 1, Y, 2κ)
    end
    _rconv!(fp, fp.βh)
    for κ in 1:fp.np
        _rscatter!(out, (2κ - 2) * p, fp, κ, Val(false))
        _rscatter!(out, (2κ - 1) * p, fp, κ, Val(true))
    end
    return out
end

"Two forward transforms (one Rader transform for B = 1: x1 and x2 must have comparable sizes, the error being relative to the larger)."
function affwd2!(Z1::Matrix{ComplexF64}, Z2::Matrix{ComplexF64}, x1::AbstractVector, x2::AbstractVector, fp::AFP)
    affwd!(Z1, x1, fp); affwd!(Z2, x2, fp)
    return nothing
end
function affwd2!(Z1::Matrix{ComplexF64}, Z2::Matrix{ComplexF64}, x1::AbstractVector, x2::AbstractVector, fp::AFPlanR)
    fp.B == 1 || return invoke(affwd2!, Tuple{Matrix{ComplexF64},Matrix{ComplexF64},AbstractVector,AbstractVector,AFP},
                               Z1, Z2, x1, x2, fp)
    _rgather!(fp, 1, x1, 0, x2, 0)
    _rconv!(fp, fp.bh)
    _runpack!(fp, 1, x1, 0, x2, 0, Z1, 1, Z2, 1)
    return nothing
end

"Two inverse transforms (one Rader transform for B = 1); Z1, Z2 are preserved."
function afinv2!(o1::AbstractVector{UInt64}, o2::AbstractVector{UInt64}, Z1::AbstractMatrix{ComplexF64},
                 Z2::AbstractMatrix{ComplexF64}, fp::AFP)
    afinv!(o1, Z1, fp); afinv!(o2, Z2, fp)
    return nothing
end
function afinv2!(o1::AbstractVector{UInt64}, o2::AbstractVector{UInt64}, Z1::AbstractMatrix{ComplexF64},
                 Z2::AbstractMatrix{ComplexF64}, fp::AFPlanR)
    fp.B == 1 || return invoke(afinv2!, Tuple{AbstractVector{UInt64},AbstractVector{UInt64},AbstractMatrix{ComplexF64},
                                               AbstractMatrix{ComplexF64},AFP}, o1, o2, Z1, Z2, fp)
    _rpack!(fp, 1, Z1, 1, Z2, 1)                                     # B = 1: no transform along k
    _rconv!(fp, fp.βh)
    _rscatter!(o1, 0, fp, 1, Val(false))
    _rscatter!(o2, 0, fp, 1, Val(true))
    return nothing
end

# -------- :pad, zero padding to a power of 2

struct AFPlanZ{PR,PB,PF,PI} <: AFP
    p::Int; B::Int; h::Int; L::Int
    xr::Matrix{Float64}           # L × B
    Y::Matrix{ComplexF64}         # h × B, h = L/2 + 1
    tw::Vector{ComplexF64}
    itw::Vector{ComplexF64}
    prf::PR; pbr::PB; pf::PF; pb::PI
    acc::Vector{Float64}          # p
end

function AFPlanZ(Rg::ARing; flags = FFTW.MEASURE)
    p, B = Rg.p, Rg.B; L = nextpow(2, 2p - 2); h = L ÷ 2 + 1
    xr = zeros(Float64, L, B); Y = zeros(ComplexF64, h, B)
    prf = plan_rfft(xr, 1; flags); pbr = plan_brfft(Y, L, 1; flags)
    pf = B > 1 ? plan_fft!(Y, 2; flags) : nothing
    pb = B > 1 ? plan_bfft!(Y, 2; flags) : nothing
    return AFPlanZ(p, B, h, L, xr, Y, _utw(B)..., prf, pbr, pf, pb, zeros(p))
end

fexps(fp::AFPlanZ) = error(":pad does not support the public products of the comb bootstrap (chained products)")

function affwd!(Z::Matrix{ComplexF64}, x::AbstractVector, fp::AFPlanZ)
    xr = fp.xr; p = fp.p
    fill!(xr, 0.0)
    @inbounds for k in 1:fp.B, v in 1:p
        xr[v, k] = tofloat(x[(k-1)*p+v])
    end
    mul!(Z, fp.prf, xr)
    return _ufwd!(Z, fp)
end

function afinv!(out::AbstractVector{UInt64}, Z::AbstractMatrix{ComplexF64}, fp::AFPlanZ)
    p, L = fp.p, fp.L
    Y = fp.Y; copyto!(Y, Z); _uinv!(Y, fp)
    mul!(fp.xr, fp.pbr, Y)
    xr = fp.xr; acc = fp.acc; iL = 1 / L
    for k in 0:fp.B-1
        fill!(acc, 0.0)
        v = 0
        @inbounds for w in 1:L                               # fold modulo V^p - 1
            v += 1; v > p && (v = 1)
            acc[v] += xr[w, k+1]
        end
        _round_column!(out, k, p, v -> acc[v] * iL)
    end
    return out
end

"Largest prime factor of n."
lpf(n::Int) = (f = 1; d = 2; while n > 1; while n % d == 0; n ÷= d; f = d; end; d += 1; end; f)

"""FFT plan of case (A): mode :rader (default if B ≥ 2 or p - 1 is 7-smooth), :pad (default otherwise; no chained
products), :fftw."""
AFPlan(Rg::ARing; mode::Symbol = Rg.B >= 2 || lpf(Rg.p - 1) <= 7 ? :rader : :pad, flags = FFTW.MEASURE) =
    mode == :rader ? AFPlanR(Rg; flags) : mode == :pad ? AFPlanZ(Rg; flags) : mode == :fftw ? AFPlanF(Rg; flags) :
    error("mode: :rader, :pad or :fftw")

affwd(x::AbstractVector, fp::AFP) = affwd!(Matrix{ComplexF64}(undef, fp.h, fp.B), x, fp)
afinv(Z::AbstractMatrix{ComplexF64}, fp::AFP) = afinv!(Vector{UInt64}(undef, fp.p * fp.B), Z, fp)

"Permutation of the columns of the Fourier values for τ_c (c ≡ 1 mod p): (τ_c Z)[:, j] = Z[:, perm[j]]."
afperm(Rg::ARing, c::Integer) = [(mod(c * (2j + 1), 2Rg.B) - 1) ÷ 2 + 1 for j in 0:Rg.B-1]

# ---------------------------------------------------------------- keys

const ARow = NTuple{2,Matrix{ComplexF64}}

struct AFGGSW
    rows::Vector{ARow}
    G::Gadget
end

"GGSW(μ) over T^∨ (rows as in CaseA.aggsw_encrypt: μ Ω*_{0,0} g_l added to the mask, then to the body)."
function afggsw(Rg::ARing, fp::AFP, sk::ASecretKey, μ::Integer, G::Gadget, σ::Float64, rng)
    rows = ARow[]; z = zeros(UInt64, Rg.B * Rg.p)
    for part in 1:2, l in 1:G.ell
        c = aglwe_encrypt(Rg, sk, z, σ, rng)
        g = UInt64(μ) * gvalue(G, l)
        t = part == 1 ? c.a : c.b
        t[1] += g; t[Rg.p] -= g                              # entries (k, v) = (0, 0) and (0, p - 1)
        push!(rows, (affwd(c.a, fp), affwd(c.b, fp)))
    end
    return AFGGSW(rows, G)
end

struct AFKS
    c::Int
    perm::Vector{Int}
    rows::Vector{ARow}
    G::Gadget
end

"Key-switching key from τ_c(s') to s' (c ≡ 1 mod p), as CaseA.aks_keygen."
function afks_keygen(Rg::ARing, fp::AFP, sk::ASecretKey, c::Integer, G::Gadget, σ::Float64, rng)
    mod(c, Rg.p) == 1 || error("afks_keygen: c ≢ 1 (mod p)")
    base = omega00_times(Rg, normR!(Rg, aut_A(Rg, sk.s, c)))
    rows = ARow[]
    for l in 1:G.ell
        ct = aglwe_encrypt(Rg, sk, [reinterpret(UInt64, x) * gvalue(G, l) for x in base], σ, rng)
        push!(rows, (affwd(ct.a, fp), affwd(ct.b, fp)))
    end
    return AFKS(Int(c), afperm(Rg, c), rows, G)
end

struct AFKeys
    m::Int
    sk::ASecretKey
    s::Vector{Int64}
    bsk::Vector{Vector{AFGGSW}}    # bsk[t][j] = GGSW(δ_{j, pattern of group t})
    ksk::Vector{AFKS}              # τ_c, c ∈ U \ {1}
end

"Keys of CaseA.akeygen, for the FFT backend."
function afkeygen(Rg::ARing, P::AParams, fp::AFP, rng; m::Int = 1)
    P.n % m == 0 || error("n must be a multiple of m")
    sk = akeygen_sk(Rg, rng)
    s = sk.svec[1:P.n]
    bsk = [[afggsw(Rg, fp, sk, pattern(s, t, m) == j ? 1 : 0, P.Gbr, P.σ, rng) for j in 1:(1<<m)-1] for t in 1:P.n÷m]
    ksk = [afks_keygen(Rg, fp, sk, c, P.Gks, P.σ, rng) for c in Ugroup_A(Rg)[2:end]]
    return AFKeys(m, sk, s, bsk, ksk)
end

# ---------------------------------------------------------------- operations

@inline function _mac2!(accA::Matrix{ComplexF64}, accB::Matrix{ComplexF64}, D::Matrix{ComplexF64}, row::ARow)
    ra, rb = row
    @inbounds @simd for i in eachindex(D)
        accA[i] = muladd(D[i], ra[i], accA[i])
        accB[i] = muladd(D[i], rb[i], accB[i])
    end
    return nothing
end

"(accA, accB) += Fourier values of Dec(c)·GGSW (D, D2: buffers h × B; digits transformed by pairs for Rader with B = 1)."
function afextprod_acc!(accA::Matrix{ComplexF64}, accB::Matrix{ComplexF64}, Rg::ARing, fp::AFP, c::AGLWE,
                        K::AFGGSW, D::Matrix{ComplexF64} = similar(accA), D2::Matrix{ComplexF64} = similar(accA))
    ℓ = K.G.ell
    ds = vcat(adecompose(Rg, K.G, c.a), adecompose(Rg, K.G, c.b))   # row (part - 1) ℓ + l
    if fp isa AFPlanR && fp.B == 1
        for i in 1:2:2ℓ
            affwd2!(D, D2, ds[i], ds[i+1], fp)
            _mac2!(accA, accB, D, K.rows[i]); _mac2!(accA, accB, D2, K.rows[i+1])
        end
    else
        for i in 1:2ℓ
            affwd!(D, ds[i], fp)
            _mac2!(accA, accB, D, K.rows[i])
        end
    end
    return nothing
end

"Encryption from accumulated Fourier values (two inverse transforms)."
function afglwe(fp::AFP, accA::AbstractMatrix{ComplexF64}, accB::AbstractMatrix{ComplexF64})
    a = Vector{UInt64}(undef, fp.p * fp.B); b = similar(a)
    afinv2!(a, b, accA, accB, fp)
    return AGLWE(a, b)
end

"Dec(c)·GGSW(μ): an encryption of μ·(phase of c)."
function afextprod(Rg::ARing, fp::AFP, c::AGLWE, K::AFGGSW)
    accA = zeros(ComplexF64, fp.h, fp.B); accB = zeros(ComplexF64, fp.h, fp.B)
    afextprod_acc!(accA, accB, Rg, fp, c, K)
    return afglwe(fp, accA, accB)
end

"Hoisted key switching, outputs in the Fourier domain: (0, τ_c(b)) - τ_c(Dec(a))·K_c; B0 = Fourier values of C.b."
function afks_hoisted(Rg::ARing, fp::AFP, C::AGLWE, B0::Matrix{ComplexF64}, keys::Vector{AFKS})
    isempty(keys) && return ARow[]
    G = keys[1].G; ℓ = G.ell; h, B = fp.h, fp.B
    Ds = [affwd(d, fp) for d in adecompose(Rg, G, C.a)]
    out = ARow[]
    for K in keys
        oA = zeros(ComplexF64, h, B); oB = Matrix{ComplexF64}(undef, h, B)
        @inbounds for j in 1:B
            jp = K.perm[j]
            @simd for f in 1:h
                oB[f, j] = B0[f, jp]
            end
        end
        for l in 1:ℓ
            ra, rb = K.rows[l]; Dl = Ds[l]
            @inbounds for j in 1:B
                jp = K.perm[j]
                @simd for f in 1:h
                    d = Dl[f, jp]
                    oA[f, j] = muladd(-d, ra[f, j], oA[f, j])
                    oB[f, j] = muladd(-d, rb[f, j], oB[f, j])
                end
            end
        end
        push!(out, (oA, oB))
    end
    return out
end

"Single key switching τ_c, output in the coefficient domain."
function afks(Rg::ARing, fp::AFP, C::AGLWE, K::AFKS)
    oA, oB = afks_hoisted(Rg, fp, C, affwd(C.b, fp), [K])[1]
    return afglwe(fp, oA, oB)
end

"Constants of the public products of the comb bootstrap."
struct AFComb
    U::Vector{Int}                 # the group U, c = 1 first
    E::Matrix{ComplexF64}          # E[i, a+1] = V_f^a = e^{-2iπ fa/p}, f = fexps(fp)[i] (row i of the Fourier values)
    W::Matrix{ComplexF64}          # W[t+1, k+1] = e^{-2iπ tk/B}
    tix::Vector{Vector{Int}}       # tix[i][j+1] = 1 + ((2j + 1)(1 - c_i)/2 mod B)
end

function AFComb(Rg::ARing, fp::AFP)
    p, B = Rg.p, Rg.B
    U = Ugroup_A(Rg)
    E = [cispi(-2 * mod(f * a, p) / p) for f in fexps(fp), a in 0:p-1]
    W = [cispi(-2 * mod(t * k, B) / B) for t in 0:B-1, k in 0:B-1]
    tix = [[1 + mod((2j + 1) * ((1 - c) ÷ 2), B) for j in 0:B-1] for c in U]
    return AFComb(U, E, W, tix)
end

"T[f, t] = Σ_k V_f^{a_k} e^{-2iπ tk/B}: Fourier values of the public polynomials B·A_c of the exponents acol (a_k mod p)."
function aftable!(T::Matrix{ComplexF64}, cb::AFComb, acol::AbstractVector{<:Integer})
    fill!(T, 0)
    h, B = size(T)
    @inbounds for k in 1:B
        a = acol[k] + 1
        for t in 1:B
            w = cb.W[t, k]
            @simd for f in 1:h
                T[f, t] = muladd(w, cb.E[f, a], T[f, t])
            end
        end
    end
    return T
end

"(GA, GB) = Σ_c B·A_c ⊙ Cs[c] in the Fourier domain (Cs in the order of cb.U)."
function afpublic!(GA::Matrix{ComplexF64}, GB::Matrix{ComplexF64}, cb::AFComb, T::Matrix{ComplexF64}, Cs::Vector{ARow})
    fill!(GA, 0); fill!(GB, 0)
    h, B = size(T)
    for (i, (cA, cB)) in enumerate(Cs)
        ti = cb.tix[i]
        @inbounds for j in 1:B
            t = ti[j]
            @simd for f in 1:h
                q = T[f, t]
                GA[f, j] = muladd(q, cA[f, j], GA[f, j])
                GB[f, j] = muladd(q, cB[f, j], GB[f, j])
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the error bootstrap and the comb bootstrap

"Error bootstrap (B = 1), mask grouped by K.m: acc + Σ_j Dec(V^{e_j} acc - acc)·GGSW(δ_j), summed in the Fourier domain."
function blind_rotation_A_fft(Rg::ARing, P::AParams, fp::AFP, K::AFKeys, arow::AbstractVector{<:Integer},
                              bint::Integer)
    m = K.m
    C = AGLWE(zeros(UInt64, Rg.B * Rg.p), aacc0(Rg, R_of(P), [bint]))
    accA = zeros(ComplexF64, fp.h, fp.B); accB = similar(accA); D = similar(accA); D2 = similar(accA)
    for t in 1:length(arow)÷m
        fill!(accA, 0); fill!(accB, 0)
        for j in 1:(1<<m)-1
            afextprod_acc!(accA, accB, Rg, fp, rotate_v(Rg, C, pattern_exp(arow, t, j, m, Rg.p)) - C, K.bsk[t][j], D, D2)
        end
        C = C + afglwe(fp, accA, accB)
    end
    return aextract(Rg, C, 0)
end

"""
    combboot_A_fft(Rg, P, fp, K, aints, bints; cb = AFComb(Rg, fp))

Algorithm 1 in case (A) with the FFT backend (as CaseA.combboot_A: mask grouped by K.m, h = 1): per group, the division
by B, the Fourier values of C_0 and the hoisted key switchings (shared by the patterns); per pattern, the table of the
public polynomials, C_L in the Fourier domain, its two inverse FFTs and the external product, accumulated in the Fourier
domain; the accumulator is updated once per group.
"""
function combboot_A_fft(Rg::ARing, P::AParams, fp::AFP, K::AFKeys, aints::AbstractMatrix{<:Integer},
                        bints::AbstractVector{<:Integer}; cb::AFComb = AFComb(Rg, fp))
    m = K.m; h, B = fp.h, fp.B
    C = AGLWE(zeros(UInt64, B * Rg.p), aacc0(Rg, R_of(P), bints))
    T = zeros(ComplexF64, h, B); GA = similar(T); GB = similar(T)
    accA = similar(T); accB = similar(T); D = similar(T)
    acol = zeros(Int, B)
    for t in 1:size(aints, 2)÷m
        C0 = divB_A(Rg, C)
        B0 = affwd(C0.b, fp)
        Cs = vcat([(affwd(C0.a, fp), B0)], afks_hoisted(Rg, fp, C0, B0, K.ksk))
        fill!(accA, 0); fill!(accB, 0)
        for j in 1:(1<<m)-1
            for k in 1:B
                acol[k] = pattern_exp(view(aints, k, :), t, j, m, Rg.p)
            end
            aftable!(T, cb, acol)
            afpublic!(GA, GB, cb, T, Cs)
            CL = afglwe(fp, GA, GB)
            afextprod_acc!(accA, accB, Rg, fp, CL - C, K.bsk[t][j], D)
        end
        C = C + afglwe(fp, accA, accB)
    end
    return [aextract(Rg, C, k) for k in 0:B-1]
end

"""
    refresh_A_fft(Rg, P, fp, K, cts; ksk, pool, Vstar, recenter, rng, info, cb)

CaseA.refresh_A with the FFT backend: key switch N -> n, rounding at R = p̂ p with recentering and selection, rotation
of the errors (blind_rotation_A_fft for B = 1, combboot_A_fft for B ≥ 2), subtraction.
"""
function refresh_A_fft(Rg::ARing, P::AParams, fp::AFP, K::AFKeys, cts::AbstractVector{LWE};
                       ksk::Union{StairKey,Nothing} = nothing, pool = nothing, Vstar::Real = Inf, recenter::Bool = true,
                       rng::AbstractRNG = Random.default_rng(), info::Bool = false,
                       cb::Union{AFComb,Nothing} = Rg.B > 1 ? AFComb(Rg, fp) : nothing)
    length(cts) == Rg.B || error("B = $(Rg.B) ciphertexts expected")
    R = R_of(P); n = P.n
    t0 = time_ns()
    lw = ksk === nothing ? cts : [stair_ks(ksk, ct) for ct in cts]
    t1 = time_ns()
    rs = [round_select(ct, R, pool; Vstar, recenter, rng) for ct in lw]
    t2 = time_ns()
    aints = [mod(rs[k].abar[u], P.p) for k in 1:Rg.B, u in 1:n]
    bints = [mod(rs[k].bbar, P.p) for k in 1:Rg.B]
    errs = Rg.B == 1 ? [blind_rotation_A_fft(Rg, P, fp, K, view(aints, 1, :), bints[1])] :
                       combboot_A_fft(Rg, P, fp, K, aints, bints; cb)
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

include("caseA_opt.jl")

end # module
