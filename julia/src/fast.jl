# Floating-point FFT backend (as in TFHE-rs): products computed with a negacyclic FFT in Float64,
# the FFT error being an extra noise. Requires FFTW.jl:  using Pkg; Pkg.add("FFTW")
#
# Usage (after CombBoot):
#     include("src/CombBoot.jl"); using .CombBoot
#     include("src/fast.jl");      using .Fast
#
# Fourier representation of a real polynomial x of degree < N: the H = N/2 values x(ζ^(4k+1)),
# ζ = e^{iπ/N}, obtained by one complex FFT of size H on z_j = (x_j + i x_{j+H}) ζ^j.
# The other N/2 roots are the conjugates; X ↦ X^a maps ζ^(4k+1) to ζ^(a(4k+1)), i.e. a
# permutation of the H values, with a conjugation when a(4k+1) ≡ 3 (mod 4).

module Fast

using FFTW
using ..CombBoot
import ..CombBoot: decompose, aut_poly, mulX, extract, divB, acc0, vstar, trivial, gvalue, gadget_lift,
                    glwe_encrypt, SecretKey, GLWE, Gadget, CBParams, LWE, Ring

export FPlan, FKeys, fast_keygen, combboot_fast, separate_rotations_fast, fextprod, fcmux, fks

const TWO64 = 18446744073709551616.0

# ---------------------------------------------------------------- negacyclic FFT

struct FPlan{PF,PI}
    N::Int
    H::Int
    tw::Vector{ComplexF64}      # ζ^j
    itw::Vector{ComplexF64}     # ζ^{-j} / H
    Z::Vector{ComplexF64}       # ζ^m, m = 0..2N-1
    pf::PF                      # in-place backward FFT (e^{+2πi jk/H})
    pi::PI                      # in-place forward FFT
    buf::Vector{ComplexF64}
    nthreads::Int               # threads of the FFTW plans
end

"Threads of the FFTs: environment variable FFT_THREADS (default 1); julia must be started with -t ≥ that number."
fft_threads() = parse(Int, get(ENV, "FFT_THREADS", "1"))

function FPlan(N::Int; nthreads::Int = fft_threads())
    H = N ÷ 2
    buf = zeros(ComplexF64, H)
    tw = [cispi(j / N) for j in 0:H-1]
    itw = [cispi(-j / N) / H for j in 0:H-1]
    Z = [cispi(m / N) for m in 0:2N-1]
    return FPlan(N, H, tw, itw, Z, plan_bfft!(copy(buf); flags = FFTW.MEASURE, num_threads = nthreads),
                 plan_fft!(copy(buf); flags = FFTW.MEASURE, num_threads = nthreads), buf, nthreads)
end

@inline tofloat(t::UInt64) = Float64(reinterpret(Int64, t))
@inline tofloat(t::Int64) = Float64(t)

"Fourier values of a torus (UInt64, read as signed) or small integer polynomial."
function ffwd!(Y::Vector{ComplexF64}, x::AbstractVector, fp::FPlan)
    H = fp.H
    @inbounds for j in 1:H
        Y[j] = complex(tofloat(x[j]), tofloat(x[j+H])) * fp.tw[j]
    end
    fp.pf * Y
    return Y
end
ffwd(x::AbstractVector, fp::FPlan) = ffwd!(Vector{ComplexF64}(undef, fp.H), x, fp)

@inline function f2t(v::Float64)
    r = v - TWO64 * round(v / TWO64)
    return reinterpret(UInt64, unsafe_trunc(Int64, round(r)))
end

"Torus polynomial from Fourier values (rounded, modulo 2^64)."
function finv!(out::Vector{UInt64}, Y::Vector{ComplexF64}, fp::FPlan)
    H = fp.H
    z = fp.buf
    copyto!(z, Y)
    fp.pi * z
    @inbounds for j in 1:H
        w = z[j] * fp.itw[j]
        out[j] = f2t(real(w))
        out[j+H] = f2t(imag(w))
    end
    return out
end
finv(Y::Vector{ComplexF64}, fp::FPlan) = finv!(Vector{UInt64}(undef, fp.N), Y, fp)

"Map of X ↦ X^a on the Fourier values: (τ_a Y)[k] = Y[src[k]], conjugated where cj[k]."
function faut_map(N::Int, a::Integer)
    return _faut_map(N, a)
end
function _faut_map(N::Int, a::Integer)
    H = N ÷ 2; twoN = 2N
    src = Vector{Int}(undef, H); cj = Vector{Bool}(undef, H)
    for k in 0:H-1
        c = mod(Int(a) * (4k + 1), twoN)
        if c % 4 == 1
            src[k+1] = (c - 1) ÷ 4 + 1; cj[k+1] = false
        else
            c2 = twoN - c
            src[k+1] = (c2 - 1) ÷ 4 + 1; cj[k+1] = true
        end
    end
    return src, cj
end

# ---------------------------------------------------------------- keys

const Row = NTuple{2,Vector{ComplexF64}}

struct FGGSWf
    rows::Vector{Row}
    G::Gadget
end

struct FKSf
    a::Int
    src::Vector{Int}
    cj::Vector{Bool}
    sg::Vector{Float64}         # -1 where conjugated, +1 otherwise
    rows::Vector{Row}
    G::Gadget
end

struct FKeys
    sk::SecretKey
    s::Vector{Int64}
    bsk::Vector{FGGSWf}
    baby::Vector{FKSf}
    giant::Vector{FKSf}
end

frow(c::GLWE, fp::FPlan) = (ffwd(c.a, fp), ffwd(c.b, fp))

function fggsw(r::Ring, fp::FPlan, sk::SecretKey, μ::Integer, G::Gadget, σ::Float64, rng)
    rows = Row[]
    z = zeros(UInt64, r.N)
    for part in 1:2, j in 1:G.ell
        c = glwe_encrypt(r, sk, z, σ, rng)
        if part == 1
            c.a[1] += UInt64(μ) * gvalue(G, j)
        else
            c.b[1] += UInt64(μ) * gvalue(G, j)
        end
        push!(rows, frow(c, fp))
    end
    return FGGSWf(rows, G)
end

function fkskey(r::Ring, fp::FPlan, sk::SecretKey, a::Integer, G::Gadget, σ::Float64, rng)
    τs = aut_poly(sk.s, a)
    rows = Row[]
    for j in 1:G.ell
        msg = [reinterpret(UInt64, τs[i]) * gvalue(G, j) for i in 1:r.N]
        push!(rows, frow(glwe_encrypt(r, sk, msg, σ, rng), fp))
    end
    src, cj = faut_map(r.N, a)
    return FKSf(Int(a), src, cj, [c ? -1.0 : 1.0 for c in cj], rows, G)
end

"Keys of the comb bootstrap for the FFT backend (the encryptions themselves are exact)."
function fast_keygen(P::CBParams, r::Ring, fp::FPlan, rng)
    sprime = Int64.(rand(rng, 0:1, P.N))
    sk = SecretKey(r, sprime)
    bsk = [fggsw(r, fp, sk, sprime[u], P.Gbr, P.σ, rng) for u in 1:P.n]
    M = 2 * P.N; γ = 1 + P.p
    baby = [fkskey(r, fp, sk, powermod(γ, j, M), P.Gks, P.σ, rng) for j in 1:P.g-1]
    giant = [fkskey(r, fp, sk, powermod(γ, P.g * k, M), P.Gks, P.σ, rng) for k in 1:P.h-1]
    return FKeys(sk, sprime[1:P.n], bsk, baby, giant)
end

# ---------------------------------------------------------------- operations

"Dec(c)·GGSW(μ)."
function fextprod(fp::FPlan, c::GLWE, K::FGGSWf)
    G = K.G; ℓ = G.ell; H = fp.H
    accA = zeros(ComplexF64, H); accB = zeros(ComplexF64, H)
    D = Vector{ComplexF64}(undef, H)
    for (part, comp) in enumerate((c.a, c.b))
        digits = decompose(G, comp)
        for j in 1:ℓ
            ffwd!(D, digits[j], fp)
            ra, rb = K.rows[(part-1)*ℓ+j]
            @inbounds @simd for k in 1:H
                accA[k] = muladd(D[k], ra[k], accA[k])
                accB[k] = muladd(D[k], rb[k], accB[k])
            end
        end
    end
    return GLWE(finv(accA, fp), finv(accB, fp))
end

fcmux(fp::FPlan, c0::GLWE, c1::GLWE, K::FGGSWf) = c0 + fextprod(fp, c1 - c0, K)

"Hoisted key switching; outputs in the Fourier domain. B0 = Fourier values of c.b."
function fks_hoisted(fp::FPlan, c::GLWE, B0::Vector{ComplexF64}, keys::Vector{FKSf})
    isempty(keys) && return Row[]
    G = keys[1].G; ℓ = G.ell; H = fp.H
    D = [ffwd(d, fp) for d in decompose(G, c.a)]
    out = Row[]
    for K in keys
        src = K.src; sg = K.sg
        oA = zeros(ComplexF64, H); oB = Vector{ComplexF64}(undef, H)
        @inbounds for k in 1:H
            b = B0[src[k]]
            oB[k] = complex(real(b), sg[k] * imag(b))
        end
        for j in 1:ℓ
            ra, rb = K.rows[j]; Dj = D[j]
            @inbounds for k in 1:H
                x = Dj[src[k]]
                d = complex(real(x), sg[k] * imag(x))
                oA[k] = muladd(-d, ra[k], oA[k])
                oB[k] = muladd(-d, rb[k], oB[k])
            end
        end
        push!(out, (oA, oB))
    end
    return out
end

"Single key switching, output in the coefficient domain."
function fks(fp::FPlan, c::GLWE, K::FKSf)
    oA, oB = fks_hoisted(fp, c, ffwd(c.b, fp), [K])[1]
    return GLWE(finv(oA, fp), finv(oB, fp))
end

"Table of c_3 in the complex domain: T[c, t] = Σ_k ζ^(M0 c a_k + p t k)."
function fba_table(P::CBParams, fp::FPlan, acol::AbstractVector{Int})
    mask = 2P.N - 1
    T = zeros(ComplexF64, P.p, P.B)
    Z = fp.Z
    @inbounds for t in 0:P.B-1, c in 1:2:P.p-1
        acc = zero(ComplexF64)
        e1 = P.M0 * c; e2 = P.p * t
        for k in 0:P.B-1
            acc += Z[((e1 * acol[k+1] + e2 * k) & mask) + 1]
        end
        T[c+1, t+1] = acc
    end
    return T
end

"Fourier values of τ_e(B·A_{u,a}) at the roots ζ^(4k+1)."
function fba_hat(P::CBParams, T::Matrix{ComplexF64}, a::Integer, e::Integer)
    N = P.N; H = N ÷ 2
    mask = 2N - 1; pm = P.p - 1; bm = P.B - 1
    rr = mod((Int(a) - 1) ÷ P.p, P.M0)
    Qh = Vector{ComplexF64}(undef, H)
    c = Int(e) & mask; step = (4 * Int(e)) & mask
    @inbounds for k in 1:H
        Qh[k] = T[(c & pm)+1, ((-c * rr) & bm)+1]
        c = (c + step) & mask
    end
    return Qh
end

"GA += τ_e(B·A_{u,a})·cA, GB += τ_e(B·A_{u,a})·cB in the Fourier domain, values read from T."
function fba_mac!(GA, GB, P::CBParams, T::Matrix{ComplexF64}, a::Integer, e::Integer, cA, cB)
    N = P.N; H = N ÷ 2
    mask = 2N - 1; pm = P.p - 1; bm = P.B - 1
    rr = mod((Int(a) - 1) ÷ P.p, P.M0)
    c = Int(e) & mask; step = (4 * Int(e)) & mask
    @inbounds for i in 1:H
        q = T[(c & pm)+1, ((-c * rr) & bm)+1]
        GA[i] = muladd(q, cA[i], GA[i])
        GB[i] = muladd(q, cB[i], GB[i])
        c = (c + step) & mask
    end
    return nothing
end

# ---------------------------------------------------------------- Algorithm 1 and baseline

function combboot_fast_ref(P::CBParams, fp::FPlan, K::FKeys, aints::AbstractMatrix{Int}, bints::Vector{Int})
    N = P.N; H = fp.H; M = 2N; B = P.B; γ = 1 + P.p
    cks = [powermod(γ, P.g * k, M) for k in 0:P.h-1]
    ckinv = [invmod(c, M) for c in cks]
    gpow = [powermod(γ, j, M) for j in 0:P.g-1]
    C = trivial(acc0(P, bints))
    for u in 1:size(aints, 2)
        acol = view(aints, :, u)
        C0 = divB(C, B)
        C0 = GLWE(gadget_lift(P.Gks, C0.a), C0.b)
        B0 = ffwd(C0.b, fp)
        Cs = vcat([(ffwd(C0.a, fp), B0)], fks_hoisted(fp, C0, B0, K.baby))
        T = fba_table(P, fp, acol)
        CL = GLWE(N)
        for k in 0:P.h-1
            GA = zeros(ComplexF64, H); GB = zeros(ComplexF64, H)
            for j in 0:P.g-1
                j + P.g * k < B || continue
                cA, cB = Cs[j+1]
                fba_mac!(GA, GB, P, T, mod(cks[k+1] * gpow[j+1], M), ckinv[k+1], cA, cB)
            end
            Gk = GLWE(finv(GA, fp), finv(GB, fp))
            CL = k == 0 ? CL + Gk : CL + fks(fp, Gk, K.giant[k])
        end
        C = fcmux(fp, C, CL, K.bsk[u])
    end
    return [extract(C, k) for k in 0:B-1]
end

function blind_rotation_fast(P::CBParams, fp::FPlan, K::FKeys, arow::AbstractVector{Int}, bint::Int)
    C = trivial(mulX(vstar(P), -P.M0 * bint))
    for u in 1:length(arow)
        C = fcmux(fp, C, mulX(C, P.M0 * arow[u]), K.bsk[u])
    end
    return extract(C, 0)
end

separate_rotations_fast_ref(P, fp, K, aints, bints) =
    [blind_rotation_fast(P, fp, K, view(aints, k+1, :), bints[k+1]) for k in 0:P.B-1]

include("fast_opt.jl")
include("grouped.jl")
include("fast_product.jl")
include("stair_ks.jl")
include("fast_refresh.jl")

end
