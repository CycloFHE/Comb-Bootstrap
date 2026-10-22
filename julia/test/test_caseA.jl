# Tests of case (A) (src/caseA.jl): algebra against explicit arithmetic in Q[X]/Φ_M with rationals, exactness of the
# NTT products, GLWE operations, error bootstrap and comb bootstrap on small rings.
# Run:  julia --project=. test/test_caseA.jl
using Test, Random
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
import .CombBoot: Gadget, LWE, lwe_phase, torus_frac, torus_to_float, gauss_torus, lwe_encrypt

# ---------------------------------------------------------------- explicit arithmetic in Q[X]/Φ_M
const QQ = Rational{BigInt}
const CYC = Dict{Int,Vector{BigInt}}()
function polydiv_exact(num::Vector{BigInt}, den::Vector{BigInt})       # ascending coefficients, den monic
    num = copy(num); dn = length(den) - 1; q = zeros(BigInt, length(num) - dn)
    for i in length(num):-1:dn+1
        c = num[i]; q[i-dn] = c
        for j in 0:dn
            num[i-dn+j] -= c * den[j+1]
        end
    end
    @assert all(iszero, num[1:dn])
    return q
end
function cyclotomic(M::Int)
    haskey(CYC, M) && return CYC[M]
    num = zeros(BigInt, M + 1); num[1] = -1; num[end] = 1
    for d in 1:M-1
        M % d == 0 && (num = polydiv_exact(num, cyclotomic(d)))
    end
    return CYC[M] = num
end
struct KX
    M::Int; N::Int; Φ::Vector{BigInt}
end
KX(M) = (Φ = cyclotomic(M); KX(M, length(Φ) - 1, Φ))
function kreduce(F::KX, P::Vector{QQ})
    P = copy(P)
    for i in length(P):-1:F.N+1
        c = P[i]; iszero(c) && continue
        for j in 0:F.N
            P[i-F.N+j] -= c * F.Φ[j+1]
        end
    end
    out = zeros(QQ, F.N); out[1:min(F.N, length(P))] .= P[1:min(F.N, length(P))]
    return out
end
xpow(F::KX, e::Integer) = (v = zeros(QQ, F.M); v[mod(e, F.M)+1] = 1; kreduce(F, v))
function kmul(F::KX, A::Vector{QQ}, B::Vector{QQ})
    out = zeros(QQ, 2F.N)
    for i in 1:F.N, j in 1:F.N
        out[i+j-1] += A[i] * B[j]
    end
    return kreduce(F, out)
end
function kcompose(F::KX, P::Vector{QQ}, a::Integer)         # P(X^a)
    out = zeros(QQ, F.M)
    for i in 0:F.N-1
        out[mod(a * i, F.M)+1] += P[i+1]
    end
    return kreduce(F, out)
end
function ktrace(F::KX, P::Vector{QQ})
    s = zeros(QQ, F.N)
    for a in 1:F.M-1
        gcd(a, F.M) == 1 && (s .+= kcompose(F, P, a))
    end
    @assert all(iszero, s[2:end])
    return s[1]
end
kpair(F::KX, P, Q) = ktrace(F, kmul(F, P, kcompose(F, Q, -1)))
"K element of augmented coordinates x: Σ_k Σ_v x[k][v]/(B p) U^k V^v."
function toK_aug(Rg, F, x)
    P = zeros(QQ, F.N)
    for k in 0:Rg.B-1, v in 0:Rg.p-1
        P .+= QQ(x[k*Rg.p+v+1]) // (Rg.B * Rg.p) .* xpow(F, Rg.p * k * (Rg.M0 > 1) + max(Rg.M0, 1) * v)
    end
    return P
end
"K element of R coordinates r: Σ r[k][v] U^k V^v."
function toK_R(Rg, F, r)
    P = zeros(QQ, F.N)
    for k in 0:Rg.B-1, v in 0:Rg.p-1
        iszero(r[k*Rg.p+v+1]) || (P .+= QQ(r[k*Rg.p+v+1]) .* xpow(F, Rg.p * k * (Rg.M0 > 1) + max(Rg.M0, 1) * v))
    end
    return P
end
"Augmented coordinates of P ∈ K: ⟨P, U^k V^v⟩."
coords_aug(Rg, F, P) = [kpair(F, P, xpow(F, Rg.p * k * (Rg.M0 > 1) + max(Rg.M0, 1) * v)) for k in 0:Rg.B-1 for v in 0:Rg.p-1]
rand_aug_q(Rg, rng) = (x = QQ.(rand(rng, -20:20, Rg.B * Rg.p)) .// rand(rng, 1:7); fixaug!(Rg, x))
function rand_R(Rg, rng; lim = 5)
    r = rand(rng, -lim:lim, Rg.B * Rg.p)
    for k in 0:Rg.B-1
        r[k*Rg.p+Rg.p] = 0
    end
    return r
end
units(M) = [a for a in 1:M-1 if gcd(a, M) == 1]

@testset "case (A): algebra" begin
    rng = MersenneTwister(1)
    for (p, M0) in ((3, 1), (5, 1), (7, 1), (3, 4), (5, 4), (3, 8), (7, 8))
        Rg = ARing(p, M0); F = KX(Rg.M)
        @test F.N == Rg.N
        x = rand_aug_q(Rg, rng); Px = toK_aug(Rg, F, x)
        # dual coordinates: ⟨P, Ω_{j,k}⟩ = x[k][j], and ⟨P, U^k V^v⟩ = x[k][v] for every v
        @test coords_aug(Rg, F, Px) == x
        # products by R
        r = rand_R(Rg, rng)
        @test coords_aug(Rg, F, kmul(F, toK_R(Rg, F, r), Px)) == conv_generic(Rg, r, x)
        # Ω*_{0,0} a' = a, with a' = a/Ω*_{0,0} ∈ K/R in Ω coordinates
        ap = div_omega00(Rg, x)
        @test omega00_times(Rg, ap) == x
        Ω00 = toK_aug(Rg, F, (e = zeros(QQ, Rg.B * Rg.p); e[1] = 1; e[Rg.p] = -1; e))
        @test kmul(F, Ω00, toK_R(Rg, F, ap)) == Px
        # X^e and the automorphisms
        for e in (1, 2, Rg.M - 1, Rg.p, max(Rg.M0, 1) + 1)
            @test coords_aug(Rg, F, kmul(F, xpow(F, e), Px)) == mulX_A(Rg, x, e)
        end
        for c in units(Rg.M)[1:min(end, 6)]
            @test coords_aug(Rg, F, kcompose(F, Px, c)) == aut_A(Rg, x, c)
            @test toK_R(Rg, F, normR!(Rg, aut_A(Rg, r, c))) == kcompose(F, toK_R(Rg, F, r), c)
        end
        # the group U
        U = Ugroup_A(Rg)
        @test length(U) == Rg.B && all(mod(c, p) == 1 for c in U) && allunique(U) && U[1] == 1
        # exact products modulo 2^64
        t = rand(rng, UInt64, Rg.B * Rg.p); fixaug!(Rg, t)
        rb = rand_R(Rg, rng; lim = 1 << 22)
        @test mulRT(Rg, kint(Rg, rb), t) == conv_generic(Rg, rb, t)
    end
    # example of Appendix F: M = 12, p = 3, M0 = 4, v* = (X^3 + 2X^2 - 2X - 1)/(18 p̂)
    Rg = ARing(3, 4); F = KX(12); phat = 5
    vq = [QQ(mods(v, 3), 3 * phat) for k in 0:1 for v in 0:2]
    @test toK_aug(Rg, F, vq) == kreduce(F, QQ[-1, -2, 2, 1] .// (18 * phat))
end

@testset "case (A): GLWE, GGSW, key switching, extraction" begin
    rng = MersenneTwister(2)
    for (p, M0) in ((17, 1), (13, 8), (17, 4))
        Rg = ARing(p, M0); sk = akeygen_sk(Rg, rng); σ = 2.0^-40
        m = rand_aug(Rg, rng)
        C = aglwe_encrypt(Rg, sk, m, σ, rng)
        err(x, y) = maximum(abs.(torus_to_float.(x .- y)))
        @test err(aphase(Rg, sk, C), m) < 2.0^-35
        for k in 0:Rg.B-1
            L = aextract(Rg, C, k)
            @test lwe_phase(L, sk.svec) == aphase(Rg, sk, C)[k*p+1]
        end
        G = Gadget(8, 3)
        for μ in (0, 1)
            K = aggsw_encrypt(Rg, sk, μ, G, σ, rng)
            @test err(aphase(Rg, sk, aextprod(Rg, C, K)), μ == 1 ? m : zero(m)) < 2.0^-18
        end
        for c in Ugroup_A(Rg)[2:end]
            Kc = aks_keygen(Rg, sk, c, Gadget(10, 3), σ, rng)
            @test err(aphase(Rg, sk, aks_hoisted(Rg, C, [Kc])[1]), aut_A(Rg, m, c)) < 2.0^-20
        end
    end
end

@testset "case (A): error bootstrap and comb bootstrap" begin
    rng = MersenneTwister(3)
    for (p, M0, n, m) in ((17, 1, 12, 1), (13, 8, 16, 1), (17, 4, 20, 1), (3, 4, 4, 1), (31, 1, 24, 1),
                          (17, 1, 12, 2), (13, 8, 16, 2), (17, 4, 18, 3), (31, 1, 24, 4))
        Rg = ARing(p, M0)
        P = AParams(Rg; phat = 4, n, Gbr = Gadget(8, 3), Gks = Gadget(10, 3), σ = 2.0^-40)
        K = akeygen(Rg, P, rng; m)
        R = P.phat * p; emax = (p - 1) ÷ 2
        for trial in 1:3
            aints = rand(rng, 0:p-1, Rg.B, n)
            E = rand(rng, -emax:emax, Rg.B)
            bints = [mod(sum(aints[k, u] * K.s[u] for u in 1:n) + E[k], p) for k in 1:Rg.B]
            outs = Rg.B == 1 ? [blind_rotation_A(Rg, P, K, view(aints, 1, :), bints[1])] : combboot_A(Rg, P, K, aints, bints)
            for k in 1:Rg.B
                e = torus_to_float(lwe_phase(outs[k], K.sk.svec) - torus_frac(E[k], R))
                @test abs(e) < 2.0^-14
            end
        end
        # the whole refresh of fresh LWE encryptions under s (dimension n, no key switch)
        mus = rand(rng, 0:P.phat-1, Rg.B)
        cts = [lwe_encrypt(K.s, torus_frac(μ, P.phat), 2.0^-20, rng) for μ in mus]
        outs = refresh_A(Rg, P, K, cts)
        for k in 1:Rg.B
            @test abs(torus_to_float(lwe_phase(outs[k], K.sk.svec) - torus_frac(mus[k], P.phat))) < 2.0^-14
        end
    end
end
