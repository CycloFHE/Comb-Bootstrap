# Comb bootstrap, case (B): p = 2^β, M = M0·p = 2N, Φ_M = X^N + 1, B = M0 columns,
# column k = coefficients X^i with i ≡ k (mod B), ϖ_k = X^k, U = {1 + p r}, γ = 1 + p.
# Algorithm 1 of the paper, with the division by B of line 4 (any q, here q = 2^64).

struct CBParams
    N::Int          # ring degree
    p::Int          # 2^β: order of X^{M0}, the errors E_k live in Z_p, window |E_k| ≤ p/4
    phat::Int       # message modulus
    n::Int          # LWE dimension (binary key s = first n coefficients of s')
    M0::Int
    B::Int          # number of columns = M0
    g::Int          # baby steps
    h::Int          # giant steps
    Gbr::Gadget     # gadget of the bootstrapping keys RGSW(s_u)
    Gks::Gadget     # gadget of the automorphism key-switching keys
    σ::Float64      # standard deviation of the GLWE errors (torus)
end

function CBParams(; N, p, phat, n, Gbr, Gks, σ, h = nothing)
    ispow2(N) && ispow2(p) && p >= 4 || error("case (B): N and p powers of 2, p ≥ 4")
    M = 2N
    M % p == 0 || error("p must divide 2N")
    M0 = M ÷ p
    B = M0                         # B = 1 (p = 2N): the error bootstrap of one ciphertext (separate rotations only)
    h === nothing && (h = B >= 4 ? 2 : 1)
    Gks.off == 0 || (h == 1 && Gks.off <= trailing_zeros(B)) ||
        error("shifted automorphism gadget: h = 1 (giant steps decompose undivided values) and off ≤ log2 B")
    g = cld(B, h)
    return CBParams(N, p, phat, n, M0, B, g, h, Gbr, Gks, σ)
end

R_of(P::CBParams) = P.phat * P.p
Emax(P::CBParams) = P.p ÷ 4

struct CBKeys
    sk::SecretKey                 # ring key s'
    s::Vector{Int64}              # LWE key s = first n coefficients of s'
    bsk::Vector{FGGSW}            # RGSW(s_u), u = 1..n
    baby::Vector{FKSKey}          # τ_{γ^j}, j = 1..g-1
    giant::Vector{FKSKey}         # τ_{c_k}, c_k = γ^{gk}, k = 1..h-1
end

function cb_keygen(P::CBParams, r::Ring, rng)
    sprime = Int64.(rand(rng, 0:1, P.N))
    sk = SecretKey(r, sprime)
    bsk = [ggsw_encrypt(r, sk, sprime[u], P.Gbr, P.σ, rng) for u in 1:P.n]
    M = 2 * P.N; γ = 1 + P.p
    baby = [ks_keygen(r, sk, powermod(γ, j, M), P.Gks, P.σ, rng) for j in 1:P.g-1]
    giant = [ks_keygen(r, sk, powermod(γ, P.g * k, M), P.Gks, P.σ, rng) for k in 1:P.h-1]
    return CBKeys(sk, sprime[1:P.n], bsk, baby, giant)
end

# ---------------------------------------------------------------- test polynomial and accumulator

"N·v*: coefficient i = min(j, p/2 - j)/R with j = floor(i/B) (the tent, repeated B times)."
function vstar(P::CBParams)
    R = R_of(P)
    vals = [torus_frac(m, R) for m in 0:P.p÷2]          # the p/2 + 1 distinct values
    v = Vector{UInt64}(undef, P.N)
    for i in 0:P.N-1
        j = i ÷ P.B
        v[i+1] = vals[min(j, P.p ÷ 2 - j)+1]
    end
    return v
end

"π_k on torus polynomials: keep the coefficients i ≡ k (mod B)."
column(v::Vector{T}, k::Int, B::Int) where {T} = [(i % B == k) ? v[i+1] : zero(T) for i in 0:length(v)-1]

"acc_0 = Σ_k X^{-M0 b_k} π_k(v*), with b_k = bints[k+1]/p."
function acc0(P::CBParams, bints::Vector{Int})
    v = vstar(P)
    acc = zeros(UInt64, P.N)
    for k in 0:P.B-1
        acc .+= mulX(column(v, k, P.B), -P.M0 * bints[k+1])
    end
    return acc
end

"Clear L_{u}(P) = Σ_k X^{M0 a_{k,u}} π_k(P)."
function L_clear(P::CBParams, acc::Vector{UInt64}, acol::AbstractVector{Int})
    out = zeros(UInt64, P.N)
    for k in 0:P.B-1
        out .+= mulX(column(acc, k, P.B), P.M0 * acol[k+1])
    end
    return out
end

"B·A_{u,a} = Σ_k X^{M0 a_{k,u} + k(1-a)}: an integer polynomial with B monomials ±1."
function BA_poly(P::CBParams, acol::AbstractVector{Int}, a::Integer)
    N = P.N; twoN = 2N
    Q = zeros(Int64, N)
    for k in 0:P.B-1
        e = mod(P.M0 * acol[k+1] + k * (1 - Int(a)), twoN)
        if e < N
            Q[e+1] += 1
        else
            Q[e-N+1] -= 1
        end
    end
    return Q
end

"The group U = {1 + p r : r ∈ Z_{M0}} as exponents in [0, 2N)."
Ugroup(P::CBParams) = [mod(1 + P.p * r, 2 * P.N) for r in 0:P.M0-1]

"""
Values of B·A_{u,a}(X^e) at the roots ψ^(2i+1), for all a = 1 + p r ∈ U, from one table
T[c, t] = Σ_k ψ^(M0 c a_k + p t k) (c odd mod p, t mod B): B·A_{u,a}(ψ^c) = T[c mod p, -c r mod B].
This is the computation of line 3 of Algorithm 1 (Appendix F), here with (p/2)·B² additions.
"""
function ba_table(P::CBParams, r::Ring, acol::AbstractVector{Int})
    pl = r.plan; Pm = pl.m.P; N = P.N
    ψpow(e) = (e = mod(e, 2N); e < N ? pl.psi[e+1] : Pm - pl.psi[e-N+1])
    T = zeros(UInt64, P.p, P.B)
    for c in 1:2:P.p-1, t in 0:P.B-1
        acc = UInt64(0)
        for k in 0:P.B-1
            acc = addmod(acc, ψpow(P.M0 * c * acol[k+1] + P.p * t * k), Pm)
        end
        T[c+1, t+1] = acc
    end
    return T
end

"NTT of τ_e(B·A_{u,a}) read from the table."
function ba_hat(P::CBParams, T::Matrix{UInt64}, a::Integer, e::Integer)
    N = P.N; twoN = 2N
    rr = mod((Int(a) - 1) ÷ P.p, P.M0)
    Qh = Vector{UInt64}(undef, N)
    @inbounds for i in 0:N-1
        c = mod((2i + 1) * Int(e), twoN)
        Qh[i+1] = T[mod(c, P.p)+1, mod(-c * rr, P.B)+1]
    end
    return Qh
end

"The public products can be done exactly in the NTT domain if g·B·(ℓ N B_aut/2 + 1)·2^31 < P/2."
ntt_products_ok(P::CBParams, r::Ring) =
    log2(P.g * P.B * (P.Gks.ell * P.N * 2.0^(P.Gks.logB - 1) + 1)) + 31 < log2(Float64(r.plan.m.P)) - 1.5

"Division by B of the coordinates (rounded shift; any representative works, Lemma 4)."
function divB(c::GLWE, B::Int)
    s = trailing_zeros(B)
    f(x::UInt64) = (x >> s) + ((x >> (s - 1)) & 0x1)
    return GLWE(f.(c.a), f.(c.b))
end

# ---------------------------------------------------------------- Algorithm 1

"""
    combboot(P, r, K, aints, bints; trace)

aints[k+1, u] = p·a_{k,u} ∈ Z_p and bints[k+1] = p·b_k ∈ Z_p (noise-free LWE ciphertexts modulo p,
b_k = a_k·s + E_k/p). Returns the B LWE encryptions of E_k/R under s' (dimension N).
`trace(u, C)` is called after each iteration (for tests).
"""
function combboot(P::CBParams, r::Ring, K::CBKeys, aints::AbstractMatrix{Int}, bints::Vector{Int};
                   trace = nothing, ntt_products::Bool = ntt_products_ok(P, r))
    N = P.N; M = 2N; B = P.B; γ = 1 + P.p
    cks = [powermod(γ, P.g * k, M) for k in 0:P.h-1]
    ckinv = [invmod(c, M) for c in cks]
    gpow = [powermod(γ, j, M) for j in 0:P.g-1]
    C = trivial(acc0(P, bints))                                   # line 1
    for u in 1:size(aints, 2)                                     # line 2 (u ≤ n)
        acol = view(aints, :, u)
        C0 = divB(C, B)                                           # line 4
        C0 = GLWE(gadget_lift(P.Gks, C0.a), C0.b)                 # representative of the digits (shifted gadget)
        CL = GLWE(N)
        if ntt_products
            # baby steps and public products in the NTT domain, as in the cost model
            T = ba_table(P, r, acol)                              # line 3 (public)
            NC0 = nglwe(r, C0)
            NCs = vcat([NC0], ks_hoisted_ntt(r, C0, NC0, K.baby)) # lines 5-7
            for k in 0:P.h-1                                      # line 8
                NG = NGLWE(N)
                for j in 0:P.g-1
                    j + P.g * k < B || continue
                    a = mod(cks[k+1] * gpow[j+1], M)
                    mac!(NG, ba_hat(P, T, a, ckinv[k+1]), NCs[j+1], r.plan.m)   # line 9
                end
                Gk = to_glwe(r, NG)
                CL = k == 0 ? CL + Gk : CL + ks(r, Gk, K.giant[k])            # line 10
            end
        else
            # exact products in the coefficient domain (always safe)
            Cs = vcat([C0], ks_hoisted(r, C0, K.baby))            # lines 5-7
            for k in 0:P.h-1                                      # line 8
                Gk = GLWE(N)
                for j in 0:P.g-1
                    j + P.g * k < B || continue
                    a = mod(cks[k+1] * gpow[j+1], M)
                    Q = aut_poly(BA_poly(P, acol, a), ckinv[k+1]) # line 3 (public)
                    sparse_mul_add!(Gk, Q, Cs[j+1])               # line 9
                end
                CL = k == 0 ? CL + Gk : CL + ks(r, Gk, K.giant[k])            # line 10
            end
        end
        C = cmux(r, C, CL, K.bsk[u])                              # line 11
        trace === nothing || trace(u, C)
    end
    return [extract(C, k) for k in 0:B-1]                         # lines 12-13 (ϖ_k = X^k)
end

# ---------------------------------------------------------------- baseline: B separate blind rotations

"One blind rotation in the same ring (CMux form), read at the constant coefficient."
function blind_rotation(P::CBParams, r::Ring, K::CBKeys, arow::AbstractVector{Int}, bint::Int)
    C = trivial(mulX(vstar(P), -P.M0 * bint))
    for u in 1:length(arow)
        C = cmux(r, C, mulX(C, P.M0 * arow[u]), K.bsk[u])
    end
    return extract(C, 0)
end

separate_rotations(P, r, K, aints, bints) =
    [blind_rotation(P, r, K, view(aints, k+1, :), bints[k+1]) for k in 0:P.B-1]

# ---------------------------------------------------------------- inputs and refresh

"Noise-free inputs modulo p: errors E_k uniform in [-Emax, Emax]."
function random_inputs(P::CBParams, s::Vector{Int64}, rng; emax = Emax(P))
    aints = rand(rng, 0:P.p-1, P.B, P.n)
    E = rand(rng, -emax:emax, P.B)
    bints = [mod(sum(aints[k, u] * s[u] for u in 1:P.n) + E[k], P.p) for k in 1:P.B]
    return aints, bints, E
end

"Fresh LWE encryption of μ under s (dimension n, modulus 2^64) with noise σ_e."
function lwe_encrypt(s::Vector{Int64}, μ::UInt64, σe::Float64, rng)
    a = rand(rng, UInt64, length(s))
    b = μ + gauss_torus(rng, σe)
    for i in eachindex(s)
        b += a[i] * reinterpret(UInt64, s[i])
    end
    return LWE(a, b)
end

"Round x ∈ T to the nearest multiple of 1/R: returns the integer in [0, R)."
round_to(x::UInt64, R::Int) = Int(mod((widemul(x, UInt64(R)) + (UInt128(1) << 63)) >> 64, R))

"""
    refresh(P, r, K, cts; pool = nothing, Vstar = Inf, recenter = true, rng = Random.default_rng(), info = false)

Error bootstrap of B ciphertexts LWE_s(μ_k) (dimension n): rounding at R = p̂p coordinate by coordinate, with
recentering and selection (round_select, src/rounding.jl; selection only with a pool of encryptions of zero and a
finite Vstar), comb bootstrap of the errors E_k = E_R + c̄ (B = 1: one blind rotation), subtraction of the result from
the noise-free encryption (ā/R, (b̄ + c̄)/R) of μ_k + E_k/R. Returns LWE_{s'}(μ_k) (dimension N), and with info = true
also the Rounded of each ciphertext.
"""
function refresh(P::CBParams, r::Ring, K::CBKeys, cts::Vector{LWE}; pool = nothing, Vstar::Real = Inf,
                 recenter::Bool = true, rng::AbstractRNG = Random.default_rng(), info::Bool = false)
    R = R_of(P); N = P.N
    rs = [round_select(ct, R, pool; Vstar, recenter, rng) for ct in cts]
    abar = [rs[k].abar[u] for k in 1:P.B, u in 1:P.n]
    bbar = [rs[k].bbar for k in 1:P.B]
    errs = P.B == 1 ? [blind_rotation(P, r, K, view(mod.(abar, P.p), 1, :), mod(bbar[1], P.p))] :
                      combboot(P, r, K, mod.(abar, P.p), mod.(bbar, P.p))
    out = LWE[]
    for k in 1:P.B
        a = zeros(UInt64, N)
        for u in 1:P.n
            a[u] = torus_frac(abar[k, u], R)
        end
        push!(out, LWE(a .- errs[k].a, torus_frac(bbar[k], R) - errs[k].b))
    end
    return info ? (out, rs) : out
end
