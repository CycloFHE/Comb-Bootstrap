# Polynomial stair key switch (Section 4.3 and Appendix B of the paper), FFT backend (included in module Fast).
#
# The LWE ciphertext (a, b) of dimension T is read as a GLWE ciphertext of rank T/d over R_d = Z[X]/(X^d + 1):
#     s_j(X) = Σ_{i<d} s'_{(j-1)d+i} X^i,   ã_j(X) = a_{(j-1)d} - Σ_{i≥1} a_{(j-1)d+d-i} X^i,   a·s' = Σ_j [ã_j s_j]_0.
# A step from y = yb d to x = k d + r switches the blocks j > k (and, if r > 0, the tail s_{k+1} - s_{k+1}^(r) of block
# k+1, whose first r coefficients are kept) with GLWE keys under (s_1, ..., s_k[, s_{k+1}^(r)]), of noise σ_128(x):
#     c ← c - Σ_j Σ_l dec_l(ã_j) S_{j,l},   S_{j,l} = GLWE(m_j g_l),   m_j = s_j (or the tail).
# The constant coefficient is extracted at the end: an LWE ciphertext of dimension n under the first n coordinates.
# Products in R_d by complex FFTs of size d/2 with 64-bit keys: their error is negligible against the noise of the
# key switch. The digits have zero mean (decomp_sym!), so that the noise of the keys gives no bias.

export StairKey, stair_keygen, stair_ks, stair_var, σ128

σ128(x::Real) = 2.0^max(-0.02591 * x + 2.77, -62.0)

struct StairStep
    yb::Int                                 # input rank
    k::Int; r::Int; ko::Int                 # x = k d + r, output rank ko = k + (r > 0)
    G::Gadget
    blocks::Vector{Int}                     # switched blocks
    K::Vector{Vector{Matrix{ComplexF64}}}   # K[jj][l]: H × (ko + 1) Fourier values (A_1 .. A_ko, B)
end

struct StairKey
    d::Int; T::Int; n::Int
    steps::Vector{StairStep}
    fp::FPlan
end

"Negacyclic product in R_d, exact modulo 2^64 (key generation)."
function rd_mul!(out::Vector{UInt64}, a::Vector{UInt64}, s::AbstractVector{Int64})
    d = length(a)
    @inbounds for j in 0:d-1
        s[j+1] == 0 && continue
        sj = reinterpret(UInt64, s[j+1])
        for i in 0:d-1
            v = a[i+1] * sj; t = i + j
            if t < d
                out[t+1] += v
            else
                out[t-d+1] -= v
            end
        end
    end
    return out
end

"""Keys of the stair from dimension T (multiple of d) down to n = chain[end][1]; chain = [(x, log2 B, ℓ), ...] gives the
dimension reached by each step and its gadget. σf(x): noise of the keys of a step to x."""
function stair_keygen(sprime::Vector{Int64}, T::Int, chain, rng; d::Int = 16, σf = σ128)
    fp = FPlan(d); H = d ÷ 2
    sp = vcat(sprime, zeros(Int64, max(0, T - length(sprime))))
    blk(j) = sp[(j-1)*d+1:j*d]
    steps = StairStep[]; y = T
    for (x, b, ℓ) in chain
        yb = y ÷ d; k, r = divrem(x, d); ko = k + (r > 0); G = Gadget(b, ℓ); σ = σf(x)
        keyb = [blk(i) for i in 1:k]
        blocks = Int[]; msgs = Vector{Vector{Int64}}()
        if r > 0
            sr = blk(k + 1); sr[r+1:end] .= 0
            push!(keyb, sr)
            push!(blocks, k + 1); push!(msgs, blk(k + 1) .- sr)
            for j in k+2:yb
                push!(blocks, j); push!(msgs, blk(j))
            end
        else
            for j in k+1:yb
                push!(blocks, j); push!(msgs, blk(j))
            end
        end
        K = Vector{Vector{Matrix{ComplexF64}}}()
        for msg in msgs
            Kl = Matrix{ComplexF64}[]
            for l in 1:ℓ
                g = UInt64(1) << (64 - l * b)
                M = Matrix{ComplexF64}(undef, H, ko + 1)
                B = [reinterpret(UInt64, msg[t]) * g + reinterpret(UInt64, round(Int64, randn(rng) * σ * 2.0^64)) for t in 1:d]
                for i in 1:ko
                    A = rand(rng, UInt64, d)
                    rd_mul!(B, A, keyb[i])
                    M[:, i] = ffwd(A, fp)
                end
                M[:, ko+1] = ffwd(B, fp)
                push!(Kl, M)
            end
            push!(K, Kl)
        end
        push!(steps, StairStep(yb, k, r, ko, G, blocks, K))
        y = x
    end
    return StairKey(d, T, chain[end][1], steps, fp)
end

const _DSTATE = Ref{UInt64}(0x9e3779b97f4a7c15)
@inline function _rbit()
    x = _DSTATE[]; x ⊻= x << 13; x ⊻= x >> 7; x ⊻= x << 17; _DSTATE[] = x
    return (x >> 63) % Bool
end

"""Balanced digits with zero mean: as decomp!, but a digit equal to B/2 is replaced by -B/2 (with a carry) with
probability 1/2. The usual rule (digits in [-B/2, B/2 - 1]) has mean -1/2, which multiplies the noise of the keys
into a fixed bias for a given key: half of the variance for B = 2."""
function decomp_sym!(out::Vector{Vector{Int64}}, t::Vector{UInt64}, G::Gadget)
    b = G.logB; ℓ = G.ell; shift = 64 - b * ℓ
    mask = (UInt64(1) << b) - 1; half = Int64(1) << (b - 1); full = Int64(1) << b
    @inbounds for i in eachindex(t)
        x = t[i]
        y = (x >> shift) + ((x >> (shift - 1)) & 0x1)
        carry = Int64(0)
        for j in ℓ:-1:1
            d = Int64(y & mask) + carry
            y >>= b
            if d > half || (d == half && _rbit())
                d -= full; carry = Int64(1)
            else
                carry = Int64(0)
            end
            out[j][i] = d
        end
    end
    return out
end

"Predicted variance of the stair (ρ = E[s_i^2] = 1/2 for a binary key)."
function stair_var(T::Int, chain; d::Int = 16, ρ::Float64 = 0.5, σf = σ128)
    y = T; v = 0.0
    for (x, b, ℓ) in chain
        k, r = divrem(x, d)
        key = ℓ * (2.0^(2b) + 2) / 12 * σf(x)^2; rnd = ρ * 2.0^(-2ℓ * b) / 12
        v += r == 0 ? (y - x) * (key + rnd) : (y - k * d) * key + (y - x) * rnd
        y = x
    end
    return v
end

"Key switch of an LWE ciphertext of dimension ≤ T (under s') to dimension n (under its first n coordinates)."
function stair_ks(K::StairKey, c::LWE)
    d = K.d; H = d ÷ 2; T = K.T; fp = K.fp; nb = T ÷ d
    comps = [zeros(UInt64, d) for _ in 1:nb]
    a = c.a; na = length(a)
    @inbounds for j in 1:nb
        base = (j - 1) * d; C = comps[j]
        C[1] = base + 1 <= na ? a[base+1] : UInt64(0)
        for i in 1:d-1
            u = base + d - i + 1
            C[i+1] = u <= na ? -a[u] : UInt64(0)
        end
    end
    bpoly = zeros(UInt64, d); bpoly[1] = c.b
    ℓmax = maximum(st.G.ell for st in K.steps)
    dig = [zeros(Int64, d) for _ in 1:ℓmax]
    D = Vector{ComplexF64}(undef, H); Y = Vector{ComplexF64}(undef, H); tmp = Vector{UInt64}(undef, d)
    for st in K.steps
        ko = st.ko
        acc = zeros(ComplexF64, H, ko + 1)
        for (jj, j) in enumerate(st.blocks)
            decomp_sym!(dig, comps[j], st.G)
            for l in 1:st.G.ell
                ffwd!(D, dig[l], fp)
                M = st.K[jj][l]
                @inbounds for col in 1:ko+1
                    @simd for h in 1:H
                        acc[h, col] = muladd(D[h], M[h, col], acc[h, col])
                    end
                end
            end
        end
        for i in 1:ko+1
            @inbounds for h in 1:H
                Y[h] = acc[h, i]
            end
            finv!(tmp, Y, fp)
            dst = i <= ko ? comps[i] : bpoly
            @inbounds for t in 1:d
                dst[t] -= tmp[t]
            end
        end
    end
    n = K.n; ao = Vector{UInt64}(undef, n)
    @inbounds for u in 0:n-1
        i, t = divrem(u, d); A = comps[i+1]
        ao[u+1] = t == 0 ? A[1] : -A[d-t+1]
    end
    return LWE(ao, bpoly[1])
end
