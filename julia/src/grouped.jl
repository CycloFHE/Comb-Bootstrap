# Grouping of the mask by m (included in module Fast, after fast_opt.jl).
#
# Key s = (s_1, ..., s_n), n = m·G, group t = coordinates u = m(t-1)+1 .. m t, pattern σ_t ∈ {0,1}^m.
# For each group and each pattern π ≠ 0 (π = 1 .. 2^m - 1, bit i of π = coordinate i of the group), the key
# GGSW(1[σ_t = π]), 1[σ_t = π] = Π_{π_i = 1} s_i · Π_{π_i = 0} (1 - s_i).
#
# Comb bootstrap (iteration of Section 5.3 grouped by m, Algorithm 1): with ã_k(π) = Σ_{π_i = 1} ã_{k,i},
#     acc ← acc + Σ_{π ≠ 0} 1[σ_t = π] (L^π(acc) - acc),   L^π(P) = Σ_k X^{ã_k(π)} π_k(P) = Σ_{a∈U} A^π_a P(X^a):
# the automorphisms P(X^a) of acc (division, FFTs, hoisted digits, baby-step key switchings) are computed once
# for the 2^m - 1 patterns; the public products, the giant steps (h ≥ 2), the inverse FFTs of C_L^π, its
# digits and the external product with GGSW(1[σ_t = π]) are done for each pattern; the external products are
# summed in the Fourier domain and the accumulator is updated once. With h = 1 (g = B), there is no giant step:
# all the automorphisms are hoisted baby steps, shared by the patterns.
#
# Separate rotations (multi-bit form): C ← C + K ⊡ C with the combined key K = Σ_{π ≠ 0} (X^{e_π} - 1)·GGSW(1[σ_t = π]),
# e_π = M0·Σ_{π_i = 1} a_{k,i}: one external product per group, plus (2^m - 1) combinations of the key rows
# (in the Fourier domain, X^e is a diagonal multiplication).
#
# Threads (nthreads > 1), as in fast_opt.jl: comb bootstrap of one batch, the threads cooperate inside each group (SPMD,
# spin barriers; the patterns give independent tasks: tables, inverse FFTs and digits of C_L^π); separate
# rotations, the B rotations are shared between the threads; wgof(wg, t): workspace of thread t.

export fast_keygen_grouped, WorkG, wgof, combboot_grouped, separate_rotations_grouped

npat(m::Int) = (1 << m) - 1

"Index of GGSW(1[σ_t = π]) in K.bsk (t, π 1-based)."
@inline kidx(t::Int, π::Int, m::Int) = (t - 1) * npat(m) + π

"Keys of the grouped comb bootstrap / separate rotations: GGSW(1[σ_t = π]) for each group t and pattern π ≠ 0."
function fast_keygen_grouped(P::CBParams, r::Ring, fp::FPlan, rng; m::Int)
    P.n % m == 0 || error("n must be a multiple of m")
    sprime = Int64.(rand(rng, 0:1, P.N))
    sk = SecretKey(r, sprime)
    bsk = FGGSWf[]
    for t in 1:P.n÷m, π in 1:npat(m)
        bit = 1
        for i in 1:m
            si = Int(sprime[(t-1)*m+i])
            bit *= ((π >> (i - 1)) & 1) == 1 ? si : 1 - si
        end
        push!(bsk, fggsw(r, fp, sk, bit, P.Gbr, P.σ, rng))
    end
    M = 2 * P.N; γ = 1 + P.p
    baby = [fkskey(r, fp, sk, powermod(γ, j, M), P.Gks, P.σ, rng) for j in 1:P.g-1]
    giant = [fkskey(r, fp, sk, powermod(γ, P.g * k, M), P.Gks, P.σ, rng) for k in 1:P.h-1]
    return FKeys(sk, sprime[1:P.n], bsk, baby, giant)
end

"""Workspace of the grouped versions: the workspace of fast_opt.jl and the buffers of the 2^m - 1 patterns;
pool: the workspaces of threads 2, 3, ... (same keys, own buffers)."""
struct WorkG
    ws::Work
    m::Int
    Qp::Vector{Vector{SV}}              # public polynomials of each pattern (one value per row κ)
    GAp::Vector{Vector{SV}}; GBp::Vector{Vector{SV}}    # giant accumulators of each pattern (T-layout)
    acol::Vector{Int}
    W::SV                               # X^e - 1 in the Fourier domain (separate rotations)
    KC::Vector{SRow}                    # combined key rows (separate rotations)
    Yp::Vector{NTuple{2,CV}}; Up::Vector{NTuple{2,Vector{UInt64}}}   # per pattern: inverse FFTs of C_L^π
    Dbrp::Vector{Vector{CV}}            # per pattern: the 2ℓ digits of C_L^π - C
    Ygp::Vector{CV}; Ugp::Vector{Vector{UInt64}}; Dgp::Vector{Vector{CV}}; DgTp::Vector{Vector{SV}}  # giant steps
    pool::Vector{WorkG}
    mem::Vector{Vector{Float64}}        # blocks behind the staggered buffers (kept alive)
end

"""n vectors of length H in one block, H + 24 apart (192 bytes). The baby steps read and write the same row of the
giant accumulators of every pattern together: buffers at the same address mod 16 KiB fall in the same sets of the
L1 cache (Apple cores: 128 KiB, 8 ways) and evict each other; the allocator of macOS produces such buffers, the
one of Linux does not. Staggering them removes the conflicts whatever the allocator. The block is pushed to mem."""
function staggered(n::Int, H::Int, mem::Vector{Vector{Float64}})
    st = H + 24
    blk = zeros(st * n); push!(mem, blk)
    p = pointer(blk)
    return [unsafe_wrap(Vector{Float64}, p + 8st * (k - 1), H) for k in 1:n]
end

function _workg(ws::Work, P::CBParams, fp::FPlan, m::Int)
    H = fp.H; L = H ÷ P.M0; np = npat(m); N = P.N
    z() = zeros(ComplexF64, H); u() = zeros(UInt64, N)
    mem = Vector{Float64}[]
    v = staggered(4np * P.h, H, mem)                     # giant accumulators GA, GB (r, i) of all patterns
    sv(π, t, ab) = (k = 4 * ((π - 1) * P.h + t - 1) + 2ab; SV(v[k+1], v[k+2]))
    return WorkG(ws, m, [[SV(L) for _ in ws.idx] for _ in 1:np], [[sv(π, t, 0) for t in 1:P.h] for π in 1:np],
                 [[sv(π, t, 1) for t in 1:P.h] for π in 1:np], zeros(Int, P.B), SV(H),
                 [(SV(H), SV(H)) for _ in 1:2P.Gbr.ell],
                 [(z(), z()) for _ in 1:np], [(u(), u()) for _ in 1:np], [[z() for _ in 1:2P.Gbr.ell] for _ in 1:np],
                 [z() for _ in 1:np], [u() for _ in 1:np], [[z() for _ in 1:P.Gks.ell] for _ in 1:np],
                 [[SV(H) for _ in 1:P.Gks.ell] for _ in 1:np], WorkG[], mem)
end

function WorkG(P::CBParams, fp::FPlan, K::FKeys; m::Int, nthreads::Int = Threads.nthreads())
    ws = Work(P, fp, K; nthreads)
    wg = _workg(ws, P, fp, m)
    for t in 2:nthreads
        push!(wg.pool, _workg(wsof(ws, t), P, fp, m))
    end
    return wg
end

"Workspace of thread t (t = 1: wg itself)."
wgof(wg::WorkG, t::Int) = t == 1 ? wg : wg.pool[t-1]

"Table of c_3 for the rotation amounts acol (p·a_k), public polynomials into Q (as table! of fast_opt.jl)."
function table_acol!(ws::Work, P::CBParams, fp::FPlan, acol::Vector{Int}, Q::Vector{SV})
    S = ws.S; Z = fp.Z; mask = 2P.N - 1
    @inbounds for ci in 1:(P.p÷2), k in 1:P.B
        S[k, ci] = Z[((P.M0 * (2ci - 1) * acol[k]) & mask) + 1]
    end
    ws.pS * S
    @inbounds for t in eachindex(ws.idx)
        id = ws.idx[t]; q = Q[t]
        for k in eachindex(id)
            x = S[id[k]]; q.r[k] = real(x); q.i[k] = imag(x)
        end
    end
    return nothing
end

"""Baby steps shared by the patterns: for each row κ, the key switched rows τ_j((0, B0) - Σ_l D_l·K'_(j,l)) are
computed once, then multiplied by the public values of each pattern and added to its giant accumulators."""
function baby_rows_pat!(::Val{M0}, κs::UnitRange{Int}, ws::Work, Qp::Vector{Vector{SV}}, GAp, GBp,
                        t1::NTuple{4,Vector{Float64}}, t2::NTuple{4,Vector{Float64}}) where {M0}
    h, g = size(ws.tix); ℓ = length(ws.DksT); Kp = ws.bpack; st = 4M0 * ℓ; np = length(Qp)
    @inbounds for κ in κs
        base = M0 * κ; ko = κ * (g - 1) * st
        for π in 1:np, t in 1:h
            q = Qp[π][ws.tix[t, 1]]
            _row_set!(GAp[π][t], GBp[π][t], ws.A0T, ws.B0T, base, q.r[κ+1], q.i[κ+1], Val(M0))
        end
        j = 1
        while j <= g - 1
            s1 = ws.bsh[j][κ+1]
            _row_ks!(t1, base, ws.DksT, Kp, ko, ws.B0T, Val(M0)); ko += st
            if j + 1 <= g - 1
                s2 = ws.bsh[j+1][κ+1]
                _row_ks!(t2, base, ws.DksT, Kp, ko, ws.B0T, Val(M0)); ko += st
                for π in 1:np
                    Q = Qp[π]
                    for t in 1:h
                        u1 = ws.tix[t, j+1]; u2 = ws.tix[t, j+2]
                        q1r = u1 == 0 ? 0.0 : Q[u1].r[κ+1]; q1i = u1 == 0 ? 0.0 : Q[u1].i[κ+1]
                        q2r = u2 == 0 ? 0.0 : Q[u2].r[κ+1]; q2i = u2 == 0 ? 0.0 : Q[u2].i[κ+1]
                        _row_pub2!(GAp[π][t], GBp[π][t], t1, s1, q1r, q1i, t2, s2, q2r, q2i, base, Val(M0))
                    end
                end
                j += 2
            else
                for π in 1:np, t in 1:h
                    u1 = ws.tix[t, j+1]
                    u1 == 0 && continue
                    _row_pub!(GAp[π][t], GBp[π][t], t1, base, s1, Qp[π][u1].r[κ+1], Qp[π][u1].i[κ+1], Val(M0))
                end
                j += 1
            end
        end
    end
    return nothing
end

"Table of c_3 of pattern π (rotation amounts Σ_{π_i = 1} a_{k,i}), with the scratch buffers of me, into Q."
function _pat_table!(me::WorkG, P::CBParams, fp::FPlan, aints::AbstractMatrix{Int}, us::UnitRange{Int}, π::Int,
                     Q::Vector{SV})
    @inbounds for k in 1:P.B
        s = 0
        for i in 1:me.m
            ((π >> (i - 1)) & 1) == 1 && (s += aints[k, us[i]])
        end
        me.acol[k] = s
    end
    table_acol!(me.ws, P, fp, me.acol, Q)
    return nothing
end

"""Groups ts of the comb bootstrap, run by thread tid of T (SPMD, all T threads call it; T = 1: sequential).
Phases, separated by spin barriers (tasks shared round-robin, or slices of rows / points):
 0. before the first group: division (slices of points) and the 2^m - 1 tables of c_3 of the first group;
 1. FFTs of A0, B0 and the ℓ_aut hoisted digits (one task each);
 2. baby steps (key switched rows computed once) and public products of every pattern (slices of rows κ);
 3. giant steps (h ≥ 2): per pattern, inverse FFT and digits (one task each), then key switching (slices);
 4. per pattern and part (a, b): inverse FFT of C_L^π (one task each);
 5. per pattern, part and j ≤ ℓ: digit j of C_L^π - C (one task each);
 6. external products with GGSW(1[σ_t = π]), summed over π (slices of points);
 7. per part: C += inverse FFT, then division of that part for the next group (one task each); in parallel, the
    tables of c_3 of the next group (one task each).
The critical path of a group holds four FFTs (1, 4, 5, 7) and two sliced phases (2, 6).
tim (optional, thread 1): times of the 7 phases, waiting included (phase 0 counted in phase 7). Same results for any
T, and as the order of the operations within each part is fixed, the same results as phases without fusion."""
function cb_group_spmd!(::Val{M0}, C::GLWE, aints::AbstractMatrix{Int}, ts::UnitRange{Int}, P::CBParams,
                        fp::FPlan, wg::WorkG, bar::Barrier, tid::Int, tim = nothing) where {M0}
    ws = wg.ws; me = wgof(wg, tid); m = wg.m; np = npat(m); T = bar.n
    H = fp.H; L = H ÷ M0; ℓ = P.Gbr.ell; ℓk = P.Gks.ell; s = trailing_zeros(P.B)
    rN = _part(P.N, T, tid); rH = _part(H, T, tid)
    κs = _part(L, T, tid) .- 1; pts = M0 * first(κs) + 1 : M0 * (last(κs) + 1)
    t1buf = ws.rows[2tid-1]; t2buf = ws.rows[2tid]
    ls = 0
    clk = tim !== nothing && tid == 1
    t0 = clk ? time_ns() : UInt64(0)
    isempty(ts) && return nothing
    _ph_div!(ws, C, rN, s, P.Gks)                                           # 0
    t = first(ts); us = (t-1)*m+1 : t*m
    for π in tid:T:np
        _pat_table!(me, P, fp, aints, us, π, wg.Qp[π])
    end
    ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[7] += t1 - t0; t0 = t1)
    for t in ts
        for it in tid:T:2+ℓk                                                # 1
            if it == 1
                fwd_torus!(ws.YA, ws.ta, fp); toT!(ws.A0T, ws.YA, M0)
            elseif it == 2
                fwd_torus!(ws.YB, ws.tb, fp); toT!(ws.B0T, ws.YB, M0)
            else
                l = it - 2
                digit_fwd!(ws.Dks[l], ws.ta, nothing, P.Gks, l, fp); toT!(ws.DksT[l], ws.Dks[l], M0)
            end
        end
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[1] += t1 - t0; t0 = t1)
        baby_rows_pat!(Val(M0), κs, ws, wg.Qp, wg.GAp, wg.GBp, t1buf, t2buf)  # 2
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[2] += t1 - t0; t0 = t1)
        for kk in 1:P.h-1                                                   # 3
            for π in tid:T:np
                fromT!(wg.Ygp[π], wg.GAp[π][kk+1], M0); inv_torus!(wg.Ugp[π], wg.Ygp[π], fp, false)
                for l in 1:ℓk
                    digit_fwd!(wg.Dgp[π][l], wg.Ugp[π], nothing, P.Gks, l, fp); toT!(wg.DgTp[π][l], wg.Dgp[π][l], M0)
                end
            end
            ls = sync!(bar, ls)
            for π in 1:np
                keyswitch!(ws.OA, ws.OB, wg.DgTp[π], ws.giant[kk], wg.GBp[π][kk+1], pts)
                rotadd!(wg.GAp[π][1], wg.GBp[π][1], ws.OA, ws.OB, ws.gsh[kk], M0, κs)
            end
            ls = sync!(bar, ls)
        end
        clk && (t1 = time_ns(); tim[3] += t1 - t0; t0 = t1)
        for it in tid:T:2np                                                 # 4
            π = (it + 1) >> 1; part = 2 - (it & 1)
            Y = wg.Yp[π][part]
            fromT!(Y, part == 1 ? wg.GAp[π][1] : wg.GBp[π][1], M0); inv_torus!(wg.Up[π][part], Y, fp, false)
        end
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[4] += t1 - t0; t0 = t1)
        for it in tid:T:2ℓ*np                                               # 5
            π = (it - 1) ÷ 2ℓ + 1; r = (it - 1) % 2ℓ + 1                    # r ≤ ℓ: mask, r > ℓ: body
            part = r <= ℓ ? 1 : 2; j = r <= ℓ ? r : r - ℓ
            digit_fwd!(wg.Dbrp[π][r], wg.Up[π][part], part == 1 ? C.a : C.b, P.Gbr, j, fp)
        end
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[5] += t1 - t0; t0 = t1)
        for π in 1:np                                                       # 6
            rows = ws.bsk[kidx(t, π, m)]; D = wg.Dbrp[π]
            for r in 1:2ℓ
                kmacc!(ws.accA, ws.accB, D[r], rows[r][1], rows[r][2], !(π == 1 && r == 1), rH)
            end
        end
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[6] += t1 - t0; t0 = t1)
        nxt = t < last(ts)                                                  # 7
        for it in tid:T:(nxt ? 2 + np : 2)
            if it == 1
                inv_torus!(C.a, ws.accA, fp, true); nxt && _ph_div_part!(ws, C, 1, 1:P.N, s, P.Gks)
            elseif it == 2
                inv_torus!(C.b, ws.accB, fp, true); nxt && _ph_div_part!(ws, C, 2, 1:P.N, s, P.Gks)
            else
                π = it - 2; un = t*m+1 : (t+1)*m
                _pat_table!(me, P, fp, aints, un, π, wg.Qp[π])
            end
        end
        ls = sync!(bar, ls); clk && (t1 = time_ns(); tim[7] += t1 - t0; t0 = t1)
    end
    return nothing
end

"""Comb bootstrap of the B errors (columns of aints), mask grouped by m (keys of fast_keygen_grouped).
nthreads > 1: the threads cooperate inside each group (latency); wg must have been built with ≥ nthreads threads."""
function combboot_grouped(P::CBParams, fp::FPlan, K::FKeys, aints::AbstractMatrix{Int}, bints::Vector{Int};
                           m::Int, wg::WorkG = WorkG(P, fp, K; m), nthreads::Int = 1, tim = nothing)
    @assert wg.m == m
    C = trivial(acc0(P, bints))
    nu = size(aints, 2); nu % m == 0 || error("number of columns must be a multiple of m")
    ts = 1:nu÷m
    if nthreads == 1
        cb_group_spmd!(Val(P.M0), C, aints, ts, P, fp, wg, Barrier(1), 1, tim)
    else
        @assert nthreads <= min(Threads.nthreads(), length(wg.pool) + 1) "start julia with -t $nthreads"
        bar = Barrier(nthreads)
        Threads.@threads :static for tid in 1:nthreads
            cb_group_spmd!(Val(P.M0), C, aints, ts, P, fp, wg, bar, tid, tim)
        end
    end
    return [extract(C, k) for k in 0:P.B-1]
end

"KC = Σ_{π ≠ 0} (X^{e_π} - 1)·GGSW(1[σ_t = π]) (rows split, Fourier domain), e_π = Σ_{π_i = 1} es[i] (mod 2N)."
function combine_key!(wg::WorkG, es::NTuple{4,Int}, t::Int, P::CBParams, fp::FPlan)
    ws = wg.ws; m = wg.m; H = fp.H; mask = 2P.N - 1; Z = fp.Z; W = wg.W; Wr = W.r; Wi = W.i
    for π in 1:npat(m)
        e = 0
        for i in 1:m
            ((π >> (i - 1)) & 1) == 1 && (e += es[i])
        end
        e &= mask
        c = e; step = (4e) & mask
        @inbounds for k in 1:H                               # W = ζ^{e(4k+1)} - 1
            z = Z[c+1]; Wr[k] = real(z) - 1.0; Wi[k] = imag(z); c = (c + step) & mask
        end
        rows = ws.bsk[kidx(t, π, m)]
        for r in eachindex(rows), part in 1:2
            src = rows[r][part]; dst = wg.KC[r][part]
            sr = src.r; si = src.i; dr = dst.r; di = dst.i
            if π == 1
                @inbounds @simd for k in 1:H
                    dr[k] = Wr[k] * sr[k] - Wi[k] * si[k]; di[k] = Wr[k] * si[k] + Wi[k] * sr[k]
                end
            else
                @inbounds @simd for k in 1:H
                    dr[k] += Wr[k] * sr[k] - Wi[k] * si[k]; di[k] += Wr[k] * si[k] + Wi[k] * sr[k]
                end
            end
        end
    end
    return nothing
end

"C ← C + K ⊡ C with the combined key of wg.KC."
function _mbit!(C::GLWE, P::CBParams, wg::WorkG, fp::FPlan)
    ws = wg.ws; ℓ = P.Gbr.ell
    digits_fwd!(ws.Dbr, 0, C.a, P.Gbr, fp)
    digits_fwd!(ws.Dbr, ℓ, C.b, P.Gbr, fp)
    for r in 1:2ℓ
        kmacc!(ws.accA, ws.accB, ws.Dbr[r], wg.KC[r][1], wg.KC[r][2], r > 1)
    end
    inv_torus!(C.a, ws.accA, fp, true); inv_torus!(C.b, ws.accB, fp, true)
    return C
end

"""B separate blind rotations, mask grouped by m (multi-bit form), batched: for each group, the B rotations
advance together. m = 1: the CMux form of fast_opt.jl (one external product per coordinate, no combination).
nthreads > 1: the B rotations are shared between the threads (each with its own workspace)."""
function separate_rotations_grouped(P::CBParams, fp::FPlan, K::FKeys, aints::AbstractMatrix{Int}, bints::Vector{Int};
                                    m::Int, wg::WorkG = WorkG(P, fp, K; m), nthreads::Int = 1)
    N = P.N; mask = 2N - 1
    v = vstar(P)
    Cs = [trivial(mulX(v, -P.M0 * bints[k])) for k in 1:P.B]
    nu = size(aints, 2)
    go(tid) = begin
        w = wgof(wg, tid); ks = tid:nthreads:P.B
        if m == 1
            for u in 1:nu, k in ks
                cmux_rot!(Cs[k], mod(P.M0 * aints[k, u], 2N), u, P, w.ws, fp)
            end
        else
            for t in 1:nu÷m, k in ks
                es = ntuple(i -> i <= m ? (P.M0 * aints[k, (t-1)*m+i]) & mask : 0, 4)
                combine_key!(w, es, t, P, fp)
                _mbit!(Cs[k], P, w, fp)
            end
        end
    end
    if nthreads == 1
        go(1)
    else
        @assert nthreads <= min(Threads.nthreads(), length(wg.pool) + 1) "start julia with -t $nthreads"
        Threads.@threads :static for tid in 1:nthreads
            go(tid)
        end
    end
    return [extract(C, 0) for C in Cs]
end
