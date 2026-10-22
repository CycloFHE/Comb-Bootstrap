# Time of an LWE key switch T -> n (Mac VM, 1 thread): nb ciphertexts share each row of the key switching key.
# Result used in the cost model: about 0.41 ns per multiply-add with a non-zero
# signed digit (batching hardly helps: no 64-bit vector multiply on NEON). Run: julia -O3 benchmarks/ks_bench.jl
using Random
function digits!(D::Matrix{Int64}, a::Vector{UInt64}, n, T, beta, ell)   # D[(j-n-1)*ell+l] signed digits
    mask = (UInt64(1) << beta) - 1; half = UInt64(1) << (beta - 1); sh = 64 - beta * ell
    @inbounds for j in n+1:T
        x = sh > 0 ? (a[j] + (UInt64(1) << (sh - 1))) >> sh : a[j]; carry = UInt64(0)
        for l in ell:-1:1
            d = (x & mask) + carry; x >>= beta; carry = d >= half ? UInt64(1) : UInt64(0)
            D[(j - n - 1) * ell + l] = reinterpret(Int64, d - (carry << beta))
        end
    end
end
function ksb!(O::Matrix{UInt64}, Ds::Vector{Vector{Int64}}, K::Matrix{UInt64}, n::Int)
    nb = length(Ds); fill!(O, 0)
    @inbounds for col in 1:size(K, 2)
        for c in 1:nb
            d = reinterpret(UInt64, Ds[c][col])
            d == 0 && continue
            @simd for i in 1:n+1
                O[i, c] -= d * K[i, col]
            end
        end
    end
end
function bench(T, n, beta, ell, nb)
    rng = Xoshiro(1)
    K = rand(rng, UInt64, n + 1, (T - n) * ell); O = zeros(UInt64, n + 1, nb)
    Ds = [zeros(Int64, (T - n) * ell) for _ in 1:nb]
    for c in 1:nb
        D = reshape(Ds[c], :, 1); digits!(D, rand(rng, UInt64, T), n, T, beta, ell)
    end
    ksb!(O, Ds, K, n)
    t = minimum(@elapsed(ksb!(O, Ds, K, n)) for _ in 1:3)
    macs = (T - n) * ell * (n + 1) * nb
    println("T=$T n=$n (2^$beta,$ell) batch $nb: $(round(t*1e3/nb, digits=2)) ms per ciphertext, $(round(t*1e9/macs, digits=3)) ns per nominal MAC")
end
for nb in (1, 4, 16)
    bench(2048, 684, 1, 15, nb)
end
bench(2048, 684, 3, 5, 16)
bench(2048, 684, 4, 4, 16)
