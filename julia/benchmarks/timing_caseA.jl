using Random
include("../src/CombBoot.jl"); using .CombBoot
include("../src/fast.jl"); using .Fast
include("../src/caseA.jl"); using .CaseA
import .CombBoot: Gadget
rng = MersenneTwister(1)
for (p, M0) in ((2039, 1), (257, 16))
    Rg = ARing(p, M0); sk = akeygen_sk(Rg, rng)
    K = aggsw_encrypt(Rg, sk, 1, Gadget(11, 3), 2.0^-50, rng)
    C = aglwe_encrypt(Rg, sk, rand_aug(Rg, rng), 2.0^-50, rng)
    aextprod(Rg, C, K)
    t = @elapsed for _ in 1:20; aextprod(Rg, C, K); end
    println("p=$p M0=$M0 N=$(Rg.N) NK=$(Rg.NK): extprod (2^11,3) $(round(t/20*1e3, digits=2)) ms")
    if M0 > 1
        ks = [aks_keygen(Rg, sk, c, Gadget(11, 3), 2.0^-50, rng) for c in Ugroup_A(Rg)[2:end]]
        aks_hoisted(Rg, C, ks)
        t = @elapsed for _ in 1:5; aks_hoisted(Rg, C, ks); end
        println("   hoisted KS ($(length(ks)) keys) $(round(t/5*1e3, digits=2)) ms")
    end
end
