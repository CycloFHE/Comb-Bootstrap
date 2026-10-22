# Configurations of the comb bootstrap at 2, 4, 8, 12 and 16 bits (Table 3 of the paper), and the predicted output
# variance, shared by run_comb_sizes.jl, run_comb_threads.jl and run_comb_keys_corr.jl (included after run_curves.jl,
# which defines Gadget and gstr).
# bits => (p, RGSW gadget, automorphism gadget, n, 16 V*/p^2, chain of the key switch N -> n)
const SIZES = Dict(
    2 => (256, Gadget(23, 1), Gadget(14, 2), 608, 2.8317e-03,
          [(1280, 15, 1), (960, 7, 2), (864, 5, 3), (816, 4, 4), (768, 3, 5), (736, 2, 7), (720, 2, 7), (704, 2, 7),
           (688, 2, 7), (672, 1, 13), (656, 1, 12), (640, 1, 12), (624, 1, 12), (608, 1, 11)]),
    4 => (256, Gadget(14, 2), Gadget(14, 2, 4), 688, 3.2219e-03,
          [(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8),
           (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13)]),
    8 => (256, Gadget(11, 3), Gadget(11, 3, 4), 896, 4.2407e-03,
          [(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7),
           (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20),
           (912, 1, 19), (896, 1, 19)]),
    12 => (512, Gadget(11, 3), Gadget(11, 3, 3), 986, 1.1707e-03,
           [(1664, 13, 2), (1488, 9, 3), (1408, 7, 4), (1328, 5, 5), (1280, 4, 7), (1248, 3, 9), (1216, 3, 9),
            (1184, 2, 13), (1168, 2, 13), (1152, 2, 13), (1136, 2, 12), (1120, 1, 24), (1104, 1, 24), (1088, 1, 24),
            (1072, 1, 23), (1056, 1, 23), (1040, 1, 23), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (986, 1, 21)]),
    16 => (512, Gadget(8, 5), Gadget(6, 6, 3), 1157, 1.3813e-03,
           [(1872, 15, 2), (1696, 10, 3), (1600, 8, 4), (1520, 6, 5), (1472, 5, 6), (1424, 4, 8), (1392, 3, 10),
            (1360, 3, 10), (1344, 2, 15), (1328, 2, 15), (1312, 2, 15), (1296, 2, 14), (1280, 1, 28), (1264, 1, 28),
            (1248, 1, 28), (1232, 1, 28), (1216, 1, 27), (1200, 1, 27), (1184, 1, 26), (1168, 1, 26), (1157, 1, 26)]),
)
gs(G) = G.off == 0 ? gstr(G) : "(2^$(G.logB),$(G.ell)) shifted by $(G.off)"

const CAUT = 2.0^-88.5                               # error of the FFT of the automorphism key switchings (fitted)
"""Predicted output variance: Proposition 1 (h = 1) for the grouping m in the rotation, with nst active groups (n1, the
number of ones of the key, for m = 1), and the errors of the FFT (per external product and per active group)."""
function var_mbs(P, nst, m = 1)
    key, rnd = var_ext(P); ne = P.n ÷ m * (2^m - 1)                     # external products: (n/m)(2^m - 1)
    return ne * key + nst * (rnd + ((P.B - P.h) * P.B + P.h - 1) * var_ks(P)) + ne * 2.0^-85 * 2.0^(2P.Gbr.logB) * P.Gbr.ell +
           nst * (P.B - 1) * P.B * CAUT * 2.0^(2P.Gks.logB) * P.Gks.ell
end
"Active groups of the key s for the grouping m (groups of m coordinates not all zero)."
nactive(s, m) = count(t -> any(!iszero, view(s, (t-1)*m+1:t*m)), 1:length(s)÷m)
