"""
    CombBoot

Prototype of the comb bootstrap of B LWE errors (Section 5 of the paper), case (B):
ring Z[X]/(X^N+1), torus modulus q = 2^64, p = 2^β, M0 = 2N/p = B columns.

All products modulo 2^64 are exact (two 32-bit limbs, NTT modulo a 62-bit prime), so that the
measured noise is the noise of the scheme, without FFT rounding errors.

Usage: `include("src/CombBoot.jl"); using .CombBoot` (only standard libraries are needed).
"""
module CombBoot

using Random
using Printf

include("ntt.jl")
include("ring.jl")
include("glwe.jl")
include("algorithm.jl")
include("rounding.jl")
include("noise.jl")
include("product.jl")

export Ring, Gadget, SecretKey, GLWE, LWE, FGGSW, FKSKey
export torus_frac, torus_to_float, gauss_torus, mulX, aut_poly, mul_small_torus, mul_schoolbook
export ntt!, intt!, ntt_small, ntt_aut_perm, decompose, gvalue, gadget_lift
export glwe_encrypt, glwe_phase, ggsw_encrypt, extprod, cmux, ks_keygen, ks_hoisted, ks
export extract, lwe_phase, lwe_encrypt
export CBParams, CBKeys, cb_keygen, vstar, acc0, L_clear, BA_poly, Ugroup, column, divB
export ba_table, ba_hat, ntt_products_ok, NGLWE, nglwe, to_glwe, ks_hoisted_ntt
export combboot, blind_rotation, separate_rotations, random_inputs, refresh, R_of, Emax
export var_ext, var_ks, var_combboot, var_separate, output_errors
export QM, QRLWE, qadd, qsub, qneg, qlift, qfrom, to_q, from_q, qdivN, q_to_float, aut_q, mul_small_q, qrlwe_encrypt, q_phase
export lwe_to_qrlwe, to_glwe_64, ProductKeysRaw, ProductKeys, product_keys_raw, square_key, qks_aut, qtrace
export lwe_to_rlwe_clean, limbs16, scale_phat, tensor, relin, lwe_product, var_ks_gadget, trace_profile, var_product

end
