//! Comb bootstrap (Algorithm 1 of the paper, case (B): p = 2^β, M = M0 p = 2N, B = M0 columns, h = 1, m = 1) on TFHE-rs 1.8.1,
//! port of julia/src/algorithm.jl (combboot, exact reference) with the products in the Fourier domain of TFHE-rs:
//!   C = acc_0 = Σ_k X^{-M0 b_k} π_k(v*);  for u = 1..n:  C0 = C/B (rounded shift; digits of the shifted gadget),
//!   C_j = KS_{τ_j}(τ_j(C0)) for the B - 1 automorphisms τ_j: X ↦ X^{γ^j} (γ = 1 + p; one decomposition: hoisting),
//!   C_L = Σ_j Q_j C_j with the public polynomials Q_j = B·A_{u,γ^j} = Σ_k X^{M0 a_{k,u} + k(1-γ^j)},
//!   C = CMux(RGSW(s_u), C, C_L) (cmux of TFHE-rs);  outputs: the B columns, LWE_{s'}(E_k/R).
//! Fourier layout ("e-layout", as the T-layout of julia/src/fast_opt.jl): the N/2 values of TFHE-rs (roots ψ^{c_i},
//! c_i read on the transform of X) are reordered by e, c_i = ±5^e (conjugated when -), e = κ + L m (L = p/4, m < B).
//! Then τ_j (γ^j = 5^{s_j}, s_j ∈ L Z) is the rotation of the vector by s_j, i.e. a rotation of the B blocks of L
//! values, and Q_j(ψ^{5^e}) depends on κ only: its L values come from a half table T[κ][t] = Σ_k ψ^{M0 (5^κ mod p) a_k}
//! ω^{t k} (one DFT of size B per κ). The key switching of a source block and its product by Q_j are done in one pass
//! (keys pre-rotated), all loops contiguous. combboot_mt: the same on t threads, bit-identical results.
use crate::kern::{ks_block, ks_src};
use crate::params::{log2_sigma128, LOG2_SIGMA_N, N};
use crate::rounding::{round_select, round_select_r, Rounded};
use crate::stair::StairKey;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial;
use tfhe::core_crypto::prelude::*;
use tfhe_fft::c64;

pub fn cm() -> CiphertextModulus<u64> { CiphertextModulus::new_native() }

/// Configuration of the comb bootstrap (julia/benchmarks/comb_sizes.jl): p, gadgets (log2 B, ℓ[, offset]), n, 16 V*/p^2,
/// chain of the stair key switch N -> n.
pub struct CombConfig { pub bits: u32, pub p: usize, pub gbr: (usize, usize), pub gks: (u32, usize, u32), pub n: usize,
                        pub vterm: f64, pub chain: &'static [(usize, u32, usize)] }

pub const COMB_CONFIGS: [CombConfig; 6] = [
    CombConfig { bits: 2, p: 256, gbr: (23, 1), gks: (14, 2, 0), n: 608, vterm: 2.8317e-03,
        chain: &[(1280, 15, 1), (960, 7, 2), (864, 5, 3), (816, 4, 4), (768, 3, 5), (736, 2, 7), (720, 2, 7), (704, 2, 7), (688, 2, 7), (672, 1, 13), (656, 1, 12), (640, 1, 12), (624, 1, 12), (608, 1, 11)] },
    CombConfig { bits: 4, p: 256, gbr: (14, 2), gks: (14, 2, 4), n: 688, vterm: 3.2219e-03,
        chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13)] },
    // 6 bits: the noise model (B = 16): cheapest at failure 2^-128, margin 0.2 bit after 1000 additions (predicted 0.212)
    CombConfig { bits: 6, p: 256, gbr: (14, 2), gks: (14, 2, 4), n: 784, vterm: 3.6915e-03,
        chain: &[(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16)] },
    CombConfig { bits: 8, p: 256, gbr: (11, 3), gks: (11, 3, 4), n: 896, vterm: 4.2407e-03,
        chain: &[(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7), (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19)] },
    CombConfig { bits: 12, p: 512, gbr: (11, 3), gks: (11, 3, 3), n: 986, vterm: 1.1707e-03,
        chain: &[(1664, 13, 2), (1488, 9, 3), (1408, 7, 4), (1328, 5, 5), (1280, 4, 7), (1248, 3, 9), (1216, 3, 9), (1184, 2, 13), (1168, 2, 13), (1152, 2, 13), (1136, 2, 12), (1120, 1, 24), (1104, 1, 24), (1088, 1, 24), (1072, 1, 23), (1056, 1, 23), (1040, 1, 23), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (986, 1, 21)] },
    CombConfig { bits: 16, p: 512, gbr: (8, 5), gks: (6, 6, 3), n: 1157, vterm: 1.3813e-03,
        chain: &[(1872, 15, 2), (1696, 10, 3), (1600, 8, 4), (1520, 6, 5), (1472, 5, 6), (1424, 4, 8), (1392, 3, 10), (1360, 3, 10), (1344, 2, 15), (1328, 2, 15), (1312, 2, 15), (1296, 2, 14), (1280, 1, 28), (1264, 1, 28), (1248, 1, 28), (1232, 1, 28), (1216, 1, 27), (1200, 1, 27), (1184, 1, 26), (1168, 1, 26), (1157, 1, 26)] },
];

/// RNS: the 8 pairwise coprime moduli p̂_k ≤ 263 of largest product (2^64.03): a 64-bit integer x is held as the 8
/// residues x mod p̂_k, one per column of a CombBoot with B = 8 (refresh_rns: the 8 refreshed together).
pub const RNS_MODULI: [u64; 8] = [263, 262, 261, 259, 257, 253, 251, 247];

/// Configuration of CombBoot for RNS_MODULI (the noise model: cheapest at failure 2^-128 with a margin of 0.2 bit
/// after 1000 additions, for p̂ = 263, case (B), B = 8, p = 512; predicted margin 0.26 bit). `bits` (8) only serves the
/// power-of-two methods (p̂ = 256); the methods _rns take the moduli of the columns.
pub const COMB_RNS: CombConfig = CombConfig { bits: 8, p: 512, gbr: (14, 2), gks: (14, 2, 3), n: 832, vterm: 9.8167e-04,
    chain: &[(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17)] };

/// RNS of 32-bit integers with 8 small moduli (bin rns32_arith): the 8 pairwise coprime integers of smallest maximum
/// whose product reaches 2^32 (2^33.05). CombBoot configurations for p̂ = 25, the largest (the noise model,
/// B = 16 and 8: cheapest at failure 2^-128 with a margin of 0.2 bit after 1000 additions, case (B); predicted margins 0.24 and
/// 0.20 bit): B = 16 (p = 256) refreshes two integers per call, B = 8 (p = 512) one. `bits` only serves the
/// power-of-two methods.
pub const RNS32_MODULI: [u64; 8] = [25, 23, 21, 19, 17, 16, 13, 11];
pub const COMB_RNS32_B16: CombConfig = CombConfig { bits: 4, p: 256, gbr: (14, 2), gks: (14, 2, 4), n: 720, vterm: 3.3783e-03,
    chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14)] };
pub const COMB_RNS32_B8: CombConfig = CombConfig { bits: 4, p: 512, gbr: (14, 2), gks: (14, 2, 3), n: 680, vterm: 7.9571e-04,
    chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (680, 1, 13)] };

/// RNS of 16- and 32-bit integers with small moduli (bin rns_small_arith), one integer per CombBoot: refresh (R) and refresh
/// before a product (P). P is the cheapest configuration that is itself a valid refresh and whose outputs, taken as the
/// two operands of the product (trace (2^8,5), relinearization (2^10,4)), are refreshed by P after the product with
/// failure 2^-128 under the exact law of the main term; the operands come from two distinct calls (independent errors).
/// the noise model (margin 0.2 bit, 1000 additions, case (B)).
/// 32 bits: RNS32_MODULI (p̂ ≤ 25), B = 8 (p = 512): R = COMB_RNS32_B8, P below (predicted output 2^-24.61).
pub const COMB_RNS32P_B8: CombConfig = CombConfig { bits: 4, p: 512, gbr: (11, 3), gks: (11, 3, 3), n: 680, vterm: 7.9571e-04,
    chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (680, 1, 13)] };
/// 32 bits, two integers per call: B = 16 (p = 256): R = COMB_RNS32_B16, P below (predicted output 2^-23.99).
pub const COMB_RNS32P_B16: CombConfig = CombConfig { bits: 4, p: 256, gbr: (11, 3), gks: (11, 3, 4), n: 720, vterm: 3.3783e-03,
    chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14)] };
/// 16 bits: the 4 pairwise coprime integers of smallest maximum whose product reaches 2^16 (77520, 2^16.24), B = 4
/// (p = 1024), one integer per call.
pub const RNS16_MODULI: [u64; 4] = [19, 17, 16, 15];
pub const COMB_RNS16_B4: CombConfig = CombConfig { bits: 4, p: 1024, gbr: (14, 2), gks: (14, 2, 2), n: 656, vterm: 1.9161e-04,
    chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13), (672, 1, 13), (656, 1, 13)] };
pub const COMB_RNS16P_B4: CombConfig = CombConfig { bits: 4, p: 1024, gbr: (14, 2), gks: (11, 3, 2), n: 688, vterm: 2.0137e-04,
    chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13)] };
/// 16 bits, two integers per call: B = 8 (p = 512).
pub const COMB_RNS16_B8: CombConfig = CombConfig { bits: 4, p: 512, gbr: (14, 2), gks: (14, 2, 3), n: 667, vterm: 7.7985e-04,
    chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13), (672, 1, 13), (667, 1, 13)] };
pub const COMB_RNS16P_B8: CombConfig = CombConfig { bits: 4, p: 512, gbr: (11, 3), gks: (14, 2, 3), n: 688, vterm: 8.0548e-04,
    chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14)] };

#[inline] fn cmul(a: c64, b: c64) -> c64 { c64 { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re } }

/// Shifted gadget (log2 B, ℓ, off): g_j = 2^(64 - off - j log2 B); balanced digits in [-B/2, B/2) (decompose of glwe.jl),
/// by the offset trick: y + Σ_j (B/2) B^j has the digits d_j + B/2 in [0, B) (no carries).
fn decompose(b: u32, l: usize, off: u32, t: &[u64], out: &mut [Vec<i64>]) {
    let shift = 64 - off - b * l as u32;
    let mask = (1u64 << b) - 1; let half = 1u64 << (b - 1);
    let offs: u64 = (0..l).fold(0u64, |acc, j| acc.wrapping_add(half << (j as u32 * b)));
    for j in 0..l {
        let sj = (l - 1 - j) as u32 * b; let o = &mut out[j][..t.len()];
        for (d, &x) in o.iter_mut().zip(t) {
            let y = ((x >> shift) + ((x >> (shift - 1)) & 1)).wrapping_add(offs);
            *d = ((y >> sj) & mask) as i64 - half as i64;
        }
    }
}

/// Representative of t reconstructed by its digits (gadget_lift of glwe.jl): differs from t by a multiple of 2^-off.
fn gadget_lift(b: u32, l: usize, off: u32, t: &mut [u64], dig: &mut [Vec<i64>]) {
    if off == 0 { return; }
    decompose(b, l, off, t, dig);
    let shift = 64 - off - b * l as u32;
    for i in 0..t.len() {
        let x = t[i]; let y = (x >> shift) + ((x >> (shift - 1)) & 1);
        let mut rec = 0u64;
        for j in 0..l { rec = rec.wrapping_add((dig[j][i] as u64).wrapping_mul(1u64 << (64 - off - (j as u32 + 1) * b))); }
        t[i] = x.wrapping_sub(y << shift).wrapping_add(rec);
    }
}

pub struct CombKeys {
    pub bits: u32, pub p: usize, pub bcols: usize, pub n: usize, pub log_r: u32, pub vstar: f64, pub gks: (u32, usize, u32),
    pub big_sk: LweSecretKeyOwned<u64>, pub small_sk: LweSecretKeyOwned<u64>,
    /// grouping of the mask in the rotation: m = 1 (CMux of TFHE-rs, keys GGSW(s_u)) or m = 2 (keys GGSW of the
    /// indicators of the patterns 10, 01, 11 of each pair, and GGSW(s_n) for the last coordinate when n is odd)
    pub m: usize,
    bsk: Option<FourierLweBootstrapKeyOwned>, ind: Option<FourierLweBootstrapKeyOwned>,
    pub stair: StairKey, pub pool: Vec<(Vec<u64>, u64)>,
    roots: Vec<c64>,                       // ψ^e, e = 0..2N-1
    vstar_poly: Vec<u64>,
    fft: Fft,
    // e-layout: index e = κ + L·m (L = p/4, m < B) holds the value at ψ^{5^e}; τ_{γ^j} is the rotation e ↦ e + s_j
    // (s_j multiple of L) and Q_j depends on κ only
    pub lrow: usize,
    eord: Vec<u32>,                        // [e]: Fourier index of TFHE-rs whose root is ±5^e
    esgn: Vec<f64>,                        // -1 if that root is -5^e (conjugate the value)
    eshift: Vec<usize>,                    // [j]: s_j
    ksk_e: Vec<Vec<[Vec<f64>; 4]>>,        // [j][level]: keys in the e-layout, pre-rotated (source space of τ_j)
    ksk_s: Vec<f64>,                       // m = 1: the same, source-major: [s][j-1][level][A.re, A.im, B.re, B.im][L]
                                           // for the source block s (KS_MODE = 1)
    kpad: Vec<f64>,                        // m = 1: ksk_e at the stride N/2 + PADW: [j][level][A.re, A.im, B.re, B.im] (KS_MODE = 0)
    tq: Vec<Vec<u32>>,                     // [j][κ]: index of Q_j at 5^κ in the half table (L rows)
    ekc_e: Vec<usize>,                     // [κ]: M0 (5^κ mod p)
}

/// In-place DFT of size B (power of two) with the root w = ψ^p of order B: x[t] <- Σ_k x[k] w^{t k} (radix 2).
fn dft_pow2(x: &mut [c64], tw: &[c64]) {
    let n = x.len(); let mut j = 0;
    for i in 1..n { let mut bit = n >> 1; while j & bit != 0 { j ^= bit; bit >>= 1; } j |= bit; if i < j { x.swap(i, j); } }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for st in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = tw[k * step];
                let u = x[st + k]; let v = cmul(x[st + k + len / 2], w);
                x[st + k] = c64 { re: u.re + v.re, im: u.im + v.im };
                x[st + k + len / 2] = c64 { re: u.re - v.re, im: u.im - v.im };
            }
        }
        len <<= 1;
    }
}

/// Order of the key switchings of the automorphisms and of the public products in combboot (m = 1): 0, automorphism by
/// automorphism, block by block (the first version); 1 (default), source block by source block: the digits and the FFT of
/// b of a block are read once for the B - 1 automorphisms, the keys stored source-major (one contiguous stream), the
/// destination blocks of the accumulators in L1. Mode 1 sums over the automorphisms in another order: the outputs differ
/// by roundings only (same distribution of the noise). On the Apple M3 Max, with KS_PAD, 1.00 to 1.03 times as fast as
/// mode 0 (results/rust/comb_kab_mac2.txt, where it is mode 3). 2, 3, 4: the same order, the digits and the FFT of b of
/// W = 4, 2 or 8 values of the source block held in registers for the B - 1 automorphisms (kern::ks_src; fewer loads,
/// bit-identical to mode 1). KS_AUTO (default): mode 2 for a gadget of at most 2 levels (2 and 4 bits: 1.05 and 1.04 times
/// as fast as mode 1 on the Apple M3 Max), mode 1 beyond (equal or faster), results/rust/comb_kab_mac3.txt.
pub static KS_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(KS_AUTO);
pub const KS_AUTO: usize = 99;

/// Working arrays and keys of the key switchings of combboot (m = 1) at a stride of N/2 + PADW values (true, default) or
/// N/2 (false: separate arrays, as allocated). Arrays of 8 KB allocated one after the other, or on 16 KB pages, start on
/// the same sets of the L1 cache (Apple M3: 128 KB, 8 ways, lines of 128 B, sets repeating every 16 KB): the 20 to 26
/// streams of the fused kernel then evict each other. A shift of 3 lines per array spreads them. Results unchanged; on the
/// Apple M3 Max, 1.03 to 1.14 times as fast (results/rust/comb_kab_mac2.txt, modes 0u and 0).
pub static KS_PAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
const PADW: usize = 48;

/// Grouping of the comb bootstrap of a run: environment variable COMBM (default 1: on the Apple M3 Max, m = 2 is 15 to 20%
/// slower up to 12 bits and on par at 16 bits, results/rust/comb_ab_mac.txt: the public products and the backward FFTs
/// of the three patterns cost more than the key switchings they share).
pub fn comb_m() -> usize { std::env::var("COMBM").ok().and_then(|v| v.parse().ok()).unwrap_or(1) }

impl CombKeys {
    /// Keys with the grouping of the run (comb_m()).
    pub fn new(cfg: &CombConfig, seed: u64) -> Self { Self::new_m(cfg, comb_m(), seed) }

    /// Keys with the grouping m (1 or 2) in the rotation.
    pub fn new_m(cfg: &CombConfig, m: usize, seed: u64) -> Self {
        assert!(m == 1 || m == 2, "m = 1 or 2");
        let mut boxed = new_seeder(); let seeder = boxed.as_mut();
        let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(seed as u128));
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
        let glwe_sk = GlweSecretKey::generate_new_binary(GlweDimension(1), PolynomialSize(N), &mut sgen);
        let big_sk = glwe_sk.clone().into_lwe_secret_key();
        let small_sk = LweSecretKey::from_container(big_sk.as_ref()[..cfg.n].to_vec());
        let glwe_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(LOG2_SIGMA_N)), 0.0);
        // m = 1: GGSW(s_u); m = 2: GGSW of the indicators of the patterns of each pair (and GGSW(s_n) for an odd n),
        // generated by TFHE-rs as the bootstrapping key of the vector of the indicators
        let gen_bsk = |key: &LweSecretKeyOwned<u64>, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| {
            let std = par_allocate_and_generate_new_lwe_bootstrap_key(key, &glwe_sk, DecompositionBaseLog(cfg.gbr.0),
                DecompositionLevelCount(cfg.gbr.1), glwe_noise, cm(), egen);
            let mut f = FourierLweBootstrapKey::new(std.input_lwe_dimension(), std.glwe_size(), std.polynomial_size(),
                std.decomposition_base_log(), std.decomposition_level_count());
            par_convert_standard_lwe_bootstrap_key_to_fourier(&std, &mut f);
            f
        };
        let (bsk, ind) = if m == 1 { (Some(gen_bsk(&small_sk, &mut egen)), None) } else {
            let s = small_sk.as_ref();
            let mut v: Vec<u64> = s.chunks_exact(2).flat_map(|p| [p[0] & !p[1] & 1, !p[0] & p[1] & 1, p[0] & p[1]]).collect();
            if s.len() % 2 == 1 { v.push(s[s.len() - 1]); }
            (None, Some(gen_bsk(&LweSecretKey::from_container(v), &mut egen)))
        };
        let fft = Fft::new(PolynomialSize(N));
        let two_n = 2 * N;
        let roots: Vec<c64> = (0..two_n).map(|e| { let t = std::f64::consts::PI * e as f64 / N as f64; c64 { re: t.cos(), im: t.sin() } }).collect();
        let mut buf = ComputationBuffers::new();
        buf.resize(fft.as_view().forward_scratch().unaligned_bytes_required().max(fft.as_view().backward_scratch().unaligned_bytes_required()));
        // root exponents of the Fourier values of TFHE-rs: transform of X
        let mut xpoly = vec![0u64; N]; xpoly[1] = 1;
        let mut fx = FourierPolynomial::new(PolynomialSize(N));
        fft.as_view().forward_as_integer(fx.as_mut_view(), Polynomial::from_container(xpoly.as_slice()), buf.stack());
        let ci: Vec<usize> = fx.data.iter().map(|z| {
            let c = (z.im.atan2(z.re) * N as f64 / std::f64::consts::PI).round() as i64;
            c.rem_euclid(two_n as i64) as usize
        }).collect();
        let mut pos = vec![usize::MAX; two_n];
        for (i, &c) in ci.iter().enumerate() { assert!(c % 2 == 1 && pos[c] == usize::MAX, "unexpected Fourier layout"); pos[c] = i; }
        let bcols = two_n / cfg.p; let gamma = 1 + cfg.p;
        let mut aj = vec![1usize; bcols];
        for j in 1..bcols { aj[j] = aj[j - 1] * gamma % two_n; }
        let rj: Vec<usize> = aj.iter().map(|&a| ((a - 1) / cfg.p) % bcols).collect();
        assert!(aj.iter().all(|&a| a % 4 == 1));
        // e-layout
        let h = N / 2; let lrow = h / bcols;
        let mut dlog = vec![(usize::MAX, false); two_n];
        { let mut c5 = 1usize; for e in 0..h { dlog[c5] = (e, false); dlog[two_n - c5] = (e, true); c5 = c5 * 5 % two_n; } }
        let mut eord = vec![u32::MAX; h]; let mut esgn = vec![0f64; h];
        for (i, &c) in ci.iter().enumerate() {
            let (e, neg) = dlog[c]; assert!(e != usize::MAX && eord[e] == u32::MAX, "e-layout");
            eord[e] = i as u32; esgn[e] = if neg { -1.0 } else { 1.0 };
        }
        let (eg, gneg) = dlog[gamma % two_n]; assert!(!gneg && eg % lrow == 0, "γ not in 5^(L Z)");
        let eshift: Vec<usize> = (0..bcols).map(|j| j * eg % h).collect();
        let pow5: Vec<usize> = { let mut v = vec![1usize; lrow]; for k in 1..lrow { v[k] = v[k - 1] * 5 % two_n; } v };
        let ekc_e: Vec<usize> = pow5.iter().map(|&c| bcols * (c % cfg.p)).collect();
        let tq: Vec<Vec<u32>> = rj.iter().map(|&r| pow5.iter().enumerate().map(|(k, &c)| (k * bcols + (bcols - (c * r) % bcols) % bcols) as u32).collect()).collect();
        let mut ksk_e: Vec<Vec<[Vec<f64>; 4]>> = vec![Vec::new()];
        let (gb, gl, goff) = cfg.gks;
        let s: Vec<u64> = big_sk.as_ref().to_vec();
        for j in 1..bcols {
            let a = aj[j];
            let mut ts = vec![0u64; N];
            for i in 0..N { let e = i * a % two_n; if e < N { ts[e] = ts[e].wrapping_add(s[i]); } else { ts[e - N] = ts[e - N].wrapping_sub(s[i]); } }
            let mut rows_e = Vec::new();
            for l in 1..=gl {
                let g = 1u64 << (64 - goff - l as u32 * gb);
                let msg = PlaintextList::from_container(ts.iter().map(|&x| x.wrapping_mul(g)).collect::<Vec<u64>>());
                let mut ct = GlweCiphertext::new(0u64, GlweSize(2), PolynomialSize(N), cm());
                encrypt_glwe_ciphertext(&glwe_sk, &mut ct, &msg, glwe_noise, &mut egen);
                let polys: Vec<Vec<u64>> = ct.as_polynomial_list().iter().map(|p| p.as_ref().to_vec()).collect();
                let mut fa = FourierPolynomial::new(PolynomialSize(N)); let mut fb = FourierPolynomial::new(PolynomialSize(N));
                fft.as_view().forward_as_torus(fa.as_mut_view(), Polynomial::from_container(polys[0].as_slice()), buf.stack());
                fft.as_view().forward_as_torus(fb.as_mut_view(), Polynomial::from_container(polys[1].as_slice()), buf.stack());
                {
                    let sj = eshift[j]; let (mut v0, mut v1, mut v2, mut v3) = (vec![0f64; h], vec![0f64; h], vec![0f64; h], vec![0f64; h]);
                    for e in 0..h {
                        let i = eord[e] as usize; let d = (e + sj) % h;
                        v0[d] = fa.data[i].re; v1[d] = esgn[e] * fa.data[i].im; v2[d] = fb.data[i].re; v3[d] = esgn[e] * fb.data[i].im;
                    }
                    rows_e.push([v0, v1, v2, v3]);
                }
            }
            ksk_e.push(rows_e);
        }
        let (mut ksk_s, mut kpad) = (Vec::new(), Vec::new());
        if m == 1 {
            kpad = vec![0f64; bcols * gl * 4 * (h + PADW)];
            for j in 1..bcols { for l in 0..gl { for c in 0..4 {
                let o = ((j * gl + l) * 4 + c) * (h + PADW); kpad[o..o + h].copy_from_slice(&ksk_e[j][l][c]);
            } } }
            ksk_s.reserve(bcols * (bcols - 1) * gl * 4 * lrow);
            for sb in 0..bcols { for j in 1..bcols {
                let so = sb * lrow;
                for l in 0..gl { for c in 0..4 { ksk_s.extend_from_slice(&ksk_e[j][l][c][so..so + lrow]); } }
            } }
        }
        let mut rng = StdRng::seed_from_u64(seed ^ 0xc0b);
        let stair = StairKey::new(big_sk.as_ref(), N, cfg.chain, &mut rng);
        let pool_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(cfg.n))), 0.0);
        let pool = (0..128).map(|_| {
            let z: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&small_sk, Plaintext(0), pool_noise, cm(), &mut egen);
            (z.get_mask().as_ref().to_vec(), *z.get_body().data)
        }).collect();
        let log_r = cfg.bits + cfg.p.trailing_zeros();
        let vstar_poly: Vec<u64> = (0..N).map(|i| { let j = i / bcols; (j.min(cfg.p / 2 - j) as u64) << (64 - log_r) }).collect();
        let vstar = cfg.vterm * (cfg.p as f64).powi(2) / 16.0;
        CombKeys { bits: cfg.bits, p: cfg.p, bcols, n: cfg.n, log_r, vstar, gks: cfg.gks, big_sk, small_sk, m, bsk, ind, stair, pool,
                   roots, vstar_poly, fft, lrow, eord, esgn, eshift, ksk_e, ksk_s, kpad, tq, ekc_e }
    }

    /// Comb bootstrap of B noise-free LWE ciphertexts modulo p: aints[k][u] = p a_{k,u}, bints[k] = p b_k (in Z_p);
    /// returns the B LWE encryptions of E_k/R under s' (dimension N).
    pub fn combboot(&self, aints: &[Vec<u64>], bints: &[u64]) -> Vec<LweCiphertextOwned<u64>> { self.combboot_prof(aints, bints, None) }

    /// combboot, with the time of each phase added to prof (ms): digits and forward FFTs, table, public products and
    /// key switchings, backward FFTs, CMux. Products in the e-layout: τ_j is a rotation by s_j (a multiple of L) of the
    /// whole vector, and Q_j is a vector of L values repeated B times (all loops contiguous).
    pub fn combboot_prof(&self, aints: &[Vec<u64>], bints: &[u64], prof: Option<&mut [f64; 5]>) -> Vec<LweCiphertextOwned<u64>> {
        if self.m == 2 { return self.combboot2_prof(aints, bints, prof); }
        self.combboot1(aints, self.acc0(bints), prof)
    }

    /// RNS: comb bootstrap of B noise-free LWE ciphertexts modulo p, column k with the message modulus phats[k] (any
    /// integer): the test polynomial of column k is v*_k(j) = (j mods p)/(p p̂_k) (acc0_rns). m = 1 only.
    pub fn combboot_rns(&self, aints: &[Vec<u64>], bints: &[u64], phats: &[u64]) -> Vec<LweCiphertextOwned<u64>> {
        assert!(self.m == 1, "combboot_rns: m = 1");
        self.combboot1(aints, self.acc0_rns(bints, phats), None)
    }

    /// The rotation (m = 1) from the accumulator acc_0, then the extraction of the B columns.
    fn combboot1(&self, aints: &[Vec<u64>], mut c: GlweCiphertextOwned<u64>, mut prof: Option<&mut [f64; 5]>) -> Vec<LweCiphertextOwned<u64>> {
        let (bc, two_n, h, lr) = (self.bcols, 2 * N, N / 2, self.lrow);
        let mask2n = two_n - 1; let sh = bc.trailing_zeros(); let p = self.p;
        let fft = self.fft.as_view();
        let mut buf = ComputationBuffers::new();
        let req = fft.forward_scratch().unaligned_bytes_required().max(fft.backward_scratch().unaligned_bytes_required())
            .max(cmux_assign_mem_optimized_requirement::<u64>(GlweSize(2), PolynomialSize(N), fft).unaligned_bytes_required());
        buf.resize(req);
        let (gb, gl, goff) = self.gks;
        let ggsw: Vec<_> = self.bsk.as_ref().expect("m = 1").as_view().into_ggsw_iter().collect();
        let z = c64 { re: 0.0, im: 0.0 };
        let (eord, esgn) = (&self.eord[..h], &self.esgn[..h]);
        let mut dig = vec![vec![0i64; N]; gl];
        let mut du = vec![0u64; N];
        let mut a0 = vec![0u64; N]; let mut b0 = vec![0u64; N];
        // the arrays of the key switchings (KS_PAD): digits, FFT of C0, accumulators of C_L, Q_j
        let pad = KS_PAD.load(std::sync::atomic::Ordering::Relaxed);
        let hs = if pad { h + PADW } else { h };
        let nw = 2 * gl + 12;
        let mut arena = vec![0f64; if pad { nw * hs } else { 0 }];
        let mut sep = vec![vec![0f64; h]; if pad { 0 } else { nw }];
        let ws: Vec<&mut [f64]> = if pad { arena.chunks_mut(hs).map(|c| &mut c[..h]).collect() }
                                  else { sep.iter_mut().map(|v| &mut v[..]).collect() };
        let mut ws = ws.into_iter();
        let mut dre: Vec<&mut [f64]> = (0..gl).map(|_| ws.next().unwrap()).collect();
        let mut dim: Vec<&mut [f64]> = (0..gl).map(|_| ws.next().unwrap()).collect();
        let (mut fa_re, mut fa_im, mut fb_re, mut fb_im) = (ws.next().unwrap(), ws.next().unwrap(), ws.next().unwrap(), ws.next().unwrap());
        let mut fp = FourierPolynomial::new(PolynomialSize(N));
        let mut tab = vec![z; lr * bc];
        let tw: Vec<c64> = (0..bc).map(|t| self.roots[(p * t) & mask2n]).collect();
        let (mut q_re, mut q_im) = (vec![0f64; lr], vec![0f64; lr]);
        let (qa_re, qa_im) = (ws.next().unwrap(), ws.next().unwrap());
        let mode = match KS_MODE.load(std::sync::atomic::Ordering::Relaxed) { KS_AUTO => if gl <= 2 { 2 } else { 1 }, m => m };
        let kst = gl * 4 * lr;                 // keys of one (source block, automorphism) in ksk_s
        let (ca_re, ca_im, cb_re, cb_im) = (ws.next().unwrap(), ws.next().unwrap(), ws.next().unwrap(), ws.next().unwrap());
        let mut ek = vec![0usize; bc];
        let mut cl = GlweCiphertext::new(0u64, GlweSize(2), PolynomialSize(N), cm());
        let mut tk = std::time::Instant::now();
        let mut lap = |prof: &mut Option<&mut [f64; 5]>, i: usize| {
            if let Some(p) = prof.as_deref_mut() { let t = std::time::Instant::now(); p[i] += (t - tk).as_secs_f64() * 1e3; tk = t; }
        };
        // TFHE-rs order -> e-layout (split), with conjugation
        let to_e = |w: &[c64], re: &mut [f64], im: &mut [f64]| {
            for e in 0..h { let x = w[eord[e] as usize]; re[e] = x.re; im[e] = esgn[e] * x.im; }
        };
        for u in 0..self.n {
            // line 4: C0 = C / B (rounded shift), digits of the shifted gadget
            {
                let pl = c.as_polynomial_list(); let mut it = pl.iter();
                for (dst, src) in [(&mut a0, it.next().unwrap()), (&mut b0, it.next().unwrap())] {
                    for (d, &x) in dst.iter_mut().zip(src.as_ref()) { *d = (x >> sh) + ((x >> (sh - 1)) & 1); }
                }
            }
            if goff > 0 { gadget_lift(gb, gl, goff, &mut a0, &mut dig); } else { decompose(gb, gl, goff, &a0, &mut dig); }
            for l in 0..gl {
                for (d, &x) in du.iter_mut().zip(&dig[l]) { *d = x as u64; }
                fft.forward_as_integer(fp.as_mut_view(), Polynomial::from_container(du.as_slice()), buf.stack());
                to_e(&fp.data, &mut dre[l], &mut dim[l]);
            }
            fft.forward_as_torus(fp.as_mut_view(), Polynomial::from_container(a0.as_slice()), buf.stack());
            to_e(&fp.data, &mut fa_re, &mut fa_im);
            fft.forward_as_torus(fp.as_mut_view(), Polynomial::from_container(b0.as_slice()), buf.stack());
            to_e(&fp.data, &mut fb_re, &mut fb_im);
            lap(&mut prof, 0);
            // line 3: half table T[κ][t] = Σ_k ψ^{M0 (5^κ mod p) a_k} ω^{t k}: one DFT of size B per κ < L
            for k in 0..bc { ek[k] = aints[k][u] as usize; }
            for (row, &mc) in self.ekc_e.iter().enumerate() {
                let x = &mut tab[row * bc..(row + 1) * bc];
                for k in 0..bc { x[k] = self.roots[(mc * ek[k]) & mask2n]; }
                dft_pow2(x, &tw);
            }
            lap(&mut prof, 1);
            // lines 5-10 in the e-layout: C_L = Q_0 C0 + Σ_{j≥1} Q_j τ_j(KS'_j), KS'_j computed in the source space of τ_j
            if mode >= 1 {
                // source block by source block: the block so feeds the destination blocks so - s_j (mod N/2)
                for j in 0..bc {
                    for (k, &t) in self.tq[j].iter().enumerate() { let q = tab[t as usize]; qa_re[j * lr + k] = q.re; qa_im[j * lr + k] = q.im; }
                }
                {
                    let (qr, qi) = (&qa_re[..lr], &qa_im[..lr]);
                    for o in (0..h).step_by(lr) {
                        let (ar, ai, br, bi) = (&fa_re[o..o + lr], &fa_im[o..o + lr], &fb_re[o..o + lr], &fb_im[o..o + lr]);
                        let (xr, xi, yr, yi) = (&mut ca_re[o..o + lr], &mut ca_im[o..o + lr], &mut cb_re[o..o + lr], &mut cb_im[o..o + lr]);
                        for k in 0..lr {
                            xr[k] = qr[k] * ar[k] - qi[k] * ai[k]; xi[k] = qr[k] * ai[k] + qi[k] * ar[k];
                            yr[k] = qr[k] * br[k] - qi[k] * bi[k]; yi[k] = qr[k] * bi[k] + qi[k] * br[k];
                        }
                    }
                }
                for sb in 0..bc {
                    let so = sb * lr;
                    let mut dd: [[&[f64]; 2]; 6] = [[&[], &[]]; 6];
                    for l in 0..gl { dd[l] = [&dre[l][so..so + lr], &dim[l][so..so + lr]]; }
                    if mode >= 2 {
                        // the digits and the FFT of b of the block in registers for the B - 1 automorphisms
                        let mut dst = [0usize; 32];
                        for j in 1..bc { dst[j - 1] = (so + h - self.eshift[j]) % h; }
                        ks_src(ca_re, ca_im, cb_re, cb_im, qa_re, qa_im, &fb_re[so..so + lr], &fb_im[so..so + lr], &dd[..gl],
                               &self.ksk_s[sb * (bc - 1) * kst..(sb + 1) * (bc - 1) * kst], &dst[..bc - 1], lr, [4, 2, 8][mode - 2]);
                        continue;
                    }
                    for j in 1..bc {
                        let o = (so + h - self.eshift[j]) % h;
                        let kb = &self.ksk_s[(sb * (bc - 1) + j - 1) * kst..][..kst];
                        let mut kk: [[&[f64]; 4]; 6] = [[&[], &[], &[], &[]]; 6];
                        for l in 0..gl { let b = &kb[l * 4 * lr..(l + 1) * 4 * lr]; kk[l] = [&b[..lr], &b[lr..2 * lr], &b[2 * lr..3 * lr], &b[3 * lr..]]; }
                        ks_block(&mut ca_re[o..o + lr], &mut ca_im[o..o + lr], &mut cb_re[o..o + lr], &mut cb_im[o..o + lr],
                                 &qa_re[j * lr..(j + 1) * lr], &qa_im[j * lr..(j + 1) * lr], &fb_re[so..so + lr], &fb_im[so..so + lr], &dd[..gl], &kk[..gl]);
                    }
                }
            } else {
            for j in 0..bc {
                for (k, &t) in self.tq[j].iter().enumerate() { let q = tab[t as usize]; q_re[k] = q.re; q_im[k] = q.im; }
                let (qr, qi) = (&q_re[..lr], &q_im[..lr]);
                if j == 0 {
                    for m in 0..bc {
                        let o = m * lr;
                        let (ar, ai, br, bi) = (&fa_re[o..o + lr], &fa_im[o..o + lr], &fb_re[o..o + lr], &fb_im[o..o + lr]);
                        let (xr, xi, yr, yi) = (&mut ca_re[o..o + lr], &mut ca_im[o..o + lr], &mut cb_re[o..o + lr], &mut cb_im[o..o + lr]);
                        for k in 0..lr {
                            xr[k] = qr[k] * ar[k] - qi[k] * ai[k]; xi[k] = qr[k] * ai[k] + qi[k] * ar[k];
                            yr[k] = qr[k] * br[k] - qi[k] * bi[k]; yi[k] = qr[k] * bi[k] + qi[k] * br[k];
                        }
                    }
                    continue;
                }
                // destination block m (L values) reads the source block m + s_j (mod B): the key switching of that
                // source block is accumulated in L1 (t, w), then multiplied by Q_j and added (one pass over keys and digits)
                let rows: [[&[f64]; 4]; 6] = std::array::from_fn(|l| if l >= gl { [&[][..]; 4] } else if pad {
                    std::array::from_fn(|c| &self.kpad[((j * gl + l) * 4 + c) * hs..][..h])
                } else { std::array::from_fn(|c| &self.ksk_e[j][l][c][..]) });
                let sj = self.eshift[j];
                for m in 0..bc {
                    let o = m * lr; let so = (o + sj) % h;
                    let mut dd: [[&[f64]; 2]; 6] = [[&[], &[]]; 6]; let mut kk: [[&[f64]; 4]; 6] = [[&[], &[], &[], &[]]; 6];
                    for l in 0..gl {
                        let [ar, ai, br, bi] = &rows[l];
                        kk[l] = [&ar[so..so + lr], &ai[so..so + lr], &br[so..so + lr], &bi[so..so + lr]];
                        dd[l] = [&dre[l][so..so + lr], &dim[l][so..so + lr]];
                    }
                    let (xr, xi, yr, yi) = (&mut ca_re[o..o + lr], &mut ca_im[o..o + lr], &mut cb_re[o..o + lr], &mut cb_im[o..o + lr]);
                    ks_block(xr, xi, yr, yi, qr, qi, &fb_re[so..so + lr], &fb_im[so..so + lr], &dd[..gl], &kk[..gl]);
                }
            }
            }
            lap(&mut prof, 2);
            {
                let mut pl = cl.as_mut_polynomial_list();
                let mut it = pl.iter_mut();
                let mut pa = it.next().unwrap();
                for e in 0..h { fp.data[eord[e] as usize] = c64 { re: ca_re[e], im: esgn[e] * ca_im[e] }; }
                fft.backward_as_torus(pa.as_mut_view(), fp.as_view(), buf.stack());
                let mut pb = it.next().unwrap();
                for e in 0..h { fp.data[eord[e] as usize] = c64 { re: cb_re[e], im: esgn[e] * cb_im[e] }; }
                fft.backward_as_torus(pb.as_mut_view(), fp.as_view(), buf.stack());
            }
            lap(&mut prof, 3);
            // line 11: C = CMux(RGSW(s_u), C, C_L)
            cmux_assign_mem_optimized(&mut c, &mut cl, &ggsw[u], fft, buf.stack());
            lap(&mut prof, 4);
        }
        self.extract(&c)
    }

    /// Comb bootstrap with the mask grouped by m = 2 (Algorithm 1): per pair (u, u + 1), the division by B, the digits,
    /// their FFTs and the key switchings of the B - 1 automorphisms are shared by the three patterns P = 10, 01, 11
    /// (exponents a_u, a_{u+1}, a_u + a_{u+1}); per pattern: the half table, the public products (Q_j^P times the
    /// switched blocks), the backward FFTs of C_L^P and the external product Dec(C_L^P - C)·GGSW(ind_P), the three
    /// accumulated in the Fourier domain (2 backward FFTs per pair). An odd last coordinate is a CMux step (m = 1).
    /// prof (ms): digits and forward FFTs, tables, key switchings and public products, backward FFTs of the C_L^P,
    /// external products.
    fn combboot2_prof(&self, aints: &[Vec<u64>], bints: &[u64], mut prof: Option<&mut [f64; 5]>) -> Vec<LweCiphertextOwned<u64>> {
        use crate::kern::{ks_block3, ks_tw, q_acc_pub};
        let (bc, two_n, h, lr) = (self.bcols, 2 * N, N / 2, self.lrow);
        let mask2n = two_n - 1; let sh = bc.trailing_zeros(); let p = self.p;
        let fft = self.fft.as_view();
        let mut buf = ComputationBuffers::new();
        buf.resize(fft.forward_scratch().unaligned_bytes_required().max(fft.backward_scratch().unaligned_bytes_required()));
        let mut c = self.acc0(bints);
        let (gb, gl, goff) = self.gks;
        let ind = self.ind.as_ref().expect("m = 2");
        let ggsw: Vec<&[c64]> = ind.as_view().into_ggsw_iter().map(|g| g.data()).collect();
        let (bbr, lbr) = (ind.decomposition_base_log().0 as u32, ind.decomposition_level_count().0);
        let z = c64 { re: 0.0, im: 0.0 };
        let (eord, esgn) = (&self.eord[..h], &self.esgn[..h]);
        let mut dig = vec![vec![0i64; N]; gl];
        let mut du = vec![0u64; N];
        let mut a0 = vec![0u64; N]; let mut b0 = vec![0u64; N];
        let (mut dre, mut dim) = (vec![vec![0f64; h]; gl], vec![vec![0f64; h]; gl]);
        let (mut fa_re, mut fa_im, mut fb_re, mut fb_im) = (vec![0f64; h], vec![0f64; h], vec![0f64; h], vec![0f64; h]);
        let mut fp = FourierPolynomial::new(PolynomialSize(N));
        let tw: Vec<c64> = (0..bc).map(|t| self.roots[(p * t) & mask2n]).collect();
        let mut tab = vec![vec![z; lr * bc]; 3];
        let mut qv = vec![(vec![0f64; lr], vec![0f64; lr]); 3];
        let mut acc: Vec<[Vec<f64>; 4]> = (0..3).map(|_| [vec![0f64; h], vec![0f64; h], vec![0f64; h], vec![0f64; h]]).collect();
        let mut ek = vec![0usize; bc];
        let (mut tr, mut ti, mut wr, mut wi) = (vec![0f64; lr], vec![0f64; lr], vec![0f64; lr], vec![0f64; lr]);
        let mut cl = vec![0u64; 2 * N];
        let (mut ed, mut fd, mut st) = (vec![0u64; 2 * lbr * N], vec![z; 2 * lbr * h], vec![0u64; N]);
        let (mut f0, mut f1) = (vec![z; h], vec![z; h]);
        let arch = pulp::Arch::new();
        let rep = bbr * lbr as u32; let nr = 64 - rep; let rmask = u64::MAX >> (64 - rep); let mb = (1u64 << bbr) - 1;
        let mut tk = std::time::Instant::now();
        let mut lap = |prof: &mut Option<&mut [f64; 5]>, i: usize| {
            if let Some(p) = prof.as_deref_mut() { let t = std::time::Instant::now(); p[i] += (t - tk).as_secs_f64() * 1e3; tk = t; }
        };
        let to_e = |w: &[c64], re: &mut [f64], im: &mut [f64]| {
            for e in 0..h { let x = w[eord[e] as usize]; re[e] = x.re; im[e] = esgn[e] * x.im; }
        };
        let npairs = self.n / 2;
        for g in 0..npairs + self.n % 2 {
            let single = g == npairs; let u = 2 * g;
            // shared: C0 = C / B (rounded shift), digits of the shifted gadget, forward FFTs
            {
                let pl = c.as_polynomial_list(); let mut it = pl.iter();
                for (dst, src) in [(&mut a0, it.next().unwrap()), (&mut b0, it.next().unwrap())] {
                    for (d, &x) in dst.iter_mut().zip(src.as_ref()) { *d = (x >> sh) + ((x >> (sh - 1)) & 1); }
                }
            }
            if goff > 0 { gadget_lift(gb, gl, goff, &mut a0, &mut dig); } else { decompose(gb, gl, goff, &a0, &mut dig); }
            for l in 0..gl {
                for (d, &x) in du.iter_mut().zip(&dig[l]) { *d = x as u64; }
                fft.forward_as_integer(fp.as_mut_view(), Polynomial::from_container(du.as_slice()), buf.stack());
                to_e(&fp.data, &mut dre[l], &mut dim[l]);
            }
            fft.forward_as_torus(fp.as_mut_view(), Polynomial::from_container(a0.as_slice()), buf.stack());
            to_e(&fp.data, &mut fa_re, &mut fa_im);
            fft.forward_as_torus(fp.as_mut_view(), Polynomial::from_container(b0.as_slice()), buf.stack());
            to_e(&fp.data, &mut fb_re, &mut fb_im);
            lap(&mut prof, 0);
            // half tables of the patterns
            let np = if single { 1 } else { 3 };
            for pp in 0..np {
                for k in 0..bc {
                    let x0 = aints[k][u] as usize; let x1 = if single { 0 } else { aints[k][u + 1] as usize };
                    ek[k] = match pp { 0 => x0, 1 => x1, _ => x0 + x1 };
                }
                for (row, &mc) in self.ekc_e.iter().enumerate() {
                    let x = &mut tab[pp][row * bc..(row + 1) * bc];
                    for k in 0..bc { x[k] = self.roots[(mc * ek[k]) & mask2n]; }
                    dft_pow2(x, &tw);
                }
            }
            lap(&mut prof, 1);
            // key switchings of the B - 1 automorphisms (shared) and public products (per pattern), in the e-layout
            for j in 0..bc {
                for pp in 0..np {
                    for (k, &t) in self.tq[j].iter().enumerate() { let v = tab[pp][t as usize]; qv[pp].0[k] = v.re; qv[pp].1[k] = v.im; }
                }
                if j == 0 {
                    for pp in 0..np {
                        let (qr, qi) = (&qv[pp].0, &qv[pp].1);
                        let [xr, xi, yr, yi] = &mut acc[pp];
                        for m in 0..bc {
                            let o = m * lr;
                            for k in 0..lr {
                                let e = o + k;
                                xr[e] = qr[k] * fa_re[e] - qi[k] * fa_im[e]; xi[e] = qr[k] * fa_im[e] + qi[k] * fa_re[e];
                                yr[e] = qr[k] * fb_re[e] - qi[k] * fb_im[e]; yi[e] = qr[k] * fb_im[e] + qi[k] * fb_re[e];
                            }
                        }
                    }
                    continue;
                }
                let rows = &self.ksk_e[j]; let sj = self.eshift[j];
                for m in 0..bc {
                    let o = m * lr; let so = (o + sj) % h;
                    let mut dd: [[&[f64]; 2]; 6] = [[&[], &[]]; 6]; let mut kk: [[&[f64]; 4]; 6] = [[&[], &[], &[], &[]]; 6];
                    for l in 0..gl {
                        let [ar, ai, br, bi] = &rows[l];
                        kk[l] = [&ar[so..so + lr], &ai[so..so + lr], &br[so..so + lr], &bi[so..so + lr]];
                        dd[l] = [&dre[l][so..so + lr], &dim[l][so..so + lr]];
                    }
                    if np == 3 {
                        let (a0, rest) = acc.split_at_mut(1); let (a1, a2) = rest.split_at_mut(1);
                        fn blk(a: &mut [Vec<f64>; 4], o: usize, lr: usize) -> [&mut [f64]; 4] {
                            let [xr, xi, yr, yi] = a;
                            [&mut xr[o..o + lr], &mut xi[o..o + lr], &mut yr[o..o + lr], &mut yi[o..o + lr]]
                        }
                        ks_block3([blk(&mut a0[0], o, lr), blk(&mut a1[0], o, lr), blk(&mut a2[0], o, lr)],
                                  [[&qv[0].0, &qv[0].1], [&qv[1].0, &qv[1].1], [&qv[2].0, &qv[2].1]],
                                  &fb_re[so..so + lr], &fb_im[so..so + lr], &dd[..gl], &kk[..gl]);
                    } else {
                        ks_tw(&mut tr, &mut ti, &mut wr, &mut wi, &fb_re[so..so + lr], &fb_im[so..so + lr], &dd[..gl], &kk[..gl]);
                        let [xr, xi, yr, yi] = &mut acc[0];
                        q_acc_pub(&mut xr[o..o + lr], &mut xi[o..o + lr], &mut yr[o..o + lr], &mut yi[o..o + lr],
                                  &qv[0].0, &qv[0].1, &tr, &ti, &wr, &wi);
                    }
                }
            }
            lap(&mut prof, 2);
            // per pattern: C_L^P (backward FFTs), digits of C_L^P - C, forward FFTs, products by GGSW(ind_P), accumulated
            f0.fill(z); f1.fill(z);
            for pp in 0..np {
                let [xr, xi, yr, yi] = &acc[pp];
                for e in 0..h { fp.data[eord[e] as usize] = c64 { re: xr[e], im: esgn[e] * xi[e] }; }
                fft.backward_as_torus(Polynomial::from_container(&mut cl[..N]), fp.as_view(), buf.stack());
                for e in 0..h { fp.data[eord[e] as usize] = c64 { re: yr[e], im: esgn[e] * yi[e] }; }
                fft.backward_as_torus(Polynomial::from_container(&mut cl[N..]), fp.as_view(), buf.stack());
                lap(&mut prof, 3);
                {
                    let cc: &[u64] = c.as_ref();
                    for r in 0..2 {
                        for i in 0..N {
                            let x = cl[r * N + i].wrapping_sub(cc[r * N + i]);
                            let mut res = x >> (nr - 1); let rb = res & 1; res = ((res + 1) >> 1) & rmask;
                            let nb = ((res.wrapping_sub(1) | (rb << (rep - 1))) & res) >> (rep - 1);
                            st[i] = res.wrapping_sub(nb << rep);
                        }
                        for s in 0..lbr {
                            let out = &mut ed[(r * lbr + s) * N..(r * lbr + s + 1) * N];
                            for i in 0..N {
                                let v = st[i]; let rr = v & mb; let v = ((v as i64) >> bbr) as u64;
                                let carry = ((rr.wrapping_sub(1) | v) & rr) >> (bbr - 1);
                                st[i] = v.wrapping_add(carry); out[i] = rr.wrapping_sub(carry << bbr);
                            }
                        }
                    }
                }
                for k in 0..2 * lbr {
                    fft.forward_as_integer(FourierPolynomial { data: &mut fd[k * h..(k + 1) * h] },
                                           Polynomial::from_container(&ed[k * N..(k + 1) * N]), buf.stack());
                }
                let gk = ggsw[if single { 3 * npairs } else { 3 * g + pp }];
                for s in 0..lbr { for r in 0..2 {
                    let dk = &fd[(r * lbr + s) * h..(r * lbr + s + 1) * h];
                    let g0 = &gk[((s * 2 + r) * 2) * h..((s * 2 + r) * 2 + 1) * h];
                    let g1 = &gk[((s * 2 + r) * 2 + 1) * h..((s * 2 + r) * 2 + 2) * h];
                    arch.dispatch(crate::mt::Mac2 { o0: &mut f0, o1: &mut f1, d: dk, g0, g1 });
                } }
                lap(&mut prof, 4);
            }
            {
                let cm_: &mut [u64] = c.as_mut(); let (a1, b1) = cm_.split_at_mut(N);
                fft.add_backward_as_torus(Polynomial::from_container(a1), FourierPolynomial { data: &f0[..] }, buf.stack());
                fft.add_backward_as_torus(Polynomial::from_container(b1), FourierPolynomial { data: &f1[..] }, buf.stack());
            }
            lap(&mut prof, 4);
        }
        self.extract(&c)
    }

    /// combboot on t threads (workers alive for the whole bootstrap, spin barriers). Per iteration: digits and rounded
    /// division (coefficients split) | forward FFTs (gl + 2 jobs) and half table (rows split) | key switchings and
    /// public products (destination blocks split: no reduction) | backward FFTs of C_L (2 jobs) | parallel CMux
    /// (digits, 2ℓ FFTs, products, 2 backward FFTs). Same arithmetic as combboot except the CMux (ours, with the keys
    /// and the decomposition of TFHE-rs).
    pub fn combboot_mt(&self, aints: &[Vec<u64>], bints: &[u64], threads: usize) -> Vec<LweCiphertextOwned<u64>> {
        use crate::mt::{ext_prod_mt, prof_on, share, ExtProdBufs, Lap, Sh, SpinBarrier};
        let t = threads.max(1);
        let (bc, two_n, h, lr) = (self.bcols, 2 * N, N / 2, self.lrow);
        let mask2n = two_n - 1; let sh = bc.trailing_zeros(); let p = self.p;
        let (gb, gl, goff) = self.gks;
        let bsk = self.bsk.as_ref().expect("combboot_mt: m = 1 only (keys from CombKeys::new_m(cfg, 1, seed))");
        let (bbr, lbr) = (bsk.decomposition_base_log().0 as u32, bsk.decomposition_level_count().0);
        let z = c64 { re: 0.0, im: 0.0 };
        let acc = self.acc0(bints);
        let mut c_ab: Vec<u64> = acc.as_ref().to_vec();
        let mut a0 = vec![0u64; 2 * N];
        let mut dig = vec![0u64; gl * N];
        let mut fe = vec![0f64; (gl + 2) * 2 * h];
        let mut tab = vec![z; lr * bc];
        let mut fcl = vec![z; 2 * h];
        let mut cl = vec![0u64; 2 * N];
        let (mut ed, mut fd, mut fo) = (vec![0u64; 2 * lbr * N], vec![z; 2 * lbr * h], vec![z; 2 * h]);
        let ggsw: Vec<&[c64]> = bsk.as_view().into_ggsw_iter().map(|g| g.data()).collect();
        let (pc, pa0, pdig, pfe, ptab, pfcl, pcl) = (Sh::of(&mut c_ab), Sh::of(&mut a0), Sh::of(&mut dig), Sh::of(&mut fe),
                                                    Sh::of(&mut tab), Sh::of(&mut fcl), Sh::of(&mut cl));
        let bufs = ExtProdBufs { ed: Sh::of(&mut ed), fd: Sh::of(&mut fd), fo: Sh::of(&mut fo) };
        let bar = SpinBarrier::new(t);
        let tw: Vec<c64> = (0..bc).map(|k| self.roots[(p * k) & mask2n]).collect();
        let shift = 64 - goff - gb * gl as u32;
        let (gmask, ghalf) = ((1u64 << gb) - 1, 1u64 << (gb - 1));
        let offs: u64 = (0..gl).fold(0u64, |a, j| a.wrapping_add(ghalf << (j as u32 * gb)));
        let nfft = gl + 2;
        let worker = |w: usize| {
            let fft = self.fft.as_view();
            let mut buf = ComputationBuffers::new();
            buf.resize(fft.forward_scratch().unaligned_bytes_required().max(fft.backward_scratch().unaligned_bytes_required()));
            let mut fp = FourierPolynomial::new(PolynomialSize(N));
            let (mb0, mb1) = share(bc, t, w); let nbw = mb1 - mb0;
            let mut accb = vec![0f64; 4 * nbw * lr];
            let (mut qr, mut qi) = (vec![0f64; lr], vec![0f64; lr]);
            let (c0, c1) = share(N, t, w);
            let (r0, r1) = if t > nfft { share(lr, t - nfft, w.saturating_sub(nfft)) } else { share(lr, t, w) };
            let (r0, r1) = if t > nfft && w < nfft { (0, 0) } else { (r0, r1) };
            let mut ek = vec![0usize; bc];
            let (mut ybuf, mut recb) = (vec![0u64; c1 - c0], vec![0u64; c1 - c0]);
            let mut lap = Lap::new(w == 0 && prof_on());
            for u in 0..self.n {
                // A: rounded division by B, digits of the shifted gadget (representative lifted if off > 0)
                unsafe {
                    let m = c1 - c0;
                    let (ca, cb) = (pc.s(c0, m), pc.s(N + c0, m)); let (xa, xb) = (pa0.s(c0, m), pa0.s(N + c0, m));
                    for i in 0..m {
                        let x = (ca[i] >> sh) + ((ca[i] >> (sh - 1)) & 1);
                        xb[i] = (cb[i] >> sh) + ((cb[i] >> (sh - 1)) & 1);
                        xa[i] = x; ybuf[i] = (x >> shift) + ((x >> (shift - 1)) & 1);
                    }
                    if goff > 0 { recb[..m].fill(0); }
                    for j in 0..gl {
                        let dj = pdig.s(j * N + c0, m); let sj = (gl - 1 - j) as u32 * gb;
                        let gj = 1u64 << (64 - goff - (j as u32 + 1) * gb);
                        for i in 0..m { dj[i] = ((((ybuf[i].wrapping_add(offs)) >> sj) & gmask) as i64 - ghalf as i64) as u64; }
                        if goff > 0 { for i in 0..m { recb[i] = recb[i].wrapping_add(dj[i].wrapping_mul(gj)); } }
                    }
                    if goff > 0 { for i in 0..m { xa[i] = xa[i].wrapping_sub(ybuf[i] << shift).wrapping_add(recb[i]); } }
                }
                bar.wait(); lap.lap(0);
                // B: forward FFTs into the e-layout; half table
                for k in (w..nfft).step_by(t) {
                    let src: &[u64] = unsafe { if k < gl { pdig.s(k * N, N) } else { pa0.s((k - gl) * N, N) } };
                    if k < gl { fft.forward_as_integer(fp.as_mut_view(), Polynomial::from_container(src), buf.stack()); }
                    else { fft.forward_as_torus(fp.as_mut_view(), Polynomial::from_container(src), buf.stack()); }
                    let (re, im) = unsafe { (pfe.s(2 * k * h, h), pfe.s((2 * k + 1) * h, h)) };
                    for e in 0..h { let x = fp.data[self.eord[e] as usize]; re[e] = x.re; im[e] = self.esgn[e] * x.im; }
                }
                if r1 > r0 {
                    for k in 0..bc { ek[k] = aints[k][u] as usize; }
                    for row in r0..r1 {
                        let x = unsafe { ptab.s(row * bc, bc) }; let mc = self.ekc_e[row];
                        for k in 0..bc { x[k] = self.roots[(mc * ek[k]) & mask2n]; }
                        dft_pow2(x, &tw);
                    }
                }
                bar.wait(); lap.lap(1);
                // C: key switchings and public products on the destination blocks mb0..mb1, then to the order of TFHE-rs
                if nbw > 0 {
                    let tabr: &[c64] = unsafe { ptab.s(0, lr * bc) };
                    let fer: &[f64] = unsafe { pfe.s(0, (gl + 2) * 2 * h) };
                    for j in 0..bc {
                        for (k, &ix) in self.tq[j].iter().enumerate() { let q = tabr[ix as usize]; qr[k] = q.re; qi[k] = q.im; }
                        let sj = self.eshift[j];
                        let (qr, qi) = (&qr[..lr], &qi[..lr]);
                        for bi in 0..nbw {
                            let o = (mb0 + bi) * lr;
                            let blk = &mut accb[4 * bi * lr..4 * (bi + 1) * lr];
                            let (xr, rest) = blk.split_at_mut(lr); let (xi, rest) = rest.split_at_mut(lr); let (yr, yi) = rest.split_at_mut(lr);
                            if j == 0 {
                                let (ar, ai) = (&fer[2 * gl * h + o..2 * gl * h + o + lr], &fer[(2 * gl + 1) * h + o..(2 * gl + 1) * h + o + lr]);
                                let (br, bi_) = (&fer[(2 * gl + 2) * h + o..(2 * gl + 2) * h + o + lr], &fer[(2 * gl + 3) * h + o..(2 * gl + 3) * h + o + lr]);
                                for k in 0..lr {
                                    xr[k] = qr[k] * ar[k] - qi[k] * ai[k]; xi[k] = qr[k] * ai[k] + qi[k] * ar[k];
                                    yr[k] = qr[k] * br[k] - qi[k] * bi_[k]; yi[k] = qr[k] * bi_[k] + qi[k] * br[k];
                                }
                                continue;
                            }
                            let so = (o + sj) % h;
                            let rows = &self.ksk_e[j];
                            let mut dd: [[&[f64]; 2]; 6] = [[&[], &[]]; 6]; let mut kk: [[&[f64]; 4]; 6] = [[&[], &[], &[], &[]]; 6];
                            for l in 0..gl {
                                let [ar, ai, br, bi_] = &rows[l];
                                kk[l] = [&ar[so..so + lr], &ai[so..so + lr], &br[so..so + lr], &bi_[so..so + lr]];
                                dd[l] = [&fer[2 * l * h + so..2 * l * h + so + lr], &fer[(2 * l + 1) * h + so..(2 * l + 1) * h + so + lr]];
                            }
                            ks_block(xr, xi, yr, yi, qr, qi, &fer[(2 * gl + 2) * h + so..(2 * gl + 2) * h + so + lr],
                                     &fer[(2 * gl + 3) * h + so..(2 * gl + 3) * h + so + lr], &dd[..gl], &kk[..gl]);
                        }
                    }
                    for bi in 0..nbw {
                        let o = (mb0 + bi) * lr; let blk = &accb[4 * bi * lr..4 * (bi + 1) * lr];
                        for k in 0..lr {
                            let e = o + k; let i = self.eord[e] as usize; let sg = self.esgn[e];
                            unsafe {
                                *pfcl.0.add(i) = c64 { re: blk[k], im: sg * blk[lr + k] };
                                *pfcl.0.add(h + i) = c64 { re: blk[2 * lr + k], im: sg * blk[3 * lr + k] };
                            }
                        }
                    }
                }
                bar.wait(); lap.lap(2);
                // D: C_L back to the standard domain
                for k in (w..2).step_by(t) {
                    let (dst, src) = unsafe { (pcl.s(k * N, N), pfcl.s(k * h, h)) };
                    fft.backward_as_torus(Polynomial::from_container(dst), FourierPolynomial { data: &*src }, buf.stack());
                }
                bar.wait(); lap.lap(3);
                // line 11: C = CMux(RGSW(s_u), C, C_L) = C + ExtProd(RGSW(s_u), C_L - C)
                ext_prod_mt(w, t, N, pcl, pc, ggsw[u], bbr, lbr, &bufs, fft, &mut buf, &bar, &mut lap, None);
            }
        };
        std::thread::scope(|sc| {
            for w in 1..t { let wk = &worker; sc.spawn(move || wk(w)); }
            worker(0);
        });
        let c = GlweCiphertext::from_container(c_ab, PolynomialSize(N), cm());
        self.extract(&c)
    }

    /// Line 1: trivial encryption of acc_0 = Σ_k X^{-M0 b_k} π_k(v*).
    pub fn acc0(&self, bints: &[u64]) -> GlweCiphertextOwned<u64> {
        let (bc, two_n) = (self.bcols, 2 * N);
        let mut c = GlweCiphertext::new(0u64, GlweSize(2), PolynomialSize(N), cm());
        {
            let mut body = c.get_mut_body(); let bd = body.as_mut();
            for k in 0..bc {
                let sft = (two_n - (bc * bints[k] as usize) % two_n) % two_n;
                let mut i = k;
                while i < N {
                    let e = (i + sft) % two_n;
                    if e < N { bd[e] = bd[e].wrapping_add(self.vstar_poly[i]); } else { bd[e - N] = bd[e - N].wrapping_sub(self.vstar_poly[i]); }
                    i += bc;
                }
            }
        }
        c
    }

    /// RNS: acc_0 = Σ_k X^{-M0 b_k} π_k(v*_k), the test polynomial of column k for the message modulus phats[k]:
    /// v*_k at the position i = k + B j is round(2^64 min(j, p/2 - j) / (p p̂_k)) (for p̂_k = 2^bits: vstar_poly).
    pub fn acc0_rns(&self, bints: &[u64], phats: &[u64]) -> GlweCiphertextOwned<u64> {
        assert_eq!(phats.len(), self.bcols);
        let half = self.p / 2;
        let vk: Vec<Vec<u64>> = phats.iter().map(|&ph| {
            let r = self.p as u128 * ph as u128;
            (0..half).map(|j| ((((j.min(half - j)) as u128) << 64) + r / 2) / r).map(|v| v as u64).collect()
        }).collect();
        let (bc, two_n) = (self.bcols, 2 * N);
        let mut c = GlweCiphertext::new(0u64, GlweSize(2), PolynomialSize(N), cm());
        {
            let mut body = c.get_mut_body(); let bd = body.as_mut();
            for k in 0..bc {
                let sft = (two_n - (bc * bints[k] as usize) % two_n) % two_n;
                let mut i = k;
                while i < N {
                    let e = (i + sft) % two_n;
                    if e < N { bd[e] = bd[e].wrapping_add(vk[k][i / bc]); } else { bd[e - N] = bd[e - N].wrapping_sub(vk[k][i / bc]); }
                    i += bc;
                }
            }
        }
        c
    }

    /// Lines 12-13: the B columns.
    pub fn extract(&self, c: &GlweCiphertextOwned<u64>) -> Vec<LweCiphertextOwned<u64>> {
        (0..self.bcols).map(|k| {
            let mut out = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            extract_lwe_sample_from_glwe_ciphertext(c, &mut out, MonomialDegree(k));
            out
        }).collect()
    }

    /// refresh with t threads: key switches and roundings split by ciphertexts, combboot_mt, subtractions.
    pub fn refresh_mt(&self, cts: &[LweCiphertextOwned<u64>], seed: u64, threads: usize) -> Vec<LweCiphertextOwned<u64>> {
        let t = threads.max(1); let bc = cts.len();
        let mut rs: Vec<Option<Rounded>> = (0..bc).map(|_| None).collect();
        std::thread::scope(|sc| {
            let chunk = bc.div_ceil(t);
            for (w, (cc, rr)) in cts.chunks(chunk).zip(rs.chunks_mut(chunk)).enumerate() {
                sc.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed ^ (w as u64).wrapping_mul(0x9e3779b97f4a7c15)); let mut rbit = rng.gen::<u64>();
                    for (ct, r) in cc.iter().zip(rr.iter_mut()) {
                        let (a, b) = self.stair.switch(ct.get_mask().as_ref(), *ct.get_body().data, &mut rbit);
                        *r = Some(round_select(&a, b, self.log_r, &self.pool, self.vstar, &mut rng));
                    }
                });
            }
        });
        let rs: Vec<Rounded> = rs.into_iter().map(|r| r.unwrap()).collect();
        let pm = self.p as u64 - 1;
        let aints: Vec<Vec<u64>> = rs.iter().map(|r| r.abar.iter().map(|&x| x & pm).collect()).collect();
        let bints: Vec<u64> = rs.iter().map(|r| r.bbar & pm).collect();
        let errs = self.combboot_mt(&aints, &bints, t);
        let shift = 64 - self.log_r;
        rs.iter().zip(errs.iter()).map(|(r, e)| {
            let mut fin = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            { let (mut fm, fb) = fin.get_mut_mask_and_body();
              for (u, &x) in r.abar.iter().enumerate() { fm.as_mut()[u] = x << shift; }
              *fb.data = r.bbar << shift; }
            lwe_ciphertext_sub_assign(&mut fin, e);
            fin
        }).collect()
    }

    /// Whole refresh of B LWE ciphertexts of dimension N under s' (stair key switch, rounding with selection, comb
    /// bootstrap, subtraction); returns the refreshed ciphertexts and the time of each phase (ms).
    pub fn refresh<RG: Rng>(&self, cts: &[LweCiphertextOwned<u64>], rng: &mut RG, rbit: &mut u64) -> (Vec<LweCiphertextOwned<u64>>, [f64; 3]) {
        let t0 = std::time::Instant::now();
        let sw: Vec<(Vec<u64>, u64)> = cts.iter().map(|ct| self.stair.switch(ct.get_mask().as_ref(), *ct.get_body().data, rbit)).collect();
        let t1 = std::time::Instant::now();
        let rs: Vec<Rounded> = sw.iter().map(|(a, b)| round_select(a, *b, self.log_r, &self.pool, self.vstar, rng)).collect();
        let t2 = std::time::Instant::now();
        let pm = self.p as u64 - 1;
        let aints: Vec<Vec<u64>> = rs.iter().map(|r| r.abar.iter().map(|&x| x & pm).collect()).collect();
        let bints: Vec<u64> = rs.iter().map(|r| r.bbar & pm).collect();
        let errs = self.combboot(&aints, &bints);
        let shift = 64 - self.log_r;
        let outs = rs.iter().zip(errs.iter()).map(|(r, e)| {
            let mut fin = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            { let (mut fm, fb) = fin.get_mut_mask_and_body();
              for (u, &x) in r.abar.iter().enumerate() { fm.as_mut()[u] = x << shift; }
              *fb.data = r.bbar << shift; }
            lwe_ciphertext_sub_assign(&mut fin, e);
            fin
        }).collect();
        let t3 = std::time::Instant::now();
        (outs, [(t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, (t3 - t2).as_secs_f64() * 1e3])
    }

    /// RNS: refresh of B ciphertexts of dimension N, ciphertext k encrypting μ_k/p̂_k with p̂_k = phats[k] (any integers,
    /// for instance the residues of an integer modulo RNS_MODULI): stair key switch, rounding with selection at the
    /// resolution R_k = p p̂_k (multiplication by p̂_k and rounding modulo p), combboot_rns, subtraction of E_k/R_k from
    /// the rounded ciphertext (its torus representation rounded to 2^-64: an error below n 2^-65 on the phase).
    pub fn refresh_rns<RG: Rng>(&self, cts: &[LweCiphertextOwned<u64>], phats: &[u64], rng: &mut RG, rbit: &mut u64) -> (Vec<LweCiphertextOwned<u64>>, [f64; 3]) {
        let t0 = std::time::Instant::now();
        let sw: Vec<(Vec<u64>, u64)> = cts.iter().map(|ct| self.stair.switch(ct.get_mask().as_ref(), *ct.get_body().data, rbit)).collect();
        let t1 = std::time::Instant::now();
        let p = self.p as u64;
        let rs: Vec<Rounded> = sw.iter().zip(phats).map(|((a, b), &ph)| round_select_r(a, *b, p * ph, &self.pool, self.vstar, rng)).collect();
        let t2 = std::time::Instant::now();
        let aints: Vec<Vec<u64>> = rs.iter().map(|r| r.abar.iter().map(|&x| x % p).collect()).collect();
        let bints: Vec<u64> = rs.iter().map(|r| r.bbar % p).collect();
        let errs = self.combboot_rns(&aints, &bints, phats);
        let outs = rs.iter().zip(errs.iter()).zip(phats).map(|((r, e), &ph)| {
            let rr = (p * ph) as u128;
            let lift = |x: u64| -> u64 { ((((x as u128) << 64) + rr / 2) / rr) as u64 };
            let mut fin = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            { let (mut fm, fb) = fin.get_mut_mask_and_body();
              for (u, &x) in r.abar.iter().enumerate() { fm.as_mut()[u] = lift(x); }
              *fb.data = lift(r.bbar); }
            lwe_ciphertext_sub_assign(&mut fin, e);
            fin
        }).collect();
        let t3 = std::time::Instant::now();
        (outs, [(t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, (t3 - t2).as_secs_f64() * 1e3])
    }
}
