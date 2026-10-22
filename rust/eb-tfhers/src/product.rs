//! Product of two LWE ciphertexts (Section 6 of the paper) on TFHE-rs 1.8.1: port of julia/src/fast_product.jl, with the FFT of
//! TFHE-rs.
//!   1. LWE -> RLWE: A(X) = a_0 - Σ_{j≥1} a_j X^{N-j}, so that [A s']_0 = a·s'.
//!   2. Trace modulo the odd q' = 2^64 - 1: product by N^{-1} = 2^{64 - log2 N} (a rotation of the bits), then
//!      c ← c + KS_d(τ_d(c)) for d = 2^j + 1, j = log2 N, ..., 1: Tr(N^{-1} Φ) = Φ_0. The keys of the trace are split in
//!      two signed 32-bit limbs of their centered lifts (lift = lo + 2^32 hi): the products of the digits by each limb are
//!      integers below 2^53, which the FFT returns exactly; they are recombined modulo q' (2^64 ≡ 1).
//!   3. Tensor product of the lifts in [-1/2, 1/2), in signed 16-bit limbs (exact products, below 2^45), scaled by p̂:
//!      D0 - D1 s' + D2 s'^2 = p̂ Φ̃1 Φ̃2 (mod 1); relinearization of D2 with RLWE_{s'}(s'^2 g_l) (64-bit keys, the error of
//!      the FFT is not amplified); extraction of the constant coefficient: LWE_{s'}(p̂ μ1 μ2).
//! Exact integers from the FFT of TFHE-rs: one factor is transformed as integers (forward_as_integer), the other as torus
//! elements (forward_as_torus: its signed lift times 2^-64), and backward_as_torus returns the product modulo 2^64.
use crate::params::{Config, LOG2_SIGMA_N, N};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial;
use tfhe::core_crypto::prelude::*;
use tfhe_fft::c64;

/// Configuration P of the refresh before a product (k = 1..4, p̂ = 4^k): the refresh (n, gadget of the rotation,
/// 16 V*/(2N)^2, chain of the stair key switch) and the gadget (log2 B, ℓ) of the trace; relinearization (2^10, 4).
pub struct PConfig { pub cfg: Config, pub gtr: (u32, usize), pub m: usize }
pub const GRL: (u32, usize) = (10, 4);

/// Configurations chosen for this crate (the noise model, margin 0.2 bit): cheapest refresh before a product such that
/// the refresh that follows the product fails with probability at most 2^-128 under the exact law of the product
/// noise (Laplace main term), standard deviations × 2^0.2; classic rotation of TFHE-rs (m = 1), error of its FFT.
/// Beyond k = 4 the product is not possible at N = 2048 in double precision.
pub const PCONFIGS_RS: [PConfig; 4] = [
    PConfig { cfg: Config { k: 1, n: 564, b: 13, l: 2, vterm: 1.0209e-05, chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13), (672, 1, 13), (656, 1, 13), (640, 1, 12), (624, 1, 12), (608, 1, 11), (592, 1, 11), (576, 1, 11), (564, 1, 10)] }, gtr: (8, 5), m: 1 },
    PConfig { cfg: Config { k: 2, n: 650, b: 15, l: 2, vterm: 1.1843e-05, chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13), (656, 1, 13), (650, 1, 13)] }, gtr: (8, 5), m: 1 },
    PConfig { cfg: Config { k: 3, n: 732, b: 12, l: 3, vterm: 1.3406e-05, chain: &[(1600, 13, 2), (1520, 12, 2), (1376, 8, 3), (1296, 6, 4), (1248, 5, 5), (1200, 4, 6), (1168, 3, 8), (1136, 3, 8), (1104, 2, 12), (1088, 2, 12), (1072, 2, 12), (1056, 1, 23), (1040, 1, 22), (1024, 1, 22), (1008, 1, 22), (992, 1, 21), (976, 1, 21), (960, 1, 21), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19), (880, 1, 19), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (732, 1, 15)] }, gtr: (8, 5), m: 1 },
    PConfig { cfg: Config { k: 4, n: 848, b: 9, l: 4, vterm: 1.5622e-05, chain: &[(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7), (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19), (880, 1, 19), (864, 1, 18), (848, 1, 18)] }, gtr: (4, 11), m: 1 },
];

/// The same with our multi-bit rotation (m = 2, eb::Own2) in the refresh before a product (the noise model, margin 0.2 bit,
/// time model of the Mac): cheaper than m = 1 for every k on the Apple M3 Max.
pub const PCONFIGS_RS2: [PConfig; 4] = [
    PConfig { cfg: Config { k: 1, n: 564, b: 13, l: 2, vterm: 1.0209e-05, chain: &[(1424, 17, 1), (1088, 8, 2), (992, 6, 3), (896, 4, 4), (848, 3, 5), (816, 2, 8), (800, 2, 8), (784, 2, 8), (768, 2, 8), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 13), (672, 1, 13), (656, 1, 13), (640, 1, 12), (624, 1, 12), (608, 1, 11), (592, 1, 11), (576, 1, 11), (564, 1, 10)] }, gtr: (8, 5), m: 2 },
    PConfig { cfg: Config { k: 2, n: 650, b: 15, l: 2, vterm: 1.1843e-05, chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13), (656, 1, 13), (650, 1, 13)] }, gtr: (8, 5), m: 2 },
    PConfig { cfg: Config { k: 3, n: 734, b: 11, l: 3, vterm: 1.3444e-05, chain: &[(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (734, 1, 15)] }, gtr: (8, 5), m: 2 },
    PConfig { cfg: Config { k: 4, n: 816, b: 8, l: 5, vterm: 1.5010e-05, chain: &[(1824, 22, 1), (1424, 11, 2), (1296, 8, 3), (1200, 6, 4), (1136, 4, 6), (1088, 3, 7), (1056, 3, 7), (1024, 2, 11), (1008, 2, 11), (992, 2, 11), (976, 1, 21), (960, 1, 20), (944, 1, 20), (928, 1, 20), (912, 1, 19), (896, 1, 19), (880, 1, 19), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17)] }, gtr: (6, 8), m: 2 },
];

/// RNS (bin rns_arith): configuration P for the residues modulo comb::RNS_MODULI (p̂ ≤ 263), chosen by
/// the noise model (margin 0.2 bit): the refresh before a product of k = 4 (p̂ = 256) with the
/// trace gadget (2^5, 9) and its stair key switch. `k` (4) only serves the power-of-two methods.
pub const PCONFIG_RNS: PConfig = PConfig { cfg: Config { k: 4, n: 816, b: 8, l: 5, vterm: 1.5010e-05, chain: &[(1680, 20, 1), (1312, 10, 2), (1184, 7, 3), (1104, 5, 4), (1040, 4, 5), (992, 3, 7), (960, 2, 10), (944, 2, 10), (928, 2, 10), (912, 2, 9), (896, 1, 19), (880, 1, 18), (864, 1, 18), (848, 1, 18), (832, 1, 17), (816, 1, 17)] }, gtr: (5, 9), m: 2 };

/// RNS of 32-bit integers (bin rns32_arith): configuration P for the residues modulo comb::RNS32_MODULI (p̂ ≤ 25), chosen by
/// the noise model (margin 0.2 bit; predicted 15.2 ms per residue on the Mac, P(fail) 2^-160.7).
/// `k` (2) only serves the power-of-two methods.
pub const PCONFIG_RNS32: PConfig = PConfig { cfg: Config { k: 2, n: 672, b: 15, l: 2, vterm: 1.2262e-05, chain: &[(1584, 19, 1), (1200, 9, 2), (1072, 6, 3), (1008, 5, 4), (944, 3, 6), (912, 3, 6), (880, 2, 9), (864, 2, 9), (848, 2, 9), (832, 1, 17), (816, 1, 17), (800, 1, 16), (784, 1, 16), (768, 1, 16), (752, 1, 15), (736, 1, 15), (720, 1, 14), (704, 1, 14), (688, 1, 14), (672, 1, 13)] }, gtr: (8, 5), m: 2 };

/// The configurations P of a run: PCFG=rs2 (default, m = 2) or PCFG=rs (m = 1).
pub fn pconfigs() -> &'static [PConfig] {
    match std::env::var("PCFG").as_deref() { Ok("rs") => &PCONFIGS_RS, _ => &PCONFIGS_RS2 }
}

/// Predicted variances: key switching of the trace (one automorphism, gadget (2^b, ℓ), exact products), whole trace
/// noise T = N(N-1) vks (sum over the coefficients) and its constant coefficient V0 = (N^2-1)/3 vks, and the product
/// (var_product of julia/src/product.jl, random messages) for the input variance ve.
pub fn var_ks_trace(g: (u32, usize)) -> f64 {
    let (b, l) = (g.0 as i32, g.1 as f64); let n = N as f64;
    l * n * (2f64.powi(2 * b) + 2.0) / 12.0 * 2f64.powf(2.0 * LOG2_SIGMA_N) + (n / 2.0) * 2f64.powf(-2.0 * l * b as f64) / 12.0
}
pub fn var_trace(g: (u32, usize)) -> (f64, f64) {
    let n = N as f64; let v = var_ks_trace(g); (n * (n - 1.0) * v, (n * n - 1.0) / 3.0 * v)
}
pub fn var_product(phat: f64, ve: f64, t: f64, v0: f64) -> f64 {
    let n = N as f64; let (b, l) = (GRL.0 as i32, GRL.1 as f64);
    let rl = l * n * (2f64.powi(2 * b) + 2.0) / 12.0 * 2f64.powf(2.0 * LOG2_SIGMA_N) + n * 2f64.powf(-2.0 * l * b as f64) / 12.0 * n * n / 48.0;
    phat * phat * (n / 24.0 + 1.0 / 12.0) * 2.0 * (ve + t) + phat * phat / 12.0 * 2.0 * (ve + v0) + rl
}

const QM: u64 = u64::MAX; // q' = 2^64 - 1, residues in [0, q')
const H: usize = N / 2;

#[inline] fn qnorm(x: u64) -> u64 { if x == QM { 0 } else { x } }
#[inline] pub fn qadd(a: u64, b: u64) -> u64 { let (s, c) = a.overflowing_add(b); qnorm(if c { s + 1 } else { s }) }
#[inline] pub fn qneg(x: u64) -> u64 { if x == 0 { 0 } else { QM - x } }
#[inline] pub fn qsub(a: u64, b: u64) -> u64 { qadd(a, qneg(b)) }
/// Centered lift of a residue, in (-q'/2, q'/2).
#[inline] pub fn qlift(x: u64) -> i64 { if x < 1u64 << 63 { x as i64 } else { (x as i64).wrapping_add(1) } }
/// Residue of an integer.
#[inline] pub fn qfrom(v: i128) -> u64 { let r = v.rem_euclid(QM as i128); r as u64 }
/// Modulus switches 2^64 -> q' and q' -> 2^64.
#[inline] fn to_q(x: u64) -> u64 { if x < 1u64 << 63 { x } else { x - 1 } }
#[inline] fn from_q(x: u64) -> u64 { if x < 1u64 << 63 { x } else { x + 1 } }
/// Residue modulo q' of lo + 2^32 hi (exact integers, |lo|, |hi| < 2^62): with hi = 2^32 hh + hl, 0 ≤ hl < 2^32,
/// 2^32 hi ≡ hh + 2^32 hl (2^64 ≡ 1), and 2^32 hl < q'.
#[inline] fn q_combine(lo: i64, hi: i64) -> u64 {
    let y = lo + (hi >> 32);
    let ym = (y as u64).wrapping_sub((y < 0) as u64); // y modulo q' (2^64 - |y| ≡ q' - |y| + 1)
    qadd(((hi as u64) & 0xffff_ffff) << 32, qnorm(ym))
}
/// Torus value of a residue (tests).
pub fn q_to_f64(x: u64) -> f64 { qlift(x) as f64 / 18446744073709551615.0 }

/// Balanced digits of the gadget (2^b, ℓ), g_j = 2^(64 - j b), rounded to the top ℓ b bits (decompose of glwe.jl).
fn decompose(b: u32, l: usize, t: &[u64], out: &mut [Vec<u64>]) {
    let shift = 64 - b * l as u32; let mask = (1u64 << b) - 1; let half = 1u64 << (b - 1);
    // digit-major (vectorizable): y = top l b bits rounded, carries in 0/1
    let mut y: Vec<u64> = t.iter().map(|&x| (x >> shift) + ((x >> (shift - 1)) & 1)).collect();
    let mut carry = vec![0u64; t.len()];
    for j in (0..l).rev() {
        let o = &mut out[j][..t.len()];
        for i in 0..t.len() {
            let d = (y[i] & mask) + carry[i]; y[i] >>= b;
            let c = (d >= half) as u64; carry[i] = c;
            o[i] = d.wrapping_sub(c << b);
        }
    }
}

/// Negacyclic product a·s, s binary, modulo 2^64 (q = false) or q' (q = true).
fn mul_bin(a: &[u64], s: &[u64], q: bool) -> Vec<u64> {
    let mut out = vec![0u64; N];
    for (i, &si) in s.iter().enumerate() {
        if si == 0 { continue; }
        for j in 0..N {
            let k = i + j;
            if q {
                if k < N { out[k] = qadd(out[k], a[j]); } else { out[k - N] = qsub(out[k - N], a[j]); }
            } else if k < N { out[k] = out[k].wrapping_add(a[j]); } else { out[k - N] = out[k - N].wrapping_sub(a[j]); }
        }
    }
    out
}

/// τ_d(s)(X) = s(X^d) modulo X^N + 1 (signed integers).
fn aut_poly(s: &[i64], d: usize) -> Vec<i64> {
    let mut out = vec![0i64; N];
    for (i, &x) in s.iter().enumerate() {
        let e = (i * d) % (2 * N);
        if e < N { out[e] += x; } else { out[e - N] -= x; }
    }
    out
}

/// Automorphism X ↦ X^d on residues modulo q'.
fn aut_q(out: &mut [u64], p: &[u64], d: usize) {
    let mask = 2 * N - 1;
    for i in 0..N {
        let e = (i * d) & mask;
        if e < N { out[e] = p[i]; } else { out[e - N] = qneg(p[i]); }
    }
}

/// Signed 16-bit limbs of the lifts in [-2^63, 2^63): t = Σ_{u=0}^{3} 2^{16u} x_u.
fn limbs16(t: &[u64], out: &mut [Vec<u64>]) {
    for (i, &x) in t.iter().enumerate() {
        let mut v = x as i64 as i128;
        for u in 0..3 {
            let mut l = (v & 0xffff) as i64; if l >= 0x8000 { l -= 0x10000; }
            out[u][i] = l as u64; v = (v - l as i128) >> 16;
        }
        out[3][i] = v as i64 as u64;
    }
}

/// round(p̂ V / 2^64) mod 2^64 from V = Σ_w 2^{16w} P_w (V mod 2^128).
#[inline] fn scale_phat(pw: &[i64; 7], phat: u128) -> u64 {
    let mut v: u128 = 0;
    for w in 0..7 { v = v.wrapping_add(((pw[w] as i128) as u128) << (16 * w)); }
    (phat.wrapping_mul(v).wrapping_add(1u128 << 63) >> 64) as u64
}

/// acc_q += d k_q, q = 0..3, in one pass.
#[inline] fn mac4(acc: &mut [Vec<c64>], d: &[c64], k: &[Vec<c64>; 4]) {
    let (a01, a23) = acc.split_at_mut(2); let (a0, a1) = a01.split_at_mut(1); let (a2, a3) = a23.split_at_mut(1);
    let (a0, a1, a2, a3) = (&mut a0[0][..d.len()], &mut a1[0][..d.len()], &mut a2[0][..d.len()], &mut a3[0][..d.len()]);
    let (k0, k1, k2, k3) = (&k[0][..d.len()], &k[1][..d.len()], &k[2][..d.len()], &k[3][..d.len()]);
    for i in 0..d.len() {
        let x = d[i];
        let m = |o: &mut c64, y: c64| { o.re += x.re * y.re - x.im * y.im; o.im += x.re * y.im + x.im * y.re; };
        m(&mut a0[i], k0[i]); m(&mut a1[i], k1[i]); m(&mut a2[i], k2[i]); m(&mut a3[i], k3[i]);
    }
}

#[inline] fn mac(acc: &mut [c64], a: &[c64], b: &[c64]) {
    for ((o, x), y) in acc.iter_mut().zip(a).zip(b) {
        o.re += x.re * y.re - x.im * y.im; o.im += x.re * y.im + x.im * y.re;
    }
}

/// Keys of the product, Fourier domain of TFHE-rs.
pub struct ProductKeys {
    /// tr[j - 1][l]: (a_lo, a_hi, b_lo, b_hi) of RLWE_{s'}(τ_d(s') g_l) modulo q', d = 2^j + 1 (limbs as torus elements).
    tr: Vec<Vec<[Vec<c64>; 4]>>,
    /// rlk[l]: (a, b) of RLWE_{s'}(s'^2 g_l) modulo 2^64.
    rlk: Vec<[Vec<c64>; 2]>,
    pub gtr: (u32, usize), pub grl: (u32, usize),
    fft: Fft,
}

/// Work buffers of the product.
pub struct ProductWork {
    buf: ComputationBuffers,
    ta: Vec<u64>, tb: Vec<u64>, x: Vec<u64>, lo: Vec<u64>,
    dig: Vec<Vec<u64>>, fd: Vec<c64>, acc: Vec<Vec<c64>>,
    lim: Vec<Vec<u64>>, fl: Vec<Vec<c64>>, pw: Vec<Vec<u64>>,
}

impl ProductKeys {
    /// Keys under the binary key s' (coefficients of s'(X), 0/1 as u64), keys at σ = σ_128(N), gadgets (log2 B, ℓ)
    /// of the trace and of the relinearization.
    pub fn new<R: Rng>(sk: &[u64], gtr: (u32, usize), grl: (u32, usize), rng: &mut R) -> Self {
        assert_eq!(sk.len(), N);
        let fft = Fft::new(PolynomialSize(N));
        let sig = 2f64.powf(LOG2_SIGMA_N) * 18446744073709551616.0;
        let gauss = Normal::new(0.0, sig).unwrap();
        let mut buf = ComputationBuffers::new();
        let fv = fft.as_view();
        buf.resize(fv.forward_scratch().unaligned_bytes_required().max(fv.backward_scratch().unaligned_bytes_required()));
        let tor = |p: &[u64], buf: &mut ComputationBuffers| -> Vec<c64> {
            let mut f = FourierPolynomial::new(PolynomialSize(N));
            fft.as_view().forward_as_torus(f.as_mut_view(), Polynomial::from_container(p), buf.stack());
            f.data.to_vec()
        };
        let si: Vec<i64> = sk.iter().map(|&x| x as i64).collect();
        let l2n = N.trailing_zeros() as usize;
        let mut tr = Vec::new();
        for j in 1..=l2n {
            let ts = aut_poly(&si, (1 << j) + 1);
            let mut rows = Vec::new();
            for l in 1..=gtr.1 {
                let g = 1u64 << (64 - l as u32 * gtr.0);
                let a: Vec<u64> = (0..N).map(|_| qnorm(rng.gen::<u64>())).collect();
                let asq = mul_bin(&a, sk, true);
                let b: Vec<u64> = (0..N).map(|i| {
                    let m = qfrom(ts[i] as i128 * g as i128);
                    let e = qfrom(gauss.sample(rng).round() as i128);
                    qadd(qadd(asq[i], m), e)
                }).collect();
                let split = |p: &[u64]| -> (Vec<u64>, Vec<u64>) {
                    let mut lo = vec![0u64; N]; let mut hi = vec![0u64; N];
                    for i in 0..N {
                        let v = qlift(p[i]); let mut l = v & 0xffff_ffff; if l >= 1i64 << 31 { l -= 1i64 << 32; }
                        lo[i] = l as u64; hi[i] = ((v - l) >> 32) as u64;
                    }
                    (lo, hi)
                };
                let (alo, ahi) = split(&a); let (blo, bhi) = split(&b);
                rows.push([tor(&alo, &mut buf), tor(&ahi, &mut buf), tor(&blo, &mut buf), tor(&bhi, &mut buf)]);
            }
            tr.push(rows);
        }
        // s'^2 modulo X^N + 1 (exact)
        let mut s2 = vec![0i64; N];
        for i in 0..N { if si[i] == 0 { continue; } for k in 0..N { if si[k] == 0 { continue; }
            let e = i + k; if e < N { s2[e] += 1; } else { s2[e - N] -= 1; } } }
        let mut rlk = Vec::new();
        for l in 1..=grl.1 {
            let g = 1u64 << (64 - l as u32 * grl.0);
            let a: Vec<u64> = (0..N).map(|_| rng.gen::<u64>()).collect();
            let as_ = mul_bin(&a, sk, false);
            let b: Vec<u64> = (0..N).map(|i| as_[i].wrapping_add((s2[i] as u64).wrapping_mul(g))
                .wrapping_add(gauss.sample(rng).round() as i64 as u64)).collect();
            rlk.push([tor(&a, &mut buf), tor(&b, &mut buf)]);
        }
        ProductKeys { tr, rlk, gtr, grl, fft }
    }

    pub fn work(&self) -> ProductWork {
        let fv = self.fft.as_view(); let mut buf = ComputationBuffers::new();
        buf.resize(fv.forward_scratch().unaligned_bytes_required().max(fv.backward_scratch().unaligned_bytes_required()));
        let z = c64 { re: 0.0, im: 0.0 }; let lmax = self.gtr.1.max(self.grl.1);
        ProductWork { buf, ta: vec![0; N], tb: vec![0; N], x: vec![0; N], lo: vec![0; N],
            dig: vec![vec![0u64; N]; lmax], fd: vec![z; H], acc: vec![vec![z; H]; 21],
            lim: vec![vec![0u64; N]; 16], fl: vec![vec![z; H]; 16], pw: vec![vec![0u64; N]; 7] }
    }

    fn fwd_int(&self, out: &mut [c64], p: &[u64], buf: &mut ComputationBuffers) {
        let mut f = FourierPolynomial { data: out };
        self.fft.as_view().forward_as_integer(f.as_mut_view(), Polynomial::from_container(p), buf.stack());
    }
    fn fwd_tor(&self, out: &mut [c64], p: &[u64], buf: &mut ComputationBuffers) {
        let mut f = FourierPolynomial { data: out };
        self.fft.as_view().forward_as_torus(f.as_mut_view(), Polynomial::from_container(p), buf.stack());
    }
    fn bwd(&self, out: &mut [u64], f: &[c64], buf: &mut ComputationBuffers) {
        let fp = FourierPolynomial { data: f };
        self.fft.as_view().backward_as_torus(Polynomial::from_container(out), fp.as_view(), buf.stack());
    }

    /// Tr(N^{-1} c) modulo q' of the RLWE ciphertext (a, b) modulo q', in place.
    pub fn trace(&self, w: &mut ProductWork, a: &mut [u64], b: &mut [u64]) {
        let l2n = N.trailing_zeros();
        for i in 0..N { a[i] = a[i].rotate_left(64 - l2n); b[i] = b[i].rotate_left(64 - l2n); }
        let (bt, lt) = self.gtr;
        let ProductWork { buf, ta, tb, x, lo, dig, fd, acc, .. } = w;
        for j in (1..=l2n as usize).rev() {
            let d = (1 << j) + 1;
            aut_q(ta, a, d); aut_q(tb, b, d);
            decompose(bt, lt, ta, dig);
            for q in 0..4 { acc[q].iter_mut().for_each(|z| *z = c64 { re: 0.0, im: 0.0 }); }
            for l in 0..lt {
                self.fwd_int(fd, &dig[l], buf);
                mac4(&mut acc[0..4], fd, &self.tr[j - 1][l]);
            }
            // α = Σ d_l K_a, β = Σ d_l K_b (exact, recombined from the limbs); c += (-α, τ_d(b) - β)
            self.bwd(lo, &acc[0], buf); self.bwd(x, &acc[1], buf);
            for i in 0..N { a[i] = qsub(a[i], q_combine(lo[i] as i64, x[i] as i64)); }
            self.bwd(lo, &acc[2], buf); self.bwd(x, &acc[3], buf);
            for i in 0..N { b[i] = qadd(b[i], qsub(tb[i], q_combine(lo[i] as i64, x[i] as i64))); }
        }
    }

    /// LWE_{s'}(μ) of dimension N -> RLWE_{s'} modulo 2^64 whose phase is μ + e (constant coefficient) + noise of the trace.
    pub fn lwe_to_rlwe(&self, w: &mut ProductWork, c: &LweCiphertextOwned<u64>) -> (Vec<u64>, Vec<u64>) {
        let m = c.get_mask(); let ma = m.as_ref();
        let mut a = vec![0u64; N]; let mut b = vec![0u64; N];
        a[0] = to_q(ma[0]);
        for j in 1..N { a[N - j] = qneg(to_q(ma[j])); }
        b[0] = to_q(*c.get_body().data);
        self.trace(w, &mut a, &mut b);
        (a.into_iter().map(from_q).collect(), b.into_iter().map(from_q).collect())
    }

    /// Tensor product (D0, D1, D2) of two RLWE ciphertexts modulo 2^64, scaled by p̂.
    pub fn tensor(&self, w: &mut ProductWork, c1: &(Vec<u64>, Vec<u64>), c2: &(Vec<u64>, Vec<u64>), phat: u64)
                  -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let ProductWork { buf, lim, fl, acc, pw, .. } = w;
        // limbs: A1 0..4, B1 4..8 (as integers), A2 8..12, B2 12..16 (as torus elements, to get exact products)
        for (q, t) in [&c1.0, &c1.1, &c2.0, &c2.1].iter().enumerate() { limbs16(t, &mut lim[4 * q..4 * q + 4]); }
        for i in 0..16 {
            if i < 8 { self.fwd_int(&mut fl[i], &lim[i], buf); } else { self.fwd_tor(&mut fl[i], &lim[i], buf); }
        }
        for a in acc.iter_mut() { a.iter_mut().for_each(|z| *z = c64 { re: 0.0, im: 0.0 }); }
        for u in 0..4 { for v in 0..4 {
            let wgt = u + v;
            let (a1, b1, a2, b2) = (&fl[u], &fl[4 + u], &fl[8 + v], &fl[12 + v]);
            mac(&mut acc[wgt], b1, b2);                       // D0: b1 b2
            mac(&mut acc[7 + wgt], b1, a2);                   // D1: b1 a2 + a1 b2
            mac(&mut acc[7 + wgt], a1, b2);
            mac(&mut acc[14 + wgt], a1, a2);                  // D2: a1 a2
        } }
        let ph = phat as u128;
        let mut outs = (vec![0u64; N], vec![0u64; N], vec![0u64; N]);
        for o in 0..3 {
            for wgt in 0..7 { self.bwd(&mut pw[wgt], &acc[7 * o + wgt], buf); }
            let out = match o { 0 => &mut outs.0, 1 => &mut outs.1, _ => &mut outs.2 };
            for i in 0..N {
                let p: [i64; 7] = std::array::from_fn(|wgt| pw[wgt][i] as i64);
                out[i] = scale_phat(&p, ph);
            }
        }
        outs
    }

    /// Relinearization: (D1 + Σ_l d_l rlk_l.a, D0 + Σ_l d_l rlk_l.b), d_l the digits of D2.
    pub fn relin(&self, w: &mut ProductWork, d0: &[u64], d1: &[u64], d2: &[u64]) -> GlweCiphertextOwned<u64> {
        let ProductWork { buf, dig, fd, acc, x, .. } = w;
        let (b, l) = self.grl;
        decompose(b, l, d2, dig);
        for q in 0..2 { acc[q].iter_mut().for_each(|z| *z = c64 { re: 0.0, im: 0.0 }); }
        for j in 0..l {
            self.fwd_int(fd, &dig[j], buf);
            mac(&mut acc[0], fd, &self.rlk[j][0]); mac(&mut acc[1], fd, &self.rlk[j][1]);
        }
        let mut c = vec![0u64; 2 * N];
        self.bwd(x, &acc[0], buf);
        for i in 0..N { c[i] = d1[i].wrapping_add(x[i]); }
        self.bwd(x, &acc[1], buf);
        for i in 0..N { c[N + i] = d0[i].wrapping_add(x[i]); }
        GlweCiphertext::from_container(c, PolynomialSize(N), CiphertextModulus::new_native())
    }

    /// Product of LWE_{s'}(μ1) and LWE_{s'}(μ2) (dimension N, μ_i ∈ (1/p̂)Z): LWE_{s'}(p̂ μ1 μ2).
    pub fn product(&self, w: &mut ProductWork, c1: &LweCiphertextOwned<u64>, c2: &LweCiphertextOwned<u64>, phat: u64)
                   -> LweCiphertextOwned<u64> {
        let t1 = self.lwe_to_rlwe(w, c1); let t2 = self.lwe_to_rlwe(w, c2);
        let (d0, d1, d2) = self.tensor(w, &t1, &t2, phat);
        let g = self.relin(w, &d0, &d1, &d2);
        let mut out = LweCiphertext::new(0u64, LweSize(N + 1), CiphertextModulus::new_native());
        extract_lwe_sample_from_glwe_ciphertext(&g, &mut out, MonomialDegree(0));
        out
    }
}

/// Phase modulo q' of an RLWE ciphertext under the binary key s' (tests).
pub fn q_phase(sk: &[u64], a: &[u64], b: &[u64]) -> Vec<u64> {
    let as_ = mul_bin(a, sk, true);
    (0..N).map(|i| qsub(b[i], as_[i])).collect()
}

/// Phase modulo 2^64 of an RLWE ciphertext (a, b) (tests).
pub fn phase64(sk: &[u64], a: &[u64], b: &[u64]) -> Vec<u64> {
    let as_ = mul_bin(a, sk, false);
    (0..N).map(|i| b[i].wrapping_sub(as_[i])).collect()
}

/// Check of the exactness of one automorphism key switching of the trace (level j = log2 N): the products of the
/// digits by the limbs of the keys, computed by the FFT, against the exact negacyclic convolution (i128). Returns the
/// largest difference of the residues α, β (0 if exact).
pub fn check_exact_ks<R: Rng>(k: &ProductKeys, w: &mut ProductWork, sk: &[u64], rng: &mut R) -> u64 {
    // the key rows of level j are not stored in the standard domain: re-encrypt one row with known limbs
    let _ = sk;
    let (bt, lt) = k.gtr;
    let a: Vec<u64> = (0..N).map(|_| qnorm(rng.gen::<u64>())).collect();
    let mut dig = vec![vec![0u64; N]; lt];
    decompose(bt, lt, &a, &mut dig);
    let key: Vec<u64> = (0..N).map(|_| qnorm(rng.gen::<u64>())).collect();
    let mut lo = vec![0u64; N]; let mut hi = vec![0u64; N];
    for i in 0..N {
        let v = qlift(key[i]); let mut l = v & 0xffff_ffff; if l >= 1i64 << 31 { l -= 1i64 << 32; }
        lo[i] = l as u64; hi[i] = ((v - l) >> 32) as u64;
    }
    let z = c64 { re: 0.0, im: 0.0 };
    let (mut flo, mut fhi) = (vec![z; H], vec![z; H]);
    k.fwd_tor(&mut flo, &lo, &mut w.buf); k.fwd_tor(&mut fhi, &hi, &mut w.buf);
    let (mut alo, mut ahi) = (vec![z; H], vec![z; H]);
    for l in 0..lt {
        let mut fd = vec![z; H];
        k.fwd_int(&mut fd, &dig[l], &mut w.buf);
        mac(&mut alo, &fd, &flo); mac(&mut ahi, &fd, &fhi);
    }
    let (mut rlo, mut rhi) = (vec![0u64; N], vec![0u64; N]);
    k.bwd(&mut rlo, &alo, &mut w.buf); k.bwd(&mut rhi, &ahi, &mut w.buf);
    // exact: Σ_l d_l * key (negacyclic), modulo q'
    let mut exact = vec![0i128; N];
    for l in 0..lt {
        for i in 0..N {
            let d = dig[l][i] as i64 as i128; if d == 0 { continue; }
            for j in 0..N {
                let e = i + j; let v = d * qlift(key[j]) as i128;
                if e < N { exact[e] += v; } else { exact[e - N] -= v; }
            }
        }
    }
    let mut worst = 0u64;
    for i in 0..N {
        let f = q_combine(rlo[i] as i64, rhi[i] as i64); let x = qfrom(exact[i]);
        let diff = qsub(f, x); let dd = qlift(diff).unsigned_abs();
        worst = worst.max(dd);
    }
    worst
}
