//! TFHE-rs with our key switch, to neutralize the key switch in the comparisons. The blocks of TFHE-rs 1.8.1 (2 bits of
//! message and 2 bits of carry, PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128: n = 918, N = 2048, PBS (2^23,1)), as
//! in the bin tfhers_ks: a refresh of FheUint{2k} is k blocks, each a key switch to dimension n and a PBS of TFHE-rs
//! (centered modulus switch, LUT x ↦ x & 3, extraction):
//!   C: the LWE key switch of TFHE-rs ((2^4,4), TUniform(45));
//!   D: our polynomial stair key switch, with s = the first n coordinates of s' and the keys of the stair at the
//!      security level of the TFHE-rs sets (about 134 bits), noise 2^-11.33 against 2^-11.00 (measured, tfhers_ks).
//! Sequential (one thread) or blocks in parallel (rayon, one buffer per thread).
use crate::eb::cm;
use crate::stair::StairKey;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::fft_impl::fft64::math::fft::Fft;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as P;

/// Chain of the stair key switch 2048 -> 918 (as in tfhers_ks).
pub const CHAIN: &[(usize, u32, usize)] = &[(1360, 15, 1), (1264, 14, 1), (1072, 8, 2), (992, 7, 2), (918, 5, 3)];

/// log2 of the noise of a key of dimension x at the level of the TFHE-rs sets (as in tfhers_ks).
pub fn l2s134(x: usize) -> f64 {
    if x == 918 { return 45.0 - 0.5 * 3f64.log2() - 64.0; }
    let s128 = (-0.02591 * x as f64 + 2.77).max(-62.0);
    let grid = [(896.0, 5.4), (1024.0, 5.0), (1280.0, 3.9), (1536.0, 3.1), (1792.0, 2.8), (2048.0, 2.3)];
    let xf = x as f64;
    let mut slope = if xf < 896.0 { 5.4 } else { 2.3 };
    for w in grid.windows(2) {
        let ((x0, a), (x1, b)) = (w[0], w[1]);
        if xf >= x0 && xf <= x1 { slope = a + (b - a) * (xf - x0) / (x1 - x0); }
    }
    s128 + (6.0 / slope).max(1.22 + 0.00113 * (xf - 918.0))
}

fn fourier(std: &LweBootstrapKeyOwned<u64>) -> FourierLweBootstrapKeyOwned {
    let mut f = FourierLweBootstrapKey::new(std.input_lwe_dimension(), std.glwe_size(), std.polynomial_size(),
        std.decomposition_base_log(), std.decomposition_level_count());
    par_convert_standard_lwe_bootstrap_key_to_fourier(std, &mut f);
    f
}

/// PBS as in shortint (apply_programmable_bootstrap): centered modulus switch to 2N, blind rotation, extraction.
fn pbs(bsk: &FourierLweBootstrapKeyOwned, lwe: &LweCiphertextOwned<u64>, lut: &GlweCiphertextOwned<u64>,
       buf: &mut ComputationBuffers, fft: &Fft) -> LweCiphertextOwned<u64> {
    let n2 = lut.polynomial_size();
    let msed = lwe_ciphertext_centered_binary_modulus_switch::<u64, usize, _>(lwe.as_view(), n2.to_blind_rotation_input_modulus_log());
    let mut acc = lut.clone();
    let fv = fft.as_view();
    buf.resize(blind_rotate_assign_mem_optimized_requirement::<u64>(acc.glwe_size(), n2, fv).unaligned_bytes_required());
    blind_rotate_assign_mem_optimized(&msed, &mut acc, bsk, fv, buf.stack());
    let mut out = LweCiphertext::new(0u64, LweSize(n2.0 + 1), cm());
    extract_lwe_sample_from_glwe_ciphertext(&acc, &mut out, MonomialDegree(0));
    out
}

pub struct TfhersBlocks {
    pub n: usize, pub nn: usize,
    pub big_sk: LweSecretKeyOwned<u64>,
    small_c: LweSecretKeyOwned<u64>, small_d: LweSecretKeyOwned<u64>,
    ksk: LweKeyswitchKeyOwned<u64>, bsk_c: FourierLweBootstrapKeyOwned, bsk_d: FourierLweBootstrapKeyOwned,
    stair: StairKey, acc: GlweCiphertextOwned<u64>,
    pub delta: u64,
}

impl TfhersBlocks {
    pub fn new(seed: u64) -> Self {
        let (n, nn) = (P.lwe_dimension.0, P.polynomial_size.0);
        assert_eq!((n, nn), (918, 2048));
        let mut boxed = new_seeder(); let seeder = boxed.as_mut();
        let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(seed as u128));
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
        let glwe_sk = GlweSecretKey::generate_new_binary(GlweDimension(1), PolynomialSize(nn), &mut sgen);
        let big_sk = glwe_sk.clone().into_lwe_secret_key();
        let small_c = LweSecretKey::generate_new_binary(LweDimension(n), &mut sgen);
        let ksk = allocate_and_generate_new_lwe_keyswitch_key(&big_sk, &small_c, P.ks_base_log, P.ks_level,
            P.lwe_noise_distribution, cm(), &mut egen);
        let bsk_c = fourier(&par_allocate_and_generate_new_lwe_bootstrap_key(&small_c, &glwe_sk, P.pbs_base_log, P.pbs_level,
            P.glwe_noise_distribution, cm(), &mut egen));
        let small_d = LweSecretKey::from_container(big_sk.as_ref()[..n].to_vec());
        let mut krng = StdRng::seed_from_u64(seed ^ 0x5eed_134);
        let stair = StairKey::new_with(big_sk.as_ref(), nn, CHAIN, &mut krng, &l2s134);
        let bsk_d = fourier(&par_allocate_and_generate_new_lwe_bootstrap_key(&small_d, &glwe_sk, P.pbs_base_log, P.pbs_level,
            P.glwe_noise_distribution, cm(), &mut egen));
        let (_cks, sks) = tfhe::shortint::gen_keys(P);
        let acc = sks.generate_lookup_table(|x| x & 3).acc.clone();
        TfhersBlocks { n, nn, big_sk, small_c, small_d, ksk, bsk_c, bsk_d, stair, acc, delta: 1u64 << 59 }
    }

    /// k blocks under s' (messages m_i in [0, 16), noise 2^-30) and their messages.
    pub fn encrypt(&self, k: usize, seed: u64) -> (Vec<LweCiphertextOwned<u64>>, Vec<u64>) {
        let mut boxed = new_seeder(); let seeder = boxed.as_mut();
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let ms: Vec<u64> = (0..k).map(|_| rng.gen_range(0..16)).collect();
        let cts = ms.iter().map(|&m| allocate_and_encrypt_new_lwe_ciphertext(&self.big_sk, Plaintext(m * self.delta), noise, cm(), &mut egen)).collect();
        (cts, ms)
    }

    fn block(&self, ours: bool, ct: &LweCiphertextOwned<u64>, rbit: &mut u64, buf: &mut ComputationBuffers, fft: &Fft) -> LweCiphertextOwned<u64> {
        if ours {
            let (am, b) = self.stair.switch(ct.get_mask().as_ref(), *ct.get_body().data, rbit);
            let mut v = am; v.push(b);
            pbs(&self.bsk_d, &LweCiphertext::from_container(v, cm()), &self.acc, buf, fft)
        } else {
            let mut o = LweCiphertext::new(0u64, LweSize(self.n + 1), cm());
            keyswitch_lwe_ciphertext(&self.ksk, ct, &mut o);
            pbs(&self.bsk_c, &o, &self.acc, buf, fft)
        }
    }

    /// Refresh of the blocks one after the other (one thread). ours: D, otherwise C.
    pub fn refresh(&self, ours: bool, cts: &[LweCiphertextOwned<u64>], rbit: &mut u64) -> Vec<LweCiphertextOwned<u64>> {
        let fft = Fft::new(PolynomialSize(self.nn)); let mut buf = ComputationBuffers::new();
        cts.iter().map(|c| self.block(ours, c, rbit, &mut buf, &fft)).collect()
    }

    /// Refresh of the blocks in parallel (rayon pool of the caller).
    pub fn par_refresh(&self, ours: bool, cts: &[LweCiphertextOwned<u64>], seed: u64) -> Vec<LweCiphertextOwned<u64>> {
        cts.par_iter().enumerate().map_init(
            || (Fft::new(PolynomialSize(self.nn)), ComputationBuffers::new()),
            |(fft, buf), (i, c)| { let mut rb = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1); self.block(ours, c, &mut rb, buf, fft) },
        ).collect()
    }

    /// Checks that the outputs encrypt m & 3 under s' (both variants extract under the GLWE key).
    pub fn check(&self, outs: &[LweCiphertextOwned<u64>], ms: &[u64]) {
        for (o, &m) in outs.iter().zip(ms) {
            let ph = decrypt_lwe_ciphertext(&self.big_sk, o).0;
            assert_eq!((ph.wrapping_add(self.delta / 2) / self.delta) % 16, m & 3, "wrong block");
        }
        let _ = (&self.small_c, &self.small_d);
    }
}
