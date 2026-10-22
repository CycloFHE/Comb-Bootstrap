//! TFHE-rs with our key switch. Refresh of an 8-bit integer as TFHE-rs does it (4 blocks of 2 bits of message and
//! 2 bits of carry, one key switch and one PBS per block, LUT x ↦ x & 3), with the default parameters of TFHE-rs 1.8.1
//! (PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128: n = 918, N = 2048, PBS (2^23,1), LWE key switch (2^4,4),
//! TUniform(45) / TUniform(17), centered modulus switch), one thread, same session:
//!   A  high-level FheUint8 `&a & MAX`;
//!   B  shortint, 4 × apply_lookup_table;
//!   C  core, 4 × (LWE key switch of TFHE-rs + PBS);
//!   D  core, 4 × (our polynomial stair key switch + the same PBS), with s = the first n coordinates of s' and the
//!      keys of the stair at the security level of the TFHE-rs sets (about 134 bits: σ_134).
//! Also: measured standard deviation of the two key switches (D must not be noisier than C, so that the failure
//! probability of the parameter set, 2^-129.6, still holds), and correctness of the 4 blocks.
//! Times in ms, minimum over REPS runs (default 5) after one warm-up run.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin tfhers_ks
use eb_tfhers::eb::{cm, tor};
use eb_tfhers::stair::StairKey;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::fft_impl::fft64::math::fft::Fft;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as P;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};

/// Chain of the stair key switch 2048 -> 918 [(x, log2 B, ℓ)], chosen by the noise model: predicted variance
/// 2^-22.63 against 2^-22.03 for the key switch of TFHE-rs.
const CHAIN: &[(usize, u32, usize)] = &[(1360, 15, 1), (1264, 14, 1), (1072, 8, 2), (992, 7, 2), (918, 5, 3)];

/// log2 of the noise of a key of dimension x at the level of the TFHE-rs sets (about 134 bits): σ_128 of the paper
/// plus 6 bits of security divided by the slope dλ/dlog2σ of the security curve, never below the offsets of the
/// TFHE-rs sets; at x = 918, the standard deviation of TUniform(45).
fn l2s134(x: usize) -> f64 {
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

fn best<O: FnMut()>(reps: usize, mut op: O) -> f64 {
    op();
    (0..reps).map(|_| { let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min)
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

fn main() {
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let nks: usize = std::env::var("NKS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let (n, nn) = (P.lwe_dimension.0, P.polynomial_size.0);
    assert_eq!((n, nn), (918, 2048));
    assert_eq!(CHAIN.last().unwrap().0, n);
    println!("TFHE-rs 1.8.1, PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 (n = {n}, N = {nn}, PBS (2^{},{}), KS (2^{},{})), one thread",
             P.pbs_base_log.0, P.pbs_level.0, P.ks_base_log.0, P.ks_level.0);

    // ---- keys
    let mut boxed = new_seeder(); let seeder = boxed.as_mut();
    let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(0x7f4a_9e37));
    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
    let glwe_sk = GlweSecretKey::generate_new_binary(GlweDimension(1), PolynomialSize(nn), &mut sgen);
    let big_sk = glwe_sk.clone().into_lwe_secret_key();
    // C: independent small key, LWE key switch of TFHE-rs
    let small_c = LweSecretKey::generate_new_binary(LweDimension(n), &mut sgen);
    let ksk: LweKeyswitchKeyOwned<u64> = allocate_and_generate_new_lwe_keyswitch_key(&big_sk, &small_c, P.ks_base_log,
        P.ks_level, P.lwe_noise_distribution, cm(), &mut egen);
    let bsk_c = fourier(&par_allocate_and_generate_new_lwe_bootstrap_key(&small_c, &glwe_sk, P.pbs_base_log, P.pbs_level,
        P.glwe_noise_distribution, cm(), &mut egen));
    // D: s = first n coordinates of s', polynomial stair key switch
    let small_d = LweSecretKey::from_container(big_sk.as_ref()[..n].to_vec());
    let mut krng = StdRng::seed_from_u64(0x5eed_134);
    let stair = StairKey::new_with(big_sk.as_ref(), nn, CHAIN, &mut krng, &l2s134);
    let bsk_d = fourier(&par_allocate_and_generate_new_lwe_bootstrap_key(&small_d, &glwe_sk, P.pbs_base_log, P.pbs_level,
        P.glwe_noise_distribution, cm(), &mut egen));
    // LUT x ↦ x & 3 of shortint (same encoding, same rotation of the boxes as its centered modulus switch)
    let (cks, sks) = tfhe::shortint::gen_keys(P);
    let lut = sks.generate_lookup_table(|x| x & 3);
    let acc = lut.acc.clone();
    let fft = Fft::new(PolynomialSize(nn));
    let mut buf = ComputationBuffers::new();
    let delta = 1u64 << 59; // 2^63 / (4 · 4)

    // ---- noise of the two key switches (input: encryptions of 0 under s' with negligible noise)
    let tiny = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-55)), 0.0);
    let mut rbit = 0x853c49e6748fea9bu64;
    let (mut vc, mut vd) = (0f64, 0f64);
    for _ in 0..nks {
        let z: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&big_sk, Plaintext(0), tiny, cm(), &mut egen);
        let mut oc = LweCiphertext::new(0u64, LweSize(n + 1), cm());
        keyswitch_lwe_ciphertext(&ksk, &z, &mut oc);
        let e = tor(decrypt_lwe_ciphertext(&small_c, &oc).0); vc += e * e;
        let (am, b) = stair.switch(z.get_mask().as_ref(), *z.get_body().data, &mut rbit);
        let mut v = am; v.push(b);
        let od = LweCiphertext::from_container(v, cm());
        let e = tor(decrypt_lwe_ciphertext(&small_d, &od).0); vd += e * e;
    }
    let (sc, sd) = (0.5 * (vc / nks as f64).log2(), 0.5 * (vd / nks as f64).log2());
    println!("  key switch noise over {nks} samples: TFHE-rs 2^{sc:.2} (predicted 2^-11.02), ours 2^{sd:.2} (predicted 2^-11.32)");

    // ---- one block: key switch + PBS, correctness on the 16 inputs
    let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
    let mut rng = StdRng::seed_from_u64(8);
    let blocks: Vec<LweCiphertextOwned<u64>> = (0..4).map(|_| {
        let m: u64 = rng.gen_range(0..16);
        allocate_and_encrypt_new_lwe_ciphertext(&big_sk, Plaintext(m * delta), noise, cm(), &mut egen)
    }).collect();
    let ks_c = |ct: &LweCiphertextOwned<u64>| { let mut o = LweCiphertext::new(0u64, LweSize(n + 1), cm()); keyswitch_lwe_ciphertext(&ksk, ct, &mut o); o };
    let ks_d = |ct: &LweCiphertextOwned<u64>, rbit: &mut u64| {
        let (am, b) = stair.switch(ct.get_mask().as_ref(), *ct.get_body().data, rbit); let mut v = am; v.push(b); LweCiphertext::from_container(v, cm())
    };
    for m in 0..16u64 {
        let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&big_sk, Plaintext(m * delta), noise, cm(), &mut egen);
        for (name, o) in [("C", pbs(&bsk_c, &ks_c(&ct), &acc, &mut buf, &fft)), ("D", pbs(&bsk_d, &ks_d(&ct, &mut rbit), &acc, &mut buf, &fft))] {
            let ph = decrypt_lwe_ciphertext(&big_sk, &o).0;
            let r = (ph.wrapping_add(delta / 2) / delta) % 16;
            assert_eq!(r, m & 3, "{name}: wrong block for m = {m}");
        }
    }
    println!("  correctness: the 16 inputs of a block give x & 3 with both key switches");

    // ---- times
    let mut csv = Vec::new();
    let t_ksc = best(reps * 4, || { let _ = ks_c(&blocks[0]); });
    let t_ksd = best(reps * 4, || { let _ = ks_d(&blocks[0], &mut rbit); });
    let kc = ks_c(&blocks[0]); let kd = ks_d(&blocks[0], &mut rbit);
    let t_pbs_c = best(reps * 4, || { let _ = pbs(&bsk_c, &kc, &acc, &mut buf, &fft); });
    let t_pbs_d = best(reps * 4, || { let _ = pbs(&bsk_d, &kd, &acc, &mut buf, &fft); });
    println!("  one block: key switch TFHE-rs {t_ksc:.3} ms, ours {t_ksd:.3} ms; PBS {t_pbs_c:.3} ms (C) / {t_pbs_d:.3} ms (D)");
    csv.push(format!("ks,tfhers,{t_ksc:.4}")); csv.push(format!("ks,ours,{t_ksd:.4}"));
    csv.push(format!("pbs,c,{t_pbs_c:.4}")); csv.push(format!("pbs,d,{t_pbs_d:.4}"));
    let t_c = best(reps, || { for b in &blocks { let _ = pbs(&bsk_c, &ks_c(b), &acc, &mut buf, &fft); } });
    let t_d = best(reps, || { for b in &blocks { let k = ks_d(b, &mut rbit); let _ = pbs(&bsk_d, &k, &acc, &mut buf, &fft); } });
    // B: shortint
    let sct: Vec<_> = (0..4).map(|i| cks.encrypt((i as u64 * 5) % 4)).collect();
    let t_b = best(reps, || { for c in &sct { let _ = sks.apply_lookup_table(c, &lut); } });
    // A: high level
    let (hck, hsk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(hsk);
    let a = FheUint8::encrypt(0xB5u8, &hck);
    let t_a = best(reps, || { let _ = &a & 0xFFu8; });
    println!("  FheUint8 refresh (4 blocks): A high-level {t_a:.3} ms, B shortint {t_b:.3} ms, C core with the key switch of TFHE-rs {t_c:.3} ms, D core with our key switch {t_d:.3} ms");
    csv.push(format!("refresh8,A,{t_a:.4}")); csv.push(format!("refresh8,B,{t_b:.4}"));
    csv.push(format!("refresh8,C,{t_c:.4}")); csv.push(format!("refresh8,D,{t_d:.4}"));
    csv.push(format!("ksnoise,tfhers,{sc:.3}")); csv.push(format!("ksnoise,ours,{sd:.3}"));
    println!("CSV (item,variant,value)"); for l in csv { println!("{l}"); }
}
