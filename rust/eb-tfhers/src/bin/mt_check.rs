//! Checks of the multi-threaded versions: the rotation of the error bootstrap on t threads against the rotation of
//! TFHE-rs (same rounded ciphertext), and the refresh of B ciphertexts by CombBoot on t threads (decryption).
//! Run: cargo run --release --bin mt_check    (THREADS=1,2,3)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::{CONFIGS_RS, N};
use eb_tfhers::rounding::round_select;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tfhe::core_crypto::prelude::*;
fn main() {
    let ths: Vec<usize> = std::env::var("THREADS").unwrap_or("1,2,3".into()).split(',').map(|x| x.parse().unwrap()).collect();
    for cfg in CONFIGS_RS.iter().filter(|c| [1, 4, 8].contains(&c.k)) {
        let (keys, mut egen) = EbKeys::new(cfg, 1, 40 + cfg.k as u64);
        let mut rng = StdRng::seed_from_u64(5); let mut rbit = 12345u64;
        let lp = 2 * cfg.k; let mu = rng.gen_range(0..(1u64 << lp)) << (64 - lp);
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen);
        let (a, b) = keys.stair.as_ref().unwrap().switch(ct.get_mask().as_ref(), *ct.get_body().data, &mut rbit);
        let r = round_select(&a, b, keys.log_r, &keys.pool, keys.vstar, &mut rng);
        let s1 = keys.rotate_st(&r);
        let o1 = keys.finish(&r, &s1);
        let e1 = tor(decrypt_lwe_ciphertext(&keys.big_sk, &o1).0.wrapping_sub(mu));
        for &t in &ths {
            let s2 = keys.rotate_mt(&r, t);
            let o2 = keys.finish(&r, &s2);
            let e2 = tor(decrypt_lwe_ciphertext(&keys.big_sk, &o2).0.wrapping_sub(mu));
            let dc = s1.as_ref().iter().zip(s2.as_ref()).map(|(&x, &y)| tor(x.wrapping_sub(y)).abs()).fold(0.0, f64::max);
            println!("EB k = {} ({:2} bits), {t} threads: error 2^{:.2} (TFHE-rs rotation 2^{:.2}), max |coef MT - TFHE-rs| = 2^{:.1}",
                     cfg.k, lp, e2.abs().log2(), e1.abs().log2(), dc.log2());
        }
    }
    for cfg in COMB_CONFIGS.iter().filter(|c| [2, 8].contains(&c.bits)) {
        let keys = CombKeys::new_m(cfg, 1, 70 + cfg.bits as u64);   // combboot_mt: m = 1
        let mut rng = StdRng::seed_from_u64(6);
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(new_seeder().as_mut().seed(), new_seeder().as_mut());
        let lp = cfg.bits; let bc = keys.bcols;
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let mus: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..(1u64 << lp)) << (64 - lp)).collect();
        let cts: Vec<LweCiphertextOwned<u64>> = mus.iter().map(|&mu| allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen)).collect();
        let p = cfg.p as u64;
        let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
        let bints: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..p)).collect();
        // one-thread reference with the order of combboot_mt: by automorphism, unpadded arrays (KS_MODE 0, KS_PAD false)
        eb_tfhers::comb::KS_MODE.store(0, std::sync::atomic::Ordering::Relaxed);
        eb_tfhers::comb::KS_PAD.store(false, std::sync::atomic::Ordering::Relaxed);
        let st = keys.combboot(&aints, &bints);
        for &t in &ths {
            let mt = keys.combboot_mt(&aints, &bints, t);
            let d = st.iter().zip(&mt).map(|(x, y)| x.as_ref().iter().zip(y.as_ref()).map(|(&u, &v)| tor(u.wrapping_sub(v)).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            println!("CombBoot {:2} bits, {t} threads: max |coef MT - single-thread| = 2^{:.1}", lp, d.log2());
        }
        for &t in &ths {
            let outs = keys.refresh_mt(&cts, 9, t);
            let emax = outs.iter().zip(&mus).map(|(o, &mu)| tor(decrypt_lwe_ciphertext(&keys.big_sk, o).0.wrapping_sub(mu)).abs()).fold(0.0, f64::max);
            println!("CombBoot {:2} bits, {t} threads: max error 2^{:.2} over {bc} ciphertexts (window 2^{:.0})", lp, emax.log2(), -(lp as f64) - 2.0);
        }
    }
    let _ = N;
}
