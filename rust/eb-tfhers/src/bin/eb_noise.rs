//! Noise of the error bootstrap on TFHE-rs (FFT of TFHE-rs), for the configurations R of Figure 4:
//! output of the refresh (inputs under s, NREF refreshes, default 500), stair key switch (400 samples), predictions,
//! and margin of the condition (4p̂)^2 (ν σ_out^2 + σ_KS^2) + 16 V*/(2N)^2 ≤ 1/U^2, ν = 1000, U = u_128.
//! Run:  cargo run --release --bin eb_noise        (KS=1,2,...: some k only; NREF=...)
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tfhe::core_crypto::prelude::*;

fn main() {
    let ks: Vec<u32> = std::env::var("KS").unwrap_or("1,2,3,4,5,6,7,8".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let nref: usize = std::env::var("NREF").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    println!("Error bootstrap on TFHE-rs 1.8.1 (FFT of TFHE-rs; m = 1: its CMux, m = 2: our multi-bit rotation; CFG=rs2 (default), rs, jl): noise, log2 σ measured / predicted (without / with the error of the FFT)");
    for (cfg, mm) in configs().into_iter().filter(|(c, _)| ks.contains(&c.k)) {
        let t0 = std::time::Instant::now();
        let (keys, mut egen) = EbKeys::new(cfg, mm, 1000 + cfg.k as u64);
        let mut rng = StdRng::seed_from_u64(cfg.k as u64);
        let mut rbit = 0x9e3779b97f4a7c15u64;
        let lp = 2 * cfg.k; let window = 1.0 / (4.0 * 2f64.powi(lp as i32));
        let enc_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-20)), 0.0);
        let (mut s2, mut emax, mut fails) = (0.0, 0.0f64, 0usize);
        for _ in 0..nref {
            let m: u64 = rng.gen_range(0..(1u64 << lp)); let mu = m << (64 - lp);
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.small_sk, Plaintext(mu), enc_noise, cm(), &mut egen);
            let out = keys.refresh_small(ct.get_mask().as_ref(), *ct.get_body().data, &mut rng, None);
            let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu));
            s2 += e * e; emax = emax.max(e.abs()); if e.abs() >= window { fails += 1; }
        }
        let vo = s2 / nref as f64;
        let tiny = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-60)), 0.0);
        let mut k2 = 0.0; let nks = 400;
        for _ in 0..nks {
            let mu: u64 = rng.gen();
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), tiny, cm(), &mut egen);
            let (a, b) = keys.stair.as_ref().unwrap().switch(ct.get_mask().as_ref(), *ct.get_body().data, &mut rbit);
            let dot = a.iter().zip(keys.small_sk.as_ref()).fold(0u64, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y)));
            let e = tor(b.wrapping_sub(dot).wrapping_sub(mu)); k2 += e * e;
        }
        let vks = k2 / nks as f64;
        // error of the FFT: 2^-87 B^2 ℓ per CMux of TFHE-rs (m = 1), 2^-84.5 B^2 ℓ per pair for our multi-bit rotation
        let cfft = if mm == 1 { -87.0 } else { -84.5 };
        let pv = var_rot(cfg.n, mm, cfg.b, cfg.l); let pfftw = pv + (cfg.n as f64 / mm as f64) * 2f64.powf(cfft) * 2f64.powi(2 * cfg.b as i32) * cfg.l as f64;
        println!("k = {} (p̂ = 2^{:2}), m = {}, n = {:4}, (2^{},{}): out {:7.2} / {:7.2} / {:7.2}, max |e| 2^{:.2}, failures {}   KS {:7.2} / {:7.2}   margin {:5.2} bits   ({:.0} s)",
                 cfg.k, lp, mm, cfg.n, cfg.b, cfg.l, 0.5 * vo.log2(), 0.5 * pv.log2(), 0.5 * pfftw.log2(), emax.log2(), fails,
                 0.5 * vks.log2(), 0.5 * stair_var(N, cfg.chain, 16).log2(), margin(cfg.k, vo, vks, cfg.vterm), t0.elapsed().as_secs_f64());
    }
}
