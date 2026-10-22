//! Calibration of the error of the FFT of TFHE-rs in the rotation of the error bootstrap: output noise of
//! refresh_small (no key switch) for several gadgets, m = 1 (classic rotation) and m = 2 (multi-bit), at n = NN
//! (default 800), against the prediction without FFT error; the excess is written as (n/m) c B^2 ℓ (c per m, log2).
//! Also the time of one rotation (min over 5 runs). Run:  cargo run --release --bin eb_fft    (NREF, NN, GADS="b:l,...")
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

fn main() {
    let nref: usize = std::env::var("NREF").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let n: usize = std::env::var("NN").ok().and_then(|v| v.parse().ok()).unwrap_or(800);
    let ms: Vec<usize> = std::env::var("MS").unwrap_or("1,2".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let gads: Vec<(usize, usize)> = std::env::var("GADS").unwrap_or("23:1,22:1,20:1,18:1,16:2,15:2,14:2,13:2,12:3,11:3,10:3,9:4,8:5".into())
        .split(',').map(|g| { let v: Vec<usize> = g.split(':').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    println!("FFT of TFHE-rs: output of the rotation, n = {n}, k = 4 (p̂ = 2^8), {nref} refreshes, log2 σ measured / predicted without FFT");
    for &m in &ms { for &(b, l) in &gads {
        let cfg = Config { k: 4, n, b, l, vterm: 1e9, chain: &[] };
        let (keys, mut egen) = EbKeys::new(&cfg, m, 7 + (b * 10 + l) as u64);
        let mut rng = StdRng::seed_from_u64(b as u64 * 31 + l as u64);
        let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-20)), 0.0);
        let mut s2 = 0.0; let lp = 8;
        let mut tmin = f64::INFINITY;
        for i in 0..nref {
            let mu = (rng.gen_range(0..256u64)) << (64 - lp);
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.small_sk, Plaintext(mu), enc, cm(), &mut egen);
            let t0 = Instant::now();
            let out = keys.refresh_small(ct.get_mask().as_ref(), *ct.get_body().data, &mut rng, None);
            if i < 6 && i > 0 { tmin = tmin.min(t0.elapsed().as_secs_f64() * 1e3); }
            let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu)); s2 += e * e;
        }
        let vo = s2 / nref as f64; let pv = var_rot(n, m, b, l);
        let g = n as f64 / m as f64; let bb = 2f64.powi(2 * b as i32) * l as f64;
        let c = if vo > pv { ((vo - pv) / (g * bb)).log2() } else { f64::NAN };
        println!("m = {m}, (2^{b:2},{l}): out {:7.2} / {:7.2}   FFT: log2 c = {:7.2} (FFTW model -85)   rotation {:7.3} ms",
                 0.5 * vo.log2(), 0.5 * pv.log2(), c, tmin);
        println!("CSV,{m},{b},{l},{:.4},{:.4},{:.3},{:.4}", 0.5 * vo.log2(), 0.5 * pv.log2(), c, tmin);
    } }
}
