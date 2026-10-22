//! Tail of the error at the bootstrap (review C4): the failure probability 2^-128 rests on a Gaussian law of
//! chi = e_in + e_KS - T/R (Section 4.3), with the error of the FFT taken as an independent Gaussian term. We test
//! this law in the least favorable case, nu = 1: the input of each bootstrap is one fresh output of the error
//! bootstrap (no sum of outputs to make it Gaussian), key-switched, rounded with recentering and selection.
//! For each sample: a fresh encryption of a random mu (p_hat = 4^k) under s' is refreshed (configuration of the paper,
//! CONFIGS_RS2, rotation of TFHE-rs or Own2); the output is key-switched and rounded as for the next bootstrap, and
//! the integer error D = (b + c - a.s) - R mu mod R (centered), which the window |D| < p/4 bounds, is normalized by
//! its predicted standard deviation,
//!     sigma_D^2 = R^2 (sigma_out^2 + sigma_KS^2 [+ 2 sigma_128(n)^2 if selected]) + V_sel + 1/12,
//! sigma_out^2 from the model with the error of the FFT (2^-87.5 B^2 l per CMux, 2^-84.5 per pair for m = 2),
//! sigma_KS^2 from the model of the stair key switch, V_sel the rounding variance of the selected mask.
//! Prints the empirical standard deviation of z = D / sigma_D and the counts of |z| >= t against the Gaussian
//! expectation N 2Q(t), for t = 2 to 5.5, also with z normalized by its empirical standard deviation (shape alone);
//! the same for the output error alone, z_out = e_out / sigma_out (the term that carries the error of the FFT, which
//! the key switch dominates in D at small p_hat).
//! Run:  RAYON_NUM_THREADS=1 KS=1,4 NS=1000000 THREADS=8 cargo run --release --bin eb_tail | tee eb_tail_mac.txt
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::{config_rs2, log2_sigma128, stair_var, var_rot, N};
use eb_tfhers::rounding::round_select;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

fn envs(name: &str, d: &str) -> String { std::env::var(name).unwrap_or(d.into()) }
fn envu(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// Gaussian tail Q(x) = P(G > x), by erfc (Abramowitz-Stegun 7.1.26 is too coarse here: continued fraction).
fn q(x: f64) -> f64 { 0.5 * erfc(x / std::f64::consts::SQRT_2) }
fn erfc(x: f64) -> f64 {
    // complementary error function, Numerical Recipes erfcc (relative error < 1.2e-7)
    let z = x.abs(); let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807
        + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

fn main() {
    let ks: Vec<u32> = envs("KS", "1,4").split(',').map(|x| x.parse().unwrap()).collect();
    let ns = envu("NS", 20000);
    let threads = envu("THREADS", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1));
    let ts = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];
    println!("Tail of the error at the bootstrap, nu = 1 (one fresh output, key switch, rounding with recentering and selection); {ns} samples per k, {threads} threads");
    for &k in &ks {
        let (cfg, m) = config_rs2(k);
        let t0 = Instant::now();
        let (keys, _) = EbKeys::new(cfg, m, 4242 + k as u64);
        let lp = 2 * k; let log_r = keys.log_r; let rr = 1u128 << log_r;
        let cfft = if m == 1 { -87.5 } else { -84.5 };
        let vout = var_rot(cfg.n, m, cfg.b, cfg.l) + (cfg.n as f64 / m as f64) * 2f64.powf(cfft) * 2f64.powi(2 * cfg.b as i32) * cfg.l as f64;
        let vks = stair_var(N, cfg.chain, 16);
        let vsel = 2.0 * 2f64.powf(2.0 * log2_sigma128(cfg.n));
        let r2 = (rr as f64).powi(2);
        let per = ns / threads;
        let res: Vec<(Vec<f32>, f64, f64, Vec<f32>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads).map(|w| {
                let keys = &keys;
                sc.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(0x7a11 + 1000 * k as u64 + w as u64);
                    let mut boxed = new_seeder(); let seeder = boxed.as_mut();
                    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
                    let mut rbit = 0x9e3779b97f4a7c15u64 ^ (w as u64 + 1);
                    let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(N))), 0.0);
                    let s = keys.small_sk.as_ref();
                    let (mut zs, mut so2, mut sd2, mut zo) = (Vec::with_capacity(per), 0.0, 0.0, Vec::with_capacity(per));
                    for _ in 0..per {
                        let msg: u64 = rng.gen_range(0..(1u64 << lp)); let mu = msg << (64 - lp);
                        let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), enc, cm(), &mut egen);
                        let out = keys.refresh(&ct, &mut rng, &mut rbit, None);
                        let eo = tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu)); so2 += eo * eo; zo.push((eo / vout.sqrt()) as f32);
                        // next bootstrap: key switch, rounding with recentering and selection
                        let (a, b) = keys.stair.as_ref().unwrap().switch(out.get_mask().as_ref(), *out.get_body().data, &mut rbit);
                        let r = round_select(&a, b, log_r, &keys.pool, keys.vstar, &mut rng);
                        let dot = r.abar.iter().zip(s).fold(0u128, |acc, (&x, &y)| (acc + x as u128 * y as u128) % rr);
                        let rmu = (msg as u128) << (log_r - lp);
                        let d = ((r.bbar as u128 + 2 * rr - dot - rmu) % rr) as i128;
                        let d = if d > (rr / 2) as i128 { d - rr as i128 } else { d } as f64;
                        sd2 += d * d;
                        let pv = r2 * (vout + vks + if r.trials > 0 { vsel } else { 0.0 }) + r.v + 1.0 / 12.0;
                        zs.push((d / pv.sqrt()) as f32);
                    }
                    (zs, so2, sd2, zo)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let z: Vec<f64> = res.iter().flat_map(|r| r.0.iter().map(|&x| x as f64)).collect();
        let n = z.len() as f64;
        let so = (res.iter().map(|r| r.1).sum::<f64>() / n).sqrt();
        let sd = (res.iter().map(|r| r.2).sum::<f64>() / n).sqrt();
        let mz = z.iter().sum::<f64>() / n; let sz = (z.iter().map(|x| (x - mz) * (x - mz)).sum::<f64>() / n).sqrt();
        let kurt = z.iter().map(|x| ((x - mz) / sz).powi(4)).sum::<f64>() / n;
        let window = (1u128 << (log_r - lp)) as f64 / 4.0; // p/4
        println!("k = {k} (p_hat = 2^{lp}), m = {m}, n = {}, (2^{},{}): sigma_out 2^{:.2} (predicted 2^{:.2}), sigma_D/R 2^{:.2}, window/sigma_D {:.1}; z: mean {:.4}, std {:.4}, kurtosis {:.3} ({:.0} s)",
                 cfg.n, cfg.b, cfg.l, so.log2(), 0.5 * vout.log2(), (sd / rr as f64).log2(), window / sd, mz, sz, kurt, t0.elapsed().as_secs_f64());
        let tail = |z: &[f64], label: &str| {
            let n = z.len() as f64;
            let mz = z.iter().sum::<f64>() / n; let sz = (z.iter().map(|x| (x - mz) * (x - mz)).sum::<f64>() / n).sqrt();
            let kurt = z.iter().map(|x| ((x - mz) / sz).powi(4)).sum::<f64>() / n;
            println!("   {label}: mean {mz:.4}, std {sz:.4}, kurtosis {kurt:.3}");
            println!("   t    |z|>=t observed / expected (model)    |z/std|>=t observed / expected");
            for &t in &ts {
                let o1 = z.iter().filter(|x| x.abs() >= t).count();
                let o2 = z.iter().filter(|x| ((*x - mz) / sz).abs() >= t).count();
                let e = n * 2.0 * q(t);
                println!("  {t:3.1}   {o1:8} / {e:10.1}  (ratio {:.2})        {o2:8} / {e:10.1}  (ratio {:.2})", o1 as f64 / e, o2 as f64 / e);
                println!("CSV,{label},{k},{t},{o1},{o2},{e:.3},{n}");
            }
        };
        tail(&z, "D (bootstrap input)");
        let zo: Vec<f64> = res.iter().flat_map(|r| r.3.iter().map(|&x| x as f64)).collect();
        tail(&zo, "e_out (output of the refresh)");
    }
}
