//! A/B timing of the blind rotation, one thread, same configurations (CONFIGS_RS) and same rounded exponents:
//! classic CMux rotation of TFHE-rs (m = 1) against our multi-bit rotation for m = 2 on the FFT of TFHE-rs (eb::Own2)
//! with its three kernels for the products (eb::OWN_KERNEL: 0 = one pattern at a time, fused; 1 = row by row as
//! TFHE-rs, then the monomial; 2 = the three patterns at once), interleaved, minimum over REPS runs (default 20);
//! check that all rotate the test polynomial to the same message; time per pair of the steps of the multi-bit
//! rotation (decomposition, forward FFTs, monomials, products, backward FFTs) for each kernel.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin rot_ab        (KS=1,4,6,8 by default)
use eb_tfhers::eb::{EbKeys, OPROF, OWN_KERNEL};
use eb_tfhers::params::*;
use eb_tfhers::rounding::Rounded;
use rand::{Rng, SeedableRng};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

fn main() {
    std::env::set_var("OWNPROF", "1");
    let ks: Vec<u32> = std::env::var("KS").unwrap_or("1,4,6,8".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    println!("Blind rotation, one thread: CMux of TFHE-rs (m = 1) / our multi-bit rotation (m = 2, kernels 0, 1, 2), min over {reps} runs");
    let mut csv = Vec::new();
    for cfg in CONFIGS_RS.iter().filter(|c| ks.contains(&c.k)) {
        let (k1, _) = EbKeys::new(cfg, 1, 5); let (k2, _) = EbKeys::new(cfg, 2, 5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3 + cfg.k as u64);
        let r = Rounded { abar: (0..cfg.n).map(|_| rng.gen_range(0..2 * N as u64)).collect(), bbar: rng.gen_range(0..2 * N as u64), cbar: 0, trials: 0, v: 0.0 };
        let ph = |acc: &GlweCiphertextOwned<u64>, k: &EbKeys| {
            let mut o = LweCiphertext::new(0u64, LweSize(N + 1), CiphertextModulus::new_native());
            extract_lwe_sample_from_glwe_ciphertext(acc, &mut o, MonomialDegree(0));
            decrypt_lwe_ciphertext(&k.big_sk, &o).0
        };
        let p1 = ph(&k1.rotate_st(&r), &k1);
        let mut dmax = 0f64;
        for kern in 0..3 {
            OWN_KERNEL.store(kern, Relaxed);
            let d = (p1.wrapping_sub(ph(&k2.rotate_st(&r), &k2))) as i64 as f64 * 2f64.powi(-64);
            assert!(d.abs() < 2f64.powi(-12), "different rotations (kernel {kern})");
            dmax = dmax.max(d.abs());
        }
        let mut t = [f64::INFINITY; 4];
        for _ in 0..reps {
            let s = Instant::now(); let _ = k1.rotate_st(&r); t[0] = t[0].min(s.elapsed().as_secs_f64() * 1e3);
            for kern in 0..3 {
                OWN_KERNEL.store(kern, Relaxed);
                let s = Instant::now(); let _ = k2.rotate_st(&r); t[1 + kern] = t[1 + kern].min(s.elapsed().as_secs_f64() * 1e3);
            }
        }
        let us = |x: f64| x * 1e3 / cfg.n as f64;
        println!("  k = {} n = {:4} (2^{},{}): m = 1 {:7.3} ms ({:5.2} us/coord); m = 2: kernel 0 {:7.3} ms ({:5.2}), kernel 1 {:7.3} ms ({:5.2}), kernel 2 {:7.3} ms ({:5.2}); best ratio {:.3}   |message diff| <= 2^{:.1}",
                 cfg.k, cfg.n, cfg.b, cfg.l, t[0], us(t[0]), t[1], us(t[1]), t[2], us(t[2]), t[3], us(t[3]),
                 t[0] / t[1].min(t[2]).min(t[3]), dmax.log2());
        let mut line = format!("rot,{},{},{},{},{:.4},{:.4},{:.4},{:.4}", cfg.k, cfg.n, cfg.b, cfg.l, t[0], t[1], t[2], t[3]);
        for kern in 0..3 {
            OWN_KERNEL.store(kern, Relaxed);
            for p in OPROF.iter() { p.store(0, Relaxed); }
            let pr_reps = 5;
            for _ in 0..pr_reps { let _ = k2.rotate_st(&r); }
            let pr: Vec<f64> = OPROF.iter().map(|x| x.load(Relaxed) as f64 / 1e3 / pr_reps as f64 / (cfg.n / 2) as f64).collect();
            println!("    kernel {kern}, us per pair (mean): decomposition {:.2}, forward FFTs {:.2}, monomials {:.2}, products {:.2}, backward FFTs {:.2}",
                     pr[0], pr[1], pr[2], pr[3], pr[4]);
            line += &format!(",{:.3}", pr[3]);
        }
        csv.push(line);
    }
    println!("CSV (rot,k,n,logB,l,ms m=1,ms m=2 kernels 0 1 2,us per pair of the products for kernels 0 1 2)");
    for l in csv { println!("{l}"); }
}
