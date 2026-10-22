//! A/B timing of the comb bootstrap, one thread, same configurations (COMB_CONFIGS) and same inputs: grouping m = 1
//! (CMux of TFHE-rs) against m = 2 (shared key switchings, three patterns, our external products on the FFT of
//! TFHE-rs), interleaved, minimum over REPS calls (default 6); check that both give the same errors E_k/R; time of each
//! phase (ms per call, mean): digits and forward FFTs, tables, key switchings and public products, backward FFTs of
//! C_L, CMux / external products.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin comb_ab        (BITS=2,4,8,12,16)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::tor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

fn main() {
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("2,4,8,12,16".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    println!("Comb bootstrap, one thread: m = 1 (CMux of TFHE-rs) / m = 2, ms per call (B errors), min over {reps} calls");
    let mut csv = Vec::new();
    for cfg in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let k1 = CombKeys::new_m(cfg, 1, 31 + cfg.bits as u64); let k2 = CombKeys::new_m(cfg, 2, 31 + cfg.bits as u64);
        let mut rng = StdRng::seed_from_u64(cfg.bits as u64);
        let (p, bc) = (cfg.p as u64, k1.bcols);
        let s = k1.small_sk.as_ref().to_vec();
        assert_eq!(s, k2.small_sk.as_ref().to_vec());
        let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
        let e: Vec<i64> = (0..bc).map(|_| rng.gen_range(-(p as i64) / 4..=(p as i64) / 4)).collect();
        let bints: Vec<u64> = (0..bc).map(|k| {
            let dot = aints[k].iter().zip(&s).fold(0u64, |acc, (&a, &x)| acc.wrapping_add(a * x));
            ((dot as i64 + e[k]).rem_euclid(p as i64)) as u64
        }).collect();
        let r = 2f64.powi(k1.log_r as i32);
        let mut emax = [0f64; 2];
        for (i, k) in [&k1, &k2].iter().enumerate() {
            let outs = k.combboot(&aints, &bints);
            for kk in 0..bc {
                let want = ((e[kk] as f64) / r * 2f64.powi(64)).round() as i64 as u64;
                let err = tor(decrypt_lwe_ciphertext(&k.big_sk, &outs[kk]).0.wrapping_sub(want));
                emax[i] = emax[i].max(err.abs());
            }
            assert!(emax[i] < 1.0 / (4.0 * 2f64.powi(cfg.bits as i32)), "wrong errors (m = {})", i + 1);
        }
        let (mut t1, mut t2) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let t = Instant::now(); let _ = k1.combboot(&aints, &bints); t1 = t1.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now(); let _ = k2.combboot(&aints, &bints); t2 = t2.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let (mut p1, mut p2) = ([0f64; 5], [0f64; 5]);
        for _ in 0..3 { let _ = k1.combboot_prof(&aints, &bints, Some(&mut p1)); let _ = k2.combboot_prof(&aints, &bints, Some(&mut p2)); }
        let f = |x: [f64; 5]| x.iter().map(|v| format!("{:.1}", v / 3.0)).collect::<Vec<_>>().join(", ");
        println!("  {:2} bits, p = {}, B = {:2}, n = {:4}: m = 1 {:8.2} ms ({:.2} per error), m = 2 {:8.2} ms ({:.2} per error), ratio {:.3}   max |e| 2^{:.1} / 2^{:.1}",
                 cfg.bits, cfg.p, bc, cfg.n, t1, t1 / bc as f64, t2, t2 / bc as f64, t1 / t2, emax[0].log2(), emax[1].log2());
        println!("      phases (ms per call): m = 1 [{}]   m = 2 [{}]", f(p1), f(p2));
        csv.push(format!("comb_ab,{},{:.4},{:.4}", cfg.bits, t1 / bc as f64, t2 / bc as f64));
    }
    println!("CSV (comb_ab,bits,ms per error m=1,ms per error m=2)"); for l in csv { println!("{l}"); }
}
