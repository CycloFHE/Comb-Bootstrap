//! Noise of the comb bootstrap on TFHE-rs (h = 1, grouping m of COMBM, default 1; MT=t: version on t threads, m = 1):
//! NE errors (default 1600), inputs noise-free LWE modulo p with E_k uniform in the window |E_k| ≤ p/4 (random_inputs
//! of julia/src/algorithm.jl), output errors against E_k/R, and the prediction of Proposition 1 (var_mbs of
//! julia/benchmarks/comb_sizes.jl: (n/m)(2^m - 1) external products, the rounding and automorphism terms for the active
//! groups) with the error of the FFT of the external products of TFHE-rs (2^-87.5 B^2 ℓ each) and, for comparison, with
//! the error of the FFT of the automorphism key switchings fitted in Julia (2^-88.5 B^2 ℓ).
//! Run:  cargo run --release --bin comb_noise     (BITS=2,4,8,12,16; NE=...; QUICK=1: n = 64, test only)
use eb_tfhers::comb::{CombConfig, CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::tor;
use eb_tfhers::params::{stair_var, LOG2_SIGMA_N, N, NU, U128};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tfhe::core_crypto::prelude::*;

fn main() {
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("2,4,8,12,16".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let ne: usize = std::env::var("NE").ok().and_then(|v| v.parse().ok()).unwrap_or(1600);
    let quick = std::env::var("QUICK").map(|v| v == "1").unwrap_or(false);
    let mt: usize = std::env::var("MT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    println!("Comb bootstrap on TFHE-rs 1.8.1 (h = 1, m = {}): output noise, log2 σ measured / predicted{}", if mt > 0 { 1 } else { eb_tfhers::comb::comb_m() }, if quick { "  (QUICK: n = 64)" } else { "" });
    for c0 in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let qc = CombConfig { bits: c0.bits, p: c0.p, gbr: c0.gbr, gks: c0.gks, n: 64, vterm: c0.vterm, chain: &[(64, 1, 30)] };
        let cfg = if quick { &qc } else { c0 };
        let t0 = std::time::Instant::now();
        let keys = if mt > 0 { CombKeys::new_m(cfg, 1, 500 + cfg.bits as u64) } else { CombKeys::new(cfg, 500 + cfg.bits as u64) };
        let tk = t0.elapsed().as_secs_f64();
        let mut rng = StdRng::seed_from_u64(cfg.bits as u64);
        let (p, bc) = (cfg.p as u64, keys.bcols);
        let s = keys.small_sk.as_ref().to_vec();
        // external products and active groups (groups of m coordinates not all zero; an odd last coordinate alone)
        let (nep, n1) = if keys.m == 1 { (cfg.n, s.iter().filter(|&&x| x == 1).count()) } else {
            let act = s.chunks(2).filter(|g| g.iter().any(|&x| x == 1)).count();
            (3 * (cfg.n / 2) + cfg.n % 2, act)
        };
        let r = 2f64.powi(keys.log_r as i32);
        let (mut s2, mut cnt, mut emax) = (0.0, 0usize, 0.0f64);
        let t1 = std::time::Instant::now(); let mut calls = 0;
        while cnt < ne {
            let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
            let e: Vec<i64> = (0..bc).map(|_| rng.gen_range(-(p as i64) / 4..=(p as i64) / 4)).collect();
            let bints: Vec<u64> = (0..bc).map(|k| {
                let dot = aints[k].iter().zip(&s).fold(0u64, |acc, (&a, &x)| acc.wrapping_add(a * x));
                ((dot as i64 + e[k]).rem_euclid(p as i64)) as u64
            }).collect();
            let outs = if mt > 0 { keys.combboot_mt(&aints, &bints, mt) } else { keys.combboot(&aints, &bints) }; calls += 1;
            for k in 0..bc {
                let ph = decrypt_lwe_ciphertext(&keys.big_sk, &outs[k]).0;
                let want = ((e[k] as f64) / r * 2f64.powi(64)).round() as i64 as u64;
                let err = tor(ph.wrapping_sub(want)); s2 += err * err; emax = emax.max(err.abs()); cnt += 1;
            }
        }
        let tc = t1.elapsed().as_secs_f64() / calls as f64;
        let vo = s2 / cnt as f64;
        let sig2 = 2f64.powf(2.0 * LOG2_SIGMA_N); let nn = N as f64;
        let (b, l) = (cfg.gbr.0 as i32, cfg.gbr.1 as f64);
        let key = 2.0 * l * nn * (2f64.powi(2 * b) + 2.0) / 12.0 * sig2;
        let rnd = (1.0 + nn / 2.0) * 2f64.powf(-2.0 * l * b as f64) / 12.0;
        let (gb, gl, go) = cfg.gks; let gbf = gb as i32; let glf = gl as f64;
        let vks = glf * nn * (2f64.powi(2 * gbf) + 2.0) / 12.0 * sig2 + (nn / 2.0) * 2f64.powf(-2.0 * (glf * gb as f64 + go as f64)) / 12.0;
        let bb = bc as f64; let n = nep as f64; let n1f = n1 as f64;
        let pred = n * key + n1f * (rnd + (bb - 1.0) * bb * vks) + n * 2f64.powf(-87.5) * 2f64.powi(2 * b) * l;
        let pred_aut = pred + n1f * (bb - 1.0) * bb * 2f64.powf(-88.5) * 2f64.powi(2 * gbf) * glf;
        let vks_ks = stair_var(N, cfg.chain, 16); let phat = 2f64.powi(cfg.bits as i32);
        let marg = 0.5 * ((1.0 / (U128 * U128)) / ((4.0 * phat).powi(2) * (NU * vo + vks_ks) + cfg.vterm)).log2();
        println!("{:2} bits: p = {}, B = {}, n = {}, RGSW (2^{},{}), aut (2^{},{}) shift {}: out {:7.2} / {:7.2} (with aut FFT 2^-88.5: {:7.2}), max |e| 2^{:.2}, window 2^{:.0}, margin {:.2} bits (KS predicted {:.2})   comb {:.1} ms per call ({:.2} ms per error), keygen {:.0} s",
                 cfg.bits, cfg.p, bc, cfg.n, cfg.gbr.0, cfg.gbr.1, gb, gl, go, 0.5 * vo.log2(), 0.5 * pred.log2(), 0.5 * pred_aut.log2(),
                 emax.log2(), -(cfg.bits as f64) - 2.0, marg, 0.5 * vks_ks.log2(), tc * 1e3, tc * 1e3 / bb, tk);
    }
}
