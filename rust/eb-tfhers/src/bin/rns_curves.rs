//! RNS: a 64-bit integer x held as its 8 residues x mod p̂_k (comb::RNS_MODULI = 263, 262, 261, 259, 257, 253, 251, 247,
//! product 2^64.03), residue k encrypted as LWE(r_k/p̂_k) of dimension N, the 8 refreshed together by one CombBoot with
//! B = 8 (comb::COMB_RNS, case (B), p = 512; refresh_rns: stair key switch, rounding at the resolution p p̂_k with
//! selection, the test polynomial of column k for p̂_k, subtraction), against the refresh of FheUint64 by TFHE-rs with its
//! default parameters (`&a & u64::MAX`, one PBS per block of 2 bits), one thread, same session.
//!   check:  CHECK random integers, inputs with noise 2^-30, decrypted and recombined by the CRT;
//!   time:   minimum over REPS refreshes (default 5) after one warm-up, phases (key switches, roundings, CombBoot);
//!   noise:  NE output errors (default 1600) of combboot_rns on noise-free inputs modulo p with E_k uniform in |E_k| ≤ p/4,
//!           against E_k/(p p̂_k), and the margin at failure 2^-128 after 1000 additions for p̂ = 263 (as comb_noise).
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin rns_curves      (REPS=5, NE=1600, CHECK=20, TF=0: no TFHE-rs)
use eb_tfhers::comb::{cm, CombKeys, COMB_RNS, RNS_MODULI};
use eb_tfhers::eb::tor;
use eb_tfhers::params::{stair_var, LOG2_SIGMA_N, N, NU, U128};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint64};

fn env_usize(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// x from its residues modulo the moduli (pairwise coprime), by Garner's mixed radix, modulo the product (< 2^65).
fn crt(res: &[u64], m: &[u64]) -> u128 {
    let inv = |a: u64, md: u64| -> u64 { (1..md).find(|&y| (a % md) * y % md == 1).unwrap() };
    let k = m.len(); let mut v = vec![0u64; k];
    for i in 0..k {
        let mut t = res[i] % m[i];
        for j in 0..i { t = (t + m[i] - v[j] % m[i]) % m[i] * inv(m[j], m[i]) % m[i]; }
        v[i] = t;
    }
    let (mut x, mut w) = (0u128, 1u128);
    for i in 0..k { x += v[i] as u128 * w; w *= m[i] as u128; }
    x
}

fn main() {
    let reps = env_usize("REPS", 5); let ne = env_usize("NE", 1600); let nchk = env_usize("CHECK", 20);
    let tf = std::env::var("TF").map(|v| v != "0").unwrap_or(true);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let cfg = &COMB_RNS; let mods: Vec<u64> = RNS_MODULI.to_vec();
    let prod: f64 = mods.iter().map(|&m| (m as f64).log2()).sum();
    println!("RNS on TFHE-rs 1.8.1, one thread: moduli {:?} (product 2^{:.2}); CombBoot B = 8, p = {}, n = {}, RGSW (2^{},{}), aut (2^{},{}) shift {}",
             mods, prod, cfg.p, cfg.n, cfg.gbr.0, cfg.gbr.1, cfg.gks.0, cfg.gks.1, cfg.gks.2);
    let t0 = Instant::now();
    let keys = CombKeys::new_m(cfg, 1, 7001);
    let tk = t0.elapsed().as_secs_f64();
    assert_eq!(keys.bcols, mods.len());
    let mut rng = StdRng::seed_from_u64(64); let mut rbit = 0x853c49e6748fea9bu64;
    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(new_seeder().as_mut().seed(), new_seeder().as_mut());
    let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
    let enc = |x: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Vec<LweCiphertextOwned<u64>> {
        mods.iter().map(|&m| {
            let mu = ((((x % m) as u128) << 64) / m as u128) as u64;          // r/p̂ on the torus (rounded down: < 2^-64)
            allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), egen)
        }).collect()
    };
    // check: refresh and recombination of random 64-bit integers
    let mut emax = 0f64;
    for _ in 0..nchk {
        let x: u64 = rng.gen();
        let (outs, _) = keys.refresh_rns(&enc(x, &mut egen), &mods, &mut rng, &mut rbit);
        let res: Vec<u64> = outs.iter().zip(&mods).map(|(o, &m)| {
            let ph = decrypt_lwe_ciphertext(&keys.big_sk, o).0;
            let r = (((ph as u128) * m as u128 + (1u128 << 63)) >> 64) as u64 % m;
            let want = ((((x % m) as u128) << 64) / m as u128) as u64;
            emax = emax.max(tor(ph.wrapping_sub(want)).abs());
            r
        }).collect();
        assert_eq!(crt(&res, &mods), x as u128, "wrong RNS refresh");
    }
    println!("  check: {nchk} integers refreshed and recombined (CRT), max |output error| 2^{:.2} (window 2^{:.2})",
             emax.log2(), -(263f64).log2() - 2.0);
    // time of the refresh of one integer (its 8 residues)
    let cts = enc(rng.gen(), &mut egen);
    let _ = keys.refresh_rns(&cts, &mods, &mut rng, &mut rbit);
    let mut best = f64::INFINITY; let mut ph = [0f64; 3];
    for _ in 0..reps {
        let t = Instant::now(); let (_, p) = keys.refresh_rns(&cts, &mods, &mut rng, &mut rbit); let tt = t.elapsed().as_secs_f64() * 1e3;
        if tt < best { best = tt; ph = p; }
    }
    println!("  rns  64 bits (8 residues, one CombBoot): {:8.3} ms per integer  [key switches {:.2}, roundings {:.2}, CombBoot {:.1}; {:.3} ms per residue]  (keys {:.0} s)",
             best, ph[0], ph[1], ph[2], best / 8.0, tk);
    // noise
    if ne > 0 {
        let (p, bc) = (cfg.p as u64, keys.bcols);
        let s = keys.small_sk.as_ref().to_vec();
        let (mut s2, mut cnt, mut emx) = (0.0, 0usize, 0.0f64);
        while cnt < ne {
            let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
            let e: Vec<i64> = (0..bc).map(|_| rng.gen_range(-(p as i64) / 4..=(p as i64) / 4)).collect();
            let bints: Vec<u64> = (0..bc).map(|k| {
                let dot = aints[k].iter().zip(&s).fold(0u64, |acc, (&a, &x)| acc.wrapping_add(a * x));
                ((dot as i64 + e[k]).rem_euclid(p as i64)) as u64
            }).collect();
            let outs = keys.combboot_rns(&aints, &bints, &mods);
            for k in 0..bc {
                let phs = decrypt_lwe_ciphertext(&keys.big_sk, &outs[k]).0;
                let want = ((e[k] as f64) / (p as f64 * mods[k] as f64) * 2f64.powi(64)).round() as i64 as u64;
                let err = tor(phs.wrapping_sub(want)); s2 += err * err; emx = emx.max(err.abs()); cnt += 1;
            }
        }
        let vo = s2 / cnt as f64;
        let sig2 = 2f64.powf(2.0 * LOG2_SIGMA_N); let nn = N as f64;
        let (b, l) = (cfg.gbr.0 as i32, cfg.gbr.1 as f64);
        let key = 2.0 * l * nn * (2f64.powi(2 * b) + 2.0) / 12.0 * sig2;
        let rnd = (1.0 + nn / 2.0) * 2f64.powf(-2.0 * l * b as f64) / 12.0;
        let (gb, gl, go) = cfg.gks; let glf = gl as f64;
        let vks = glf * nn * (2f64.powi(2 * gb as i32) + 2.0) / 12.0 * sig2 + (nn / 2.0) * 2f64.powf(-2.0 * (glf * gb as f64 + go as f64)) / 12.0;
        let bb = bc as f64; let n1 = s.iter().filter(|&&x| x == 1).count() as f64; let n = cfg.n as f64;
        let pred = n * key + n1 * (rnd + (bb - 1.0) * bb * vks) + n * 2f64.powf(-87.5) * 2f64.powi(2 * b) * l
                   + n1 * (bb - 1.0) * bb * 2f64.powf(-88.5) * 2f64.powi(2 * gb as i32) * glf;
        let vks_ks = stair_var(N, cfg.chain, 16); let phat = 263f64;
        let marg = 0.5 * ((1.0 / (U128 * U128)) / ((4.0 * phat).powi(2) * (NU * vo + vks_ks) + cfg.vterm)).log2();
        println!("  noise ({cnt} errors): out {:7.2} / {:7.2} predicted, max |e| 2^{:.2}, margin {:.2} bits at p̂ = 263 (KS {:.2})",
                 0.5 * vo.log2(), 0.5 * pred.log2(), emx.log2(), marg, 0.5 * vks_ks.log2());
    }
    drop(keys);
    if tf {
        let (ck, sk) = generate_keys(ConfigBuilder::default().build());
        set_server_key(sk);
        let x: u64 = 0xB5E3_6A7D_1234_9F0Eu64;
        let a = FheUint64::encrypt(x, &ck);
        let r: u64 = (&a & u64::MAX).decrypt(&ck); assert_eq!(r, x);
        let t = (0..reps).map(|_| { let t = Instant::now(); let _ = &a & u64::MAX; t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        println!("  tfhers FheUint64 refresh (&a & MAX): {:8.3} ms", t);
        println!("CSV\nrns,64,{:.4}\ntfhers,64,{:.4}", best, t);
    }
}
