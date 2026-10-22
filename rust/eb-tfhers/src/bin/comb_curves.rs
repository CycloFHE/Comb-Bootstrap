//! Times, one thread, all on TFHE-rs 1.8.1 and in the same session: the comb bootstrap (h = 1, grouping m of COMBM,
//! default 1) at 2, 4, 6, 8, 12 and 16 bits (whole refresh of B ciphertexts of dimension N: stair key switches, roundings
//! with selection, comb bootstrap, subtractions; time per ciphertext), the error bootstrap alone at the same sizes
//! (CONFIGS_RS2, m chosen for each k; CFG=rs: CONFIGS_RS, m = 1), and the refresh of FheUint{bits} by TFHE-rs with its
//! default parameters (`&a & MAX`).
//! Times in ms, minimum over REPS runs (default 5) after one warm-up run.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin comb_curves     (BITS=2,4,8,12,16; 6 also available)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::{configs, Config};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint12, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};

fn main() {
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("2,4,8,12,16".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let mut csv = Vec::new();
    println!("Times (ms), one thread, TFHE-rs 1.8.1: comb bootstrap (per ciphertext), error bootstrap alone, TFHE-rs");
    for cfg in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let keys = CombKeys::new(cfg, 900 + cfg.bits as u64);
        let mut rng = StdRng::seed_from_u64(cfg.bits as u64); let mut rbit = 0x853c49e6748fea9bu64;
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(new_seeder().as_mut().seed(), new_seeder().as_mut());
        let lp = cfg.bits; let bc = keys.bcols;
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let mus: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..(1u64 << lp)) << (64 - lp)).collect();
        let cts: Vec<LweCiphertextOwned<u64>> = mus.iter().map(|&mu| allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen)).collect();
        let (outs, _) = keys.refresh(&cts, &mut rng, &mut rbit);
        for (o, &mu) in outs.iter().zip(&mus) {
            let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, o).0.wrapping_sub(mu));
            assert!(e.abs() < 1.0 / (4.0 * 2f64.powi(lp as i32)), "wrong refresh");
        }
        let mut best = f64::INFINITY; let mut ph = [0f64; 3];
        for _ in 0..reps {
            let t = Instant::now(); let (_, p) = keys.refresh(&cts, &mut rng, &mut rbit); let tt = t.elapsed().as_secs_f64() * 1e3;
            if tt < best { best = tt; ph = p; }
        }
        println!("  comb {:2} bits: m = {}, p = {}, B = {:2}, n = {:4}: {:8.3} ms per ciphertext  [{:.1} ms for B: key switches {:.2}, roundings {:.2}, comb bootstrap {:.1}]",
                 cfg.bits, keys.m, cfg.p, bc, cfg.n, best / bc as f64, best, ph[0], ph[1], ph[2]);
        csv.push(format!("comb,{},{:.4}", cfg.bits, best / bc as f64));
        drop(keys);
        // the error bootstrap alone at the same size
        let (ecfg, em): (&Config, usize) = configs().into_iter().find(|(c, _)| 2 * c.k == cfg.bits).unwrap();
        let (ek, mut egen2) = EbKeys::new(ecfg, em, 950 + cfg.bits as u64);
        let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&ek.big_sk, Plaintext(mus[0]), noise, cm(), &mut egen2);
        let _ = ek.refresh(&ct, &mut rng, &mut rbit, None);
        let te = (0..reps).map(|_| { let t = Instant::now(); let _ = ek.refresh(&ct, &mut rng, &mut rbit, None); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        println!("  eb   {:2} bits: m = {}, n = {:4}, (2^{},{}): {:8.3} ms", cfg.bits, em, ecfg.n, ecfg.b, ecfg.l, te);
        csv.push(format!("eb,{},{:.4}", cfg.bits, te));
    }
    let (ck, sk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    macro_rules! tf { ($T:ty, $C:ty, $b:expr) => {{
        if bits.contains(&$b) {
            let max: $C = ((1u32 << $b) - 1) as $C; let a = <$T>::encrypt((0xB5E3u32 as $C) & max, &ck);
            let _ = &a & max;
            let t = (0..reps).map(|_| { let t = Instant::now(); let _ = &a & max; t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
            println!("  tfhers FheUint{:<2}: {:8.3} ms", $b, t); csv.push(format!("tfhers,{},{:.4}", $b, t));
        }
    }}; }
    tf!(FheUint2, u8, 2); tf!(FheUint4, u8, 4); tf!(FheUint6, u8, 6); tf!(FheUint8, u8, 8); tf!(FheUint12, u16, 12); tf!(FheUint16, u16, 16);
    println!("CSV (scheme,bits,ms)"); for l in csv { println!("{l}"); }
}
