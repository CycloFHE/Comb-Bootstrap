//! Times on T threads (THREADS, default 8), TFHE-rs 1.8.1, same session, at 2, 4, 8, 12 and 16 bits:
//!   latency of one ciphertext: TFHE-rs FheUint{bits} refresh (`&a & MAX`, blocks in parallel) with its default
//!   (classic) parameters and with its multi-bit parameters (GROUP_4); the error bootstrap on 1 thread and with its
//!   rotation on t threads (EBT, default 2,4,T);
//!   TFHE-rs core with its key switch and with ours (tfhers_ours.rs: the key switch neutralized), blocks in parallel;
//!   one CombBoot call on T threads (B = 16 ciphertexts up to 8 bits, 8 beyond; time of the call and per ciphertext).
//! Times in ms, minimum over REPS runs (default 5) after one warm-up run.
//! Run:  cargo run --release --bin mt_curves      (THREADS=8, BITS=2,4,8,12,16 (6 also available), REPS=5)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::CONFIGS_RS;
use eb_tfhers::tfhers_ours::TfhersBlocks;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::shortint::parameters::v1_8::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint12, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};

fn best<O: FnMut()>(reps: usize, mut op: O) -> f64 {
    op();
    (0..reps).map(|_| { let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min)
}

fn main() {
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let tt: usize = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let ebt: Vec<usize> = std::env::var("EBT").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![2, 4, tt]);
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("2,4,8,12,16".into()).split(',').map(|x| x.parse().unwrap()).collect();
    rayon::ThreadPoolBuilder::new().num_threads(tt).build_global().unwrap();
    println!("Times (ms), T = {tt} threads, TFHE-rs 1.8.1, same session (rayon pool of {} threads)", rayon::current_num_threads());
    let mut csv = Vec::new();
    for cfg in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let lp = cfg.bits;
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let mut rng = StdRng::seed_from_u64(lp as u64);
        // error bootstrap: latency on 1 and t threads
        let ecfg = CONFIGS_RS.iter().find(|c| 2 * c.k == lp).unwrap();
        let (ek, mut egen) = EbKeys::new(ecfg, 1, 950 + lp as u64);
        let bc = 2 * 2048 / cfg.p;
        let mus: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..(1u64 << lp)) << (64 - lp)).collect();
        let ects: Vec<LweCiphertextOwned<u64>> = mus.iter().map(|&mu| allocate_and_encrypt_new_lwe_ciphertext(&ek.big_sk, Plaintext(mu), noise, cm(), &mut egen)).collect();
        let mut rbit = 0x853c49e6748fea9bu64;
        let t1 = best(reps, || { let _ = ek.refresh(&ects[0], &mut rng, &mut rbit, None); });
        println!("  {lp:2} bits: EB 1 thread {t1:8.3} ms   (n = {}, (2^{},{}))", ecfg.n, ecfg.b, ecfg.l);
        csv.push(format!("eb_latency,{lp},1,{t1:.4}"));
        for &t in &ebt {
            let o = ek.refresh_mt(&ects[0], &mut rng, &mut rbit, t);
            assert!(tor(decrypt_lwe_ciphertext(&ek.big_sk, &o).0.wrapping_sub(mus[0])).abs() < 1.0 / (4.0 * 2f64.powi(lp as i32)));
            let tm = best(reps, || { let _ = ek.refresh_mt(&ects[0], &mut rng, &mut rbit, t); });
            println!("           EB {t} threads {tm:8.3} ms  (speed-up {:.2})", t1 / tm);
            csv.push(format!("eb_latency,{lp},{t},{tm:.4}"));
        }
        drop(ek);
        // CombBoot on T threads
        let keys = CombKeys::new_m(cfg, 1, 900 + lp as u64);   // combboot_mt: m = 1
        let cts: Vec<LweCiphertextOwned<u64>> = mus.iter().map(|&mu| allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen)).collect();
        let outs = keys.refresh_mt(&cts, 1, tt);
        for (o, &mu) in outs.iter().zip(&mus) { assert!(tor(decrypt_lwe_ciphertext(&keys.big_sk, o).0.wrapping_sub(mu)).abs() < 1.0 / (4.0 * 2f64.powi(lp as i32)), "wrong refresh"); }
        let mut s = 2u64;
        let tc = best(reps, || { s += 1; let _ = keys.refresh_mt(&cts, s, tt); });
        println!("           CombBoot batch of {bc} on {tt} threads: {:8.3} ms per ciphertext ({tc:.1} ms)", tc / bc as f64);
        csv.push(format!("comb_batch,{lp},{tt},{:.4}", tc / bc as f64));
        drop(keys);
    }
    // TFHE-rs: classic (default) and multi-bit GROUP_4 parameters
    for (lab, conf) in [("classic", ConfigBuilder::default().build()),
                        ("multibit4", ConfigBuilder::with_custom_parameters(V1_8_PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128).build())] {
        let (ck, sk) = generate_keys(conf);
        let sk2 = sk.clone();
        rayon::broadcast(|_| set_server_key(sk2.clone()));
        set_server_key(sk);
        macro_rules! tf { ($T:ty, $C:ty, $b:expr) => {{
            if bits.contains(&$b) {
                let bc = 1;
                let max: $C = ((1u32 << $b) - 1) as $C;
                let xs: Vec<$T> = (0..bc).map(|i| <$T>::encrypt(((0xB5E3u32 + 7 * i as u32) as $C) & max, &ck)).collect();
                let r: $C = (&xs[0] & max).decrypt(&ck); assert_eq!(r, ((0xB5E3u32 as $C) & max));
                let tl = best(reps, || { let _ = &xs[0] & max; });
                println!("  TFHE-rs {lab:9} FheUint{:<2}: latency {tl:8.3} ms", $b);
                csv.push(format!("tfhers_{lab}_latency,{},{tt},{tl:.4}", $b));
            }
        }}; }
        tf!(FheUint2, u8, 2); tf!(FheUint4, u8, 4); tf!(FheUint6, u8, 6); tf!(FheUint8, u8, 8); tf!(FheUint12, u16, 12); tf!(FheUint16, u16, 16);
    }
    // TFHE-rs core (default parameters), blocks in parallel: its key switch (core) or ours (ours), same PBS
    let tb = TfhersBlocks::new(0x7f4a_9e37);
    for &b in &bits {
        let k = (b / 2) as usize;
        let (cts, ms) = tb.encrypt(k, 41 + k as u64);
        for (lab, ours) in [("core", false), ("ours", true)] {
            tb.check(&tb.par_refresh(ours, &cts, 1), &ms);
            let mut s = 1u64;
            let tl = best(reps, || { s += 1; let _ = tb.par_refresh(ours, &cts, s); });
            println!("  TFHE-rs {lab:9} FheUint{b:<2}: latency {tl:8.3} ms  (blocks in parallel, {} key switch)", if ours { "our" } else { "its" });
            csv.push(format!("tfhers_{lab}_latency,{b},{tt},{tl:.4}"));
        }
    }
    println!("CSV (what,bits,threads,ms)"); for l in csv { println!("{l}"); }
}
