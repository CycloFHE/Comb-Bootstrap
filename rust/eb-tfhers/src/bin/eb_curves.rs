//! Times, one thread, same library: the error bootstrap built on TFHE-rs (configurations R of Figure 4) against
//! TFHE-rs 1.8.1 itself:
//!   ebrs_m1:     whole refresh (stair key switch, rounding with selection, multi-bit rotation of TFHE-rs, extraction,
//!                subtraction) of one ciphertext of dimension N = 2048, k = 1..8 (p̂ = 4^k);
//!   tfhers:      refresh of FheUint2k with the default parameters (`&a & MAX`, one PBS per block), high-level API;
//!   tfhers_core: the same k blocks, core_crypto, one after the other: key switch of TFHE-rs + PBS of TFHE-rs;
//!   tfhers_ours: the same with our polynomial stair key switch in place of its own (tfhers_ours.rs, as in tfhers_ks):
//!                TFHE-rs with the key switch neutralized;
//!   tfhers_pbs:  one key switch + PBS of one 2_2 block, core_crypto, ThreadCount(1): classic default parameters
//!                (n = 918, (2^23,1)) and multi-bit parameters V1_8 GROUP_3 / GROUP_4 (2^-128).
//! Times in ms: minimum over REPS runs (default 10) after one warm-up run (and median for tfhers).
//! Configurations: CONFIGS_RS2 (default: m = 2, our multi-bit rotation, or m = 1, CMux of TFHE-rs, chosen for each k),
//! CFG=rs (CONFIGS_RS, m = 1) or CFG=jl (those of the Julia prototype, m = 2); EBM=1,2 runs both m.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin eb_curves      (KS=1,2,...: some k only)
use eb_tfhers::eb::{cm, tor, EbKeys, Times};
use eb_tfhers::params::*;
use eb_tfhers::tfhers_ours::TfhersBlocks;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::shortint::parameters::v1_8::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder};
use tfhe::{FheUint10, FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};

fn env_usize(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

fn bench<O: FnMut()>(reps: usize, mut op: O) -> (f64, f64) {
    op();
    let mut ts: Vec<f64> = (0..reps).map(|_| { let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e3 }).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[reps / 2], ts[0])
}

macro_rules! tfhers_refresh {
    ($T:ty, $C:ty, $k:expr, $ck:expr, $reps:expr, $csv:expr) => {{
        let k: u32 = $k; let max: $C = ((1u32 << (2 * k)) - 1) as $C;
        let x: $C = (0xB5E3u32 as $C) & max;
        let a = <$T>::encrypt(x, $ck);
        let r: $C = (&a & max).decrypt($ck); assert_eq!(r, x);
        let t = bench($reps, || { let _ = &a & max; });
        println!("  tfhers FheUint{:<2} (k = {}) refresh {:>9.3} ms (min {:>9.3})", 2 * k, k, t.0, t.1);
        $csv.push(format!("tfhers,{},refresh,{:.4},{:.4}", k, t.0, t.1));
    }};
}

fn block_pbs(label: &str, lwe_dim: LweDimension, poly: PolynomialSize, lwe_noise: DynamicDistribution<u64>,
             glwe_noise: DynamicDistribution<u64>, pbs: (DecompositionBaseLog, DecompositionLevelCount),
             ks: (DecompositionBaseLog, DecompositionLevelCount), grouping: Option<usize>, reps: usize, csv: &mut Vec<String>) {
    let mut boxed = new_seeder(); let seeder = boxed.as_mut();
    let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
    let small = LweSecretKey::generate_new_binary(lwe_dim, &mut sgen);
    let glwe_sk = GlweSecretKey::generate_new_binary(GlweDimension(1), poly, &mut sgen);
    let big = glwe_sk.clone().into_lwe_secret_key();
    let ksk = allocate_and_generate_new_lwe_keyswitch_key(&big, &small, ks.0, ks.1, lwe_noise, cm(), &mut egen);
    let delta = 1u64 << 59; // 2_2: 4 bits + padding
    let ct_big: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&big, Plaintext(5 * delta), glwe_noise, cm(), &mut egen);
    let mut ct_small = LweCiphertext::new(0u64, small.lwe_dimension().to_lwe_size(), cm());
    let acc = generate_programmable_bootstrap_glwe_lut(poly, GlweDimension(1).to_glwe_size(), 16, cm(), delta, |x: u64| x);
    let mut out = LweCiphertext::new(0u64, big.lwe_dimension().to_lwe_size(), cm());
    let (tks, tpbs);
    match grouping {
        None => {
            let bsk = allocate_and_generate_new_lwe_bootstrap_key(&small, &glwe_sk, pbs.0, pbs.1, glwe_noise, cm(), &mut egen);
            let mut fbsk = FourierLweBootstrapKey::new(bsk.input_lwe_dimension(), bsk.glwe_size(), bsk.polynomial_size(),
                bsk.decomposition_base_log(), bsk.decomposition_level_count());
            convert_standard_lwe_bootstrap_key_to_fourier(&bsk, &mut fbsk);
            tks = bench(reps, || keyswitch_lwe_ciphertext(&ksk, &ct_big, &mut ct_small)).1;
            tpbs = bench(reps, || programmable_bootstrap_lwe_ciphertext(&ct_small, &mut out, &acc, &fbsk)).1;
        }
        Some(g) => {
            let mut bsk = LweMultiBitBootstrapKey::new(0u64, GlweDimension(1).to_glwe_size(), poly, pbs.0, pbs.1, lwe_dim, LweBskGroupingFactor(g), cm());
            par_generate_lwe_multi_bit_bootstrap_key(&small, &glwe_sk, &mut bsk, glwe_noise, &mut egen);
            let mut fbsk = FourierLweMultiBitBootstrapKey::new(bsk.input_lwe_dimension(), bsk.glwe_size(), bsk.polynomial_size(),
                bsk.decomposition_base_log(), bsk.decomposition_level_count(), bsk.grouping_factor());
            par_convert_standard_lwe_multi_bit_bootstrap_key_to_fourier(&bsk, &mut fbsk);
            tks = bench(reps, || keyswitch_lwe_ciphertext(&ksk, &ct_big, &mut ct_small)).1;
            tpbs = bench(reps, || multi_bit_programmable_bootstrap_lwe_ciphertext(&ct_small, &mut out, &acc, &fbsk, ThreadCount(1), true)).1;
        }
    }
    let m = (decrypt_lwe_ciphertext(&big, &out).0.wrapping_add(delta / 2)) / delta;
    assert_eq!(m % 16, 5, "{label}: wrong PBS result");
    println!("  tfhers_pbs {:<34} n = {:4}: key switch {:6.3} + PBS {:7.3} = {:7.3} ms per block", label, lwe_dim.0, tks, tpbs, tks + tpbs);
    csv.push(format!("tfhers_pbs,{},{},{:.4},{:.4},{:.4}", label, lwe_dim.0, tks, tpbs, tks + tpbs));
}

fn main() {
    let reps = env_usize("REPS", 10);
    let ks: Vec<u32> = std::env::var("KS").unwrap_or("1,2,3,4,5,6,7,8".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let mut csv: Vec<String> = Vec::new();
    println!("Times (ms), one thread, TFHE-rs 1.8.1 for both");
    let ebm: Option<Vec<usize>> = std::env::var("EBM").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect());
    for (cfg, m0) in configs().into_iter().filter(|(c, _)| ks.contains(&c.k)) { for &m in ebm.as_ref().unwrap_or(&vec![m0]) {
        let (keys, mut egen) = EbKeys::new(cfg, m, 2000 + cfg.k as u64);
        let mut rng = StdRng::seed_from_u64(77 + cfg.k as u64); let mut rbit = 0x2545F4914F6CDD1Du64;
        let lp = 2 * cfg.k; let msg: u64 = rng.gen_range(0..(1u64 << lp)); let mu = msg << (64 - lp);
        let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen);
        let out = keys.refresh(&ct, &mut rng, &mut rbit, None);
        let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu));
        assert!(e.abs() < 1.0 / (4.0 * 2f64.powi(lp as i32)), "wrong refresh");
        let t = bench(reps, || { let _ = keys.refresh(&ct, &mut rng, &mut rbit, None); }).1;
        let mut tc = Times { ks: 0.0, rounding: 0.0, rotation: 0.0, rest: 0.0 };
        for _ in 0..reps { let _ = keys.refresh(&ct, &mut rng, &mut rbit, Some(&mut tc)); }
        let r = reps as f64;
        println!("  ebrs k = {} ({:2} bits), m = {}, n = {:4}, (2^{},{}): refresh {:7.3} ms  [mean: key switch {:.3}, rounding {:.3}, rotation {:.3}, rest {:.3}]",
                 cfg.k, lp, m, cfg.n, cfg.b, cfg.l, t, tc.ks / r, tc.rounding / r, tc.rotation / r, tc.rest / r);
        csv.push(format!("ebrs_m{},{},refresh,{:.4}", m, cfg.k, t));
        drop(keys);
    } }
    let p = V1_8_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128;
    block_pbs("classic (2^23,1), default", p.lwe_dimension, p.polynomial_size, p.lwe_noise_distribution, p.glwe_noise_distribution,
              (p.pbs_base_log, p.pbs_level), (p.ks_base_log, p.ks_level), None, reps, &mut csv);
    for (lab, q) in [("multi-bit group 3 (2^15,2)", V1_8_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128),
                     ("multi-bit group 4 (2^22,1)", V1_8_PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128)] {
        block_pbs(lab, q.lwe_dimension, q.polynomial_size, q.lwe_noise_distribution, q.glwe_noise_distribution,
                  (q.pbs_base_log, q.pbs_level), (q.ks_base_log, q.ks_level), Some(q.grouping_factor.0), reps, &mut csv);
    }
    let (ck, sk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    tfhers_refresh!(FheUint2, u8, 1, &ck, reps, csv);
    tfhers_refresh!(FheUint4, u8, 2, &ck, reps, csv);
    tfhers_refresh!(FheUint6, u8, 3, &ck, reps, csv);
    tfhers_refresh!(FheUint8, u8, 4, &ck, reps, csv);
    tfhers_refresh!(FheUint10, u16, 5, &ck, reps, csv);
    tfhers_refresh!(FheUint12, u16, 6, &ck, reps, csv);
    tfhers_refresh!(FheUint14, u16, 7, &ck, reps, csv);
    tfhers_refresh!(FheUint16, u16, 8, &ck, reps, csv);
    // TFHE-rs core, k blocks one after the other: its key switch (core) or ours (ours), same PBS
    let tb = TfhersBlocks::new(0x7f4a_9e37);
    let mut rbit = 0x9E37_79B9_7F4A_7C15u64;
    for &k in &ks {
        let (cts, ms) = tb.encrypt(k as usize, 31 + k as u64);
        for (lab, ours) in [("core", false), ("ours", true)] {
            tb.check(&tb.refresh(ours, &cts, &mut rbit), &ms);
            let t = bench(reps, || { let _ = tb.refresh(ours, &cts, &mut rbit); });
            println!("  tfhers_{lab} FheUint{:<2} (k = {}) refresh {:>9.3} ms (min {:>9.3})  [{k} x ({} key switch + PBS of TFHE-rs)]",
                     2 * k, k, t.0, t.1, if ours { "our" } else { "its" });
            csv.push(format!("tfhers_{lab},{},refresh,{:.4},{:.4}", k, t.0, t.1));
        }
    }
    println!("CSV");
    for l in csv { println!("{l}"); }
}
