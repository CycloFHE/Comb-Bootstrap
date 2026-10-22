//! Product of two ciphertexts (Section 6 of the paper) built on TFHE-rs, against the product of TFHE-rs, one thread, same session
//! (configurations P of src/product.rs, k = 1..4, p̂ = 4^k):
//!   ebrs:    refresh before a product (stair key switch, rounding with selection, rotation (m = 2: ours on the FFT
//!            of TFHE-rs, PCONFIGS_RS2, default; PCFG=rs: CMux of TFHE-rs, m = 1, PCONFIGS_RS), extraction, subtraction) + the product itself (two traces, tensor product, relinearization, extraction);
//!            the time of a product is that of one refresh plus one product (each operation is followed by a refresh);
//!   tfhers:  `&a * &b` on FheUint{2k}, default parameters, high-level API (carries propagated), k = 1..8 (TKS=...).
//! The product of the error bootstrap is not possible beyond k = 4 at N = 2048 (error of the FFT, noise of the keys
//! of the trace).
//! Times in ms: minimum over REPS runs (default 10) after one warm-up run (and median for tfhers).
//! Noise (NPROD products, default 200; 0: none): inputs refreshed with the configuration P, trace of the first one,
//! product against p̂ μ1 μ2; log2 σ measured / predicted, and the variances for the noise model (margin of
//! the refresh that follows, exact law).
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin product_curves      (KS=1,2,...: some k only; TKS=... for TFHE-rs)
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::*;
use eb_tfhers::product::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder};
use tfhe::{FheUint10, FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};

fn env_usize(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

fn bench<O: FnMut()>(reps: usize, mut op: O) -> (f64, f64) {
    op();
    let mut ts: Vec<f64> = (0..reps).map(|_| { let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e3 }).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[reps / 2], ts[0])
}

macro_rules! tfhers_product {
    ($T:ty, $C:ty, $k:expr, $ck:expr, $reps:expr, $csv:expr) => {{
        let k: u32 = $k; let max: u32 = (1u32 << (2 * k)) - 1;
        let (x, y): ($C, $C) = ((0xB5E3u32 & max) as $C, (0x6A7Du32 & max) as $C);
        let a = <$T>::encrypt(x, $ck); let b = <$T>::encrypt(y, $ck);
        let r: $C = (&a * &b).decrypt($ck); assert_eq!(r as u32, (x as u32).wrapping_mul(y as u32) & max);
        let t = bench($reps, || { let _ = &a * &b; });
        println!("  tfhers FheUint{:<2} (k = {}) product {:>9.3} ms (min {:>9.3})", 2 * k, k, t.0, t.1);
        $csv.push(format!("tfhers,{},product,{:.4},{:.4}", k, t.0, t.1));
    }};
}

fn phase(sk: &[u64], c: &LweCiphertextOwned<u64>) -> u64 {
    let m = c.get_mask(); let a = m.as_ref();
    a.iter().zip(sk).fold(*c.get_body().data, |p, (&x, &s)| p.wrapping_sub(x.wrapping_mul(s)))
}

fn main() {
    let reps = env_usize("REPS", 10);
    let nprod = env_usize("NPROD", 200);
    let ks: Vec<u32> = std::env::var("KS").unwrap_or("1,2,3,4".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let tks: Vec<u32> = std::env::var("TKS").unwrap_or("1,2,3,4,5,6,7,8".into()).split(',').filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap()).collect();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let l2 = |v: f64| 0.5 * v.log2();
    let mut csv: Vec<String> = Vec::new();
    println!("Product (ms), one thread, TFHE-rs 1.8.1 for both; relinearization (2^{},{})", GRL.0, GRL.1);
    for pc in pconfigs().iter().filter(|c| ks.contains(&c.cfg.k)) {
        let cfg = &pc.cfg; let k = cfg.k; let lp = 2 * k; let phat = 1u64 << lp; let d = 64 - lp;
        let t0 = Instant::now();
        let (keys, mut egen) = EbKeys::new(cfg, pc.m, 3000 + k as u64);
        let sk: Vec<u64> = keys.big_sk.as_ref().to_vec();
        let mut rng = StdRng::seed_from_u64(91 + k as u64); let mut rbit = 0x2545F4914F6CDD1Du64;
        let pk = ProductKeys::new(&sk, pc.gtr, GRL, &mut rng);
        let mut w = pk.work();
        let tkeys = t0.elapsed().as_secs_f64();
        let enc_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
        let fresh = |m: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> LweCiphertextOwned<u64> {
            allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(m << d), enc_noise, cm(), egen)
        };
        // correctness and times
        let (m1, m2) = (rng.gen_range(0..phat), rng.gen_range(0..phat));
        let ct = fresh(m1, &mut egen);
        let c1 = keys.refresh(&ct, &mut rng, &mut rbit, None);
        let c2 = keys.refresh(&fresh(m2, &mut egen), &mut rng, &mut rbit, None);
        let o = pk.product(&mut w, &c1, &c2, phat);
        let e = tor(phase(&sk, &o).wrapping_sub(((m1 * m2) % phat) << d));
        assert!(e.abs() < 1.0 / (4.0 * phat as f64), "wrong product");
        let tr = bench(reps, || { let _ = keys.refresh(&ct, &mut rng, &mut rbit, None); }).1;
        let ttr = bench(reps, || { let _ = pk.lwe_to_rlwe(&mut w, &c1); }).1;
        let tp = bench(reps, || { let _ = pk.product(&mut w, &c1, &c2, phat); }).1;
        println!("  ebrs k = {} ({:2} bits) before a product: m = {}, n = {:4}, (2^{},{}), refresh {:7.3} ms;  product, trace (2^{},{}) {:6.3} ms (one trace {:.3});  total {:7.3} ms   (keys {:.0} s)",
                 k, lp, pc.m, cfg.n, cfg.b, cfg.l, tr, pc.gtr.0, pc.gtr.1, tp, ttr, tr + tp, tkeys);
        csv.push(format!("ebrs,{},product,{:.4},{:.4},{:.4}", k, tr + tp, tr, tp));
        // noise
        if nprod > 0 {
            let t1 = Instant::now();
            let (mut ve, mut v0, mut tt, mut vp) = (0.0, 0.0, 0.0, 0.0);
            for _ in 0..nprod {
                let (m1, m2) = (rng.gen_range(0..phat), rng.gen_range(0..phat));
                let c1 = keys.refresh(&fresh(m1, &mut egen), &mut rng, &mut rbit, None);
                let c2 = keys.refresh(&fresh(m2, &mut egen), &mut rng, &mut rbit, None);
                let p1 = phase(&sk, &c1);
                for (c, m) in [(&c1, m1), (&c2, m2)] { let e = tor(phase(&sk, c).wrapping_sub(m << d)); ve += e * e; }
                let (a, b) = pk.lwe_to_rlwe(&mut w, &c1);
                let ph = phase64(&sk, &a, &b);
                let e0 = tor(ph[0].wrapping_sub(p1)); v0 += e0 * e0;
                tt += e0 * e0 + ph[1..].iter().map(|&x| tor(x) * tor(x)).sum::<f64>();
                let o = pk.product(&mut w, &c1, &c2, phat);
                let e = tor(phase(&sk, &o).wrapping_sub(((m1 * m2) % phat) << d)); vp += e * e;
            }
            let np = nprod as f64; ve /= 2.0 * np; v0 /= np; tt /= np; vp /= np;
            let tiny = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-60)), 0.0);
            let mut vks = 0.0; let nks = 400;
            for _ in 0..nks {
                let mu: u64 = rng.gen();
                let c: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), tiny, cm(), &mut egen);
                let (a, b) = keys.stair.as_ref().unwrap().switch(c.get_mask().as_ref(), *c.get_body().data, &mut rbit);
                let dot = a.iter().zip(keys.small_sk.as_ref()).fold(0u64, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y)));
                let e = tor(b.wrapping_sub(dot).wrapping_sub(mu)); vks += e * e;
            }
            vks /= nks as f64;
            let cfft = if pc.m == 1 { -87.0 } else { -84.5 };   // error of the FFT per CMux / per pair
            let pe = var_rot(cfg.n, pc.m, cfg.b, cfg.l) + (cfg.n / pc.m) as f64 * 2f64.powf(cfft) * 2f64.powi(2 * cfg.b as i32) * cfg.l as f64;
            let (pt, pv0) = var_trace(pc.gtr);
            println!("    noise ({} products): in {:7.2} / {:7.2}   √T {:7.2} / {:7.2}   √V0 {:7.2} / {:7.2}   product {:7.2} / {:7.2} (measured inputs {:7.2})   KS {:7.2} / {:7.2}   ({:.0} s)",
                     nprod, l2(ve), l2(pe), l2(tt), l2(pt), l2(v0), l2(pv0), l2(vp), l2(var_product(phat as f64, pe, pt, pv0)),
                     l2(var_product(phat as f64, ve, tt, v0)), l2(vks), l2(stair_var(N, cfg.chain, 16)), t1.elapsed().as_secs_f64());
            csv.push(format!("noise,{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", k, ve, tt, v0, vp, vks, cfg.vterm));
        }
        drop(keys);
    }
    let (ck, sk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    if tks.contains(&1) { tfhers_product!(FheUint2, u8, 1, &ck, reps, csv); }
    if tks.contains(&2) { tfhers_product!(FheUint4, u8, 2, &ck, reps, csv); }
    if tks.contains(&3) { tfhers_product!(FheUint6, u8, 3, &ck, reps, csv); }
    if tks.contains(&4) { tfhers_product!(FheUint8, u8, 4, &ck, reps, csv); }
    if tks.contains(&5) { tfhers_product!(FheUint10, u16, 5, &ck, reps, csv); }
    if tks.contains(&6) { tfhers_product!(FheUint12, u16, 6, &ck, reps, csv); }
    if tks.contains(&7) { tfhers_product!(FheUint14, u16, 7, &ck, reps, csv); }
    if tks.contains(&8) { tfhers_product!(FheUint16, u16, 8, &ck, reps, csv); }
    println!("CSV");
    for l in csv { println!("{l}"); }
}
