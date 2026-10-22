//! RNS arithmetic on 64-bit integers, one thread, one session: x held as its residues x mod p̂_k (comb::RNS_MODULI, 8
//! moduli ≤ 263, product M ≈ 2^64.03), residue k encrypted as LWE_{s'}(r_k/p̂_k) of dimension N, all under one key s':
//!   refresh:        one CombBoot with B = 8 for the 8 residues (comb::COMB_RNS, refresh_rns);
//!   addition:       residue by residue (LWE additions: modulo p̂_k through the torus);
//!   multiplication: residue by residue, the refresh before a product (error bootstrap, product::PCONFIG_RNS, our m = 2
//!                   rotation, refresh_rns) and the product of Section 6 with p̂_k; as in product_curves, the time of a
//!                   multiplication is that of one refresh before a product plus the product, for each residue (each
//!                   operation is followed by a refresh; the other operand was refreshed after its own operation).
//!   check:  CHECK pairs of random 64-bit integers (default 5): x + y and x y modulo M by the CRT, then the product
//!           refreshed by CombBoot and decoded again.
//!   noise:  NPROD products per modulus (default 25, so 200): inputs refreshed by refresh_rns, product against
//!           p̂ μ1 μ2, measured variances (line "noise,263,ve,T,V0,vp,vks,vterm" for the noise model with PH=263).
//!   TFHE-rs FheUint64, default parameters: refresh (`&a & MAX`), `&a + &b`, `&a * &b` (MREPS runs, default 1).
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin rns_arith     (REPS=5, CHECK=5, NPROD=25, MREPS=1, TF=0: no TFHE-rs)
use eb_tfhers::comb::{cm, CombKeys, COMB_RNS, RNS_MODULI};
use eb_tfhers::eb::{tor, EbKeys};
use eb_tfhers::params::{stair_var, N};
use eb_tfhers::product::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint64};

fn env_usize(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

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

fn phase(sk: &[u64], c: &LweCiphertextOwned<u64>) -> u64 {
    let m = c.get_mask(); let a = m.as_ref();
    a.iter().zip(sk).fold(*c.get_body().data, |p, (&x, &s)| p.wrapping_sub(x.wrapping_mul(s)))
}

fn main() {
    let reps = env_usize("REPS", 5); let nchk = env_usize("CHECK", 5); let nprod = env_usize("NPROD", 25);
    let mreps = env_usize("MREPS", 1);
    let tf = std::env::var("TF").map(|v| v != "0").unwrap_or(true);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let mods: Vec<u64> = RNS_MODULI.to_vec();
    let mm: u128 = mods.iter().map(|&m| m as u128).product();
    let pc = &PCONFIG_RNS;
    println!("RNS arithmetic on TFHE-rs 1.8.1, one thread: moduli {:?} (M = 2^{:.2}); refresh: CombBoot B = 8, n = {}; before a product: error bootstrap m = {}, n = {}, (2^{},{}), trace (2^{},{})",
             mods, (mm as f64).log2(), COMB_RNS.n, pc.m, pc.cfg.n, pc.cfg.b, pc.cfg.l, pc.gtr.0, pc.gtr.1);
    let t0 = Instant::now();
    let seed = 7001;
    let comb = CombKeys::new_m(&COMB_RNS, 1, seed);
    let (eb, mut egen) = EbKeys::new(&pc.cfg, pc.m, seed);
    assert_eq!(comb.big_sk.as_ref(), eb.big_sk.as_ref(), "one key s' for CombBoot and the error bootstrap");
    let sk: Vec<u64> = eb.big_sk.as_ref().to_vec();
    let mut rng = StdRng::seed_from_u64(64); let mut rbit = 0x853c49e6748fea9bu64;
    let pk = ProductKeys::new(&sk, pc.gtr, GRL, &mut rng);
    let mut w = pk.work();
    let tk = t0.elapsed().as_secs_f64();
    let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
    let enc1 = |r: u64, m: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> LweCiphertextOwned<u64> {
        allocate_and_encrypt_new_lwe_ciphertext(&eb.big_sk, Plaintext((((r as u128) << 64) / m as u128) as u64), noise, cm(), egen)
    };
    let enc = |x: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Vec<LweCiphertextOwned<u64>> {
        mods.iter().map(|&m| enc1(x % m, m, egen)).collect()
    };
    let dec = |cs: &[LweCiphertextOwned<u64>]| -> u128 {
        let res: Vec<u64> = cs.iter().zip(&mods).map(|(c, &m)| ((((phase(&sk, c) as u128) * m as u128 + (1u128 << 63)) >> 64) as u64) % m).collect();
        crt(&res, &mods)
    };
    let add = |a: &[LweCiphertextOwned<u64>], b: &[LweCiphertextOwned<u64>], out: &mut [LweCiphertextOwned<u64>]| {
        for k in 0..a.len() { lwe_ciphertext_add(&mut out[k], &a[k], &b[k]); }
    };
    // check
    for _ in 0..nchk {
        let (x, y): (u64, u64) = (rng.gen(), rng.gen());
        let (xr, _) = comb.refresh_rns(&enc(x, &mut egen), &mods, &mut rng, &mut rbit);
        let (yr, _) = comb.refresh_rns(&enc(y, &mut egen), &mods, &mut rng, &mut rbit);
        let mut z = xr.clone(); add(&xr, &yr, &mut z);
        assert_eq!(dec(&z), (x as u128 + y as u128) % mm, "wrong addition");
        let xp: Vec<_> = xr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
        let yp: Vec<_> = yr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
        let pr: Vec<_> = (0..8).map(|k| pk.product(&mut w, &xp[k], &yp[k], mods[k])).collect();
        let want = (x as u128 * y as u128) % mm;
        assert_eq!(dec(&pr), want, "wrong multiplication");
        let (pr2, _) = comb.refresh_rns(&pr, &mods, &mut rng, &mut rbit);
        assert_eq!(dec(&pr2), want, "wrong refresh after a multiplication");
    }
    println!("  check: {nchk} pairs of 64-bit integers: x + y, x y (mod M) and the refresh of x y, correct (CRT)");
    // times
    let (x, y): (u64, u64) = (rng.gen(), rng.gen());
    let (xr, _) = comb.refresh_rns(&enc(x, &mut egen), &mods, &mut rng, &mut rbit);
    let (yr, _) = comb.refresh_rns(&enc(y, &mut egen), &mods, &mut rng, &mut rbit);
    let cts = enc(x, &mut egen);
    let _ = comb.refresh_rns(&cts, &mods, &mut rng, &mut rbit);
    let t_ref = (0..reps).map(|_| { let t = Instant::now(); let _ = comb.refresh_rns(&cts, &mods, &mut rng, &mut rbit); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
    let mut z = xr.clone();
    let t_add = (0..1000).map(|_| { let t = Instant::now(); add(&xr, &yr, &mut z); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
    let _ = eb.refresh_rns(&xr[0], mods[0], &mut rng, &mut rbit);
    let t_eb = (0..reps).map(|_| { let t = Instant::now(); for k in 0..8 { let _ = eb.refresh_rns(&xr[k], mods[k], &mut rng, &mut rbit); } t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
    let xp: Vec<_> = xr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
    let yp: Vec<_> = yr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
    let _ = pk.product(&mut w, &xp[0], &yp[0], mods[0]);
    let t_pr = (0..reps).map(|_| { let t = Instant::now(); for k in 0..8 { let _ = pk.product(&mut w, &xp[k], &yp[k], mods[k]); } t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
    println!("  rns  64 bits: refresh (CombBoot) {:8.3} ms;  addition {:8.4} ms;  multiplication {:8.3} ms (refresh before a product {:.3} + product {:.3}, 8 residues)   (keys {:.0} s)",
             t_ref, t_add, t_eb + t_pr, t_eb, t_pr, tk);
    let mut csv = vec![format!("rns,64,refresh,{t_ref:.4}"), format!("rns,64,add,{t_add:.5}"), format!("rns,64,mul,{:.4},{t_eb:.4},{t_pr:.4}", t_eb + t_pr)];
    // noise of the products (inputs refreshed by the error bootstrap), as product_curves
    if nprod > 0 {
        let t1 = Instant::now();
        let (mut ve, mut v0, mut tt, mut vp, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for _ in 0..nprod { for (k, &m) in mods.iter().enumerate() {
            let (m1, m2) = (rng.gen_range(0..m), rng.gen_range(0..m));
            let c1 = eb.refresh_rns(&enc1(m1, m, &mut egen), m, &mut rng, &mut rbit);
            let c2 = eb.refresh_rns(&enc1(m2, m, &mut egen), m, &mut rng, &mut rbit);
            let mu = |r: u64| -> u64 { (((r as u128) << 64) / m as u128) as u64 };
            let p1 = phase(&sk, &c1);
            for (c, r) in [(&c1, m1), (&c2, m2)] { let e = tor(phase(&sk, c).wrapping_sub(mu(r))); ve += e * e; }
            let (a, b) = pk.lwe_to_rlwe(&mut w, &c1);
            let ph = phase64(&sk, &a, &b);
            let e0 = tor(ph[0].wrapping_sub(p1)); v0 += e0 * e0;
            tt += e0 * e0 + ph[1..].iter().map(|&x| tor(x) * tor(x)).sum::<f64>();
            let o = pk.product(&mut w, &c1, &c2, m);
            let e = tor(phase(&sk, &o).wrapping_sub(mu((m1 * m2) % m))); vp += e * e; cnt += 1;
            let _ = k;
        } }
        let np = cnt as f64; ve /= 2.0 * np; v0 /= np; tt /= np; vp /= np;
        let vks = stair_var(N, pc.cfg.chain, 16);
        let l2 = |v: f64| 0.5 * v.log2();
        println!("    noise ({cnt} products, moduli in turn): in {:7.2}   √T {:7.2}   √V0 {:7.2}   product {:7.2} (predicted at p̂ = 263 from the measured inputs {:7.2})   KS {:7.2} predicted   ({:.0} s)",
                 l2(ve), l2(tt), l2(v0), l2(vp), l2(var_product(263.0, ve, tt, v0)), l2(vks), t1.elapsed().as_secs_f64());
        csv.push(format!("noise,263,{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", ve, tt, v0, vp, vks, pc.cfg.vterm));
    }
    drop(comb); drop(eb); drop(pk);
    if tf {
        let (ck, sk) = generate_keys(ConfigBuilder::default().build());
        set_server_key(sk);
        let (x, y) = (0xB5E3_6A7D_1234_9F0Eu64, 0x6A7D_0F1E_B5E3_2C4Du64);
        let (a, b) = (FheUint64::encrypt(x, &ck), FheUint64::encrypt(y, &ck));
        let r: u64 = (&a & u64::MAX).decrypt(&ck); assert_eq!(r, x);
        let r: u64 = (&a + &b).decrypt(&ck); assert_eq!(r, x.wrapping_add(y));
        let tr = (0..reps).map(|_| { let t = Instant::now(); let _ = &a & u64::MAX; t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        let ta = (0..reps).map(|_| { let t = Instant::now(); let _ = &a + &b; t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        let t = Instant::now(); let p = &a * &b; let tm0 = t.elapsed().as_secs_f64() * 1e3;
        let r: u64 = p.decrypt(&ck); assert_eq!(r, x.wrapping_mul(y));
        let tm = (0..mreps).map(|_| { let t = Instant::now(); let _ = &a * &b; t.elapsed().as_secs_f64() * 1e3 }).fold(tm0, f64::min);
        println!("  tfhers FheUint64: refresh (&a & MAX) {tr:9.3} ms;  addition {ta:9.3} ms;  multiplication {tm:10.3} ms");
        csv.push(format!("tfhers,64,refresh,{tr:.4}")); csv.push(format!("tfhers,64,add,{ta:.4}")); csv.push(format!("tfhers,64,mul,{tm:.4}"));
    }
    println!("CSV"); for l in csv { println!("{l}"); }
}
