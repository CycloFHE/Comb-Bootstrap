//! RNS arithmetic on 32-bit integers against the CRT integers of TFHE-rs, one thread, one session.
//! Ours: x held as its residues modulo pairwise coprime moduli p̂_k, residue k encrypted as LWE_{s'}(r_k/p̂_k) of
//! dimension N = 2048, all under one key s'. Three variants (VARIANTS, default 4,16,8):
//!   4:  the 4 largest moduli of comb::RNS_MODULI, {263, 262, 261, 259} (2^32.12): CombBoot comb::COMB_RNS (B = 8,
//!       two integers per call), product product::PCONFIG_RNS (p̂ ≤ 263);
//!   16: the 8 moduli comb::RNS32_MODULI, {25, 23, 21, 19, 17, 16, 13, 11} (2^33.05, the smallest largest modulus for 8
//!       residues): CombBoot comb::COMB_RNS32_B16 (B = 16, p = 256, two integers per call), product PCONFIG_RNS32 (p̂ ≤ 25);
//!   8:  the same moduli, CombBoot comb::COMB_RNS32_B8 (B = 8, p = 512, one integer per call), product PCONFIG_RNS32.
//! For each: refresh = one CombBoot (time per call, the latency of one integer, and per integer); addition = residue by
//! residue (LWE additions); multiplication = residue by residue, one refresh before a product (error bootstrap, m = 2,
//! refresh_rns) and the product of Section 6, as in rns_arith (one refresh before a product per multiplication).
//!   check:  CHECK pairs of random 32-bit integers (default 5): refresh, x + y, x y modulo the product of the moduli, and
//!           the refresh of x y and x + y, decrypted and recombined by the CRT;
//!   noise:  NE output errors of the CombBoot (default 1600; noise-free inputs modulo p, E_k uniform in |E_k| ≤ p/4) and
//!           the margin at failure 2^-128 after 1000 additions for the largest modulus (as rns_curves); NPROD products per
//!           modulus (default 25) with inputs refreshed before a product: line "noise,P̂MAX,ve,T,V0,vp,vks,vterm" for
//!           the noise model (margin of the refresh that follows the product, p̂ = the largest modulus).
//! TFHE-rs, CRT (integer::gen_keys_crt, the RNS of TFHE-rs): blocks with message modulus 32 (residues below 32), set
//!   V1_6_PARAM_MESSAGE_5_CARRY_1_KS_PBS_GAUSSIAN_2M128 (n = 1006, N = 8192, PBS (2^15,2), KS (2^3,7)), the set of the
//!   CRT tests of TFHE-rs; basis CRT_BASIS (default 19,23,25,27,29,31,32: product 2^32.98, the fewest blocks, 7, for a
//!   32-bit integer with residues below 32; the tests of TFHE-rs take 3,11,13,19,23,29,31,32, 8 blocks):
//!   refresh = full_extract_message_assign (1 PBS per block), addition = unchecked_crt_add (no PBS; carry room for one
//!   addition only before a refresh), multiplication = unchecked_crt_mul (2 PBS per block: (x+y)^2/4 - (x-y)^2/4)
//!   followed by one refresh (1 PBS per block), as ours counts one refresh per multiplication.
//!   Key switch: the time of one LWE key switch of this set (N -> n) is measured apart; "without KS" subtracts it from
//!   each PBS (a lower bound of TFHE-rs with any key switch, ours included).
//! TFHE-rs, radix FheUint32 (default parameters, 16 blocks of 2+2 bits, mod 2^32): refresh (`&a & MAX`), addition,
//!   multiplication; and the refresh of its 16 blocks with our key switch (tfhers_ours, as for Figure 4).
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin rns32_arith
//!       (REPS=5, CHECK=5, NE=1600, NPROD=25, MREPS=1, VARIANTS=4,16,8, TF=1, RADIX=1)
use eb_tfhers::comb::{cm, CombConfig, CombKeys, COMB_RNS, COMB_RNS32_B16, COMB_RNS32_B8, RNS32_MODULI, RNS_MODULI};
use eb_tfhers::eb::{tor, EbKeys};
use eb_tfhers::params::{stair_var, LOG2_SIGMA_N, N, NU, U128};
use eb_tfhers::product::*;
use eb_tfhers::tfhers_ours::TfhersBlocks;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::gen_keys_crt;
use tfhe::prelude::*;
use tfhe::shortint::parameters::v1_6::V1_6_PARAM_MESSAGE_5_CARRY_1_KS_PBS_GAUSSIAN_2M128 as P51;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint32};

type Ct = LweCiphertextOwned<u64>;

fn env_usize(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn env_on(name: &str) -> bool { std::env::var(name).map(|v| v != "0").unwrap_or(true) }

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

fn phase(sk: &[u64], c: &Ct) -> u64 {
    let m = c.get_mask(); let a = m.as_ref();
    a.iter().zip(sk).fold(*c.get_body().data, |p, (&x, &s)| p.wrapping_sub(x.wrapping_mul(s)))
}

fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f();
    (0..reps).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min)
}

struct Variant { tag: &'static str, mods: &'static [u64], comb: &'static CombConfig, pc: &'static PConfig }

struct Opts { reps: usize, nchk: usize, ne: usize, nprod: usize }

fn run_variant(v: &Variant, o: &Opts, csv: &mut Vec<String>) {
    let mods: Vec<u64> = v.mods.to_vec(); let r = mods.len();
    let mm: u128 = mods.iter().map(|&m| m as u128).product();
    let phmax = *mods.iter().max().unwrap();
    let (cfg, pc) = (v.comb, v.pc);
    let t0 = Instant::now();
    let seed = 7032;
    let comb = CombKeys::new_m(cfg, 1, seed);
    let (eb, mut egen) = EbKeys::new(&pc.cfg, pc.m, seed);
    assert_eq!(comb.big_sk.as_ref(), eb.big_sk.as_ref(), "one key s' for CombBoot and the error bootstrap");
    let bc = comb.bcols; assert!(bc % r == 0, "B must be a multiple of the number of residues");
    let ipc = bc / r;                                        // integers per CombBoot
    let phats: Vec<u64> = (0..ipc).flat_map(|_| mods.iter().copied()).collect();
    let sk: Vec<u64> = eb.big_sk.as_ref().to_vec();
    let mut rng = StdRng::seed_from_u64(32); let mut rbit = 0x853c49e6748fea9bu64;
    let pk = ProductKeys::new(&sk, pc.gtr, GRL, &mut rng);
    let mut w = pk.work();
    let tk = t0.elapsed().as_secs_f64();
    println!("  [{}] moduli {:?} (2^{:.2}): CombBoot B = {bc} (p = {}, {ipc} integer(s) per call), n = {}, RGSW (2^{},{}), aut (2^{},{}) shift {}; before a product: m = {}, n = {}, (2^{},{}), trace (2^{},{})   (keys {:.0} s)",
             v.tag, mods, (mm as f64).log2(), cfg.p, cfg.n, cfg.gbr.0, cfg.gbr.1, cfg.gks.0, cfg.gks.1, cfg.gks.2, pc.m, pc.cfg.n, pc.cfg.b, pc.cfg.l, pc.gtr.0, pc.gtr.1, tk);
    let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
    let enc1 = |x: u64, m: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Ct {
        allocate_and_encrypt_new_lwe_ciphertext(&eb.big_sk, Plaintext((((x as u128) << 64) / m as u128) as u64), noise, cm(), egen)
    };
    let enc = |x: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Vec<Ct> { mods.iter().map(|&m| enc1(x % m, m, egen)).collect() };
    let dec = |cs: &[Ct]| -> u128 {
        let res: Vec<u64> = cs.iter().zip(&mods).map(|(c, &m)| ((((phase(&sk, c) as u128) * m as u128 + (1u128 << 63)) >> 64) as u64) % m).collect();
        crt(&res, &mods)
    };
    // refresh of a list of integers, ipc per CombBoot (the last call padded with copies)
    let refresh = |ints: &[&[Ct]], rng: &mut StdRng, rbit: &mut u64| -> Vec<Vec<Ct>> {
        let mut out = Vec::new();
        for chunk in ints.chunks(ipc) {
            let cts: Vec<Ct> = (0..ipc).flat_map(|i| chunk[i.min(chunk.len() - 1)].iter().cloned()).collect();
            let (o, _) = comb.refresh_rns(&cts, &phats, rng, rbit);
            for i in 0..chunk.len() { out.push(o[i * r..(i + 1) * r].to_vec()); }
        }
        out
    };
    let add = |a: &[Ct], b: &[Ct], out: &mut [Ct]| { for k in 0..a.len() { lwe_ciphertext_add(&mut out[k], &a[k], &b[k]); } };
    for _ in 0..o.nchk {
        let (x, y) = (rng.gen::<u32>() as u64, rng.gen::<u32>() as u64);
        let (cx, cy) = (enc(x, &mut egen), enc(y, &mut egen));
        let rr = refresh(&[&cx, &cy], &mut rng, &mut rbit);
        let (xr, yr) = (&rr[0], &rr[1]);
        assert_eq!((dec(xr), dec(yr)), (x as u128, y as u128), "wrong refresh");
        let mut z = xr.clone(); add(xr, yr, &mut z);
        assert_eq!(dec(&z), (x as u128 + y as u128) % mm, "wrong addition");
        let xp: Vec<Ct> = xr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
        let yp: Vec<Ct> = yr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
        let pr: Vec<Ct> = (0..r).map(|k| pk.product(&mut w, &xp[k], &yp[k], mods[k])).collect();
        let want = (x as u128 * y as u128) % mm;
        assert_eq!(dec(&pr), want, "wrong multiplication");
        let rr2 = refresh(&[&pr, &z], &mut rng, &mut rbit);
        assert_eq!((dec(&rr2[0]), dec(&rr2[1])), (want, (x as u128 + y as u128) % mm), "wrong refresh after the operations");
    }
    println!("    check: {} pairs of 32-bit integers: refresh, x + y, x y (mod {}) and the refresh of x y and x + y, correct (CRT)", o.nchk, mm);
    // times
    let (x, y) = (rng.gen::<u32>() as u64, rng.gen::<u32>() as u64);
    let (cx, cy) = (enc(x, &mut egen), enc(y, &mut egen));
    let rr = refresh(&[&cx, &cy], &mut rng, &mut rbit);
    let (xr, yr) = (rr[0].clone(), rr[1].clone());
    let ints: Vec<&[Ct]> = (0..ipc).map(|i| if i % 2 == 0 { &cx[..] } else { &cy[..] }).collect();
    let t_call = best(o.reps, || { let _ = refresh(&ints, &mut rng, &mut rbit); });
    let mut z = xr.clone();
    let t_add = best(1000, || add(&xr, &yr, &mut z));
    let t_eb = best(o.reps, || { for k in 0..r { let _ = eb.refresh_rns(&xr[k], mods[k], &mut rng, &mut rbit); } });
    let xp: Vec<Ct> = xr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
    let yp: Vec<Ct> = yr.iter().zip(&mods).map(|(c, &m)| eb.refresh_rns(c, m, &mut rng, &mut rbit)).collect();
    let t_pr = best(o.reps, || { for k in 0..r { let _ = pk.product(&mut w, &xp[k], &yp[k], mods[k]); } });
    let t_int = t_call / ipc as f64;
    println!("    times: refresh {t_call:8.3} ms per CombBoot ({ipc} integer(s)), {t_int:8.3} ms per integer;  addition {t_add:8.5} ms ({:.5} with a thousandth of a refresh);  multiplication {:8.3} ms (refresh before a product {t_eb:.3} + product {t_pr:.3}, {r} residues: {:.2} + {:.2} per residue)",
             t_add + t_int / NU, t_eb + t_pr, t_eb / r as f64, t_pr / r as f64);
    csv.push(format!("rns{},32,refresh_call,{t_call:.4},{ipc}", v.tag)); csv.push(format!("rns{},32,refresh_int,{t_int:.4}", v.tag));
    csv.push(format!("rns{},32,add,{t_add:.6},{:.6}", v.tag, t_add + t_int / NU));
    csv.push(format!("rns{},32,mul,{:.4},{t_eb:.4},{t_pr:.4}", v.tag, t_eb + t_pr));
    // noise of the CombBoot (as rns_curves), margin for the largest modulus
    if o.ne > 0 {
        let p = cfg.p as u64; let s = comb.small_sk.as_ref().to_vec();
        let (mut s2, mut cnt, mut emx) = (0.0, 0usize, 0.0f64);
        while cnt < o.ne {
            let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
            let e: Vec<i64> = (0..bc).map(|_| rng.gen_range(-(p as i64) / 4..=(p as i64) / 4)).collect();
            let bints: Vec<u64> = (0..bc).map(|k| {
                let dot = aints[k].iter().zip(&s).fold(0u64, |acc, (&a, &x)| acc.wrapping_add(a * x));
                ((dot as i64 + e[k]).rem_euclid(p as i64)) as u64
            }).collect();
            let outs = comb.combboot_rns(&aints, &bints, &phats);
            for k in 0..bc {
                let phs = decrypt_lwe_ciphertext(&comb.big_sk, &outs[k]).0;
                let want = ((e[k] as f64) / (p as f64 * phats[k] as f64) * 2f64.powi(64)).round() as i64 as u64;
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
        let vks_ks = stair_var(N, cfg.chain, 16); let ph = phmax as f64;
        let marg = 0.5 * ((1.0 / (U128 * U128)) / ((4.0 * ph).powi(2) * (NU * vo + vks_ks) + cfg.vterm)).log2();
        println!("    CombBoot noise ({cnt} errors): out {:7.2} / {:7.2} predicted, max |e| 2^{:.2}, margin {:.2} bits at p̂ = {phmax} after {NU} additions (KS {:.2})",
                 0.5 * vo.log2(), 0.5 * pred.log2(), emx.log2(), marg, 0.5 * vks_ks.log2());
        csv.push(format!("rns{},32,comb_margin,{marg:.3}", v.tag));
    }
    // noise of the products (as rns_arith), inputs refreshed before a product
    if o.nprod > 0 {
        let t1 = Instant::now();
        let (mut ve, mut v0, mut tt, mut vp, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for _ in 0..o.nprod { for &m in mods.iter() {
            let (m1, m2) = (rng.gen_range(0..m), rng.gen_range(0..m));
            let c1 = eb.refresh_rns(&enc1(m1, m, &mut egen), m, &mut rng, &mut rbit);
            let c2 = eb.refresh_rns(&enc1(m2, m, &mut egen), m, &mut rng, &mut rbit);
            let mu = |x: u64| -> u64 { (((x as u128) << 64) / m as u128) as u64 };
            let p1 = phase(&sk, &c1);
            for (c, x) in [(&c1, m1), (&c2, m2)] { let e = tor(phase(&sk, c).wrapping_sub(mu(x))); ve += e * e; }
            let (a, b) = pk.lwe_to_rlwe(&mut w, &c1);
            let ph = phase64(&sk, &a, &b);
            let e0 = tor(ph[0].wrapping_sub(p1)); v0 += e0 * e0;
            tt += e0 * e0 + ph[1..].iter().map(|&x| tor(x) * tor(x)).sum::<f64>();
            let out = pk.product(&mut w, &c1, &c2, m);
            let e = tor(phase(&sk, &out).wrapping_sub(mu((m1 * m2) % m))); vp += e * e; cnt += 1;
        } }
        let np = cnt as f64; ve /= 2.0 * np; v0 /= np; tt /= np; vp /= np;
        let vks = stair_var(N, pc.cfg.chain, 16);
        let l2 = |x: f64| 0.5 * x.log2();
        println!("    product noise ({cnt} products, moduli in turn): in {:7.2}   √T {:7.2}   √V0 {:7.2}   product {:7.2} (predicted at p̂ = {phmax} from the measured inputs {:7.2})   KS {:7.2} predicted   ({:.0} s)",
                 l2(ve), l2(tt), l2(v0), l2(vp), l2(var_product(phmax as f64, ve, tt, v0)), l2(vks), t1.elapsed().as_secs_f64());
        csv.push(format!("noise,{phmax},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", ve, tt, v0, vp, vks, pc.cfg.vterm));
    }
}

fn main() {
    let o = Opts { reps: env_usize("REPS", 5), nchk: env_usize("CHECK", 5), ne: env_usize("NE", 1600), nprod: env_usize("NPROD", 25) };
    let mreps = env_usize("MREPS", 1);
    let (tf, radix) = (env_on("TF"), env_on("RADIX"));
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let all = [
        Variant { tag: "4", mods: &RNS_MODULI[..4], comb: &COMB_RNS, pc: &PCONFIG_RNS },
        Variant { tag: "16", mods: &RNS32_MODULI, comb: &COMB_RNS32_B16, pc: &PCONFIG_RNS32 },
        Variant { tag: "8", mods: &RNS32_MODULI, comb: &COMB_RNS32_B8, pc: &PCONFIG_RNS32 },
    ];
    let want: Vec<String> = std::env::var("VARIANTS").unwrap_or("4,16,8".into()).split(',').map(|s| s.trim().to_string()).collect();
    println!("RNS arithmetic on 32-bit integers, TFHE-rs 1.8.1, one thread, N = {N}; variants {:?} (4: 4 moduli up to 263, B = 8; 16: 8 moduli up to 25, B = 16; 8: the same, B = 8)", want);
    let mut csv: Vec<String> = Vec::new();
    for v in all.iter().filter(|v| want.iter().any(|t| t == v.tag)) { run_variant(v, &o, &mut csv); }
    let reps = o.reps; let nchk = o.nchk;
    if tf {
        // CRT integers of TFHE-rs
        let basis: Vec<u64> = std::env::var("CRT_BASIS").ok().map(|s| s.split(',').map(|v| v.trim().parse().unwrap()).collect())
            .unwrap_or(vec![19, 23, 25, 27, 29, 31, 32]);
        let mb: u128 = basis.iter().map(|&m| m as u128).product();
        assert!(basis.iter().all(|&m| m <= P51.message_modulus.0), "residues below the message modulus");
        assert!(mb >= 1u128 << 32, "the basis must hold 32 bits");
        let t0 = Instant::now();
        let (ck, sk) = gen_keys_crt(P51, basis.clone());
        let tk = t0.elapsed().as_secs_f64();
        let mut rng = StdRng::seed_from_u64(3232);
        for _ in 0..nchk {
            let (x, y) = (rng.gen::<u32>() as u64, rng.gen::<u32>() as u64);
            let (a, b) = (ck.encrypt(x), ck.encrypt(y));
            let mut s = sk.unchecked_crt_add(&a, &b); sk.full_extract_message_assign(&mut s);
            assert_eq!(ck.decrypt(&s) as u128, (x as u128 + y as u128) % mb, "TFHE-rs CRT: wrong addition");
            let mut p = sk.unchecked_crt_mul(&a, &b); sk.full_extract_message_assign(&mut p);
            let want = (x as u128 * y as u128) % mb;
            assert_eq!(ck.decrypt(&p) as u128, want, "TFHE-rs CRT: wrong multiplication");
            sk.full_extract_message_assign(&mut p);
            assert_eq!(ck.decrypt(&p) as u128, want, "TFHE-rs CRT: wrong refresh");
        }
        let (x, y) = (rng.gen::<u32>() as u64, rng.gen::<u32>() as u64);
        let (a, b) = (ck.encrypt(x), ck.encrypt(y));
        let mut r = a.clone();
        let tr = best(reps, || sk.full_extract_message_assign(&mut r));
        let ta = best(1000, || { let _ = sk.unchecked_crt_add(&a, &b); });
        let tm = best(mreps.max(1), || { let _ = sk.unchecked_crt_mul(&a, &b); });
        let tme = best(mreps.max(1), || { let mut p = sk.unchecked_crt_mul(&a, &b); sk.full_extract_message_assign(&mut p); });
        // one key switch of the set, N -> n (core_crypto, as in shortint)
        let (n, nn) = (P51.lwe_dimension.0, P51.polynomial_size.0 * P51.glwe_dimension.0);
        let mut boxed = new_seeder(); let seeder = boxed.as_mut();
        let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(51));
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
        let big = LweSecretKey::generate_new_binary(LweDimension(nn), &mut sgen);
        let small = LweSecretKey::generate_new_binary(LweDimension(n), &mut sgen);
        let ksk = allocate_and_generate_new_lwe_keyswitch_key(&big, &small, P51.ks_base_log, P51.ks_level, P51.lwe_noise_distribution, cm(), &mut egen);
        let ct = allocate_and_encrypt_new_lwe_ciphertext(&big, Plaintext(1u64 << 58), P51.lwe_noise_distribution, cm(), &mut egen);
        let mut out = LweCiphertext::new(0u64, LweSize(n + 1), cm());
        let tks = best(reps * 4, || keyswitch_lwe_ciphertext(&ksk, &ct, &mut out));
        let nb = basis.len() as f64;
        println!("  check: {nchk} pairs of 32-bit integers, TFHE-rs CRT: x + y, x y (mod {mb}) and the refresh of x y, correct");
        println!("  tfhers CRT {:?} ({} blocks, 2^{:.2}, MESSAGE_5_CARRY_1, n = {n}, N = {nn}): refresh {tr:9.3} ms ({:.2} per block);  addition {ta:8.5} ms;  multiplication {tm:9.3} ms, {tme:9.3} ms with the refresh   (keys {tk:.0} s)",
                 basis, basis.len(), (mb as f64).log2(), tr / nb);
        println!("     key switch {tks:.3} ms ({:.0} % of a block refresh); without KS: refresh {:9.3} ms, multiplication with the refresh {:9.3} ms",
                 100.0 * tks / (tr / nb), tr - nb * tks, tme - 3.0 * nb * tks);
        csv.push(format!("tfhers_crt,32,refresh,{tr:.4}")); csv.push(format!("tfhers_crt,32,add,{ta:.6}"));
        csv.push(format!("tfhers_crt,32,mul,{tme:.4},{tm:.4}")); csv.push(format!("tfhers_crt,32,ks,{tks:.4}"));
        csv.push(format!("tfhers_crt,32,refresh_noks,{:.4}", tr - nb * tks)); csv.push(format!("tfhers_crt,32,mul_noks,{:.4}", tme - 3.0 * nb * tks));
        drop(ck); drop(sk); drop(ksk);
        if radix {
            // radix FheUint32 of TFHE-rs (default parameters), and its 16 blocks refreshed with our key switch
            let (ck, sk) = generate_keys(ConfigBuilder::default().build());
            set_server_key(sk);
            let (x, y) = (0xB5E3_6A7Du32, 0x0F1E_2C4Du32);
            let (a, b) = (FheUint32::encrypt(x, &ck), FheUint32::encrypt(y, &ck));
            let r: u32 = (&a & u32::MAX).decrypt(&ck); assert_eq!(r, x);
            let r: u32 = (&a + &b).decrypt(&ck); assert_eq!(r, x.wrapping_add(y));
            let r: u32 = (&a * &b).decrypt(&ck); assert_eq!(r, x.wrapping_mul(y));
            let tr = best(reps, || { let _ = &a & u32::MAX; });
            let ta = best(reps, || { let _ = &a + &b; });
            let tm = best(mreps, || { let _ = &a * &b; });
            let blocks = TfhersBlocks::new(9032);
            let (cts, ms) = blocks.encrypt(16, 5);
            let mut rb = 0x9e3779b97f4a7c15u64;
            blocks.check(&blocks.refresh(true, &cts, &mut rb), &ms);
            let to = best(reps, || { let _ = blocks.refresh(true, &cts, &mut rb); });
            let tc = best(reps, || { let _ = blocks.refresh(false, &cts, &mut rb); });
            println!("  tfhers FheUint32 (radix, 16 blocks): refresh (&a & MAX) {tr:9.3} ms;  addition {ta:9.3} ms;  multiplication {tm:10.3} ms;  16 blocks refreshed: {tc:8.3} ms with its key switch, {to:8.3} ms with ours");
            csv.push(format!("tfhers,32,refresh,{tr:.4}")); csv.push(format!("tfhers,32,add,{ta:.4}")); csv.push(format!("tfhers,32,mul,{tm:.4}"));
            csv.push(format!("tfhers,32,refresh_blocks_theirs,{tc:.4}")); csv.push(format!("tfhers,32,refresh_blocks_ours,{to:.4}"));
        }
    }
    println!("CSV"); for l in csv { println!("{l}"); }
}
