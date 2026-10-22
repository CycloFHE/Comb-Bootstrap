//! RNS arithmetic on 16- and 32-bit integers with small moduli, against the CRT integers of TFHE-rs, one thread.
//! Ours: x held as its residues modulo pairwise coprime moduli p̂_k, residue k encrypted as LWE_{s'}(r_k/p̂_k) of
//! dimension N = 2048, all under one key s'. Two CombBoot configurations (comb.rs, chosen by the noise model):
//!   R, refresh: one CombBoot refreshes the residues of one integer (B = number of residues);
//!   P, refresh before a product: a valid refresh whose outputs, taken as the two operands of the product of Section 6
//!      (trace (2^8,5), relinearization (2^10,4)), are refreshed again by P after the product at failure 2^-128.
//! Multiplication: the residues are multiplied one by one (product::ProductKeys::product, Section 6), the two operands
//! coming from two distinct P calls (independent errors); as for TFHE-rs (unchecked_crt_mul + one refresh), its time
//! counts one refresh, by P, which makes the result ready for another product: mul = P call + products. The time
//! with both operands refreshed (2 P calls + products) is printed too.
//! Variants (VARIANTS, default 32,16):
//!   32:  RNS32_MODULI {25,23,21,19,17,16,13,11} (2^33.05), B = 8: R COMB_RNS32_B8, P COMB_RNS32P_B8;
//!   16:  RNS16_MODULI {19,17,16,15} (2^16.24), B = 4: R COMB_RNS16_B4, P COMB_RNS16P_B4;
//!   32w: the same moduli, B = 16, two integers per call (COMB_RNS32_B16, COMB_RNS32P_B16);
//!   16w: B = 8, two integers per call (COMB_RNS16_B8, COMB_RNS16P_B8).
//!   check:  CHECK pairs of random integers (default 5): refresh, x + y, x y modulo the product of the moduli, and the
//!           refresh of x y (R and P), decrypted and recombined by the CRT;
//!   noise:  NE output errors of each CombBoot (default 1600; noise-free inputs modulo p, E_k uniform in |E_k| ≤ p/4),
//!           predicted, and margin at failure 2^-128 after 1000 additions for the largest modulus; NPROD pairs of P calls
//!           (default 25), one product per modulus each: line "noise,P̂MAX,ve,T,V0,vp,vks,vterm" (vks, vterm of P, the
//!           refresh after the product) for the noise model.
//! TFHE-rs, CRT (integer::gen_keys_crt), the fastest set for each size (algorithmic cost of the set times the blocks):
//!   16 bits: V1_6_PARAM_MESSAGE_4_CARRY_1_KS_PBS_GAUSSIAN_2M128, basis 16,15,13,11,7 (5 blocks, 2^17.87; four moduli
//!            up to 16 reach only 2^15.07), CRT16 to change it (CRT16SET=51: MESSAGE_5_CARRY_1, 32,31,29,27, for comparison);
//!   32 bits: V1_6_PARAM_MESSAGE_5_CARRY_1_KS_PBS_GAUSSIAN_2M128, basis 19,23,25,27,29,31,32 (7 blocks, 2^32.98), CRT32.
//!   refresh = full_extract_message_assign (1 PBS per block), addition = unchecked_crt_add (no PBS), multiplication =
//!   unchecked_crt_mul (2 PBS per block, (x+y)^2/4 - (x-y)^2/4, the carry being 1 bit) followed by one refresh;
//!   the key switch of the set (N -> n) is timed apart: "without KS" subtracts it from each PBS.
//! TFHE-rs radix (RADIX=1): FheUint16 / FheUint32, default parameters: refresh (&a & MAX), addition, multiplication.
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin rns_small_arith | tee ../../results/rust/rns_small_mac.txt
//!       (REPS=5, CHECK=5, NE=1600, NPROD=25, MREPS=1, VARIANTS=32,16, TF=1, RADIX=1)
use eb_tfhers::comb::*;
use eb_tfhers::eb::tor;
use eb_tfhers::params::{stair_var, LOG2_SIGMA_N, N, NU, U128};
use eb_tfhers::product::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::gen_keys_crt;
use tfhe::prelude::*;
use tfhe::shortint::parameters::v1_6::V1_6_PARAM_MESSAGE_4_CARRY_1_KS_PBS_GAUSSIAN_2M128 as P41;
use tfhe::shortint::parameters::v1_6::V1_6_PARAM_MESSAGE_5_CARRY_1_KS_PBS_GAUSSIAN_2M128 as P51;
use tfhe::shortint::parameters::ClassicPBSParameters;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint16, FheUint32};

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

struct Variant { tag: &'static str, bits: u32, mods: &'static [u64], r: &'static CombConfig, p: &'static CombConfig }

struct Opts { reps: usize, nchk: usize, ne: usize, nprod: usize }

/// Output noise of a CombBoot (as rns_curves): measured and predicted variance, max |e|, margin after NU additions.
fn comb_noise(comb: &CombKeys, cfg: &CombConfig, phats: &[u64], ne: usize, rng: &mut StdRng) -> (f64, f64, f64, f64, f64) {
    let bc = comb.bcols; let p = cfg.p as u64; let s = comb.small_sk.as_ref().to_vec();
    let (mut s2, mut cnt, mut emx) = (0.0, 0usize, 0.0f64);
    while cnt < ne {
        let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
        let e: Vec<i64> = (0..bc).map(|_| rng.gen_range(-(p as i64) / 4..=(p as i64) / 4)).collect();
        let bints: Vec<u64> = (0..bc).map(|k| {
            let dot = aints[k].iter().zip(&s).fold(0u64, |acc, (&a, &x)| acc.wrapping_add(a * x));
            ((dot as i64 + e[k]).rem_euclid(p as i64)) as u64
        }).collect();
        let outs = comb.combboot_rns(&aints, &bints, phats);
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
    let vks_ks = stair_var(N, cfg.chain, 16); let ph = *phats.iter().max().unwrap() as f64;
    let marg = 0.5 * ((1.0 / (U128 * U128)) / ((4.0 * ph).powi(2) * (NU * vo + vks_ks) + cfg.vterm)).log2();
    (vo, pred, emx, marg, vks_ks)
}

fn run_variant(v: &Variant, o: &Opts, csv: &mut Vec<String>) {
    let mods: Vec<u64> = v.mods.to_vec(); let r = mods.len(); let bits = v.bits;
    let mm: u128 = mods.iter().map(|&m| m as u128).product();
    assert!(mm >= 1u128 << bits, "the moduli must hold the integers");
    let phmax = *mods.iter().max().unwrap();
    let t0 = Instant::now();
    let seed = 7016;
    let cr = CombKeys::new_m(v.r, 1, seed);
    let cp = CombKeys::new_m(v.p, 1, seed);
    assert_eq!(cr.big_sk.as_ref(), cp.big_sk.as_ref(), "one key s' for both CombBoots");
    let bc = cr.bcols; assert!(bc == cp.bcols && bc % r == 0, "B must be a multiple of the number of residues");
    let ipc = bc / r;
    let phats: Vec<u64> = (0..ipc).flat_map(|_| mods.iter().copied()).collect();
    let sk: Vec<u64> = cr.big_sk.as_ref().to_vec();
    let mut rng = StdRng::seed_from_u64(16 + bits as u64); let mut rbit = 0x853c49e6748fea9bu64;
    let pk = ProductKeys::new(&sk, (8, 5), GRL, &mut rng);
    let mut w = pk.work();
    let mut boxed = new_seeder(); let seeder = boxed.as_mut();
    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
    let tk = t0.elapsed().as_secs_f64();
    let pr_cfg = |c: &CombConfig| format!("n = {}, RGSW (2^{},{}), aut (2^{},{}) shift {}", c.n, c.gbr.0, c.gbr.1, c.gks.0, c.gks.1, c.gks.2);
    println!("  [{}] {bits}-bit integers, moduli {:?} (2^{:.2}), B = {bc} (p = {}, {ipc} integer(s) per call); R: {}; P: {}; product: trace (2^8,5), relin (2^{},{})   (keys {tk:.0} s)",
             v.tag, mods, (mm as f64).log2(), v.r.p, pr_cfg(v.r), pr_cfg(v.p), GRL.0, GRL.1);
    let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
    let enc1 = |x: u64, m: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Ct {
        allocate_and_encrypt_new_lwe_ciphertext(&cr.big_sk, Plaintext((((x as u128) << 64) / m as u128) as u64), noise, cm(), egen)
    };
    let enc = |x: u64, egen: &mut EncryptionRandomGenerator<DefaultRandomGenerator>| -> Vec<Ct> { mods.iter().map(|&m| enc1(x % m, m, egen)).collect() };
    let dec = |cs: &[Ct]| -> u128 {
        let res: Vec<u64> = cs.iter().zip(&mods).map(|(c, &m)| ((((phase(&sk, c) as u128) * m as u128 + (1u128 << 63)) >> 64) as u64) % m).collect();
        crt(&res, &mods)
    };
    // refresh of a list of integers by the CombBoot c, ipc per call (the last call padded with copies)
    let refresh = |c: &CombKeys, ints: &[&[Ct]], rng: &mut StdRng, rbit: &mut u64| -> Vec<Vec<Ct>> {
        let mut out = Vec::new();
        for chunk in ints.chunks(ipc) {
            let cts: Vec<Ct> = (0..ipc).flat_map(|i| chunk[i.min(chunk.len() - 1)].iter().cloned()).collect();
            let (o, _) = c.refresh_rns(&cts, &phats, rng, rbit);
            for i in 0..chunk.len() { out.push(o[i * r..(i + 1) * r].to_vec()); }
        }
        out
    };
    let add = |a: &[Ct], b: &[Ct], out: &mut [Ct]| { for k in 0..a.len() { lwe_ciphertext_add(&mut out[k], &a[k], &b[k]); } };
    let rand_int = |rng: &mut StdRng| -> u64 { if bits == 32 { rng.gen::<u32>() as u64 } else { rng.gen::<u16>() as u64 } };
    for _ in 0..o.nchk {
        let (x, y) = (rand_int(&mut rng), rand_int(&mut rng));
        let (cx, cy) = (enc(x, &mut egen), enc(y, &mut egen));
        let rr = refresh(&cr, &[&cx, &cy], &mut rng, &mut rbit);
        assert_eq!((dec(&rr[0]), dec(&rr[1])), (x as u128, y as u128), "wrong refresh (R)");
        let mut z = rr[0].clone(); add(&rr[0], &rr[1], &mut z);
        assert_eq!(dec(&z), (x as u128 + y as u128) % mm, "wrong addition");
        // the two operands in two distinct P calls
        let xp = refresh(&cp, &[&z], &mut rng, &mut rbit).remove(0);
        let yp = refresh(&cp, &[&cy], &mut rng, &mut rbit).remove(0);
        let pr: Vec<Ct> = (0..r).map(|k| pk.product(&mut w, &xp[k], &yp[k], mods[k])).collect();
        let want = ((x as u128 + y as u128) % mm * y as u128) % mm;
        assert_eq!(dec(&pr), want, "wrong multiplication");
        let (pp, pr2) = (refresh(&cp, &[&pr], &mut rng, &mut rbit).remove(0), refresh(&cr, &[&pr], &mut rng, &mut rbit).remove(0));
        assert_eq!((dec(&pp), dec(&pr2)), (want, want), "wrong refresh after the product");
    }
    println!("    check: {} pairs of {bits}-bit integers: refresh, x + y, (x + y) y (mod {mm}) with the operands from two P calls, and the refresh of the product by P and R, correct (CRT)", o.nchk);
    // times
    let (x, y) = (rand_int(&mut rng), rand_int(&mut rng));
    let (cx, cy) = (enc(x, &mut egen), enc(y, &mut egen));
    let ints: Vec<&[Ct]> = (0..ipc).map(|i| if i % 2 == 0 { &cx[..] } else { &cy[..] }).collect();
    let t_r = best(o.reps, || { let _ = refresh(&cr, &ints, &mut rng, &mut rbit); });
    let t_p = best(o.reps, || { let _ = refresh(&cp, &ints, &mut rng, &mut rbit); });
    let xp = refresh(&cp, &[&cx], &mut rng, &mut rbit).remove(0);
    let yp = refresh(&cp, &[&cy], &mut rng, &mut rbit).remove(0);
    let mut z = xp.clone();
    let t_add = best(1000, || add(&xp, &yp, &mut z));
    let t_pr = best(o.reps, || { for k in 0..r { let _ = pk.product(&mut w, &xp[k], &yp[k], mods[k]); } });
    let (ri, pi) = (t_r / ipc as f64, t_p / ipc as f64);
    println!("    times: refresh (R) {t_r:8.3} ms per call ({ipc} integer(s)), {ri:8.3} ms per integer;  refresh before a product (P) {t_p:8.3} ms per call, {pi:8.3} per integer");
    println!("           addition {t_add:8.5} ms ({:.5} with a thousandth of a refresh);  products {t_pr:7.3} ms ({r} residues, {:.3} each);  multiplication {:8.3} ms (one P refresh + products),  {:8.3} ms with both operands refreshed",
             t_add + ri / NU, t_pr / r as f64, pi + t_pr, 2.0 * pi + t_pr);
    let tg = format!("rns{}", v.tag);
    csv.push(format!("{tg},{bits},refresh_call,{t_r:.4},{ipc}")); csv.push(format!("{tg},{bits},refresh_int,{ri:.4}"));
    csv.push(format!("{tg},{bits},refresh_p_call,{t_p:.4},{ipc}")); csv.push(format!("{tg},{bits},refresh_p_int,{pi:.4}"));
    csv.push(format!("{tg},{bits},add,{t_add:.6},{:.6}", t_add + ri / NU));
    csv.push(format!("{tg},{bits},mul,{:.4},{pi:.4},{t_pr:.4}", pi + t_pr)); csv.push(format!("{tg},{bits},mul_both,{:.4}", 2.0 * pi + t_pr));
    // noise of the two CombBoots
    if o.ne > 0 {
        for (name, c, cfg) in [("R", &cr, v.r), ("P", &cp, v.p)] {
            let (vo, pred, emx, marg, vks) = comb_noise(c, cfg, &phats, o.ne, &mut rng);
            println!("    CombBoot {name} noise ({} errors): out {:7.2} / {:7.2} predicted, max |e| 2^{:.2}, margin {:.2} bits at p̂ = {phmax} after {NU} additions (KS {:.2})",
                     o.ne, 0.5 * vo.log2(), 0.5 * pred.log2(), emx.log2(), marg, 0.5 * vks.log2());
            csv.push(format!("{tg},{bits},comb_{name},{:.3},{:.3},{marg:.3}", 0.5 * vo.log2(), 0.5 * pred.log2()));
        }
    }
    // noise of the products: operands from two distinct P calls; the refresh after the product is P
    if o.nprod > 0 {
        let t1 = Instant::now();
        let (mut ve, mut v0, mut tt, mut vp, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for _ in 0..o.nprod {
            let (m1, m2): (Vec<u64>, Vec<u64>) = (mods.iter().map(|&m| rng.gen_range(0..m)).collect(), mods.iter().map(|&m| rng.gen_range(0..m)).collect());
            let c1v: Vec<Ct> = mods.iter().zip(&m1).map(|(&m, &a)| enc1(a, m, &mut egen)).collect();
            let c2v: Vec<Ct> = mods.iter().zip(&m2).map(|(&m, &a)| enc1(a, m, &mut egen)).collect();
            let c1v = refresh(&cp, &[&c1v], &mut rng, &mut rbit).remove(0);
            let c2v = refresh(&cp, &[&c2v], &mut rng, &mut rbit).remove(0);
            for k in 0..r {
                let m = mods[k];
                let mu = |x: u64| -> u64 { (((x as u128) << 64) / m as u128) as u64 };
                let (c1, c2) = (&c1v[k], &c2v[k]);
                let p1 = phase(&sk, c1);
                for (c, x) in [(c1, m1[k]), (c2, m2[k])] { let e = tor(phase(&sk, c).wrapping_sub(mu(x))); ve += e * e; }
                let (a, b) = pk.lwe_to_rlwe(&mut w, c1);
                let ph = phase64(&sk, &a, &b);
                let e0 = tor(ph[0].wrapping_sub(p1)); v0 += e0 * e0;
                tt += e0 * e0 + ph[1..].iter().map(|&x| tor(x) * tor(x)).sum::<f64>();
                let out = pk.product(&mut w, c1, c2, m);
                let e = tor(phase(&sk, &out).wrapping_sub(mu((m1[k] * m2[k]) % m))); vp += e * e; cnt += 1;
            }
        }
        let np = cnt as f64; ve /= 2.0 * np; v0 /= np; tt /= np; vp /= np;
        let vks = stair_var(N, v.p.chain, 16);
        let l2 = |x: f64| 0.5 * x.log2();
        println!("    product noise ({cnt} products, operands from two P calls): in {:7.2}   √T {:7.2}   √V0 {:7.2}   product {:7.2} (predicted at p̂ = {phmax} from the measured inputs {:7.2})   KS of P {:7.2}   ({:.0} s)",
                 l2(ve), l2(tt), l2(v0), l2(vp), l2(var_product(phmax as f64, ve, tt, v0)), l2(vks), t1.elapsed().as_secs_f64());
        csv.push(format!("noise,{phmax},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", ve, tt, v0, vp, vks, v.p.vterm));
    }
}

/// CRT integers of TFHE-rs with the set prm and the basis: check, refresh, addition, multiplication, key switch.
fn tfhers_crt(bits: u32, prm: ClassicPBSParameters, name: &str, basis: Vec<u64>, reps: usize, mreps: usize, nchk: usize, csv: &mut Vec<String>) {
    let mb: u128 = basis.iter().map(|&m| m as u128).product();
    assert!(basis.iter().all(|&m| m <= prm.message_modulus.0), "residues below the message modulus");
    assert!(mb >= 1u128 << bits, "the basis must hold the integers");
    let t0 = Instant::now();
    let (ck, sk) = gen_keys_crt(prm, basis.clone());
    let tk = t0.elapsed().as_secs_f64();
    let mut rng = StdRng::seed_from_u64(3200 + bits as u64);
    let rand_int = |rng: &mut StdRng| -> u64 { if bits == 32 { rng.gen::<u32>() as u64 } else { rng.gen::<u16>() as u64 } };
    for _ in 0..nchk {
        let (x, y) = (rand_int(&mut rng), rand_int(&mut rng));
        let (a, b) = (ck.encrypt(x), ck.encrypt(y));
        let mut s = sk.unchecked_crt_add(&a, &b); sk.full_extract_message_assign(&mut s);
        assert_eq!(ck.decrypt(&s) as u128, (x as u128 + y as u128) % mb, "TFHE-rs CRT: wrong addition");
        let mut p = sk.unchecked_crt_mul(&a, &b); sk.full_extract_message_assign(&mut p);
        let want = (x as u128 * y as u128) % mb;
        assert_eq!(ck.decrypt(&p) as u128, want, "TFHE-rs CRT: wrong multiplication");
        sk.full_extract_message_assign(&mut p);
        assert_eq!(ck.decrypt(&p) as u128, want, "TFHE-rs CRT: wrong refresh");
    }
    let (x, y) = (rand_int(&mut rng), rand_int(&mut rng));
    let (a, b) = (ck.encrypt(x), ck.encrypt(y));
    let mut r = a.clone();
    let tr = best(reps, || sk.full_extract_message_assign(&mut r));
    let ta = best(1000, || { let _ = sk.unchecked_crt_add(&a, &b); });
    let tm = best(mreps.max(1), || { let _ = sk.unchecked_crt_mul(&a, &b); });
    let tme = best(mreps.max(1), || { let mut p = sk.unchecked_crt_mul(&a, &b); sk.full_extract_message_assign(&mut p); });
    // one key switch of the set, N -> n (core_crypto, as in shortint)
    let (n, nn) = (prm.lwe_dimension.0, prm.polynomial_size.0 * prm.glwe_dimension.0);
    let mut boxed = new_seeder(); let seeder = boxed.as_mut();
    let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(51));
    let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
    let big = LweSecretKey::generate_new_binary(LweDimension(nn), &mut sgen);
    let small = LweSecretKey::generate_new_binary(LweDimension(n), &mut sgen);
    let ksk = allocate_and_generate_new_lwe_keyswitch_key(&big, &small, prm.ks_base_log, prm.ks_level, prm.lwe_noise_distribution, cm(), &mut egen);
    let ct = allocate_and_encrypt_new_lwe_ciphertext(&big, Plaintext(1u64 << 58), prm.lwe_noise_distribution, cm(), &mut egen);
    let mut out = LweCiphertext::new(0u64, LweSize(n + 1), cm());
    let tks = best(reps * 4, || keyswitch_lwe_ciphertext(&ksk, &ct, &mut out));
    let nb = basis.len() as f64;
    println!("  check: {nchk} pairs of {bits}-bit integers, TFHE-rs CRT: x + y, x y (mod {mb}) and the refresh of x y, correct");
    println!("  tfhers CRT {bits} bits {:?} ({} blocks, 2^{:.2}, {name}, n = {n}, N = {nn}): refresh {tr:9.3} ms ({:.2} per block);  addition {ta:8.5} ms;  multiplication {tm:9.3} ms, {tme:9.3} ms with the refresh   (keys {tk:.0} s)",
             basis, basis.len(), (mb as f64).log2(), tr / nb);
    println!("     key switch {tks:.3} ms ({:.0} % of a block refresh); without KS: refresh {:9.3} ms, multiplication with the refresh {:9.3} ms",
             100.0 * tks / (tr / nb), tr - nb * tks, tme - 3.0 * nb * tks);
    csv.push(format!("tfhers_crt,{bits},refresh,{tr:.4},{}", basis.len())); csv.push(format!("tfhers_crt,{bits},add,{ta:.6}"));
    csv.push(format!("tfhers_crt,{bits},mul,{tme:.4},{tm:.4}")); csv.push(format!("tfhers_crt,{bits},ks,{tks:.4}"));
    csv.push(format!("tfhers_crt,{bits},refresh_noks,{:.4}", tr - nb * tks)); csv.push(format!("tfhers_crt,{bits},mul_noks,{:.4}", tme - 3.0 * nb * tks));
}

fn basis_env(name: &str, d: &[u64]) -> Vec<u64> {
    std::env::var(name).ok().map(|s| s.split(',').map(|v| v.trim().parse().unwrap()).collect()).unwrap_or(d.to_vec())
}

fn main() {
    let o = Opts { reps: env_usize("REPS", 5), nchk: env_usize("CHECK", 5), ne: env_usize("NE", 1600), nprod: env_usize("NPROD", 25) };
    let mreps = env_usize("MREPS", 1);
    let (tf, radix) = (env_on("TF"), env_on("RADIX"));
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    let all = [
        Variant { tag: "32", bits: 32, mods: &RNS32_MODULI, r: &COMB_RNS32_B8, p: &COMB_RNS32P_B8 },
        Variant { tag: "16", bits: 16, mods: &RNS16_MODULI, r: &COMB_RNS16_B4, p: &COMB_RNS16P_B4 },
        Variant { tag: "32w", bits: 32, mods: &RNS32_MODULI, r: &COMB_RNS32_B16, p: &COMB_RNS32P_B16 },
        Variant { tag: "16w", bits: 16, mods: &RNS16_MODULI, r: &COMB_RNS16_B8, p: &COMB_RNS16P_B8 },
    ];
    let want: Vec<String> = std::env::var("VARIANTS").unwrap_or("32,16".into()).split(',').map(|s| s.trim().to_string()).collect();
    println!("RNS arithmetic on 16- and 32-bit integers with small moduli, TFHE-rs 1.8.1, one thread, N = {N}; variants {:?}", want);
    let mut csv: Vec<String> = Vec::new();
    for v in all.iter().filter(|v| want.iter().any(|t| t == v.tag)) { run_variant(v, &o, &mut csv); }
    let sizes: Vec<u32> = [32u32, 16].into_iter().filter(|b| all.iter().any(|v| v.bits == *b && want.iter().any(|t| t == v.tag))).collect();
    if tf {
        for &bits in &sizes {
            if bits == 32 {
                tfhers_crt(32, P51, "MESSAGE_5_CARRY_1", basis_env("CRT32", &[19, 23, 25, 27, 29, 31, 32]), o.reps, mreps, o.nchk, &mut csv);
            } else {
                if std::env::var("CRT16SET").map(|v| v == "51").unwrap_or(false) {   // comparison: 4 blocks of the 32-bit set
                    tfhers_crt(16, P51, "MESSAGE_5_CARRY_1", basis_env("CRT16", &[32, 31, 29, 27]), o.reps, mreps, o.nchk, &mut csv);
                } else {
                    tfhers_crt(16, P41, "MESSAGE_4_CARRY_1", basis_env("CRT16", &[16, 15, 13, 11, 7]), o.reps, mreps, o.nchk, &mut csv);
                }
            }
        }
    }
    if radix {
        let (ck, sk) = generate_keys(ConfigBuilder::default().build());
        set_server_key(sk);
        for &bits in &sizes {
            let (tr, ta, tm) = if bits == 32 {
                let (x, y) = (0xB5E3_6A7Du32, 0x0F1E_2C4Du32);
                let (a, b) = (FheUint32::encrypt(x, &ck), FheUint32::encrypt(y, &ck));
                let r: u32 = (&a * &b).decrypt(&ck); assert_eq!(r, x.wrapping_mul(y));
                (best(o.reps, || { let _ = &a & u32::MAX; }), best(o.reps, || { let _ = &a + &b; }), best(mreps, || { let _ = &a * &b; }))
            } else {
                let (x, y) = (0xB5E3u16, 0x2C4Du16);
                let (a, b) = (FheUint16::encrypt(x, &ck), FheUint16::encrypt(y, &ck));
                let r: u16 = (&a * &b).decrypt(&ck); assert_eq!(r, x.wrapping_mul(y));
                (best(o.reps, || { let _ = &a & u16::MAX; }), best(o.reps, || { let _ = &a + &b; }), best(mreps, || { let _ = &a * &b; }))
            };
            println!("  tfhers FheUint{bits} (radix, {} blocks, its key switch): refresh (&a & MAX) {tr:9.3} ms;  addition {ta:9.3} ms;  multiplication {tm:10.3} ms", bits / 2);
            csv.push(format!("tfhers,{bits},refresh,{tr:.4}")); csv.push(format!("tfhers,{bits},add,{ta:.4}")); csv.push(format!("tfhers,{bits},mul,{tm:.4}"));
        }
    }
    println!("CSV"); for l in csv { println!("{l}"); }
}
