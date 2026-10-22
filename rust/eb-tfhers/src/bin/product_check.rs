//! Checks of the product on TFHE-rs (src/product.rs): exactness of the key switchings of the trace, phase of the trace,
//! and products of LWE ciphertexts against p̂ μ1 μ2.
use eb_tfhers::params::N;
use eb_tfhers::product::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use tfhe::core_crypto::prelude::*;

fn enc(sk: &[u64], mu: u64, sig: f64, rng: &mut StdRng) -> LweCiphertextOwned<u64> {
    let g = Normal::new(0.0, sig * 18446744073709551616.0).unwrap();
    let mut c: Vec<u64> = (0..N).map(|_| rng.gen()).collect();
    let mut b = mu.wrapping_add(g.sample(rng).round() as i64 as u64);
    for i in 0..N { b = b.wrapping_add(c[i].wrapping_mul(sk[i])); }
    c.push(b);
    LweCiphertext::from_container(c, CiphertextModulus::new_native())
}
fn dec(sk: &[u64], c: &LweCiphertextOwned<u64>) -> u64 {
    let m = c.get_mask(); let a = m.as_ref();
    let mut p = *c.get_body().data;
    for i in 0..N { p = p.wrapping_sub(a[i].wrapping_mul(sk[i])); }
    p
}
fn tf(x: u64) -> f64 { x as i64 as f64 / 18446744073709551616.0 }

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let sk: Vec<u64> = (0..N).map(|_| rng.gen_range(0..2u64)).collect();
    let gtr = (8u32, 5usize); let grl = (10u32, 4usize);
    let t = std::time::Instant::now();
    let k = ProductKeys::new(&sk, gtr, grl, &mut rng);
    println!("keys: {:.1} s", t.elapsed().as_secs_f64());
    let mut w = k.work();
    for _ in 0..3 { println!("exactness of a key switching of the trace: largest difference {}", check_exact_ks(&k, &mut w, &sk, &mut rng)); }
    // trace: LWE_{s'}(μ) -> RLWE with phase μ + e in the constant coefficient
    for &phat in &[4u64, 16, 256] {
        let mut worst = 0f64; let mut sq = 0f64; let cnt = 20;
        for _ in 0..cnt {
            let m1 = rng.gen_range(0..phat); let m2 = rng.gen_range(0..phat);
            let d = 64 - phat.trailing_zeros();
            let c1 = enc(&sk, m1 << d, 2f64.powi(-40), &mut rng); let c2 = enc(&sk, m2 << d, 2f64.powi(-40), &mut rng);
            let t = std::time::Instant::now();
            let o = k.product(&mut w, &c1, &c2, phat);
            let el = t.elapsed().as_secs_f64() * 1e3;
            let want = ((m1 * m2) % phat) << d;
            let e = tf(dec(&sk, &o).wrapping_sub(want));
            worst = worst.max(e.abs()); sq += e * e;
            if phat == 256 && worst == e.abs() { print!(""); }
            let _ = el;
        }
        println!("p̂ = {:3}: {} products, error σ 2^{:.2}, max 2^{:.2} (window 2^{:.0})", phat, cnt,
                 0.5 * (sq / cnt as f64).log2(), worst.log2(), -(2.0 * phat as f64).log2());
    }
    let c1 = enc(&sk, 1 << 60, 2f64.powi(-40), &mut rng); let c2 = enc(&sk, 3 << 60, 2f64.powi(-40), &mut rng);
    let reps = 10; let t = std::time::Instant::now();
    for _ in 0..reps { let _ = k.product(&mut w, &c1, &c2, 16); }
    println!("product: {:.3} ms (gadgets trace {:?}, relinearization {:?})", t.elapsed().as_secs_f64() * 1e3 / reps as f64, gtr, grl);
}
