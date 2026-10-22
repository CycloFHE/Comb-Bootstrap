//! Microbenchmark of TFHE-rs (tfhe 1.8.1, default parameters PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128):
//! addition and multiplication of two encrypted 8-bit integers (4 blocks of 2 bits of message and 2 bits of carry),
//! with and without carry propagation, one thread by default (THREADS=k to change it), REPS repetitions (default 20).
//!
//!   high-level API:  FheUint8, `&a + &b` (= add_parallelized) and `&a * &b` (= mul_parallelized), clean outputs;
//!   integer API, same parameters, 4 blocks:
//!     addition:  unchecked_add (no PBS, carries left in the blocks), then full_propagate (sequential) or
//!                full_propagate_parallelized (carry propagation alone), and add_parallelized (both);
//!     product:   unchecked_mul (sequential schoolbook, carries left), then full_propagate, and
//!                mul_parallelized (partial products, sum and carry propagation);
//!   shortint: one PBS (key switching + blind rotation) on one block, for reference.
//! Run:  cargo run --release      (times in ms: median and minimum over REPS runs)
use std::time::Instant;
use tfhe::integer::{gen_keys_radix, RadixCiphertext};
use tfhe::prelude::*;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as PARAMS;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Times `op` on a fresh input built by `prep` (not timed): (median, min) in ms.
fn bench<T, P: FnMut() -> T, O: FnMut(&mut T)>(reps: usize, mut prep: P, mut op: O) -> (f64, f64) {
    for _ in 0..3 {
        let mut x = prep();
        op(&mut x);
    }
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let mut x = prep();
            let t = Instant::now();
            op(&mut x);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[reps / 2], ts[0])
}

fn line(label: &str, (med, min): (f64, f64)) {
    println!("  {:<70} {:>9.3} ms  (min {:>8.3})", label, med, min);
}

fn main() {
    let threads = env_usize("THREADS", 1);
    let reps = env_usize("REPS", 20);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    println!("TFHE-rs 1.8.1, {} thread(s), {} repetitions", threads, reps);
    println!("parameters: {:?}", PARAMS);

    let (x, y): (u8, u8) = (173, 59);
    let (sum, prod) = (x.wrapping_add(y) as u64, x.wrapping_mul(y) as u64);

    // ---------------------------------------------------------------- high-level API
    println!("high-level API (FheUint8)");
    let (ck, sk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    let a = FheUint8::encrypt(x, &ck);
    let b = FheUint8::encrypt(y, &ck);
    let s: u8 = (&a + &b).decrypt(&ck);
    let p: u8 = (&a * &b).decrypt(&ck);
    assert_eq!((s as u64, p as u64), (sum, prod));
    line("a + b (add_parallelized: with carry propagation)", bench(reps, || (), |_| {
        let _ = &a + &b;
    }));
    line("a * b (mul_parallelized: with carry propagation)", bench(reps, || (), |_| {
        let _ = &a * &b;
    }));

    // ---------------------------------------------------------------- integer API
    println!("integer API (radix, 4 blocks, same parameters)");
    let (cks, sks) = gen_keys_radix(PARAMS, 4);
    let ca: RadixCiphertext = cks.encrypt(x as u64);
    let cb: RadixCiphertext = cks.encrypt(y as u64);
    let dec = |c: &RadixCiphertext| -> u64 { cks.decrypt::<u64>(c) % 256 };

    // addition
    let u = sks.unchecked_add(&ca, &cb);
    let mut u1 = u.clone();
    sks.full_propagate(&mut u1);
    let mut u2 = u.clone();
    sks.full_propagate_parallelized(&mut u2);
    let ap = sks.add_parallelized(&ca, &cb);
    assert_eq!((dec(&u), dec(&u1), dec(&u2), dec(&ap)), (sum, sum, sum, sum));
    assert!(u1.block_carries_are_empty() && u2.block_carries_are_empty() && ap.block_carries_are_empty());
    line("addition without propagation (unchecked_add)", bench(reps, || (), |_| {
        let _ = sks.unchecked_add(&ca, &cb);
    }));
    line("carry propagation alone, sequential (full_propagate)", bench(reps, || u.clone(), |c| {
        sks.full_propagate(c)
    }));
    line("carry propagation alone (full_propagate_parallelized)", bench(reps, || u.clone(), |c| {
        sks.full_propagate_parallelized(c)
    }));
    line("addition + propagation, sequential (unchecked_add + full_propagate)", bench(reps, || (), |_| {
        let mut c = sks.unchecked_add(&ca, &cb);
        sks.full_propagate(&mut c);
    }));
    line("addition with propagation (add_parallelized)", bench(reps, || (), |_| {
        let _ = sks.add_parallelized(&ca, &cb);
    }));

    // product
    let m = sks.unchecked_mul(&ca, &cb);
    let mut m1 = m.clone();
    sks.full_propagate(&mut m1);
    let mp = sks.mul_parallelized(&ca, &cb);
    assert_eq!((dec(&m1), dec(&mp)), (prod, prod));
    assert!(m1.block_carries_are_empty() && mp.block_carries_are_empty());
    line("product without final propagation (unchecked_mul, sequential)", bench(reps, || (), |_| {
        let _ = sks.unchecked_mul(&ca, &cb);
    }));
    line("final carry propagation of the product (full_propagate)", bench(reps, || m.clone(), |c| {
        sks.full_propagate(c)
    }));
    line("product + propagation, sequential (unchecked_mul + full_propagate)", bench(reps, || (), |_| {
        let mut c = sks.unchecked_mul(&ca, &cb);
        sks.full_propagate(&mut c);
    }));
    line("product with propagation (mul_parallelized)", bench(reps, || (), |_| {
        let _ = sks.mul_parallelized(&ca, &cb);
    }));

    // ---------------------------------------------------------------- one PBS
    println!("shortint (one block)");
    let (ck1, sk1) = tfhe::shortint::gen_keys(PARAMS);
    let c1 = ck1.encrypt(3);
    assert_eq!(ck1.decrypt(&sk1.message_extract(&c1)), 3);
    line("one PBS (key switching + blind rotation: message_extract)", bench(reps, || (), |_| {
        let _ = sk1.message_extract(&c1);
    }));
}
