//! TFHE-rs 1.8.1 (default parameters PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128) on FheUint2, 4, ..., 16
//! (k = 1..8 blocks of 2 bits of message and 2 bits of carry), high-level API, one thread by default (THREADS):
//!   refresh:   `&a & MAX`  (scalar_bitand_parallelized: one PBS, key switching included, per block);
//!   addition:  `&a + &b`   (add_parallelized: sum and carry propagation, clean output);
//!   product:   `&a * &b`   (mul_parallelized: partial products, sum and carry propagation, clean output).
//! The results are checked against the clear computation modulo 2^(2k). Times in ms, median and minimum over REPS
//! runs (default 10) after one warm-up run.
//! Run:  cargo run --release --bin curves      (output: lines "tfhers,k,operation,median_ms,min_ms")
use std::time::Instant;
use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ClientKey, ConfigBuilder};
use tfhe::{FheUint10, FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn bench<O: FnMut()>(reps: usize, mut op: O) -> (f64, f64) {
    op();
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            op();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[reps / 2], ts[0])
}

macro_rules! run {
    ($T:ty, $C:ty, $k:expr, $ck:expr, $reps:expr, $csv:expr) => {{
        let k: u32 = $k;
        let max: $C = ((1u32 << (2 * k)) - 1) as $C;
        let (x, y): ($C, $C) = ((0xB5E3u32 as $C) & max, (0x6A97u32 as $C) & max);
        let a = <$T>::encrypt(x, $ck);
        let b = <$T>::encrypt(y, $ck);
        let r: $C = (&a & max).decrypt($ck);
        let s: $C = (&a + &b).decrypt($ck);
        let p: $C = (&a * &b).decrypt($ck);
        assert_eq!((r, s, p), (x, x.wrapping_add(y) & max, x.wrapping_mul(y) & max));
        for (op, t) in [
            ("refresh", bench($reps, || { let _ = &a & max; })),
            ("addition", bench($reps, || { let _ = &a + &b; })),
            ("product", bench($reps, || { let _ = &a * &b; })),
        ] {
            println!("  FheUint{:<2} (k = {}) {:<9} {:>10.3} ms  (min {:>10.3})", 2 * k, k, op, t.0, t.1);
            $csv.push(format!("tfhers,{},{},{:.4},{:.4}", k, op, t.0, t.1));
        }
    }};
}

fn main() {
    let threads = env_usize("THREADS", 1);
    let reps = env_usize("REPS", 10);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    println!("TFHE-rs 1.8.1, {} thread(s), {} repetitions, default parameters", threads, reps);
    let (ck, sk): (ClientKey, _) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    let mut csv: Vec<String> = Vec::new();
    run!(FheUint2, u8, 1, &ck, reps, csv);
    run!(FheUint4, u8, 2, &ck, reps, csv);
    run!(FheUint6, u8, 3, &ck, reps, csv);
    run!(FheUint8, u8, 4, &ck, reps, csv);
    run!(FheUint10, u16, 5, &ck, reps, csv);
    run!(FheUint12, u16, 6, &ck, reps, csv);
    run!(FheUint14, u16, 7, &ck, reps, csv);
    run!(FheUint16, u16, 8, &ck, reps, csv);
    println!("CSV (scheme,k,operation,median_ms,min_ms)");
    for l in csv {
        println!("{}", l);
    }
}
