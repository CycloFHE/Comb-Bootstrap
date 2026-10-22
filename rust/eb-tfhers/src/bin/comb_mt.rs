//! Time of one comb bootstrap (ms per call and per error), single-thread version (combboot, CMux of TFHE-rs) and
//! multi-threaded version on t threads (combboot_mt), same keys and inputs.
//! Run: cargo run --release --bin comb_mt    (BITS=8, THREADS=1,2,4,8, REPS=5)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
fn main() {
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("8".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let ths: Vec<usize> = std::env::var("THREADS").unwrap_or("1,2,4,8".into()).split(',').map(|x| x.parse().unwrap()).collect();
    for cfg in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let keys = CombKeys::new_m(cfg, 1, 1);   // combboot_mt: m = 1
        let mut rng = StdRng::seed_from_u64(3);
        let p = cfg.p as u64; let bc = keys.bcols;
        let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
        let bints: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..p)).collect();
        // one-thread reference with the order of combboot_mt: by automorphism, unpadded arrays (KS_MODE 0, KS_PAD false)
        eb_tfhers::comb::KS_MODE.store(0, std::sync::atomic::Ordering::Relaxed);
        eb_tfhers::comb::KS_PAD.store(false, std::sync::atomic::Ordering::Relaxed);
        let _ = keys.combboot(&aints, &bints);
        let t1 = (0..reps).map(|_| { let t = Instant::now(); let _ = keys.combboot(&aints, &bints); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        println!("{:2} bits, n = {}, B = {}: single-thread {:.1} ms per call ({:.2} ms per error)", cfg.bits, cfg.n, bc, t1, t1 / bc as f64);
        for &t in &ths {
            let _ = keys.combboot_mt(&aints, &bints, t);
            let tt = (0..reps).map(|_| { let s = Instant::now(); let _ = keys.combboot_mt(&aints, &bints, t); s.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
            println!("    {t} threads: {:.1} ms per call ({:.2} ms per error), speed-up {:.2}", tt, tt / bc as f64, t1 / tt);
            if eb_tfhers::mt::prof_on() {
                let v: Vec<f64> = eb_tfhers::mt::PROF.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / (reps + 1) as f64).collect();
                println!("      worker 0, ms per call: digits {:.1}, FFT+table {:.1}, products+KS {:.1}, backward {:.1}, CMux: digits {:.1}, FFT {:.1}, products {:.1}, backward {:.1}", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            }
        }
    }
}
