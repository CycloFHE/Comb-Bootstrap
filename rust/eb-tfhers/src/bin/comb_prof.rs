//! Time of the phases of one comb bootstrap (ms per call): digits and forward FFTs, table, public products and key
//! switchings, backward FFTs, CMux.  Run: cargo run --release --bin comb_prof   (BITS=8)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
fn main() {
    let bits: u32 = std::env::var("BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let cfg = COMB_CONFIGS.iter().find(|c| c.bits == bits).unwrap();
    let keys = CombKeys::new(cfg, 1);
    let mut rng = StdRng::seed_from_u64(3);
    let p = cfg.p as u64; let bc = keys.bcols;
    let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
    let bints: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..p)).collect();
    let _ = keys.combboot(&aints, &bints);
    let mut pr = [0f64; 5]; let reps = 3;
    for _ in 0..reps { let _ = keys.combboot_prof(&aints, &bints, Some(&mut pr)); }
    let tot: f64 = pr.iter().sum::<f64>() / reps as f64;
    println!("{bits} bits, n = {}, B = {bc}: {:.1} ms per call ({:.3} ms per error): digits+FFT {:.1}, table {:.1}, products+KS {:.1}, backward FFT {:.1}, CMux {:.1}",
             cfg.n, tot, tot / bc as f64, pr[0] / reps as f64, pr[1] / reps as f64, pr[2] / reps as f64, pr[3] / reps as f64, pr[4] / reps as f64);
}
