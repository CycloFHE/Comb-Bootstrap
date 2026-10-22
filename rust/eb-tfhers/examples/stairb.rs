//! Time of the stair key switch alone (chain of CombBoot at 2 bits), µs per ciphertext.
use eb_tfhers::comb::COMB_CONFIGS;
use eb_tfhers::stair::StairKey;
use rand::{Rng, SeedableRng};
use std::time::Instant;
fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    for cfg in COMB_CONFIGS.iter().filter(|c| c.bits == 2 || c.bits == 8) {
        let s: Vec<u64> = (0..2048).map(|_| rng.gen_range(0..2)).collect();
        let key = StairKey::new(&s, 2048, cfg.chain, &mut rng);
        let a: Vec<u64> = (0..2048).map(|_| rng.gen()).collect(); let b: u64 = rng.gen();
        let mut rbit = 0x1234567u64; let mut best = f64::INFINITY; let mut chk = 0u64;
        for _ in 0..200 { let t = Instant::now(); let (ao, bo) = key.switch(&a, b, &mut rbit); best = best.min(t.elapsed().as_secs_f64() * 1e6); chk ^= ao[3] ^ bo; }
        println!("{} bits, n = {}: stair key switch {:.1} µs  ({chk:x})", cfg.bits, cfg.n, best);
    }
}
