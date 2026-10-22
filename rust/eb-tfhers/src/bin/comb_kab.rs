//! A/B timing of the orders of the key switchings of CombBoot (m = 1, comb::KS_MODE) and of the padding of their arrays
//! (comb::KS_PAD), one thread, same keys and inputs, interleaved, minimum over REPS calls (default 8). Variants: 0u, mode 0
//! (automorphism by automorphism, block by block) with separate arrays, as in the first version; 0, the same with padded
//! arrays (bit-identical outputs); 1, source block by source block, padded (default for ℓ ≥ 3; sums in another order, so the masks
//! may differ while the phases agree up to the noise); 2, 3, 4 (default for ℓ ≤ 2: 2; comb::KS_AUTO): as 1, the digits and the FFT of b of W = 4, 2 or 8 values in
//! registers for the B - 1 automorphisms (bit-identical to 1). Time of each phase (µs per coordinate, minimum over the calls):
//! digits and forward FFTs, table, key switchings and public products, backward FFTs of C_L, CMux.
//! (results/rust/comb_kab_mac2.txt was run with four exploratory orders: 0, 1 (whole vector), 2 (by destination block)
//! and 3 (by source block, now mode 1).)
//! Run:  RAYON_NUM_THREADS=1 cargo run --release --bin comb_kab        (BITS=2,4,8,12,16; MODES=0,1,2,3,4; 0u: mode 0 unpadded)
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS, KS_AUTO, KS_MODE, KS_PAD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() {
    let bits: Vec<u32> = std::env::var("BITS").unwrap_or("2,4,8,12,16".into()).split(',').map(|x| x.parse().unwrap()).collect();
    // variants: KS_MODE, followed by "u" for unpadded arrays (KS_PAD = false); the first one is the reference
    let vars: Vec<(usize, bool)> = std::env::var("MODES").unwrap_or("0,1,2,3,4".into()).split(',')
        .map(|x| (x.trim_end_matches('u').parse().unwrap(), !x.ends_with('u'))).collect();
    let name = |v: &(usize, bool)| format!("{}{}", v.0, if v.1 { "" } else { "u" });
    let set = |v: &(usize, bool)| { KS_MODE.store(v.0, Relaxed); KS_PAD.store(v.1, Relaxed); };
    let modes: Vec<String> = vars.iter().map(name).collect();
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    println!("CombBoot (m = 1), one thread: orders of the key switchings {modes:?} (0: by automorphism; 1: by source block; 2, 3, 4: by source block, digits in registers, W = 4, 2, 8; u: arrays not padded), min over {reps}");
    for cfg in COMB_CONFIGS.iter().filter(|c| bits.contains(&c.bits)) {
        let k = CombKeys::new_m(cfg, 1, 41 + cfg.bits as u64);
        let mut rng = StdRng::seed_from_u64(cfg.bits as u64);
        let (p, bc) = (cfg.p as u64, k.bcols);
        let aints: Vec<Vec<u64>> = (0..bc).map(|_| (0..cfg.n).map(|_| rng.gen_range(0..p)).collect()).collect();
        let bints: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..p)).collect();
        let outs: Vec<_> = vars.iter().map(|v| { set(v); k.combboot(&aints, &bints) }).collect();
        // phases under s' (dimension N): mode 3 sums in another order, so the masks may differ (a digit of the gadget
        // flipped by a rounding) while the phases agree up to the noise
        let sk: Vec<u64> = k.big_sk.as_ref().to_vec();
        let phase = |c: &tfhe::core_crypto::prelude::LweCiphertextOwned<u64>| -> u64 {
            let (a, b) = c.as_ref().split_at(sk.len());
            a.iter().zip(&sk).fold(b[0], |p, (&x, &s)| p.wrapping_sub(x.wrapping_mul(s)))
        };
        let same: Vec<&String> = outs.iter().zip(&modes).filter(|(o, _)| o.iter().zip(&outs[0]).all(|(x, y)| x.as_ref() == y.as_ref())).map(|(_, m)| m).collect();
        let diffs: Vec<String> = outs.iter().zip(&modes).map(|(o, m)| {
            let d = o.iter().zip(&outs[0]).map(|(x, y)| (phase(x).wrapping_sub(phase(y)) as i64).unsigned_abs()).max().unwrap();
            format!("{m}: {}", if d == 0 { "0".to_string() } else { format!("2^{:.1}", (d as f64).log2() - 64.0) })
        }).collect();
        let mut t = vec![f64::INFINITY; modes.len()];
        let mut ph = vec![[f64::INFINITY; 5]; modes.len()];
        for _ in 0..reps {
            for (i, v) in vars.iter().enumerate() {
                set(v);
                let t0 = Instant::now(); let _ = k.combboot(&aints, &bints); t[i] = t[i].min(t0.elapsed().as_secs_f64() * 1e3);
                let mut pr = [0f64; 5]; let _ = k.combboot_prof(&aints, &bints, Some(&mut pr));
                for q in 0..5 { ph[i][q] = ph[i][q].min(pr[q] * 1e3 / cfg.n as f64); }
            }
        }
        KS_MODE.store(KS_AUTO, Relaxed); KS_PAD.store(true, Relaxed);
        println!("  {:2} bits, B = {:2}, n = {:4}, KS gadget (2^{},{}):  outputs bit-identical for the modes {same:?}; max |difference of the phases| to mode {}: {}", cfg.bits, bc, cfg.n, cfg.gks.0, cfg.gks.1, modes[0], diffs.join(", "));
        for (i, m) in modes.iter().enumerate() {
            println!("    mode {m:2}: {:8.2} ms per call, {:.3} ms per ciphertext, ratio to mode {} {:.3};  µs per coordinate: digits+FFT {:.1}, table {:.1}, KS+PP {:.1}, bwd {:.1}, CMux {:.1}",
                     t[i], t[i] / bc as f64, modes[0], t[0] / t[i], ph[i][0], ph[i][1], ph[i][2], ph[i][3], ph[i][4]);
        }
    }
}
