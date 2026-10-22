//! Kernel alone (kern::ks_block), random keys and digits, as in CombBoot (m = 1, B = 16, L = 64): the key switchings of the
//! 15 automorphisms and the public products for one coordinate, block by block and over the whole vector, ℓ = 2 and 3 (µs),
//! with separate arrays (as allocated) and with all the arrays in one buffer at a stride of 1024 + PAD values (PAD = 0:
//! 8 KB apart, the starts on the same sets of the L1 cache; 48: shifted by 3 lines of 128 B each).
//! Run:  cargo run --release --example ksrs
use eb_tfhers::kern::ks_block;
use rand::{Rng, SeedableRng};
use std::time::Instant;
const H: usize = 1024; const BC: usize = 16; const LR: usize = H / BC;
fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for gl in [2usize, 3] {
        for layout in ["separate", "pad 0", "pad 48"] {
            let narr = 2 * gl + BC * gl * 4 + 8;
            let pad = match layout { "pad 0" => 0, "pad 48" => 48, _ => 0 };
            let mut arena: Vec<f64> = (0..narr * (H + pad)).map(|_| rng.gen::<f64>() - 0.5).collect();
            let mut sep: Vec<Vec<f64>> = (0..narr).map(|_| (0..H).map(|_| rng.gen::<f64>() - 0.5).collect()).collect();
            let mut arrs: Vec<&mut [f64]> = if layout == "separate" { sep.iter_mut().map(|v| &mut v[..]).collect() }
                                            else { arena.chunks_mut(H + pad).map(|c| &mut c[..H]).collect() };
            let (xr, rest) = arrs.split_at_mut(1); let (xi, rest) = rest.split_at_mut(1);
            let (yr, rest) = rest.split_at_mut(1); let (yi, rest) = rest.split_at_mut(1);
            let (xr, xi, yr, yi) = (&mut *xr[0], &mut *xi[0], &mut *yr[0], &mut *yi[0]);
            let ro: Vec<&[f64]> = rest.iter().map(|a| &**a).collect();
            let (fr, fi, qr, qi) = (ro[0], ro[1], ro[2], ro[3]);
            let dr = &ro[4..4 + gl]; let di = &ro[4 + gl..4 + 2 * gl];
            let key = |j: usize, l: usize, c: usize| ro[4 + 2 * gl + (j * gl + l) * 4 + c];
            let mut best = [f64::INFINITY; 2];
            for _ in 0..300 { for mode in 0..2 {
                let t = Instant::now();
                for j in 1..BC {
                    let mut sj = (j * 3 % BC) * LR; if sj == 0 { sj = LR; }
                    let segs: Vec<(usize, usize, usize)> = if mode == 0 { (0..BC).map(|m| (m * LR, LR, (m * LR + sj) % H)).collect() } else { vec![(0, H - sj, sj), (H - sj, sj, 0)] };
                    for (o, n, so) in segs {
                        let mut dd: [[&[f64]; 2]; 6] = [[&[], &[]]; 6]; let mut kk: [[&[f64]; 4]; 6] = [[&[], &[], &[], &[]]; 6];
                        for l in 0..gl { kk[l] = [&key(j, l, 0)[so..so + n], &key(j, l, 1)[so..so + n], &key(j, l, 2)[so..so + n], &key(j, l, 3)[so..so + n]]; dd[l] = [&dr[l][so..so + n], &di[l][so..so + n]]; }
                        ks_block(&mut xr[o..o + n], &mut xi[o..o + n], &mut yr[o..o + n], &mut yi[o..o + n], &qr[o..o + n], &qi[o..o + n], &fr[so..so + n], &fi[so..so + n], &dd[..gl], &kk[..gl]);
                    }
                }
                best[mode] = best[mode].min(t.elapsed().as_secs_f64() * 1e6);
            } }
            println!("gl = {gl}, {layout:8}: µs per coordinate: blocks {:.2}, whole {:.2}   ({})", best[0], best[1], xr[5]);
        }
    }
}
