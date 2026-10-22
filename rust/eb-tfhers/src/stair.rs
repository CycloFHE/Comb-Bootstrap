//! Polynomial stair key switch (Section 4.3 and Appendix B of the paper), port of julia/src/stair_ks.jl.
//!
//! The LWE ciphertext (a, b) of dimension T is read as a GLWE ciphertext of rank T/d over R_d (d = 16):
//!     s_j(X) = Σ_{i<d} s'_{jd+i} X^i,   ã_j(X) = a_{jd} - Σ_{i≥1} a_{jd+d-i} X^i,   a·s' = Σ_j [ã_j s_j]_0.
//! A step from y to x = k d + r switches the blocks beyond k (and, if r > 0, the tail of block k whose first r
//! coefficients are kept) with GLWE keys under (s_0, ..., s_{k-1}[, s_k^(r)]), of noise σ_128(x):
//!     c ← c - Σ_j Σ_l dec_l(ã_j) S_{j,l},   S_{j,l} = GLWE(m_j g_l).
//! The output is an LWE ciphertext of dimension n under the first n coordinates of s'.
use crate::params::log2_sigma128;
use crate::rd::{rd_mul_acc, Tw, C, D, H};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub struct Step {
    pub k: usize, pub r: usize, pub ko: usize, pub b: u32, pub l: usize,
    pub blocks: Vec<usize>,        // switched blocks (0-based)
    keys: Vec<f64>,                // [block][level][column 0..=ko][re (H values), im (H values)]
}

pub struct StairKey { pub t: usize, pub n: usize, pub steps: Vec<Step>, tw: Tw }

impl StairKey {
    /// Keys from dimension t (multiple of d) down to n = chain.last().0; chain = [(x, log2 B, ℓ)]; sprime: binary key.
    pub fn new<R: Rng>(sprime: &[u64], t: usize, chain: &[(usize, u32, usize)], rng: &mut R) -> Self {
        Self::new_with(sprime, t, chain, rng, &log2_sigma128)
    }

    /// The same with the keys of dimension x at the noise 2^l2s(x) (standard deviation on the torus).
    pub fn new_with<R: Rng>(sprime: &[u64], t: usize, chain: &[(usize, u32, usize)], rng: &mut R, l2s: &dyn Fn(usize) -> f64) -> Self {
        assert!(t % D == 0);
        let tw = Tw::new();
        let mut sp: Vec<i64> = sprime.iter().map(|&x| x as i64).collect();
        sp.resize(t, 0);
        let blk = |j: usize| -> Vec<i64> { sp[j * D..(j + 1) * D].to_vec() };
        let mut steps = Vec::new();
        let mut y = t;
        for &(x, b, l) in chain {
            let yb = y / D; let (k, r) = (x / D, x % D); let ko = k + (r > 0) as usize;
            let sigma = 2f64.powf(l2s(x));
            let mut keyb: Vec<Vec<i64>> = (0..k).map(|i| blk(i)).collect();
            let mut blocks = Vec::new(); let mut msgs: Vec<Vec<i64>> = Vec::new();
            if r > 0 {
                let mut sr = blk(k); for v in sr.iter_mut().skip(r) { *v = 0; }
                keyb.push(sr.clone());
                blocks.push(k); msgs.push(blk(k).iter().zip(&sr).map(|(a, c)| a - c).collect());
                for j in k + 1..yb { blocks.push(j); msgs.push(blk(j)); }
            } else {
                for j in k..yb { blocks.push(j); msgs.push(blk(j)); }
            }
            let mut keys = Vec::with_capacity(msgs.len() * l * (ko + 1) * H);
            for msg in &msgs {
                for lev in 1..=l {
                    let g: u64 = 1u64 << (64 - lev * b as usize);
                    let mut bpoly = [0u64; D];
                    for tt in 0..D {
                        let z: f64 = StandardNormal.sample(rng);
                        bpoly[tt] = (msg[tt] as u64).wrapping_mul(g).wrapping_add((z * sigma * 2f64.powi(64)).round() as i64 as u64);
                    }
                    let mut cols: Vec<[C; H]> = Vec::with_capacity(ko + 1);
                    for kb in keyb.iter().take(ko) {
                        let a: Vec<u64> = (0..D).map(|_| rng.gen::<u64>()).collect();
                        rd_mul_acc(&mut bpoly, &a, kb);
                        let mut f = [C::default(); H]; tw.fwd_u64(&a, &mut f); cols.push(f);
                    }
                    let mut f = [C::default(); H]; tw.fwd_u64(&bpoly, &mut f); cols.push(f);
                    for c in cols { keys.extend(c.iter().map(|z| z.re)); keys.extend(c.iter().map(|z| z.im)); }
                }
            }
            steps.push(Step { k, r, ko, b, l, blocks, keys });
            y = x;
        }
        StairKey { t, n: chain.last().unwrap().0, steps, tw }
    }

    /// Key switch of (a, b) under s' (dimension ≤ t) to dimension n under the first n coordinates of s'.
    pub fn switch(&self, a: &[u64], b: u64, rbit: &mut u64) -> (Vec<u64>, u64) {
        let nb = self.t / D; let na = a.len();
        let mut comps = vec![[0u64; D]; nb];
        for j in 0..nb {
            let base = j * D; let c = &mut comps[j];
            c[0] = if base < na { a[base] } else { 0 };
            for i in 1..D { let u = base + D - i; c[i] = if u < na { a[u].wrapping_neg() } else { 0 }; }
        }
        let mut bpoly = [0u64; D]; bpoly[0] = b;
        let lmax = self.steps.iter().map(|s| s.l).max().unwrap();
        let mut dig = vec![[0i64; D]; lmax];
        let mut dft = [C::default(); H];
        let mut tmp = [0u64; D];
        for st in &self.steps {
            let ncol = st.ko + 1;
            // accumulators and keys split re / im by column (the same operations as complex numbers: vectorized)
            let mut acc = vec![0f64; ncol * D];
            for (jj, &j) in st.blocks.iter().enumerate() {
                decomp_sym(&mut dig, &comps[j], st.b, st.l, rbit);
                for lev in 0..st.l {
                    self.tw.fwd_i64(&dig[lev], &mut dft);
                    let (dr, di): ([f64; H], [f64; H]) = (std::array::from_fn(|h| dft[h].re), std::array::from_fn(|h| dft[h].im));
                    let off = ((jj * st.l + lev) * ncol) * D;
                    let kk = &st.keys[off..off + ncol * D];
                    for (a, m) in acc.chunks_exact_mut(D).zip(kk.chunks_exact(D)) {
                        let (ar, ai) = a.split_at_mut(H); let (mr, mi) = m.split_at(H);
                        for h in 0..H {
                            ar[h] += dr[h] * mr[h] - di[h] * mi[h];
                            ai[h] += dr[h] * mi[h] + di[h] * mr[h];
                        }
                    }
                }
            }
            for i in 0..ncol {
                let y: [C; H] = std::array::from_fn(|h| C { re: acc[i * D + h], im: acc[i * D + H + h] });
                self.tw.inv_u64(&y, &mut tmp);
                let dst = if i < st.ko { &mut comps[i] } else { &mut bpoly };
                for t in 0..D { dst[t] = dst[t].wrapping_sub(tmp[t]); }
            }
        }
        let n = self.n; let mut ao = vec![0u64; n];
        for u in 0..n {
            let (i, t) = (u / D, u % D); let c = &comps[i];
            ao[u] = if t == 0 { c[0] } else { c[D - t].wrapping_neg() };
        }
        (ao, bpoly[0])
    }
}

#[inline]
fn xbit(state: &mut u64) -> bool {
    let mut x = *state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *state = x; (x >> 63) == 1
}

/// Balanced digits with zero mean (a digit equal to B/2 becomes -B/2 with a carry with probability 1/2), as decomp_sym!
fn decomp_sym(out: &mut [[i64; D]], t: &[u64; D], b: u32, l: usize, rbit: &mut u64) {
    let shift = 64 - b as usize * l;
    let mask = (1u64 << b) - 1; let half = 1i64 << (b - 1); let full = 1i64 << b;
    for i in 0..D {
        let x = t[i];
        let mut y = (x >> shift) + ((x >> (shift - 1)) & 1);
        let mut carry = 0i64;
        for j in (0..l).rev() {
            let mut d = (y & mask) as i64 + carry;
            y >>= b;
            if d > half || (d == half && xbit(rbit)) { d -= full; carry = 1; } else { carry = 0; }
            out[j][i] = d;
        }
    }
}
