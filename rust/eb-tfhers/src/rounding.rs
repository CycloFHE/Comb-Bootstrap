//! Rounding of an LWE ciphertext at the resolution R = p̂·2N (a power of two), coordinate by coordinate, with
//! recentering and selection; port of julia/src/rounding.jl (binary key):
//!   ā_u = round(R a_u), X_u = ā_u - R a_u (public), b̄ = round(R b); over a uniform binary key, T = Σ s_u X_u has the
//!   public mean c = Σ X_u / 2 and variance V = Σ X_u^2 / 4. Recentering adds c̄ = round(c) to b̄; selection
//!   re-randomizes the ciphertext with a random pair of public encryptions of zero while V > V*.
use rand::Rng;

#[inline]
pub fn rounding_error(x: u64, r: u64) -> f64 { -((x.wrapping_mul(r)) as i64 as f64) * 2f64.powi(-64) }

#[inline]
pub fn round_to(x: u64, log_r: u32) -> u64 { x.wrapping_add(1u64 << (63 - log_r)) >> (64 - log_r) }

pub fn stats(a: &[u64], r: u64) -> (f64, f64) {
    let (mut c, mut v) = (0.0, 0.0);
    for &x in a { let e = rounding_error(x, r); c += e; v += e * e; }
    (c / 2.0, v / 4.0)
}

fn stats3(a: &[u64], z1: &[u64], z2: &[u64], r: u64) -> (f64, f64) {
    let (mut c, mut v) = (0.0, 0.0);
    for u in 0..a.len() { let e = rounding_error(a[u].wrapping_add(z1[u]).wrapping_add(z2[u]), r); c += e; v += e * e; }
    (c / 2.0, v / 4.0)
}

/// Rounded ciphertext: ā (in [0, R)), b̄ + c̄ (in [0, R)), c̄, number of trials of the selection.
pub struct Rounded { pub abar: Vec<u64>, pub bbar: u64, pub cbar: i64, pub trials: usize, pub v: f64 }

/// pool: masks and bodies of the public encryptions of zero under s (dimension n).
pub fn round_select<RG: Rng>(a: &[u64], b: u64, log_r: u32, pool: &[(Vec<u64>, u64)], vstar: f64, rng: &mut RG) -> Rounded {
    let r = 1u64 << log_r;
    let (mut c, mut v) = stats(a, r);
    let mut trials = 0usize;
    let mut sel: Option<(usize, usize)> = None;
    if v > vstar && pool.len() >= 2 {
        let p = pool.len(); let mut best = (v, usize::MAX, usize::MAX, c);
        while trials < 100_000 {
            trials += 1;
            let i = rng.gen_range(0..p); let mut j = rng.gen_range(0..p - 1); if j >= i { j += 1; }
            let (c2, v2) = stats3(a, &pool[i].0, &pool[j].0, r);
            if v2 < best.0 { best = (v2, i, j, c2); }
            if v2 <= vstar { break; }
        }
        if best.1 != usize::MAX { v = best.0; c = best.3; sel = Some((best.1, best.2)); }
    }
    let (abar, bsel) = match sel {
        Some((i, j)) => (
            (0..a.len()).map(|u| round_to(a[u].wrapping_add(pool[i].0[u]).wrapping_add(pool[j].0[u]), log_r)).collect(),
            b.wrapping_add(pool[i].1).wrapping_add(pool[j].1),
        ),
        None => (a.iter().map(|&x| round_to(x, log_r)).collect(), b),
    };
    let cbar = c.round() as i64;
    let bbar = (round_to(bsel, log_r) as i64 + cbar).rem_euclid(r as i64) as u64;
    Rounded { abar, bbar, cbar, trials, v }
}

/// Rounding at any integer resolution r (RNS: r = p p̂ with p̂ not a power of two): round(r x / 2^64) mod r.
#[inline]
pub fn round_to_r(x: u64, r: u64) -> u64 { ((((x as u128) * (r as u128) + (1u128 << 63)) >> 64) as u64) % r }

/// round_select at any integer resolution r (the statistics of rounding_error hold for any r); for r = 2^log_r, the same
/// values as round_select.
pub fn round_select_r<RG: Rng>(a: &[u64], b: u64, r: u64, pool: &[(Vec<u64>, u64)], vstar: f64, rng: &mut RG) -> Rounded {
    let (mut c, mut v) = stats(a, r);
    let mut trials = 0usize;
    let mut sel: Option<(usize, usize)> = None;
    if v > vstar && pool.len() >= 2 {
        let p = pool.len(); let mut best = (v, usize::MAX, usize::MAX, c);
        while trials < 100_000 {
            trials += 1;
            let i = rng.gen_range(0..p); let mut j = rng.gen_range(0..p - 1); if j >= i { j += 1; }
            let (c2, v2) = stats3(a, &pool[i].0, &pool[j].0, r);
            if v2 < best.0 { best = (v2, i, j, c2); }
            if v2 <= vstar { break; }
        }
        if best.1 != usize::MAX { v = best.0; c = best.3; sel = Some((best.1, best.2)); }
    }
    let (abar, bsel) = match sel {
        Some((i, j)) => (
            (0..a.len()).map(|u| round_to_r(a[u].wrapping_add(pool[i].0[u]).wrapping_add(pool[j].0[u]), r)).collect(),
            b.wrapping_add(pool[i].1).wrapping_add(pool[j].1),
        ),
        None => (a.iter().map(|&x| round_to_r(x, r)).collect(), b),
    };
    let cbar = c.round() as i64;
    let bbar = (round_to_r(bsel, r) as i64 + cbar).rem_euclid(r as i64) as u64;
    Rounded { abar, bbar, cbar, trials, v }
}
