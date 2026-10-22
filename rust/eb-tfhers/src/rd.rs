//! Arithmetic in R_d = Z[X]/(X^d + 1), d = 16, for the polynomial stair key switch: negacyclic products of small
//! digits by 64-bit keys through the evaluations at the H = d/2 roots ζ_k = exp(iπ(2k+1)/d) (the other half are their
//! conjugates). As in julia/src/stair_ks.jl, the floating-point error is negligible against the noise of the key switch.
pub const D: usize = 16;
pub const H: usize = D / 2;

#[derive(Clone, Copy, Default, Debug)]
pub struct C { pub re: f64, pub im: f64 }

pub struct Tw { fw: [[C; D]; H], inv: [[C; H]; D] }

impl Default for Tw { fn default() -> Self { Self::new() } }

impl Tw {
    pub fn new() -> Self {
        let mut fw = [[C::default(); D]; H];
        let mut inv = [[C::default(); H]; D];
        for k in 0..H {
            for t in 0..D {
                let ang = std::f64::consts::PI * ((2 * k + 1) * t) as f64 / D as f64;
                fw[k][t] = C { re: ang.cos(), im: ang.sin() };
                inv[t][k] = C { re: ang.cos(), im: -ang.sin() };
            }
        }
        Tw { fw, inv }
    }
    /// Evaluations of a polynomial with real coefficients x[t].
    #[inline]
    pub fn fwd(&self, x: &[f64; D], out: &mut [C; H]) {
        for k in 0..H {
            let (mut re, mut im) = (0.0, 0.0);
            let w = &self.fw[k];
            for t in 0..D { re += x[t] * w[t].re; im += x[t] * w[t].im; }
            out[k] = C { re, im };
        }
    }
    pub fn fwd_u64(&self, a: &[u64], out: &mut [C; H]) {
        let mut x = [0f64; D];
        for t in 0..D { x[t] = a[t] as i64 as f64; }
        self.fwd(&x, out);
    }
    pub fn fwd_i64(&self, a: &[i64], out: &mut [C; H]) {
        let mut x = [0f64; D];
        for t in 0..D { x[t] = a[t] as f64; }
        self.fwd(&x, out);
    }
    /// Coefficients (mod 2^64) of the polynomial with evaluations y: a_t = (2/d) Re Σ_k y_k ζ_k^{-t}.
    pub fn inv_u64(&self, y: &[C; H], out: &mut [u64; D]) {
        for t in 0..D {
            let w = &self.inv[t];
            let mut re = 0.0;
            for k in 0..H { re += y[k].re * w[k].re - y[k].im * w[k].im; }
            out[t] = wrap64((re * (2.0 / D as f64)).round());
        }
    }
}

/// v mod 2^64 for an integer-valued f64 (|v| < 2^126): v - 2^64 round(2^-64 v) is exact (|·| ≤ 2^63, a multiple of the
/// ulp of v) and congruent to v, so the result is that of `v as i128 as u64`, without the slow conversion to i128.
#[inline]
pub fn wrap64(v: f64) -> u64 {
    let q = (v * 2f64.powi(-64)).round();
    let r = q.mul_add(-18446744073709551616.0, v);
    let r = if r >= 9223372036854775808.0 { r - 18446744073709551616.0 } else { r };   // r = 2^63: -2^63
    (r as i64) as u64
}

/// Exact negacyclic product-accumulate out += a·s in R_d mod 2^64, s with small integer coefficients (key generation).
pub fn rd_mul_acc(out: &mut [u64], a: &[u64], s: &[i64]) {
    for j in 0..D {
        if s[j] == 0 { continue; }
        let sj = s[j] as u64;
        for i in 0..D {
            let v = a[i].wrapping_mul(sj);
            let t = i + j;
            if t < D { out[t] = out[t].wrapping_add(v); } else { out[t - D] = out[t - D].wrapping_sub(v); }
        }
    }
}
