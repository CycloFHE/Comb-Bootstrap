//! Fused kernels of CombBoot (generated for ℓ = 1..6 levels of the automorphism gadget): for one block of L values,
//! the key switching of the source block (all levels kept in registers) and its product by Q_j, added to the
//! destination block:  t = Σ_l d_l a_l,  w = fb - Σ_l d_l b_l,  x -= q t,  y += q w  (complex, split re/im, FMA in the
//! same order as level by level: bit-identical). One slice parameter each, so that LLVM knows they do not alias.

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_1(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_2(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_3(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64], d2r: &[f64], d2i: &[f64], a2r: &[f64], a2i: &[f64], b2r: &[f64], b2i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i, d2r, d2i, a2r, a2i, b2r, b2i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n], &d2r[..n], &d2i[..n], &a2r[..n], &a2i[..n], &b2r[..n], &b2i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d2r[i], d2i[i], a2r[i], a2i[i], b2r[i], b2i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_4(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64], d2r: &[f64], d2i: &[f64], a2r: &[f64], a2i: &[f64], b2r: &[f64], b2i: &[f64], d3r: &[f64], d3i: &[f64], a3r: &[f64], a3i: &[f64], b3r: &[f64], b3i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i, d2r, d2i, a2r, a2i, b2r, b2i, d3r, d3i, a3r, a3i, b3r, b3i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n], &d2r[..n], &d2i[..n], &a2r[..n], &a2i[..n], &b2r[..n], &b2i[..n], &d3r[..n], &d3i[..n], &a3r[..n], &a3i[..n], &b3r[..n], &b3i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d2r[i], d2i[i], a2r[i], a2i[i], b2r[i], b2i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d3r[i], d3i[i], a3r[i], a3i[i], b3r[i], b3i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_5(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64], d2r: &[f64], d2i: &[f64], a2r: &[f64], a2i: &[f64], b2r: &[f64], b2i: &[f64], d3r: &[f64], d3i: &[f64], a3r: &[f64], a3i: &[f64], b3r: &[f64], b3i: &[f64], d4r: &[f64], d4i: &[f64], a4r: &[f64], a4i: &[f64], b4r: &[f64], b4i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i, d2r, d2i, a2r, a2i, b2r, b2i, d3r, d3i, a3r, a3i, b3r, b3i, d4r, d4i, a4r, a4i, b4r, b4i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n], &d2r[..n], &d2i[..n], &a2r[..n], &a2i[..n], &b2r[..n], &b2i[..n], &d3r[..n], &d3i[..n], &a3r[..n], &a3i[..n], &b3r[..n], &b3i[..n], &d4r[..n], &d4i[..n], &a4r[..n], &a4i[..n], &b4r[..n], &b4i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d2r[i], d2i[i], a2r[i], a2i[i], b2r[i], b2i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d3r[i], d3i[i], a3r[i], a3i[i], b3r[i], b3i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d4r[i], d4i[i], a4r[i], a4i[i], b4r[i], b4i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_fused_6(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64], d2r: &[f64], d2i: &[f64], a2r: &[f64], a2i: &[f64], b2r: &[f64], b2i: &[f64], d3r: &[f64], d3i: &[f64], a3r: &[f64], a3i: &[f64], b3r: &[f64], b3i: &[f64], d4r: &[f64], d4i: &[f64], a4r: &[f64], a4i: &[f64], b4r: &[f64], b4i: &[f64], d5r: &[f64], d5i: &[f64], a5r: &[f64], a5i: &[f64], b5r: &[f64], b5i: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n]);
    let (qr, qi, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i, d2r, d2i, a2r, a2i, b2r, b2i, d3r, d3i, a3r, a3i, b3r, b3i, d4r, d4i, a4r, a4i, b4r, b4i, d5r, d5i, a5r, a5i, b5r, b5i) = (&qr[..n], &qi[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n], &d2r[..n], &d2i[..n], &a2r[..n], &a2i[..n], &b2r[..n], &b2i[..n], &d3r[..n], &d3i[..n], &a3r[..n], &a3i[..n], &b3r[..n], &b3i[..n], &d4r[..n], &d4i[..n], &a4r[..n], &a4i[..n], &b4r[..n], &b4i[..n], &d5r[..n], &d5i[..n], &a5r[..n], &a5i[..n], &b5r[..n], &b5i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d2r[i], d2i[i], a2r[i], a2i[i], b2r[i], b2i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d3r[i], d3i[i], a3r[i], a3i[i], b3r[i], b3i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d4r[i], d4i[i], a4r[i], a4i[i], b4r[i], b4i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d5r[i], d5i[i], a5r[i], a5i[i], b5r[i], b5i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (qr[i], qi[i]);
        xr[i] = p.mul_add(ti, q.mul_add(-tr, xr[i])); xi[i] = p.mul_add(-tr, q.mul_add(-ti, xi[i]));
        yr[i] = p.mul_add(-wi, q.mul_add(wr, yr[i])); yi[i] = p.mul_add(wr, q.mul_add(wi, yi[i]));
    }
}

/// Dispatch on the number of levels: d[l] = (re, im) of the digits l, k[l] = (A.re, A.im, B.re, B.im) of the key l.
#[allow(clippy::too_many_arguments)]
pub fn ks_fused(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64],
                d: &[[&[f64]; 2]], k: &[[&[f64]; 4]]) {
    match d.len() {
        1 => ks_fused_1(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3]),
        2 => ks_fused_2(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3]),
        3 => ks_fused_3(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3], d[2][0], d[2][1], k[2][0], k[2][1], k[2][2], k[2][3]),
        4 => ks_fused_4(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3], d[2][0], d[2][1], k[2][0], k[2][1], k[2][2], k[2][3], d[3][0], d[3][1], k[3][0], k[3][1], k[3][2], k[3][3]),
        5 => ks_fused_5(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3], d[2][0], d[2][1], k[2][0], k[2][1], k[2][2], k[2][3], d[3][0], d[3][1], k[3][0], k[3][1], k[3][2], k[3][3], d[4][0], d[4][1], k[4][0], k[4][1], k[4][2], k[4][3]),
        6 => ks_fused_6(xr, xi, yr, yi, qr, qi, fr, fi, d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3], d[2][0], d[2][1], k[2][0], k[2][1], k[2][2], k[2][3], d[3][0], d[3][1], k[3][0], k[3][1], k[3][2], k[3][3], d[4][0], d[4][1], k[4][0], k[4][1], k[4][2], k[4][3], d[5][0], d[5][1], k[5][0], k[5][1], k[5][2], k[5][3]),
        l => panic!("ks_fused: {l} levels"),
    }
}

/// t += d·a, w -= d·b (complex, split): one level of the key switching of a block.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_level(tr: &mut [f64], ti: &mut [f64], wr: &mut [f64], wi: &mut [f64], dr: &[f64], di: &[f64], ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64]) {
    let n = tr.len();
    let (ti, wr, wi, dr, di, ar, ai, br, bi) = (&mut ti[..n], &mut wr[..n], &mut wi[..n], &dr[..n], &di[..n], &ar[..n], &ai[..n], &br[..n], &bi[..n]);
    for k in 0..n {
        tr[k] = di[k].mul_add(-ai[k], dr[k].mul_add(ar[k], tr[k])); ti[k] = di[k].mul_add(ar[k], dr[k].mul_add(ai[k], ti[k]));
        wr[k] = di[k].mul_add(bi[k], dr[k].mul_add(-br[k], wr[k])); wi[k] = di[k].mul_add(-br[k], dr[k].mul_add(-bi[k], wi[k]));
    }
}

/// x -= q·t, y += q·w (complex, split).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn q_acc(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], tr: &[f64], ti: &[f64], wr: &[f64], wi: &[f64]) {
    let n = xr.len();
    let (xi, yr, yi, qr, qi, tr, ti, wr, wi) = (&mut xi[..n], &mut yr[..n], &mut yi[..n], &qr[..n], &qi[..n], &tr[..n], &ti[..n], &wr[..n], &wi[..n]);
    for k in 0..n {
        xr[k] = qi[k].mul_add(ti[k], qr[k].mul_add(-tr[k], xr[k])); xi[k] = qi[k].mul_add(-tr[k], qr[k].mul_add(-ti[k], xi[k]));
        yr[k] = qi[k].mul_add(-wi[k], qr[k].mul_add(wr[k], yr[k])); yi[k] = qi[k].mul_add(wr[k], qr[k].mul_add(wi[k], yi[k]));
    }
}

/// Key switching of a block alone (m = 2, the result is then multiplied by the Q_j of the three patterns): t = Σ_l d_l a_l,
/// w = f - Σ_l d_l b_l (the switched block is (-t, w)), in arrays of at most 128 values.
#[allow(clippy::too_many_arguments)]
pub fn ks_tw(tr: &mut [f64], ti: &mut [f64], wr: &mut [f64], wi: &mut [f64], fr: &[f64], fi: &[f64], d: &[[&[f64]; 2]], k: &[[&[f64]; 4]]) {
    let n = tr.len();
    tr.fill(0.0); ti.fill(0.0); wr.copy_from_slice(&fr[..n]); wi.copy_from_slice(&fi[..n]);
    for l in 0..d.len() { ks_level(tr, ti, wr, wi, d[l][0], d[l][1], k[l][0], k[l][1], k[l][2], k[l][3]); }
}

/// x -= q·t, y += q·w (complex, split): product of a switched block by a public polynomial, accumulated.
#[allow(clippy::too_many_arguments)]
pub fn q_acc_pub(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], tr: &[f64], ti: &[f64], wr: &[f64], wi: &[f64]) {
    q_acc(xr, xi, yr, yi, qr, qi, tr, ti, wr, wi)
}

/// Key switching of a block and product by Q_j: fused kernel up to 3 levels; beyond (16 bits: 6 levels), level by
/// level with the accumulators in L1, which is faster there (Apple M3 Max: 21 against 27 ms per ciphertext at 16 bits).
/// Both orders give the same results, bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn ks_block(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qr: &[f64], qi: &[f64], fr: &[f64], fi: &[f64],
                d: &[[&[f64]; 2]], k: &[[&[f64]; 4]]) {
    if d.len() <= 3 { return ks_fused(xr, xi, yr, yi, qr, qi, fr, fi, d, k); }
    let n = xr.len(); assert!(n <= 128);
    let (mut tr, mut ti, mut wr, mut wi) = ([0f64; 128], [0f64; 128], [0f64; 128], [0f64; 128]);
    wr[..n].copy_from_slice(&fr[..n]); wi[..n].copy_from_slice(&fi[..n]);
    for l in 0..d.len() {
        ks_level(&mut tr[..n], &mut ti[..n], &mut wr[..n], &mut wi[..n], d[l][0], d[l][1], k[l][0], k[l][1], k[l][2], k[l][3]);
    }
    q_acc(xr, xi, yr, yi, qr, qi, &tr[..n], &ti[..n], &wr[..n], &wi[..n]);
}

// m = 2 (three patterns): the switched block is computed once, in registers, and multiplied by the Q_j of the three
// patterns, added to their three destination blocks.

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks3_fused_1(x0r: &mut [f64], x0i: &mut [f64], y0r: &mut [f64], y0i: &mut [f64], x1r: &mut [f64], x1i: &mut [f64], y1r: &mut [f64], y1i: &mut [f64], x2r: &mut [f64], x2i: &mut [f64], y2r: &mut [f64], y2i: &mut [f64], q0r: &[f64], q0i: &[f64], q1r: &[f64], q1i: &[f64], q2r: &[f64], q2i: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64]) {
    let n = x0r.len();
    let (x0i, y0r, y0i, x1r, x1i, x2r, x2i, y1r, y1i, y2r, y2i) = (&mut x0i[..n], &mut y0r[..n], &mut y0i[..n], &mut x1r[..n], &mut x1i[..n], &mut x2r[..n], &mut x2i[..n], &mut y1r[..n], &mut y1i[..n], &mut y2r[..n], &mut y2i[..n]);
    let (q0r, q0i, q1r, q1i, q2r, q2i, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i) = (&q0r[..n], &q0i[..n], &q1r[..n], &q1i[..n], &q2r[..n], &q2i[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (q0r[i], q0i[i]);
        x0r[i] = p.mul_add(ti, q.mul_add(-tr, x0r[i])); x0i[i] = p.mul_add(-tr, q.mul_add(-ti, x0i[i]));
        y0r[i] = p.mul_add(-wi, q.mul_add(wr, y0r[i])); y0i[i] = p.mul_add(wr, q.mul_add(wi, y0i[i]));
        let (q, p) = (q1r[i], q1i[i]);
        x1r[i] = p.mul_add(ti, q.mul_add(-tr, x1r[i])); x1i[i] = p.mul_add(-tr, q.mul_add(-ti, x1i[i]));
        y1r[i] = p.mul_add(-wi, q.mul_add(wr, y1r[i])); y1i[i] = p.mul_add(wr, q.mul_add(wi, y1i[i]));
        let (q, p) = (q2r[i], q2i[i]);
        x2r[i] = p.mul_add(ti, q.mul_add(-tr, x2r[i])); x2i[i] = p.mul_add(-tr, q.mul_add(-ti, x2i[i]));
        y2r[i] = p.mul_add(-wi, q.mul_add(wr, y2r[i])); y2i[i] = p.mul_add(wr, q.mul_add(wi, y2i[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks3_fused_2(x0r: &mut [f64], x0i: &mut [f64], y0r: &mut [f64], y0i: &mut [f64], x1r: &mut [f64], x1i: &mut [f64], y1r: &mut [f64], y1i: &mut [f64], x2r: &mut [f64], x2i: &mut [f64], y2r: &mut [f64], y2i: &mut [f64], q0r: &[f64], q0i: &[f64], q1r: &[f64], q1i: &[f64], q2r: &[f64], q2i: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64]) {
    let n = x0r.len();
    let (x0i, y0r, y0i, x1r, x1i, x2r, x2i, y1r, y1i, y2r, y2i) = (&mut x0i[..n], &mut y0r[..n], &mut y0i[..n], &mut x1r[..n], &mut x1i[..n], &mut x2r[..n], &mut x2i[..n], &mut y1r[..n], &mut y1i[..n], &mut y2r[..n], &mut y2i[..n]);
    let (q0r, q0i, q1r, q1i, q2r, q2i, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i) = (&q0r[..n], &q0i[..n], &q1r[..n], &q1i[..n], &q2r[..n], &q2i[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (q0r[i], q0i[i]);
        x0r[i] = p.mul_add(ti, q.mul_add(-tr, x0r[i])); x0i[i] = p.mul_add(-tr, q.mul_add(-ti, x0i[i]));
        y0r[i] = p.mul_add(-wi, q.mul_add(wr, y0r[i])); y0i[i] = p.mul_add(wr, q.mul_add(wi, y0i[i]));
        let (q, p) = (q1r[i], q1i[i]);
        x1r[i] = p.mul_add(ti, q.mul_add(-tr, x1r[i])); x1i[i] = p.mul_add(-tr, q.mul_add(-ti, x1i[i]));
        y1r[i] = p.mul_add(-wi, q.mul_add(wr, y1r[i])); y1i[i] = p.mul_add(wr, q.mul_add(wi, y1i[i]));
        let (q, p) = (q2r[i], q2i[i]);
        x2r[i] = p.mul_add(ti, q.mul_add(-tr, x2r[i])); x2i[i] = p.mul_add(-tr, q.mul_add(-ti, x2i[i]));
        y2r[i] = p.mul_add(-wi, q.mul_add(wr, y2r[i])); y2i[i] = p.mul_add(wr, q.mul_add(wi, y2i[i]));
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks3_fused_3(x0r: &mut [f64], x0i: &mut [f64], y0r: &mut [f64], y0i: &mut [f64], x1r: &mut [f64], x1i: &mut [f64], y1r: &mut [f64], y1i: &mut [f64], x2r: &mut [f64], x2i: &mut [f64], y2r: &mut [f64], y2i: &mut [f64], q0r: &[f64], q0i: &[f64], q1r: &[f64], q1i: &[f64], q2r: &[f64], q2i: &[f64], fr: &[f64], fi: &[f64], d0r: &[f64], d0i: &[f64], a0r: &[f64], a0i: &[f64], b0r: &[f64], b0i: &[f64], d1r: &[f64], d1i: &[f64], a1r: &[f64], a1i: &[f64], b1r: &[f64], b1i: &[f64], d2r: &[f64], d2i: &[f64], a2r: &[f64], a2i: &[f64], b2r: &[f64], b2i: &[f64]) {
    let n = x0r.len();
    let (x0i, y0r, y0i, x1r, x1i, x2r, x2i, y1r, y1i, y2r, y2i) = (&mut x0i[..n], &mut y0r[..n], &mut y0i[..n], &mut x1r[..n], &mut x1i[..n], &mut x2r[..n], &mut x2i[..n], &mut y1r[..n], &mut y1i[..n], &mut y2r[..n], &mut y2i[..n]);
    let (q0r, q0i, q1r, q1i, q2r, q2i, fr, fi, d0r, d0i, a0r, a0i, b0r, b0i, d1r, d1i, a1r, a1i, b1r, b1i, d2r, d2i, a2r, a2i, b2r, b2i) = (&q0r[..n], &q0i[..n], &q1r[..n], &q1i[..n], &q2r[..n], &q2i[..n], &fr[..n], &fi[..n], &d0r[..n], &d0i[..n], &a0r[..n], &a0i[..n], &b0r[..n], &b0i[..n], &d1r[..n], &d1i[..n], &a1r[..n], &a1i[..n], &b1r[..n], &b1i[..n], &d2r[..n], &d2i[..n], &a2r[..n], &a2i[..n], &b2r[..n], &b2i[..n]);
    for i in 0..n {
        let (mut tr, mut ti, mut wr, mut wi) = (0.0f64, 0.0f64, fr[i], fi[i]);
        let (dr, di, ar, ai, br, bi) = (d0r[i], d0i[i], a0r[i], a0i[i], b0r[i], b0i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d1r[i], d1i[i], a1r[i], a1i[i], b1r[i], b1i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (dr, di, ar, ai, br, bi) = (d2r[i], d2i[i], a2r[i], a2i[i], b2r[i], b2i[i]);
        tr = di.mul_add(-ai, dr.mul_add(ar, tr)); ti = di.mul_add(ar, dr.mul_add(ai, ti));
        wr = di.mul_add(bi, dr.mul_add(-br, wr)); wi = di.mul_add(-br, dr.mul_add(-bi, wi));
        let (q, p) = (q0r[i], q0i[i]);
        x0r[i] = p.mul_add(ti, q.mul_add(-tr, x0r[i])); x0i[i] = p.mul_add(-tr, q.mul_add(-ti, x0i[i]));
        y0r[i] = p.mul_add(-wi, q.mul_add(wr, y0r[i])); y0i[i] = p.mul_add(wr, q.mul_add(wi, y0i[i]));
        let (q, p) = (q1r[i], q1i[i]);
        x1r[i] = p.mul_add(ti, q.mul_add(-tr, x1r[i])); x1i[i] = p.mul_add(-tr, q.mul_add(-ti, x1i[i]));
        y1r[i] = p.mul_add(-wi, q.mul_add(wr, y1r[i])); y1i[i] = p.mul_add(wr, q.mul_add(wi, y1i[i]));
        let (q, p) = (q2r[i], q2i[i]);
        x2r[i] = p.mul_add(ti, q.mul_add(-tr, x2r[i])); x2i[i] = p.mul_add(-tr, q.mul_add(-ti, x2i[i]));
        y2r[i] = p.mul_add(-wi, q.mul_add(wr, y2r[i])); y2i[i] = p.mul_add(wr, q.mul_add(wi, y2i[i]));
    }
}

/// Key switching of a block, multiplied by the Q_j of the three patterns (m = 2): fused up to 3 levels, level by level
/// beyond (accumulators in L1). o[P] = (x.re, x.im, y.re, y.im) of the destination block of pattern P, q[P] = Q_j^P.
#[allow(clippy::too_many_arguments)]
pub fn ks_block3(o: [[&mut [f64]; 4]; 3], q: [[&[f64]; 2]; 3], fr: &[f64], fi: &[f64], d: &[[&[f64]; 2]], k: &[[&[f64]; 4]]) {
    let [[x0r, x0i, y0r, y0i], [x1r, x1i, y1r, y1i], [x2r, x2i, y2r, y2i]] = o;
    match d.len() {
        1 => return ks3_fused_1(x0r, x0i, y0r, y0i, x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], fr, fi,
                                d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3]),
        2 => return ks3_fused_2(x0r, x0i, y0r, y0i, x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], fr, fi,
                                d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3]),
        3 => return ks3_fused_3(x0r, x0i, y0r, y0i, x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], fr, fi,
                                d[0][0], d[0][1], k[0][0], k[0][1], k[0][2], k[0][3], d[1][0], d[1][1], k[1][0], k[1][1], k[1][2], k[1][3],
                                d[2][0], d[2][1], k[2][0], k[2][1], k[2][2], k[2][3]),
        _ => {}
    }
    let n = x0r.len(); assert!(n <= 128);
    let (mut tr, mut ti, mut wr, mut wi) = ([0f64; 128], [0f64; 128], [0f64; 128], [0f64; 128]);
    ks_tw(&mut tr[..n], &mut ti[..n], &mut wr[..n], &mut wi[..n], fr, fi, d, k);
    q_acc(x0r, x0i, y0r, y0i, q[0][0], q[0][1], &tr[..n], &ti[..n], &wr[..n], &wi[..n]);
    q_acc(x1r, x1i, y1r, y1i, q[1][0], q[1][1], &tr[..n], &ti[..n], &wr[..n], &wi[..n]);
    q_acc(x2r, x2i, y2r, y2i, q[2][0], q[2][1], &tr[..n], &ti[..n], &wr[..n], &wi[..n]);
}

// CombBoot, key switchings source block by source block with the digits and the FFT of b held in registers (KS_MODE 2
// and 3): for W values of the source block, the 2ℓ + 2 inputs are loaded once and used by the B - 1 automorphisms, whose
// keys are read in one contiguous stream ([j-1][level][A.re, A.im, B.re, B.im][L]), the products by Q_j added to the
// destination blocks dst[j-1] (same operations, in the same order for each value, as ks_block over j: bit-identical to
// KS_MODE 1).

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn ks_src_g<const GL: usize, const W: usize>(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qa_re: &[f64], qa_im: &[f64],
                                              fr: &[f64], fi: &[f64], d: &[[&[f64]; 2]], kb: &[f64], dst: &[usize], lr: usize) {
    let (h, nj, kst) = (xr.len(), dst.len(), GL * 4 * lr);
    assert!(xi.len() == h && yr.len() == h && yi.len() == h && lr % W == 0 && fr.len() >= lr && fi.len() >= lr);
    assert!(kb.len() >= nj * kst && qa_re.len() >= (nj + 1) * lr && qa_im.len() >= (nj + 1) * lr && d.len() == GL);
    for l in 0..GL { assert!(d[l][0].len() >= lr && d[l][1].len() >= lr); }
    for &o in dst { assert!(o + lr <= h); }
    unsafe {
        for i0 in (0..lr).step_by(W) {
            let (mut dr, mut di) = ([[0f64; W]; GL], [[0f64; W]; GL]);
            for l in 0..GL { for w in 0..W { dr[l][w] = *d[l][0].get_unchecked(i0 + w); di[l][w] = *d[l][1].get_unchecked(i0 + w); } }
            let (mut f_r, mut f_i) = ([0f64; W], [0f64; W]);
            for w in 0..W { f_r[w] = *fr.get_unchecked(i0 + w); f_i[w] = *fi.get_unchecked(i0 + w); }
            for jj in 0..nj {
                let o = *dst.get_unchecked(jj) + i0;
                let k = kb.as_ptr().add(jj * kst + i0);
                let (mut tr, mut ti, mut wr, mut wi) = ([0f64; W], [0f64; W], f_r, f_i);
                for l in 0..GL {
                    let kl = k.add(l * 4 * lr);
                    for w in 0..W {
                        let (ar, ai, br, bi) = (*kl.add(w), *kl.add(lr + w), *kl.add(2 * lr + w), *kl.add(3 * lr + w));
                        let (dr_, di_) = (dr[l][w], di[l][w]);
                        tr[w] = di_.mul_add(-ai, dr_.mul_add(ar, tr[w])); ti[w] = di_.mul_add(ar, dr_.mul_add(ai, ti[w]));
                        wr[w] = di_.mul_add(bi, dr_.mul_add(-br, wr[w])); wi[w] = di_.mul_add(-br, dr_.mul_add(-bi, wi[w]));
                    }
                }
                let (qp, pp) = (qa_re.as_ptr().add((jj + 1) * lr + i0), qa_im.as_ptr().add((jj + 1) * lr + i0));
                for w in 0..W {
                    let (q, p) = (*qp.add(w), *pp.add(w));
                    let x = xr.get_unchecked_mut(o + w); *x = p.mul_add(ti[w], q.mul_add(-tr[w], *x));
                    let x = xi.get_unchecked_mut(o + w); *x = p.mul_add(-tr[w], q.mul_add(-ti[w], *x));
                    let y = yr.get_unchecked_mut(o + w); *y = p.mul_add(-wi[w], q.mul_add(wr[w], *y));
                    let y = yi.get_unchecked_mut(o + w); *y = p.mul_add(wr[w], q.mul_add(wi[w], *y));
                }
            }
        }
    }
}

/// Dispatch of ks_src_g on ℓ = d.len() (1 to 6) and the number W of values per step (2, 4 or 8; 8 up to ℓ = 2 and
/// 4 up to ℓ = 3, else the largest allowed).
#[allow(clippy::too_many_arguments)]
pub fn ks_src(xr: &mut [f64], xi: &mut [f64], yr: &mut [f64], yi: &mut [f64], qa_re: &[f64], qa_im: &[f64], fr: &[f64], fi: &[f64],
              d: &[[&[f64]; 2]], kb: &[f64], dst: &[usize], lr: usize, w: usize) {
    macro_rules! go { ($l:literal, $w:literal) => { ks_src_g::<$l, $w>(xr, xi, yr, yi, qa_re, qa_im, fr, fi, d, kb, dst, lr) } }
    match (d.len(), w) {
        (1, 8) => go!(1, 8), (1, 4) => go!(1, 4), (1, _) => go!(1, 2),
        (2, 8) => go!(2, 8), (2, 4) => go!(2, 4), (2, _) => go!(2, 2),
        (3, 4) | (3, 8) => go!(3, 4), (3, _) => go!(3, 2),
        (4, _) => go!(4, 2), (5, _) => go!(5, 2), (6, _) => go!(6, 2),
        (l, _) => panic!("ks_src: {l} levels"),
    }
}
