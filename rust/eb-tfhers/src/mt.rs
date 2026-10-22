//! Tools of the multi-threaded versions: a spin barrier (workers that live for the whole bootstrap and synchronize a
//! few times per iteration, at a cost of a fraction of a microsecond), shared buffers written on disjoint ranges, the
//! signed decomposition of TFHE-rs and a parallel external product (the CMux of the rotation) in the Fourier domain
//! of TFHE-rs, with the Fourier bootstrapping keys of TFHE-rs.
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial;
use tfhe::core_crypto::prelude::*;
use tfhe_fft::c64;

/// Optional profile of worker 0 (ns per phase), enabled by MTPROF=1.
pub static PROF: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
pub fn prof_on() -> bool { std::env::var("MTPROF").is_ok() }
pub struct Lap { on: bool, t: std::time::Instant }
impl Lap {
    pub fn new(on: bool) -> Self { Lap { on, t: std::time::Instant::now() } }
    #[inline] pub fn lap(&mut self, i: usize) {
        if self.on { let n = std::time::Instant::now(); PROF[i].fetch_add((n - self.t).as_nanos() as u64, Ordering::Relaxed); self.t = n; }
    }
}

pub struct SpinBarrier { n: usize, count: AtomicUsize, gen: AtomicUsize }
impl SpinBarrier {
    pub fn new(n: usize) -> Self { SpinBarrier { n, count: AtomicUsize::new(0), gen: AtomicUsize::new(0) } }
    #[inline]
    pub fn wait(&self) {
        if self.n == 1 { return; }
        let g = self.gen.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.gen.store(g.wrapping_add(1), Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.gen.load(Ordering::Acquire) == g {
                std::hint::spin_loop(); spins += 1;
                if spins > (1 << 22) { std::thread::yield_now(); }
            }
        }
    }
}

/// Shared buffer: each worker writes a range that no other worker reads or writes in the same phase.
#[derive(Clone, Copy)]
pub struct Sh<T>(pub *mut T);
unsafe impl<T> Send for Sh<T> {}
unsafe impl<T> Sync for Sh<T> {}
impl<T> Sh<T> {
    pub fn of(v: &mut [T]) -> Self { Sh(v.as_mut_ptr()) }
    /// # Safety: no overlapping mutable access from another worker in the same phase.
    #[inline] #[allow(clippy::mut_from_ref)]
    pub unsafe fn s<'a>(self, off: usize, len: usize) -> &'a mut [T] { std::slice::from_raw_parts_mut(self.0.add(off), len) }
}

/// Range of the share w of t in 0..n (contiguous, sizes differing by at most one).
#[inline] pub fn share(n: usize, t: usize, w: usize) -> (usize, usize) { (n * w / t, n * (w + 1) / t) }

/// Signed decomposition of TFHE-rs (SignedDecomposer, native modulus): the digits of x, least significant level first
/// (the storage order of the levels of its GGSW ciphertexts), as u64 (two's complement).
#[inline]
pub fn tfhe_digits(x: u64, b: u32, l: usize, out: &mut [u64]) {
    let rep = b * l as u32; let nr = 64 - rep;
    let mut res = x >> (nr - 1); let rb = res & 1; res = (res + 1) >> 1;
    let mask = u64::MAX >> (64 - rep); res &= mask;
    let nb = ((res.wrapping_sub(1) | (rb << (rep - 1))) & res) >> (rep - 1);
    let mut state = res.wrapping_sub(nb << rep);
    let mb = (1u64 << b) - 1;
    for o in out.iter_mut().take(l) {
        let r = state & mb; state = ((state as i64) >> b) as u64;
        let carry = ((r.wrapping_sub(1) | state) & r) >> (b - 1);
        state = state.wrapping_add(carry);
        *o = r.wrapping_sub(carry << b);
    }
}

/// Parallel external product C <- C + ExtProd(GGSW, D - C), with D = (da, db) given in the standard domain, or
/// D = X^rot C when rot is given (the CMux of the blind rotation, computed on the fly), split in
/// three phases separated by barriers: digits (coefficients split), forward FFTs (2ℓ jobs) and products (frequencies
/// split), backward FFTs added to C (2 jobs). Buffers: ed (2ℓN u64), fd (2ℓ N/2 c64), fo (N c64).
pub struct ExtProdBufs { pub ed: Sh<u64>, pub fd: Sh<c64>, pub fo: Sh<c64> }

#[allow(clippy::too_many_arguments)]
pub fn ext_prod_mt(w: usize, t: usize, n: usize, d: Sh<u64>, c: Sh<u64>, ggsw: &[c64], b: u32, l: usize, bufs: &ExtProdBufs,
                   fft: FftView<'_>, stack: &mut tfhe::core_crypto::commons::computation_buffers::ComputationBuffers, bar: &SpinBarrier, lap: &mut Lap,
                   rot: Option<usize>) {
    let h = n / 2;
    // digits of D - C (D holds C_L): coefficients split, level by level (contiguous, vectorizable)
    let (c0, c1) = share(n, t, w); let m = c1 - c0;
    let rep = b * l as u32; let nr = 64 - rep; let mask = u64::MAX >> (64 - rep); let mb = (1u64 << b) - 1;
    let mut st = vec![0u64; m];
    for r in 0..2 {
        let cc = unsafe { c.s(r * n, n) };
        for i in 0..m {
            let j = c0 + i;
            let dv = match rot {
                None => unsafe { *d.0.add(r * n + j) },
                Some(dg) => { let s = (j + 2 * n - dg) % (2 * n); if s < n { cc[s] } else { cc[s - n].wrapping_neg() } }
            };
            let x = dv.wrapping_sub(cc[j]);
            let mut res = x >> (nr - 1); let rb = res & 1; res = ((res + 1) >> 1) & mask;
            let nb = ((res.wrapping_sub(1) | (rb << (rep - 1))) & res) >> (rep - 1);
            st[i] = res.wrapping_sub(nb << rep);
        }
        for s in 0..l {
            let out = unsafe { bufs.ed.s((r * l + s) * n + c0, m) };
            for i in 0..m {
                let v = st[i]; let rr = v & mb; let v = ((v as i64) >> b) as u64;
                let carry = ((rr.wrapping_sub(1) | v) & rr) >> (b - 1);
                st[i] = v.wrapping_add(carry); out[i] = rr.wrapping_sub(carry << b);
            }
        }
    }
    bar.wait(); lap.lap(4);
    for k in (w..2 * l).step_by(t) {
        let src = unsafe { bufs.ed.s(k * n, n) }; let dst = unsafe { bufs.fd.s(k * h, h) };
        fft.forward_as_integer(FourierPolynomial { data: dst }, Polynomial::from_container(&*src), stack.stack());
    }
    bar.wait(); lap.lap(5);
    // products: frequencies split; GGSW layout [level (least significant first)][input row][output poly][N/2]
    let (f0, f1) = share(h / 8, t, w); let (f0, f1) = (8 * f0, 8 * f1);   // multiples of 8: same SIMD lanes for every t
    let (o0, o1) = unsafe { (bufs.fo.s(f0, f1 - f0), bufs.fo.s(h + f0, f1 - f0)) };
    let zero = c64 { re: 0.0, im: 0.0 };
    o0.fill(zero); o1.fill(zero);
    for s in 0..l {
        for r in 0..2 {
            let dk = unsafe { bufs.fd.s((r * l + s) * h + f0, f1 - f0) };
            let g0 = &ggsw[((s * 2 + r) * 2) * h + f0..((s * 2 + r) * 2) * h + f1];
            let g1 = &ggsw[((s * 2 + r) * 2 + 1) * h + f0..((s * 2 + r) * 2 + 1) * h + f1];
            pulp::Arch::new().dispatch(Mac2 { o0: &mut *o0, o1: &mut *o1, d: dk, g0, g1 });
        }
    }
    bar.wait(); lap.lap(6);
    for o in (w..2).step_by(t) {
        let dst = unsafe { c.s(o * n, n) }; let src = unsafe { bufs.fo.s(o * h, h) };
        fft.add_backward_as_torus(Polynomial::from_container(dst), FourierPolynomial { data: &*src }, stack.stack());
    }
    bar.wait(); lap.lap(7);
}

/// o0 += d g0, o1 += d g1 (complex, SIMD of pulp as in update_with_fmadd of TFHE-rs).
pub struct Mac2<'a> { pub o0: &'a mut [c64], pub o1: &'a mut [c64], pub d: &'a [c64], pub g0: &'a [c64], pub g1: &'a [c64] }
impl pulp::WithSimd for Mac2<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let (a, at) = S::as_mut_simd_c64s(self.o0); let (b, bt) = S::as_mut_simd_c64s(self.o1);
        let (d, dt) = S::as_simd_c64s(self.d); let (x, xt) = S::as_simd_c64s(self.g0); let (y, yt) = S::as_simd_c64s(self.g1);
        for i in 0..d.len() { a[i] = simd.mul_add_c64s(x[i], d[i], a[i]); b[i] = simd.mul_add_c64s(y[i], d[i], b[i]); }
        for i in 0..dt.len() {
            let (u, v, z) = (dt[i], xt[i], at[i]);
            at[i] = c64 { re: u.im.mul_add(-v.im, u.re.mul_add(v.re, z.re)), im: u.im.mul_add(v.re, u.re.mul_add(v.im, z.im)) };
            let (v, z) = (yt[i], bt[i]);
            bt[i] = c64 { re: u.im.mul_add(-v.im, u.re.mul_add(v.re, z.re)), im: u.im.mul_add(v.re, u.re.mul_add(v.im, z.im)) };
        }
    }
}
