//! Error bootstrap in GLWE of rank kk and polynomial size np, kk np = N = 2048 (review R1, M2): the alternative to
//! CombBoot that bootstraps each ciphertext on its own, at p = 2 np, with the same extracted LWE key of dimension 2048
//! and the same GLWE noise σ_128(2048). Same steps as eb::EbKeys::refresh: stair key switch 2048 -> n, rounding at
//! R = p̂ p with recentering and selection, rotation of the test polynomial v_j = min(j, np - j)/R (j < np) in the body
//! of a trivial GLWE of rank kk, sample extraction (LWE of dimension kk np = 2048), subtraction.
//! Rotation: m = 1, the CMux of TFHE-rs (blind_rotate_assign, rank kk); m = 2, our multi-bit form (as eb::Own2, extended
//! to rank kk: one decomposition and (kk+1) ℓ forward FFTs per pair, 3 (kk+1)^2 ℓ pointwise products, kk+1 backward FFTs).
//! Modes (MODE=...):
//!   cal:  for each (kk, np) of SHAPES and each gadget of GADS, at n = NN (default 700): time of the rotation per
//!         coordinate (minimum over REPS) and noise of the output (NREF refreshes, no key switch) against the model
//!         n [(kk+1) ℓ np (B^2+2)/12 σ^2] + n/2 (1 + kk np/2) B^-2ℓ/12 (m = 1; multi-bit form for m = 2): the excess
//!         is written as c B^2 ℓ per CMux (m = 1) or per pair (m = 2), log2 c.
//!   full: whole refreshes of the configurations GCONFIGS (chosen by the noise model), noise and margin, times
//!         (median and minimum of REPS), against, in the same session, the error bootstrap at N = 2048 (CONFIGS_RS2, EB=0
//!         to skip), CombBoot per ciphertext (COMB_CONFIGS at 2, 4, 6, 8, 12, 16 bits, COMB=0 to skip) and the refresh of
//!         FheUint{2k} by TFHE-rs with its default parameters, on one thread (TF=0 to skip); KKS=2,4 keeps some shapes
//!         (KKS= : none).
//!   ftime: whole refreshes interleaved round by round, for each k: error bootstrap at N = 2048, GLWE configurations
//!         (KKS), CombBoot per ciphertext (COMB=0 to skip) and TFHE-rs (FheUint{2k}), minimum and median of REPS rounds,
//!         ratios to TFHE-rs.
//! Run (from rust/eb-tfhers; results in results/rust):
//!   RAYON_NUM_THREADS=1 MODE=cal cargo run --release --bin eb_glwe | tee ../../results/rust/glwe_cal_mac.txt
//!   RAYON_NUM_THREADS=1 MODE=time cargo run --release --bin eb_glwe | tee ../../results/rust/glwe_time_mac.txt
//!   (the noise model then chooses the configurations of src/glwe_cfgs.rs from these two logs, margin 0.2 bit)
//!   RAYON_NUM_THREADS=1 MODE=ftime cargo run --release --bin eb_glwe | tee ../../results/rust/glwe_ftime_mac.txt
//!   RAYON_NUM_THREADS=1 MODE=full EB=0 COMB=0 TF=0 cargo run --release --bin eb_glwe | tee ../../results/rust/glwe_full_mac.txt
use eb_tfhers::comb::{CombKeys, COMB_CONFIGS};
use eb_tfhers::eb::{cm, tor, EbKeys};
use eb_tfhers::params::{config_rs2, log2_sigma128, stair_var, LOG2_SIGMA_N, N, NU, U128};
use eb_tfhers::rounding::{round_select, Rounded};
use eb_tfhers::stair::StairKey;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use tfhe::core_crypto::commons::math::random::Seed;
use tfhe::core_crypto::prelude::*;
use tfhe_fft::c64;

struct Msed1<'a> { r: &'a Rounded, two_n: u64 }
impl ModulusSwitchedLweCiphertext<usize> for Msed1<'_> {
    fn log_modulus(&self) -> CiphertextModulusLog { CiphertextModulusLog(self.two_n.trailing_zeros() as usize) }
    fn lwe_dimension(&self) -> LweDimension { LweDimension(self.r.abar.len()) }
    fn body(&self) -> usize { (self.r.bbar % self.two_n) as usize }
    fn mask(&self) -> impl ExactSizeIterator<Item = usize> + '_ { self.r.abar.iter().map(move |&a| (a % self.two_n) as usize) }
}

/// Our multi-bit rotation (m = 2) in rank kk, polynomial size np (eb::Own2 for kk = 1, np = 2048).
struct OwnK { f: FourierLweBootstrapKeyOwned, theta: Vec<usize>, roots: Vec<c64>, fft: Fft, kk: usize, np: usize }

/// For one pair: f_c += Σ_P w_P ⊙ Σ_{s,r} d_{r,s} ⊙ K_P[s][r][c], c = 0..kk (t: scratch of kk+1 Fourier polynomials).
struct PairK<'a> { f: &'a mut [c64], t: &'a mut [c64], w: &'a [c64], d: &'a [c64], k: [&'a [c64]; 3], nz: [bool; 3], l: usize, h: usize, kp: usize }
impl pulp::WithSimd for PairK<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let (h, l, kp) = (self.h, self.l, self.kp);
        let zero = simd.splat_c64s(c64 { re: 0.0, im: 0.0 });
        for p in 0..3 {
            if !self.nz[p] { continue; }
            { let (t, tt) = S::as_mut_simd_c64s(self.t); assert!(tt.is_empty()); for x in t.iter_mut() { *x = zero; } }
            for s in 0..l { for r in 0..kp {
                let d = S::as_simd_c64s(&self.d[(r * l + s) * h..(r * l + s + 1) * h]).0;
                for c in 0..kp {
                    let g = S::as_simd_c64s(&self.k[p][((s * kp + r) * kp + c) * h..((s * kp + r) * kp + c + 1) * h]).0;
                    let o = S::as_mut_simd_c64s(&mut self.t[c * h..(c + 1) * h]).0;
                    for i in 0..d.len() { o[i] = simd.mul_add_c64s(g[i], d[i], o[i]); }
                }
            } }
            let w = S::as_simd_c64s(&self.w[p * h..(p + 1) * h]).0;
            for c in 0..kp {
                let t = S::as_simd_c64s(&self.t[c * h..(c + 1) * h]).0;
                let f = S::as_mut_simd_c64s(&mut self.f[c * h..(c + 1) * h]).0;
                for i in 0..w.len() { f[i] = simd.mul_add_c64s(w[i], t[i], f[i]); }
            }
        }
    }
}

impl OwnK {
    fn new<G: ParallelByteRandomGenerator>(small_sk: &LweSecretKeyOwned<u64>, glwe_sk: &GlweSecretKeyOwned<u64>, b: usize, l: usize,
                                           noise: Gaussian<f64>, egen: &mut EncryptionRandomGenerator<G>) -> Self {
        let (kk, np) = (glwe_sk.glwe_dimension().0, glwe_sk.polynomial_size().0);
        let s = small_sk.as_ref(); assert!(s.len() % 2 == 0, "n must be even for m = 2");
        let ind: Vec<u64> = s.chunks(2).flat_map(|p| [p[0] & !p[1] & 1, !p[0] & p[1] & 1, p[0] & p[1]]).collect();
        let ind_sk = LweSecretKey::from_container(ind);
        let std = par_allocate_and_generate_new_lwe_bootstrap_key(&ind_sk, glwe_sk, DecompositionBaseLog(b),
            DecompositionLevelCount(l), noise, cm(), egen);
        let mut f = FourierLweBootstrapKey::new(std.input_lwe_dimension(), std.glwe_size(), std.polynomial_size(),
            std.decomposition_base_log(), std.decomposition_level_count());
        par_convert_standard_lwe_bootstrap_key_to_fourier(&std, &mut f);
        drop(std);
        let fft = Fft::new(PolynomialSize(np)); let two_n = 2 * np;
        let roots = (0..two_n).map(|e| { let t = std::f64::consts::PI * e as f64 / np as f64; c64 { re: t.cos(), im: t.sin() } }).collect();
        let mut buf = ComputationBuffers::new();
        buf.resize(fft.as_view().forward_scratch().unaligned_bytes_required());
        let mut xpoly = vec![0u64; np]; xpoly[1] = 1;
        let mut fx = tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial::new(PolynomialSize(np));
        fft.as_view().forward_as_integer(fx.as_mut_view(), Polynomial::from_container(xpoly.as_slice()), buf.stack());
        let theta: Vec<usize> = fx.data.iter().map(|z| {
            let c = (z.im.atan2(z.re) * np as f64 / std::f64::consts::PI).round() as i64; c.rem_euclid(two_n as i64) as usize
        }).collect();
        assert!(theta.iter().all(|&c| c % 2 == 1), "unexpected Fourier layout");
        OwnK { f, theta, roots, fft, kk, np }
    }

    /// acc <- X^{Σ_u e_u s_u} acc (exponents modulo 2 np, n even); acc: kk+1 polynomials of size np.
    fn rotate(&self, exps: &[usize], acc: &mut [u64]) {
        use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomial;
        let (np, kp) = (self.np, self.kk + 1); let (two_n, h) = (2 * np, np / 2);
        let (b, l) = (self.f.decomposition_base_log().0 as u32, self.f.decomposition_level_count().0);
        let fv = self.fft.as_view();
        let mut buf = ComputationBuffers::new();
        buf.resize(fv.forward_scratch().unaligned_bytes_required().max(fv.backward_scratch().unaligned_bytes_required()));
        let z = c64 { re: 0.0, im: 0.0 };
        let (mut ed, mut fd, mut st) = (vec![0u64; kp * l * np], vec![z; kp * l * h], vec![0u64; np]);
        let (mut f, mut t, mut w) = (vec![z; kp * h], vec![z; kp * h], vec![z; 3 * h]);
        let arch = pulp::Arch::new();
        let ggsw: Vec<&[c64]> = self.f.as_view().into_ggsw_iter().map(|g| g.data()).collect();
        let rep = b * l as u32; let nr = 64 - rep; let mask = u64::MAX >> (64 - rep); let mb = (1u64 << b) - 1;
        for g in 0..exps.len() / 2 {
            let (e0, e1) = (exps[2 * g], exps[2 * g + 1]);
            if e0 == 0 && e1 == 0 { continue; }
            let es = [e0, e1, (e0 + e1) % two_n];
            for r in 0..kp {
                let cc = &acc[r * np..(r + 1) * np];
                for i in 0..np {
                    let x = cc[i];
                    let mut res = x >> (nr - 1); let rb = res & 1; res = ((res + 1) >> 1) & mask;
                    let nb = ((res.wrapping_sub(1) | (rb << (rep - 1))) & res) >> (rep - 1);
                    st[i] = res.wrapping_sub(nb << rep);
                }
                for s in 0..l {
                    let out = &mut ed[(r * l + s) * np..(r * l + s + 1) * np];
                    for i in 0..np {
                        let v = st[i]; let rr = v & mb; let v = ((v as i64) >> b) as u64;
                        let carry = ((rr.wrapping_sub(1) | v) & rr) >> (b - 1);
                        st[i] = v.wrapping_add(carry); out[i] = rr.wrapping_sub(carry << b);
                    }
                }
            }
            for k in 0..kp * l {
                fv.forward_as_integer(FourierPolynomial { data: &mut fd[k * h..(k + 1) * h] },
                                      Polynomial::from_container(&ed[k * np..(k + 1) * np]), buf.stack());
            }
            for j in 0..h {
                let r0 = self.roots[(self.theta[j] * e0) & (two_n - 1)]; let r1 = self.roots[(self.theta[j] * e1) & (two_n - 1)];
                w[j] = c64 { re: r0.re - 1.0, im: r0.im }; w[h + j] = c64 { re: r1.re - 1.0, im: r1.im };
                w[2 * h + j] = c64 { re: r0.re * r1.re - r0.im * r1.im - 1.0, im: r0.re * r1.im + r0.im * r1.re };
            }
            f.fill(z);
            arch.dispatch(PairK { f: &mut f, t: &mut t, w: &w, d: &fd, k: [ggsw[3 * g], ggsw[3 * g + 1], ggsw[3 * g + 2]],
                                  nz: [e0 != 0, e1 != 0, es[2] != 0], l, h, kp });
            for c in 0..kp {
                fv.add_backward_as_torus(Polynomial::from_container(&mut acc[c * np..(c + 1) * np]), FourierPolynomial { data: &f[c * h..(c + 1) * h] }, buf.stack());
            }
        }
    }
}

enum GBsk { Classic(FourierLweBootstrapKeyOwned), OwnK(OwnK) }

struct GKeys {
    kk: usize, np: usize, log_r: u32, vstar: f64,
    big_sk: LweSecretKeyOwned<u64>, small_sk: LweSecretKeyOwned<u64>, bsk: GBsk, stair: Option<StairKey>,
    pool: Vec<(Vec<u64>, u64)>, lut: PlaintextListOwned<u64>,
}

impl GKeys {
    /// prec: p̂ = 4^prec; vterm = 16 V*/p^2 (V* = vterm p^2/16).
    fn new(kk: usize, np: usize, n: usize, m: usize, b: usize, l: usize, prec: u32, vterm: f64, chain: &[(usize, u32, usize)], seed: u64)
           -> (Self, EncryptionRandomGenerator<DefaultRandomGenerator>) {
        assert_eq!(kk * np, N);
        let mut boxed = new_seeder(); let seeder = boxed.as_mut();
        let mut sgen = SecretRandomGenerator::<DefaultRandomGenerator>::new(Seed(seed as u128));
        let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder);
        let glwe_sk = GlweSecretKey::generate_new_binary(GlweDimension(kk), PolynomialSize(np), &mut sgen);
        let big_sk = glwe_sk.clone().into_lwe_secret_key();
        let small_sk = LweSecretKey::from_container(big_sk.as_ref()[..n].to_vec());
        let glwe_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(LOG2_SIGMA_N)), 0.0);
        let bsk = if m == 1 {
            let std = par_allocate_and_generate_new_lwe_bootstrap_key(&small_sk, &glwe_sk, DecompositionBaseLog(b),
                DecompositionLevelCount(l), glwe_noise, cm(), &mut egen);
            let mut f = FourierLweBootstrapKey::new(std.input_lwe_dimension(), std.glwe_size(), std.polynomial_size(),
                std.decomposition_base_log(), std.decomposition_level_count());
            par_convert_standard_lwe_bootstrap_key_to_fourier(&std, &mut f);
            GBsk::Classic(f)
        } else {
            assert_eq!(m, 2); GBsk::OwnK(OwnK::new(&small_sk, &glwe_sk, b, l, glwe_noise, &mut egen))
        };
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
        let stair = if chain.is_empty() { None } else { Some(StairKey::new(big_sk.as_ref(), N, chain, &mut rng)) };
        let pool_noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(n))), 0.0);
        let pool = (0..128).map(|_| {
            let z: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&small_sk, Plaintext(0), pool_noise, cm(), &mut egen);
            (z.get_mask().as_ref().to_vec(), *z.get_body().data)
        }).collect();
        let p = 2 * np;
        let log_r = 2 * prec + p.trailing_zeros();
        let scale = 1u64 << (64 - log_r);
        let lut = PlaintextList::from_container((0..np).map(|i| (if i < np / 2 { i } else { np - i }) as u64 * scale).collect::<Vec<u64>>());
        let vstar = vterm * (p as f64).powi(2) / 16.0;
        (GKeys { kk, np, log_r, vstar, big_sk, small_sk, bsk, stair, pool, lut }, egen)
    }

    fn rotate(&self, r: &Rounded) -> GlweCiphertextOwned<u64> {
        let mut acc = allocate_and_trivially_encrypt_new_glwe_ciphertext(GlweDimension(self.kk).to_glwe_size(), &self.lut, cm());
        let two_n = 2 * self.np as u64;
        match &self.bsk {
            GBsk::Classic(f) => blind_rotate_assign(&Msed1 { r, two_n }, &mut acc, f),
            GBsk::OwnK(o) => {
                let (np, tn) = (self.np, 2 * self.np);
                let dg = (tn - (r.bbar as usize % tn)) % tn;
                let c = acc.as_mut();
                let off = self.kk * np;
                let body: Vec<u64> = c[off..].to_vec();
                for j in 0..np { let s = (j + tn - dg) % tn; c[off + j] = if s < np { body[s] } else { body[s - np].wrapping_neg() }; }
                let exps: Vec<usize> = r.abar.iter().map(|&a| (a % tn as u64) as usize).collect();
                o.rotate(&exps, c);
            }
        }
        acc
    }

    fn refresh_small<RG: Rng>(&self, a: &[u64], b: u64, rng: &mut RG) -> LweCiphertextOwned<u64> {
        let r = round_select(a, b, self.log_r, &self.pool, self.vstar, rng);
        let acc = self.rotate(&r);
        let mut out = LweCiphertext::new(0u64, LweSize(N + 1), cm());
        extract_lwe_sample_from_glwe_ciphertext(&acc, &mut out, MonomialDegree(0));
        let shift = 64 - self.log_r;
        let mut fin = LweCiphertext::new(0u64, LweSize(N + 1), cm());
        {
            let (mut fm, fb) = fin.get_mut_mask_and_body();
            for (u, &x) in r.abar.iter().enumerate() { fm.as_mut()[u] = x << shift; }
            *fb.data = r.bbar << shift;
        }
        lwe_ciphertext_sub_assign(&mut fin, &out);
        fin
    }

    fn refresh<RG: Rng>(&self, ct: &LweCiphertextOwned<u64>, rng: &mut RG, rbit: &mut u64) -> LweCiphertextOwned<u64> {
        let (a, b) = self.stair.as_ref().expect("no key switch").switch(ct.get_mask().as_ref(), *ct.get_body().data, rbit);
        self.refresh_small(&a, b, rng)
    }
}

/// Model of the output variance of the rotation, without the error of the FFT (binary keys; m = 2: multi-bit form).
fn var_rot_g(kk: usize, np: usize, n: usize, m: usize, b: usize, l: usize) -> f64 {
    let s2 = 2f64.powf(2.0 * LOG2_SIGMA_N);
    let key = ((kk + 1) * l * np) as f64 * (2f64.powi(2 * b as i32) + 2.0) / 12.0 * s2;
    let rnd = (1.0 + (kk * np) as f64 / 2.0) * 2f64.powf(-2.0 * (l * b) as f64) / 12.0;
    let g = n as f64 / m as f64;
    ((((1 << m) - 1) as f64) * g * key + g * (1.0 - 2f64.powi(-(m as i32))) * rnd) * if m > 1 { 2.0 } else { 1.0 }
}

fn envs(name: &str, d: &str) -> String { std::env::var(name).unwrap_or(d.into()) }
fn envu(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

fn cal() {
    let shapes: Vec<(usize, usize)> = envs("SHAPES", "1:2048,2:1024,4:512,8:256,16:128").split(',')
        .map(|s| { let v: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    let gads: Vec<(usize, usize)> = envs("GADS", "23:1,20:1,15:2,14:2,11:3,9:4,8:5").split(',')
        .map(|g| { let v: Vec<usize> = g.split(':').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    let ms: Vec<usize> = envs("MS", "1,2").split(',').map(|x| x.parse().unwrap()).collect();
    let n = envu("NN", 700); let nref = envu("NREF", 200); let reps = envu("REPS", 5);
    println!("GLWE error bootstrap, calibration: n = {n}, p̂ = 4, {nref} refreshes (no key switch); log2 σ measured / model without FFT; FFT: log2 c, c B^2 ℓ per CMux (m = 1) or pair (m = 2); time of the rotation per coordinate (µs, min of {reps})");
    for &(kk, np) in &shapes { for &m in &ms { for &(b, l) in &gads {
        let (keys, mut egen) = GKeys::new(kk, np, n, m, b, l, 1, 1e9, &[], 11 + (kk * 100 + b * 10 + l) as u64);
        let mut rng = StdRng::seed_from_u64((kk * 1000 + b * 10 + l + m) as u64);
        let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-20)), 0.0);
        let mut s2 = 0.0; let mut tmin = f64::INFINITY;
        for i in 0..nref.max(reps + 1) {
            let mu = rng.gen_range(0..4u64) << 62;
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.small_sk, Plaintext(mu), enc, cm(), &mut egen);
            let r = round_select(ct.get_mask().as_ref(), *ct.get_body().data, keys.log_r, &keys.pool, keys.vstar, &mut rng);
            let t0 = Instant::now();
            let acc = keys.rotate(&r);
            let dt = t0.elapsed().as_secs_f64() * 1e6;
            if i >= 1 && i <= reps { tmin = tmin.min(dt); }
            // finish (as refresh_small)
            let mut out = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            extract_lwe_sample_from_glwe_ciphertext(&acc, &mut out, MonomialDegree(0));
            let shift = 64 - keys.log_r;
            let mut fin = LweCiphertext::new(0u64, LweSize(N + 1), cm());
            { let (mut fm, fb) = fin.get_mut_mask_and_body();
              for (u, &x) in r.abar.iter().enumerate() { fm.as_mut()[u] = x << shift; } *fb.data = r.bbar << shift; }
            lwe_ciphertext_sub_assign(&mut fin, &out);
            if i < nref { let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, &fin).0.wrapping_sub(mu)); s2 += e * e; }
        }
        let vo = s2 / nref as f64; let pv = var_rot_g(kk, np, n, m, b, l);
        let g = n as f64 / m as f64; let bb = 2f64.powi(2 * b as i32) * l as f64;
        let c = if vo > pv { ((vo - pv) / (g * bb)).log2() } else { f64::NAN };
        println!("kk = {kk:2}, np = {np:4}, m = {m}, (2^{b:2},{l}): out {:7.2} / {:7.2}   FFT log2 c = {:7.2}   rotation {:8.2} µs/coord ({:.2} ms)",
                 0.5 * vo.log2(), 0.5 * pv.log2(), c, tmin / n as f64, tmin / 1e3);
        println!("CSV,cal,{kk},{np},{m},{b},{l},{n},{:.4},{:.4},{:.3},{:.4}", 0.5 * vo.log2(), 0.5 * pv.log2(), c, tmin / n as f64);
    } } }
}

/// Configurations chosen by the noise model: (prec, kk, np, m, n, b, l, vterm, FFT log2 c, chain).
struct GCfg { prec: u32, kk: usize, np: usize, m: usize, n: usize, b: usize, l: usize, vterm: f64, cfft: f64, chain: &'static [(usize, u32, usize)] }
include!("../glwe_cfgs.rs");

fn bench<O: FnMut()>(reps: usize, mut op: O) -> (f64, f64) {
    op();
    let mut ts: Vec<f64> = (0..reps).map(|_| { let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e3 }).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[reps / 2], ts[0])
}

fn full() {
    let reps = envu("REPS", 10); let nref = envu("NREF", 300);
    let precs: Vec<u32> = envs("KS", "1,2,3,4,5,6,7,8").split(',').map(|x| x.parse().unwrap()).collect();
    let eb = envu("EB", 1) == 1; let comb = envu("COMB", 1) == 1; let tf = envu("TF", 1) == 1;
    let shapes: Vec<usize> = envs("KKS", "1,2,4,8,16").split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    println!("GLWE error bootstrap, whole refresh (stair key switch from 2048, rounding with selection, rotation, extraction, subtraction), one thread; times in ms (median, min of {reps}); noise on {nref} refreshes; margin of (4p̂)^2 (ν σ_out^2 + σ_KS^2) + 16 V*/p^2 ≤ 1/U^2");
    for &prec in &precs {
        if comb {
            for cfg in COMB_CONFIGS.iter().filter(|c| c.bits == 2 * prec) {
                let keys = CombKeys::new(cfg, 900 + cfg.bits as u64);
                let mut rng = StdRng::seed_from_u64(cfg.bits as u64); let mut rbit = 0x853c49e6748fea9bu64;
                let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(new_seeder().as_mut().seed(), new_seeder().as_mut());
                let lp = cfg.bits; let bc = keys.bcols;
                let noise = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-30)), 0.0);
                let mus: Vec<u64> = (0..bc).map(|_| rng.gen_range(0..(1u64 << lp)) << (64 - lp)).collect();
                let cts: Vec<LweCiphertextOwned<u64>> = mus.iter().map(|&mu| allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), noise, cm(), &mut egen)).collect();
                let (outs, _) = keys.refresh(&cts, &mut rng, &mut rbit);
                for (o, &mu) in outs.iter().zip(&mus) {
                    let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, o).0.wrapping_sub(mu));
                    assert!(e.abs() < 1.0 / (4.0 * 2f64.powi(lp as i32)), "wrong refresh");
                }
                let t = bench(reps.min(5), || { let _ = keys.refresh(&cts, &mut rng, &mut rbit); });
                println!("k = {prec} (p̂ = 2^{:2}): CombBoot B = {:2}, p = {}, m = {}, n = {}: {:7.3} ms per ciphertext (min {:7.3}; {:.1} ms per call)",
                         2 * prec, bc, cfg.p, keys.m, cfg.n, t.0 / bc as f64, t.1 / bc as f64, t.1);
                println!("CSV,comb,{prec},{},{},{},{},{:.4},{:.4}", cfg.p, bc, keys.m, cfg.n, t.0 / bc as f64, t.1 / bc as f64);
            }
        }
        if eb {
            let (cfg, mm) = config_rs2(prec);
            let (keys, mut egen) = EbKeys::new(cfg, mm, 1000 + prec as u64);
            let mut rng = StdRng::seed_from_u64(prec as u64); let mut rbit = 0x9e3779b97f4a7c15u64;
            let mu = 1u64 << (64 - 2 * prec);
            let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(N))), 0.0);
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), enc, cm(), &mut egen);
            let t = bench(reps, || { let _ = keys.refresh(&ct, &mut rng, &mut rbit, None); });
            println!("k = {prec} (p̂ = 2^{:2}): EB N = 2048, p = 4096, m = {mm}, n = {}, (2^{},{}): refresh {:7.3} ms (min {:7.3})", 2 * prec, cfg.n, cfg.b, cfg.l, t.0, t.1);
            println!("CSV,eb,{prec},4096,{mm},{},{},{},{:.4},{:.4}", cfg.n, cfg.b, cfg.l, t.0, t.1);
        }
        for c in GCONFIGS.iter().filter(|c| c.prec == prec && shapes.contains(&c.kk)) {
            let t0 = Instant::now();
            let (keys, mut egen) = GKeys::new(c.kk, c.np, c.n, c.m, c.b, c.l, c.prec, c.vterm, c.chain, 2000 + prec as u64 * 10 + c.kk as u64);
            let tkey = t0.elapsed().as_secs_f64();
            let mut rng = StdRng::seed_from_u64(77 + prec as u64); let mut rbit = 0x2545f4914f6cdd1du64;
            let lp = 2 * prec; let window = 1.0 / (4.0 * 2f64.powi(lp as i32));
            let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(N))), 0.0);
            let mu0 = 1u64 << (64 - lp);
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu0), enc, cm(), &mut egen);
            let t = bench(reps, || { let _ = keys.refresh(&ct, &mut rng, &mut rbit); });
            // noise of the output (inputs under s: no key switch) and of the key switch
            let enc_s = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-20)), 0.0);
            let (mut s2, mut emax, mut fails) = (0.0, 0.0f64, 0usize);
            for _ in 0..nref {
                let m: u64 = rng.gen_range(0..(1u64 << lp)); let mu = m << (64 - lp);
                let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.small_sk, Plaintext(mu), enc_s, cm(), &mut egen);
                let out = keys.refresh_small(ct.get_mask().as_ref(), *ct.get_body().data, &mut rng);
                let e = tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu));
                s2 += e * e; emax = emax.max(e.abs()); if e.abs() >= window { fails += 1; }
            }
            let vo = s2 / nref as f64;
            let tiny = Gaussian::from_dispersion_parameter(StandardDev(2f64.powi(-60)), 0.0);
            let mut k2 = 0.0; let nks = 300;
            for _ in 0..nks {
                let mu: u64 = rng.gen();
                let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), tiny, cm(), &mut egen);
                let (a, b) = keys.stair.as_ref().unwrap().switch(ct.get_mask().as_ref(), *ct.get_body().data, &mut rbit);
                let dot = a.iter().zip(keys.small_sk.as_ref()).fold(0u64, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y)));
                let e = tor(b.wrapping_sub(dot).wrapping_sub(mu)); k2 += e * e;
            }
            let vks = k2 / nks as f64;
            let pv = var_rot_g(c.kk, c.np, c.n, c.m, c.b, c.l) + (c.n as f64 / c.m as f64) * 2f64.powf(c.cfft) * 2f64.powi(2 * c.b as i32) * c.l as f64;
            let phat = 4f64.powi(prec as i32);
            let marg = 0.5 * ((1.0 / (U128 * U128)) / ((4.0 * phat).powi(2) * (NU * vo + vks) + c.vterm)).log2();
            let bsk_mb = (c.n / c.m * ((1 << c.m) - 1)) as f64 * ((c.kk + 1) * (c.kk + 1) * c.l * c.np) as f64 * 8.0 / 1e6;
            println!("k = {prec} (p̂ = 2^{:2}): GLWE kk = {:2}, np = {:4}, p = {:4}, m = {}, n = {:4}, (2^{},{}): refresh {:7.3} ms (min {:7.3}); out {:7.2} / {:7.2}, max 2^{:.2}, failures {}; KS {:7.2} / {:7.2}; margin {:5.2} bits; vterm {:.3e}; BSK {:.0} MB (keys {:.0} s)",
                     2 * prec, c.kk, c.np, 2 * c.np, c.m, c.n, c.b, c.l, t.0, t.1, 0.5 * vo.log2(), 0.5 * pv.log2(), emax.log2(), fails,
                     0.5 * vks.log2(), 0.5 * stair_var(N, c.chain, 16).log2(), marg, c.vterm, bsk_mb, tkey);
            println!("CSV,glwe,{prec},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.3},{}", 2 * c.np, c.kk, c.m, c.n, c.b, c.l, t.0, t.1, 0.5 * vo.log2(), 0.5 * vks.log2(), marg, fails);
        }
    }
    if tf {
        // TFHE-rs runs its integer operations on the rayon pool: one thread if the pool is not built yet (otherwise, run
        // with RAYON_NUM_THREADS=1)
        let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
        use tfhe::prelude::*;
        use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint10, FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};
        let (ck, sk) = generate_keys(ConfigBuilder::default().build());
        set_server_key(sk);
        macro_rules! tfr { ($T:ty, $C:ty, $k:expr) => {{
            if precs.contains(&$k) {
                let max: $C = ((1u32 << (2 * $k)) - 1) as $C; let a = <$T>::encrypt((0xB5E3u32 as $C) & max, &ck);
                let r: $C = (&a & max).decrypt(&ck); assert_eq!(r, (0xB5E3u32 as $C) & max);
                let t = bench(reps, || { let _ = &a & max; });
                println!("k = {} (p̂ = 2^{:2}): TFHE-rs FheUint{} refresh (default parameters) {:7.3} ms (min {:7.3})", $k, 2 * $k, 2 * $k, t.0, t.1);
                println!("CSV,tfhers,{},{:.4},{:.4}", $k, t.0, t.1);
            }
        }}; }
        tfr!(FheUint2, u8, 1); tfr!(FheUint4, u8, 2); tfr!(FheUint6, u8, 3); tfr!(FheUint8, u8, 4);
        tfr!(FheUint10, u16, 5); tfr!(FheUint12, u16, 6); tfr!(FheUint14, u16, 7); tfr!(FheUint16, u16, 8);
    }
}

/// Times of the rotation only, the shapes interleaved (same keys size n, same gadget): for each (m, gadget), the keys of
/// all shapes are built, then REPS rounds each time one rotation per shape (minimum per shape); with OWN2=1, also the
/// rotation of eb::EbKeys at N = 2048 (Own2 for m = 2, the CMux of TFHE-rs for m = 1) in the same rounds.
fn timing() {
    use eb_tfhers::params::Config;
    let shapes: Vec<(usize, usize)> = envs("SHAPES", "1:2048,2:1024,4:512,8:256,16:128").split(',')
        .map(|s| { let v: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    let gads: Vec<(usize, usize)> = envs("GADS", "23:1,14:2,11:3,9:4,8:5").split(',')
        .map(|g| { let v: Vec<usize> = g.split(':').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).collect();
    let ms: Vec<usize> = envs("MS", "1,2").split(',').map(|x| x.parse().unwrap()).collect();
    let n = envu("NN", 700); let reps = envu("REPS", 15); let own2 = envu("OWN2", 1) == 1;
    println!("GLWE error bootstrap: time of the rotation (µs per coordinate, minimum of {reps} interleaved rounds), n = {n}");
    for &m in &ms { for &(b, l) in &gads {
        let keys: Vec<GKeys> = shapes.iter().map(|&(kk, np)| GKeys::new(kk, np, n, m, b, l, 1, 1e9, &[], 5 + kk as u64).0).collect();
        let cfg = Config { k: 1, n, b, l, vterm: 1e9, chain: &[] };
        let eb = if own2 { Some(EbKeys::new(&cfg, m, 9).0) } else { None };
        let mut rng = StdRng::seed_from_u64(3);
        let rs: Vec<Rounded> = shapes.iter().map(|&(_, np)| {
            let r = 4 * 2 * np as u64;
            Rounded { abar: (0..n).map(|_| rng.gen_range(0..r)).collect(), bbar: rng.gen_range(0..r), cbar: 0, trials: 0, v: 0.0 }
        }).collect();
        let r2 = Rounded { abar: (0..n).map(|_| rng.gen_range(0..4 * 4096u64)).collect(), bbar: 5, cbar: 0, trials: 0, v: 0.0 };
        let mut mins = vec![f64::INFINITY; shapes.len() + 1];
        for _ in 0..reps {
            for (i, k) in keys.iter().enumerate() {
                let t0 = Instant::now(); let acc = k.rotate(&rs[i]); let dt = t0.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(&acc); mins[i] = mins[i].min(dt);
            }
            if let Some(e) = &eb {
                let t0 = Instant::now(); let acc = e.rotate_st(&r2); let dt = t0.elapsed().as_secs_f64() * 1e6;
                std::hint::black_box(&acc); mins[shapes.len()] = mins[shapes.len()].min(dt);
            }
        }
        for (i, &(kk, np)) in shapes.iter().enumerate() {
            println!("m = {m}, (2^{b:2},{l}), kk = {kk:2}, np = {np:4}: {:8.2} µs/coord  ({:.3} of kk = 1)", mins[i] / n as f64, mins[i] / mins[0]);
            println!("CSV,time,{kk},{np},{m},{b},{l},{n},{:.4}", mins[i] / n as f64);
        }
        if eb.is_some() {
            println!("m = {m}, (2^{b:2},{l}), eb::EbKeys N = 2048 ({}): {:8.2} µs/coord", if m == 2 { "Own2" } else { "CMux" }, mins[shapes.len()] / n as f64);
            println!("CSV,time,eb,2048,{m},{b},{l},{n},{:.4}", mins[shapes.len()] / n as f64);
        }
    } }
}

/// Times of whole refreshes, interleaved, for each k: the error bootstrap at N = 2048 (CONFIGS_RS2), the GLWE
/// configurations of GCONFIGS (shapes KKS), CombBoot per ciphertext (COMB_CONFIGS at 2k bits, if any) and the refresh of
/// FheUint{2k} by TFHE-rs (default parameters, one thread): REPS rounds, each running every operation once; minimum and
/// median per operation, and ratios to TFHE-rs (minimum).
fn ftime() {
    use tfhe::prelude::*;
    use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint10, FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8};
    let reps = envu("REPS", 15);
    let precs: Vec<u32> = envs("KS", "1,2,3,4,5,6,7,8").split(',').map(|x| x.parse().unwrap()).collect();
    let shapes: Vec<usize> = envs("KKS", "1,2,4,8").split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    let (ck, sk) = generate_keys(ConfigBuilder::default().build());
    set_server_key(sk);
    println!("Whole refreshes, one thread, interleaved ({reps} rounds): ms, minimum (median); ratio = TFHE-rs / operation (minima)");
    for &prec in &precs {
        let mut ops: Vec<(String, Box<dyn FnMut()>)> = Vec::new();
        let enc = Gaussian::from_dispersion_parameter(StandardDev(2f64.powf(log2_sigma128(N))), 0.0);
        // TFHE-rs
        macro_rules! tfo { ($T:ty, $C:ty, $k:expr) => {{
            if prec == $k {
                let max: $C = ((1u32 << (2 * $k)) - 1) as $C; let a = <$T>::encrypt((0xB5E3u32 as $C) & max, &ck);
                let r: $C = (&a & max).decrypt(&ck); assert_eq!(r, (0xB5E3u32 as $C) & max);
                ops.push((format!("TFHE-rs FheUint{}", 2 * $k), Box::new(move || { let _ = &a & max; })));
            }
        }}; }
        tfo!(FheUint2, u8, 1); tfo!(FheUint4, u8, 2); tfo!(FheUint6, u8, 3); tfo!(FheUint8, u8, 4);
        tfo!(FheUint10, u16, 5); tfo!(FheUint12, u16, 6); tfo!(FheUint14, u16, 7); tfo!(FheUint16, u16, 8);
        // error bootstrap at N = 2048
        {
            let (cfg, mm) = config_rs2(prec);
            let (keys, mut egen) = EbKeys::new(cfg, mm, 1000 + prec as u64);
            let mut rng = StdRng::seed_from_u64(prec as u64); let mut rbit = 0x9e3779b97f4a7c15u64;
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(1u64 << (64 - 2 * prec)), enc, cm(), &mut egen);
            ops.push((format!("EB N = 2048, m = {mm}, n = {}, (2^{},{})", cfg.n, cfg.b, cfg.l),
                      Box::new(move || { let _ = keys.refresh(&ct, &mut rng, &mut rbit, None); })));
        }
        // GLWE
        for c in GCONFIGS.iter().filter(|c| c.prec == prec && shapes.contains(&c.kk)) {
            let (keys, mut egen) = GKeys::new(c.kk, c.np, c.n, c.m, c.b, c.l, c.prec, c.vterm, c.chain, 2000 + prec as u64 * 10 + c.kk as u64);
            let mut rng = StdRng::seed_from_u64(77 + prec as u64 + c.kk as u64); let mut rbit = 0x2545f4914f6cdd1du64;
            let mu = 1u64 << (64 - 2 * prec);
            let ct: LweCiphertextOwned<u64> = allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk, Plaintext(mu), enc, cm(), &mut egen);
            let out = keys.refresh(&ct, &mut rng, &mut rbit);
            assert!(tor(decrypt_lwe_ciphertext(&keys.big_sk, &out).0.wrapping_sub(mu)).abs() < 1.0 / (4.0 * 4f64.powi(prec as i32)));
            ops.push((format!("GLWE kk = {}, np = {}, p = {}, m = {}, n = {}, (2^{},{})", c.kk, c.np, 2 * c.np, c.m, c.n, c.b, c.l),
                      Box::new(move || { let _ = keys.refresh(&ct, &mut rng, &mut rbit); })));
        }
        // CombBoot (per ciphertext)
        let mut bcs = Vec::new();
        for cfg in COMB_CONFIGS.iter().filter(|c| c.bits == 2 * prec && envu("COMB", 1) == 1) {
            let keys = CombKeys::new(cfg, 900 + cfg.bits as u64);
            let mut rng = StdRng::seed_from_u64(cfg.bits as u64); let mut rbit = 0x853c49e6748fea9bu64;
            let mut egen = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(new_seeder().as_mut().seed(), new_seeder().as_mut());
            let bc = keys.bcols; bcs.push((ops.len(), bc));
            let cts: Vec<LweCiphertextOwned<u64>> = (0..bc).map(|i| allocate_and_encrypt_new_lwe_ciphertext(&keys.big_sk,
                Plaintext(((i as u64 * 7 + 1) % (1u64 << cfg.bits)) << (64 - cfg.bits)), enc, cm(), &mut egen)).collect();
            ops.push((format!("CombBoot B = {bc}, p = {}, m = {}, n = {} (per ciphertext)", cfg.p, keys.m, cfg.n),
                      Box::new(move || { let _ = keys.refresh(&cts, &mut rng, &mut rbit); })));
        }
        for (_, op) in ops.iter_mut() { op(); }
        let mut ts: Vec<Vec<f64>> = vec![Vec::new(); ops.len()];
        for _ in 0..reps {
            for (i, (_, op)) in ops.iter_mut().enumerate() { let t = Instant::now(); op(); ts[i].push(t.elapsed().as_secs_f64() * 1e3); }
        }
        for (i, bc) in &bcs { for x in ts[*i].iter_mut() { *x /= *bc as f64; } }
        let mm: Vec<(f64, f64)> = ts.iter_mut().map(|v| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[0], v[v.len() / 2]) }).collect();
        let tf = mm[0].0;
        println!("k = {prec} ({:2} bits):", 2 * prec);
        for (i, (name, _)) in ops.iter().enumerate() {
            println!("   {:<62} {:8.3} ({:8.3})  ratio {:5.2}", name, mm[i].0, mm[i].1, tf / mm[i].0);
            println!("CSV,ftime,{prec},{},{:.4},{:.4}", name.replace(',', ";"), mm[i].0, mm[i].1);
        }
    }
}

fn main() {
    match envs("MODE", "cal").as_str() { "full" => full(), "time" => timing(), "ftime" => ftime(), _ => cal() }
}
