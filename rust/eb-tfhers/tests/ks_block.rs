use eb_tfhers::kern::{ks_block, ks_fused};
fn rnd(s: &mut u64, n: usize) -> Vec<f64> { (0..n).map(|_| { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 }).collect() }
#[test]
fn ks_block_matches_fused() {
    let mut s = 0x9e3779b97f4a7c15u64;
    for &gl in &[1usize, 2, 3, 4, 5, 6] { for &n in &[16usize, 64, 128] {
        let q = (rnd(&mut s, n), rnd(&mut s, n)); let f = (rnd(&mut s, n), rnd(&mut s, n));
        let dv: Vec<[Vec<f64>; 2]> = (0..gl).map(|_| [rnd(&mut s, n), rnd(&mut s, n)]).collect();
        let kv: Vec<[Vec<f64>; 4]> = (0..gl).map(|_| [rnd(&mut s, n), rnd(&mut s, n), rnd(&mut s, n), rnd(&mut s, n)]).collect();
        let d: Vec<[&[f64]; 2]> = dv.iter().map(|a| [&a[0][..], &a[1][..]]).collect();
        let k: Vec<[&[f64]; 4]> = kv.iter().map(|a| [&a[0][..], &a[1][..], &a[2][..], &a[3][..]]).collect();
        let x0 = (rnd(&mut s, n), rnd(&mut s, n), rnd(&mut s, n), rnd(&mut s, n));
        let (mut a, mut b) = (x0.clone(), x0.clone());
        ks_fused(&mut a.0, &mut a.1, &mut a.2, &mut a.3, &q.0, &q.1, &f.0, &f.1, &d, &k);
        ks_block(&mut b.0, &mut b.1, &mut b.2, &mut b.3, &q.0, &q.1, &f.0, &f.1, &d, &k);
        for (u, v) in [(&a.0, &b.0), (&a.1, &b.1), (&a.2, &b.2), (&a.3, &b.3)] {
            assert!(u.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits()), "gl={gl} n={n}");
        }
    }}
}
