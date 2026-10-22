# Exact Error Bootstrapping and Comb Bootstrapping in FHEW/TFHE — Rust/Julia code and measurements

Philippe Chartier (Inria, IRMAR, Université de Rennes) and Mohammed Lemou (CNRS, IRMAR, Université de Rennes).

This folder contains the code (Rust, on TFHE-rs, and Julia), the benchmarks and the raw logs behind the measured
numbers of the paper. Our code is under the MIT license (`LICENSE`). TFHE-rs 1.8.1 is a dependency fetched by cargo, under its own
license. The code of BatchBoot and of Guimarães and Pereira is not included: `third_party/` holds only our build and
run scripts for it (`third_party/README.md` says where to get it).

All the times of the paper were measured on one laptop (Apple M3 Max, macOS), one thread unless stated otherwise.
Times depend on the machine; the ratios of the paper compare programs measured in the same session. The noise does
not depend on the machine.

## Contents

| Folder | Contents |
|---|---|
| `rust/eb-tfhers/` | the error bootstrap, CombBoot and the product of two ciphertexts built on TFHE-rs 1.8.1 (its FFT, CMux, blind rotation and keys; our stair key switch, rounding with recentering and selection, multi-bit rotation and automorphism key switchings), and the benchmarks against TFHE-rs (`src/bin/`) |
| `tfhe-rs-bench/` | TFHE-rs 1.8.1 alone: refresh, addition and product of `FheUint2k`, `k = 1..8` |
| `julia/` | the second implementation (Appendix J): all the variants (`m ≤ 4`, giant steps, case (A)), an FFT backend for the times and an exact backend for the noise; tests in `julia/test/`, benchmarks in `julia/benchmarks/` |
| `bench/` | the protocol of the timing sessions (`run_sessions.sh`) |
| `third_party/` | our portable builds and run scripts for BatchBoot and for Fast-Amortized-Bootstrapping (Guimarães–Pereira) (`batchBoot_port/`, `fab_port/`); their code is not included |
| `results/` | the raw logs used in the paper: `*_mac.txt` measured on the laptop; `*_vm.txt` and `*_cloud*.txt` noise measurements or indicative runs on Linux virtual machines (never used for a time of the paper) |

## Requirements

- Rust (stable) and cargo; TFHE-rs 1.8.1 is fetched by cargo (feature `integer`).
- Julia ≥ 1.10 (measurements: 1.12) with FFTW.jl, for Appendix J: `cd julia && julia --project -e 'using Pkg; Pkg.instantiate()'`.

## Quick check

```
cd rust/eb-tfhers
cargo build --release
RAYON_NUM_THREADS=1 KS=1,4 REPS=3 cargo run --release --bin eb_curves      # error bootstrap and TFHE-rs, 2 and 8 bits
RAYON_NUM_THREADS=1 BITS=8 REPS=2 cargo run --release --bin comb_curves    # CombBoot at 8 bits
cd ../../julia && julia --project test/runtests.jl                          # Julia: exact backend
```

## From the paper to the commands

Rust benchmarks run from `rust/eb-tfhers/` (`RAYON_NUM_THREADS=1 cargo run --release --bin <name>`), TFHE-rs alone from
`tfhe-rs-bench/`, Julia benchmarks from `julia/` (`julia --project -t 1 benchmarks/<script>`). Each program writes its log on stdout; the logs of the paper are in `results/`. The times of Section 7, Table 1 and
Figures 1 and 4 are the means of 10 sessions (`results/sessions/20261018_1717`, see Timing sessions below): the logs named in
the table are single sessions with the same programs, and `results/sessions/20261018_1717/mean/` holds the means in the same formats.

| Paper | Command | Log |
|---|---|---|
| Section 7, table at 8 bits; refresh of the error bootstrap and of TFHE-rs with our key switch (Figure 4, refresh) | `eb_curves` | `results/rust/eb_curves_rs2_mac2.txt` |
| Section 7 and Figure 4, CombBoot (16 ciphertexts per rotation up to 8 bits, 8 beyond) | `comb_curves` | `results/rust/comb_curves_mac9.txt` |
| Section 7 and Figure 4, product (refreshes of both operands, `k ≤ 4`) against TFHE-rs `a * b` | `product_curves` | `results/rust/product_curves_rs2_mac.txt` |
| Section 7 and Figure 4, TFHE-rs refresh, addition and product | `cd tfhe-rs-bench && cargo run --release --bin curves` | `results/tfhe-rs/tfhe_curves_mac.txt` |
| Table 1 (residues of 16 and 32 bits against the radix and CRT integers of TFHE-rs) | `VARIANTS=32,16,32w,16w TF=1 RADIX=1 cargo run --release --bin rns_small_arith` | `results/rust/rns_small_mac.txt` |
| Section 7, amortized bootstraps (and Figure 1) | `sh third_party/batchBoot_port/run_mac.sh`, `sh third_party/fab_port/run_mac.sh`; failure probabilities: `scripts/noise_estimator.py` of Guimarães and Pereira | `results/batchboot/batchboot_mac.txt`, `results/gp25/gp25_mac.txt`, `results/gp25/gp25_failure_estimator.txt` |
| Section 7 and Appendix J, 8 threads: latency of one CombBoot (its rotation shared by the threads), and of the error bootstrap and TFHE-rs | `BITS=2,4,8,12,16 THREADS=1,8,10 REPS=10 cargo run --release --bin comb_mt`; `cargo run --release --bin mt_curves` | `results/rust/comb_mt_mac4.txt`, `results/rust/mt_curves_mac2.txt` |
| Section 4.3 and Table 4: tail of the error with `ν = 1` | `KS=1,4 NS=1000000 THREADS=12 cargo run --release --bin eb_tail` (several threads: only the counts matter) | `results/rust/eb_tail_mac.txt` |
| Section 5, separate rotations in smaller rings (GLWE) | `MODE=ftime cargo run --release --bin eb_glwe` (configurations from `MODE=cal` and `MODE=time`) | `results/rust/glwe_ftime_mac.txt` (`glwe_cfg_mac.txt`, `glwe_time_mac.txt`, `glwe_full_mac.txt`) |
| Section 7 and Appendix J, noise on TFHE-rs; calibration of the error of its FFT | `NREF=500 cargo run --release --bin eb_noise` (500 refreshes per configuration), `comb_noise` (1600 outputs per size); `eb_fft` | `results/rust/eb_noise_rs2_mac500.txt`, `results/rust/comb_noise_mac2.txt`; `results/rust/eb_fft_cloud.txt` |
| Section 7 and Appendix J, TFHE-rs with our key switch; our multi-bit rotation against the CMux | `tfhers_ks`; `rot_ab`, `comb_ab` | `results/rust/tfhers_ks_mac.txt`; `results/rust/rot_ab_mac3.txt`, `results/rust/comb_ab_mac.txt` |
| Section 7 and Appendix J, product: exactness of the trace and tensor product on the FFT of TFHE-rs | `product_check` | — |
| Appendix J, Table 3 (CombBoot in Julia, 2 to 16 bits) | `M=2 PART=NET julia --project -t 1 benchmarks/run_comb_sizes.jl` | `results/julia/mb_sizes_m2_mac.txt` |
| Appendix J, validation: end to end, independence of the outputs of one CombBoot | `M=2 PART=E julia --project -t 1 benchmarks/run_comb_sizes.jl`; `M=2 julia --project -t 1 benchmarks/run_comb_keys_corr.jl` | `results/julia/comb_e2e_m2_vm.txt`, `results/julia/comb_keys_corr_m2_vm.txt` |
| Appendix J, times in Julia (CombBoot and separate rotations, `m = 1..3`) | `P=256 julia --project -t 8 benchmarks/run_grouped.jl` (and `P=128`, `P=512`) | `results/julia/mt8_p*.txt`, `results/julia/mt8b_p*.txt` |
| Appendix J, product in Julia (noise and margins) | `PCFG=exact PART=N julia --project -t 1 benchmarks/run_curves.jl` | `results/julia/curves_noise_exact_vm.txt` |
| Case (A), rings of odd prime conductor (Appendix C; Julia only) | `julia --project test/test_caseA.jl`, `julia -t 2 --project benchmarks/run_caseA.jl all` | `results/julia/caseA_mac.txt` |

## Timing sessions

The protocol of the averaged times: `bench/run_sessions.sh` runs, in each session, `eb_curves`, `comb_curves`,
`product_curves`, `tfhe-rs-bench/curves`, `rns_small_arith` and `eb_glwe`, in an order rotated from one session to the
next, with rests between programs and a check of the load of the machine. Every number of the paper is the mean over
the sessions of the minimum over the repetitions within each session; `results/sessions/20261018_1717/` holds the logs
of the sessions, their statistics (`stats.csv`, `summary.txt`) and the means in the formats above (`mean/`).

```
BUILD_ONLY=1 bash bench/run_sessions.sh
bash bench/run_sessions.sh 10
```

## Notes

- **Error of the FFT.** TFHE-rs computes in double precision; its error is modelled as an independent Gaussian term,
  `2^-87.5 B^2 ℓ` per external product and `2^-84.5 B^2 ℓ` per pair of coordinates in our multi-bit form (calibrated by
  `eb_fft`), and the configurations keep a margin of 0.2 bit on the standard deviations.
- **Rounding.** The mask is rounded at the resolution `R = p̂ p` coordinate by coordinate, its mean is recentered,
  and its variance is brought below the threshold `V*` by selection (pairs from a pool of 128 public encryptions of
  zero, about 100 trials).
- **Not implemented:** case (A) on TFHE-rs (Julia only), and the grouped rounding with its transfer for `m ≥ 2`.

## Citation

```
@misc{ChartierLemou2026,
  author       = {Philippe Chartier and Mohammed Lemou},
  title        = {Exact Error Bootstrapping and Comb Bootstrapping in {FHEW/TFHE}},
  howpublished = {IACR Cryptology ePrint Archive},
  year         = {2026},
  note         = {Code: \url{https://github.com/CycloFHE/Comb-Bootstrap}}
}
```
