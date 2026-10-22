# BatchBoot on the same laptop

BatchBoot (Li et al., USENIX Security 2026; code: doi.org/10.5281/zenodo.17936945, in `third_party/batchBoot`),
built by `third_party/batchBoot_port` (MOSFHET with its portable pure-C FFT, FFNT, and SHAKE: the default build
needs AVX-512 and VAES), sources unchanged, one thread, Apple clang. One process per parameter set (`run_mac.sh`):
the secret keys are allocated without being zeroed, so a second set run in the same process may get a garbage key
(`batchboot_mac_oneprocess.txt`: Boot4 and Boot8 fail there).

`batchboot_mac.txt`: time of one batch (clock()) and amortized time; all coefficients decrypt correctly.

| set   | bits | batch | batch time | amortized | failure (paper) | published (Xeon 6258R, AVX-512) |
|-------|------|-------|------------|-----------|-----------------|---------------------------------|
| Boot2 | 2    | 2048  | 5.5 s      | 2.68 ms   | 2^-120          | 3.57 ms                         |
| Boot4 | 4    | 2048  | 5.7 s      | 2.79 ms   | 2^-94           | 3.60 ms                         |
| Boot6 | 6    | 4096  | 31.6 s     | 7.70 ms   | 2^-64           | 8.43 ms                         |
| Boot8 | 8    | 4096  | 70.0 s     | 17.09 ms  | 2^-62           | 25.11 ms                        |

What is timed: a packed RLWE ciphertext (degree n) in, an RLWE ciphertext under another key (degree N) out; neither
the packing of LWE ciphertexts nor the key switch back to the input key. Sparse binary keys (Hamming weight 34 to 42).
