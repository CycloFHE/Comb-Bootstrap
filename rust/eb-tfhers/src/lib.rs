//! Error bootstrap and CombBoot (case (B), N = 2048) built on TFHE-rs 1.8.1.
//!
//! Reused from TFHE-rs: the binary GLWE key, the bootstrapping keys (classic or multi-bit) and their Fourier conversion,
//! the FFT, the blind rotation (`blind_rotate_assign`, `multi_bit_blind_rotate_assign`) fed with our exponents through
//! its public traits, the CMux, the signed decomposition, the sample extraction and the LWE arithmetic. Ours: the
//! polynomial stair key switch (`stair`), the rounding at R = p̂·2N with recentering and selection (`rounding`), the
//! error bootstrap (`eb`), CombBoot (`comb`), the multi-threaded versions (`mt`: spin barrier and parallel external
//! product in the Fourier domain of TFHE-rs, bit-identical to its CMux) and the product of two ciphertexts of Section 6
//! (`product`: trace modulo 2^64 - 1, tensor product, relinearization, with exact integer products from the FFT of
//! TFHE-rs). Configurations: `params` (error bootstrap), `comb::COMB_CONFIGS` and `product::PCONFIGS_RS`.
pub mod params;
pub mod rd;
pub mod stair;
pub mod rounding;
pub mod eb;
pub mod comb;
pub mod mt;
pub mod kern;
pub mod product;
pub mod tfhers_ours;
