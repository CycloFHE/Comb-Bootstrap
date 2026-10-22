# Code of other teams

Only our build and run scripts are here. Their code is not included; put it next to them before `make`:

- `Fast-Amortized-Bootstrapping/`: Guimarães and Pereira, https://github.com/antoniocgj/Fast-Amortized-Bootstrapping,
  commit 8938ab5bd42a80c93ed5ee4caef36efbb01bf7ef (Apache-2.0):
  `git clone https://github.com/antoniocgj/Fast-Amortized-Bootstrapping && git -C Fast-Amortized-Bootstrapping checkout 8938ab5`
- `batchBoot/`: BatchBoot, https://doi.org/10.5281/zenodo.17936945 (CC BY 4.0), unzipped here.

`batchBoot_port/` and `fab_port/` build their MOSFHET with its pure-C FFT (no AVX-512, VAES or AES-NI), sources
unchanged, and run the parameter sets of the paper (`sh run_mac.sh`); the logs go to `../results/batchboot/` and
`../results/gp25/`.
