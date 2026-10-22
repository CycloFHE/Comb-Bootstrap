# BatchBoot (Li et al., USENIX Security 2026; code: doi.org/10.5281/zenodo.17936945, unzipped in ../batchBoot),
# portable build of MOSFHET (pure-C FFT FFNT, SHAKE; no AVX-512/VAES/AES-NI) for Apple silicon or any CPU.
# The sources of ../batchBoot are used unchanged; bench_main.c only runs the parameter sets given on the command line.
#   make            (CC=clang on macOS; ARCHFLAGS=-march=native on x86)
#   ./batchBoot_port 2 4 6 8
BB = ../batchBoot
M = $(BB)/third-party/MOSFHET/src
SRC_MOSFHET = $(M)/keyswitch.c $(M)/bootstrap.c $(M)/bootstrap_ga.c $(M)/tlwe.c $(M)/trlwe.c $(M)/trgsw.c $(M)/misc.c \
  $(M)/polynomial.c $(M)/register.c $(M)/sha3/fips202.c $(M)/fft/karatsuba.c $(M)/trlwe_compressed.c $(M)/fft/ffnt/ffnt.c
CC ?= cc
ARCHFLAGS ?=
CFLAGS = -O3 -funroll-loops $(ARCHFLAGS) -DNDEBUG -DPORTABLE_BUILD -DUSE_SHAKE -DUSE_COMPRESSED_TRLWE \
  -I$(BB)/include -I$(BB)/third-party/MOSFHET/include -Wno-unused-function -Wno-unused-result
batchBoot_port: bench_main.c $(wildcard $(BB)/src/*.c) $(SRC_MOSFHET)
	$(CC) -o $@ $^ $(CFLAGS) -lm
clean:
	rm -f batchBoot_port
