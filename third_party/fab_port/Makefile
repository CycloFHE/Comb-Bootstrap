# Guimarães and Pereira, "Fast amortized bootstrapping with small keys and polynomial noise overhead" (CCS 2025;
# code: github.com/antoniocgj/Fast-Amortized-Bootstrapping, in ../Fast-Amortized-Bootstrapping), portable build of
# its MOSFHET (pure-C FFT FFNT, SHAKE; no AVX-512/VAES/AES-NI), sources unchanged. COMPRESS=1 keeps the packing keys
# compressed (their default, expanded on the fly by VAES; here by SHAKE), otherwise they are stored in full.
#   make PARAM=SET_8_9_4096 [KEY=BINARY|TERNARY|ARBITRARY] [CC=clang]  ->  ./sab_<KEY>_<PARAM>
FAB = ../Fast-Amortized-Bootstrapping
M = $(FAB)/src/mosfhet/src
SRC_MOSFHET = $(M)/keyswitch.c $(M)/bootstrap.c $(M)/bootstrap_ga.c $(M)/tlwe.c $(M)/trlwe.c $(M)/trgsw.c $(M)/misc.c \
  $(M)/polynomial.c $(M)/register.c $(M)/sha3/fips202.c $(M)/fft/karatsuba.c $(M)/fft/ffnt/ffnt.c
KEY ?= BINARY
PARAM ?= SET_2_3_2048
CC ?= cc
ARCHFLAGS ?=
ifdef COMPRESS
  SRC_MOSFHET += $(M)/trlwe_compressed.c
  EXTRA = -DUSE_COMPRESSED_TRLWE
endif
CFLAGS = -O3 -funroll-loops $(ARCHFLAGS) -DNDEBUG -DPORTABLE_BUILD -DUSE_SHAKE $(EXTRA) -D$(KEY) -D$(PARAM) \
  -I$(FAB)/include -I$(FAB)/src/mosfhet/include -Wno-unused-function -Wno-unused-result -Wno-format -Wno-gnu-folding-constant
sab_$(KEY)_$(PARAM): $(FAB)/main.c $(FAB)/src/sparse_amortized_bootstrap.c $(SRC_MOSFHET)
	$(CC) -o $@ $^ $(CFLAGS) -lm
clean:
	rm -f sab_*
