#!/bin/sh
# Build and run BatchBoot (portable build, one thread) on this machine; results in ../../results/batchboot/.
# One process per parameter set, as in test/batchBoot_main.c: the secret keys of BatchBoot are allocated without
# being zeroed (trlwe_alloc_key) and only their ones are set, so in a process that already freed memory the key may
# contain garbage and every coefficient fails (observed for Boot4 run after Boot2 in the same process).
set -e
cd "$(dirname "$0")"
make CC=clang
mkdir -p ../../results/batchboot
OUT=../../results/batchboot/batchboot_mac.txt
: > $OUT
for c in ${SETS:-2 4 6 8}; do ./batchBoot_port $c | tee -a $OUT; done
