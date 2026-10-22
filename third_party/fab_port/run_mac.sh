#!/bin/sh
# Build and run the amortized bootstrapping of Guimarães and Pereira (portable build, one thread, binary keys) on this
# machine, one process per parameter set (sizes in increasing order); results in ../../results/gp25/.
# Each run: key generation, then 3 bootstraps of a packed RLWE ciphertext of in_N messages back to the same key
# (blind rotations, extraction, packing key switch, key switch to the sparse input key), mean time, then "Pass".
# SETS="SET_2_3_2048 ..." to choose; all binary sets take about an hour.
set -e
cd "$(dirname "$0")"
mkdir -p ../../results/gp25
OUT=../../results/gp25/gp25_mac.txt
: > $OUT
for s in ${SETS:-SET_2_3_2048 SET_4_5_2048 SET_2_3_4096 SET_4_5_4096 SET_6_7_4096 SET_8_9_4096 SET_2_3_8192 SET_4_5_8192 SET_6_7_8192 SET_8_9_8192}; do
  make -s CC=clang PARAM=$s
  echo "== $s" | tee -a $OUT
  ./sab_BINARY_$s | tee -a $OUT
done
