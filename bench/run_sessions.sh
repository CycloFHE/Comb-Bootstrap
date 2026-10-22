#!/bin/bash
# Timing sessions of the paper: NSESS sessions of the one-thread benchmarks of Section 7, so that the times and the ratios can be averaged over sessions.
# Each session runs, in an order rotated from one session to the next:
#   eb_curves       error bootstrap and TFHE-rs refresh, k = 1..8 (Figure 4, Figure 1, Section 7)
#   comb_curves     CombBoot, error bootstrap and TFHE-rs, 2..16 bits (Figure 4, Figure 1, Appendix)
#   product_curves  product of the error bootstrap (k <= 4) and of TFHE-rs (k = 1..8), times only (NPROD=0)
#   tfhe_curves     TFHE-rs refresh, addition and product (tfhe-rs-bench, Figure 4)
#   rns_small       residues of 16 and 32 bits against the radix and CRT integers of TFHE-rs (Table 1)
#   glwe_ftime      separate rotations in GLWE of rank 2..8 (paragraph of Section 5; GLWE=0 to skip)
#
# Run from the root of the repository (Terminal, Mac on AC power, other applications closed):
#   BUILD_ONLY=1 bash bench/run_sessions.sh         # 1. build (may need the network), then cut the network
#   QUICK=1 bash bench/run_sessions.sh 1            # 2. smoke test (a few minutes, results/sessions/quick_*)
#   bash bench/run_sessions.sh 10                   # 3. ten sessions (the duration of each is in run.log)
# Interrupted? Run it again with the same OUT: the programs already done (.ok) are skipped.
#
# Arguments: NSESS (10), OUT (results/sessions/YYYYMMDD_HHMM).
# Environment: COOL (60: seconds of rest before each program), COOL_SESS (180: rest between sessions),
#   LOADMAX (1.5: wait until the load average over one minute is below it, at most WAITMAX = 900 s),
#   GLWE (1), MREPS (3: repetitions of the multiplications of TFHE-rs in
#   rns_small), QUICK (0), DRY (0: print the commands only), BUILD_ONLY (0), NOBUILD (0), FORCE (0: run even if
#   cargo, rustc, julia or eb_tail are running), EBDIR, TFDIR (the two crates).
set -u
export LC_ALL=C          # decimal points in the load averages read by awk
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NSESS=${1:-10}
QUICK=${QUICK:-0}
if [ "$QUICK" = 1 ]; then DEF_OUT="$ROOT/results/sessions/quick_$(date +%Y%m%d_%H%M)"; else DEF_OUT="$ROOT/results/sessions/$(date +%Y%m%d_%H%M)"; fi
OUT=${2:-$DEF_OUT}
EBDIR=${EBDIR:-$ROOT/rust/eb-tfhers}
TFDIR=${TFDIR:-$ROOT/tfhe-rs-bench}
EBB="$EBDIR/target/release"
TFB="$TFDIR/target/release"
GLWE=${GLWE:-1}; MREPS=${MREPS:-3}
DRY=${DRY:-0}; BUILD_ONLY=${BUILD_ONLY:-0}; NOBUILD=${NOBUILD:-0}; FORCE=${FORCE:-0}
LOADMAX=${LOADMAX:-1.5}; WAITMAX=${WAITMAX:-900}
if [ "$QUICK" = 1 ]; then
  R5=2; R10=2; R15=2; KS8=1,4; PKS=1; BITS=2,8; VARS=16; KKS=4; MREPS=1
  COOL=${COOL:-5}; COOL_SESS=${COOL_SESS:-5}
else
  R5=5; R10=10; R15=15; KS8=1,2,3,4,5,6,7,8; PKS=1,2,3,4; BITS=2,4,8,12,16; VARS=32,16,32w,16w; KKS=1,2,4,8
  COOL=${COOL:-60}; COOL_SESS=${COOL_SESS:-180}
fi
EB_BINS="eb_curves comb_curves product_curves rns_small_arith eb_glwe"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }
md5f() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
load1() { if [ -r /proc/loadavg ]; then cut -d' ' -f1 /proc/loadavg; else sysctl -n vm.loadavg | awk '{print $2}'; fi; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }
therm() { if command -v pmset >/dev/null 2>&1; then pmset -g therm 2>/dev/null | grep -v '^Note' | tr '\n' ' ' | tr -s ' '; else echo "n/a"; fi; }
power() { if command -v pmset >/dev/null 2>&1; then pmset -g batt 2>/dev/null | head -1 | sed "s/Now drawing from //"; else echo "n/a"; fi; }

build() {
  echo "Building the benchmarks (release)..."
  (cd "$EBDIR" && cargo build --release $(for b in $EB_BINS; do printf ' --bin %s' "$b"; done)) || { echo "build of eb-tfhers failed"; exit 1; }
  (cd "$TFDIR" && cargo build --release --bin curves) || { echo "build of tfhe-rs-bench failed"; exit 1; }
}

sysinfo() {
  echo "date: $(ts)"
  echo "uname: $(uname -a)"
  if command -v sw_vers >/dev/null 2>&1; then sw_vers | tr '\n' ' '; echo; fi
  echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo)"
  echo "cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc); performance $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo ?), efficiency $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo ?)"
  echo "power: $(power)"
  if command -v pmset >/dev/null 2>&1; then echo "low power mode: $(pmset -g 2>/dev/null | awk '/lowpowermode|powermode/ {print $1"="$2}' | tr '\n' ' ')"; fi
  echo "thermal: $(therm)"
  echo "uptime: $(uptime)"
  echo "--- processes using the most CPU"
  if [ "$(uname)" = Darwin ]; then top -l 2 -s 1 -o cpu -n 12 -stats pid,command,cpu,mem 2>/dev/null | awk '/^PID/ {n++} n == 2'; else ps -eo pid,comm,pcpu,rss --sort=-pcpu | head -13; fi
}

# the command of each program (also written to commands.txt)
cmd_of() {
  case $1 in
    eb_curves)      echo "env RAYON_NUM_THREADS=1 REPS=$R10 KS=$KS8 $EBB/eb_curves" ;;
    comb_curves)    echo "env RAYON_NUM_THREADS=1 REPS=$R5 BITS=$BITS $EBB/comb_curves" ;;
    product_curves) echo "env RAYON_NUM_THREADS=1 REPS=$R10 NPROD=0 KS=$PKS TKS=$KS8 $EBB/product_curves" ;;
    tfhe_curves)    echo "env RAYON_NUM_THREADS=1 THREADS=1 REPS=$R10 $TFB/curves" ;;
    rns_small)      echo "env RAYON_NUM_THREADS=1 VARIANTS=$VARS TF=1 RADIX=1 NE=0 NPROD=0 CHECK=1 REPS=$R5 MREPS=$MREPS $EBB/rns_small_arith" ;;
    glwe_ftime)     echo "env RAYON_NUM_THREADS=1 MODE=ftime KS=$KS8 KKS=$KKS REPS=$R15 $EBB/eb_glwe" ;;
  esac
}

wait_load() {
  local w=0 l
  l=$(load1)
  while ! lt "$l" "$LOADMAX" && [ $w -lt "$WAITMAX" ]; do sleep 15; w=$((w + 15)); l=$(load1); done
  if lt "$l" "$LOADMAX"; then
    [ $w -gt 0 ] && log "    waited ${w} s for the load average to drop below $LOADMAX (now $l)"
  else
    log "    WARNING: load average still $l after ${w} s (something else is running)"
  fi
}

# ------------------------------------------------------------------------------------------------------------------
if [ "$BUILD_ONLY" = 1 ]; then build; echo "Built. You can now cut the network and run: bash bench/run_sessions.sh $NSESS"; exit 0; fi
mkdir -p "$OUT"
if [ "$DRY" != 1 ] && [ "$FORCE" != 1 ]; then
  others=$(pgrep -lf 'cargo|rustc|julia|eb_tail|eb_noise' 2>/dev/null | grep -v "$$" | grep -v pgrep || true)
  if [ -n "$others" ]; then echo "Other computations are running (FORCE=1 to ignore):"; echo "$others"; exit 1; fi
fi
[ "$NOBUILD" = 1 ] || [ "$DRY" = 1 ] || build
for b in $EB_BINS; do [ -x "$EBB/$b" ] || [ "$DRY" = 1 ] || { echo "missing $EBB/$b"; exit 1; }; done
[ -x "$TFB/curves" ] || [ "$DRY" = 1 ] || { echo "missing $TFB/curves"; exit 1; }
if command -v caffeinate >/dev/null 2>&1 && [ "$DRY" != 1 ]; then caffeinate -ims -w $$ & fi

TASKS="eb_curves comb_curves product_curves tfhe_curves rns_small"
[ "$GLWE" = 1 ] && TASKS="$TASKS glwe_ftime"
T=($TASKS); NT=${#T[@]}
{
  echo "sessions: $NSESS, out: $OUT, QUICK=$QUICK COOL=$COOL COOL_SESS=$COOL_SESS LOADMAX=$LOADMAX GLWE=$GLWE MREPS=$MREPS"
  echo "order: $TASKS rotated by one program per session"
  echo "rustc: $(rustc --version 2>/dev/null)"
  for t in $TASKS; do echo "$t: $(cmd_of "$t")"; done
  if [ "$DRY" != 1 ]; then
    for b in $EB_BINS; do echo "md5 $b $(md5f "$EBB/$b")"; done
    echo "md5 tfhe-rs-bench/curves $(md5f "$TFB/curves")"
  fi
} > "$OUT/commands.txt"
log "start: $NSESS sessions into $OUT (power: $(power))"
[ "$(uname)" = Darwin ] && ! power | grep -q "AC Power" && log "WARNING: not on AC power"

t_start=$(date +%s)
for s in $(seq 1 "$NSESS"); do
  sd=$(printf "%s/s%02d" "$OUT" "$s"); mkdir -p "$sd"
  [ "$DRY" = 1 ] || sysinfo > "$sd/sysinfo.txt" 2>&1
  order=""
  for j in $(seq 0 $((NT - 1))); do order="$order ${T[$(( (j + s - 1) % NT ))]}"; done
  log "session $s/$NSESS:$order"
  ts_sess=$(date +%s)
  for t in $order; do
    if [ -f "$sd/$t.ok" ]; then log "  $t: already done, skipped"; continue; fi
    c=$(cmd_of "$t")
    if [ "$DRY" = 1 ]; then echo "  [dry] $c > $sd/$t.txt"; continue; fi
    sleep "$COOL"; wait_load
    log "  $t: start (load $(load1); thermal: $(therm))"
    t0=$(date +%s)
    $c > "$sd/$t.txt" 2>&1; rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
      echo "$dt" > "$sd/$t.ok"; log "  $t: done in ${dt} s"
    else
      log "  $t: FAILED (exit $rc after ${dt} s), see $sd/$t.txt"
    fi
  done
  if [ "$DRY" != 1 ]; then
    el=$(( $(date +%s) - ts_sess )); tot=$(( $(date +%s) - t_start ))
    left=$(( tot * (NSESS - s) / s ))
    log "session $s done in $((el / 60)) min; about $((left / 60)) min left (thermal: $(therm))"
    [ "$s" -lt "$NSESS" ] && sleep "$COOL_SESS"
  fi
done
log "all sessions done: $OUT"
