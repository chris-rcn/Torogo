#!/bin/sh
# run-ppat-parallel.sh — launch K train_ppat workers that parameter-average θ,
# plus one dedicated monitor that tests the averaged checkpoint without stalling
# the trainers.
#
#   ./run-ppat-parallel.sh <K> <data> [extra train_ppat args...]
#
# The K trainers run silently in the background; every --sync-every positions they
# file-barrier all-reduce (mean) their weights, and worker 0 writes the averaged
# checkpoint.  The foreground process is the monitor: it repeatedly loads that
# checkpoint, runs the test, and prints the live |ΔV| table.  Because testing is
# off the training critical path, the trainers never wait on it.
#
# Example (10 trainers, sync every 2000 positions):
#   ./run-ppat-parallel.sh 10 out/evals-s9-500k.txt --M 100 --N 100 --lr 5 \
#       --train-pos 24000 --test-pos 2000 --sync-every 2000
set -e
# Locate the binary by absolute path but keep the caller's cwd, so relative
# data/output paths (e.g. out/evals-s9-500k.txt, out/ppat-data-*.js) resolve.
BIN="$(cd "$(dirname "$0")" && pwd)/train_ppat.bin"

K="$1"; DATA="$2"; shift 2
RUN="$$"
SYNC_DIR="out/ppat-sync-$RUN"
CKPT="out/ppat-data-$RUN.js"
BEST="${CKPT%.js}-best.js"

# Pull --train-pos out of the passed args, for reporting.
TRAIN_POS=""; prev=""
for a in "$@"; do
    [ "$prev" = "--train-pos" ] && { TRAIN_POS="$a"; break; }
    prev="$a"
done
if [ -n "$TRAIN_POS" ]; then
    TP_MSG="train-pos/worker: $TRAIN_POS"
else
    TP_MSG="train-pos/worker: (train_ppat default)"
fi

# Fresh sync dir (avoid stale round files from a previous run).
rm -rf "$SYNC_DIR"; mkdir -p "$SYNC_DIR"

echo "launching $K trainers + 1 monitor;  $TP_MSG"
echo "sync-dir: $SYNC_DIR  checkpoint: $CKPT  best: $BEST"
PIDS=""
i=0
while [ "$i" -lt "$K" ]; do
    "$BIN" "$DATA" --workers "$K" --worker-id "$i" --sync-dir "$SYNC_DIR" --save "$CKPT" "$@" >/dev/null 2>&1 &
    PIDS="$PIDS $!"
    i=$((i + 1))
done

# Kill all trainers if the monitor exits (Ctrl-C, etc.); leave the checkpoint.
trap 'kill $PIDS 2>/dev/null; rm -rf "$SYNC_DIR"' INT TERM EXIT

# Monitor in the foreground: tests the averaged checkpoint and prints the table.
# It gets the test args from "$@" (test-pos/test-playouts/test-moves/phases) and
# --workers so it can report aggregate positions; it ignores the training args.
"$BIN" "$DATA" --workers "$K" --monitor "$CKPT" "$@"
