#!/bin/sh
# Retrain a truncation vpat model each time its eval set has grown, and score
# each retrained model on move choice.
#
# The eval set is the concatenation of its chunk files (<EVALS>-NN, written by
# gen-agent-evals runs still in progress).  Whenever the concatenation has grown
# by more than 1/8 since the last round: hash-split it 8:1 into -train / -test,
# train MODEL from scratch on it, then compare puct-trunc with its best-teMSE
# checkpoint MODEL_BEST (p2) against default puct-trunc (p1) on MD_FILE.  Each round prints one line: the record
# count (thousands), the hash-split summary, the trainer's last best (*) row,
# and the compare-move-selection summary.  Runs until killed.
#
# Example eval generation (one chunk per process; run several in parallel):
#   p=2000; PLAYOUTS=$p node gen-agent-evals.js --position-agent ref-ply8 --value-agent puct-ppat-fp --min-phase 0.00 --max-phase 0.35 --prefix-delta 0.2 --komi 0.5 >> out/eval-s13-rp8-00ph35-d20-ppf$p-$(printf '%02d\n' $((RANDOM % 100))) &
# The value agent's labels must share one currency across the target phase
# range: it must either always truncate or never truncate there (puct-ppat-fp
# never does; a puct-trunc labeller whose TRUNC_ROOT_PHASE falls inside the
# range mixes truncated and full-playout labels).
#
# usage: train-trunc-model.sh

LABEL_PLAYOUTS=2000                       # playouts of the labelling search (ppf<N> in the file names)
SPEC=tp9,2:4,3:4
MODEL=out/vpat-trunc00ph35d20-$LABEL_PLAYOUTS.js
MODEL_BEST=${MODEL%.js}-best.js            # the trainer's best-teMSE checkpoint (the '*' row)
EVALS=out/eval-s13-rp8-00ph35-d20-ppf$LABEL_PLAYOUTS
EPOCHS=30
TEST_NTH=9                                # hash-split: every 9th record goes to the test set
MD_FILE=out/md-trunc30k_all-1976.md
MAX_PHASE=0.35
BUDGET_MS=2000
P2_ROOT_PHASE=0.35
POLL_SECONDS=9

echo "$MODEL"
last_count=1
while :; do
  # Wait until the concatenated chunks hold more than 9/8 of last round's records.
  while :; do
    cat "$EVALS"-?? > "$EVALS"
    count=$(wc -l < "$EVALS")
    [ "$count" -gt "$((last_count * 9 / 8))" ] && break
    sleep "$POLL_SECONDS"
  done

  printf "%4sk:  Train: " $((count / 1000))
  ./hash-split.py "$EVALS" "$TEST_NTH" 2>/dev/null

  best_row=$(node train-vpat-supervised.js --evals "$EVALS-train" --test-evals "$EVALS-test" \
               --epochs "$EPOCHS" --spec "$SPEC" --save "$MODEL" | grep '\*' | tail -1)
  printf "%s  Move: " "$best_row"

  P2_TRUNC_VPAT_DATA="$MODEL_BEST" P2_TRUNC_ROOT_PHASE="$P2_ROOT_PHASE" \
    node compare-move-selection.js --md "$MD_FILE" --p1 puct-trunc --p2 puct-trunc \
      --max-phase "$MAX_PHASE" --budget "$BUDGET_MS" 2>&1 | tail -1

  last_count=$count
done
