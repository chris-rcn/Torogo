#!/bin/sh
# Retrain a truncation vpat model each time its eval set has grown, and score
# each retrained model on move choice.
#
# The eval set is the concatenation of its chunk files (<EVALS>-NN, written by
# gen-agent-evals runs still in progress).  Whenever the concatenation has grown
# by more than 1/8 since the last round: hash-split it 8:1 into -train / -test,
# train MODEL from scratch on it, then score puct-trunc with its best-teMSE
# checkpoint MODEL_BEST on MD_FILE (evalmovedetails: the mean gap to each
# position's best move).  Each round prints one line: the record count
# (thousands), the hash-split summary, the trainer's last best (*) row, and the
# evalmovedetails SUMMARY.  Runs until killed.
#
# Example eval generation (one chunk per process; run several in parallel):
#   p=2000; PLAYOUTS=$p node gen-agent-evals.js --position-agent ref-ply8 --value-agent puct-ppat-fp --min-phase 0.00 --max-phase 0.35 --prefix-delta 0.2 --komi 0.5 >> out/eval-s13-rp8-00ph35-d20-ppf$p-$(printf '%02d\n' $((RANDOM % 100))) &
# The value agent's labels must share one currency across the target phase
# range: it must either always truncate or never truncate there (puct-ppat-fp
# never does; a puct-trunc labeller whose TRUNC_ROOT_PHASE falls inside the
# range mixes truncated and full-playout labels).
#
# usage: [NAME=value ...] train-trunc-model.sh   (any setting below can be set from outside)

LABEL_PLAYOUTS=${LABEL_PLAYOUTS:-2000}    # playouts of the labelling search (ppf<N> in the file names)
SPEC=${SPEC:-tp9,2:4,3:4}
MODEL=${MODEL:-out/vpat-trunc00ph35d20-$LABEL_PLAYOUTS.js}
MODEL_BEST=${MODEL_BEST:-${MODEL%.js}-best.js} # the trainer's best-teMSE checkpoint (the '*' row)
EVALS=${EVALS:-out/eval-s13-rp8-00ph35-d20-ppf$LABEL_PLAYOUTS}
EPOCHS=${EPOCHS:-30}
DELTA=${DELTA:-0.2}                       # baked into the model; must match the evals' --prefix-delta
TEST_NTH=${TEST_NTH:-9}                   # hash-split: every 9th record goes to the test set
MD_FILE=${MD_FILE:-out/md-trunc30k_all-1976.md}
MAX_PHASE=${MAX_PHASE:-0.35}
TEST_LIMIT=${TEST_LIMIT:-200}             # score only the first N positions inside the phase window (same set every round)
SEARCH_PLAYOUTS=${SEARCH_PLAYOUTS:-10000} # puct-trunc playouts per move when scoring (fixed, not timed)
ROOT_PHASE=${ROOT_PHASE:-0.35}            # puct-trunc's TRUNC_ROOT_PHASE
POLL_SECONDS=${POLL_SECONDS:-9}

echo "train: evals: $EVALS  test-nth: $TEST_NTH  spec: $SPEC  delta: $DELTA  epochs: $EPOCHS  model: $MODEL"
echo "test:  md: $MD_FILE  max-phase: $MAX_PHASE  limit: $TEST_LIMIT  playouts: $SEARCH_PLAYOUTS  root-phase: $ROOT_PHASE"
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
               --epochs "$EPOCHS" --spec "$SPEC" --delta "$DELTA" --save "$MODEL" | grep '\*' | tail -1)
  printf "%s  Move: " "$best_row"

  TRUNC_VPAT_DATA="$MODEL_BEST" TRUNC_ROOT_PHASE="$ROOT_PHASE" PLAYOUTS="$SEARCH_PLAYOUTS" \
    node evalmovedetails.js --agent puct-trunc --md "$MD_FILE" \
      --max-phase "$MAX_PHASE" --limit "$TEST_LIMIT" 2>&1 | grep SUMMARY

  last_count=$count
done
