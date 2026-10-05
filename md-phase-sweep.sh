#!/bin/sh
# Per-phase movedetails MAE for every rated CGOS house agent, in cost order.
# One output file per agent: <dir>/<agent>.txt (the full evalmovedetails
# output, phase-band table and SUMMARY at the end).  Run with --verbose, so
# each file also holds every position's gap (rows keyed by the position's index
# in the MD file) for per-position analysis across agents; md-phase-fit.js
# reads only the band table and SUMMARY, which verbose output leaves intact.  An agent whose file already
# holds a SUMMARY line is skipped, so the sweep can be resumed.
#
# usage: md-phase-sweep.sh [--file F] [--dir D] [--limit N] [agent ...]
#   --file F    movedetails file (default movedetails_5059.md)
#   --dir D     output directory (default md-phase)
#   --limit N   positions per agent (default: all)
#   agent ...   restrict to these agents (default: all, in cost order)

file=movedetails_5059.md; dir=md-phase; limit=
while :; do
  case $1 in
    --file)  file=$2; shift 2;;
    --dir)   dir=$2; shift 2;;
    --limit) limit="--limit $2"; shift 2;;
    *) break;;
  esac
done

agents="ref-ppat random ref-featurepol-softmax ref-npat-softmax ref-search-top2-fp
ref-fp-heavy ref-ab2-fp4-vpat ref-vlibpat-or-fp ref-mc-200 ref-vlibpat
ref-cascade-k3-v2-p100 ref-cascade-k3-v2-p300 ref-rave-1k ref-puct-ppat-fp-e2-u6-300
ref-rave-2k ref-puct-ppat-300 prod-2026-08-18-5k ref-puct-ppat-fp-e2-u6-1k
ref-puct-ppat-fp-e2-1k ref-puct-ppat-1k prod-2026-08-18-10k ref-puct-ppat-fp-e2-3k
prod-2026-08-18-20k ref-puct-ppat-3k ref-puct-ppat-fp-e2-u6-6k"
[ $# -gt 0 ] && agents="$*"

cd "$(dirname "$0")"
mkdir -p "$dir"
for a in $agents; do
  f=$dir/$a.txt
  if grep -q '^SUMMARY' $f 2>/dev/null; then echo "skip $a (done)"; continue; fi
  echo "$(date +%H:%M) $a"
  node evalmovedetails.js --file $file --agent $a $limit --show-phases --verbose > $f 2>&1
  grep '^SUMMARY' $f
done
