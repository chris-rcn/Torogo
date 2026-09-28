#!/bin/sh
# Per-phase movedetails MAE for every rated CGOS house agent, in cost order.
# One output file per agent: md-phase/<agent>.txt (the full evalmovedetails
# output, phase-band table and SUMMARY at the end), also echoed to stdout as
# it runs.  An agent whose file already
# holds a SUMMARY line is skipped, so the sweep can be resumed.
#
# usage: md-phase-sweep.sh [--limit N] [agent ...]
#   --limit N   positions per agent (default 1000)
#   agent ...   restrict to these agents (default: all, in cost order)

limit=1000
if [ "$1" = "--limit" ]; then limit=$2; shift 2; fi

agents="ref-ppat random ref-featurepol-softmax ref-npat-softmax ref-search-top2-fp
ref-fp-heavy ref-ab2-fp4-vpat ref-vlibpat-or-fp ref-mc-200 ref-vlibpat
ref-cascade-k3-v2-p100 ref-cascade-k3-v2-p300 ref-rave-1k ref-puct-ppat-fp-e2-u6-300
ref-rave-2k ref-puct-ppat-300 prod-2026-08-18-5k ref-puct-ppat-fp-e2-u6-1k
ref-puct-ppat-fp-e2-1k ref-puct-ppat-1k prod-2026-08-18-10k ref-puct-ppat-fp-e2-3k
prod-2026-08-18-20k ref-puct-ppat-3k ref-puct-ppat-fp-e2-u6-6k"
[ $# -gt 0 ] && agents="$*"

cd "$(dirname "$0")"
mkdir -p md-phase
for a in $agents; do
  f=md-phase/$a.txt
  if grep -q '^SUMMARY' $f 2>/dev/null; then echo "skip $a (done)"; continue; fi
  echo "$(date +%H:%M) $a"
  node evalmovedetails.js --file movedetails_5059.md --agent $a --limit $limit --show-phases 10 2>&1 | tee $f
done
