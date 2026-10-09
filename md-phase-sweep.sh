#!/bin/sh
# Per-phase movedetails MAE for every CGOS house agent (cgos/refs.json), in cost
# order: ms/mv from cgos/standings.js, cheapest first; agents it has no ms/mv
# for go last, in refs.json order.
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
#   agent ...   restrict to these agents (default: the house fleet, in cost order)

file=movedetails_5059.md; dir=md-phase; limit=
while :; do
  case $1 in
    --file)  file=$2; shift 2;;
    --dir)   dir=$2; shift 2;;
    --limit) limit="--limit $2"; shift 2;;
    *) break;;
  esac
done

if [ $# -gt 0 ]; then
  agents="$*"
else
  agents=$(cd "$(dirname "$0")" && node -e '
    const { execFileSync } = require("child_process");
    const refs = JSON.parse(require("fs").readFileSync("cgos/refs.json", "utf8")).refs.map(r => r.name);
    const cost = new Map();
    try {
      const out = execFileSync("node", ["cgos/standings.js"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
      for (const line of out.split("\n")) {
        const m = line.match(/^(\S+)\s+-?\d+\??\s+\d+\s+(\d+)/);
        if (m) cost.set(m[1], +m[2]);
      }
    } catch (e) { console.error("md-phase-sweep: cgos/standings.js failed; refs.json order"); }
    const order = refs.map((a, i) => ({ a, i, c: cost.has(a) ? cost.get(a) : Infinity }))
                      .sort((x, y) => x.c - y.c || x.i - y.i);
    console.log(order.map(o => o.a).join(" "));
  ') || exit 1
fi


cd "$(dirname "$0")"
mkdir -p "$dir"
for a in $agents; do
  f=$dir/$a.txt
  if grep -q '^SUMMARY' $f 2>/dev/null; then echo "skip $a (done)"; continue; fi
  echo "$(date +%H:%M) $a"
  node evalmovedetails.js --file $file --agent $a $limit --show-phases --verbose > $f 2>&1
  grep '^SUMMARY' $f
done
