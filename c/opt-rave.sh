#!/bin/bash
# opt-rave.sh — synchronous finite-difference coordinate ascent of RAVE constants.
#
# Each ROUND runs 8 fixed-length probes — K, C, N, INH each stepped up/down on a
# 1.5x ladder — against the current anchor, with NO early stop (every probe plays
# exactly GAMES games).  When all 8 finish, each parameter HALF-steps toward its
# winning (>50%) direction, and every improving parameter steps at once ("combined
# positive directions").  Then the next round.  Half-stepping + averaging across
# rounds makes it robust to per-round noise: a consistent gradient accumulates,
# zero-mean noise cancels.
#
#   PO=2000 GAMES=200 ./opt-rave.sh
#   PO=300  GAMES=30  ./opt-rave.sh     # fast smoke test
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"; BIN="$DIR/tune_rave.bin"
PO=${PO:-2000}; SIZE=${SIZE:-9}; GAMES=${GAMES:-200}; RANDM=${RANDM:-3}
K=${K:-800}; C=${C:-0.4}; N=${N:-2}; INH=${INH:-0.2}
RUN=$(mktemp -d "${TMPDIR:-/tmp}/opt-rave.XXXXXX")

PIDS=()
cleanup(){ [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null; rm -rf "$RUN"; }
trap cleanup EXIT INT TERM
say(){ echo "[$(date +%H:%M:%S)] $*"; }
af(){ awk "BEGIN{printf \"%.6g\", $1}"; }

# 1.5x ladder target for (param, direction u/d).
ptv(){
  case "$1" in
    K)   [ "$2" = u ] && af "$K*1.5"   || af "$K/1.5";;
    C)   [ "$2" = u ] && af "$C*1.5"   || af "$C/1.5";;
    INH) [ "$2" = u ] && af "$INH*1.5" || af "$INH/1.5";;
    N)   [ "$2" = u ] && awk "BEGIN{print int($N*1.5+0.5)}" || awk "BEGIN{n=int($N/1.5+0.5);print n<1?1:n}";;
  esac
}

say "opt-rave (sync FD): PO=$PO GAMES=$GAMES SIZE=$SIZE  start: K=$K C=$C N=$N INH=$INH  ($RUN)"
DIRS="K:u K:d C:u C:d N:u N:d INH:u INH:d"

round=0
while :; do
  round=$((round + 1))
  PIDS=(); declare -A VAL
  for pd in $DIRS; do
    p=${pd%:*}; d=${pd#*:}; v=$(ptv "$p" "$d"); VAL["$pd"]="$v"
    pk=$K pc=$C pn=$N pinh=$INH
    case "$p" in K) pk=$v;; C) pc=$v;; N) pn=$v;; INH) pinh=$v;; esac
    env PLAYOUTS=$PO RAVE_K=$pk EXPLORATION_C=$pc N_EXPAND=$pn RAVE_INHERIT=$pinh \
        RAVE_K_B=$K EXPLORATION_C_B=$C N_EXPAND_B=$N RAVE_INHERIT_B=$INH \
        "$BIN" --size "$SIZE" --rand-moves "$RANDM" --limit "$GAMES" \
        > "$RUN/r${round}_${p}${d}.log" 2>&1 &
    PIDS+=("$!")
  done
  wait

  declare -A WIN
  for pd in $DIRS; do
    p=${pd%:*}; d=${pd#*:}
    WIN["$pd"]=$(grep -m1 '^RESULT' "$RUN/r${round}_${p}${d}.log" | sed -n 's/.*Awin=\([0-9.]*\).*/\1/p')
    WIN["$pd"]=${WIN["$pd"]:-0}
  done

  # Combined step: each param half-steps toward its better, >50% direction.
  moved=""
  for p in K C N INH; do
    wu=${WIN[$p:u]}; wd=${WIN[$p:d]}
    bd=u; bestw=$wu
    awk "BEGIN{exit !($wd > $wu)}" && { bd=d; bestw=$wd; }
    awk "BEGIN{exit !($bestw > 50)}" || continue
    v=${VAL[$p:$bd]}
    case "$p" in
      K)   K=$(af "($K+$v)/2");;
      C)   C=$(af "($C+$v)/2");;
      INH) INH=$(af "($INH+$v)/2");;
      N)   N=$(awk "BEGIN{print int(($N+$v)/2+0.5)}");;
    esac
    moved="$moved ${p}${bd}=$(printf '%s' "$bestw")%"
  done
  unset VAL WIN

  if [ -z "$moved" ]; then
    say "round $round: no positive direction; CONVERGED  K=$K C=$C N=$N INH=$INH"
    break
  fi
  say "round $round: step$moved  ->  K=$K C=$C N=$N INH=$INH"
done
