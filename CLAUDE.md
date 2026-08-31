# Torogo — notes for Claude

## Board

- **Toroidal board (no edges).** Every cell wraps in both directions, so
  there are no edge or corner cells — every position has the same local
  neighbourhood structure. Anything reasoning about "edge proximity" or
  "corners" does not apply here.

## Invariants

- **Valid-position replay invariant.** When replaying a valid game position stone
  by stone (setting `current` to the placed stone's colour before each call),
  every placement must be legal and no placement may cause a capture. If either
  happens, the "position" was never reachable by legal play, or the replay
  order/colour is wrong.

## Terminology

- **"Ratio", never "rate", for a fraction of a whole.** A rate is a quantity per
  unit time; a fraction of some total is a ratio. Name flags, variables and prose
  accordingly (`--hpat-pos-ratio`, not `--hpat-pos-rate`).
- **Name the unit a knob counts.** The two hpat knobs are easy to confuse, so
  always say which one is meant: `--hpat-pos-ratio R` is the fraction of
  POSITIONS in which the hpat feature is computed at all; `--hpat-topn N` is how
  many MOVES get ranked within such a position (0 = every candidate). Never
  write a bare "rate", "ratio" or "N" where either could be meant.

## Workflow

- **Never use `pkill`.** To stop a process, kill specific PIDs (e.g. track
  the child PIDs you spawned, or `pgrep -f '<pattern>' | xargs -r kill`
  with a pattern that cannot match unrelated processes or the shell
  running the command itself).

- **After a background job completes, cat the log file.** When a Bash command
  was started with `run_in_background: true` and the harness reports its
  completion, immediately `cat` the output file so the user can review it
  before deciding next steps.

- **Training external policy: `ref-npat-softmax`.** For training runs that
  need an external opponent (`--ext`), use `ref-npat-softmax` — same npat
  weights as plain `npat` but with softmax sampling instead of greedy
  argmax, so it provides stochastic moves.  All parameters are hardcoded
  in the agent file; do **not** pass any env vars when using it.
  Do **not** use `vlibpat-ref-2x2` as `--ext`: it loads a 175k-weight
  model and runs its own search at every move, roughly doubling per-move
  cost.
- **Elo rating via local CGOS.** `cgos/` contains a toroidal CGOS server
  (vendored + patched) for rating agents against the reference fleet.
  Start the ladder with `node cgos/run.js`, attach a candidate with
  `node cgos/join.js --p <agent> --budget <ms> --games <n>`, read ratings
  with `node cgos/standings.js` (quote the `mleElo` column and game count).
  The scale is anchored at `random` = 0; all other ratings float.
  The server must keep
  `ko = SIMPLE` and komi 3.5 to match Game2.  Rule/compat tests:
  `python3 -m unittest discover -s cgos/tests` and
  `node cgos/tests/replay-compat.js`.
- **Eval reporting.** When summarising selfplay/eval results, report the
  win-rate (e.g. p2 win% or p1 win%) and game count.  Do **not** report
  the `p2Better%` column.
- **Training reporting.** When summarising training-run output, do **not**
  report the training-accuracy (`acc%`) column — it's not informative.
