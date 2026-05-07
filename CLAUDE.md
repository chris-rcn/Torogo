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

## Workflow

- **After a background job completes, cat the log file.** When a Bash command
  was started with `run_in_background: true` and the harness reports its
  completion, immediately `cat` the output file so the user can review it
  before deciding next steps.

- **Reference player: `vlibpat-ref-2x2`.** Use `ai/vlibpat-ref-2x2.js` as the
  default reference player for both training (`--ext vlibpat-ref-2x2`) and
  evaluation (`--p1 vlibpat-ref-2x2` or `--p2 vlibpat-ref-2x2`). It's a
  hardcoded-config 234k-game vlibpat checkpoint with verified strength of
  68.5% / 200g vs rave-500 and 62.5% / 200g vs npat.  Do not pass any env
  vars when using it; all parameters are baked into the agent file.
