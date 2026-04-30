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
