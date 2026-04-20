# Torogo — notes for Claude

## Invariants

- **Valid-position replay invariant.** When replaying a valid game position stone
  by stone (setting `current` to the placed stone's colour before each call),
  every placement must be legal and no placement may cause a capture. If either
  happens, the "position" was never reachable by legal play, or the replay
  order/colour is wrong.
