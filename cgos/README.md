# Toroidal CGOS — Elo rating for Torogo agents

A local [CGOS](https://github.com/zakki/cgos) (Computer Go Server) adapted to
Torogo's toroidal board.  A fleet of reference engines plays continuously;
any new agent connects as one more engine and gets an Elo estimate quickly
(new players start at K=200, so ~15–20 games already give a usable number,
and the Bradley–Terry fit in `standings.js` converges even faster).

## Quick start

```sh
# 1. Start the server + the reference fleet (runs until Ctrl-C):
node cgos/run.js

# 2. In another terminal, connect the agent you want rated:
node cgos/join.js --p my-new-agent --budget 100 --games 50

# 3. Watch the ratings:
node cgos/standings.js
```

State persists in `cgos/data/9x9/` (ratings, game archive, SGF records), so
the reference ladder keeps improving across runs.

## Pieces

| file            | what it is |
|-----------------|------------|
| `gtp.js`        | GTP wrapper: exposes any `ai/*.js` agent as a GTP engine (`node cgos/gtp.js --p rave --budget 500`) |
| `run.js`        | Launcher: server + one client per engine in `refs.json`, writes rating anchors |
| `join.js`       | Connect one engine (your candidate) to a running server; prints its rating after each game |
| `standings.js`  | Rating table: server's incremental Elo + anchored Bradley–Terry MLE fit with 95% CI (uses `elo-lib.js`) |
| `refs.json`     | Reference fleet: engine names, agents, per-move budgets, and rating anchors |
| `torogo9.ini`   | Server config: 9×9, komi 3.5, **simple ko** (matches Game2), 300 s/side |
| `server/`       | Vendored `zakki/cgos` python server, patched for toroidal rules (see `server/UPSTREAM.txt`) |
| `client/`       | Vendored python CGOS client (unmodified) — bridges any GTP engine to the server |
| `tests/`        | Toroidal rule tests + Game2↔server replay-equivalence check |

## How ratings work

- The server runs classic CGOS incremental Elo: new engines start with
  K=200, decaying toward K=3 as games accrue.  A `?` marks provisional
  ratings (K>16).
- Engines in the `anchors` map of `refs.json` are pinned to a fixed rating
  and define the scale (default: `vlibpat-ref-3x3` = 1700).
- `standings.js` additionally fits all games with an anchored
  Bradley–Terry maximum-likelihood model (`elo-lib.js`); this uses every
  game optimally and is the number to quote.
- **Adding lower anchors:** after a burn-in run, copy the fast engines'
  stable `mleElo` values into the `anchors` map of `refs.json` (they are
  written to the server on the next `run.js` start).  Multiple pinned
  rungs make new-engine ratings settle faster and stop ladder drift;
  anchor-vs-anchor games are throttled by `anchor_match_rate`.  Don't
  guess anchor values — a mispinned anchor warps the whole scale.
- Matchmaking pairs engines of adjacent (jittered) rating each round, so a
  new engine climbs to its level within a few rounds and then plays
  informative near-50% games — this is what makes the estimate fast.

## Throughput

Rounds are barrier-synchronized: the server pairs every waiting engine,
then waits for **all** games to finish before pairing the next round.  Two
consequences:

- One slow engine (e.g. `rave-500` at ~30 s/game) gates every round even
  for the fast pairs.  Keep the standing fleet fast; attach slow engines
  deliberately (`join.js`) when you want them rated, or run them on a
  second server (`--ini`/`--data`/port of their own).
- The stock CGOS delays (45 s startup, 15 s between rounds, 3 s round
  start) are meant for internet play.  `torogo9.ini` overrides them
  (`startupDelay`/`scheduleInterval`/`matchStartDelay` = 3/2/0.25 s), so
  round overhead is well under a second per game with a fast fleet.

## Budgets and engine identity

A slow searcher and its fast setting are *different players*.  The per-move
budget is fixed per engine name (`rave-100`, `rave-500`), never taken from
the server clock, so ratings stay comparable across machines and runs.  The
server clock (300 s/side) is just an anti-hang forfeit.

## Toroidal adaptations (vs stock CGOS)

- `server/cgos/gogame/go.py`: adjacency wraps in both directions
  (precomputed `nbrs` table) — affects captures, suicide, and the
  Tromp–Taylor scoring flood-fill.  Rule tests: `python3 -m unittest
  discover -s cgos/tests`.  (The scoring flood-fill is *not* dead-stone
  adjudication — Tromp–Taylor counts every stone as alive, same as
  `Game2.calcWinner`; it only assigns empty regions (eyes/territory) to
  the color that fully surrounds them.  Agents must capture dead stones
  to get credit, as everywhere else in Torogo.)
- Ko: server must use `ko = SIMPLE` (Game2 implements simple ko; positional
  superko would forfeit engines for legal-in-Game2 moves).
- First move: `gtp.js` answers the game's first `genmove` with the center
  point — Game2 auto-plays black's first move there, and on a torus all
  first moves are equivalent.
- Equivalence guarantee: `node cgos/tests/replay-compat.js --games 50`
  replays random Game2 games into the server's GoGame and checks that every
  legal move is accepted and every winner matches `calcWinner()`.

## Requirements

`python3` with `passlib` (`python3 -m pip install passlib`).  Node ≥ 22
(`node:sqlite` for `join.js`/`standings.js`).

## Notes

- Engines whose weights live in gitignored dirs (`ref/`, `out/`) only work
  on machines that have those files; edit `refs.json` accordingly.
- Deterministic agents (greedy `npat`) make poor references — identical
  games repeat.  Prefer stochastic variants (`ref-npat-softmax`, dithered
  vlibpat refs, search agents).
- The web page builder (`webuild.py`) and Tcl viewer were not vendored;
  `standings.js` replaces them.  Game records land in
  `cgos/data/9x9/SGF/` (note: standard SGF viewers will render toroidal
  games with flat-board assumptions).
