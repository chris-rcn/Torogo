# Toroidal CGOS — Elo rating for Torogo agents

A local [CGOS](https://github.com/zakki/cgos) (Computer Go Server) adapted to
Torogo's toroidal board.  A fleet of **house** reference engines sits idle,
always ready; connecting a **trial** engine (the agent you want rated)
starts a continuous stream of games against the nearest-rated opponents,
and the Elo estimate arrives quickly (new players start at K=200, so
~15–20 games already give a usable number, and the Bradley–Terry fit in
`standings.js` converges even faster).

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
- **Adding lower anchors:** after a burn-in run (e.g. join each reference
  agent as a trial for a while so the house fleet gets played), copy the
  fast engines' stable `mleElo` values into the `anchors` map of
  `refs.json` (they are written to the server on the next `run.js`
  start).  Multiple pinned rungs make new-engine ratings settle faster
  and stop ladder drift.  Don't guess anchor values — a mispinned anchor
  warps the whole scale.
- Matchmaking pairs each trial engine with the waiting opponent of nearest
  (jittered) rating, so a new engine climbs to its level within a few games
  and then plays informative near-50% games — this is what makes the
  estimate fast.

## House vs trial engines, and scheduling

Every engine is either **house** (listed in the server's `house` table)
or **trial** (everything else):

- *House* engines never initiate a game.  With no trial connected the
  server sits completely idle — the reference fleet costs nothing while
  it waits.
- *Trial* engines play continuously while connected.  Every scheduler
  tick (`scheduleInterval`, default 2 s) each waiting trial is paired
  with the waiting engine of nearest jittered rating (house or another
  trial).

`run.js` registers its whole fleet as house; `join.js` joins as a trial
by default (`--house` to add an extra house engine on the fly — the
server re-reads the table every tick).

There is **no round barrier** (stock CGOS waits for all games to finish
before pairing the next round): a finished engine is re-paired on the
next tick while other games keep running, so several trials play in
parallel and a slow game never blocks a fast one.  Concurrency = number
of connected trials (each gets its own game; every engine is its own OS
process).  Keep total engines under your core count or per-move budgets
get noisy.

The stock CGOS delays (45 s startup, 15 s scheduler ticks, 3 s match
start) are meant for internet play; `torogo9.ini` overrides them
(`startupDelay`/`scheduleInterval`/`matchStartDelay` = 3/2/0.25 s), so
per-game overhead is roughly one tick.

## Budgets and engine identity

A slow searcher and its fast setting are *different players*.  The per-move
budget is fixed per engine name (`rave-100`, `rave-500`), never taken from
the server clock, so ratings stay comparable across machines and runs.  The
server clock (300 s/side) is just an anti-hang forfeit.

## Toroidal adaptations (vs stock CGOS)

- `server/cgos/gogame/go.py`: adjacency wraps in both directions
  (precomputed `nbrs` table) — affects captures, suicide, and scoring.
  Rule tests: `python3 -m unittest discover -s cgos/tests`.
- Scoring: upstream CGOS uses a Tromp–Taylor region flood-fill; this
  server instead uses the same 1-step area count as
  `Game2.estimateScore()` (which is what `calcWinner()` calls): every
  stone counts as alive, an empty point counts only when all its
  adjacent stones are one color, and deeper empty points are neutral.
  Dead stones must be captured to get credit for them.  The two scorings
  agree on played-out boards but can differ after an early double-pass
  over a large open region — matching Game2 exactly means the server's
  verdict can never disagree with the engines' own.
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
