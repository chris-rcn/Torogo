# Torogo

A Go engine for the **toroidal** board — a board with no edges, where every
point wraps in both directions so all cells share the same local neighbourhood.
The target size is 13×13 (much development happens on 9×9); komi is 3.5 and ko is
simple ko.

Because the board is a torus there are no edges, corners, or first/second-line
shapes. Any reasoning that assumes edge proximity does not apply here — every
pattern below is defined on the wrapped neighbourhood.

## Requirements

Plain Node.js (developed on v22) — there is no `package.json`, no build step, and
no dependencies for the JavaScript. The C playout trainer is built separately
with `c/build.sh` (see [C engine](#c-engine)).

## Quick start

Play against the engine in a browser:

```
# open index.html in a browser (it loads the JS engine + data files directly)
```

Play two agents against each other:

```
node selfplay.js --p1 prod --p2 rave-ppat --budget 1000 --size 13 --limit 100
```

`--p1`/`--p2` name agents from `ai/` (default `prod`); `--budget` is ms/move.
Pass `--help` for the full option set (randomised/mirror openings, phase windows,
fallback agent, …).

## Architecture

The playing strength comes from a small stack of learned models, each a hashed
pattern family over the toroidal neighbourhood:

- **featurepol** — a spec-driven, hash-keyed softmax policy used for move priors
  and top-K candidate pruning (`featurepol-lib.js`; the fielded weights are
  `featurepol-cbk7wa32.js`).
- **vpat** (`vpatterns.js`) — a truncated-playout **value** model: windowed
  board-shape features (sizes 1–4, plus the 2×3/3×4 and size-8 octagon windows)
  hashed to weights, evaluated at a truncation point instead of playing to the end.
- **ppat** (`ppat-lib.js`) — the **playout policy**, trained by softmax-based
  reinforcement (the Huang/Coulom/Lin CG2010 "simulation balancing" method).
- **health** (`health-lib.js`) — a chain-survival model.

`game2.js` is the core toroidal board engine (`game3.js` adds ladder tracking).
`Game2(N)` starts with a free black centre stone (white to move); `Game2(N, false)`
is empty.

### Agents

Agents live in `ai/` and expose `getMove(game, budgetMs, options)` (factory
agents export `create(cfg)`). The currently *fielded* agent is **`ai/prod.js`**
(a self-contained copy of `puct-ppat-fp`), but the **strongest** agent is
`puct-ppat-fp-trunc` — described below. Other notable families: the `puct-*`
search agents, `mc-ppat` (a minimal playout-policy probe), `rave-*`, and the
frozen `ref-*` reference agents used for rating. `ref-*` agents are immutable
once fielded — any strength-affecting change gets a new name.

### puct-ppat-fp-trunc (strongest)

`ai/puct-ppat-fp-trunc.js` is where the whole model stack comes together in one
search. It is a PUCT Monte-Carlo tree search whose every component is driven by a
learned model:

- **featurepol → priors and pruning.** At each node the featurepol policy
  supplies the prior `P(s,a)` in the PUCT term and prunes the candidate moves to
  its top-K. The width is phase-conditioned — narrow in the opening, wider late
  (default ramp ~10 → 40 by board fullness) — since the best pruning width rises
  with phase.
- **PUCT selection.** A child is chosen by
  `score = Q + C_PUCT · P(s,a) · √N_total / (1 + N_a)`, balancing the exploited
  mean value `Q` against the policy-weighted exploration term.
- **RAVE.** `Q` is RAVE/AMAF-blended (all-moves-as-first), so early estimates
  borrow strength from move outcomes seen elsewhere in the subtree; playout
  prefix moves fill the RAVE trace.
- **ppat → leaf playouts.** A newly expanded leaf is evaluated by a **ppat**
  policy playout (uniform below a phase threshold, ppat-policy above it).
- **vpat → truncation (the "-trunc" part, and the source of its strength).**
  This is what distinguishes it from `puct-ppat-fp`. Truncation is decided once
  per move at the root: when the root's phase is below a threshold, every playout
  runs only a short **prefix** (a fullness advance, `TRUNC_PHASE_DELTA`) and then
  takes its leaf value from a static **vpat** value evaluation
  (`train-vpat-supervised` checkpoint, `V(s) = P(BLACK wins)`) instead of playing
  to the end; above the threshold, playouts run full. The cut point is measured
  by net board-filling progress (an integer empty-count drop), so captures during
  the prefix delay it correctly. Truncating trades a little per-playout accuracy
  for many more playouts in the phase where the value model is reliable.
- **Root symmetry reduction.** In symmetric opening positions, moves in the same
  symmetry orbit are equal in value, so only one representative is searched —
  exact, and self-limiting once the board becomes asymmetric.

Values backpropagate fractionally (each chooser credited `value` for Black,
`1 − value` for White); terminal positions are scored exactly; the move played is
the most-visited root child. So the pieces fit as a pipeline: **featurepol**
shapes *where* the search looks (priors + pruning), PUCT+RAVE decide *how the
budget is spent*, and **ppat** playouts truncated by the **vpat** value model
supply the *leaf evaluations* — the playout policy, value model, and priors all
pulling together in one tree.

## Training

Each model has its own trainer; all print a live progress table.

- **ppat** — the C trainer, run in parallel with a monitor:
  `c/train-ppat-parallel <workers> <evals-file> [args]`. Reports `trMSE`,
  `directWR` (vs a fixed reference), and `mdMae` (mc-ppat move-selection regret
  vs a labelled movedetails file).
- **vpat** — `node train-vpatterns.js --eval <ref> --spec '<spec>' [args]`
  (TD self-play), or `train-vpat-supervised.js` (from a labelled corpus).
  `--bootstrap <N>` runs N random games first to seed the value table and
  greatly speeds convergence; `--start-phase <f|uniform>` trains a phase band.
- **health** — `node train-health.js [args]`.
- **featurepol / others** — `train-featurepol-reinforce.js`, `train-vlibpat.js`,
  `train-hpatterns.js`.

Board fullness (**phase** = 1 − empty/area) is the pervasive conditioning
variable — many models and training runs are restricted to a phase band.

## Evaluation

- `selfplay.js` — head-to-head win rate between two agents.
- `evalmovedetails.js` — agent move-selection regret against a file of positions
  with labelled candidate win-ratios (`*.md` format, see `movedetails-format.js`;
  generate with `createmovedetails.js`).
- `eval-value-accuracy.js`, `eval-uniform.js`, `evalladders.js` — value-model
  accuracy and tactical (ladder) checks.

### Elo rating (local CGOS)

`cgos/` vendors a toroidal CGOS server for rating agents against the reference
fleet (scale anchored at `random` = 0):

```
node cgos/run.js                                             # start the ladder
node cgos/join.js --p <agent> --budget <ms> --games <n>     # attach a candidate
node cgos/standings.js                                       # read ratings (mleElo)
```

## C engine

`c/` holds a C port of the board and the ppat trainer (the JS trainer is too slow
for full runs). Build everything with `c/build.sh`; it produces the `train-ppat`
binary that `c/train-ppat-parallel` drives.

## Tests

The test suites are standalone Node scripts — run them directly:

```
node test-game2.js
node test-patterns.js
node test-symmetry.js
```

The CGOS rule/compat tests use Python: `python3 -m unittest discover -s cgos/tests`.

## Conventions

`CLAUDE.md` records the project's working conventions (terminology — e.g. always
"ratio" not "rate", naming, the valid-position replay invariant, CGOS usage, and
reporting norms). Read it before contributing.
