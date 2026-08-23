'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * RAVE (Rapid Action Value Estimation) MCTS policy.
 *
 * Node structure: all stats kept in compact child-indexed arrays on the parent.
 * Child nodes are promoted lazily after N_EXPAND playout visits.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;
const { makeRng } = _isNode ? require('../xorshift.js') : window.XorShift;
const _ppat = _isNode ? require('../ppat-lib.js') : window.PPatterns;
const { createState, ppatMove, loadWeights } = _ppat;

// ── Factory ──────────────────────────────────────────────────────────────────
// create(cfg) → { getMove }.  cfg is a Util.makeCfg reader (slot-aware env:
// P1_/P2_ prefixes in selfplay), so two instances in one process never
// collide.  All per-instance state lives in this closure.
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const _model = _isNode
    ? loadWeights(cfg.str('PPAT_DATA', ''))
    : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);

  // Use uniform-random playout moves while board fullness < this fraction [0,1] (0 = off).
  if (_model) _model.uniformBelowPhase = cfg.float('PPAT_UNIFORM_BELOW_PHASE', 0);

  let _ppatState = null;
  function _ensurePpatState(N) {
    if (_ppatState === null || _ppatState.moves.length < N * N)
      _ppatState = createState(N);
  }

  const EXPLORATION_C = cfg.float('EXPLORATION_C', 0.4);

  const RAVE_K = cfg.float('RAVE_K', 800);

  const PLAYOUTS = cfg.int('PLAYOUTS', 0);

  // Minimum playout visits before a child node is promoted (allocated).
  const N_EXPAND = cfg.int('N_EXPAND', 2);

  // Fraction of parent RAVE stats inherited by a newly created child node.
  // Must be < 1 to prevent unbounded growth down the tree.
  const RAVE_INHERIT = cfg.float('RAVE_INHERIT', 0.2);

  const PRIOR_WINS   = 0.001;
  const PRIOR_VISITS = 2 * PRIOR_WINS;
  const RESIGN_MIN_PLAYOUTS = 20000;

  // Number of playout moves to use ppat policy before switching to uniform.
  // -1 = unlimited (ppat for entire playout).
  const PPAT_MOVES = cfg.int('PPAT_MOVES', -1);

  // Per-move probability of using the ppat policy (vs a uniform-random move) during
  // playouts, within the PPAT_MOVES window.  1 = always ppat (default).  e.g. 0.5
  // mixes in 50% uniform moves: cheaper (skips ppat feature extraction half the
  // time) and injects playout variety.
  const PPAT_RATIO = cfg.float('PPAT_RATIO', 1);

  // ── Fast playout helpers ──────────────────────────────────────────────────────

  // Returns { winner, played }.
  // played: reused Float32Array(cap) — caller zeroes it each call.
  function playTracked(game2, node, played) {
    const wasAlreadyOver = game2.gameOver;
    const N   = game2.N;
    const cap = N * N;

    played.fill(0, 0, cap);

    const moveLimit = 3 * game2.emptyCount + 20;
    const weightStep = 1 / cap;
    let moves = 0;
    let weight = 1.0;

    _ensurePpatState(N);

    while (!game2.gameOver && moves < moveLimit) {
      const current = game2.current;
      // usePolicy: use the ppat policy this move — within the PPAT_MOVES window and
      // (subject to PPAT_RATIO) not a randomly-mixed uniform move.
      const ppatActive = _model && (PPAT_MOVES < 0 || moves < PPAT_MOVES);
      const usePolicy  = ppatActive && (PPAT_RATIO >= 1 || Math.random() < PPAT_RATIO);
      const idx = usePolicy ? ppatMove(game2, _ppatState, _model) : game2.randomLegalMove();

      if (idx === PASS) {
        game2.play(PASS);
        moves++;
        continue;
      }
      if (weight > 0 && played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
      game2.play(idx);
      moves++;
      weight -= weightStep;
    }

    let winner;
    if (wasAlreadyOver) {
      winner = game2.calcWinner();
    } else {
      winner = game2.estimateWinner();
    }
    return { winner, played };
  }

  // ── Tree node ─────────────────────────────────────────────────────────────────

  // Enumerate legal non-true-eye moves from a Game2 state as integers.
  // Place moves are y*N+x; pass is PASS (-1).
  function getLegalMoves(game2) {
    const N     = game2.N;
    const cap   = N * N;
    const cells = game2.cells;
    const moves = [];
    for (let i = 0; i < cap; i++) {
      if (cells[i] !== 0) continue;
      if (game2.isTrueEye(i)) continue;
      if (game2.isLegal(i)) moves.push(i);
    }
    // Pass move must be at the end (if present).
    if (moves.length < cap / 3 || game2.consecutivePasses > 0) {
      moves.push(PASS);
    }
    return moves;
  }

  // Create a node for the position reached by `move`.
  // `game2` is the game state AFTER `move` was played (or initial state for root).
  // `ci`    is this node's index in parent.children / parent.child* arrays (-1 for root).
  // A new node seeds its RAVE stats by inheriting a fraction (RAVE_INHERIT) of the
  // grandparent's; the root and its children start from flat priors.
  function makeNode(move, parent, ci, game2, N) {
    const movesArr = getLegalMoves(game2);
    const M = movesArr.length;
    const area = N * N;

    // Copy into Int32Array for compact, cache-friendly storage.
    const legalMoves = new Int32Array(M);
    for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

    const children   = new Array(M).fill(null);
    const wins       = new Float32Array(M).fill(PRIOR_WINS);
    const visits     = new Float32Array(M).fill(PRIOR_VISITS);
    const raveWins   = new Float32Array(area);
    const raveVisits = new Float32Array(area);

    if (parent === null || parent.parent === null) {
      raveWins.fill(PRIOR_WINS);
      raveVisits.fill(PRIOR_VISITS);
    } else {
      const gparent = parent.parent;
      for (let m = 0; m < area; m++) {
        raveWins[m]   = RAVE_INHERIT * gparent.raveWins[m];
        raveVisits[m] = RAVE_INHERIT * gparent.raveVisits[m];
      }
    }

    const mover = -game2.current;

    return {
      move,
      parent,
      ci,           // this node's index in parent.children / parent.child* arrays (-1 for root)
      mover,        // player who made `move` to reach this node
      totalVisits:  0.1,  // sum of visits; incremented each playout
      selectedChild: -1,  // set by selectAndExpand; read by backpropagate

      legalMoves,   // Int32Array(M)
      children,     // Array(M) — promoted child node or null

      wins,    // Float32Array(M) — playout wins per child
      visits,  // Float32Array(M) — playout visits per child

      raveWins,     // Float32Array(N*N) — RAVE wins indexed by cell; updated by rollouts
      raveVisits
    };
  }

  // RAVE-blended UCT score for child index i of node.  Q blends the move's real
  // win-rate with its RAVE (AMAF) win-rate, weighted RAVE_K/(RAVE_K+realV); the
  // UCT exploration term (large while realV sits at its PRIOR_VISITS floor) keeps
  // unvisited children preferred until they accumulate real visits.

  function ucbScore(moveIdx, node, rng) {
    const move  = node.legalMoves[moveIdx];

    // RAVE
    const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);

    // Real
    const realW = node.wins[moveIdx];
    const realV = node.visits[moveIdx];
    const realWR = realW / realV;

    const raveWeight = RAVE_K / (RAVE_K + realV);
    const realWeight = 1 - raveWeight;

    // Combined win ratio
    const wr = realWeight * realWR + raveWeight * raveWR;

    const scoreBase = wr + 0.001 * rng.random();
    return scoreBase + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / realV);
  }

  // ── RAVE-MCTS core ────────────────────────────────────────────────────────────

  function selectAndExpand(root, rootGame2, N, rng) {
    let node = root;
    const game2 = rootGame2.clone();

    while (!game2.gameOver) {
      const M = node.legalMoves.length;
      if (M === 0) break;

      // Select best child by RAVE-blended score.
      let best = 0, bestScore = -Infinity;
      for (let i = 0; i < M; i++) {
        const s = ucbScore(i, node, rng);
        if (s > bestScore) { bestScore = s; best = i; }
      }

      game2.play(node.legalMoves[best]);

      // Promote child to a full node once it has accumulated enough visits.
      // Fall through to the descent check so the loop continues into the new node;
      // its children all have cv=0 < N_EXPAND, so the leaf case fires next iteration
      // (exactly one makeNode per playout, same as rave2's one-expansion-per-playout).
      if (node.children[best] === null && node.visits[best] >= N_EXPAND) {
        node.children[best] = makeNode(node.legalMoves[best], node, best, game2, N);
      }

      // After a pass, always force a second pass so the playout scores the current
      // board position (consecutive passes end the game).  This prevents rollouts
      // from a single-pass state playing on for many random moves and inflating
      // the pass move's apparent win rate.
      if (!game2.gameOver && game2.consecutivePasses > 0) {
        game2.play(PASS);
        node.selectedChild = best;
        break;
      }

      // Descend into the promoted child, if available.
      if (node.children[best] !== null) {
        node = node.children[best];
        node.selectedChild = -1;  // reset in case game ends before we select below
        continue;
      }

      // Unpromoted leaf — run playout from here.
      node.selectedChild = best;
      break;
    }

    return { node, game2 };
  }

  // Backpropagate playout result and update RAVE statistics.
  //
  // node.selectedChild holds the unpromoted child index that was played last,
  // or -1 if we descended all the way to a promoted node (game already over).
  //
  // childMover(n): the player choosing the next move from node n = opponent of n.mover.
  function backpropagate(node, winner, played, rootPlayer) {
    function childMover(n) {
      return -n.mover;
    }

    function updateRave(node, won, played, chooser) {
      if (chooser === BLACK) {
        for (let k = 0; k < played.length; k++) {
          const weight = played[k];
          if (weight > 0) {
            node.raveVisits[k] += weight;
            node.raveWins[k]   += won * weight;
          }
        }
      } else {
        for (let k = 0; k < played.length; k++) {
          const weight = played[k];
          if (weight < 0) {
            node.raveVisits[k] -= weight;
            node.raveWins[k]   -= won * weight;
          }
        }
      }
    }

    // Update the unpromoted leaf child stats (if we stopped before descending).
    // Also update RAVE at this node so root's RAVE is populated even when no
    // deeper promoted nodes exist (e.g. N_EXPAND=9999).
    const leafIdx = node.selectedChild;
    if (leafIdx !== -1) {
      const chooser = childMover(node);
      const won     = winner === chooser ? 1 : 0;
      node.visits[leafIdx]++;
      node.wins[leafIdx] += won;
      node.totalVisits++;
      updateRave(node, won, played, chooser);
    }

    // Walk up the tree, updating each parent's child arrays and RAVE arrays.
    while (node.parent !== null) {
      const ci      = node.ci;   // stored at promotion time — no lookup needed
      const chooser = childMover(node.parent);
      const won     = winner === chooser ? 1 : 0;
      node.parent.visits[ci]++;
      node.parent.wins[ci] += won;
      node.parent.totalVisits++;
      updateRave(node.parent, won, played, chooser);
      node = node.parent;
    }
  }

  // ── Public interface ──────────────────────────────────────────────────────────

  function getMove(game, timeBudgetMs, options = {}) {
    const rng = options.rng || makeRng();
    if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

    const N          = game.cells ? game.N : game.boardSize;
    const game2      = game.cells ? game.clone() : game.toGame2();
    const rootPlayer = game2.current;

    // Obvious pass: opponent just passed and we're already winning — end the game.
    if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
      return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };
    }

    const root = makeNode(null, null, -1, game2, N);

    // Pre-allocate played buffer — reused across all playouts.
    const played = new Float32Array(N * N);

    const playoutLimit = options.playoutLimit || PLAYOUTS;
    const deadline = performance.now() + timeBudgetMs;
    let playouts = 0;

    do {
      playouts++;
      const { node, game2: simGame2 } = selectAndExpand(root, game2, N, rng);
      const { winner, played: p } = playTracked(simGame2, node, played);
      backpropagate(node, winner, p, rootPlayer);
    } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

    // Best child: most playout visits; ties broken by RAVE-blended score.
    const M = root.legalMoves.length;
    let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const cv = root.visits[i];
      if (cv > bestVisits || (cv === bestVisits && ucbScore(i, root, rng) > bestScore)) {
        bestVisits = cv;
        bestScore  = ucbScore(i, root, rng);
        bestIdx    = i;
      }
    }

    const children = [];
    for (let i = 0; i < M; i++) {
      const m = root.legalMoves[i];
      children.push({
        move: m === PASS ? { type: 'pass' } : { type: 'place', x: m % N, y: (m / N) | 0 },
        visits: root.visits[i],
        wins:   root.wins[i],
      });
    }
    children.sort((a, b) => b.visits - a.visits);

    let totalChildWins = 0;
    for (let i = 0; i < M; i++) totalChildWins += root.wins[i];
    const rootWinRatio = totalChildWins / root.totalVisits;

    if (playouts >= RESIGN_MIN_PLAYOUTS && game2.emptyCount <= N * N / 2 && root.wins[bestIdx] <= PRIOR_WINS) {
      return { type: 'pass', move: PASS, info: 'no winning line found', children, rootWinRatio };
    }

    const m = root.legalMoves[bestIdx];
    const cv = root.visits[bestIdx];
    const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

    const result = m === PASS ? { type: 'pass', move: PASS, children, rootWinRatio }
                              : { type: 'place', move: m, x: m % N, y: (m / N) | 0, children, rootWinRatio };
    result.info = `value=${(game.current===BLACK?bestWinRatio:(1-bestWinRatio)).toFixed(3)}`;
    return result;
  }


  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
