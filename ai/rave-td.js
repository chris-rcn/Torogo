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

const { PASS, BLACK, WHITE, EMPTY } = _isNode ? require('../game2.js') : window.Game2;
const Util    = _isNode ? require('../util.js')    : window.Util;
const IntMap  = _isNode ? require('../int-map.js') : window.IntMap;
const { makeIntMap } = IntMap;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = Util.envInt('PLAYOUTS', 0);

// Minimum playout visits before a child node is promoted (allocated).
const N_EXPAND = Util.envInt('N_EXPAND', 2);

// TD learning rates: LR0 at start of budget, LR1 at end.
const LR0 = Util.envFloat('TD_LR0', 0.6);
const LR1 = Util.envFloat('TD_LR1', 0.3);

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.4);

const RAVE_K = Util.envFloat('RAVE_K', 600);

// Equivalence weight for the TD prior (virtual visits).  Override with TD_WEIGHT=<n>.
const TD_WEIGHT = Util.envFloat('TD_WEIGHT', 0);


const VERBOSE = Util.envInt('VERBOSE', 0);

// Fraction of parent RAVE stats inherited by a newly created child node.
// Must be < 1 to prevent unbounded growth down the tree.
const RAVE_INHERIT = Util.envFloat('RAVE_INHERIT', 0.2);

// ── TD model ─────────────────────────────────────────────────────────────────
//
// keyToIdx: IntMap<featureKey, weightIndex>
// weightsArr: number[] — weight values indexed by weightIndex
//
// resolveKey returns the weight index for a feature key, inserting weight 0 if new.
function resolveKey(key, ctx) {
  let wi = ctx.keyToIdx.get(key);
  if (wi < 0) {
    wi = ctx.weightsArr.length;
    ctx.keyToIdx.set(key, wi);
    ctx.weightsArr.push(0);
  }
  return wi;
}

// Module-level model state (persists across getMove calls).
let keyToIdx_start = makeIntMap();   // snapshot saved at move 1
let weightsArr_start = [];
let keyToIdx   = makeIntMap();       // current model
let weightsArr = [];
let lastMoveCount = 0;

// ── Feature buffers and extraction ───────────────────────────────────────────
// 1×1 patterns only: each occupied cell contributes one feature.
// posOf[s] = board index of the stone whose feature occupies slot s.

function makeBuf(area) {
  return {
    idxs:  new Int32Array(area),
    n:     0,
    sum:   0,
    slotA: new Int32Array(area).fill(-1),  // slot index for position j (-1 if none)
    posOf: new Int32Array(area),           // posOf[s] = board index of slot s
  };
}


// V(s) = σ(Σ w_k) = P(BLACK wins)
function evaluate(buf, weightsArr) {
  let z = 0;
  const { idxs, n } = buf;
  for (let i = 0; i < n; i++) z += weightsArr[idxs[i]];
  buf.sum = z;
  buf.val = 1 / (1 + Math.exp(-z));
}

// Δw_k = (lr / n) · (target − buf.val)
function tdUpdate(buf, target, weightsArr, lr) {
  const { idxs, n } = buf;
  if (n === 0) return;
  const step = lr * (target - buf.val) / n;
  for (let i = 0; i < n; i++) weightsArr[idxs[i]] += step;
}

// Evaluate current primary state into feats, update prev2 toward it, rotate buffers.
// Called after each non-terminal move (tree descent or playout).
function tdStep(td) {
  const feats = td.feats;
  feats.n = td.primary.n;
  feats.idxs.set(td.primary.idxs.subarray(0, td.primary.n));
  evaluate(feats, td.weightsArr);
  if (td.prev2.n > 0) tdUpdate(td.prev2, feats.val, td.weightsArr, td.lr);
  // Rotate: feats → prev1, old prev1 → prev2, old prev2 → feats (recycled).
  const recycled = td.prev2;
  td.prev2 = td.prev1;
  td.prev1 = feats;
  td.feats = recycled;
}

// Terminal updates: bring both tracked positions toward the game outcome.
function tdTerminal(td, outcome) {
  if (td.prev2.n > 0) tdUpdate(td.prev2, outcome, td.weightsArr, td.lr);
  if (td.prev1.n > 0) tdUpdate(td.prev1, outcome, td.weightsArr, td.lr);
}

// ── Incremental feature maintenance ──────────────────────────────────────────

function _removeSlot(buf, s, j) {
  const last     = --buf.n;
  buf.slotA[j]   = -1;
  if (s === last) return;
  buf.idxs[s]    = buf.idxs[last];
  const lastJ    = buf.posOf[last];
  buf.posOf[s]   = lastJ;
  buf.slotA[lastJ] = s;
}

// Copy a fully-initialised feature buffer (src) into dst.
// Used to restore the root state at the start of each playout instead of
// re-running findFeaturesInit, which is more expensive (full board scan + IntMap lookups).
function copyBuf(src, dst, area) {
  dst.n = src.n;
  dst.idxs.set(src.idxs.subarray(0, src.n));
  dst.slotA.set(src.slotA.subarray(0, area));
  dst.posOf.set(src.posOf.subarray(0, src.n));
}

// Initialise buf with 1×1 features for the current board position.
// Call once per playout; findFeaturesIncremental maintains it thereafter.
function findFeaturesInit(game, buf, ctx) {
  const area  = game.N * game.N;
  const cells = game.cells;
  buf.n = 0;
  buf.slotA.fill(-1, 0, area);

  for (let idx = 0; idx < area; idx++) {
    const v0 = cells[idx];
    if (v0) {
      const s = buf.n++;
      buf.idxs[s]    = resolveKey(Math.imul(835 + idx, 691 + v0), ctx);
      buf.slotA[idx] = s;
      buf.posOf[s]   = idx;
    }
  }
}

// Update buf incrementally for the move just played via game.play(nextMove).
// Must be called AFTER game.play(nextMove); reads captures from game._lastCaptures.
function findFeaturesIncremental(game, buf, ctx, nextMove) {
  const nCap = game._lastCaptureCount;
  const cap  = game._lastCaptures;

  // Remove features for captured stones (now EMPTY on the board).
  for (let ci = 0; ci < nCap; ci++) {
    const p = cap[ci];
    if (buf.slotA[p] >= 0) _removeSlot(buf, buf.slotA[p], p);
  }

  // Add feature for the new stone (cells already updated by game.play).
  const v0 = game.cells[nextMove];
  if (v0) {
    const s = buf.n++;
    buf.idxs[s]         = resolveKey(Math.imul(835 + nextMove, 691 + v0), ctx);
    buf.slotA[nextMove] = s;
    buf.posOf[s]        = nextMove;
  }
}

// ── Fast playout helpers ──────────────────────────────────────────────────────

// Returns { winner, played }.
function playTracked(game2, node, td, played) {
  const wasAlreadyOver = game2.gameOver;
  const N   = game2.N;
  const cap = N * N;
  // Signed weight per cell: positive = played by BLACK, negative = played by WHITE.
  // Zero means not yet played.  First play on a cell wins; recaptures are ignored.

  // td.primary already reflects the leaf position after selectAndExpand.
  const moveLimit = 3 * game2.emptyCount + 20;
  const weightStep = 1 / cap;
  let moves = 0;
  let weight = 1.0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = game2.randomLegalMove();
    if (idx === PASS) {
      game2.play(PASS);
      moves++;
      if (!game2.gameOver) tdStep(td);
      continue;
    }
    if (played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
    game2.play(idx);
    findFeaturesIncremental(game2, td.primary, td.ctx, idx);
    moves++;
    weight -= weightStep;
    if (!game2.gameOver) tdStep(td);
  }

  let winner;
  if (wasAlreadyOver) {
    winner = game2.calcWinner();
  } else {
    winner = game2.estimateWinner();
  }

  tdTerminal(td, winner === BLACK ? 1 : 0);

  return { winner };
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
// Priors (ladder + pattern) are computed eagerly from the current game state.
function makeNode(move, parent, ci, game2, N) {
  const movesArr = getLegalMoves(game2);
  const M = movesArr.length;
  const area = N * N;

  // Copy into Int32Array for compact, cache-friendly storage.
  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  const children   = new Array(M).fill(null);
  const wins       = new Float32Array(M).fill(0.001);
  const visits     = new Float32Array(M).fill(0.002);
  const raveWins   = new Float32Array(area);
  const raveVisits = new Float32Array(area);

  if (parent === null || parent.parent === null) {
    raveWins.fill(0.001);
    raveVisits.fill(0.002);
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
    totalWins:  0.5,  // sum of wins; incremented each playout
    totalVisits:  1,  // sum of visits; incremented each playout
    selectedChild: -1,  // set by selectAndExpand; read by backpropagate

    legalMoves,   // Int32Array(M)
    children,     // Array(M) — promoted child node or null

    wins,    // Float32Array(M) — playout wins per child
    visits,  // Float32Array(M) — playout visits per child

    raveWins,     // Float32Array(N*N) — RAVE wins indexed by cell; updated by rollouts
    raveVisits,

    tdVals: new Float32Array(M).fill(0.5),  // TD estimate after each child move; updated during search
  };
}

// RAVE-blended UCT score for child index i of node.
// Unvisited children get a large exploration bonus so they are always preferred
// over visited children; RAVE ranks them within the unvisited tier.                                

function ucbScore(moveIdx, node) {
  const move  = node.legalMoves[moveIdx];

  // RAVE
  const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);

  // TD prior — evaluate() returns P(BLACK wins); flip for BLACK-mover nodes
  // since the chooser is the opponent of the mover.
  // TD_WEIGHT scales with node.totalVisits: all children benefit equally as the
  // model trains on simulations through this node.
  const tdVal   = node.tdVals[moveIdx];
  const tdWR    = (node.mover === BLACK) ? 1 - tdVal : tdVal;

  // Real
  const realW = node.wins[moveIdx];
  const realV = node.visits[moveIdx];
  const realWR = realW / realV;

  const raveWeight = RAVE_K / (RAVE_K + realV);

  // Combine win ratios
  const raveRealWR = raveWeight * raveWR + (1 - raveWeight) * realWR;
  const tdRaveRealWR = TD_WEIGHT * tdWR + (1 - TD_WEIGHT) * raveRealWR;

  const scoreBase = tdRaveRealWR + 0.001 * Math.random();
  return scoreBase + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / realV);
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N, td) {
  let node = root;
  const game2 = rootGame2.clone();

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    // Select best child by RAVE-blended score.
    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const s = ucbScore(i, node);
      if (s > bestScore) { bestScore = s; best = i; }
    }

    const move = node.legalMoves[best];
    game2.play(move);
    if (move !== PASS) {
      findFeaturesIncremental(game2, td.primary, td.ctx, move);
    }
    if (!game2.gameOver) {
      tdStep(td);
      node.tdVals[best] = td.prev1.val;
    }

    // Promote child to a full node once it has accumulated enough visits.
    // Fall through to the descent check so the loop continues into the new node;
    // its children all have cv=0 < N_EXPAND, so the leaf case fires next iteration
    // (exactly one makeNode per playout, same as rave2's one-expansion-per-playout).
    if (node.children[best] === null && node.visits[best] >= N_EXPAND) {
      node.children[best] = makeNode(move, node, best, game2, N);
    }

    // After a pass, always force a second pass so the playout scores the current
    // board position (consecutive passes end the game).  This prevents rollouts
    // from a single-pass state playing on for many random moves and inflating
    // the pass move's apparent win rate.
    if (!game2.gameOver && game2.consecutivePasses > 0) {
      game2.play(PASS);
      if (!game2.gameOver) tdStep(td);
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
    node.totalWins += won;
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
    node.parent.totalWins += won;
    node.parent.totalVisits++;
    updateRave(node.parent, won, played, chooser);
    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs, options = {}) {
  if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

  const N          = game.cells ? game.N : game.boardSize;
  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;

  // Model management: restore start-of-game snapshot on move 1; reset on
  // unexpected move count jumps (new game or position was set externally).
  const moveCountDiff = game.moveCount - lastMoveCount;
  if (game.moveCount === 1) {
    keyToIdx   = keyToIdx_start;
    weightsArr = weightsArr_start;
  } else if (moveCountDiff < 0 || moveCountDiff > 2) {
    keyToIdx   = makeIntMap();
    weightsArr = [];
  }
  lastMoveCount = game.moveCount;

  const ctx = { keyToIdx, weightsArr };

  // Obvious pass: opponent just passed and we're already winning — end the game.
  if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
    return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };
  }

  const root = makeNode(null, null, -1, game2, N);

  const area = N * N;
  const rootBuf   = makeBuf(area);
  const tdPrimary = makeBuf(area);
  const tdPrev2   = makeBuf(area);
  const tdPrev1   = makeBuf(area);
  const tdFeats   = makeBuf(area);
  const td = { primary: tdPrimary, prev2: tdPrev2, prev1: tdPrev1, feats: tdFeats,
               ctx, weightsArr, lr: LR1 };
  const played = new Float32Array(area);

  findFeaturesInit(game2, rootBuf, ctx);

  const tStart  = performance.now();
  const deadline = tStart + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;

    // Ramp LR from LR1 down toward 0 over the playout budget.
    const progress = PLAYOUTS > 0 ? playouts / PLAYOUTS
                                  : (performance.now() - tStart) / timeBudgetMs;
    td.lr = LR1 - progress * (LR0 - LR1);

    // Reset TD rotation state for this playout.
    copyBuf(rootBuf, td.primary, area);
    td.prev2 = tdPrev2; tdPrev2.n = 0;
    td.prev1 = tdPrev1; tdPrev1.n = 0;
    td.feats = tdFeats;

    const { node, game2: simGame2 } = selectAndExpand(root, game2, N, td);
    played.fill(0);
    const { winner } = playTracked(simGame2, node, td, played);
    backpropagate(node, winner, played, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  if (game.moveCount === 1) {
    keyToIdx_start   = keyToIdx.clone();
    weightsArr_start = weightsArr.slice();
  }

  // Best child: most playout visits; ties broken by RAVE-blended score.
  const M = root.legalMoves.length;
  let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
  for (let i = 0; i < M; i++) {
    const cv = root.visits[i];
    if (cv > bestVisits || (cv === bestVisits && ucbScore(i, root) > bestScore)) {
      bestVisits = cv;
      bestScore  = ucbScore(i, root);
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
  const rootWinRatio = root.totalVisits > 0 ? totalChildWins / root.totalVisits : 0.5;

  if (root.wins[bestIdx] === 0 && game.moveCount >= N * N / 2) {
    return { type: 'pass', move: PASS, info: 'no winning line found', children, rootWinRatio };
  }

  if (options.polite) {
    const passIdx = root.legalMoves.indexOf(PASS);
    if (passIdx !== -1) {
      const pv = root.visits[passIdx];
      const passWR = root.wins[passIdx] / pv;
      if (passWR > 0.999) {
        bestIdx = passIdx;
      }
    }
  }

  const m = root.legalMoves[bestIdx];
  const cv = root.visits[bestIdx];
  const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

  if (VERBOSE) {
    const topK = 1000;
    const moverStr = rootPlayer === BLACK ? 'B' : 'W';
    console.log(`[rave-td] move=${game.moveCount} player=${moverStr} playouts=${playouts} weights=${weightsArr.length}`);

    // Top children by visit count.
    const ranked = [];
    for (let i = 0; i < M; i++) {
      const mv = root.legalMoves[i];
      const v  = root.visits[i];
      const wr = v > 0 ? root.wins[i] / v : 0;
      const rv = mv !== PASS ? root.raveVisits[mv] : 0;
      const rw = mv !== PASS ? root.raveWins[mv] / (rv || 1) : 0;
      const tdV = root.tdVals[i];
      const tdW = (root.mover === BLACK) ? 1 - tdV : tdV;
      const label = mv === PASS ? 'pass' : `${String.fromCharCode(65 + mv % N)}${(mv / N | 0) + 1}`;
      const ucb = ucbScore(i, root);
      ranked.push({ label, v, wr, rw, tdW, ucb, best: i === bestIdx });
    }
    ranked.sort((a, b) => b.v - a.v);
    for (const r of ranked.slice(0, topK)) {
      const mark = r.best ? '*' : ' ';
      console.log(`${mark} ${r.label.padEnd(4)} visits=${r.v.toFixed(0).padStart(5)}  wr=${r.wr.toFixed(3)}  rave=${r.rw.toFixed(3)}  td=${r.tdW.toFixed(3)}  ucb=${r.ucb.toFixed(3)}`);
    }
  }

  const result = m === PASS ? { type: 'pass', move: PASS, children, rootWinRatio }
                            : { type: 'place', move: m, x: m % N, y: (m / N) | 0, children, rootWinRatio };
  result.info = `value=${(game.current===BLACK?bestWinRatio:(1-bestWinRatio)).toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
