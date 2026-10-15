'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// puct-vnet-wasm: puct-ppat-fp's search with the value network (cnn.h) in
// place of playouts, the net run in WebAssembly — vnet-eval.wasm via
// vnet-wasm.js (c/wasm_vnet.c).
//
// PUCT MCTS with featurepol priors and top-K candidate pruning at every node
// including the root (TOP_K_A at phase 0 up to TOP_K_B at phase 1).  Each
// simulation descends by
//   score = Q + C_PUCT · P(s,a) · √N_total / (1 + N_a)
// (Q the edge's mean backpropagated value, P featurepol's softmax probability)
// to an unexpanded edge, expands it — one new node, featurepol extraction and
// priors — and values the new node's position ONCE with the net, P(side to
// move wins) mapped to P(BLACK wins); that value is backpropagated as a
// playout's result would be (each chooser credited value or 1 − value).  No
// RAVE: it needs playout traces.  Terminal positions (a double pass, or a
// descent into a finished game) are scored exactly.  The move played is the
// most visited root edge.
//
// Config (cfg reader, slot-aware):
//   C_PUCT      exploration weight of the prior                    (default 0.1)
//   TOP_K_A     kept moves at phase 0 (0 with TOP_K_B 0 = full width) (default 10)
//   TOP_K_B     kept moves at phase 1                              (default 40)
//   PLAYOUTS    simulations (= net evaluations) per decision; 0 = the time
//               budget                                            (default 0)
//   FPOL_DATA   featurepol weights (priors, top-K)      (default ref/ref-fp-fast.js)
//   VN_MODEL    the net, a VNET1 file   (default out/vnet-rfs-ptw1000-lad-6x32-best.bin)
// In the browser the page loads the pieces first: window.vnetWasm = await
// VnetWasm.load(url), window.vnetModel = the VNET1 file's bytes (Uint8Array),
// and window.featurepolModel.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const FeaturePol            = Util.load('./featurepol-lib.js', 'FeaturePol');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const VnetWasm              = Util.load('./vnet-wasm.js', 'VnetWasm');

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;
const RESIGN_MIN_PLAYOUTS = 20000;

function create(cfg) {
  cfg = cfg || Util.makeCfg();
  const tag = `puct-vnet-wasm[${cfg.slot != null ? cfg.slot : '-'}]`;

  const C_PUCT   = cfg.float('C_PUCT', 0.1);
  const TOP_K_A  = cfg.int('TOP_K_A', 10);
  const TOP_K_B  = cfg.int('TOP_K_B', 40);
  const PLAYOUTS = cfg.int('PLAYOUTS', 0);

  const fpModel   = FeaturePol.loadModel({ name: 'puct-vnet-wasm',
    path: _isNode ? cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'ref', 'ref-fp-fast.js')) : undefined });
  const fpWeights = fpModel.weights;
  // Rank the vpat<n> feature over the best 3 candidates when the fp spec ranks
  // (as puct-ppat-fp); no-op for specs without it.
  if (fpWeights.spec.rankSpaces && fpWeights.spec.rankSpaces.length > 0) fpWeights.rankTopN = 3;

  const modelPath  = _isNode ? cfg.str('VN_MODEL', require('path').join(__dirname, '..', 'out', 'vnet-rfs-ptw1000-lad-6x32-best.bin')) : null;
  const modelBytes = _isNode ? new Uint8Array(require('fs').readFileSync(modelPath))
                             : (typeof window !== 'undefined' && window.vnetModel) || null;
  if (!modelBytes) throw new Error(`${tag}: window.vnetModel is not set — the VNET1 file's bytes (Uint8Array)`);
  const vw = _isNode ? VnetWasm.loadSync() : (typeof window !== 'undefined' && window.vnetWasm) || null;
  if (!vw) throw new Error(`${tag}: window.vnetWasm is not set — load it first: window.vnetWasm = await VnetWasm.load(url)`);
  const nParams = vw.setModel(modelBytes);
  let vwN = 0;   // board size the module is initialised for

  console.error(`${tag}: C_PUCT ${C_PUCT}, top-K ${TOP_K_A}->${TOP_K_B}, ` +
                `${PLAYOUTS > 0 ? `${PLAYOUTS} simulations` : 'time budget'}, ` +
                `fp: ${fpModel.modelName} (${fpWeights.size} weights${fpWeights.rankTopN > 0 ? `, rank-topn ${fpWeights.rankTopN}` : ''}), ` +
                `vnet: ${_isNode ? require('path').basename(modelPath) : 'window.vnetModel'} (${nParams} parameters)`);

  const stateByN = new Map();

  // Featurepol extraction + softmax for game2 (game3: its lockstep mirror).
  // Returns the shared state (moves / probs) or null.
  function runFp(game2, game3) {
    if (game2.gameOver) return null;
    const N = game2.N;
    let state = stateByN.get(N);
    if (!state) { state = FeaturePol.createState(N, fpWeights.spec); stateByN.set(N, state); }
    FeaturePol.extractFeatures(game2, state, fpWeights, game3);
    if (state.count === 0) return null;
    FeaturePol.computeSoftmax(state, fpWeights);
    return state;
  }

  // The net's P(BLACK wins) of a position that is not over.
  function netValueB(game2) {
    if (vwN !== game2.N) { vw.init(game2.N); vwN = game2.N; }
    vw.setRoot(game2);
    const v = vw.evalRoot();
    return game2.current === BLACK ? v : 1 - v;
  }

  function makeNode(move, parent, ci, game2, N, game3) {
    const fpState = runFp(game2, game3);
    let movesArr = getLegalMoves(game2);
    const k = Math.round(TOP_K_A + (TOP_K_B - TOP_K_A) * game2.phase());
    if (k > 0) movesArr = pruneToTopK(movesArr, fpState, k, N);
    const M = movesArr.length, area = N * N;
    const legalMoves = Int32Array.from(movesArr);

    // PUCT priors per move (renormalised to sum 1); PASS and moves featurepol
    // does not list get a 1/area floor.
    const priors = new Float32Array(M);
    if (fpState) {
      const probByMove = new Float64Array(area);
      for (let i = 0; i < fpState.count; i++) probByMove[fpState.moves[i]] = fpState.probs[i];
      const floor = 1 / area;
      let sum = 0;
      for (let i = 0; i < M; i++) {
        const m = legalMoves[i];
        priors[i] = m === PASS ? floor : (probByMove[m] || floor);
        sum += priors[i];
      }
      if (sum > 0) for (let i = 0; i < M; i++) priors[i] /= sum;
    } else {
      for (let i = 0; i < M; i++) priors[i] = 1 / M;
    }

    return {
      move, parent, ci,
      mover: -game2.current,
      totalVisits: 0.1,
      selectedChild: -1,
      legalMoves,
      children: new Array(M).fill(null),
      priors,
      wins:   new Float32Array(M).fill(PRIOR_WINS),
      visits: new Float32Array(M).fill(PRIOR_VISITS),
    };
  }

  function nodeScore(i, node, rng) {
    const Q = node.wins[i] / node.visits[i];
    const U = C_PUCT * node.priors[i] * Math.sqrt(node.totalVisits) / (1 + node.visits[i]);
    return Q + U + 0.001 * rng.random();
  }

  // Descend from the root to an unexpanded edge (expanding it) or a terminal
  // position.  game3 mirrors the descent; `depth` plays to undo afterwards.
  // `evaluate`: the new node's position needs a net value.
  function selectAndExpand(root, rootGame2, N, rng, game3) {
    let node = root;
    const game2 = rootGame2.clone();
    let depth = 0, evaluate = false;
    while (!game2.gameOver) {
      const M = node.legalMoves.length;
      if (M === 0) break;
      let best = 0, bestScore = -Infinity;
      for (let i = 0; i < M; i++) {
        const s = nodeScore(i, node, rng);
        if (s > bestScore) { bestScore = s; best = i; }
      }
      const move = node.legalMoves[best];
      game2.play(move);
      game3.play(move);
      depth++;
      // A pass answered by a pass ends the game: scored exactly.
      if (!game2.gameOver && game2.consecutivePasses > 0) {
        game2.play(PASS);
        game3.play(PASS);
        depth++;
        node.selectedChild = best;
        break;
      }
      if (node.children[best] === null) {
        if (game2.gameOver) { node.selectedChild = best; break; }
        node.children[best] = makeNode(move, node, best, game2, N, game3);
        node = node.children[best];
        node.selectedChild = -1;
        evaluate = true;
        break;
      }
      node = node.children[best];
      node.selectedChild = -1;
    }
    return { node, game2, depth, evaluate };
  }

  function backpropagate(node, value) {
    const leafIdx = node.selectedChild;
    if (leafIdx !== -1) {
      const chooser = -node.mover;
      node.visits[leafIdx]++;
      node.wins[leafIdx] += chooser === BLACK ? value : 1 - value;
      node.totalVisits++;
    }
    while (node.parent !== null) {
      const chooser = -node.parent.mover;
      node.parent.visits[node.ci]++;
      node.parent.wins[node.ci] += chooser === BLACK ? value : 1 - value;
      node.parent.totalVisits++;
      node = node.parent;
    }
  }

  // The search from game2; returns the populated root and the simulation count.
  function runSearch(game2, N, rng, playoutLimit, timeBudgetMs) {
    const game3 = game3FromGame2(game2);
    const root = makeNode(null, null, -1, game2, N, game3);
    const deadline = performance.now() + timeBudgetMs;
    let playouts = 0;
    do {
      playouts++;
      const { node, game2: g, depth, evaluate } = selectAndExpand(root, game2, N, rng, game3);
      const value = evaluate ? netValueB(g) : (g.calcWinner() === BLACK ? 1 : 0);
      for (let i = 0; i < depth; i++) game3.undo();
      backpropagate(node, value);
    } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);
    return { root, playouts };
  }

  function getMove(game, timeBudgetMs, options = {}) {
    if (game.gameOver) return { move: PASS, info: 'game already over' };
    const N = game.N, game2 = game.clone();
    if (game2.consecutivePasses > 0 && game2.calcWinner() === game2.current)
      return { move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };

    const rng = options.rng || makeRng();
    const playoutLimit = options.playoutLimit || PLAYOUTS;
    const { root, playouts } = runSearch(game2, N, rng, playoutLimit, timeBudgetMs);

    const M = root.legalMoves.length;
    let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const cv = root.visits[i];
      if (cv > bestVisits || (cv === bestVisits && nodeScore(i, root, rng) > bestScore)) {
        bestVisits = cv;
        bestScore  = nodeScore(i, root, rng);
        bestIdx    = i;
      }
    }
    const children = [];
    for (let i = 0; i < M; i++) children.push({ move: root.legalMoves[i], visits: root.visits[i], wins: root.wins[i] });
    children.sort((a, b) => b.visits - a.visits);
    let totalChildWins = 0;
    for (let i = 0; i < M; i++) totalChildWins += root.wins[i];
    const rootWinRatio = totalChildWins / root.totalVisits;   // P(side to move wins)

    if (playouts >= RESIGN_MIN_PLAYOUTS && game2.emptyCount <= N * N / 2 && root.wins[bestIdx] <= PRIOR_WINS)
      return { move: PASS, info: 'no winning line found', children, rootWinRatio };

    const cv = root.visits[bestIdx];
    const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;
    return { move: root.legalMoves[bestIdx], children, rootWinRatio,
             info: `value=${(game.current === BLACK ? bestWinRatio : 1 - bestWinRatio).toFixed(3)} sims=${playouts}` };
  }

  // P(BLACK wins) from a search (as puct-ppat-fp): PLAYOUTS, else
  // options.budgetMs, else 1000 simulations; the root's win ratio.
  function valueB(game, options = {}) {
    const N = game.N, game2 = game.clone();
    if (game2.gameOver) return game2.calcWinner() === BLACK ? 1 : 0;
    const r = options.rng || makeRng();
    const budgetMs = options.budgetMs > 0 ? options.budgetMs : 0;
    const playoutLimit = PLAYOUTS > 0 ? PLAYOUTS : (budgetMs > 0 ? 0 : 1000);
    const { root } = runSearch(game2, N, r, playoutLimit, budgetMs);
    let totalChildWins = 0;
    for (let i = 0; i < root.legalMoves.length; i++) totalChildWins += root.wins[i];
    const rootWinRatio = totalChildWins / root.totalVisits;
    return game2.current === BLACK ? rootWinRatio : 1 - rootWinRatio;
  }

  return { getMove, valueB };
}

function getLegalMoves(game2) {
  const cap = game2.N * game2.N, cells = game2.cells, moves = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0 || game2.isTrueEye(i)) continue;
    if (game2.isLegal(i)) moves.push(i);
  }
  if (moves.length < cap / 3 || game2.consecutivePasses > 0) moves.push(PASS);
  return moves;
}

// The K placements featurepol rates most probable; PASS, when listed, kept.
function pruneToTopK(allMoves, state, K, N) {
  if (K <= 0 || !state) return allMoves;
  const probByMove = new Float64Array(N * N);
  for (let i = 0; i < state.count; i++) probByMove[state.moves[i]] = state.probs[i];
  const placements = [];
  let hasPass = false;
  for (const m of allMoves) { if (m === PASS) hasPass = true; else placements.push(m); }
  placements.sort((a, b) => probByMove[b] - probByMove[a]);
  const top = placements.slice(0, K);
  if (hasPass) top.push(PASS);
  return top;
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') {
  module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
} else {
  window.create  = create;
  window.getMove = (g, b, o) => _def().getMove(g, b, o);
  window.valueB  = (g, o)    => _def().valueB(g, o);
}

})();
