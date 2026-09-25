'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * AMAF (All-Moves-As-First) Monte Carlo policy, with ppat playouts.
 *
 * Runs N flat playouts (PLAYOUTS, else a time budget); each opens with a
 * randomly chosen candidate and then plays out.  Every move made during a
 * playout is credited to the corresponding cell with a linearly decaying weight
 * (1.0 → 0, clamped), so a single playout also updates estimates for the other
 * moves it plays, giving all candidates far more data than mc.js provides.
 *
 * Playouts are uniform below PPAT_MIN_PHASE and follow the ppat policy above it
 * (the same standard playout the mc-ppat / puct-ppat agents run).  The default
 * PPAT_MIN_PHASE is 1, i.e. fully uniform — ppat wins per-playout but not per
 * unit time here, so uniform is the fielded default; lower the knob to enable
 * ppat above a given phase.
 *
 * create(cfg) → { getMove }.  Env (per-slot P<n>_ under selfplay):
 *   PPAT_DATA       ppat weight file (default out/ppat-data-233162-best-ref-candidate.js)
 *   PPAT_MIN_PHASE  uniform playout moves below this board fullness (default 1,
 *                   i.e. fully uniform; lower to enable ppat above that phase)
 *   PLAYOUTS        fixed total playouts per decision (0 = use the time budget)
 *   AMAF_OPP_WEIGHT weight multiplier for opponent moves (default 0.3)
 *   AMAF_DECAY      per-move weight-decay multiplier on 1/area (default 0.5;
 *                   higher = faster decay, 0 = no decay)
 *   AMAF_PRIOR_WEIGHT  prior weight on each cell's ratio denominator (default 0.1;
 *                   shrinks thin cells toward 0 and breaks ties, 0 = off)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;
const { makeRng } = _isNode ? require('../xorshift.js') : window.XorShift;
const { createState, ppatMove, loadWeights } = Util.load('./ppat-lib.js', 'PPatterns');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const PLAYOUTS        = cfg.int('PLAYOUTS', 0);
  // Weight multiplier for opponent moves.  Override with AMAF_OPP_WEIGHT=<n>.
  const AMAF_OPP_WEIGHT = cfg.float('AMAF_OPP_WEIGHT', 0.3);
  // Decay multiplier on the per-move weight step (weight loses AMAF_DECAY/area
  // per move): 1 = weight reaches 0 after one full board of moves (the old
  // fixed 1/area), higher = faster decay / more first-move emphasis, 0 = no
  // decay (every played move keeps weight 1.0).
  const AMAF_DECAY      = cfg.float('AMAF_DECAY', 0.5);
  // Prior WEIGHT added to each cell's win-ratio DENOMINATOR (wins prior 0), so
  // ratio = wins / (AMAF_PRIOR_WEIGHT + plays).  plays is accumulated decayed
  // weight, not a count, so the prior is weight too.  Shrinks thinly-sampled
  // cells toward 0 and breaks ties among equal ratios toward the cell with more
  // accumulated weight (played earlier / more often) rather than at random.  0 = off.
  const AMAF_PRIOR_WEIGHT = cfg.float('AMAF_PRIOR_WEIGHT', 0.1);

  // ppat playout policy.  A hard failure, not a fallback: this agent now runs
  // ppat playouts, so silently reverting to uniform would misrepresent it.
  const _ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
    : null;
  const model = _isNode
    ? loadWeights(_ppatPath)
    : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!model) throw new Error(`amaf: cannot load ppat weights from ${_isNode ? _ppatPath : 'window.PPATWeights'}`);
  model.uniformBelowPhase = cfg.float('PPAT_MIN_PHASE', 1);
  const stateByN = new Map();

  console.log(`amaf[${cfg.slot != null ? cfg.slot : '-'}]: ppat playouts from ` +
    `${_isNode ? require('path').basename(_ppatPath) : 'window.PPATWeights'}, ` +
    `uniform-below ${model.uniformBelowPhase}`);

  // ppat playout.  Returns { winner, played } where played is a Float32Array of
  // length cap: positive value = played by BLACK, negative = played by WHITE,
  // zero = not played.  Weight decreases linearly from 1.0; first play wins.
  function playTracked(game2, rng) {
    const cap        = game2.N * game2.N;
    const played     = new Float32Array(cap);
    const moveLimit  = cap + 20;
    const weightStep = AMAF_DECAY / cap;
    let moves = 0, weight = 1.0;

    let state = stateByN.get(game2.N);
    if (!state) { state = createState(game2.N); stateByN.set(game2.N, state); }

    while (!game2.gameOver && moves < moveLimit) {
      const current = game2.current;
      const idx     = ppatMove(game2, state, model, rng);
      if (idx === PASS) { game2.play(PASS); moves++; continue; }
      if (played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
      game2.play(idx);
      moves++;
      // Clamp at 0: moves played past the linear ramp get zero credit rather
      // than a negative weight (which would flip the credited colour).
      weight = Math.max(0, weight - weightStep);
    }

    return { winner: game2.estimateWinner(), played };
  }

  function getMove(game, timeBudgetMs, options = {}) {
    if (game.gameOver) return { type: 'pass', move: PASS };

    const rng = options.rng || makeRng();
    const playoutLimit = options.playoutLimit || PLAYOUTS;

    const game2  = game.cells ? game.clone() : game.toGame2();
    const player = game2.current;
    const N      = game2.N;
    const cap    = N * N;

    // Build list of legal candidate moves (flat cell indices; pass at cap).
    const candidates = [];
    const cells = game2.cells;
    for (let i = 0; i < cap; i++) {
      if (game2.isLegal(i) && !game2.isTrueEye(i)) candidates.push(i);
    }
    if (game2.moveCount >= cap / 2 || game2.consecutivePasses > 0) {
      candidates.push(PASS);
    }
    Util.shuffle(candidates, rng);

    // AMAF stats indexed by cell (0..N*N-1); pass stored at N*N.
    const wins  = new Float32Array(cap + 1);
    const plays = new Float32Array(cap + 1);
    const PASS_IDX = cap;
    // Seed the denominator with AMAF_PRIOR_WEIGHT of prior weight on the board
    // cells (not pass, whose selection stays gated on real credit): shrinks each
    // ratio toward 0 and breaks equal-ratio ties toward the higher-weight cell.
    if (AMAF_PRIOR_WEIGHT > 0) plays.fill(AMAF_PRIOR_WEIGHT, 0, cap);

    // N flat playouts (no rounds): each opens with a randomly chosen candidate,
    // then plays out.  The opening gets the direct credit and every playout move
    // feeds the all-moves-as-first credit; over the run each candidate is opened
    // ~playoutLimit / candidates.length times in expectation.  playoutLimit > 0
    // fixes the count; otherwise run until the time budget.
    const deadline    = performance.now() + timeBudgetMs;
    const playerSign  = player === BLACK ? 1 : -1;
    const nCandidates = candidates.length;
    let playoutCount = 0;
    while (playoutLimit > 0 ? playoutCount < playoutLimit : performance.now() < deadline) {
      playoutCount++;
      const move  = candidates[(rng.random() * nCandidates) | 0];
      const clone = game2.clone();
      clone.play(move);

      const { winner, played } = playTracked(clone, rng);
      const won = winner === player ? 1 : 0;

      // Credit the opening move at full weight (it was played "first").
      const firstIdx = move === PASS ? PASS_IDX : move;
      plays[firstIdx] += 1.0;
      wins[firstIdx]  += won;

      // Credit playout moves using signed weights from played[].
      for (let k = 0; k < cap; k++) {
        const w = played[k];
        if (w === 0) continue;
        if (w * playerSign > 0) {
          const wt = Math.abs(w);
          plays[k] += wt;
          wins[k]  += won * wt;
        } else if (AMAF_OPP_WEIGHT > 0) {
          const wt = Math.abs(w) * AMAF_OPP_WEIGHT;
          plays[k] += wt;
          wins[k]  += (1 - won) * wt;
        }
      }
    }

    // Select the candidate with the highest AMAF win ratio; ties broken randomly.
    let bestRatio = -1;
    let bestCount = 0;
    let bestIdx   = 0;
    for (let i = 0; i < candidates.length; i++) {
      const idx   = candidates[i] === PASS ? PASS_IDX : candidates[i];
      if (plays[idx] === 0) continue;
      const ratio = wins[idx] / plays[idx];
      if (ratio > bestRatio) {
        bestRatio = ratio;
        bestIdx   = i;
        bestCount = 1;
      } else if (ratio === bestRatio) {
        bestCount++;
        if (rng.random() * bestCount < 1) bestIdx = i;
      }
    }

    // If every playout is a loss, pass — no move can help.
    if (bestRatio === 0) return { type: 'pass', move: PASS };

    // Prefer pass when it ties for best ratio.
    if (plays[PASS_IDX] > 0 && wins[PASS_IDX] / plays[PASS_IDX] === bestRatio) {
      const passCandIdx = candidates.indexOf(PASS);
      if (passCandIdx !== -1) bestIdx = passCandIdx;
    }

    // Before committing to a pass, verify that the position is actually won.
    // If calcWinner disagrees with playout winners, fall back to the best
    // non-pass candidate.
    if (candidates[bestIdx] === PASS && plays[PASS_IDX] > 0) {
      const actualWinner = game2.calcWinner();
      const allPlayoutsAgree = actualWinner === player
        ? wins[PASS_IDX] === plays[PASS_IDX]
        : true;
      if (!allPlayoutsAgree) {
        let altRatio = -1;
        let altCount = 0;
        let altIdx   = -1;
        for (let i = 0; i < candidates.length; i++) {
          if (candidates[i] === PASS) continue;
          const idx = candidates[i];
          if (plays[idx] === 0) continue;
          const ratio = wins[idx] / plays[idx];
          if (ratio > altRatio) {
            altRatio = ratio;
            altIdx   = i;
            altCount = 1;
          } else if (ratio === altRatio) {
            altCount++;
            if (rng.random() * altCount < 1) altIdx = i;
          }
        }
        if (altIdx !== -1) bestIdx = altIdx;
      }
    }

    const best = candidates[bestIdx];
    return best === PASS
      ? { type: 'pass', move: PASS }
      : { type: 'place', move: best, x: best % N, y: (best / N) | 0 };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
const _api = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };

if (typeof module !== 'undefined') module.exports = _api;
else window.Amaf = _api;

})();
