'use strict';

// mc-ppat: a deliberately minimal agent whose ONLY decision input is the
// ppat playout policy — built to measure playout quality directly.
//
// Each move: sample CANDIDATES legal non-eye moves uniformly at random, split
// the PLAYOUTS budget evenly between them, and play the one with the best mean
// result (from the mover's perspective).  No tree, no priors, no RAVE.
// PLAYOUTS is the TOTAL per decision, as in every other agent, so changing
// CANDIDATES redistributes a fixed budget rather than inflating it.  With
// FP_TOP > 0 the uniform sampler is replaced by featurepol pruning (the
// cascade stage-1 rule: top moves by raw fp score) — no longer a pure
// playout-quality probe, but a cheap strong configuration.
//
// Why this rather than puct-ppat-fp: in the puct agents the featurepol priors
// and the tree make most of the decision, so swapping the ppat model barely
// moves the result — two different ppat checkpoints once produced identical
// movedetails output down to the same worst-case move choices.  Here the ppat
// policy decides every move, so an A/B between two models measures the models.
//
// Cost is fixed per decision (PLAYOUTS) and independent of the time budget,
// which is ignored.  Sized for fast head-to-heads on small boards.
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (slot-aware env),
// so P1_/P2_ prefixes select a different model per side in selfplay:
//   P1_PPAT_DATA=a.js P2_PPAT_DATA=b.js node selfplay.js --p1 mc-ppat --p2 mc-ppat
//
// Config:
//   CANDIDATES    moves sampled per decision; 0 = all legal moves (default 0)
//   PLAYOUTS      TOTAL playouts per decision, split evenly       (default 1)
//   CAND_PLAYOUTS playouts PER CANDIDATE; overrides PLAYOUTS when > 0 (default 0)
//                 The two conventions allocate to opposite ends of the game: a
//                 fixed total gives each candidate more playouts as the move
//                 list shrinks (endgame-weighted), while a fixed per-candidate
//                 count spends more total playouts while the list is long
//                 (midgame-weighted).
//   FP_TOP        featurepol pruning: candidates are fp's top-N moves by raw
//                 score instead of a uniform sample; overrides CANDIDATES
//                 (default 0 = off, uniform sampling)
//   FPOL_DATA     featurepol weights for FP_TOP        (default featurepol-cbk7wa32.js)
//   SIM_PHASE     below this board fullness, skip the sims and play fp's
//                 top-scored move outright (the cascade rule: early-game
//                 sims are long and noisy, so the budget is saved for the
//                 phase where it discriminates).  Requires FP_TOP > 0.
//                 (default 0 = sims at every phase)
//   FP_GAP_SKIP   when fp's raw-score gap between its #1 and #2 exceeds
//                 this, play #1 without any sims (the decision is not
//                 contested; the budget is saved for moves that are).
//                 Requires FP_TOP > 0.  (default 0 = off)
//   FP_GAP_SKIP_MAX_PHASE  the gap skip only applies below this board
//                 fullness (measured: fp's big-gap calls are reliable early
//                 but materially wrong ~1-in-6 late, so late moves always
//                 get their vote).  (default 1 = skip at every phase)
//   VOTE_BLOCK    sequential voting: sims run in blocks of this many per
//                 candidate, stopping early once the leader's margin over
//                 the runner-up exceeds VOTE_Z standard errors (or the
//                 CAND_PLAYOUTS/PLAYOUTS cap is reached).  Decided moves
//                 stop after one block; contested moves run to the cap.
//                 (default 0 = single fixed-depth vote)
//   VOTE_Z        the stopping margin in standard errors (default 2)
//   VPAT_PICK     1 = below SIM_PHASE, instead of fp's top move, choose
//                 among the FP_TOP candidates by the vpat evaluator's value
//                 of the successor position (1-ply pick; the ref-search
//                 pattern).  Needs FP_TOP > 0 and TRUNC_VPAT_DATA.
//                 SIM_PHASE=1 with VPAT_PICK=1 is a pure 1-ply agent.
//                 (default 0)
//   PPAT_DATA     ppat weight file                 (default out/ppat-data-233162-best-ref-candidate.js)
//   PPAT_MIN_PHASE  uniform playout moves below this board fullness
//                 (default 0.6, matching the standard playout)
//   TRUNC_VPAT_DATA   vpatterns evaluator for the truncation point.  Its
//                 PRESENCE turns truncation ON (absent = plain full playouts):
//                 a playout whose START phase ph satisfies ph + delta <=
//                 TRUNC_MAX_PHASE_B stops after the board fullness has gained
//                 delta and returns the vpat value at the truncation point
//                 instead of playing out (other playouts run to the end).
//   TRUNC_PHASE_DELTA  truncation delta; overrides the model file's baked
//                 'trunc.delta'.  0 is valid — truncate at the leaf, i.e. a
//                 pure static vpat evaluation.  With TRUNC_VPAT_DATA set and no
//                 delta from either source the agent throws.
//   TRUNC_MAX_PHASE_B gate bound on the truncated ENDPOINT (default 0.55)
//   TRUNC_VALUE_OFFSET "a,b": measured lean correction, applied as a logit
//                 shift 4*(a + b*ph) at the endpoint phase; overrides the
//                 model file's baked 'trunc.offset' (default: model's, else
//                 0,0 — offsets are per (model, delta, band) and never transfer)

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK, EMPTY } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const PPat = require('../ppat-lib.js');
const VPat = require('../vpatterns.js');
const FeaturePol = require('../featurepol-lib.js');
const { game3FromGame2 } = require('../game3.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const CANDIDATES = Math.max(0, cfg.int('CANDIDATES', 0));   // 0 = all legal non-eye moves
  const PLAYOUTS   = Math.max(1, cfg.int('PLAYOUTS', 1));
  const CAND_PLAYOUTS = Math.max(0, cfg.int('CAND_PLAYOUTS', 0));

  const ppatPath = cfg.str('PPAT_DATA', path.join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'));
  const model    = PPat.loadWeights(ppatPath);
  // Hard failure, not a fallback: this agent exists to measure a ppat model, so
  // silently running uniform playouts would produce a meaningless comparison.
  if (!model) throw new Error(`mc-ppat: cannot load ppat weights from ${ppatPath}`);
  model.uniformBelowPhase = cfg.float('PPAT_MIN_PHASE', 0.6);

  // Optional featurepol pruning (off unless FP_TOP > 0).
  const FP_TOP = Math.max(0, cfg.int('FP_TOP', 0));
  const SIM_PHASE = cfg.float('SIM_PHASE', 0);
  if (SIM_PHASE > 0 && FP_TOP === 0) {
    throw new Error('mc-ppat: SIM_PHASE needs FP_TOP > 0 — below the gate there is no cheap pick to play');
  }
  const VPAT_PICK = cfg.int('VPAT_PICK', 0);
  const FP_GAP_SKIP = cfg.float('FP_GAP_SKIP', 0);
  const FP_GAP_SKIP_MAX_PHASE = cfg.float('FP_GAP_SKIP_MAX_PHASE', 1);
  const VOTE_BLOCK  = Math.max(0, cfg.int('VOTE_BLOCK', 0));
  const VOTE_Z      = cfg.float('VOTE_Z', 2);
  if (FP_GAP_SKIP > 0 && FP_TOP === 0) {
    throw new Error('mc-ppat: FP_GAP_SKIP needs FP_TOP > 0 — the gap is between fp\'s ranked candidates');
  }
  if (VPAT_PICK && FP_TOP === 0) {
    throw new Error('mc-ppat: VPAT_PICK needs FP_TOP > 0 — it chooses among the fp candidates');
  }
  let fpWeights = null, fpState = null, fpScores = null;
  if (FP_TOP > 0) {
    fpWeights = FeaturePol.loadModel({ name: 'mc-ppat',
      path: cfg.str('FPOL_DATA', path.join(__dirname, '..', 'featurepol-cbk7wa32.js')) }).weights;
  }

  // Truncation is ON when a vpat evaluator is specified (TRUNC_VPAT_DATA);
  // absent = plain full playouts.  delta and offset come from the env vars,
  // else the model file's baked 'trunc' block (train-vpat-playout-eval writes
  // it).  VPAT_PICK also uses the evaluator, so it requires TRUNC_VPAT_DATA too.
  const TRUNC_VPAT = cfg.str('TRUNC_VPAT_DATA', '');
  const TRUNC_ON   = TRUNC_VPAT !== '';
  const TRUNC_B    = cfg.float('TRUNC_MAX_PHASE_B', 0.55);
  let vpatModel = null, TRUNC_DELTA = 0, VO_A = 0, VO_B = 0, _deltaSrc = '', _voSrc = '';
  if (TRUNC_ON || VPAT_PICK) {
    if (VPAT_PICK && !TRUNC_ON) {
      throw new Error(`mc-ppat[${cfg.slot != null ? cfg.slot : '-'}]: VPAT_PICK needs TRUNC_VPAT_DATA (the evaluator it picks with)`);
    }
    vpatModel = VPat.loadWeights(TRUNC_VPAT, cfg.str('HEALTH_DATA', ''));   // throws if unset/unloadable — no silent full-playout fallback
    const tm = vpatModel.trunc || {};
    // delta: env override, else the model's baked delta.  delta 0 is valid
    // (truncate at the leaf — a pure static vpat evaluation); only a delta from
    // NEITHER source is an error.
    let deltaResolved = false;
    if (cfg.has('TRUNC_PHASE_DELTA')) { TRUNC_DELTA = cfg.float('TRUNC_PHASE_DELTA', 0); deltaResolved = true; _deltaSrc = 'env'; }
    else if (tm.delta != null)        { TRUNC_DELTA = tm.delta;                          deltaResolved = true; _deltaSrc = 'model'; }
    if (!deltaResolved) {
      throw new Error(`mc-ppat[${cfg.slot != null ? cfg.slot : '-'}]: TRUNC_VPAT_DATA is set (truncation on) but no ` +
        `delta — pass TRUNC_PHASE_DELTA or use a model with a baked trunc block`);
    }
    if (TRUNC_DELTA < 0) {
      throw new Error(`mc-ppat[${cfg.slot != null ? cfg.slot : '-'}]: TRUNC_PHASE_DELTA must be >= 0 (got ${TRUNC_DELTA})`);
    }
    // offset: env override, else the model's baked offset, else none.
    if (cfg.has('TRUNC_VALUE_OFFSET')) {
      const vo = cfg.str('TRUNC_VALUE_OFFSET', '0,0').split(',').map(parseFloat); VO_A = vo[0]; VO_B = vo[1] || 0; _voSrc = 'env';
    } else if (Array.isArray(tm.offset)) {
      VO_A = tm.offset[0]; VO_B = tm.offset[1] || 0; _voSrc = 'model';
    }
  }
  console.log(`mc-ppat[${cfg.slot != null ? cfg.slot : '-'}]: ${model.weights.length} ppat weights ` +
              `from ${path.basename(ppatPath)}, ` +
              (CAND_PLAYOUTS > 0 ? `${CAND_PLAYOUTS} playouts/candidate` : `${PLAYOUTS} playouts/move`) +
              (FP_TOP > 0 ? ` over fp top-${FP_TOP}` : ` over ${CANDIDATES > 0 ? CANDIDATES : 'all'} candidates`) +
              (SIM_PHASE > 0 ? `  sims from phase ${SIM_PHASE}` : '') +
              (VPAT_PICK ? `  vpat-pick below the gate` : '') +
              (FP_GAP_SKIP > 0 ? `  gap-skip>${FP_GAP_SKIP}` : '') +
              (VOTE_BLOCK > 0 ? `  seq-vote block=${VOTE_BLOCK} z=${VOTE_Z}` : '') +
              (TRUNC_ON ? `  trunc: delta=${TRUNC_DELTA}${_deltaSrc === 'model' ? '(model)' : ''} B=${TRUNC_B} ` +
                          `offset=${VO_A},${VO_B}${_voSrc === 'model' ? '(model)' : ''}` : ''));

  const rng = makeRng();
  let ppatState = null;

  // vpat value at a truncation point, offset-corrected: P(BLACK wins).
  function vpatValueB(g) {
    const f = VPat.extractFeatures(g, vpatModel.preparedSpecs);
    VPat.evaluateFeatures(f, vpatModel.weights);
    const ph = 1 - g.emptyCount / (g.N * g.N);
    return 1 / (1 + Math.exp(-(f.z - 4 * (VO_A + VO_B * ph))));
  }

  // ppat playout (mutates game2).  Returns P(BLACK wins): 1/0 from the
  // terminal position, or — when truncation is on and this playout's start
  // qualifies (start + delta <= B) — the vpat value at the truncation point.
  function playout(game2, r) {
    if (ppatState === null || ppatState.moves.length < game2.N * game2.N)
      ppatState = PPat.createState(game2.N);
    const area = game2.N * game2.N;
    // Prefix length in MOVES, not a fullness check repeated per move: delta is
    // a fullness fraction only so one number carries across board sizes, and a
    // per-move check descends further whenever the prefix captures.  Matches
    // ai/puct-ppat-fp-trunc.js and the offline prefix generators.
    let prefixLen = -1;
    if (TRUNC_ON && (1 - game2.emptyCount / area) + TRUNC_DELTA <= TRUNC_B) {
      prefixLen = Math.ceil(TRUNC_DELTA * area);   // 0 when delta == 0 → static eval at the leaf
    }
    const moveLimit = 3 * game2.emptyCount + 20;
    let moves = 0;
    while (!game2.gameOver && moves < moveLimit) {
      if (prefixLen >= 0 && moves >= prefixLen) return vpatValueB(game2);
      game2.play(PPat.ppatMove(game2, ppatState, model, r));
      moves++;
    }
    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  // Reservoir-sample k legal non-eye moves without building the full list:
  // one pass over the empty cells, O(area) with no allocation beyond `out`.
  function sampleMoves(game, k, out, r) {
    const area = game.N * game.N;
    let seen = 0;
    for (let i = 0; i < area; i++) {
      if (game.cells[i] !== EMPTY || game.isTrueEye(i) || !game.isLegal(i)) continue;
      if (seen < k) out[seen] = i;
      else { const j = r.int(seen + 1); if (j < k) out[j] = i; }
      seen++;
    }
    return seen < k ? seen : k;
  }

  let _cand = new Int32Array(64);

  // Featurepol pruning (the cascade stage-1 rule): fp's top FP_TOP moves by
  // raw linear score, written into `out` best-first.  Returns the candidate
  // count; _fpGap holds the #1-vs-#2 raw-score gap (Infinity when only one).
  let _fpGap = 0;
  function fpCandidates(game, out) {
    const N = game.N;
    if (!fpState || fpState.moves.length < N * N) {
      fpState  = FeaturePol.createState(N, fpWeights.spec);
      fpScores = new Float64Array(N * N + 1);
    }
    const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
    FeaturePol.extractFeatures(game, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    if (n === 0) return 0;
    const k = Math.min(FP_TOP, n);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => fpScores[b] - fpScores[a]);
    _fpGap = n > 1 ? fpScores[order[0]] - fpScores[order[1]] : Infinity;
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return k;
  }

  // options.rng (when provided) drives candidate sampling, playouts and the
  // dither, making the decision reproducible (e.g. evalmovedetails' per-
  // position seed schedule); otherwise the unseeded closure rng is used.
  function getMove(game, timeBudgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const r = options.rng || rng;
    const want = FP_TOP > 0 ? FP_TOP : (CANDIDATES > 0 ? CANDIDATES : game.N * game.N);
    if (_cand.length < want) _cand = new Int32Array(want);
    const n = FP_TOP > 0 ? fpCandidates(game, _cand) : sampleMoves(game, want, _cand, r);
    if (n === 0) return { move: PASS };
    // Below SIM_PHASE the sims don't discriminate (long, noisy playouts vs
    // tiny value gaps): play fp's top-scored candidate outright — or, with
    // VPAT_PICK, the candidate whose successor the vpat evaluator likes best
    // (constant lean is common-mode across same-position candidates).
    if (SIM_PHASE > 0 && game.phase() < SIM_PHASE) {
      if (!VPAT_PICK || n === 1) return { move: _cand[0], info: 'fp-top (below sim-phase)' };
      const mover = game.current;
      let best = _cand[0], bestV = -Infinity;
      for (let c = 0; c < n; c++) {
        const g = game.clone();
        g.play(_cand[c]);
        const v = vpatValueB(g);
        const mv2 = mover === BLACK ? v : 1 - v;
        if (mv2 > bestV) { bestV = mv2; best = _cand[c]; }
      }
      return { move: best, info: `vpat-pick=${bestV.toFixed(3)}` };
    }

    // Uncontested by fp's own scores: play #1, save the sims for moves
    // that are contested.
    if (FP_GAP_SKIP > 0 && _fpGap > FP_GAP_SKIP && game.phase() < FP_GAP_SKIP_MAX_PHASE) {
      return { move: _cand[0], info: `fp-gap=${_fpGap.toFixed(2)}` };
    }

    // CAND_PLAYOUTS fixes the per-candidate count (total then scales with n);
    // otherwise split the total budget evenly across the candidates actually
    // found (n can be < CANDIDATES late), keeping per-decision cost at PLAYOUTS.
    const per = CAND_PLAYOUTS > 0 ? CAND_PLAYOUTS : Math.max(1, Math.round(PLAYOUTS / n));
    const mover = game.current;

    // Sequential vote: blocks of VOTE_BLOCK sims per candidate, stopping as
    // soon as the leader's mean beats the runner-up's by VOTE_Z standard
    // errors.  Decided moves cost one block; contested moves run to `per`.
    if (VOTE_BLOCK > 0 && n > 1) {
      const sum = new Float64Array(n), sq = new Float64Array(n);
      let done = 0;
      while (done < per) {
        const b = Math.min(VOTE_BLOCK, per - done);
        for (let c = 0; c < n; c++) {
          for (let p = 0; p < b; p++) {
            const g = game.clone();
            g.play(_cand[c]);
            const res = playout(g, r);
            const v = mover === BLACK ? res : 1 - res;
            sum[c] += v; sq[c] += v * v;
          }
        }
        done += b;
        let li = 0, ri = 1;
        for (let c = 1; c < n; c++) if (sum[c] > sum[li]) li = c;
        ri = li === 0 ? 1 : 0;
        for (let c = 0; c < n; c++) if (c !== li && sum[c] > sum[ri]) ri = c;
        const mL = sum[li] / done, mR = sum[ri] / done;
        const vL = Math.max(1e-6, sq[li] / done - mL * mL), vR = Math.max(1e-6, sq[ri] / done - mR * mR);
        if (mL - mR > VOTE_Z * Math.sqrt((vL + vR) / done)) break;
      }
      let li = 0;
      for (let c = 1; c < n; c++) if (sum[c] + r.random() * 1e-9 > sum[li]) li = c;
      return { move: _cand[li], info: `wr=${(sum[li] / done).toFixed(3)} of ${n}x${done}` };
    }
    let best = _cand[0], bestV = -1;
    for (let c = 0; c < n; c++) {
      let wins = 0;
      for (let p = 0; p < per; p++) {
        const g = game.clone();
        g.play(_cand[c]);
        const res = playout(g, r);
        wins += mover === BLACK ? res : 1 - res;
      }
      // Fractional dither breaks exact ties randomly; it can never reorder
      // distinct means (multiples of 1/per when playouts run to the end;
      // truncated returns are continuous, where a 1e-9 dither is still moot).
      const v = wins / per + r.random() * 1e-9;
      if (v > bestV) { bestV = v; best = _cand[c]; }
    }
    return { move: best, info: `wr=${bestV.toFixed(3)} of ${n}x${per}` };
  }

  // Position value oracle: mean outcome of PLAYOUTS standard playouts from
  // the position itself (no candidate machinery).  Returns P(BLACK wins),
  // matching the valueB convention (gen-agent-evals, ref-vlibpat, ...).
  function valueB(game, options = {}) {
    const game2 = game.cells ? game : game.toGame2();
    if (game2.gameOver) return game2.calcWinner() === BLACK ? 1 : 0;
    const r = options.rng || rng;
    let wins = 0, n = 0;
    if (options.budgetMs > 0) {
      const deadline = Date.now() + options.budgetMs;
      do { wins += playout(game2.clone(), r); n++; } while (Date.now() < deadline);
    } else {
      for (let p = 0; p < PLAYOUTS; p++) wins += playout(game2.clone(), r);
      n = PLAYOUTS;
    }
    return wins / n;
  }

  return { getMove, valueB };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
