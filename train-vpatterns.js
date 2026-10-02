#!/usr/bin/env node
'use strict';

// train-vpatterns.js — learn pattern weights via TD(λ) self-play.
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — λ-return TD applied at episode end, 2-ply lookahead
// (bootstraps from the next position where the same player moves):
//   G_t^λ      = (1−λ)·V(s_{t+2}) + λ·G_{t+2}^λ      (recursive form)
//   G_{M-1}^λ  = G_{M-2}^λ = outcome ∈ {1, 0.5, 0}   (terminal, M = #moves)
//   Δw_k       = (LR / n_features) · (G_t^λ − V_t) · polarity_k
//
//   Targets are absolute (P(BLACK wins)), independent of current player.
//   The two parity classes (even-t and odd-t) form independent chains.
//   λ = 0  → pure 2-step TD (target = V(s_{t+2}))
//   λ = 1  → pure Monte Carlo (target = outcome)
//   0<λ<1  → exponentially-weighted bootstrap trading bias for variance
//
// Training: pure self-play — both colours use the pattern policy.
//   Move selection uses absolute V = P(BLACK wins).
//   BLACK maximises V(s'), WHITE minimises V(s')  (full-width single-ply)
//
// Evaluation: play eval games against a configurable reference agent to
//   measure how much the policy has improved.  Eval games do not update weights.
//
// Features: pattern1 + pattern2 + pattern3 (maxLibs = 1), all cells.
//
// Status is printed at an exponentially increasing interval (× 1.4 each time),
// capped so the gap between prints never exceeds 6 hours.
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const { Game2, BLACK, PASS, setKomi, KOMI } = require('./game2.js');
const { evaluateFeatures, extractFeatures, prepareSpecs, deltaZ, loadWeights, saveWeights, makeWeights, specTag, specString, needsGame3 } = require('./vpatterns.js');
const { search } = require('./ai/vpatsearch.js');
const FeaturePol = require('./featurepol-lib.js');
const { game3FromGame2 } = require('./game3.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2), ['help', 'no-add'], ['accuracy-file', 'accuracy-games', 'budget', 'epsilon', 'eval', 'eval-size', 'ext', 'fp-width', 'fp-data', 'komi', 'ladder-file', 'ladder-min-chain', 'limit', 'load', 'lr', 'smooth-weights', 'md-file', 'on-policy', 'save', 'size', 'spec', 'start-phase', 'train-size', 'bootstrap']);
if (opts.help) {
  console.log(`Usage: node train-vpatterns.js [options]

TD(lambda) self-play trainer for vpatterns value weights (V(s) = P(BLACK
wins), 2-ply lookahead).  Runs indefinitely unless --limit is given; the
checkpoint is written at every print.

  --spec S          comma list of "size:maxLibs[f]" tokens (size 1-4, or
                    23 = the 2x3/3x2 and 34 = the 3x4/4x3 rectangle
                    pairs, both orientations;
                    maxLibs 1 = presence only, or L = ladder-coded cells
                    (vlibpat 7-state tactical alphabet; game3 pass per
                    position, not incremental); trailing 'f' freezes that
                    spec's loaded weights).  Default 1:6,2:6,3:6.  With
                    --load, --spec overrides the checkpoint's specs: shared
                    specs keep their trained weights, new ones start at zero
  --train-size N    self-play board size (default 13)
  --eval-size N     evaluation board size (default 13)
  --size N          sets both of the above
  --komi K          auto | auto:<start> | <number>.  auto (default) steps komi
                    by +/-1 every 500 self-play games while black's win share
                    sits outside [45%, 55%]; a number fixes komi and disables
                    the controller.  Eval games always use a fixed komi
  --limit N         stop after N games (default 0 = run indefinitely)

  --lr F            step size for the TD update (default 0.3)
  --smooth-weights A  Polyak EMA decay, applied every 100 games; 0 = off
                    (default 0.9).  The EMA weights are what gets saved
  --epsilon F       share of moves played uniformly at random (default 0.1)
  --on-policy F     share of the NON-random moves from this model's own
                    search; the rest come from --ext (default 1)
  --ext AGENT       ai/<name>.js supplying the off-policy moves; only
                    consulted when --on-policy < 1
  --fp-width K      restrict the trainee's GREEDY move to featurepol's top-K
                    candidates (0 = off, default), in self-play and in the
                    --eval reference games — matches deploying the vpat as a
                    re-ranker after an fp filter.  Epsilon exploration stays
                    full-width, for coverage of off-filter states
  --fp-data PATH    featurepol weights for --fp-width (default ref/ref-fp2-data.js)
  --start-phase F   fill the board with random stones to this phase before
                    normal training moves begin (backward curriculum).
                    'uniform' draws a fresh random phase per game
  --bootstrap N     run N fully-random (epsilon=1) games to seed the value
                    estimates before normal training; ignored with --load

  --load PATH       resume from a checkpoint; periodic saves start once training
                    time exceeds 2x the load time
  --ladder-min-chain N  smallest chain the ladder-coded (size:L) specs read:
                    2 skips single stones (~57% of the ladder time).  Saved
                    with the model; default: the --load model's value, else 1
  --no-add          fine-tune ONLY the keys already in the loaded model; never
                    intern new ones (requires --load)
  --save PATH       checkpoint path (default out/vpat-<random>.js)

  --eval AGENT      ai/<name>.js played as the reference in test games
                    (default: none, which disables the test games)
  --budget MS       per-move time budget for the reference agent (default 1)
  --md-file F       evalmovedetails positions, single full pass each print
                    (mdMae column)
  --ladder-file F   evalladders2 suite scored each print (ladr column)
  --accuracy-file F game corpus for winner-prediction accuracy each print
                    (vacc column)
  --accuracy-games N  games sampled from --accuracy-file (default 100)
  --help            show this message`);
  process.exit(0);
}
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '13',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '13', 10);
const SAVE_PATH  = opts.save  || `out/vpat-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const NO_ADD     = opts['no-add'] === true;   // fine-tune ONLY the loaded keys; never intern new ones
if (NO_ADD && !LOAD_PATH) { console.error('Error: --no-add needs a loaded model to fine-tune (pass --load).'); process.exit(1); }
const EVAL_AGENT = opts.eval  || '';     // empty disables in-training reference test games
const EXT_AGENT  = opts.ext   || '';     // off-policy move source: (1-epsilon) fraction of moves come from this agent
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
// --bootstrap N: run N fully-random (epsilon=1) training games before the normal
// loop, to seed the value estimates.  Ignored when resuming (--load).
const BOOTSTRAP = opts.bootstrap !== undefined ? parseInt(opts.bootstrap, 10) : 0;
if (opts.bootstrap !== undefined && !(BOOTSTRAP >= 0)) {
  console.error(`--bootstrap must be a non-negative integer (got '${opts.bootstrap}')`);
  process.exit(1);
}
const EPSILON    = parseFloat(opts.epsilon      || '0.1');
const ON_POLICY  = parseFloat(opts['on-policy'] || '1');   // share of non-random moves from own search1ply (vs --ext)
const FP_WIDTH   = opts['fp-width'] !== undefined ? parseInt(opts['fp-width'], 10) : 0;   // 0 = off; featurepol top-K filter
const FP_DATA    = opts['fp-data'] || path.join(__dirname, 'ref', 'ref-fp2-data.js');
// random stones until this board phase, then normal training.  'uniform' draws a
// fresh random target phase per game (see trainGame).
const START_PHASE_UNIFORM = (opts['start-phase'] || '0') === 'uniform';
const START_PHASE = START_PHASE_UNIFORM ? 0 : parseFloat(opts['start-phase'] || '0');
if (!START_PHASE_UNIFORM && !(START_PHASE >= 0 && START_PHASE <= 1)) {
  console.error(`--start-phase must be a phase in [0,1] or 'uniform' (got '${opts['start-phase']}')`);
  process.exit(1);
}
const MD_FILE         = opts['md-file']          || null;   // evalmovedetails positions for the single-pass mdMae column
const LADDER_FILE     = opts['ladder-file']      || null;   // evalladders2 suite to score each status print (the ladr column)
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']       || '0.3');
// Polyak EMA, applied every EMA_PERIOD games; 0 = off.
// Window ≈ EMA_PERIOD/(1-alpha) games: --smooth-weights 0.9 ≈ 1k games, 0.99 ≈ 10k.
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');
const EMA_PERIOD = 100;
const BUDGET     = parseFloat(opts['budget']   || '1');

// Komi controller (the train-hpatterns design): every KOMI_WINDOW self-play
// games, step komi by +/-1 while black's win share sits outside [45%, 55%].
// --komi auto:<start> seeds the start; --komi <number> fixes komi and
// disables the controller.  The current komi is persisted in the checkpoint
// and restored on --load (auto mode only).
let AUTO_KOMI = true;
if (opts.komi !== undefined) {
  const m = /^auto(?::(-?[0-9.]+))?$/.exec(opts.komi);
  if (m) {
    if (m[1] !== undefined) { setKomi(TRAIN_SIZE, parseFloat(m[1])); setKomi(EVAL_SIZE, parseFloat(m[1])); }
  } else {
    AUTO_KOMI = false;
    setKomi(TRAIN_SIZE, parseFloat(opts.komi));
    setKomi(EVAL_SIZE,  parseFloat(opts.komi));
  }
}
// Eval games ALWAYS use a fixed komi (see train-hpatterns): the controller
// only ever moves the TRAIN_SIZE komi, and the eval batch pins EVAL_SIZE.
const EVAL_KOMI = KOMI(EVAL_SIZE);
const KOMI_WINDOW = 500;
let komiGames = 0, komiBlackWins = 0;
let komiSum = 0, komiSumGames = 0;   // per-interval avg komi (avgK column)

// ── Features ───────────────────────────

// --spec: comma list of "size:maxLibs[f]" tokens (e.g. "1:6,2:6,3:6").  size is
// 1-4 or 34 (the 3×4/4×3 rectangle pair, both orientations under one tag);
// maxLibs caps the per-cell liberty count (1 = presence only).  A trailing
// 'f' freezes that spec's weights (loaded values stay fixed; gradient updates
// skipped).  Default: sizes 1/2/3 at maxLibs 6.  With --load, --spec overrides
// the checkpoint's specs: shared specs keep their trained weights (freeze the
// carried-over ones with 'f' for a curriculum), new specs start at zero.
let specs;
const FROZEN = new Set();
// Health model for health-coded specs / the C survival attribute; the library
// no longer reads the environment itself.
const HEALTH_PATH = (typeof process !== 'undefined' && process.env.HEALTH_DATA) || '';   // spec tags ((maxLibs << 3) | size) excluded from updates
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    if (tok[0] === 't') {
      let body = tok.slice(1);
      let phaseBins = 0;
      const pm = /^p(\d+)$/.exec(body);
      if (pm) { phaseBins = parseInt(pm[1], 10); body = ''; }
      if (body !== '' || (pm && !(phaseBins >= 1 && phaseBins <= 64))) {
        console.error(`--spec: bad t token '${tok}' (expected t or t p<1-64>, e.g. tp9)`);
        process.exit(1);
      }
      return phaseBins > 1 ? { turn: true, phaseBins } : { turn: true };
    }
    const [s, mRaw] = tok.split(':');
    const size = parseInt(s, 10);
    const frozen = /f$/.test(mRaw);
    let body = frozen ? mRaw.slice(0, -1) : mRaw;
    // optional phase-bin suffix pN (e.g. 2:1p2): per-spec phase-salted keys
    let patBins = 0;
    const pbm = /p(\d+)$/.exec(body);
    if (pbm) { patBins = parseInt(pbm[1], 10); body = body.slice(0, -pbm[0].length); }
    // 'L' = the ladder-coded family (vlibpat 7-state tactical alphabet),
    // internally maxLibs 0.  Not incremental: the 1-ply search falls back to
    // a full extraction per candidate (several times slower per move).
    // 'H<N>' = the health-coded family: each stone takes its chain's survival
    // bucket (1..N under the frozen chain-survival model) in place of its
    // liberty count.  Internally maxLibs = -N; needs HEALTH_DATA, and like
    // 'L' it is not incremental.
    const hm = /^H(\d+)$/.exec(body);
    if (hm) {
      const hb = parseInt(hm[1], 10);
      if (!((size >= 1 && size <= 4) || size === 8 || size === 34 || size === 23) || !(hb >= 2 && hb <= 15)) {
        console.error(`--spec: bad token '${tok}' (expected size:H<N>, size 1-4, 8, 23 or 34, N 2-15)`);
        process.exit(1);
      }
      if (frozen) FROZEN.add(specTag({ size, maxLibs: -hb }));
      return patBins > 1 ? { size, maxLibs: -hb, phaseBins: patBins } : { size, maxLibs: -hb };
    }
    // Strict: parseInt would silently accept trailing garbage ('2H' -> 2), so
    // a mistyped family marker would train as a liberty spec instead of
    // erroring.  Only 'L', 'H<N>' (handled above) or bare digits are valid.
    if (body !== 'L' && !/^\d+$/.test(body)) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs, size:L or size:H<N>; got '${body}')`);
      process.exit(1);
    }
    const maxLibs = body === 'L' ? 0 : parseInt(body, 10);
    if (!((size >= 1 && size <= 4) || size === 8 || size === 34 || size === 23) || !(maxLibs >= 1 || body === 'L')) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs[f] or size:L[f], size 1-4, 8, 23 or 34, maxLibs >= 1)`);
      process.exit(1);
    }
    if (frozen) FROZEN.add(specTag({ size, maxLibs }));
    return patBins > 1 ? { size, maxLibs, phaseBins: patBins } : { size, maxLibs };
  });
} else {
  specs = [
    { size: 1, maxLibs: 6 },
    { size: 2, maxLibs: 6 },
    { size: 3, maxLibs: 6 },
  ];
}
// --ladder-min-chain: given, it wins; else a --load model's own value (below), else 1.
const LADDER_MIN_CHAIN_FLAG = opts['ladder-min-chain'] !== undefined ? parseInt(opts['ladder-min-chain'], 10) : null;
if (LADDER_MIN_CHAIN_FLAG !== null && !(LADDER_MIN_CHAIN_FLAG >= 1)) { console.error('--ladder-min-chain must be a positive integer'); process.exit(1); }
let prepSpecs = prepareSpecs(specs, { health: HEALTH_PATH, ladderMinChain: LADDER_MIN_CHAIN_FLAG || 1 });

// ── Weight table ──────────────────────────────────────────────────────────────

let weights  = makeWeights();  // pattern key (int32) → weight (float)
let weightsEMA = makeWeights();  // Polyak-averaged shadow (saved/eval'd when EMA on)
let weightsEMAInit = false;  // first applyEMA seeds EMA = weights

// Polyak / SWA averaging: weightsEMA[k] = alpha·weightsEMA[k] + (1-alpha)·weights[k].
// New keys (interned since the last call) are seeded at their live value.
function applyEMA(alpha) {
  if (!weightsEMAInit) {
    weights.forEach((k, v) => weightsEMA.set(k, v));
    weightsEMAInit = true;
    return;
  }
  weights.forEach((k, v) => {
    const e = weightsEMA.get(k);
    weightsEMA.set(k, e === undefined ? v : alpha * e + (1 - alpha) * v);
  });
}

// Eval ≡ save: the model that save writes and eval matches measure — the EMA
// shadow when enabled (and initialized), else the live weights.
function saveSource() {
  return (EMA_ALPHA > 0 && weightsEMAInit) ? weightsEMA : weights;
}
let wAbsSum = 0, wUpdateCount = 0;  // per-interval |weight| sum/count over feature updates (avgW; reset each print)

// ── Training helpers ──────────────────────────────────────────────────────────

// Absolute terminal outcome: 1=BLACK wins, 0=WHITE wins.
function absoluteOutcome(game) {
//  return game.estimateWinner() === BLACK ? 1 : 0;
  return game.calcWinner() === BLACK ? 1 : 0;
}

// Plain SGD:
//   g_k  = (target − V) / n · polarity_k
//   Δw_k = lr · g_k
function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const { keys, pols, tags } = features;
  const hasFrozen = FROZEN.size > 0;
  let nActive = n;
  if (hasFrozen) {
    nActive = 0;
    for (let i = 0; i < n; i++) if (!FROZEN.has(tags[i])) nActive++;
    if (nActive === 0) return;
  }
  const step = lr * (target - features.val) / nActive;
  for (let i = 0; i < n; i++) {
    if (hasFrozen && FROZEN.has(tags[i])) continue;
    const k = keys[i];
    const cur = weights.get(k);
    if (NO_ADD && cur === undefined) continue;   // key not in the loaded model: skip, don't intern
    const w = (cur ?? 0) + pols[i] * step;
    weights.set(k, w);
    wAbsSum += Math.abs(w);
    wUpdateCount++;
  }
}

// ── Self-play training ────────────────────────────────────────────────────────

// 1-ply move selection: vpatsearch's incremental depth-1 search (one base
// extraction, deltaZ per candidate, capture fallback on its own prepSpecs).
// The same search serves self-play, the ladder/md suites, and eval games.
// Note the vpatsearch tie/PASS semantics: first-strict-improvement for both
// colours, and a terminal double-pass is scored exactly.
function search1ply(game) {
  return search(game, { weights, specs, preparedSpecs: prepSpecs });
}

// Both colours use the policy.  Per-position features and values are collected
// during play; at episode end the λ-return target is computed by a single
// backward pass and applied to each position.
function trainGame(N, epsilon = EPSILON) {
  const game     = new Game2(N, true);   // free initial stone (applyFirstMove=true)
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  // --start-phase: fill the board with random stones up to the target phase
  // before training begins.  The prefix is untrained (no features recorded).
  // 'uniform' draws a fresh target in [0,1) each game, spreading coverage across
  // all phases instead of one fixed curriculum stage.
  const startPhase = START_PHASE_UNIFORM ? Math.random() : START_PHASE;
  while (game.phase() < startPhase && !game.gameOver) game.play(game.randomLegalMove());

  // One Game3, advanced in lockstep with `game`, reused for ladder-coded
  // extraction each ply instead of rebuilding one from scratch.
  const useG3 = needsGame3(prepSpecs);
  const g3 = useG3 ? game3FromGame2(game) : undefined;

  let moves = 0;
  const featsArr = [];
  const vals = [];

  while (!game.gameOver && moves < maxMoves) {
    const features = extractFeatures(game, prepSpecs, undefined, undefined, undefined, g3);
    evaluateFeatures(features, weights);
    featsArr.push(features);
    vals.push(features.val);

    let move;
    if (Math.random() < epsilon) {
      move = game.randomLegalMove();                 // full-width exploration (coverage off the filter)
    } else if (extGetMove && Math.random() > ON_POLICY) {
      move = extGetMove(game).move;
    } else if (fpWeights) {
      const cand = fpTopK(game, FP_WIDTH);           // greedy only within the fp filter
      move = (cand && cand.length) ? bestFiltered(game, cand, weights) : search1ply(game);
    } else {
      move = search1ply(game);
    }
    game.play(move);
    if (useG3) g3.play(move);
    moves++;
  }

  const elapsedMs = Date.now() - tStartMs;
  const outcome   = absoluteOutcome(game);

  // TD(0) backward pass with 2-ply lookahead.  Each parity class (even-t and
  // odd-t) is its own chain — same-player moves are 2 apart.  The target for
  // step t is the value of the next same-parity step (V_{t+2}); base cases
  // G_{M-1} = G_{M-2} = outcome.
  let G0 = outcome, G1 = outcome;
  for (let t = featsArr.length - 1; t >= 0; t--) {
    if ((t & 1) === 0) {
      tdUpdate(featsArr[t], G0, LR);
      G0 = vals[t];
    } else {
      tdUpdate(featsArr[t], G1, LR);
      G1 = vals[t];
    }
  }

  return { winner: game.estimateWinner(), elapsedMs, moves };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Play nGames of policy vs agent, alternating colours.
// Returns { results } where each element is 1 (policy win), 0 (agent win), or 0.5 (draw).
function evalVsReference(N, refGetMove, nGames, budget) {
  const results = [];
  let totalMoves = 0;
  let accCorrect = 0, accN = 0;   // per-position winner prediction (test-side acc)
  let predHit = 0, predN = 0;     // mvPred: subject moves that match the eval agent's move (50% sampled)

  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game     = new Game2(N, true);   // free initial stone (applyFirstMove=true)
    // Random opening: 3 random legal moves to diversify positions (same as
    // selfplay.js --rand-moves default).
    for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
    const maxMoves = N * N * 4;
    let   moves    = 0;

    // One Game3 per game, advanced in lockstep, reused for ladder extraction.
    const useG3 = needsGame3(prepSpecs);
    const g3 = useG3 ? game3FromGame2(game) : undefined;

    const evalW = saveSource();
    const gameVals = [];
    while (!game.gameOver && moves++ < maxMoves) {
      const f = extractFeatures(game, prepSpecs, undefined, undefined, undefined, g3);
      evaluateFeatures(f, evalW);
      gameVals.push(f.val);
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        const cand = fpWeights ? fpTopK(game, FP_WIDTH) : null;
        idx = (cand && cand.length) ? bestFiltered(game, cand, evalW)
                                    : search(game, { weights: evalW, specs, preparedSpecs: prepSpecs });
        // mvPred: how often the subject's move matches the eval agent's.  Sample
        // half the subject's moves — the eval agent call is a full search.
        if (Math.random() < 0.5) {
          predN++;
          const mv = refGetMove(game, budget);
          if ((mv.move !== undefined ? mv.move : PASS) === idx) predHit++;
        }
      } else {
        const mv = refGetMove(game, budget);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      if (!game.play(idx)) {
        console.log("Illegal move!");
      } else if (useG3) {
        g3.play(idx);
      }
    }

    const winner = game.calcWinner();
    totalMoves += moves;
    for (const v of gameVals) if ((v >= 0.5) === (winner === BLACK)) accCorrect++;
    accN += gameVals.length;
    if ((winner === BLACK) === policyIsBlack) {
      results.push(1);
    } else {
      results.push(0);
    }
  }

  return { results, moves: totalMoves, accCorrect, accN, predHit, predN };
}

// ── CLI ───────────────────────────────────────────────────────────────────────

// Load eval agent from ai/ folder (only when --eval was supplied).  Factory
// agents (exporting create) are instantiated eagerly here — before the header
// prints — so their create-time load banner lands in the startup block rather
// than splitting the header from the first data row (a lazy .getMove would defer
// create() to the first eval move).  Non-factory agents load at require.
const evalGetMove = EVAL_AGENT
  ? (() => {
      const m = require(path.join(__dirname, 'ai', EVAL_AGENT + '.js'));
      return (typeof m.create === 'function' ? m.create(Util.makeCfg()) : m).getMove;
    })()
  : null;

// Load off-policy move-source agent (only when --ext was supplied).  Factory
// agents (exporting create) are instantiated eagerly here — before the header
// prints — so their load banner lands in the startup block rather than splitting
// the header from the first data row.  Non-factory agents load at require.
const extGetMove = EXT_AGENT
  ? (() => {
      const m = require(path.join(__dirname, 'ai', EXT_AGENT + '.js'));
      return (typeof m.create === 'function' ? m.create(Util.makeCfg()) : m).getMove;
    })()
  : null;

// Featurepol candidate filter (--fp-width): restrict the trainee's greedy and
// epsilon moves — in self-play AND in the --eval reference games — to
// featurepol's top-K, matching how the vpat is deployed as a re-ranker after an
// fp filter.  Off when FP_WIDTH is 0.
const fpWeights = FP_WIDTH > 0 ? FeaturePol.loadModel({ name: 'td-fp-filter', path: FP_DATA }).weights : null;
let _fpState = null, _fpScores = null;
function fpTopK(game, K) {
  const N = game.N;
  if (!_fpState || _fpState.moves.length < N * N) {
    _fpState  = FeaturePol.createState(N, fpWeights.spec);
    _fpScores = new Float64Array(N * N + 1);
  }
  const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
  FeaturePol.extractFeatures(game, _fpState, fpWeights, game3);
  const n = FeaturePol.scoreAll(_fpState, fpWeights, _fpScores);
  if (n === 0) return null;
  const k = Math.min(K, n);
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => _fpScores[b] - _fpScores[a]);
  const out = new Int32Array(k);
  for (let j = 0; j < k; j++) out[j] = _fpState.moves[order[j]];
  return out;
}
// Greedy vpat move within the filtered set under weights w, mirroring
// vpatsearch's depth-1 loop: extract the base once, then deltaZ per candidate
// (a capture returns NaN -> full extraction on a SEPARATE prep so the base
// planes survive).  Non-incremental specs (turn/ladder/health/phased) have no
// incremental contract, so they fall back to full extraction per candidate.
// Mover-relative like vpatsearch (BLACK maximises V, WHITE minimises).
let _fbPrep = null;
function bestFiltered(game, cand, w) {
  const prep = prepSpecs;
  const incremental = !(prep.hasLadder || prep.hasPhasedPatterns || prep.hasHealth || prep.hasTurn);
  const black = game.current === BLACK;
  let zBase = 0;
  if (incremental) {
    const f = extractFeatures(game, prep);   // primes the base planes deltaZ reads
    evaluateFeatures(f, w);
    zBase = f.z;
  }
  let best = cand[0], bestV = -Infinity;
  for (let i = 0; i < cand.length; i++) {
    const c = cand[i];
    let z;
    if (incremental) {
      const d = deltaZ(game, prep, w, c);
      if (d === d) {
        z = zBase + d;
      } else {                               // capture: full extraction on a separate prep
        const g = game.clone(); g.play(c);
        const ff = extractFeatures(g, _fbPrep || (_fbPrep = prepareSpecs(specs, { ladderMinChain: prep.ladderMinChain })));
        evaluateFeatures(ff, w); z = ff.z;
      }
    } else {
      const g = game.clone(); g.play(c);
      const ff = extractFeatures(g, prep); evaluateFeatures(ff, w); z = ff.z;
    }
    const V = 1 / (1 + Math.exp(-z));
    const v = black ? V : 1 - V;
    if (v > bestV) { bestV = v; best = c; }
  }
  return best;
}

// Move-quality suite (evalmovedetails): a single full pass scoring the trainee
// against --md-file at each status print (the `mdMae` column — mean win-ratio
// gap to the top move).
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);

// Ladder suite (evalladders2): score the trainee's own 1-ply argmax (search1ply)
// against --ladder-file at each status print (the `ladr` column).
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const ladderAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

// Wall time of the --load read; the periodic save waits until training has
// run longer than twice this (0 without --load: save from the first row).
let loadMs = 0;
if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    // Compare canonical fields, not whole objects (spec objects can carry
    // derived properties that would false-positive the comparison).
    const specKey = ss => ss.map(x => x.turn ? 't' : `${x.size}:${x.maxLibs === 0 ? 'L' : x.maxLibs}`).join(',');
    const cliSpecs = opts.spec ? specs : null;
    const tLoad = Date.now();
    const loaded = loadWeights(LOAD_PATH, HEALTH_PATH);
    loadMs = Date.now() - tLoad;
    ({ weights, specs, preparedSpecs: prepSpecs } = loaded);
    if (LADDER_MIN_CHAIN_FLAG !== null) prepSpecs.ladderMinChain = LADDER_MIN_CHAIN_FLAG;   // the flag overrides the file
    // Saved komi wins over any auto:<start> seed (auto mode only; the eval
    // komi stays pinned at EVAL_KOMI).
    if (AUTO_KOMI && loaded.komi !== undefined) setKomi(TRAIN_SIZE, loaded.komi);
    if (cliSpecs !== null && specKey(cliSpecs) !== specKey(specs)) {
      // --spec overrides the checkpoint's specs.  Loaded weights are kept —
      // each spec hashes patterns with its own mixer (same collision
      // assumption as any multi-spec run), so shared specs continue from
      // their trained values (typically frozen with 'f'), new specs start at
      // zero, and dropped specs' weights stay in the table, never extracted.
      console.warn(`WARNING: --spec overrides checkpoint specs (${specKey(specs)} -> ${specKey(cliSpecs)}); shared specs keep their weights.`);
      specs = cliSpecs;
      prepSpecs = prepareSpecs(specs, { health: HEALTH_PATH, ladderMinChain: prepSpecs.ladderMinChain });
    }
    if (EMA_ALPHA > 0) {   // continue averaging on top of the persisted values
      weightsEMA = weights.clone();
      weightsEMAInit = true;
    }
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH} in ${Util.fmtMs(loadMs)}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}
if (NO_ADD) console.log(`no-add: key set frozen at ${weights.size} loaded weights (unknown keys skipped, no update)`);


console.log(`LR=${LR}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  smooth-weights=${EMA_ALPHA}  start-phase=${START_PHASE_UNIFORM ? 'uniform' : START_PHASE}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}${FP_WIDTH > 0 ? `  fp-filter=top${FP_WIDTH} (${path.basename(FP_DATA)})` : ''}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log(`Specs: ${specString(specs)}${FROZEN.size > 0 ? `  frozen: [${specString(specs.filter(sp => FROZEN.has(specTag(sp))))}]` : ''}` +
            (prepSpecs.hasLadder ? `  ladder-min-chain: ${prepSpecs.ladderMinChain}` : ''));
console.log();

// --bootstrap N: seed the value estimates with N fully-random (epsilon=1) games
// before the normal loop.  Skipped when resuming (--load already has weights).
// Produces no table output and its updates are discarded from the interval stats.
if (BOOTSTRAP > 0 && !LOAD_PATH) {
  process.stdout.write('Bootstrapping... ');
  const bootStart = Date.now();
  let nextPct = 5;   // report progress every 5% (integer compare avoids float drift)
  for (let i = 0; i < BOOTSTRAP; i++) {
    trainGame(TRAIN_SIZE, 1);
    while (nextPct <= 100 && (i + 1) * 100 >= nextPct * BOOTSTRAP) {
      process.stdout.write(` ${nextPct}%`);
      nextPct += 5;
    }
  }
  wAbsSum = 0; wUpdateCount = 0;   // discard bootstrap updates from the first avgW
  console.log(` elapsed=${Util.fmtMs(Date.now() - bootStart)}`);
}

// Print header.
console.log([
  // Training columns (left).
  'game'.padStart(4),
  'avgK'.padStart(6),
  'tElp'.padStart(5),
  'tMv '.padStart(5),
  'nWts'.padStart(4),
  'avgL'.padStart(4),
  'avgW'.padStart(6),
  'tTran'.padStart(5),
  // Test / eval columns (right).
  ...(evalGetMove ? ['mvPred'.padStart(6)] : []),
  // winRatio: "wr(g)/avg(ga)" — wr/avg fmtRatio4, g/ga fmt4 game counts (this
  // interval's, and the rolling-half window).  Fixed 21 chars wide.
  ...(evalGetMove ? ['winRatio'.padStart(21), ' acc'.padStart(4)] : []),
  ...(ladderCases ? ['ladr'.padStart(4)] : []),
  ...(ACCURACY_FILE     ? ['vacc'.padStart(4)] : []),
  ...(mdPositions ? ['mdMae'.padStart(5)] : []),
  // tTest = whole eval pass; tTurn = wall-clock per move of the reference
  // MATCHES only (both sides' moves); training's per-move time is tMv.
  ...(evalGetMove ? ['tTest'.padStart(5), 'tTurn'.padStart(5)] : []),
].join('  '));

const t0 = Date.now();
let saveSkipped = false, saveSkipNoted = false;   // periodic save held back by the load-time rule
const MAX_PRINT_INTERVAL_MS = 4 * 60 * 60 * 1000;  // cap status-print gap at 4 hours
let nextPrintAt = t0 + 1000;
let g = 0;
let intervalGames = 0;
let intervalMoves = 0;
let intervalTrainMs = 0;
let refBudgetMs = BUDGET;
const evalHistory = [];   // per-interval game results (1/0.5/0)

while (true) {
  g++;
  const { winner, moves, elapsedMs } = trainGame(TRAIN_SIZE);
  komiSum += KOMI(TRAIN_SIZE); komiSumGames++;
  if (AUTO_KOMI) {
    komiGames++; komiBlackWins += winner === BLACK ? 1 : 0;
    if (komiGames >= KOMI_WINDOW) {
      const bw = komiBlackWins / komiGames;
      if (bw > 0.55 || bw < 0.45) {
        setKomi(TRAIN_SIZE, KOMI(TRAIN_SIZE) + (bw > 0.55 ? 1 : -1));
      }
      komiGames = 0; komiBlackWins = 0;
    }
  }
  if (EMA_ALPHA > 0 && g % EMA_PERIOD === 0) applyEMA(EMA_ALPHA);
  intervalGames++;
  intervalMoves += moves;
  intervalTrainMs += elapsedMs;

  // Force a final stats row when the game limit is reached.
  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let latestWR = null, avgWR = null, resultsBatchLen = 0, evalHalf = 0;
    let evalMatchMs = 0, evalMatchMoves = 0, evalAccC = 0, evalAccN = 0, evalPredHit = 0, evalPredN = 0;
    if (evalGetMove) {
      const trainKomi = KOMI(TRAIN_SIZE);
      setKomi(EVAL_SIZE, EVAL_KOMI);
      const resultsBatch = [];
      while (true) {
        const { results, moves, accCorrect, accN, predHit, predN } = evalVsReference(EVAL_SIZE, evalGetMove, 2, refBudgetMs);
        for (const r of results) resultsBatch.push(r);
        evalMatchMoves += moves;
        evalAccC += accCorrect; evalAccN += accN;
        evalPredHit += predHit; evalPredN += predN;
        evalMatchMs = Date.now() - tTestStart;
        if (evalMatchMs > 0.3 * intervalTrainMs) break;
        if (resultsBatch.length >= 2000) break;
      }
      setKomi(TRAIN_SIZE, trainKomi);   // restore (same entry when sizes match)
      for (const r of resultsBatch) evalHistory.push(r);

      latestWR = resultsBatch.reduce((s, r) => s + r, 0) / resultsBatch.length;
      evalHalf = Math.max(1, Math.floor(evalHistory.length / 2));
      avgWR = evalHistory.slice(-evalHalf).reduce((s, r) => s + r, 0) / evalHalf;
      resultsBatchLen = resultsBatch.length;
    }

    const avgLen  = intervalMoves / intervalGames;
    const tMvMs   = intervalTrainMs / Math.max(1, intervalMoves);
    const kAvg    = komiSumGames > 0 ? komiSum / komiSumGames : KOMI(TRAIN_SIZE);
    komiSum = 0; komiSumGames = 0;   // per-interval avgK: reset at each print
    intervalGames = 0;
    intervalMoves = 0;
    let ladrCol = null;
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, ladderAgent, { budgetMs: 1, oversample: 1 });
      ladrCol = Util.fmtRatio4(total ? passed / total : 0);
    }
    let vaccCol = null;
    if (ACCURACY_FILE) {
      const { accuracy } = evalValueAccuracy(ACCURACY_FILE, { weights, specs }, { nGames: ACCURACY_GAMES });
      vaccCol = Util.fmtRatio4(accuracy);
    }
    let mdMaeCol = null;
    if (mdPositions) {
      const { maeErr } = evalPositions(game => ({ move: search(game, { weights, specs, preparedSpecs: prepSpecs }) }), mdPositions, 0);
      mdMaeCol = Util.fmtRatio4(maeErr).padStart(5);
    }
    const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;
    wAbsSum = 0; wUpdateCount = 0;   // per-interval avgW: reset at each print

    const tTestMs   = Date.now() - tTestStart;
    const elapsedMs = Date.now() - t0;
    const trainMs   = intervalTrainMs;
    intervalTrainMs = 0;
    const nextMs    = elapsedMs;
    console.log([
      // Training columns (left).
      Util.fmt4i(g),
      Util.fmt4(kAvg).padStart(6),
      Util.fmtMs(elapsedMs),
      Util.fmtMs(tMvMs),
      Util.fmt4i(weights.size),
      Util.fmt4(avgLen),
      wAvg.toFixed(4).padStart(6),
      Util.fmtMs(trainMs),
      // Test / eval columns (right).
      ...(evalGetMove ? [Util.fmtRatio4(evalPredN > 0 ? evalPredHit / evalPredN : 0).padStart(6)] : []),
      ...(evalGetMove ? [(`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(resultsBatchLen)})` +
                          `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`).padStart(21),
                         Util.fmtRatio4(evalAccN > 0 ? evalAccC / evalAccN : 0)] : []),
      ...(ladrCol ? [ladrCol]               : []),
      ...(vaccCol ? [vaccCol]               : []),
      ...(mdMaeCol ? [mdMaeCol]             : []),
      ...(evalGetMove ? [Util.fmtMs(tTestMs),
                         Util.fmtMs(evalMatchMoves > 0 ? evalMatchMs / evalMatchMoves : 0)] : []),
    ].join('  '));
    if (elapsedMs > 2 * loadMs) {
      saveWeights(SAVE_PATH, { weights: saveSource(), specs, preparedSpecs: prepSpecs, komi: KOMI(TRAIN_SIZE) });
      saveSkipped = false;
    } else {
      if (!saveSkipNoted) {
        console.log(`(save skipped until training time exceeds 2x the model load time, ${Util.fmtMs(2 * loadMs)})`);
        saveSkipNoted = true;
      }
      saveSkipped = true;
    }
    nextPrintAt = Math.min(t0 + Math.round(nextMs * 1.4), Date.now() + MAX_PRINT_INTERVAL_MS);
  }

  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) {
    if (saveSkipped) saveWeights(SAVE_PATH, { weights: saveSource(), specs, preparedSpecs: prepSpecs, komi: KOMI(TRAIN_SIZE) });
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
