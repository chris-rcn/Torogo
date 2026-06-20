#!/usr/bin/env node
'use strict';

// train-search-values.js — offline vlibpat value trainer that regresses V(s)
// toward the puct-static SEARCH value at each position (from gen-search-values.js),
// instead of a TD bootstrap.
//
//   Δw_k = (lr / n_features) · (target − V) · polarity_k     [logistic update]
//   target = the recorded absolute P(BLACK wins) search value of s
//
// Each position's target is the search's value of that exact position (same
// player to move, by construction) — a direct per-position regression, with no
// cross-parity bootstrap.  The search reads ladders (npat priors + tree, 18/19
// on evalladders) even though the vlibpat leaf alone does not, so the value
// target carries ladder knowledge raw TD targets cannot.
//
// Data files are read in chunks (handles multi-GB / still-growing files).
// `ladr` column = the live weights' own full-width 1-ply argmax on the ladder
// suite (the same selection self-play would use).
//
// Usage:
//   node train-search-values.js --file a.ndjson[,b.ndjson,...] [--save model.js]
//        [--load model.js] [--spec 2,3] [--lr 0.05] [--epochs 0] [--seed 1]
//        [--ema-alpha 0.95] [--eval-size 13]

const fs = require('fs');
const { performance } = require('perf_hooks');
const { Game2, BLACK, PASS, parseMove } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { extractFeatures, evaluateFeatures, prepareSpecs, loadWeights, saveWeights } = require('./vlibpat.js');
const Ladders = require('./evalladders.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help || !opts.file) {
  console.log('Usage: node train-search-values.js --file <a.ndjson[,b,...]> [--save model.js] ' +
              '[--load model.js] [--spec 2,3] [--lr 0.05] [--epochs 0] [--seed 1] [--ema-alpha 0.95]');
  process.exit(opts.help ? 0 : 1);
}

const FILES     = opts.file.split(',');
const SAVE_PATH  = opts.save || `out/vsearch-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;
const LR         = parseFloat(opts.lr || '0.05');
const EPOCHS     = parseInt(opts.epochs || '0', 10);   // 0 = unlimited
const EMA_ALPHA  = parseFloat(opts['ema-alpha'] || '0.95');
const EVAL_SIZE  = parseInt(opts['eval-size'] || '13', 10);
const POS_LIMIT  = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;  // max positions loaded
const ALPHA      = parseFloat(opts.alpha !== undefined ? opts.alpha : '0');  // optional search-value weight blended into the bootstrap target (0 = pure TD)
// ── Hinge (margin ranking) options ─────────────────────────────────────────
// Add a logit-space hinge that pushes the CHOSEN move's candidate value past
// sampled sibling moves by a small margin (sign by player-to-move).  Trains the
// 1-ply argmax (ranking) directly while keeping siblings close/comparable —
// runs ON TOP of the TD+outcome anchor (which keeps the absolute level valid).
const HINGE        = opts.hinge !== undefined;
const MARGIN       = parseFloat(opts.margin !== undefined ? opts.margin : '0.5');   // logit-space margin
const NEG_K        = parseInt(opts['neg-k'] !== undefined ? opts['neg-k'] : '8', 10); // sampled sibling negatives / position
const HINGE_LR     = parseFloat(opts['hinge-lr'] !== undefined ? opts['hinge-lr'] : String(parseFloat(opts.lr || '0.05')));
const OPENING_SKIP = parseInt(opts['opening-skip'] !== undefined ? opts['opening-skip'] : '20', 10); // skip visit-sampled opening (chosen != search argmax there)

// ── Spec list (same tokens as train-vlibpat) ──────────────────────────────────
let specs;
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    if (tok === '1')  return { size: 1 };
    if (tok === '2')  return { size: 2 };
    if (tok === '3')  return { size: 3 };
    if (tok === '1p') return { size: 1, ladder: false };
    if (tok === '3p') return { size: 3, ladder: false };
    console.error(`--spec: unknown token '${tok}' (use 1, 2, 3, 1p, 3p)`);
    process.exit(1);
  });
} else {
  specs = [{ size: 2 }, { size: 3 }, { size: 3, ladder: false }];
}
const tacticsOpts = { tactics: opts.tactics || 'ladder2' };
let prepSpecs = prepareSpecs(specs, tacticsOpts);

// ── Weights ───────────────────────────────────────────────────────────────────
let weights = new Map();
let weightsEMA = new Map();
let weightsEMAInit = false;
if (LOAD_PATH) {
  const m = loadWeights(LOAD_PATH);
  weights = m.weights;
  if (m.specs) { specs = m.specs; prepSpecs = m.preparedSpecs || prepareSpecs(specs, tacticsOpts); }
  console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
}

function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const perFeature = (target - features.val) / n;
  const step = lr * perFeature;
  const { keys, pols } = features;
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    weights.set(k, (weights.get(k) ?? 0) + pols[i] * step);
  }
}

function applyEMA(alpha) {
  if (!weightsEMAInit) { for (const [k, v] of weights) weightsEMA.set(k, v); weightsEMAInit = true; return; }
  for (const [k, v] of weights) {
    const e = weightsEMA.get(k);
    weightsEMA.set(k, e === undefined ? v : alpha * e + (1 - alpha) * v);
  }
}

// Full-width 1-ply argmax over the live weights (the ladder gate / play policy).
function search1ply(game, game3) {
  const area = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestMove = PASS, bestScore = isBlack ? -Infinity : Infinity;
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
    const f = extractFeatures(game, prepSpecs, true, coord, game3);
    evaluateFeatures(f, weights);
    if (isBlack === (f.val > bestScore)) { bestScore = f.val; bestMove = coord; }
  }
  if (bestMove !== PASS && (game.consecutivePasses > 0 || game.emptyCount < area / 2)) {
    const pf = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(pf, weights);
    if (isBlack === (pf.val > bestScore)) bestMove = PASS;
  }
  return bestMove;
}

// ── Data ────────────────────────────────────────────────────────────────────
// Parse the compact per-game NDJSON — one line per game: { boardSize, moves
// (CSV string), values (per-position), explore (array of exploration move
// indices), z } — into { boardSize, tokens, outcome, recs:[{histLen, value,
// explore}] }, where rec i is the position before move i.
// Stream games from a file, calling onGame(game) per parsed game; stops after
// posBudget positions.  O(1) games in memory — the file is re-read each epoch
// (the OS page cache keeps it warm, and re-parse is negligible next to the
// per-epoch replay + feature extraction).  Returns positions streamed.
function streamFile(filePath, onGame, posBudget) {
  let tornError = null, pos = 0;
  function processLine(line) {   // returns true to signal "budget reached, stop"
    line = line.trim();
    if (!line || line.startsWith('#')) return false;
    if (pos >= posBudget) return true;
    if (tornError) throw tornError;   // a prior parse failure was not the last line — real corruption
    let g;
    try { g = JSON.parse(line); } catch (e) { tornError = e; return false; }
    const tokens = g.moves ? g.moves.split(',') : [];
    const exploreSet = (g.explore && g.explore.length) ? new Set(g.explore) : null;
    const recs = [];
    for (let i = 0; i < tokens.length; i++) {
      recs.push({ histLen: i, value: g.values[i], explore: exploreSet ? exploreSet.has(i) : false });
    }
    onGame({ boardSize: g.boardSize, tokens, outcome: g.z, recs });
    pos += recs.length;
    return false;
  }
  const fd = fs.openSync(filePath, 'r');
  try {
    const chunk = Buffer.alloc(1 << 22);
    let leftover = Buffer.alloc(0), bytes, stop = false;
    while (!stop && (bytes = fs.readSync(fd, chunk, 0, chunk.length, null)) > 0) {
      const buf = Buffer.concat([leftover, chunk.subarray(0, bytes)]);
      let start = 0, nl;
      while ((nl = buf.indexOf(10, start)) !== -1) {
        if (processLine(buf.toString('utf8', start, nl))) { stop = true; break; }
        start = nl + 1;
      }
      leftover = stop ? Buffer.alloc(0) : Buffer.from(buf.subarray(start));
    }
    if (!stop && leftover.length) processLine(leftover.toString('utf8'));
  } finally { fs.closeSync(fd); }
  if (tornError) console.warn(`train-search-values: dropped torn final line in ${filePath}`);
  return pos;
}

// No shuffle: the file is already in random generation order, so streaming in
// file order each epoch is sufficient.
console.log(`files=${FILES.length}  lr=${LR}  alpha=${ALPHA}` +
  (HINGE ? `  HINGE margin=${MARGIN} negK=${NEG_K} hingeLr=${HINGE_LR} openSkip=${OPENING_SKIP}` : '') +
  `  specs=${JSON.stringify(specs)}  (streaming per epoch)`);
console.log(`Out: ${SAVE_PATH}`);
console.log();
// Training columns (left), then test/eval columns (right): ladr.
console.log(['epoch'.padStart(5), 'pos'.padStart(7), 'elapsed'.padStart(7), 'tPos'.padStart(5),
             'acc'.padStart(5), 'avgW'.padStart(6), 'maxW'.padStart(6),
             ...(HINGE ? ['hViol'.padStart(5)] : []), 'ladr'.padStart(5)].join('  '));

// Two-step logistic TD (same player to move) on the recorded trajectories,
// anchored by the actual game OUTCOME — exactly train-vlibpat's update, but
// offline:
//   bootstrap:  target(s_t) = (1−α)·V(s_{t+2}) + α·searchValue(s_t)   [α=0 default]
//   terminal:   the trailing two positions regress to the game outcome z.
// The TD bootstrap toward the model's own value two moves later keeps the
// value SURFACE smooth (siblings differentiable); the clean outcome anchor
// gives an unbiased target — the noisy search value as the anchor (the old
// version) made the model accurate-on-average but unable to rank moves.  α>0
// optionally re-injects the search value into the bootstrap target.
// Logit (pre-sigmoid score) of a candidate feature set under the live weights.
function scoreLogit(features) {
  let z = 0;
  const { keys, pols, count } = features;
  for (let i = 0; i < count; i++) z += pols[i] * (weights.get(keys[i]) ?? 0);
  return z;
}
// Margin update: nudge this candidate's logit in direction `dir` (+1 up / -1 down).
function hingeUpdate(features, dir, lr) {
  const n = features.count;
  if (n === 0) return;
  const step = lr * dir / n;
  const { keys, pols } = features;
  for (let i = 0; i < n; i++) { const k = keys[i]; weights.set(k, (weights.get(k) ?? 0) + pols[i] * step); }
}

// Hinge at position s: push the chosen move's candidate logit past NEG_K sampled
// legal siblings by MARGIN (pol=+1 if BLACK to move, −1 if WHITE).  Only at
// non-explore, post-opening positions, where the played move IS the search argmax.
let hingeViol = 0, hingeChecks = 0;
function hingeAt(game, game3, chosen) {
  const pol = game.current === BLACK ? 1 : -1;
  const area = game.N * game.N;
  const fc = extractFeatures(game, prepSpecs, true, chosen, game3);
  let done = 0, tries = 0;
  while (done < NEG_K && tries < NEG_K * 6) {
    tries++;
    const a = (Math.random() * area) | 0;
    if (a === chosen || !game.isLegal(a) || game.isTrueEye(a)) continue;
    done++;
    const fs = extractFeatures(game, prepSpecs, true, a, game3);
    hingeChecks++;
    if (pol * (scoreLogit(fc) - scoreLogit(fs)) < MARGIN) {   // margin violated
      hingeViol++;
      hingeUpdate(fc, pol, HINGE_LR);
      hingeUpdate(fs, -pol, HINGE_LR);
    }
  }
}

let intervalCorrect = 0, intervalN = 0;
function trainGame(g) {
  const game = new Game2(g.boardSize);   // free initial stone (matches gen-search-values)
  const game3 = game3FromGame2(game);    // lockstep mirror incl. the opening stone
  let played = 0;
  let prev2 = null, prev1 = null;   // { features, searchVal } two/one positions back
  const vals = [];
  for (const rec of g.recs) {
    while (played < rec.histLen) {
      const idx = parseMove(g.tokens[played], g.boardSize);
      game.play(idx); game3.play(idx); played++;
    }
    const f = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(f, weights);
    vals.push(f.val);

    if (prev2 !== null) {
      const target = (1 - ALPHA) * f.val + ALPHA * prev2.searchVal;
      tdUpdate(prev2.features, target, LR);
    }
    // Hinge: at non-explore, post-opening positions the played move is the
    // search argmax — rank it above sampled siblings.  `game` is at position s.
    if (HINGE && !rec.explore && rec.histLen >= OPENING_SKIP) {
      const chosen = parseMove(g.tokens[rec.histLen], g.boardSize);
      if (chosen !== PASS && game.isLegal(chosen) && !game.isTrueEye(chosen)) hingeAt(game, game3, chosen);
    }
    prev2 = prev1;
    prev1 = { features: f, searchVal: rec.value };
  }
  // The replay already left `game` at the final recorded position — area-score
  // it here (once; cached) for the outcome.  No extra replay.
  if (g.outcome === undefined) g.outcome = game.estimateWinner() === BLACK ? 1 : 0;
  const outcome = g.outcome;
  // Terminal anchor: the trailing two positions regress to the game outcome.
  for (const p of [prev2, prev1]) if (p) tdUpdate(p.features, outcome, LR);
  // acc diagnostic.
  for (const v of vals) { if ((v >= 0.5) === (outcome >= 0.5)) intervalCorrect++; intervalN++; }
}

const ladderArgmax = gm => ({ move: gm.gameOver ? PASS : search1ply(gm, game3FromGame2(gm)) });

const t0 = performance.now();
let nextPrintAt = t0 + 1000;
let totalTrained = 0;
let epoch = 0;
const EMA_PERIOD = 1000;
let sinceEMA = 0;

function printStats() {
  const { passed, total } = Ladders.evalLadders(ladderArgmax, { oversample: 1 });
  let wAbs = 0, wMax = 0;
  for (const w of weights.values()) { const a = Math.abs(w); wAbs += a; if (a > wMax) wMax = a; }
  const el = performance.now() - t0;
  console.log([
    Util.fmt4(epoch)                       .padStart(5),
    Util.fmt4(totalTrained)                .padStart(7),
    Util.fmtMs(el)                         .padStart(7),
    Util.fmtMs(el / Math.max(1, totalTrained)).padStart(5),
    Util.fmtRatio4(intervalN ? intervalCorrect / intervalN : 0).padStart(5),
    (weights.size ? (wAbs / weights.size) : 0).toFixed(4).padStart(6),
    wMax.toFixed(3)                        .padStart(6),
    ...(HINGE ? [Util.fmtRatio4(hingeChecks ? hingeViol / hingeChecks : 0).padStart(5)] : []),
    Util.fmtRatio4(passed / total)         .padStart(5),
  ].join('  '));
  intervalCorrect = 0; intervalN = 0;
  hingeViol = 0; hingeChecks = 0;
  const src = weightsEMAInit ? weightsEMA : weights;
  saveWeights(SAVE_PATH, { weights: src, specs, opts: tacticsOpts, preparedSpecs: prepSpecs });
}

function onTrain(g) {
  trainGame(g);
  totalTrained += g.recs.length;
  if (++sinceEMA >= EMA_PERIOD) { applyEMA(EMA_ALPHA); sinceEMA = 0; }
  if (performance.now() >= nextPrintAt) { printStats(); nextPrintAt = performance.now() + 1000; }
}

while (EPOCHS === 0 || epoch < EPOCHS) {
  epoch++;
  const before = totalTrained;
  for (const f of FILES) streamFile(f, onTrain, POS_LIMIT);
  if (epoch === 1 && totalTrained === before) { console.error('no games loaded'); process.exit(1); }
}
printStats();
console.log(`Reached --epochs ${EPOCHS} — saved ${SAVE_PATH}`);
