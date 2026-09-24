#!/usr/bin/env node
'use strict';

// train-jpats.js — logistic TD(2) self-play trainer for jpats gated patterns.
//
// Same learning rule as train-hpatterns.js:
//   V(s)  = sigma( sum polarity_i * w[key_i] )        (absolute, P(BLACK wins))
//   dw_k  = (lr / n_features) * (target - V) * polarity_k
//   target = V(s_{t+2}) while bootstrapping, else the terminal outcome.
//
// What differs is the feature set, and two rules it imposes on the loop:
//
//   collectPending is set ONLY for the trajectory positions this run updates
//   on.  Candidate evaluation inside search1ply must never pass it: search
//   touches ~N^2 positions per move against one played, so rolling for
//   admission there would inflate the effective rate by two orders of
//   magnitude and fill the table with patterns from moves never played.
//
//   maybePrune runs ONCE PER GAME.  A prune is an O(n) collect/sort/rebuild of
//   the whole table; checking after every admitted pattern turns the converged
//   regime into thousands of full rebuilds per hundred games.
//
// jpats has no incremental delta yet, so a 1-ply search costs one full
// extraction per candidate.  Off-policy training (--on-policy 0 --ext <agent>)
// skips the search entirely and is much faster per game.

const path = require('path');
const fs   = require('fs');
const { Game2, BLACK, PASS, setKomi, KOMI } = require('./game2.js');
const J = require('./jpats-lib.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['min-psize', 'max-psize', 'p-add', 'max-weights', 'prune-ratio', 'smooth-weights',
   'epsilon', 'eval', 'eval-size', 'ext', 'komi', 'limit', 'load', 'lr',
   'on-policy', 'save', 'size', 'train-size']);
if (opts.help) {
  console.log(`Usage: node train-jpats.js [options]

Logistic TD(2) self-play trainer for jpats gated hierarchical patterns.  Runs
indefinitely unless --limit is given; the checkpoint is written at every print.

  --min-psize N      smallest size that emits features and anchors the gate;
                    sizes below it are hash scaffolding only (default 3)
  --max-psize N      largest size to climb to (default boardSize-1)
  --p-add F         admission probability per sighting of an absent pattern
                    (default 0.1; a pattern seen n times enters with
                    probability 1-(1-F)^n, so ~7 sightings at the default)
  --max-weights N   table ceiling; admission stops here until a prune makes
                    room (default 4000000)
  --prune-ratio F   a prune drops this fraction of the weights admitted since
                    the last prune, taking those nearest zero (default 0.5)
  --train-size N    self-play board size (default 13)
  --eval-size N     evaluation board size (default 13)
  --size N          sets both of the above
  --komi K          auto | auto:<start> | <number>; auto (default) steps komi
                    by +/-1 every 500 games while black's win share sits
                    outside [45%, 55%].  Eval games use a fixed komi.
  --limit N         stop after N games (default 0 = run indefinitely)

  --lr F            step size for the TD update (default 0.3)
  --smooth-weights A  Polyak EMA decay applied every 1000 games; 0 = off
                    (default 0.9)
  --epsilon F       share of moves played uniformly at random (default 0.1)
  --on-policy F     share of the NON-random moves taken from this model's own
                    1-ply search; the rest come from --ext (default 1)
  --ext AGENT       ai/<name>.js supplying off-policy moves; only consulted
                    when --on-policy < 1.  Not necessarily cheaper than the
                    model's own search: with the incremental delta a 1-ply
                    search runs ~4ms/game at size 8, while a full-spec
                    featurepol reference costs ~8ms/game.

  --load PATH       resume from a checkpoint (restores komi, and the geometry
                    for whichever of --min-psize/--max-psize you did NOT pass;
                    an explicit flag wins, so you can raise max-psize to keep
                    growing the pyramid from an existing model)
  --save PATH       checkpoint path (default out/jpats-<random>.js)
  --eval AGENT      ai/<name>.js played as the reference in test games
                    (default: none, which disables the test games)
  --help            show this message`);
  process.exit(0);
}

const TRAIN_SIZE = parseInt(opts['train-size'] || opts.size || '13', 10);
const EVAL_SIZE  = parseInt(opts['eval-size']  || opts.size || '13', 10);
const SAVE_PATH  = opts.save || `out/jpats-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;
const EVAL_AGENT = opts.eval || '';
const EXT_AGENT  = opts.ext  || '';
const EPSILON    = Math.min(parseFloat(opts.epsilon || '0.1'), 1);
const ON_POLICY  = Math.min(parseFloat(opts['on-policy'] || '1'), 1);
const LR         = parseFloat(opts.lr || '0.3');
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');
const EMA_PERIOD = 1000;
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;

// Whether the geometry was named on the command line.  An explicit flag beats
// the checkpoint on a resume — raising max-psize to let the climb go further
// (and admit larger patterns) is a normal thing to want.  Without a flag the
// checkpoint wins, since a model's features only mean anything under the
// gating that produced them.
const MIN_PSIZE_GIVEN = opts['min-psize'] !== undefined;
const MAX_PSIZE_GIVEN = opts['max-psize'] !== undefined;
const MIN_PSIZE    = parseInt(opts['min-psize'] || '3', 10);
const MAX_PSIZE    = opts['max-psize'] !== undefined ? parseInt(opts['max-psize'], 10) : TRAIN_SIZE - 1;
const P_ADD       = parseFloat(opts['p-add'] || '0.1');
const MAX_WEIGHTS = parseInt(opts['max-weights'] || '4000000', 10);
const PRUNE_RATIO = parseFloat(opts['prune-ratio'] || '0.5');

let AUTO_KOMI = true;
if (opts.komi !== undefined) {
  const m = /^auto(?::(-?[0-9.]+))?$/.exec(opts.komi);
  if (m) { if (m[1] !== undefined) { setKomi(TRAIN_SIZE, parseFloat(m[1])); setKomi(EVAL_SIZE, parseFloat(m[1])); } }
  else { AUTO_KOMI = false; setKomi(TRAIN_SIZE, parseFloat(opts.komi)); setKomi(EVAL_SIZE, parseFloat(opts.komi)); }
}
const EVAL_KOMI = KOMI(EVAL_SIZE);
const KOMI_WINDOW = 500;
let komiGames = 0, komiBlackWins = 0, komiSum = 0, komiSumGames = 0;

// ── Model ─────────────────────────────────────────────────────────────────────

let model = J.createModel({ minPSize: MIN_PSIZE, maxPSize: MAX_PSIZE, pAdd: P_ADD,
                            maxWeights: MAX_WEIGHTS, pruneRatio: PRUNE_RATIO });
let PRIOR_TRAIN_MS = 0;
let wAbsSum = 0, wUpdateCount = 0;

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const raw = require(path.resolve(LOAD_PATH));
    model.weights = J.weightsMap(raw);
    if (raw.minPSize !== undefined && !MIN_PSIZE_GIVEN) model.minPSize = raw.minPSize;
    if (raw.maxPSize !== undefined && !MAX_PSIZE_GIVEN)
      model.maxPSize = raw.maxPSize === null ? Infinity : raw.maxPSize;
    if (AUTO_KOMI && raw.komi !== undefined) setKomi(TRAIN_SIZE, raw.komi);
    PRIOR_TRAIN_MS = raw.trainMs || 0;
    console.log(`Loaded ${model.weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}

const evalGetMove = EVAL_AGENT ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove : null;
const extGetMove  = EXT_AGENT  ? require(path.join(__dirname, 'ai', EXT_AGENT  + '.js')).getMove : null;

// ── Persistence (same int16 packing as train-hpatterns.js) ────────────────────

function saveModel(filePath, m) {
  const useEMA = m.weightsEMAInit;
  const source = useEMA ? m.weightsEMA : m.weights;
  const count = source.size;
  let maxAbs = 0;
  source.forEach((k, v) => { const a = v < 0 ? -v : v; if (a > maxAbs) maxAbs = a; });
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;
  const keys = new Int32Array(count), qvals = new Int16Array(count);
  let i = 0;
  source.forEach((k, v) => {
    keys[i] = k;
    let q = Math.round(v * scale);
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    qvals[i] = q; i++;
  });
  const buf = Buffer.alloc(count * 6);
  Buffer.from(keys.buffer, keys.byteOffset, count * 4).copy(buf, 0);
  Buffer.from(qvals.buffer, qvals.byteOffset, count * 2).copy(buf, count * 4);
  const src = [
    "'use strict';",
    '// Auto-generated by train-jpats.js — do not edit by hand.',
    '// Weights int16-quantised: weight = qvals[i] / scale.',
    'const jpatsModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const minPSize = ${m.minPSize};`,
    `  const maxPSize = ${m.maxPSize === Infinity ? 'null' : m.maxPSize};`,
    `  const komi = ${KOMI(TRAIN_SIZE)};`,
    `  const trainMs = ${PRIOR_TRAIN_MS + (Date.now() - t0)};`,
    `  const weightsAreEMA = ${useEMA};`,
    `  const b64 = '${buf.toString('base64')}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf2 = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf2, 0, count);",
    "  const qvals = new Int16Array(buf2, count * 4, count);",
    "  return { minPSize, maxPSize, komi, trainMs, weightsAreEMA, count, scale, keys, qvals };",
    "})();",
    "if (typeof module !== 'undefined') module.exports = jpatsModel;",
    "else window.jpatsModel = jpatsModel;",
  ].join('\n') + '\n';
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, src);
}

// ── TD update ─────────────────────────────────────────────────────────────────

function tdUpdate(step, target, lr) {
  const w = model.weights;
  const n = step.count;
  if (n === 0) return;
  const scale = lr * (target - step.val) / n;
  for (let i = 0; i < n; i++) {
    const key = step.keys[i], cur = w.get(key);
    if (cur === undefined) continue;               // pruned since extraction
    const next = cur + scale * step.pols[i];
    w.set(key, next);
    wAbsSum += next < 0 ? -next : next; wUpdateCount++;
  }
}

// ── One self-play game ────────────────────────────────────────────────────────

// 1-ply search over the model's own value.  Candidates are scored
// incrementally as z_base + deltaZ over the windows containing the move —
// ~3x cheaper than re-extracting per candidate, and verified to match a full
// extraction exactly.  Capture moves disturb windows around every captured
// stone, so deltaZ declines them (NaN) and those fall back to extraction.
//
// NOTE: collectPending is NOT passed anywhere here.  Candidate positions must
// never grow the model (see the header); admission belongs to the played move.
function search1ply(game, w = model.weights) {
  const area = game.N * game.N;
  const isBlack = game.current === BLACK;
  const coords = [];
  for (let c = 0; c < area; c++) {
    if (!game.isLegal(c) || game.isTrueEye(c)) continue;
    coords.push(c);
  }
  if (game.consecutivePasses > 0 || game.emptyCount < area / 2) coords.push(PASS);
  if (coords.length === 0) return PASS;

  // Fills the hash/depth buffers deltaZ reads; nothing may overwrite them
  // between here and the last candidate, so the fallback re-extracts after use.
  const zBase = J.zOf(J.extractFeatures(game, model), w);

  let best = PASS, bestZ = isBlack ? -Infinity : Infinity;
  for (const c of coords) {
    let z;
    if (c === PASS) z = zBase;
    else {
      const d = J.deltaZ(game, model, w, c);
      if (Number.isNaN(d)) {
        z = J.zOf(J.extractFeatures(game, model, { nextMove: c }), w);
        J.extractFeatures(game, model);         // restore the current-position buffers
      } else z = zBase + d;
    }
    z += (Math.random() - 0.5) * 1e-6;          // break exact ties
    if (isBlack ? z > bestZ : z < bestZ) { bestZ = z; best = c; }
  }
  return best;
}

function playGame(N) {
  const game = new Game2(N, true);
  const maxMoves = N * N * 4;
  const tStart = Date.now();
  let prev1 = null, prev2 = null, moves = 0, pSizeSum = 0, admitted = 0;

  while (!game.gameOver && moves < maxMoves) {
    // The one extraction this position learns from: collectPending on.
    const f = J.extractFeatures(game, model, { collectPending: true });
    f.val = J.evaluateFeatures(f, model.weights);
    admitted += J.admitPending(model, f);
    pSizeSum += f.meanPSize;

    if (prev2 !== null) tdUpdate(prev2, f.val, LR);
    prev2 = prev1;
    prev1 = { keys: f.keys.slice(0, f.count), pols: f.pols.slice(0, f.count), count: f.count, val: f.val };

    let move;
    if (Math.random() < EPSILON)                             move = game.randomLegalMove();
    else if (extGetMove && Math.random() > ON_POLICY)        move = extGetMove(game).move;
    else                                                     move = search1ply(game);
    game.play(move);
    moves++;
  }

  const winner = game.calcWinner();
  const outcome = winner === BLACK ? 1 : (winner === 0 ? 0.5 : 0);
  if (prev2 !== null) tdUpdate(prev2, outcome, LR);
  if (prev1 !== null) tdUpdate(prev1, outcome, LR);
  return { elapsedMs: Date.now() - tStart, moves, outcome, admitted,
           avgPSize: moves > 0 ? pSizeSum / moves : 0 };
}

// ── Eval vs reference ─────────────────────────────────────────────────────────

// Returns the per-game results (1/0), not a total: the rolling average needs
// the actual sequence, and colours alternate so a partial batch stays fair.
function evalVsReference(N, nGames, seat) {
  const out = [];
  // Eval == save: score the weights the checkpoint would actually contain, so
  // the column describes the model on disk rather than the live iterate.
  const evalW = (EMA_ALPHA > 0 && model.weightsEMAInit) ? model.weightsEMA : model.weights;
  for (let g = 0; g < nGames; g++) {
    const modelIsBlack = ((seat + g) % 2 === 0);
    const game = new Game2(N, true);
    // Random opening (3 moves, as train-hpatterns and selfplay's --rand-moves).
    // Without it every game starts from the same position and the policy is
    // deterministic bar the tie-break, so the only variation is the opponent's
    // sampling: the games are heavily correlated and the win ratio swings far
    // more than the nominal game count suggests.
    for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
    let m = 0;
    while (!game.gameOver && m++ < N * N * 4) {
      const move = (game.current === BLACK) === modelIsBlack
        ? search1ply(game, evalW)
        : (evalGetMove(game).move);
      game.play(move);
    }
    out.push((game.calcWinner() === BLACK) === modelIsBlack ? 1 : 0);
  }
  return out;
}

// ── Main loop ─────────────────────────────────────────────────────────────────

console.log(`min-psize=${model.minPSize}  max-psize=${model.maxPSize === Infinity ? TRAIN_SIZE - 1 : model.maxPSize}` +
            `  p-add=${P_ADD}  max-weights=${MAX_WEIGHTS}  prune-ratio=${PRUNE_RATIO}`);
console.log(`lr=${LR}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  smooth-weights=${EMA_ALPHA}  train-size=${TRAIN_SIZE}` +
            (EVAL_AGENT ? `  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT}` : '  (no eval)') +
            (EXT_AGENT ? `  ext=${EXT_AGENT}` : ''));
console.log(`komi=${KOMI(TRAIN_SIZE)}${AUTO_KOMI ? ' (auto)' : ''}  eval-komi=${EVAL_KOMI} (fixed)`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log();

// Column order mirrors train-hpatterns.js: T, TT, game, avgK, tMv, nWts, avgW,
// then the jpats-only pair (psize, prunes), then the eval column.
//   T/TT     this leg's elapsed time, and the total including prior legs
//   avgK     mean komi over the interval (the auto controller moves it)
//   psize    mean largest-present pattern size over the anchors that are
//            present — the adaptive receptive field, not a max
//   prunes   prune passes run so far
//   winRatio "wr(g)/avg(ga)" — this interval's ratio and game count, then the
//            rolling-half window's.  Fixed 21 chars wide.
const COLS = ['T', 'TT', 'game', 'avgK', 'tMv ', 'nWts', 'avgW', 'psize', 'prunes',
              ...(EVAL_AGENT ? ['winRatio'] : [])];
const COLW = [5, 5, 4, 6, 5, 4, 6, 5, 6, ...(EVAL_AGENT ? [21] : [])];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

const t0 = Date.now();
let nextPrintAt = t0 + 1000, g = 0, lastG = 0;
let intervalMs = 0, intervalMoves = 0, pSizeAcc = 0;
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;
const evalHistory = [];
let evalSeat = 0;                 // alternates which colour the model takes
const MAX_EVAL_GAMES = 2000;

while (true) {
  g++;
  const r = playGame(TRAIN_SIZE);
  intervalMs += r.elapsedMs; intervalMoves += r.moves; pSizeAcc += r.avgPSize;

  J.maybePrune(model);                       // once per game, never per admission
  if (EMA_ALPHA > 0 && g % EMA_PERIOD === 0) J.applyEMA(model, EMA_ALPHA);

  komiGames++; komiSum += KOMI(TRAIN_SIZE); komiSumGames++;
  if (r.outcome === 1) komiBlackWins++;
  if (AUTO_KOMI && komiGames >= KOMI_WINDOW) {
    const share = komiBlackWins / komiGames;
    if (share > 0.55) setKomi(TRAIN_SIZE, KOMI(TRAIN_SIZE) + 1);
    else if (share < 0.45) setKomi(TRAIN_SIZE, KOMI(TRAIN_SIZE) - 1);
    komiGames = 0; komiBlackWins = 0;
  }

  const limitReached = LIMIT_GAMES > 0 && g >= LIMIT_GAMES;
  if (limitReached) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let wr = '';
    if (EVAL_AGENT) {
      const trainKomi = KOMI(TRAIN_SIZE);
      setKomi(EVAL_SIZE, EVAL_KOMI);
      // Batch sized by TIME, as train-hpatterns does: keep playing until the
      // test has cost ~30% of the interval's training time, so eval precision
      // grows with the run instead of being pinned at a fixed game count.
      const batch = [];
      const budgetMs = 0.3 * intervalMs;
      while (batch.length < MAX_EVAL_GAMES && Date.now() - tTestStart < budgetMs) {
        for (const r of evalVsReference(EVAL_SIZE, 2, evalSeat)) batch.push(r);
        evalSeat += 2;
      }
      if (batch.length === 0) for (const r of evalVsReference(EVAL_SIZE, 2, evalSeat)) { batch.push(r); }
      setKomi(TRAIN_SIZE, trainKomi);
      for (const r of batch) evalHistory.push(r);
      const latest = batch.reduce((s, x) => s + x, 0) / batch.length;
      const half = Math.max(1, Math.floor(evalHistory.length / 2));
      const avg = evalHistory.slice(-half).reduce((s, x) => s + x, 0) / half;
      wr = `${Util.fmtRatio4(latest)}(${Util.fmt4i(batch.length)})` +
           `/${Util.fmtRatio4(avg)}(${Util.fmt4i(half)})`;
    }
    const tTestMs = Date.now() - tTestStart;
    const nGames = Math.max(1, g - lastG);
    printRow([Util.fmtMs(Date.now() - t0),
              Util.fmtMs(PRIOR_TRAIN_MS + (Date.now() - t0)),
              Util.fmt4i(g),
              Util.fmt4(komiSumGames > 0 ? komiSum / komiSumGames : KOMI(TRAIN_SIZE)),
              Util.fmtMs(intervalMoves > 0 ? intervalMs / intervalMoves : 0),
              Util.fmt4i(model.weights.size),
              (wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0).toFixed(4),
              (pSizeAcc / nGames).toFixed(2),
              Util.fmt4i(model.prunes),
              ...(EVAL_AGENT ? [wr] : [])]);
    saveModel(SAVE_PATH, model);
    lastG = g; intervalMs = 0; intervalMoves = 0; pSizeAcc = 0;
    wAbsSum = 0; wUpdateCount = 0; komiSum = 0; komiSumGames = 0;
    const now = Date.now();
    nextPrintAt = Math.max(Math.min(t0 + Math.round((now - t0) * 1.3), now + MAX_PRINT_GAP_MS), now + tTestMs);
  }

  if (limitReached) { console.log(`\nReached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`); break; }
}
