'use strict';

// train-featurepol.js — REINFORCE self-play trainer for the featurepol policy.
//
//   node train-featurepol.js --spec 'capture6,atari4+stones4' [options]
//
// Plays self-play games with the current policy and applies a REINFORCE update
// after each game (advantage = outcome − EMA baseline), periodically evaluating
// the greedy policy against a reference agent and saving the checkpoint.

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, BLACK, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help || (!opts.spec && !opts.load)) {
  console.log(`Usage: node train-featurepol.js --spec '<spec>' [options]
  --spec S          feature spec; ',' = independent spaces, '+' = conjunction
                    (required unless --load is given, which supplies its spec)
                    term types: stones{4,8,12,20}  adjLib<n>  stone8AdjLib<n>  capture<n>  atari<n>  selfAtari<n>  lib<n>  joins  flags  ko  anyKo  local  koSolve  dist<n>
                                ladderStatus  {urgentKill,urgentSave,wastedExtend,wastedAttack}<n>
  --train-size N | --size N   self-play board size (default 9)
  --eval-size N     evaluation board size (default 13)
  --lr F            learning rate (default 0.02)
  --reward-ema F    EMA decay for the reward baseline; 0 disables (default 0.99)
  --weight-decay F  decoupled L2 shrink per update (default 0.000002; 0 = off)
  --temperature F   softmax sampling temperature for training (default 1)
  --eval-agent S    reference agent in ai/ for eval games; if omitted, eval is skipped
  --ladder-file P   evalladders2 suite scored each status print (ladr column)
  --md-file P       evalmovedetails positions scored each status print; greedy
                    move-quality RMS gap to best (mdRms column)
  --load PATH       resume from saved weights
  --save PATH       where to save (default out/featurepol-<rand>.js)`);
  process.exit(opts.help ? 0 : 1);
}

const TRAIN_SIZE  = parseInt(opts['train-size'] || opts.size || '9', 10);
const EVAL_SIZE   = parseInt(opts['eval-size'] || '13', 10);
const LR          = parseFloat(opts.lr || '0.02');
const REWARD_EMA  = parseFloat(opts['reward-ema'] || '0.99');
const WEIGHT_DECAY = parseFloat(opts['weight-decay'] || '0.000002');
const TEMPERATURE = Math.max(0, parseFloat(opts.temperature || '1'));
const EVAL_AGENT  = opts.eval || opts['eval-agent'] || null;   // no default — eval is skipped unless given
const LADDER_FILE = opts['ladder-file'] || null;   // evalladders2 suite scored each status print (ladr column)
const MD_FILE     = opts['md-file'] || null;       // evalmovedetails positions scored each status print (mdRms column)
const SAVE_PATH   = opts.save || `out/featurepol-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH   = opts.load || null;

// Load the saved model up front (if present) so it can supply the spec when
// --spec is omitted — i.e. just continue training the model as it stands.
const loaded = (LOAD_PATH && fs.existsSync(LOAD_PATH))
  ? FeaturePol.loadModel({ name: 'featurepol', path: LOAD_PATH })
  : null;
if (LOAD_PATH && !loaded) console.warn(`Warning: --load file not found: ${LOAD_PATH}`);

const SPEC = opts.spec || (loaded && loaded.spec.str);
if (!SPEC) {
  console.error('Error: --spec is required unless --load points to an existing model.');
  process.exit(1);
}

// The library throws on a malformed spec (it stays browser-safe / Node-API-free);
// at the CLI boundary turn that into a clean stderr message + non-zero exit.
let weights;
try {
  weights = FeaturePol.createWeights({ spec: SPEC });
} catch (e) {
  console.error(`Error: ${e.message}`);
  process.exit(1);
}
let ema = 0;
let totalUpdates = 0;
// avgW: running frequency-weighted mean |weight| over every weight update this run.
const wStats = { absSum: 0, count: 0 };

if (loaded) {
  ema = loaded.ema; totalUpdates = loaded.totalUpdates;
  // Resume into the CLI-spec model (built above) rather than adopting the saved
  // spec.  A feature space's hashes depend only on its own spec substring, so
  // importing the loaded (hash → weight) table makes every space present in BOTH
  // specs resume with its trained weights; CLI-only spaces start at 0; saved-only
  // spaces are dropped (their imported weights are never regenerated under the new
  // spec, so they sit unused).
  const cliSpaces   = new Set(weights.spec.spaces.map(s => s.str));
  const savedSpaces = new Set(loaded.weights.spec.spaces.map(s => s.str));
  const kept    = [...cliSpaces].filter(s => savedSpaces.has(s));
  const added   = [...cliSpaces].filter(s => !savedSpaces.has(s));
  const removed = [...savedSpaces].filter(s => !cliSpaces.has(s));
  let imported = 0;
  for (const [hash, srcIdx] of loaded.weights.map) {
    weights.vals[FeaturePol.internKey(weights, hash)] = loaded.weights.vals[srcIdx];
    imported++;
  }
  console.log(`Resumed from ${LOAD_PATH}: imported ${imported} weights, ema=${ema.toFixed(3)}`);
  console.log(`  spaces: kept ${kept.length}` +
              `, added ${added.length}${added.length ? ` (${added.join(', ')})` : ''}` +
              `, removed ${removed.length}${removed.length ? ` (${removed.join(', ')})` : ''}`);
  if (kept.length === 0) {
    console.warn('  WARNING: CLI --spec shares no feature space with the saved model — nothing carried over (effectively a fresh start).');
  }
}

const evalGetMove = EVAL_AGENT ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove : null;

// Greedy featurepol policy as a getMove(game) for the ladder / move-detail suites.
const _suiteStates = new Map();
function fpGreedyMove(game) {
  let st = _suiteStates.get(game.N);
  if (!st) { st = FeaturePol.createState(game.N, weights.spec); _suiteStates.set(game.N, st); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  return { move: FeaturePol.greedyMove(game, st, weights, game3) };
}
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;

function saveWeights() {
  fs.mkdirSync(path.dirname(SAVE_PATH), { recursive: true });
  fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema, totalUpdates }));
}

// ── One self-play game + REINFORCE update ─────────────────────────────────────
function trainGame(N) {
  const game  = new Game2(N);
  const game3 = game3FromGame2(game);
  const maxMoves = N * N * 4;
  const tStart = Date.now();
  const steps = [];
  const state = FeaturePol.createState(N, weights.spec);
  const needTac = weights.spec.needsLadder;

  let moves = 0;
  while (!game.gameOver && moves < maxMoves) {
    const player = game.current;
    const choice = FeaturePol.policyMove(game, state, weights, Math, needTac ? game3 : undefined, TEMPERATURE);
    if (choice.index >= 0) {
      const n = state.count;
      const nk = state.keyOff[n];
      const keys = new Int32Array(nk);
      keys.set(state.keys.subarray(0, nk));
      const keyOff = new Int32Array(n + 1);
      keyOff.set(state.keyOff.subarray(0, n + 1));
      const probs = new Float64Array(n);
      probs.set(state.probs.subarray(0, n));
      steps.push({ player, chosenIndex: choice.index, count: n, keys, keyOff, probs, touched: state.touched });
    }
    game.play(choice.move);
    game3.play(choice.move);
    moves++;
  }

  const winner = game.calcWinner();
  const outcomeBlack = winner === BLACK ? 1 : (winner === 0 ? 0 : -1);

  let weightUpdates = 0;
  for (const s of steps) {
    const R = s.player === BLACK ? outcomeBlack : -outcomeBlack;
    const adv = REWARD_EMA > 0 ? (R - ema) : R;
    weightUpdates += FeaturePol.reinforceUpdate(s, s.chosenIndex, adv, weights, LR, WEIGHT_DECAY, wStats) || 0;
  }
  if (REWARD_EMA > 0 && steps.length > 0) ema = REWARD_EMA * ema + (1 - REWARD_EMA) * outcomeBlack;

  let maxProbSum = 0;
  for (const s of steps) { let mx = 0; for (let i = 0; i < s.count; i++) if (s.probs[i] > mx) mx = s.probs[i]; maxProbSum += mx; }

  return { elapsedMs: Date.now() - tStart, weightUpdates, maxProbSum, maxProbN: steps.length };
}

// ── Eval vs reference ─────────────────────────────────────────────────────────
function evalVsReference(N, nGames) {
  const state = FeaturePol.createState(N, weights.spec);
  const needTac = weights.spec.needsLadder;
  let wins = 0;
  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game = new Game2(N);
    const game3 = game3FromGame2(game);
    for (let r = 0; r < 4 && !game.gameOver; r++) { const mv = game.randomLegalMove(); game.play(mv); game3.play(mv); }
    let m = 0;
    while (!game.gameOver && m++ < N * N * 4) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = FeaturePol.greedyMove(game, state, weights, needTac ? game3 : undefined);
      } else {
        const mv = evalGetMove(game);
        idx = mv && mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx); game3.play(idx);
    }
    const winner = game.calcWinner();
    if ((winner === BLACK) === policyIsBlack) wins++;
  }
  return wins;
}

console.log(`spec='${weights.spec.str}'  spaces=${weights.nSpaces}  needsLadder=${weights.spec.needsLadder}`);
console.log(`lr=${LR}  reward-ema=${REWARD_EMA}  weight-decay=${WEIGHT_DECAY}  temperature=${TEMPERATURE}`);
console.log(`train-size=${TRAIN_SIZE}` + (EVAL_AGENT ? `  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT}` : '  (no eval)'));
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log();
// Fixed per-column widths; header and data rows pad to the same widths so they
// line up regardless of the individual formatters' string lengths.
// winRatio column: "wr(g)/avg(ga)" — wr/avg are fmtRatio4, g/ga are fmt4 game
// counts (this interval's, and the rolling-half window).  Fixed 21 chars wide.
const COLS = ['elapsed', 'game', 'tGm', 'nWts', 'avgW', 'maxP',
  ...(EVAL_AGENT ? ['winRatio'] : []), ...(ladderCases ? ['ladr'] : []), ...(mdPositions ? ['mdRms'] : [])];
const COLW = [7, 5, 6, 6, 7, 5,
  ...(EVAL_AGENT ? [21] : []), ...(ladderCases ? [6] : []), ...(mdPositions ? [6] : [])];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

const t0 = Date.now();
let nextPrintAt = t0 + 1000, lastPrintAt = t0;
let g = 0, lastG = 0, elapsedAcc = 0, maxPSum = 0, maxPN = 0;
const evalHistory = [];   // per-game eval results (1/0); the rolling-half avg (wrAv) uses this

while (true) {
  g++;
  const r = trainGame(TRAIN_SIZE);
  totalUpdates += r.weightUpdates;
  elapsedAcc += r.elapsedMs; maxPSum += r.maxProbSum; maxPN += r.maxProbN;

  if (Date.now() >= nextPrintAt) {
    let wNz = 0;
    for (let i = 0; i < weights.size; i++) if (weights.vals[i] !== 0) wNz++;
    const avgW = wStats.count > 0 ? wStats.absSum / wStats.count : 0;
    const row = [
      Util.fmtMs(Date.now() - t0),
      Util.fmt4(g),
      Util.fmtMs(elapsedAcc / Math.max(1, g - lastG)),
      Util.fmt4(wNz),
      avgW.toFixed(4),
      Util.fmtRatio4(maxPN > 0 ? maxPSum / maxPN : 0),
    ];
    if (EVAL_AGENT) {
      const evalBudget = (Date.now() - lastPrintAt) * 0.2, evalStart = Date.now();
      let evalWins = 0, evalGames = 0;
      while (evalGames < 1000 && Date.now() - evalStart < evalBudget) {
        const w1 = evalVsReference(EVAL_SIZE, 1);
        evalHistory.push(w1); evalWins += w1; evalGames++;
      }
      // avg: rolling win ratio over the most recent half of all eval games.
      const avgHalf = Math.max(1, Math.floor(evalHistory.length / 2));
      const avgWR = evalHistory.length > 0
        ? evalHistory.slice(-avgHalf).reduce((s, x) => s + x, 0) / avgHalf : 0;
      row.push(`${Util.fmtRatio4(evalGames > 0 ? evalWins / evalGames : 0)}(${Util.fmt4(evalGames)})` +
               `/${Util.fmtRatio4(avgWR)}(${Util.fmt4(avgHalf)})`);
    }
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, fpGreedyMove, { budgetMs: 1, oversample: 1 });
      row.push(Util.fmtRatio4(total ? passed / total : 0));
    }
    if (mdPositions) {
      const { rmsErr } = evalPositions(fpGreedyMove, mdPositions, 0);
      row.push(Util.fmtRatio4(rmsErr));
    }
    printRow(row);
    saveWeights();
    maxPSum = 0; maxPN = 0; elapsedAcc = 0; lastG = g;
    nextPrintAt = t0 + Math.round((Date.now() - t0) * 1.4);
    lastPrintAt = Date.now();
  }
}
