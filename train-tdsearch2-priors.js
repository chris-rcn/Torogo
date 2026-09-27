'use strict';

// train-tdsearch2-priors.js — distil tdsearch2's per-move searches into its
// location-independent priors (tdsearch2-priors.js) by self-play.
//
// Loop: the agent searches a root; every feature the root exhibits moves its
// prior weight toward prior + residual (agent.distilPriors); a move SAMPLED
// from the actor's softmax is played (TD_ROOT_SELECT softmax, for position
// diversity); repeat.  Both sides are the same agent instance, so the priors
// see both colours.  The agent's online tables reset per game as usual; the
// priors persist and are saved at every progress row.
//
// Usage: node train-tdsearch2-priors.js [options]
//   --size N        board size                              (default 13)
//   --sims N        tdsearch2 simulations per move           (default 1000)
//   --games N       games to play; 0 = until stopped         (default 0)
//   --lr F          distillation step per root              (default 0.1)
//   --save PATH     priors file, saved at every row (default out/tdsearch2-priors-<id>.js)
//   --load PATH     start from an existing priors file
//   --seed N        rng seed (default: random, printed)
//   --md-file PATH  movedetails file for the progress columns (default movedetails_5059.md)
//   --md-limit N    positions of it to score, the same every row; 0 = off (default 1000)
//   Agent knobs come from the environment as usual (TD_*, PPAT_*, TRUNC_*),
//   except that TD_CRITIC_LAYERS defaults to 1,4,9 here (the 3×3 critic prior
//   is distilled from the online 3×3 layer, so a plain run trains all three
//   priors) and TD_ROOT_SELECT to softmax; the environment still overrides both.
//
// Progress columns actMae / crtMae score the PRIORS ALONE (no tables, no
// sims) on the md positions, as evalmovedetails does: actMae plays the actor
// prior's argmax; crtMae plays a one-ply search over the critic prior (each
// candidate played on a clone, the prior summed over the result's 2×2
// windows for the side then to move, best in the mover's view).

const path = require('path');
const Util = require('./util.js');
const { Game2, BLACK } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Priors = require('./tdsearch2-priors.js');
const { create, codes } = require('./ai/tdsearch2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['size', 'sims', 'games', 'lr', 'save', 'load', 'seed', 'md-file', 'md-limit']);
if (opts.help) {
  console.log(`Usage: node train-tdsearch2-priors.js [options]
  --size N        board size                              (default 13)
  --sims N        tdsearch2 simulations per move           (default 1000)
  --games N       games to play; 0 = until stopped         (default 0)
  --lr F          distillation step per root              (default 0.1)
  --save PATH     priors file, saved at every row (default out/tdsearch2-priors-<id>.js)
  --load PATH     start from an existing priors file
  --seed N        rng seed (default: random, printed)
  --md-file PATH  movedetails file for the progress columns (default movedetails_5059.md)
  --md-limit N    positions of it to score, the same every row; 0 = off (default 1000)`);
  process.exit(0);
}
const SIZE     = parseInt(opts.size || '13', 10);
const SIMS     = parseInt(opts.sims || '1000', 10);
const GAMES    = parseInt(opts.games || '0', 10);
const PRIOR_LR = parseFloat(opts['lr'] || '0.1');
const SEED     = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
const SAVE     = opts.save || path.join('out', `tdsearch2-priors-${Util.randomSeed().toString(36)}.js`);
const MD_FILE  = opts['md-file'] || 'movedetails_5059.md';
const MD_LIMIT = parseInt(opts['md-limit'] || '1000', 10);
const mdPositions = MD_LIMIT > 0 ? loadPositions(MD_FILE).slice(0, MD_LIMIT) : [];

const priors = opts.load ? Priors.load(opts.load) : Priors.make();
const agent  = create(Util.makeCfg(null, {
  TD_SIMS: String(SIMS),                      // a fixed sim count, not a time budget
  ...(process.env.TD_CRITIC_LAYERS === undefined ? { TD_CRITIC_LAYERS: '1,4,9' } : {}),   // the 3x3 layer feeds critic9
  ...(process.env.TD_ROOT_SELECT === undefined ? { TD_ROOT_SELECT: 'softmax' } : {}),      // play sampled moves
}));
agent.setPriors(priors);

console.log(`size: ${SIZE}  sims: ${SIMS}  games: ${GAMES || 'unlimited'}  lr: ${PRIOR_LR}  seed: ${SEED}`);
console.log(`save: ${SAVE}${opts.load ? `  load: ${opts.load}` : ''}`);
if (mdPositions.length) console.log(`md: ${MD_FILE}  positions: ${mdPositions.length}`);

// ── Priors-only pickers for the md columns ───────────────────────────────────
const { PASS, EMPTY } = require('./game2.js');
function legalPoints(g) {
  const pts = [];
  for (let p = 0; p < g.N * g.N; p++) if (g.cells[p] === EMPTY && g.isLegal(p) && !g.isTrueEye(p)) pts.push(p);
  return pts;
}
// Actor prior alone: argmax of the (mover, 8-cell code) weight.
function actorPick(g, budgetMs, options) {
  const rng = options.rng, m = g.current === BLACK ? 0 : 1, cells = g.cells, nbr = g._nbr, dnbr = g._dnbr;
  let best = PASS, bestS = -Infinity;
  for (const p of legalPoints(g)) {
    const s = priors.actor9[m * 6561 + codes.code9(cells, dnbr, p, codes.code5(cells, nbr, p))] + rng.random() * 1e-9;
    if (s > bestS) { bestS = s; best = p; }
  }
  return { move: best };
}
// Critic prior alone, one ply: the prior's logit of each candidate's result.
function criticLogit(g) {
  const m = g.current === BLACK ? 0 : 1, cells = g.cells, nbr = g._nbr, dnbr = g._dnbr, area = g.N * g.N;
  let z = 0;
  for (let p = 0; p < area; p++) {
    const k4 = codes.code4(cells, nbr, dnbr, p); if (k4 !== 0) z += priors.critic4[m * 81 + k4];
    const k9 = codes.code9(cells, dnbr, p, codes.code5(cells, nbr, p)) + 6561 * (cells[p] + 1); if (k9 !== 0) z += priors.critic9[m * 19683 + k9];
  }
  return z;
}
function criticPick(g, budgetMs, options) {
  const rng = options.rng, isBlack = g.current === BLACK;
  let best = PASS, bestV = -Infinity;
  for (const p of legalPoints(g)) {
    const c = g.clone(); c.play(p);
    const vB = 1 / (1 + Math.exp(-criticLogit(c)));
    const v = (isBlack ? vB : 1 - vB) + rng.random() * 1e-9;
    if (v > bestV) { bestV = v; best = p; }
  }
  return { move: best };
}
let mdMs = 0;   // time spent in the md columns, kept out of tMv
function mdColumns() {
  if (!mdPositions.length) return [];
  const t = Date.now();
  const cols = [evalPositions(actorPick, mdPositions, 0).maeErr.toFixed(4).padStart(6),
                evalPositions(criticPick, mdPositions, 0).maeErr.toFixed(4).padStart(6)];
  mdMs += Date.now() - t;
  return cols;
}

function meanAbs(arr) { let s = 0, n = 0; for (let i = 0; i < arr.length; i++) if (arr[i] !== 0) { s += Math.abs(arr[i]); n++; } return { mean: n ? s / n : 0, n }; }

// Progress table: header once, rows at games 1, 2, 3, ... growing ×1.4.
console.log([
  'game'.padStart(4), 'moves'.padStart(5), 'avgLen'.padStart(6), 'blkWR'.padStart(6),
  'nzA9'.padStart(5), 'avgA9'.padStart(6), 'nzC9'.padStart(5), 'avgC9'.padStart(6),
  ...(mdPositions.length ? ['actMae'.padStart(6), 'crtMae'.padStart(6)] : []),
  'tMv'.padStart(5), 'elapsed'.padStart(7),
].join('  '));

const t0 = Date.now();
let games = 0, moves = 0, blackWins = 0, nextRow = 1, lastRow = 0;
const rng = makeRng(SEED);
const maxMoves = 3 * SIZE * SIZE + 20;
row();   // baseline: the starting priors (zero, or --load)

function row() {
  const p9 = meanAbs(priors.actor9), pc9 = meanAbs(priors.critic9);
  const md = mdColumns();                // scored first, so elapsed is the wall clock at print time
  const el = Date.now() - t0;
  console.log([
    Util.fmt4i(games), Util.fmt4i(moves).padStart(5), (games ? Util.fmt4(moves / games) : '-').padStart(6),
    (games ? Util.fmtRatio4(blackWins / games) : '-').padStart(6),
    Util.fmt4i(p9.n).padStart(5), p9.mean.toFixed(4).padStart(6),
    Util.fmt4i(pc9.n).padStart(5), pc9.mean.toFixed(4).padStart(6),
    ...md,
    (moves ? Util.fmtMs((el - mdMs) / moves) : '-').padStart(5), Util.fmtMs(el).padStart(7),
  ].join('  '));
  Priors.save(SAVE, priors, `Generated by train-tdsearch2-priors.js — games: ${games}, moves: ${moves}, size: ${SIZE}, sims: ${SIMS}, lr: ${PRIOR_LR}, seed: ${SEED}, elapsed: ${Util.fmtMs(el).trim()}`);
}

while (GAMES === 0 || games < GAMES) {
  const game = new Game2(SIZE, true);
  let n = 0;
  while (!game.gameOver && n < maxMoves) {
    const r = agent.getMove(game, 0, { rng });   // TD_SIMS decides; the budget is ignored
    agent.distilPriors(game, PRIOR_LR);
    game.play(r.move);
    n++;
  }
  games++; moves += n;
  if (game.calcWinner() === BLACK) blackWins++;
  if (games >= nextRow) { row(); lastRow = games; nextRow = Math.max(nextRow + 1, Math.round(nextRow * 1.4)); }
}
if (games !== lastRow) row();
