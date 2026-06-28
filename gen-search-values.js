'use strict';

// gen-search-values.js — generate (position, search-value) training data for
// the offline vlibpat value trainer.  Self-play with puct-static using move
// ordering but NO pruning (NPAT_K forced to 0); at each position record the
// absolute P(BLACK wins) value the search returns.  The offline trainer
// regresses V(s) toward this value instead of a TD bootstrap: the search
// reads ladders (npat priors + tree exploration, 18/19 on evalladders) even
// though the vlibpat leaf eval alone does not, so the value target carries
// ladder knowledge that raw TD targets cannot.
//
// Pure data generation — no model, no in-run training (so memory is bounded).
// Games are flushed to disk whole; runs forever (Ctrl-C to stop) unless
// --limit is given.
//
// Usage:
//   node gen-search-values.js [--playouts 10] [--size 11] [--temp-moves 20]
//                             [--temp 1] [--epsilon 0.1] [--seed n]
//                             [--limit games] [--save path.ndjson]

process.env.NPAT_K = '0';   // move ordering, no pruning — set before the agent loads

const fs = require('fs');
const { performance } = require('perf_hooks');
const { Game2, BLACK, PASS, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');
const PuctStatic = require('./ai/puct-static.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['epsilon', 'limit', 'playouts', 'save', 'seed', 'size', 'temp', 'temp-moves']);
if (opts.help) {
  console.log('Usage: node gen-search-values.js [--playouts 10] [--size 11] [--temp-moves 20] ' +
              '[--temp 1] [--epsilon 0.1] [--seed n] [--limit games] [--save path]');
  process.exit(0);
}

const playouts  = parseInt(opts.playouts || '10', 10);
const size      = parseInt(opts.size || '11', 10);
const tempMoves = parseInt(opts['temp-moves'] || '20', 10);
const temp      = parseFloat(opts.temp || '1');
const epsilon   = parseFloat(opts.epsilon !== undefined ? opts.epsilon : '0.1');
const seed      = opts.seed !== undefined ? parseInt(opts.seed, 10) : undefined;
const gameLimit = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const SAVE_PATH = opts.save || `out/searchval-p${playouts}-${Math.random().toString(36).slice(2, 8)}.ndjson`;

const rng = makeRng(seed);

fs.writeFileSync(SAVE_PATH,
  `# gen-search-values.js  agent=puct-static(NPAT_K=0)  size=${size}  playouts=${playouts}` +
  `  temp-moves=${tempMoves}  temp=${temp}  epsilon=${epsilon}${seed !== undefined ? `  seed=${seed}` : ''}\n`);
console.log(`size=${size}  playouts=${playouts}  temp-moves=${tempMoves}  temp=${temp}  epsilon=${epsilon}`);
console.log(`Out: ${SAVE_PATH}`);
console.log();
console.log(['games'.padStart(5), 'pos'.padStart(7), 'elapsed'.padStart(7), 'tMv'.padStart(5)].join('  '));

// Sample a child index ∝ visits^(1/temp) for opening diversity.
function sampleByVisits(children) {
  const invT = 1 / temp;
  let sum = 0;
  const w = children.map(c => { const x = Math.pow(c.visits, invT); sum += x; return x; });
  let r = rng.random() * sum;
  for (let i = 0; i < children.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return children.length - 1;
}

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let gamesDone = 0;
let posCount  = 0;
let moveMs    = 0;

function printStats() {
  const el = performance.now() - startTime;
  console.log([
    Util.fmt4i(gamesDone)          .padStart(5),
    Util.fmt4i(posCount)           .padStart(7),
    Util.fmtMs(el)                 .padStart(7),
    Util.fmtMs(moveMs / posCount)  .padStart(5),
  ].join('  '));
}

while (gamesDone < gameLimit) {
  const game     = new Game2(size);   // free initial stone (applyFirstMove=true); replayers must match with Game2(size) + game3FromGame2(game)
  const maxMoves = size * size * 4;
  // Compact per-game record (one NDJSON line per game):
  //   moves   — the full move list as a comma-separated string (no quotes)
  //   values  — search value (4 dp) of the position BEFORE each move
  //   explore — indices of the moves that were epsilon-exploration moves
  //   z       — final outcome, P(BLACK wins)
  const moves      = [];
  const values     = [];
  const exploreIdx = [];
  let decision   = 0;

  while (!game.gameOver) {
    if (game.moveCount >= maxMoves) break;

    const t0 = performance.now();
    const r = PuctStatic.getMove(game, 0, { playoutLimit: playouts, rng });
    moveMs += performance.now() - t0;

    // Continue the game: epsilon-random for exploration, visit-sampled for the
    // opening (diversity), else the search's chosen move.
    let move, isExplore = false;
    if (epsilon > 0 && rng.random() < epsilon) {
      move = game.randomLegalMove(rng);
      isExplore = true;
    } else if (decision < tempMoves && r.children && r.children.length) {
      const placements = r.children.filter(c => c.move !== PASS && c.visits > 0);
      move = placements.length ? placements[sampleByVisits(placements)].move : r.move;
    } else {
      move = r.move;
    }

    values.push(r.value !== undefined ? Math.round(r.value * 1e4) / 1e4 : null);
    if (isExplore) exploreIdx.push(moves.length);   // index of this move
    if (!game.play(move)) { console.error(`illegal move ${move}`); process.exit(1); }
    moves.push(coordStr(move, size));
    decision++;
    posCount++;

    const now = performance.now();
    if (now - lastPrintTime >= printPeriodMs) {
      printStats();
      lastPrintTime = performance.now();
      printPeriodMs = Math.round(printPeriodMs * 1.5);
    }
  }

  gamesDone++;
  // Final outcome (P(BLACK wins)): 1 if BLACK won, else 0.
  const z = game.calcWinner() === BLACK ? 1 : 0;
  fs.appendFileSync(SAVE_PATH,
    JSON.stringify({ boardSize: size, moves: moves.join(','), values, explore: exploreIdx, z }) + '\n');
}

printStats();
console.log(`Reached --limit ${gameLimit} games — saved ${SAVE_PATH}`);
