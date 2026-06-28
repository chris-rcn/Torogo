#!/usr/bin/env node
'use strict';

// compare-moves.js — head-to-head move-quality diagnostic.
//
// Plays games between two agents (colors alternating per game, random
// opening moves for diversity) and, at sampled decisions, asks BOTH agents
// for their move.  Where the choices disagree, each chosen move is labelled
// by a referee search, createmovedetails-style: play the move, ask the
// referee for the resulting position's rootWinRatio, take wr = 1 − that
// (mover's perspective).  The win-ratio difference d = wr(p1 move) −
// wr(p2 move) is then reported bucketed by game phase and by the referee's
// view of p2's standing, plus the worst individual p2 choices — localizing
// where one agent bleeds win ratio on the actual match distribution.
//
// Games continue with the mover's own choice, so positions are drawn from
// real p1-vs-p2 play.  The run stops after --positions disagreements have
// been labelled.
//
// Usage:
//   node compare-moves.js [--p1 npat] [--p2 policy] [--referee prod]
//                         [--budget 1000] [--positions 200] [--sample 0.25]
//                         [--rand-moves 6] [--size 13] [--seed 1]

const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, BLACK, WHITE, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['budget', 'p1', 'p2', 'positions', 'rand-moves', 'referee', 'sample', 'seed', 'size']);

if (opts.help) {
  console.log('Usage: node compare-moves.js [--p1 <name>] [--p2 <name>] [--referee <name>] ' +
              '[--budget <ms>] [--positions <n>] [--sample <f>] [--rand-moves <n>] [--size <n>] [--seed <n>]');
  process.exit(0);
}

const p1Name    = opts.p1 || 'npat';
const p2Name    = opts.p2 || 'policy';
const refName   = opts.referee || 'prod';
const budgetMs  = parseInt(opts.budget || '1000', 10);
const maxLabels = parseInt(opts.positions || '200', 10);
const sample    = parseFloat(opts.sample || '0.25');
const randMoves = parseInt(opts['rand-moves'] || '6', 10);
const size      = parseInt(opts.size || '13', 10);
const seed      = parseInt(opts.seed || '1', 10);

const { getMove: p1 }  = require(path.join(__dirname, 'ai', p1Name + '.js'));
const { getMove: p2 }  = require(path.join(__dirname, 'ai', p2Name + '.js'));
const { getMove: ref } = require(path.join(__dirname, 'ai', refName + '.js'));

const rng = makeRng(seed);
let refSeed = 1000;   // per-invocation referee rng, evalmovedetails-style

// Win ratio (mover's perspective) of playing `move` at `game`: the referee
// searches the resulting position and reports its mover's (the opponent's)
// rootWinRatio.
function labelMove(game, move) {
  const clone = game.clone();
  clone.play(move);
  if (clone.gameOver) {
    return clone.calcWinner() === game.current ? 1 : 0;
  }
  const r = ref(clone, budgetMs, { rng: makeRng(refSeed++) });
  if (r.rootWinRatio === undefined) { console.error('referee did not return rootWinRatio'); process.exit(1); }
  return 1 - r.rootWinRatio;
}

console.log(`p1=${p1Name}  p2=${p2Name}  referee=${refName}@${budgetMs}ms  positions=${maxLabels}  ` +
            `sample=${sample}  rand-moves=${randMoves}  size=${size}`);
console.log();
console.log([
  'disag'  .padStart(5),
  'sampled'.padStart(7),
  'elapsed'.padStart(7),
  'agree'  .padStart(5),
].join('  '));

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;

const area = size * size;
const records = [];   // { phase, p2Standing, d, game, moveNo, m1, m2, wr1, wr2 }
let sampled = 0, agreed = 0;
let gameNo = 0;

function maybePrint(force) {
  const now = performance.now();
  if (!force && now - lastPrintTime < printPeriodMs) return;
  lastPrintTime = now;
  printPeriodMs = Math.round(printPeriodMs * 1.5);
  console.log([
    Util.fmt4i(records.length)             .padStart(5),
    Util.fmt4i(sampled)                    .padStart(7),
    Util.fmtMs(performance.now() - startTime).padStart(7),
    Util.fmtRatio4(agreed / sampled)       .padStart(5),
  ].join('  '));
}

outer:
while (records.length < maxLabels) {
  gameNo++;
  const game = new Game2(size);
  const p1IsBlack = gameNo % 2 === 1;
  const maxMoves = area * 4;
  for (let i = 0; i < randMoves && !game.gameOver; i++) game.play(game.randomLegalMove(rng));

  while (!game.gameOver && game.moveCount < maxMoves) {
    const moverIsP1 = (game.current === (p1IsBlack ? BLACK : WHITE));
    const moverMove = (moverIsP1 ? p1 : p2)(game, 1).move;

    if (rng.random() < sample) {
      sampled++;
      const otherMove = (moverIsP1 ? p2 : p1)(game, 1).move;
      const m1 = moverIsP1 ? moverMove : otherMove;   // p1's choice here
      const m2 = moverIsP1 ? otherMove : moverMove;   // p2's choice here
      if (m1 === m2) {
        agreed++;
      } else {
        const wr1 = labelMove(game, m1);   // mover-perspective; same mover for
        const wr2 = labelMove(game, m2);   // both, so directly comparable
        // p2's standing at this position (referee's view, p2's perspective);
        // position value ≈ the better of the two labelled moves.
        const best = Math.max(wr1, wr2);
        const p2Standing = moverIsP1 ? 1 - best : best;
        records.push({
          phase: 1 - game.emptyCount / area,
          p2Standing,
          d: wr1 - wr2,
          game: gameNo,
          moveNo: game.moveCount,
          m1: coordStr(m1, size),
          m2: coordStr(m2, size),
          wr1, wr2,
        });
        if (records.length >= maxLabels) { maybePrint(true); break outer; }
      }
    }

    game.play(moverMove);
    maybePrint(false);
  }
}
maybePrint(true);

// ── Report ────────────────────────────────────────────────────────────────────

function bucketReport(title, buckets, of) {
  console.log();
  console.log(title);
  console.log(['bucket'.padEnd(18), 'n'.padStart(4), 'p1Better'.padStart(8), 'p2Better'.padStart(8),
               'mean d (p1−p2)'.padStart(15)].join('  '));
  for (const b of buckets) {
    const rs = records.filter(r => of(r) >= b.lo && of(r) < b.hi);
    if (rs.length === 0) { console.log(b.name.padEnd(18), '   0'); continue; }
    const dMean = rs.reduce((s, r) => s + r.d, 0) / rs.length;
    const p1b = rs.filter(r => r.d > 0.005).length;
    const p2b = rs.filter(r => r.d < -0.005).length;
    console.log([
      b.name.padEnd(18),
      String(rs.length).padStart(4),
      Util.fmtRatio4(p1b / rs.length).padStart(8),
      Util.fmtRatio4(p2b / rs.length).padStart(8),
      ((dMean >= 0 ? '+' : '') + dMean.toFixed(4)).padStart(15),
    ].join('  '));
  }
}

console.log();
console.log(`sampled ${sampled} positions, agreement ${Util.fmtRatio4(agreed / sampled)}, labelled ${records.length} disagreements`);
const dAll = records.reduce((s, r) => s + r.d, 0) / records.length;
console.log(`overall mean wr diff (${p1Name} − ${p2Name}): ${(dAll >= 0 ? '+' : '') + dAll.toFixed(4)}`);

bucketReport('By phase:', [
  { name: '  phase<0.15',      lo: 0,    hi: 0.15 },
  { name: '  0.15<=phase<0.35', lo: 0.15, hi: 0.35 },
  { name: '  phase>=0.35',     lo: 0.35, hi: 9 },
], r => r.phase);

bucketReport(`By ${p2Name} standing (referee view):`, [
  { name: '  behind (<0.4)',   lo: 0,   hi: 0.4 },
  { name: '  close (0.4-0.6)', lo: 0.4, hi: 0.6 },
  { name: '  ahead (>=0.6)',   lo: 0.6, hi: 9 },
], r => r.p2Standing);

console.log();
console.log(`Worst ${p2Name} choices:`);
const worst = records.slice().sort((a, b) => b.d - a.d).slice(0, 10);
for (const r of worst) {
  console.log(`  game ${r.game} move ${r.moveNo} (phase ${r.phase.toFixed(2)}): ` +
    `${p1Name} ${r.m1} wr ${r.wr1.toFixed(3)}  ${p2Name} ${r.m2} wr ${r.wr2.toFixed(3)}  d ${(r.d >= 0 ? '+' : '') + r.d.toFixed(3)}`);
}
