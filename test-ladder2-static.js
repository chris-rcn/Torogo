'use strict';

const { Game2, PASS, BLACK, WHITE, parseBoard, coordStr } = require('./game2.js');
const Ladder2 = require('./ladder2.js');
const { game3FromGame2 } = require('./game3.js');
const Ladder2Static = require('./ladder2-static.js');

const SIZE = 13;

// ── Helpers ─────────────────────────────────────────────────────────────────

function compareResults(expected, actual) {
  if (expected.length !== actual.length) return 'length';
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i], a = actual[i];
    if (e.gid !== a.gid || e.color !== a.color) return `gid/color at ${i}`;
    const es = e.status, as = a.status;
    if (es.moverSucceeds !== as.moverSucceeds) return `moverSucceeds at gid=${e.gid}`;
    if (es.urgentLibs.length !== as.urgentLibs.length) return `urgentLibs.length at gid=${e.gid}`;
    for (let j = 0; j < es.urgentLibs.length; j++) {
      if (es.urgentLibs[j] !== as.urgentLibs[j]) return `urgentLibs[${j}] at gid=${e.gid}`;
    }
  }
  return null;
}

// ── Correctness: play through random games, check every position ────────────

const NUM_GAMES = 200;
let totalPositions = 0, mismatches = 0;

Ladder2Static.resetFallbackCount();
Ladder2Static.setDebugVerify(true);

for (let gi = 0; gi < NUM_GAMES; gi++) {
  const g = new Game2(SIZE);
  while (true) {
    const move = g.randomLegalMove();
    if (move === PASS) break;
    g.play(move);

    const expected = Ladder2.getAllLadderStatuses(game3FromGame2(g), 2);
    const actual = Ladder2Static.getAllLadderStatuses(g, 2);
    totalPositions++;
    const diff = compareResults(expected, actual);
    if (diff) mismatches++;
  }
}

console.log(`Correctness: ${totalPositions - mismatches}/${totalPositions} (${mismatches} mismatches)`);
console.log(`Fallbacks: ${Ladder2Static.getFallbackCount()}`);
console.log(`Divergences caught: ${Ladder2Static.getDivergences()}`);

Ladder2Static.setDebugVerify(false);

// ── Performance comparison ──────────────────────────────────────────────────

const PERF_GAMES = 50;
const positions = [];
for (let i = 0; i < PERF_GAMES; i++) {
  const g = new Game2(SIZE);
  while (true) {
    const move = g.randomLegalMove();
    if (move === PASS) break;
    g.play(move);
    positions.push(g.clone());
  }
}

let t0 = performance.now();
for (const g of positions) Ladder2.getAllLadderStatuses(game3FromGame2(g), 2);
const refTime = performance.now() - t0;

Ladder2Static.resetFallbackCount();
t0 = performance.now();
for (const g of positions) Ladder2Static.getAllLadderStatuses(g, 2);
const staticTime = performance.now() - t0;

console.log(`\nPerformance (${positions.length} positions from ${PERF_GAMES} games):`);
console.log(`  ladder2:        ${refTime.toFixed(0)} ms  (${(refTime / positions.length).toFixed(3)} ms/pos)`);
console.log(`  ladder2-static: ${staticTime.toFixed(0)} ms  (${(staticTime / positions.length).toFixed(3)} ms/pos)`);
console.log(`  speedup: ${(refTime / staticTime).toFixed(2)}x`);
const fb = Ladder2Static.getFallbackCount();
const tc = Ladder2Static.getTotalCalls();
console.log(`  fallbacks: ${fb}/${tc} (${(100*(tc-fb)/tc).toFixed(1)}% solved statically)`);
