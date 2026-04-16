'use strict';

// Benchmark ladder detection: Game2 vs Game3-Precise

const { Game2 } = require('./game2.js');
const { Game3Precise, PASS, BLACK, WHITE } = require('./game3-precise.js');

// Import ladder detection (now using Game3-Precise)
const { getAllLadderStatuses } = require('./ladder2.js');

// Helper to get ladder status using Game2 directly
// (We need to test Game2's ladder detection before it was modified)
const Ladder2Game2 = (function() {
  const PASS = -1;

  function _canReach3Libs(game, idx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
    if (lc >= 3) return true;
    if (lc === 0) return false;
    const defColor = game.cells[idx];
    if (game.current === defColor) {
      const libs = lc === 1 ? [lib0] : [lib0, lib1];
      for (const libIdx of libs) {
        const g = game.clone();
        if (!g.play(libIdx)) continue;
        if (g.cells[idx] === 0) continue;
        if (_canReach3Libs(g, idx)) return true;
      }
      return false;
    }
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      const g = game.clone();
      if (!g.play(libIdx)) continue;
      if (g.cells[idx] === 0) return false;
      const afterLc = g.groupLibs2(idx).count;
      if (afterLc === 0) return false;
      if (afterLc === 1 && !_canReach3Libs(g, idx)) return false;
    }
    return true;
  }

  function getLadderStatus2(game, stoneIdx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
    if (lc < 1 || lc > 2) return null;
    const atari = lc === 1;
    const libs = atari ? [lib0] : [lib0, lib1];
    const gColor = game.cells[stoneIdx];
    const mover = game.current;
    const defending = gColor === mover;
    let escape;
    if (defending && atari) {
      escape = false;
    } else {
      const g = game.clone();
      g.play(PASS);
      escape = _canReach3Libs(g, stoneIdx);
    }
    if (defending === escape) {
      return { libs, moverSucceeds: true, urgentLibs: [] };
    }
    let moverSucceeds = false;
    let urgentLibs = [];
    for (const libIdx of libs) {
      if (!defending && atari) {
        escape = false;
      } else {
        const g = game.clone();
        escape = g.play(libIdx) && _canReach3Libs(g, stoneIdx);
      }
      if (defending === escape) {
        moverSucceeds = true;
        urgentLibs.push(libIdx);
      }
    }
    return { libs, moverSucceeds, urgentLibs };
  }

  function getAllLadderStatusesGame2(game, minChainSize = 1) {
    const cap = game.N * game.N;
    const results = [];
    const visited = new Set();
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      if (game.groupSize(gid) < minChainSize) continue;
      const { count: lc } = game.groupLibs2(i);
      if (lc === 0 || lc > 2) continue;
      const status = getLadderStatus2(game, i);
      results.push({ gid, color: game.cells[i], status });
    }
    return results;
  }

  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

// Build a test position with ladder groups
function buildTestPosition(N) {
  const game = new Game3Precise(N);

  // Create a complex position with multiple ladder groups
  // Place some stones to create atari and ladder situations
  const moves = [
    10, 11, 12, 13,  // W stones
    20, 21, 22, 23,  // B stones
    30, 31, 32, 33,  // W stones
    40, 41, 42, 43,  // B stones
    50, 51, 52, 53,  // W stones
    60, 61, 62, 63,  // B stones
    70, 71, 72, 73,  // W stones
  ];

  for (const move of moves) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }

  return game;
}

function benchmark(name, fn, iterations = 100) {
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = process.hrtime.bigint();
  const elapsed = Number(end - start) / 1e6; // convert to ms
  const perIter = elapsed / iterations;
  console.log(`${name}: ${elapsed.toFixed(2)}ms total, ${perIter.toFixed(4)}ms per iteration`);
  return elapsed;
}

console.log('Ladder Detection Benchmark');
console.log('='.repeat(60));

// Test on 9x9
console.log('\n9x9 Board:');
const game9_2 = buildTestPosition(9);
const game9_3 = buildTestPosition(9);

let time2, time3;

time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game9_2);
}, 100);

time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game9_3);
}, 100);

const improvement9 = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`Improvement: ${improvement9}%`);
const ratio9 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio9}x`);

// Test on 13x13
console.log('\n13x13 Board:');
const game13_2 = buildTestPosition(13);
const game13_3 = buildTestPosition(13);

time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game13_2);
}, 50);

time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game13_3);
}, 50);

const improvement13 = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`Improvement: ${improvement13}%`);
const ratio13 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio13}x`);

// Test on 19x19
console.log('\n19x19 Board:');
const game19_2 = buildTestPosition(19);
const game19_3 = buildTestPosition(19);

time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game19_2);
}, 20);

time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game19_3);
}, 20);

const improvement19 = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`Improvement: ${improvement19}%`);
const ratio19 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio19}x`);

console.log('\n' + '='.repeat(60));
console.log('Summary:');
console.log(`9x9:   ${ratio9}x faster (${improvement9}%)`);
console.log(`13x13: ${ratio13}x faster (${improvement13}%)`);
console.log(`19x19: ${ratio19}x faster (${improvement19}%)`);
