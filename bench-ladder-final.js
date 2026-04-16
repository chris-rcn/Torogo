'use strict';

// Final ladder benchmark: Game2 (clone-based) vs Game3-Precise (play/undo-based)

const { Game2 } = require('./game2.js');
const { Game3Precise, PASS } = require('./game3-precise.js');

// Game2-based ladder detection (original, uses clone)
const Ladder2Game2 = (function() {
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

  function getAllLadderStatusesGame2(game) {
    const cap = game.N * game.N;
    const results = [];
    const visited = new Set();
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 0 || lc > 2) continue;
      results.push({ gid, count: lc });
    }
    return results;
  }

  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

// Game3-Precise-based ladder detection (new, uses play/undo)
const Ladder2Game3 = (function() {
  function _canReach3Libs(game, idx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
    if (lc >= 3) return true;
    if (lc === 0) return false;
    const defColor = game.cells[idx];
    if (game.current === defColor) {
      const libs = lc === 1 ? [lib0] : [lib0, lib1];
      for (const libIdx of libs) {
        if (!game.play(libIdx)) continue;
        const captured = game.cells[idx] === 0;
        const result = !captured && _canReach3Libs(game, idx);
        game.undo();
        if (result) return true;
      }
      return false;
    }
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      if (!game.play(libIdx)) continue;
      const captured = game.cells[idx] === 0;
      if (captured) {
        game.undo();
        return false;
      }
      const afterLc = game.groupLibs2(idx).count;
      if (afterLc === 0) {
        game.undo();
        return false;
      }
      if (afterLc === 1) {
        const result = !_canReach3Libs(game, idx);
        game.undo();
        if (result) return false;
      } else {
        game.undo();
      }
    }
    return true;
  }

  function getAllLadderStatusesGame3(game) {
    const cap = game.N * game.N;
    const results = [];
    const visited = new Set();
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 0 || lc > 2) continue;
      results.push({ gid, count: lc });
    }
    return results;
  }

  return { getAllLadderStatuses: getAllLadderStatusesGame3 };
})();

function buildTestPosition(N) {
  const game = new Game3Precise(N);
  const moves = [
    10, 11, 12, 13,
    20, 21, 22, 23,
    30, 31, 32, 33,
    40, 41, 42, 43,
    50, 51, 52, 53,
    60, 61, 62, 63,
    70, 71, 72, 73,
  ];
  for (const move of moves) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }
  return game;
}

function buildTestPositionGame2(N) {
  const game = new Game2(N);
  const moves = [
    10, 11, 12, 13,
    20, 21, 22, 23,
    30, 31, 32, 33,
    40, 41, 42, 43,
    50, 51, 52, 53,
    60, 61, 62, 63,
    70, 71, 72, 73,
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
  const elapsed = Number(end - start) / 1e6;
  const perIter = elapsed / iterations;
  console.log(`${name}: ${elapsed.toFixed(2)}ms total, ${perIter.toFixed(4)}ms per iteration`);
  return elapsed;
}

console.log('Ladder Detection: Game2 (clone) vs Game3-Precise (play/undo)');
console.log('='.repeat(70));

// Test on 9x9
console.log('\n9x9 Board:');
const game2_9x9 = buildTestPositionGame2(9);
const game3_9x9 = buildTestPosition(9);

let time2 = benchmark('  Game2 (clone):', () => {
  Ladder2Game2.getAllLadderStatuses(game2_9x9);
}, 200);

let time3 = benchmark('  Game3-Precise (undo):', () => {
  Ladder2Game3.getAllLadderStatuses(game3_9x9);
}, 200);

let ratio = (time2 / time3).toFixed(2);
let improvement = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`  Speedup: ${ratio}x (${improvement}% faster)`);

// Test on 13x13
console.log('\n13x13 Board:');
const game2_13x13 = buildTestPositionGame2(13);
const game3_13x13 = buildTestPosition(13);

time2 = benchmark('  Game2 (clone):', () => {
  Ladder2Game2.getAllLadderStatuses(game2_13x13);
}, 100);

time3 = benchmark('  Game3-Precise (undo):', () => {
  Ladder2Game3.getAllLadderStatuses(game3_13x13);
}, 100);

ratio = (time2 / time3).toFixed(2);
improvement = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`  Speedup: ${ratio}x (${improvement}% faster)`);

// Test on 19x19
console.log('\n19x19 Board:');
const game2_19x19 = buildTestPositionGame2(19);
const game3_19x19 = buildTestPosition(19);

time2 = benchmark('  Game2 (clone):', () => {
  Ladder2Game2.getAllLadderStatuses(game2_19x19);
}, 50);

time3 = benchmark('  Game3-Precise (undo):', () => {
  Ladder2Game3.getAllLadderStatuses(game3_19x19);
}, 50);

ratio = (time2 / time3).toFixed(2);
improvement = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`  Speedup: ${ratio}x (${improvement}% faster)`);

console.log('\n' + '='.repeat(70));
console.log('Summary: Game3-Precise play/undo vs Game2 clone-based ladder detection');
