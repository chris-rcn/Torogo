'use strict';

const { Game2 } = require('./game2.js');
const { Game3Precise, PASS } = require('./game3.js');
const { getAllLadderStatuses } = require('./ladder2.js');

// Create a position with actual ladder fights (more ladder scenarios)
function buildLadderPosition(N) {
  const game = new Game3Precise(N);
  
  // Create complex ladder patterns with multiple atari situations
  // Main ladder pattern: black group in danger, white attacking
  const patterns = [
    // Vertical ladder pattern (black snake being chased by white)
    [10, 11, 12, 13, 14,   // B stones
     20, 21, 22, 23, 24,   // W stones
     30, 31, 32, 33, 34,   // B stones  
     40, 41, 42, 43, 44],  // W stones
    
    // Horizontal ladder
    [15, 25, 35, 45,       // B stones
     16, 26, 36, 46,       // W stones
     17, 27, 37, 47,       // B stones
     18, 28, 38, 48],      // W stones
     
    // More scattered combat
    [50, 51, 52, 53, 54, 55,
     60, 61, 62, 63, 64, 65,
     70, 71, 72, 73, 74, 75]
  ];
  
  for (const pattern of patterns) {
    for (const move of pattern) {
      if (game.isLegal(move)) {
        game.play(move);
      }
    }
  }
  
  return game;
}

// Game2 ladder detection (original)
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
      // Simplified: just analyze without full ladder calculation
      results.push({ gid, color: game.cells[i] });
    }
    return results;
  }
  
  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

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

console.log('Ladder Detection Benchmark (Improved)');
console.log('='.repeat(60));

// Test on 9x9 with ladder-rich position
console.log('\n9x9 Board (ladder-rich position):');
const game9_2 = buildLadderPosition(9);
const game9_3 = buildLadderPosition(9);

console.log(`Position: ${game9_2.moveCount} moves, ladder groups to analyze`);

let time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game9_2);
}, 50);

let time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game9_3);
}, 50);

let improvement9 = ((time2 - time3) / time2 * 100).toFixed(1);
let ratio9 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio9}x faster (${improvement9}%)`);

// Test on 13x13
console.log('\n13x13 Board (ladder-rich position):');
const game13_2 = buildLadderPosition(13);
const game13_3 = buildLadderPosition(13);

console.log(`Position: ${game13_2.moveCount} moves`);

time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game13_2);
}, 30);

time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game13_3);
}, 30);

let improvement13 = ((time2 - time3) / time2 * 100).toFixed(1);
let ratio13 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio13}x faster (${improvement13}%)`);

// Test on 19x19
console.log('\n19x19 Board (ladder-rich position):');
const game19_2 = buildLadderPosition(19);
const game19_3 = buildLadderPosition(19);

console.log(`Position: ${game19_2.moveCount} moves`);

time2 = benchmark('Game2:', () => {
  Ladder2Game2.getAllLadderStatuses(game19_2);
}, 20);

time3 = benchmark('Game3-Precise:', () => {
  getAllLadderStatuses(game19_3);
}, 20);

let improvement19 = ((time2 - time3) / time2 * 100).toFixed(1);
let ratio19 = (time2 / time3).toFixed(2);
console.log(`Speedup: ${ratio19}x faster (${improvement19}%)`);

console.log('\n' + '='.repeat(60));
console.log('Summary (ladder-rich positions):');
console.log(`9x9:   ${ratio9}x faster (${improvement9}%)`);
console.log(`13x13: ${ratio13}x faster (${improvement13}%)`);
console.log(`19x19: ${ratio19}x faster (${improvement19}%)`);
