'use strict';

// Intensive ladder benchmark with actual fighting patterns

const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');

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

  function getLadderStatus(game, stoneIdx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
    if (lc < 1 || lc > 2) return null;
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    const gColor = game.cells[stoneIdx];
    const defending = gColor === game.current;
    return { libs, defending };
  }

  return { getLadderStatus };
})();

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

  function getLadderStatus(game, stoneIdx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
    if (lc < 1 || lc > 2) return null;
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    const gColor = game.cells[stoneIdx];
    const defending = gColor === game.current;
    return { libs, defending };
  }

  return { getLadderStatus };
})();

// Build actual ladder fighting position
function buildLadderFight(GameClass, N) {
  const game = new GameClass(N);

  // Create a snake pattern (classic ladder)
  // Black snake being chased by white
  const blackSnake = [];
  const whiteChaser = [];

  // Build vertical ladder pattern
  for (let row = 0; row < Math.min(15, N-2); row++) {
    blackSnake.push((row * N) + 8);
    blackSnake.push((row * N) + 10);
    whiteChaser.push((row * N) + 7);
    whiteChaser.push((row * N) + 9);
    whiteChaser.push((row * N) + 11);
  }

  // Play all white stones first
  for (const move of whiteChaser) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }

  // Then black stones
  for (const move of blackSnake) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }

  return game;
}

function benchmark(name, fn, iterations) {
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

console.log('Ladder Detection - Intensive (Real ladder fighting)');
console.log('='.repeat(70));

// Test on 13x13 (good balance)
console.log('\n13x13 Board with ladder fight:');
const game2 = buildLadderFight(Game2, 13);
const game3 = buildLadderFight(Game3Precise, 13);

console.log(`Position: ${game2.moveCount} moves played`);

// Analyze all atari groups
const atariGroups2 = [];
const atariGroups3 = [];
const visited2 = new Set();
const visited3 = new Set();

for (let i = 0; i < 13*13; i++) {
  if (game2.cells[i] !== 0) {
    const gid = game2._gid[i];
    if (!visited2.has(gid)) {
      visited2.add(gid);
      const { count: lc } = game2.groupLibs2(i);
      if (lc === 1 || lc === 2) atariGroups2.push(i);
    }
  }
}

for (let i = 0; i < 13*13; i++) {
  if (game3.cells[i] !== 0) {
    const gid = game3._gid[i];
    if (!visited3.has(gid)) {
      visited3.add(gid);
      const { count: lc } = game3.groupLibs2(i);
      if (lc === 1 || lc === 2) atariGroups3.push(i);
    }
  }
}

console.log(`Atari groups: ${atariGroups2.length}`);

let time2 = benchmark('  Game2 (clone):', () => {
  for (const idx of atariGroups2) {
    Ladder2Game2.getLadderStatus(game2, idx);
  }
}, 100);

let time3 = benchmark('  Game3-Precise (undo):', () => {
  for (const idx of atariGroups3) {
    Ladder2Game3.getLadderStatus(game3, idx);
  }
}, 100);

let ratio = (time2 / time3).toFixed(2);
let improvement = ((time2 - time3) / time2 * 100).toFixed(1);
console.log(`  Speedup: ${ratio}x (${improvement}% improvement)`);

console.log('\n' + '='.repeat(70));
