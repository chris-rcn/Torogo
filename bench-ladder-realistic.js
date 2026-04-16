'use strict';

// Realistic ladder detection benchmark:
// Play random moves, analyze ladder status at every position
// Repeat for ~200 games on 13x13

const { Game2 } = require('./game2.js');
const { Game3Precise, PASS, BLACK, WHITE } = require('./game3-precise.js');
const { getAllLadderStatuses } = require('./ladder2.js');

// Game2-based ladder detection
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
      g.play(-1);  // PASS
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
      const status = getLadderStatus(game, i);
      if (status) results.push({ gid, color: game.cells[i], status });
    }
    return results;
  }

  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

function getRandomLegalMove(game) {
  const N = game.N;
  const cap = N * N;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (game.isLegal(i)) {
      moves.push(i);
    }
  }
  if (moves.length === 0) return -1;
  return moves[Math.floor(Math.random() * moves.length)];
}

function playRandomGame(GameClass) {
  const game = new GameClass(13);
  const moveCount = [];

  while (true) {
    const move = getRandomLegalMove(game);
    if (move === -1) break;
    if (!game.play(move)) break;
    moveCount.push(move);

    // Stop after reasonable number of moves
    if (moveCount.length > 150) break;
  }

  return { game, moveCount };
}

console.log('Realistic Ladder Detection Benchmark');
console.log('13x13 board, random play, analyze at each position');
console.log('='.repeat(70));

// Run multiple games
const numGames = 200;
let game2TotalTime = 0;
let game3TotalTime = 0;
let game2AnalysisCount = 0;
let game3AnalysisCount = 0;

console.log(`\nRunning ${numGames} games...`);

for (let gameNum = 0; gameNum < numGames; gameNum++) {
  if ((gameNum + 1) % 50 === 0) {
    console.log(`  Game ${gameNum + 1}/${numGames}...`);
  }

  // Play random game with Game2
  const { game: game2, moveCount: moves2 } = playRandomGame(Game2);

  // Replay same moves with Game3-Precise
  const game3 = new Game3Precise(13);
  for (const move of moves2) {
    game3.play(move);
  }

  // Now analyze every position in the game tree
  // (simplified: just analyze the final position for now, would need tree traversal for full benchmark)

  // For a more realistic test, let's analyze at every move
  const game2Copy = new Game2(13);
  const game3Copy = new Game3Precise(13);

  for (const move of moves2) {
    game2Copy.play(move);
    game3Copy.play(move);

    // Measure Game2 analysis
    let start = process.hrtime.bigint();
    const results2 = Ladder2Game2.getAllLadderStatuses(game2Copy);
    let end = process.hrtime.bigint();
    game2TotalTime += Number(end - start);
    game2AnalysisCount += results2.length || 1;

    // Measure Game3-Precise analysis
    start = process.hrtime.bigint();
    const results3 = getAllLadderStatuses(game3Copy);
    end = process.hrtime.bigint();
    game3TotalTime += Number(end - start);
    game3AnalysisCount += results3.length || 1;
  }
}

const game2Ms = game2TotalTime / 1e6;
const game3Ms = game3TotalTime / 1e6;

console.log('\n' + '='.repeat(70));
console.log('Results:');
console.log(`Game2 (clone):        ${game2Ms.toFixed(2)}ms total`);
console.log(`Game3-Precise (undo): ${game3Ms.toFixed(2)}ms total`);

const ratio = (game2Ms / game3Ms).toFixed(2);
const improvement = ((game2Ms - game3Ms) / game2Ms * 100).toFixed(1);

console.log(`\nSpeedup: ${ratio}x`);
console.log(`Improvement: ${improvement}%`);

console.log(`\nAnalysis operations:`);
console.log(`  Game2: ${game2AnalysisCount} groups analyzed`);
console.log(`  Game3-Precise: ${game3AnalysisCount} groups analyzed`);
console.log(`\nAverage per analysis:`);
console.log(`  Game2: ${(game2Ms / game2AnalysisCount * 1000).toFixed(4)}µs`);
console.log(`  Game3-Precise: ${(game3Ms / game3AnalysisCount * 1000).toFixed(4)}µs`);
