#!/usr/bin/env node
'use strict';

// Compare solve ratio (definitive vs inconclusive) with different depth limits

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Test with current depth limit (20) vs reduced limit (15)
// We'll monkey-patch tactics3 to test different limits

function runTestWithDepthLimit(depthLimit, numGames = 10, movesPerGame = 50) {
  // Dynamically create canReach4Libs with specified depth limit
  function createCanReach4Libs(limit) {
    return function canReach4Libs(game, idx, credits, depth = 0) {
      if (depth > limit) return [null, credits];
      if (credits <= 0) return [null, 0];
      credits--;

      const libs = game.groupLibs(idx);
      const lc = libs.length;
      if (lc >= 4) return [true, credits];
      if (lc === 0) return [false, credits];

      const defColor = game.cells[idx];

      if (game.current === defColor) {
        let hasUnknown = false;
        for (let k = 0; k < lc; k++) {
          const libIdx = libs[k];
          if (!game.play(libIdx)) {
            game.undo();
            continue;
          }
          if (game.cells[idx] === 0) {
            game.undo();
            continue;
          }
          const budget = Math.floor(credits / (lc - k));
          credits -= budget;
          let result, unused;
          [result, unused] = canReach4Libs(game, idx, budget, depth + 1);
          credits += unused;
          game.undo();
          if (result === true) return [true, credits];
          if (result === null) hasUnknown = true;
        }
        return [hasUnknown ? null : false, credits];
      }

      let hasUnknown = false;
      for (let k = 0; k < lc; k++) {
        const libIdx = libs[k];
        if (!game.play(libIdx)) {
          game.undo();
          continue;
        }
        if (game.cells[idx] === 0) {
          game.undo();
          return [false, credits];
        }
        const afterLc = game.groupLibs(idx).length;
        if (afterLc === 0) {
          game.undo();
          return [false, credits];
        }
        if (afterLc < 4) {
          const budget = Math.floor(credits / (lc - k));
          credits -= budget;
          let result, unused;
          [result, unused] = canReach4Libs(game, idx, budget, depth + 1);
          credits += unused;
          game.undo();
          if (result === false) return [false, credits];
          if (result === null) hasUnknown = true;
        } else {
          game.undo();
        }
      }

      return [hasUnknown ? null : true, credits];
    };
  }

  function searchChain(game, stoneIdx, canReach4LibsFunc, nodeLimit = Infinity) {
    const libs = game.groupLibs(stoneIdx);
    const lc = libs.length;
    if (lc < 1 || lc > 3) return null;

    const gColor = game.cells[stoneIdx];
    const mover = game.current;
    const defending = gColor === mover;
    const atari = lc === 1;

    const opponentCalls = (defending && atari) ? 0 : 1;
    const moverCalls = (!defending && atari) ? 0 : lc;
    let callsLeft = opponentCalls + moverCalls;
    let credits = nodeLimit;

    let escape;
    if (defending && atari) {
      escape = false;
    } else {
      const budget = Math.floor(credits / callsLeft);
      credits -= budget;
      callsLeft--;
      game.play(-1); // PASS
      let unused;
      [escape, unused] = canReach4LibsFunc(game, stoneIdx, budget);
      credits += unused;
      game.undo();
    }

    if (escape !== null && defending === escape) {
      return { libs, moverSucceeds: true, urgentLibs: [] };
    }

    let moverSucceeds = false;
    let hasUnknown = escape === null;
    let urgentLibs = [];

    for (let k = 0; k < lc; k++) {
      const libIdx = libs[k];
      if (!defending && atari) {
        escape = false;
      } else {
        const budget = Math.floor(credits / callsLeft);
        credits -= budget;
        callsLeft--;
        const played = game.play(libIdx);
        if (played) {
          let unused;
          [escape, unused] = canReach4LibsFunc(game, stoneIdx, budget);
          credits += unused;
        } else {
          escape = false;
          credits += budget;
        }
        game.undo();
      }
      if (escape !== null && defending === escape) {
        moverSucceeds = true;
        urgentLibs.push(libIdx);
      } else if (escape === null) {
        hasUnknown = true;
      }
    }

    return {
      libs,
      moverSucceeds: moverSucceeds ? true : (hasUnknown ? null : false),
      urgentLibs,
    };
  }

  function searchChains(game, canReach4LibsFunc, nodeLimit = Infinity) {
    const cap = game.N * game.N;
    const results = [];
    const visited = new Set();
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const lc = game.groupLibs(i).length;
      if (lc === 0 || lc > 3) continue;
      const status = searchChain(game, i, canReach4LibsFunc, nodeLimit);
      results.push({ gid, color: game.cells[i], status });
    }
    return results;
  }

  // Run tests
  const canReach4LibsFunc = createCanReach4Libs(depthLimit);
  let definitiveCount = 0;
  let inconclusiveCount = 0;
  let totalGroups = 0;

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const g2 = new Game2(13);
    let movesPlayed = 0;

    while (movesPlayed < movesPerGame && !g2.gameOver) {
      const g3 = game3FromGame2(g2);
      const tactics = searchChains(g3, canReach4LibsFunc, 10000);

      for (const tactic of tactics) {
        totalGroups++;
        if (tactic.status && tactic.status.moverSucceeds === null) {
          inconclusiveCount++;
        } else {
          definitiveCount++;
        }
      }

      // Play random move
      for (let i = 0; i < 169; i++) {
        if (g2.isLegal(i)) {
          g2.play(i);
          movesPlayed++;
          break;
        }
      }
    }
  }

  return {
    depthLimit,
    totalGroups,
    definitive: definitiveCount,
    inconclusive: inconclusiveCount,
    definitiveRate: totalGroups > 0 ? ((definitiveCount / totalGroups) * 100).toFixed(2) : 0,
    inconclusiveRate: totalGroups > 0 ? ((inconclusiveCount / totalGroups) * 100).toFixed(2) : 0,
  };
}

console.log('Comparing solve ratio with different depth limits...\n');
console.log('Testing with 10 games, 50 moves per game...\n');

const results20 = runTestWithDepthLimit(20);
const results15 = runTestWithDepthLimit(15);

console.log('====== DEPTH LIMIT 20 ======');
console.log(`Total groups: ${results20.totalGroups}`);
console.log(`Definitive: ${results20.definitive} (${results20.definitiveRate}%)`);
console.log(`Inconclusive: ${results20.inconclusive} (${results20.inconclusiveRate}%)`);

console.log('\n====== DEPTH LIMIT 15 ======');
console.log(`Total groups: ${results15.totalGroups}`);
console.log(`Definitive: ${results15.definitive} (${results15.definitiveRate}%)`);
console.log(`Inconclusive: ${results15.inconclusive} (${results15.inconclusiveRate}%)`);

console.log('\n====== COMPARISON ======');
const defDiff = results15.definitive - results20.definitive;
const incDiff = results15.inconclusive - results20.inconclusive;
const rateDiff = parseFloat(results15.definitiveRate) - parseFloat(results20.definitiveRate);

console.log(`Definitive change: ${defDiff > 0 ? '+' : ''}${defDiff} (${rateDiff > 0 ? '+' : ''}${rateDiff}%)`);
console.log(`Inconclusive change: ${incDiff > 0 ? '+' : ''}${incDiff} (${rateDiff < 0 ? '+' : ''}${-rateDiff}%)`);

if (Math.abs(rateDiff) < 0.5) {
  console.log('\n✓ Minimal difference - depth limit 15 is viable');
} else if (rateDiff < -1.0) {
  console.log('\n✗ Significant reduction in solve rate - depth limit 15 not recommended');
} else {
  console.log('\n⚠ Moderate change - depth limit 15 may be acceptable with trade-offs');
}
