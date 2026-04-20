#!/usr/bin/env node
'use strict';

// Diagnose the overflow by tracking board positions visited during search

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('overflow-board.json', 'utf8'));

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Reconstruct the game
const g2 = new Game2(13);
for (const move of data.moves) {
  g2.play(move);
}

const g3 = game3FromGame2(g2);

// Create a board hash function
function boardHash(game) {
  let hash = '';
  for (let i = 0; i < 169; i++) {
    hash += game.cells[i] + ',';
  }
  hash += game.current + ',' + game.ko;
  return hash;
}

// Instrument canReach4Libs to track positions
let visitCount = 0;
let maxDepth = 0;
const positionsSeen = new Map();
let depthStack = [];

const originalSearchChain = require('./tactics3.js').searchChain;

// Manually implement searchChain with instrumentation
function instrumentedSearchChain(game, stoneIdx, nodeLimit = Infinity) {
  const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 3) return null;

  const libs = lc === 1 ? [lib0] : [lib0, lib1];
  const gColor = game.cells[stoneIdx];
  const mover = game.current;
  const defending = gColor === mover;
  const atari = lc === 1;

  console.log('\nStarting searchChain on position ' + stoneIdx);
  console.log('Liberties:', lc, 'Color:', gColor === 1 ? 'BLACK' : 'WHITE');

  let callDepth = 0;
  let maxCallDepth = 0;
  const positionLog = [];

  function canReach4Libs(game, idx, credits, depth) {
    callDepth = Math.max(callDepth, depth);
    maxCallDepth = Math.max(maxCallDepth, depth);

    const hash = boardHash(game);
    visitCount++;

    if (!positionsSeen.has(hash)) {
      positionsSeen.set(hash, 0);
    }
    const count = positionsSeen.get(hash);
    positionsSeen.set(hash, count + 1);

    positionLog.push({ depth, hash: hash.substring(0, 20) + '...', count: count + 1 });

    if (visitCount % 1000 === 0) {
      console.log(`Visited ${visitCount} positions, max depth ${maxCallDepth}, positions seen: ${positionsSeen.size}`);

      // Check for repeated positions
      const repeated = Array.from(positionsSeen.entries())
        .filter(([_, c]) => c > 10)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

      if (repeated.length > 0) {
        console.log('Top repeated positions:');
        for (const [hash, count] of repeated) {
          console.log(`  Seen ${count} times: ${hash.substring(0, 30)}...`);
        }
      }
    }

    if (visitCount > 100000) {
      throw new Error('Search visited 100k+ positions - likely infinite loop');
    }

    if (credits <= 0) return [null, 0];
    credits--;

    const libs2 = game.groupLibs2(idx);
    const lc2 = libs2.count;
    const lib0_2 = libs2.lib0;
    const lib1_2 = libs2.lib1;

    if (lc2 >= 4) return [true, credits];
    if (lc2 === 0) return [false, credits];

    const defColor = game.cells[idx];
    const libs_arr = lc2 === 1 ? [lib0_2] : [lib0_2, lib1_2];

    if (game.current === defColor) {
      let hasUnknown = false;
      for (let k = 0; k < lc2; k++) {
        const libIdx = libs_arr[k];
        if (!game.play(libIdx)) {
          game.undo();
          continue;
        }
        if (game.cells[idx] === 0) {
          game.undo();
          continue;
        }
        const budget = Math.floor(credits / (lc2 - k));
        credits -= budget;
        const [result, unused] = canReach4Libs(game, idx, budget, depth + 1);
        credits += unused;
        game.undo();
        if (result === true) return [true, credits];
        if (result === null) hasUnknown = true;
      }
      return [hasUnknown ? null : false, credits];
    }

    let hasUnknown = false;
    for (let k = 0; k < lc2; k++) {
      const libIdx = libs_arr[k];
      if (!game.play(libIdx)) {
        game.undo();
        continue;
      }
      if (game.cells[idx] === 0) {
        game.undo();
        return [false, credits];
      }
      const afterLc = game.groupLibs2(idx).count;
      if (afterLc === 0) {
        game.undo();
        return [false, credits];
      }
      if (afterLc < 4) {
        const budget = Math.floor(credits / (lc2 - k));
        credits -= budget;
        const [result, unused] = canReach4Libs(game, idx, budget, depth + 1);
        credits += unused;
        game.undo();
        if (result === false) return [false, credits];
        if (result === null) hasUnknown = true;
      } else {
        game.undo();
      }
    }

    return [hasUnknown ? null : true, credits];
  }

  try {
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
      [escape, unused] = canReach4Libs(game, stoneIdx, budget, 1);
      credits += unused;
      game.undo();
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
          [escape, unused] = canReach4Libs(game, stoneIdx, budget, 1);
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

    return { libs, moverSucceeds: moverSucceeds ? true : (hasUnknown ? null : false), urgentLibs };
  } catch (e) {
    console.log('\n✗ Search terminated after ' + visitCount + ' position visits');
    console.log('Max recursion depth: ' + maxCallDepth);
    console.log('Unique positions visited: ' + positionsSeen.size);

    // Find most repeated positions
    const repeated = Array.from(positionsSeen.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    console.log('\nMost repeated positions:');
    for (let i = 0; i < repeated.length; i++) {
      console.log(`${i+1}. Visited ${repeated[i][1]} times`);
    }

    throw e;
  }
}

// Run the instrumented search
try {
  instrumentedSearchChain(g3, 22);
} catch (e) {
  console.log('\nError:', e.message);
}
