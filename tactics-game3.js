'use strict';

// Example: Tactical chain search using Game3 (incremental undo approach)
// This is an equivalent to tactics3.js but optimized for Game3's incremental design

const { Game3, PASS } = require('./game3.js');

/**
 * Analyze a chain with 1-3 liberties to determine if it can reach 4+ liberties.
 * Uses Game3's incremental play/undo instead of cloning.
 *
 * @param {Game3} game - Game instance (will be modified then restored)
 * @param {number} stoneIdx - Index of a stone in the chain to analyze
 * @param {number} credits - Search budget (nodes to explore)
 * @returns {Array} [result, remainingCredits] where result is:
 *   true  - defender can reach 4+ liberties
 *   false - attacker can capture the chain
 *   null  - budget exhausted before conclusion
 */
function canReach4Libs(game, stoneIdx, credits) {
  if (credits <= 0) return [null, 0];
  credits--;

  const libs = game.groupLibs(stoneIdx);
  const lc = libs.length;

  if (lc >= 4) return [true, credits];
  if (lc === 0) return [false, credits];

  const defColor = game.cells[stoneIdx];

  if (game.current === defColor) {
    // Defender's turn: succeed if any branch is definitely true
    let hasUnknown = false;
    for (let k = 0; k < lc; k++) {
      const lib = libs[k];
      if (!game.isLegal(lib)) continue;

      game.play(lib);
      if (game.cells[stoneIdx] === 0) {
        // Stone was captured
        game.undo();
        continue;
      }

      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      const [result, unused] = canReach4Libs(game, stoneIdx, budget);
      credits += unused;

      game.undo();

      if (result === true) return [true, credits];
      if (result === null) hasUnknown = true;
    }
    return [hasUnknown ? null : false, credits];
  }

  // Attacker's turn: succeed if any branch definitely captures
  let hasUnknown = false;
  for (let k = 0; k < lc; k++) {
    const lib = libs[k];
    if (!game.isLegal(lib)) continue;

    game.play(lib);
    if (game.cells[stoneIdx] === 0) {
      // Chain captured immediately
      game.undo();
      return [false, credits];
    }

    const afterLc = game.groupLibs(stoneIdx).length;
    if (afterLc === 0) {
      game.undo();
      return [false, credits];
    }

    if (afterLc < 4) {
      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      const [result, unused] = canReach4Libs(game, stoneIdx, budget);
      credits += unused;

      if (result === false) {
        game.undo();
        return [false, credits];
      }
      if (result === null) hasUnknown = true;
    }

    game.undo();
  }

  return [hasUnknown ? null : true, credits];
}

/**
 * Analyze a single chain with 1-3 liberties.
 *
 * @param {Game3} game - Game instance
 * @param {number} stoneIdx - Index of a stone in the chain
 * @param {number} nodeLimit - Budget per liberty (default Infinity)
 * @returns {Object} { libs, moverSucceeds, urgentLibs }
 */
function searchChain(game, stoneIdx, nodeLimit = Infinity) {
  const libs = game.groupLibs(stoneIdx);
  const lc = libs.length;

  if (lc > 3) {
    console.warn(`searchChain: stone at ${stoneIdx} has ${lc} liberties (>3), skipping`);
    return null;
  }

  if (lc === 0) {
    return { libs, moverSucceeds: false, urgentLibs: [] };
  }

  const moverColor = game.current;
  const moverIsAttacker = moverColor !== game.cells[stoneIdx];
  const defenderColor = game.cells[stoneIdx];

  const [result, _] = canReach4Libs(game, stoneIdx, nodeLimit);

  let moverSucceeds;
  if (moverIsAttacker) {
    // Attacker wants to capture (canReach4Libs returns false)
    moverSucceeds = result === false;
  } else {
    // Defender wants to escape (canReach4Libs returns true)
    moverSucceeds = result === true;
  }

  // Identify urgent liberties (those that lead to immediate loss/capture)
  const urgentLibs = [];
  for (const lib of libs) {
    if (!game.isLegal(lib)) continue;

    game.play(lib);
    const capturedOrEscaped = game.cells[stoneIdx] === 0 || game.groupLibs(stoneIdx).length >= 4;
    game.undo();

    if (capturedOrEscaped) {
      urgentLibs.push(lib);
    }
  }

  return { libs: Array.from(libs), moverSucceeds, urgentLibs };
}

/**
 * Search all chains with 1-3 liberties on the board.
 *
 * @param {Game3} game - Game instance
 * @param {number} nodeLimit - Budget per sub-search (default Infinity)
 * @returns {Array} Array of { gid, color, status } objects
 */
function searchChains(game, nodeLimit = Infinity) {
  const cap = game.N * game.N;
  const results = [];
  const visited = new Set();

  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game.groupIdAt(i);
    if (visited.has(gid)) continue;
    visited.add(gid);

    const lc = game.groupLibs(i).length;
    if (lc === 0 || lc > 3) continue;

    const status = searchChain(game, i, nodeLimit);
    if (status) {
      results.push({ gid, color: game.cells[i], status });
    }
  }

  return results;
}

// ─────────────────────────────────────────────────────────────────────────────

// Example usage and testing
if (require.main === module) {
  console.log('Tactical search example using Game3\n');

  const game = new Game3(5);

  // Build a position with some chains
  const moves = [
    6,   // white
    11,  // black
    7,   // white
    2,   // black
    1,   // white
    3,   // black
    8,   // white
    13,  // black
  ];

  for (const move of moves) {
    if (game.isLegal(move)) {
      game.play(move);
      console.log(`Played at ${move}, current: ${game.current === 1 ? 'BLACK' : 'WHITE'}`);
    }
  }

  console.log('\n--- Board State ---');
  for (let y = 0; y < game.N; y++) {
    let row = '';
    for (let x = 0; x < game.N; x++) {
      const i = y * game.N + x;
      const c = game.cells[i];
      if (c === 1) row += '● ';
      else if (c === -1) row += '○ ';
      else row += '. ';
    }
    console.log(row);
  }

  console.log('\n--- Chain Analysis ---');
  const chains = searchChains(game, 100);

  if (chains.length === 0) {
    console.log('No chains with 1-3 liberties');
  } else {
    for (const chain of chains) {
      const colorName = chain.color === 1 ? 'BLACK' : 'WHITE';
      console.log(`\nGroup ${chain.gid} (${colorName}):`);
      console.log(`  Status: ${chain.status.moverSucceeds === null ? 'UNKNOWN' : (chain.status.moverSucceeds ? 'SUCCEEDS' : 'FAILS')}`);
      console.log(`  Liberties: ${chain.status.libs.join(', ')}`);
      console.log(`  Urgent: ${chain.status.urgentLibs.join(', ')}`);
    }
  }

  console.log('\n✅ Example complete (undo stack depth: ' + game._undoStack.length + ')');
}

module.exports = {
  canReach4Libs,
  searchChain,
  searchChains,
};
