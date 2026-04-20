'use strict';

// Tactical search for Game3-Optimized (compatible with searchChain interface)

const { PASS } = typeof require === 'function' ? require('./game3-optimized.js') : window.Game3Optimized;

/**
 * Can the defender reach 4+ liberties? (same logic as tactics3.js)
 */
function canReach4Libs(game, idx, credits) {
  if (credits <= 0) return [null, 0];
  credits--;

  const libs = game.groupLibs(idx);
  const lc = libs.length;
  if (lc >= 4) return [true, credits];
  if (lc === 0) return [false, credits];

  const defColor = game.cells[idx];

  if (game.current === defColor) {
    // Defender's turn
    let hasUnknown = false;
    for (let k = 0; k < lc; k++) {
      const lib = libs[k];
      if (!game.isLegal(lib)) continue;

      game.play(lib);
      if (game.cells[idx] === 0) {
        game.undo();
        continue;
      }

      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      const [result, unused] = canReach4Libs(game, idx, budget);
      credits += unused;

      game.undo();

      if (result === true) return [true, credits];
      if (result === null) hasUnknown = true;
    }
    return [hasUnknown ? null : false, credits];
  }

  // Attacker's turn
  let hasUnknown = false;
  for (let k = 0; k < lc; k++) {
    const lib = libs[k];
    if (!game.isLegal(lib)) continue;

    game.play(lib);
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
      const [result, unused] = canReach4Libs(game, idx, budget);
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
 * Analyze a single chain (compatible with searchChain from tactics3.js)
 */
function searchChain(game, stoneIdx, nodeLimit = Infinity) {
  const libs = game.groupLibs(stoneIdx);
  const lc = libs.length;

  if (lc > 3) {
    console.warn(`searchChain: stone at ${stoneIdx} has ${lc} liberties (>3), skipping`);
    return null;
  }

  if (lc === 0) {
    return { libs: Array.from(libs), moverSucceeds: false, urgentLibs: [] };
  }

  const moverColor = game.current;
  const moverIsAttacker = moverColor !== game.cells[stoneIdx];

  const [result, _] = canReach4Libs(game, stoneIdx, nodeLimit);

  let moverSucceeds;
  if (moverIsAttacker) {
    moverSucceeds = result === false;
  } else {
    moverSucceeds = result === true;
  }

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

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { canReach4Libs, searchChain };
}
