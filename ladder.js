'use strict';

// ─── Internal ─────────────────────────────────────────────────────────────────

// Returns true when the group at (x, y) can reach 3+ liberties despite best
// attacker play.  Simple two-player DFS: whose turn it is determines the move.
//   0 libs            → false (captured)
//   3+ libs           → true (escaped)
//   1 lib, defender   → defender plays it, then recurse (now attacker's turn)
//   1–2 libs, attacker → attacker tries each lib; any successful capture → false
function _canReach3Libs(game, x, y) {
  const board = game.board;
  const group = board.getGroup(x, y);
  if (group.length === 0) return false;

  const libs = board.getLiberties(group);
  if (libs.size >= 3) return true;
  if (libs.size === 0) return false;

  const defenderColor = board.get(x, y);
  const attackerColor = defenderColor === 'black' ? 'white' : 'black';

  if (libs.size === 1 && game.current === defenderColor) {
    // Defender's turn in atari: play the only liberty and see if we escape.
    const [libStr] = libs;
    const [lx, ly] = libStr.split(',').map(Number);
    const g1 = game.clone();
    if (g1.placeStone(lx, ly) === false) return false;  // suicide
    const newGroup = g1.board.getGroup(x, y);
    if (newGroup.length === 0) return false;
    return _canReach3Libs(g1, x, y);  // now attacker's turn
  }

  // 1 lib (attacker's turn) or 2 libs: attacker tries each lib as a re-atari.
  // If any re-atari leads to eventual capture, the group cannot reach 3+ libs.
  for (const lStr of libs) {
    const [lx, ly] = lStr.split(',').map(Number);

    const g2 = game.clone();
    g2.current = attackerColor;
    if (g2.placeStone(lx, ly) === false) continue;  // illegal for attacker — skip

    const afterGroup = g2.board.getGroup(x, y);
    if (afterGroup.length === 0) return false;  // group captured immediately

    const afterLibs = g2.board.getLiberties(afterGroup);
    if (afterLibs.size === 0) return false;
    if (afterLibs.size === 1 && !_canReach3Libs(g2, x, y)) return false;
    // afterLibs.size >= 2: attacker failed to re-atari — try next liberty
  }

  // No attacker move leads to capture → group can reach 3+ libs.
  return true;
}

// Examines the group containing the stone at (gx, gy), which must have 1 or 2
// liberties.  For each liberty, simulates both colours playing it first and
// searches whether the group can reach 3+ liberties.
//
// Returns an array — one entry per liberty — of:
//   { liberty: {x, y}, canEscape: boolean, canEscapeAfterPass: boolean }
//
// canEscape / canEscapeAfterPass — true when the group can escape to 3+ liberties
//   after that colour plays the liberty (canEscape = game.current moves first).
//
// Logs a warning and returns null when the group has more than 2 liberties.
function getLadderStatus(game, gx, gy) {
  const board = game.board;
  const group = board.getGroup(gx, gy);
  if (group.length === 0) return [];
  const libs = board.getLiberties(group);
  if (libs.size > 2) {
    console.warn(`getLadderStatus: group at (${gx},${gy}) has ${libs.size} liberties (expected ≤ 2)`);
    return null;
  }

  const mover    = game.current;
  const opp      = mover === 'black' ? 'white' : 'black';
  const results  = [];
  for (const lstr of libs) {
    const [lx, ly] = lstr.split(',').map(Number);
    const entry = { liberty: { x: lx, y: ly } };
    for (const color of [mover, opp]) {
      const g = game.clone();
      g.current = color;
      let escaped;
      if (g.placeStone(lx, ly) === false) {
        escaped = false;  // illegal move — liberty is unreachable for this colour
      } else {
        const grp = g.board.getGroup(gx, gy);
        escaped = grp.length > 0 && _canReach3Libs(g, gx, gy);
      }
      entry[color === mover ? 'canEscape' : 'canEscapeAfterPass'] = escaped;
    }
    results.push(entry);
  }
  return results;
}

if (typeof module !== 'undefined') module.exports = { getLadderStatus };
