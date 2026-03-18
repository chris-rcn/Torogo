'use strict';

// ─── Public API ───────────────────────────────────────────────────────────────

// Examines the group containing the stone at (x, y) and returns:
//
//   { captured: boolean, moves: Array<{x, y}> }
//
// captured — true when the group is in atari and cannot reach 3+ liberties
//            through any escape sequence (losing ladder).
//
// moves    — the immediately actionable points for the current player
//            (game.current), derived from the ladder analysis:
//
//   • Attacker's turn  + captured === true
//       The 1–2 valid re-atari points the attacker should play after the
//       defender's forced escape, or the single liberty when the group is
//       captured immediately (suicide / zero liberties after escape).
//
//   • Defender's turn  + captured === false
//       The single liberty — the one move the defender must play to escape.
//
//   • All other combinations → moves is an empty array.
//
// Returns { captured: false, moves: [] } when:
//   • there is no stone at (x, y)
//   • the group is not in atari (0 or 2+ liberties)
function isLadderCaptured(game, x, y) {
  const board = game.board;
  const group = board.getGroup(x, y);
  if (group.length === 0) return { captured: false, moves: [] };
  const libs = board.getLiberties(group);
  if (libs.size !== 1) return { captured: false, moves: [] };

  const [libStr] = libs;
  const [lx, ly] = libStr.split(',').map(Number);

  const stoneColor    = board.get(x, y);
  const attackerColor = stoneColor === 'black' ? 'white' : 'black';

  const result = _canEscape(game, x, y);

  if (result === null) {
    // Group can escape.  Only relevant for the defender.
    return {
      captured: false,
      moves: game.current === stoneColor ? [{ x: lx, y: ly }] : [],
    };
  }

  // Group is captured.  Only provide attack guidance to the attacker.
  if (game.current === attackerColor) {
    // result.length === 0 means the group is trapped immediately (escape is
    // suicide or leads to 0 liberties) — the attacker simply plays the single
    // liberty.  Otherwise result holds the 1–2 valid re-atari points.
    const attackMoves = result.length > 0 ? result : [{ x: lx, y: ly }];
    return { captured: true, moves: attackMoves };
  }
  return { captured: true, moves: [] };
}

// ─── Internal ─────────────────────────────────────────────────────────────────

// Returns null  when the group at (x, y) — currently in atari — can escape to
// 3+ liberties (not captured).
//
// Returns an Array<{x,y}> when the group cannot escape (captured).  The array
// holds the valid re-atari points at the current depth of the forced sequence:
//   • []          — group captured immediately (suicide / 0 libs after escape)
//   • [{x,y}, …]  — 1 or 2 attacker re-atari points that lead to eventual
//                   capture after the defender reaches 2 liberties
function _canEscape(game, x, y) {
  const board = game.board;
  const group = board.getGroup(x, y);
  if (group.length === 0) return [];  // already captured

  const libs = board.getLiberties(group);
  if (libs.size >= 3) return null;    // escaped
  if (libs.size === 0) return [];     // captured
  if (libs.size !== 1) return null;   // 2 liberties — not a ladder threat

  const escapingColor = board.get(x, y);
  const attackerColor = escapingColor === 'black' ? 'white' : 'black';

  // The single liberty — escaping side must play there.
  const [libStr] = libs;
  const [ex, ey] = libStr.split(',').map(Number);

  const g1 = game.clone();
  g1.current = escapingColor;
  if (g1.placeStone(ex, ey) === false) return [];  // suicide — no escape

  const newGroup = g1.board.getGroup(ex, ey);
  if (newGroup.length === 0) return [];  // captured by escape move (shouldn't occur)

  const newLibs = g1.board.getLiberties(newGroup);
  if (newLibs.size >= 3) return null;   // immediately escaped
  if (newLibs.size === 0) return [];    // captured after escape
  if (newLibs.size === 1) {
    // Still in atari after escape (ran into adjacent opponent stones).
    // Continue the escape sequence from the new position.
    return _canEscape(g1, ex, ey);
  }

  // 2 liberties after escape: collect all attacker re-atari moves that lead
  // to eventual capture (1 or 2 of the 2 available liberties).
  const validReataris = [];
  for (const lStr of newLibs) {
    const [lx, ly] = lStr.split(',').map(Number);

    const g2 = g1.clone();
    g2.current = attackerColor;
    if (g2.placeStone(lx, ly) === false) continue;  // illegal for attacker

    const afterGroup = g2.board.getGroup(ex, ey);
    if (afterGroup.length === 0) {
      validReataris.push({ x: lx, y: ly });  // group captured immediately
      continue;
    }

    const afterLibs = g2.board.getLiberties(afterGroup);
    if (afterLibs.size === 0) {
      validReataris.push({ x: lx, y: ly });
    } else if (afterLibs.size === 1) {
      // Re-atarized — recurse with escaping side to move.
      // _canEscape returning non-null means captured → this re-atari is valid.
      if (_canEscape(g2, ex, ey) !== null) {
        validReataris.push({ x: lx, y: ly });
      }
    }
    // afterLibs.size >= 2: attacker's move failed to re-atari → skip
  }

  // Any valid re-atari means the group is captured.
  return validReataris.length > 0 ? validReataris : null;
}

// Examines the group containing the stone at (gx, gy), which must have 1 or 2
// liberties.  For each liberty, simulates both colours playing it first and
// searches whether the group can reach 3+ liberties.
//
// Returns an array — one entry per liberty — of:
//   { liberty: {x, y}, current: boolean, opponent: boolean }
//
// current / opponent — true when the group can escape to 3+ liberties
//   after that colour plays the liberty (current = game.current moves first).
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
        escaped = grp.length > 0 && _canEscape(g, gx, gy) === null;
      }
      entry[color === mover ? 'current' : 'opponent'] = escaped;
    }
    results.push(entry);
  }
  return results;
}

if (typeof module !== 'undefined') module.exports = { isLadderCaptured, getLadderStatus };
