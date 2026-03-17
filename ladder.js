'use strict';

// Maximum recursion depth for ladder reading.  On a toroidal board there are
// no edges, so a ladder that meets no obstacles would run forever; the depth
// limit cuts it off and assumes the group can escape (conservative).
const MAX_LADDER_DEPTH = 200;

// ─── Public API ───────────────────────────────────────────────────────────────

// Returns true if the group containing the stone at (x, y) is in atari AND
// cannot reach 3 or more liberties through any escape sequence — i.e. it is
// caught in a losing ladder.
//
// Returns false when:
//   • there is no stone at (x, y)
//   • the group is not in atari (0 or 2+ liberties)
//   • the group can reach safety (3+ liberties) via some escape sequence
function isLadderCaptured(game, x, y) {
  const board = game.board;
  const group = board.getGroup(x, y);
  if (group.length === 0) return false;
  const libs = board.getLiberties(group);
  if (libs.size !== 1) return false;   // must be in atari
  return !_canEscape(game, x, y, 0);
}

// ─── Internal ─────────────────────────────────────────────────────────────────

// Returns true if the group at (x, y) — currently in atari — can escape to
// 3+ liberties.  Explores the forced sequence: escaping side plays on the one
// liberty; attacker tries every re-atari response; recurse.
function _canEscape(game, x, y, depth) {
  if (depth > MAX_LADDER_DEPTH) return true;  // assume escape at depth limit

  const board = game.board;
  const group = board.getGroup(x, y);
  if (group.length === 0) return false;

  const libs = board.getLiberties(group);
  if (libs.size >= 3) return true;
  if (libs.size === 0) return false;
  if (libs.size !== 1) return true;   // 2 liberties — not a ladder threat

  const escapingColor = board.get(x, y);
  const attackerColor = escapingColor === 'black' ? 'white' : 'black';

  // The single liberty — escaping side must play there.
  const [libStr] = libs;
  const [ex, ey] = libStr.split(',').map(Number);

  const g1 = game.clone();
  g1.current = escapingColor;
  if (g1.placeStone(ex, ey) === false) return false;  // suicide — no escape

  const newGroup = g1.board.getGroup(ex, ey);
  if (newGroup.length === 0) return false;

  const newLibs = g1.board.getLiberties(newGroup);
  if (newLibs.size >= 3) return true;   // immediately escaped
  if (newLibs.size === 0) return false; // captured by escape move (shouldn't occur)
  if (newLibs.size === 1) {
    // Still in atari after escape (escape ran into adjacent opponent stones).
    // Try continuing the escape sequence from the new position.
    return _canEscape(g1, ex, ey, depth + 1);
  }

  // 2 liberties after escape: attacker tries each one to re-atari the group.
  // If any attacker move leads to eventual capture, the group cannot escape.
  for (const lStr of newLibs) {
    const [lx, ly] = lStr.split(',').map(Number);

    const g2 = g1.clone();
    g2.current = attackerColor;
    if (g2.placeStone(lx, ly) === false) continue;  // illegal for attacker

    const afterGroup = g2.board.getGroup(ex, ey);
    if (afterGroup.length === 0) return false;  // group captured by attacker

    const afterLibs = g2.board.getLiberties(afterGroup);
    if (afterLibs.size === 0) return false;
    if (afterLibs.size === 1) {
      // Re-atarized — recurse with escaping side to move
      if (!_canEscape(g2, ex, ey, depth + 1)) return false;
    }
    // afterLibs.size >= 2: attacker's move failed to re-atari → group safe here
  }

  return true;
}

if (typeof module !== 'undefined') module.exports = { isLadderCaptured };
