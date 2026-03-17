#!/usr/bin/env node
'use strict';

// evalladders.js — evaluate ladder decisions across all atari groups in each position.
//
// Usage: node evalladders.js --file <path>
//   --file  path to game records produced by recordgames.js (required)
//
// For every non-pass move in every game, the script examines:
//
//   1-liberty groups (groups already in atari)
//     Defender (own group):
//       CORRECT  abandoned a doomed ladder
//       MISTAKE  played escape for a captured group
//       CORRECT  played escape when group can genuinely escape
//       MISSED   did not escape when group could have
//     Attacker (opponent group):
//       CORRECT  attacked a ladder-caught group
//       MISSED   did not attack a ladder-caught group
//       CORRECT  ignored a group that can escape
//       MISTAKE  attacked a group that can escape
//
//   2-liberty groups (attacker can initiate a ladder by playing on a liberty)
//     Attacker (opponent group with 2 liberties):
//       CORRECT  played a liberty that starts a winning ladder
//       MISSED   did not play either liberty when a winning-ladder start exists
//       CORRECT  correctly avoided playing a liberty when neither starts a
//                winning ladder
//       MISTAKE  played a liberty that the opponent can escape from

const fs = require('fs');
const { Game, DEFAULT_KOMI } = require('./game.js');
const { isLadderCaptured }   = require('./ladder.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);
if (!file) {
  console.error('Usage: node evalladders.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// ── Counters ──────────────────────────────────────────────────────────────────
const counts = {
  // 1-liberty groups — defender perspective (own group)
  def_captured_ignored:   0,  // CORRECT: abandoned a doomed ladder
  def_captured_escaped:   0,  // MISTAKE: played escape for a captured group
  def_free_escaped:       0,  // CORRECT: played escape when group can escape
  def_free_ignored:       0,  // MISSED:  did not escape when group could

  // 1-liberty groups — attacker perspective (opponent group)
  att1_captured_attacked: 0,  // CORRECT: attacked a captured group
  att1_captured_ignored:  0,  // MISSED:  did not attack a captured group
  att1_free_ignored:      0,  // CORRECT: ignored a group that can escape
  att1_free_attacked:     0,  // MISTAKE: attacked a group that can escape

  // 2-liberty groups — attacker perspective (opponent group)
  att2_winning_attacked:  0,  // CORRECT: played liberty that starts a winning ladder
  att2_winning_ignored:   0,  // MISSED:  did not play when a winning start exists
  att2_safe_ignored:      0,  // CORRECT: avoided playing when neither liberty wins
  att2_safe_attacked:     0,  // MISTAKE: played liberty the opponent can escape from
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Returns true if playing (lx, ly) on `game` leads to the group at (gx, gy)
// being captured — either immediately (0 libs after the play) or via a ladder.
function attackStartsCapture(game, gx, gy, lx, ly) {
  const g2 = game.clone();
  if (g2.placeStone(lx, ly) === false) return false;  // illegal move
  const afterGroup = g2.board.getGroup(gx, gy);
  if (afterGroup.length === 0) return true;            // captured immediately
  return isLadderCaptured(g2, gx, gy).captured;
}

// ── Game replay ───────────────────────────────────────────────────────────────
for (const line of lines) {
  const fields = line.split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1);

  const g = new Game(size, DEFAULT_KOMI);

  // moves[0] is already placed by the constructor; process from moves[1] onward.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (!token || token === '..') { g.pass(); continue; }

    const x      = token.charCodeAt(0) - 97;
    const y      = token.charCodeAt(1) - 97;
    const player = g.current;

    // Enumerate every unique group on the board that has 1 or 2 liberties.
    const visitedGroups = new Set();
    for (let gy = 0; gy < size; gy++) {
      for (let gx = 0; gx < size; gx++) {
        const color = g.board.get(gx, gy);
        if (color === null) continue;

        const cellKey = `${gx},${gy}`;
        if (visitedGroups.has(cellKey)) continue;
        const group = g.board.getGroup(gx, gy);
        for (const [sx, sy] of group) visitedGroups.add(`${sx},${sy}`);

        const libs    = g.board.getLiberties(group);
        const libSize = libs.size;
        if (libSize !== 1 && libSize !== 2) continue;

        const libArr = [...libs].map(l => {
          const [lx, ly] = l.split(',').map(Number);
          return { x: lx, y: ly };
        });

        // ── 1-liberty group ───────────────────────────────────────────────
        if (libSize === 1) {
          const { x: lx, y: ly } = libArr[0];
          const played   = (lx === x && ly === y);
          const { captured } = isLadderCaptured(g, gx, gy);

          if (color === player) {
            // Defender: own group in atari
            if (captured) {
              if (played) counts.def_captured_escaped++;
              else        counts.def_captured_ignored++;
            } else {
              if (played) counts.def_free_escaped++;
              else        counts.def_free_ignored++;
            }
          } else {
            // Attacker: opponent group in atari
            if (captured) {
              if (played) counts.att1_captured_attacked++;
              else        counts.att1_captured_ignored++;
            } else {
              if (played) counts.att1_free_attacked++;
              else        counts.att1_free_ignored++;
            }
          }
        }

        // ── 2-liberty group — attacker perspective only ───────────────────
        if (libSize === 2 && color !== player) {
          // Check whether each liberty, when played by the attacker, starts
          // a winning ladder.
          const wins = libArr.map(lib =>
            attackStartsCapture(g, gx, gy, lib.x, lib.y));

          const hasWinningAttack = wins[0] || wins[1];

          // Did the current move land on one of the two liberties?
          const playedIdx = libArr.findIndex(lib => lib.x === x && lib.y === y);

          if (playedIdx >= 0) {
            // Attacker played on a liberty of this group.
            if (wins[playedIdx]) counts.att2_winning_attacked++;
            else                 counts.att2_safe_attacked++;
          } else {
            // Attacker did not play on any liberty of this group.
            if (hasWinningAttack) counts.att2_winning_ignored++;
            else                  counts.att2_safe_ignored++;
          }
        }
      }
    }

    g.placeStone(x, y);
  }
}

// ── Report ────────────────────────────────────────────────────────────────────
function pct(n, d) {
  return d === 0 ? ' n/a ' : (100 * n / d).toFixed(1) + '%';
}

function row(label, n, d, tag) {
  const nd = `${n}/${d}`;
  console.log(`  ${label.padEnd(46)} ${nd.padStart(12)}  ${pct(n, d).padStart(6)}  [${tag}]`);
}

const defCap  = counts.def_captured_escaped   + counts.def_captured_ignored;
const defFre  = counts.def_free_escaped       + counts.def_free_ignored;
const att1Cap = counts.att1_captured_attacked + counts.att1_captured_ignored;
const att1Fre = counts.att1_free_attacked     + counts.att1_free_ignored;
const att2Win = counts.att2_winning_attacked  + counts.att2_winning_ignored;
const att2Saf = counts.att2_safe_attacked     + counts.att2_safe_ignored;

console.log('\n── Defender · 1-liberty groups (own groups in atari) ──────────────────────');
row('Abandoned a doomed ladder',             counts.def_captured_ignored,   defCap,  'CORRECT');
row('Escaped a captured group',              counts.def_captured_escaped,   defCap,  'MISTAKE');
row('Escaped when group can escape',         counts.def_free_escaped,       defFre,  'CORRECT');
row('Did not escape an escapable group',     counts.def_free_ignored,       defFre,  'MISSED ');

console.log('\n── Attacker · 1-liberty groups (opponent already in atari) ────────────────');
row('Attacked a doomed ladder',              counts.att1_captured_attacked, att1Cap, 'CORRECT');
row('Ignored a doomed ladder',               counts.att1_captured_ignored,  att1Cap, 'MISSED ');
row('Ignored an escapable ladder',           counts.att1_free_ignored,      att1Fre, 'CORRECT');
row('Attacked an escapable ladder',          counts.att1_free_attacked,     att1Fre, 'MISTAKE');

console.log('\n── Attacker · 2-liberty groups (initiating a ladder) ──────────────────────');
row('Played liberty that starts winning ladder', counts.att2_winning_attacked, att2Win, 'CORRECT');
row('Missed a winning ladder start',             counts.att2_winning_ignored,  att2Win, 'MISSED ');
row('Avoided a futile atari',                    counts.att2_safe_ignored,     att2Saf, 'CORRECT');
row('Played atari opponent can escape from',     counts.att2_safe_attacked,    att2Saf, 'MISTAKE');
console.log('');
