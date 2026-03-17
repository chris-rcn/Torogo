#!/usr/bin/env node
'use strict';

// evalladders.js — evaluate ladder decisions across all atari groups in each position.
//
// Usage: node evalladders.js --file <path>
//   --file  path to game records produced by recordgames.js (required)
//
// For every non-pass move in every game, the script examines ALL groups
// currently in atari on the board and classifies the current player's
// decision for each group:
//
//   Defender perspective (own groups in atari)
//     CORRECT  — abandoned a doomed ladder (captured group, did not play escape)
//     MISTAKE  — played escape for a captured group (escape will fail)
//     CORRECT  — played escape when group can genuinely escape
//     MISSED   — did not play escape when group could have escaped
//
//   Attacker perspective (opponent groups in atari)
//     CORRECT  — attacked a ladder-caught group (played its liberty)
//     MISSED   — did not attack a ladder-caught group
//     CORRECT  — ignored a group that can escape (did not play its liberty)
//     MISTAKE  — attacked a group that can escape (futile attack)
//
// Output: counts and percentages for all eight categories.

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
// def_ = own group in atari,  att_ = opponent group in atari
// captured  = ladder-caught,  free = can escape
// played    = current move is on the group's single liberty
const counts = {
  def_captured_ignored:  0,  // CORRECT: abandoned a doomed ladder
  def_captured_escaped:  0,  // MISTAKE: played escape for a captured group
  def_free_escaped:      0,  // CORRECT: played escape when group can escape
  def_free_ignored:      0,  // MISSED:  did not escape when group could

  att_captured_attacked: 0,  // CORRECT: attacked a captured group
  att_captured_ignored:  0,  // MISSED:  did not attack a captured group
  att_free_ignored:      0,  // CORRECT: ignored a group that can escape
  att_free_attacked:     0,  // MISTAKE: attacked a group that can escape
};

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

    // Enumerate every unique group currently in atari across the whole board.
    const visitedGroups = new Set();
    for (let gy = 0; gy < size; gy++) {
      for (let gx = 0; gx < size; gx++) {
        const color = g.board.get(gx, gy);
        if (color === null) continue;

        const cellKey = `${gx},${gy}`;
        if (visitedGroups.has(cellKey)) continue;
        const group = g.board.getGroup(gx, gy);
        for (const [sx, sy] of group) visitedGroups.add(`${sx},${sy}`);

        const libs = g.board.getLiberties(group);
        if (libs.size !== 1) continue;  // only atari groups

        const [libStr] = libs;
        const [lx, ly] = libStr.split(',').map(Number);
        // Did the current move land on this group's single liberty?
        const played = (lx === x && ly === y);

        const { captured } = isLadderCaptured(g, gx, gy);

        if (color === player) {
          // ── Defender: own group in atari ─────────────────────────────────
          if (captured) {
            if (played) counts.def_captured_escaped++;
            else        counts.def_captured_ignored++;
          } else {
            if (played) counts.def_free_escaped++;
            else        counts.def_free_ignored++;
          }
        } else {
          // ── Attacker: opponent group in atari ────────────────────────────
          if (captured) {
            if (played) counts.att_captured_attacked++;
            else        counts.att_captured_ignored++;
          } else {
            if (played) counts.att_free_attacked++;
            else        counts.att_free_ignored++;
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
  console.log(`  ${label.padEnd(44)} ${nd.padStart(12)}  ${pct(n, d).padStart(6)}  [${tag}]`);
}

const defCap = counts.def_captured_escaped  + counts.def_captured_ignored;
const defFre = counts.def_free_escaped      + counts.def_free_ignored;
const attCap = counts.att_captured_attacked + counts.att_captured_ignored;
const attFre = counts.att_free_attacked     + counts.att_free_ignored;

console.log('\n── Defender (own groups in atari) ─────────────────────────────────────────');
row('Abandoned a doomed ladder',           counts.def_captured_ignored,  defCap, 'CORRECT');
row('Escaped a captured group',            counts.def_captured_escaped,  defCap, 'MISTAKE');
row('Escaped when group can escape',       counts.def_free_escaped,      defFre, 'CORRECT');
row('Did not escape an escapable group',   counts.def_free_ignored,      defFre, 'MISSED ');

console.log('\n── Attacker (opponent groups in atari) ─────────────────────────────────────');
row('Attacked a doomed ladder',            counts.att_captured_attacked, attCap, 'CORRECT');
row('Ignored a doomed ladder',             counts.att_captured_ignored,  attCap, 'MISSED ');
row('Ignored an escapable ladder',         counts.att_free_ignored,      attFre, 'CORRECT');
row('Attacked an escapable ladder',        counts.att_free_attacked,     attFre, 'MISTAKE');
console.log('');
