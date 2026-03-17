#!/usr/bin/env node
'use strict';

// evalladders.js — count ladder mistakes in game records.
//
// Usage: node evalladders.js --file <path>
//   --file  path to game records produced by recordgames.js (required)
//
// For every non-pass move played, the script checks the groups in atari
// that are adjacent to the played cell and classifies the move:
//
//   "saves lost ladder"      — the player plays the escape liberty of one of
//                              their own groups that is already caught in a
//                              losing ladder (the escape will fail).
//
//   "attacks escapable"      — the player plays the single liberty of an
//                              opponent group that is in atari but can escape
//                              (the attack will be fruitless).
//
// A single move may satisfy either, both, or neither condition.
// Output: counts and percentages over all moves analysed.

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

let totalMoves        = 0;
let savesLost         = 0;  // move escapes a ladder-captured own group
let attacksEscapable  = 0;  // move attacks an opponent group that can escape

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
    const opp    = player === 'black' ? 'white' : 'black';

    // Examine unique groups adjacent to (x, y) that are in atari with their
    // single liberty exactly at (x, y).
    let moveSavesLost        = false;
    let moveAttacksEscapable = false;
    const visited = new Set();

    for (const [nx, ny] of g.board.getNeighbors(x, y)) {
      const color = g.board.get(nx, ny);
      if (color === null) continue;

      // Deduplicate: skip if we already processed this group.
      const key = `${nx},${ny}`;
      if (visited.has(key)) continue;
      const group = g.board.getGroup(nx, ny);
      for (const [gx, gy] of group) visited.add(`${gx},${gy}`);

      // Only care about groups in atari whose single liberty is (x, y).
      const libs = g.board.getLiberties(group);
      if (libs.size !== 1) continue;
      const [libStr] = libs;
      const [lx, ly] = libStr.split(',').map(Number);
      if (lx !== x || ly !== y) continue;

      // Group is in atari with liberty at (x, y) — the player is interacting
      // with this group by playing here.
      const { captured } = isLadderCaptured(g, nx, ny);

      if (color === player && captured) {
        // Player escapes their own group that is already caught in a ladder.
        moveSavesLost = true;
      } else if (color === opp && !captured) {
        // Player attacks an opponent group that can still escape.
        moveAttacksEscapable = true;
      }
    }

    if (moveSavesLost)        savesLost++;
    if (moveAttacksEscapable) attacksEscapable++;
    totalMoves++;

    g.placeStone(x, y);
  }
}

function pct(n, d) {
  return d === 0 ? '0.00' : (100 * n / d).toFixed(2);
}

console.log(`Total moves analysed  : ${totalMoves}`);
console.log(`Saves a lost ladder   : ${savesLost.toString().padStart(6)} (${pct(savesLost, totalMoves)}%)`);
console.log(`Attacks escapable     : ${attacksEscapable.toString().padStart(6)} (${pct(attacksEscapable, totalMoves)}%)`);
