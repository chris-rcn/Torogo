#!/usr/bin/env node
'use strict';

// deadladders.js — flag moves that attempt to extend a group known to be dead.
//
// Usage: node deadladders.js --file <path> [--min-stones N]
//   --min-stones  only flag dead groups with at least N stones (default: 1)
//
// Reads game records produced by recordgames.js.  For each non-pass move,
// checks whether the stone being played is adjacent to any friendly group that:
//   • has at least --min-stones stones,
//   • has exactly 1 liberty (already in atari), and
//   • cannot escape (getLadderStatus reports canEscape = false).
//
// When such a move is found the board position is printed with the move marked.

const fs = require('fs');
const { Game } = require('./game.js');
const { getLadderStatus }    = require('./ladder.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file      = get('--file', null);
const minStones = parseInt(get('--min-stones', '1'), 10);
if (!file) {
  console.error('Usage: node deadladders.js --file <path> [--min-stones N]');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Convert 0-based x,y to two-letter move label (same encoding as recordgames.js).
function toLabel(x, y) {
  return String.fromCharCode(97 + x) + String.fromCharCode(97 + y);
}

let gameIdx = 0;
let found   = 0;

for (const line of lines) {
  gameIdx++;
  const fields = line.split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1);

  const g = new Game(size);

  // moves[0] is the initial center stone placed by the constructor; start at 1.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (!token || token === '..') { g.pass(); continue; }

    const x      = token.charCodeAt(0) - 97;
    const y      = token.charCodeAt(1) - 97;
    const player = g.current;

    // Build the set of cell keys that are neighbors of the planned move.
    const moveNeighborKeys = new Set(
      g.board.getNeighbors(x, y).map(([nx, ny]) => `${nx},${ny}`)
    );

    // Scan every unique friendly group with 1 liberty.
    const visitedGids = new Set();
    let reported = false;

    for (let gy = 0; gy < size && !reported; gy++) {
      for (let gx = 0; gx < size && !reported; gx++) {
        if (g.board.get(gx, gy) !== player) continue;

        const gid = g.board._gid[g.board._idx(gx, gy)];
        if (visitedGids.has(gid)) continue;
        visitedGids.add(gid);

        const group = g.board.getGroup(gx, gy);
        if (group.length < minStones) continue;

        const libs  = g.board.getLiberties(group);
        if (libs.size !== 1) continue;

        if (getLadderStatus(g, gx, gy)[0].canEscape) continue;

        // Dead group confirmed.  Does the planned move touch it?
        const touches = group.some(([sx, sy]) => moveNeighborKeys.has(`${sx},${sy}`));
        if (!touches) continue;

        // Report this position.
        found++;
        reported = true;

        const [libStr]   = libs;
        const [lx, ly]   = libStr.split(',').map(Number);
        const moveLabel  = toLabel(x, y);
        const libLabel   = toLabel(lx, ly);
        const plural     = group.length !== 1 ? 's' : '';

        console.log(
          `\nGame ${gameIdx}, move ${mi}  ` +
          `${player} plays ${moveLabel} — ` +
          `extends dead ${player} group ` +
          `(${group.length} stone${plural}, liberty at ${libLabel})`
        );
        console.log(g.board.toAscii({ x, y }));
      }
    }

    g.placeStone(x, y);
  }
}

console.log(`\n${found} position${found !== 1 ? 's' : ''} found across ${gameIdx} game${gameIdx !== 1 ? 's' : ''}.`);
