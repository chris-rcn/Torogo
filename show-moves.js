#!/usr/bin/env node
'use strict';

// show-moves.js — replay games and show the agent's chosen move at every Nth position.
//
// Usage: node show-moves.js --file <games> --agent <name> [--interval <N>] [--budget <ms>]
//
//   --file      game records file (required)
//   --agent     agent name under ai/ (default: rave-bad)
//   --interval  show move at every Nth position (default: 31)
//   --budget    ms per move (default: 500)

const fs   = require('fs');
const path = require('path');
const { Game2, PASS, BLACK } = require('./game2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file      = get('--file', null);
const agentName = get('--agent', null);
const interval  = parseInt(get('--interval', '31'), 10);
const budgetMs  = parseInt(get('--budget', '1'), 10);
const showPass  = args.includes('--show-pass');

if (!file || !agentName) { process.stderr.write('Usage: node show-moves.js --file <games> --agent <name> [--interval <N>] [--budget <ms>]\n'); process.exit(1); }

const { getMove } = require(path.join(__dirname, 'ai', agentName + '.js'));

// Parse a two-letter move string ("gg") to flat index, or PASS for "..".
function parseGameMove(str, N) {
  if (str === '..') return PASS;
  const x = str.charCodeAt(0) - 97;
  const y = str.charCodeAt(1) - 97;
  return y * N + x;
}

const lines = fs.readFileSync(file, 'utf8').split('\n').filter(l => l.trim());

for (const line of lines) {
  const parts = line.split(',');
  const N     = parseInt(parts[0], 10);
  const moves = parts.slice(1);

  const game = new Game2(N);
  let nextShow = 1 + Math.floor(Math.random() * interval);
  for (let i = 0; i < moves.length; i++) {
    if (game.gameOver) break;

    if (i === nextShow) {
      nextShow += interval;
      const move = getMove(game, budgetMs);
      if (move.type === 'pass' && !showPass) continue;
      const playerName = game.current === BLACK ? 'Black' : 'White';
      const moveIdx = move.type === 'pass' ? PASS : move.y * N + move.x;
      const gNext = game.clone();
      gNext.play(moveIdx);
      process.stdout.write(`${playerName} to move:\n` + gNext.toString(moveIdx, { centerAt: moveIdx }) + '\n\n');
    }

    const idx = parseGameMove(moves[i], N);
    game.play(idx);
  }
}
