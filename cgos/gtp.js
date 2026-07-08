'use strict';

// gtp.js — GTP (Go Text Protocol) wrapper exposing any ai/ agent as an engine.
//
// Usage:
//   node cgos/gtp.js --p <policy> [--budget <ms>] [--size <n>]
//
// Options:
//   --p       <policy>  AI policy: filename without .js inside ai/  (required)
//   --budget  <ms>      Fixed time budget per move in ms            (default 1)
//   --size    <n>       Initial board size                          (default 9)
//
// Speaks GTP 2 on stdin/stdout, enough for the CGOS client (boardsize,
// clear_board, komi, play, genmove, quit + the cgos-* notifications).
// The budget is part of the engine's identity: register slow searchers
// under names that include it (e.g. rave-500).
//
// Toroidal-board notes:
//   - Game2 normally auto-plays black's first move at the center (every
//     first move is equivalent on a torus).  Under GTP the engine answers
//     its first genmove with the center point and accepts any first move
//     from the opponent.
//   - Coordinates use standard GTP columns A..T skipping I, rows 1..N;
//     column a + row 1 maps to Game2 index 0.

const path = require('path');
const readline = require('readline');
const { Game2, PASS, BLACK, WHITE, setKomi } = require('../game2.js');
const Util = require('../util.js');

const GTP_COLS = 'abcdefghjklmnopqrstuvwxyz';   // no 'i'

// GTP coordinate string → Game2 flat index, or PASS, or null if malformed.
function gtpToIdx(str, N) {
  const s = str.toLowerCase();
  if (s === 'pass') return PASS;
  const x = GTP_COLS.indexOf(s[0]);
  const y = parseInt(s.slice(1), 10) - 1;
  if (x < 0 || x >= N || !(y >= 0 && y < N)) return null;
  return y * N + x;
}

// Game2 flat index → GTP coordinate string.
function idxToGtp(idx, N) {
  if (idx === PASS) return 'pass';
  return GTP_COLS[idx % N] + ((idx / N | 0) + 1);
}

// Agent result object → flat index (mirrors selfplay.js).
function agentResultToIdx(move, N) {
  if (move.move !== undefined) return move.move;
  return move.type === 'place' ? move.y * N + move.x : PASS;
}

function makeEngine(policyName, budgetMs, size) {
  // Agents log progress via console.log; stdout belongs to the GTP stream,
  // so route all console output to stderr before loading the agent.
  const util = require('util');
  console.log = console.info = console.warn =
    (...a) => process.stderr.write(util.format(...a) + '\n');

  const { getMove } = require(path.join(__dirname, '..', 'ai', policyName + '.js'));

  const st = { size, game: new Game2(size, false) };

  function clear() { st.game = new Game2(st.size, false); }

  function colorOf(arg) {
    const c = (arg || '').toLowerCase();
    if (c === 'b' || c === 'black') return BLACK;
    if (c === 'w' || c === 'white') return WHITE;
    return null;
  }

  const commands = {
    protocol_version: () => '2',
    name:             () => `torogo-${policyName}`,
    version:          () => `budget-${budgetMs}ms`,
    list_commands:    () => Object.keys(commands).join('\n'),
    known_command:    args => commands[args[0]] ? 'true' : 'false',
    quit:             () => '',

    boardsize: args => {
      const n = parseInt(args[0], 10);
      if (!Number.isInteger(n) || n < 3 || n > 25) throw new Error('unacceptable size');
      st.size = n;
      clear();
      return '';
    },
    clear_board: () => { clear(); return ''; },
    komi: args => {
      const k = parseFloat(args[0]);
      if (isNaN(k)) throw new Error('syntax error');
      setKomi(st.size, k);   // keep agents' internal scoring in sync
      return '';
    },

    play: args => {
      const color = colorOf(args[0]);
      const idx = gtpToIdx(args[1] || '', st.size);
      if (color === null || idx === null) throw new Error('syntax error');
      if (st.game.gameOver) return '';               // trailing pass after game end
      if (st.game.current !== color) throw new Error('wrong color to move');
      if (!st.game.play(idx)) throw new Error('illegal move');
      return '';
    },

    genmove: args => {
      const color = colorOf(args[0]);
      if (color === null) throw new Error('syntax error');
      if (st.game.current !== color) throw new Error('wrong color to move');
      if (st.game.gameOver) return 'pass';

      let idx;
      if (st.game.moveCount === 0) {
        // First move of the game: on a torus all points are equivalent, and
        // Torogo agents expect the conventional center opening.
        idx = (st.size >> 1) * st.size + (st.size >> 1);
      } else {
        idx = agentResultToIdx(getMove(st.game, budgetMs), st.size);
      }
      if (!st.game.play(idx)) {
        // Never trust a stuck agent into forfeiting the game.
        idx = st.game.randomLegalMove();
        if (!st.game.play(idx)) { st.game.play(PASS); idx = PASS; }
      }
      return idxToGtp(idx, st.size);
    },

    // Accepted but ignored: budget is fixed per engine identity.
    time_settings: () => '',
    time_left:     () => '',

    // CGOS notifications.
    'cgos-opponent_name':   () => '',
    'cgos-opponent_rating': () => '',
    'cgos-gameover':        () => '',
  };

  return commands;
}

function runGtp(commands) {
  const rl = readline.createInterface({ input: process.stdin, terminal: false });

  rl.on('line', line => {
    // GTP preprocessing: strip comments, control chars, blank lines.
    let s = line.replace(/#.*/, '').replace(/[\x00-\x09\x0b-\x1f\x7f]/g, ' ').trim();
    if (!s) return;

    let id = '';
    const parts = s.split(/\s+/);
    if (/^\d+$/.test(parts[0])) id = parts.shift();
    const cmd = (parts.shift() || '').toLowerCase();

    const respond = (ok, text) => {
      const sep = text ? ' ' : '';
      process.stdout.write(`${ok ? '=' : '?'}${id}${sep}${text}\n\n`);
    };

    const fn = commands[cmd];
    if (!fn) { respond(false, 'unknown command'); return; }
    try {
      respond(true, fn(parts));
    } catch (e) {
      respond(false, e.message || 'error');
    }
    if (cmd === 'quit') process.exit(0);
  });

  rl.on('close', () => process.exit(0));
}

if (require.main === module) {
  const opts = Util.parseArgs(process.argv.slice(2), ['help']);
  if (opts.help || !opts.p) {
    console.error('Usage: node cgos/gtp.js --p <policy> [--budget <ms>] [--size <n>]');
    process.exit(opts.help ? 0 : 1);
  }
  const budgetMs = opts.getInt('budget', 1);
  const size     = opts.getInt('size', 9);
  runGtp(makeEngine(opts.p, budgetMs, size));
}

module.exports = { gtpToIdx, idxToGtp, agentResultToIdx, makeEngine };
