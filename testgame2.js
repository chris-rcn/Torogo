'use strict';

// Randomly plays 1000 games on both Game and Game2 in lockstep, comparing
// board state, group data, legality, and eye detection at every position.

const { Game } = require('./game.js');
const { Game2, PASS, BLACK, WHITE } = require('./game2.js');

let pass = 0, fail = 0;

function assert(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}

// Translate game.js color → integer (0/1/2)
function colorInt(c) { return c === 'black' ? BLACK : c === 'white' ? WHITE : 0; }

// Compare full state of game (Game) vs g2 (Game2) at the current position.
function compareState(game, g2, label) {
  const N = game.boardSize;

  // ── Cell-by-cell board comparison ──────────────────────────────────────────
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx  = y * N + x;
      const gVal = colorInt(game.board.get(x, y));
      const g2Val = g2.cells[idx];
      assert(gVal === g2Val,
        `${label} cell(${x},${y}): game=${gVal} g2=${g2Val}`);
    }
  }

  // ── Turn ───────────────────────────────────────────────────────────────────
  const curInt = colorInt(game.current);
  assert(curInt === g2.current,
    `${label} current: game=${game.current} g2=${g2.current}`);

  // ── Ko ─────────────────────────────────────────────────────────────────────
  const koGame = game.koFlag ? game.koFlag.y * N + game.koFlag.x : PASS;
  assert(koGame === g2.ko,
    `${label} ko: game=${koGame} g2=${g2.ko}`);

  // ── Scalar state ───────────────────────────────────────────────────────────
  assert(game.consecutivePasses === g2.consecutivePasses,
    `${label} consecutivePasses: ${game.consecutivePasses} vs ${g2.consecutivePasses}`);
  assert(game.gameOver === g2.gameOver,
    `${label} gameOver: ${game.gameOver} vs ${g2.gameOver}`);
  assert(game.moveCount === g2.moveCount,
    `${label} moveCount: ${game.moveCount} vs ${g2.moveCount}`);

  // ── Group stone counts and liberty counts (per occupied cell) ───────────────
  const board = game.board;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      if (g2.cells[idx] === 0) continue; // skip empty

      const gid1 = board._gid[idx];
      const gid2 = g2._gid[idx];
      if (gid1 === -1 || gid2 === -1) {
        assert(false, `${label} cell(${x},${y}) occupied but gid=-1`);
        continue;
      }

      const stones1 = board._groups.get(gid1).stones.size;
      const stones2 = g2._ss[gid2];
      assert(stones1 === stones2,
        `${label} groupSize(${x},${y}): game=${stones1} g2=${stones2}`);

      const libs1 = board._groups.get(gid1).liberties.size;
      const libs2 = g2._ls[gid2];
      assert(libs1 === libs2,
        `${label} liberties(${x},${y}): game=${libs1} g2=${libs2}`);
    }
  }

  // ── isLegal for every empty cell ───────────────────────────────────────────
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      if (g2.cells[idx] !== 0) continue;
      const legal1 = game.isLegal(x, y);
      const legal2 = g2.isLegal(idx);
      assert(legal1 === legal2,
        `${label} isLegal(${x},${y}): game=${legal1} g2=${legal2}`);
    }
  }

  // ── isTrueEye for every empty cell (current player's perspective) ──────────
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      if (g2.cells[idx] !== 0) continue;
      const eye1 = board.isTrueEye(x, y, game.current);
      const eye2 = g2.isTrueEye(idx);
      assert(eye1 === eye2,
        `${label} isTrueEye(${x},${y}) color=${game.current}: game=${eye1} g2=${eye2}`);
    }
  }
}

// ── Run N_GAMES random games ──────────────────────────────────────────────────

const N_GAMES  = 1000;
const PASS_PROB = 0.05; // 5 % chance to pass even when moves exist

let totalMoves = 0;

for (let gameIdx = 0; gameIdx < N_GAMES; gameIdx++) {
  const game = new Game(9, 3.5);
  const g2   = new Game2(9);

  // Both constructors place a black stone at center; sync initial state.
  // (Game uses komi=3.5 but that doesn't affect board/group state during play)
  compareState(game, g2, `game${gameIdx} init`);

  while (!game.gameOver) {
    // Collect all legal non-pass moves from game (ground truth).
    const legal = [];
    for (let y = 0; y < 9; y++)
      for (let x = 0; x < 9; x++)
        if (game.isLegal(x, y)) legal.push([x, y]);

    // Choose a move.
    let x, y, idx;
    if (legal.length === 0 || Math.random() < PASS_PROB) {
      // Pass
      game.pass();
      g2.play(PASS);
    } else {
      const pick = legal[(Math.random() * legal.length) | 0];
      [x, y] = pick;
      idx     = y * 9 + x;
      game.placeStone(x, y);
      g2.play(idx);
    }

    totalMoves++;
    const label = `game${gameIdx} move${totalMoves}`;
    compareState(game, g2, label);

    if (fail > 20) {
      console.error('Too many failures, stopping early.');
      process.exit(1);
    }
  }
}

console.log(`\n${N_GAMES} games, ${totalMoves} moves checked.`);
console.log(`PASS: ${pass}   FAIL: ${fail}`);
if (fail > 0) process.exit(1);
