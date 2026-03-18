'use strict';

const { Board, Game, DEFAULT_KOMI, ZOBRIST } = require('./game.js');

let pass = 0, fail = 0;

function assert(cond, msg) {
  if (cond) { pass++; }
  else { fail++; console.error('  FAIL:', msg); }
}

function section(name) { console.log(`\n── ${name} ──`); }

// ─── Board basics ────────────────────────────────────────────────────────────

section('Board construction');
{
  const b = new Board(9);
  assert(b.size === 9, 'board size');
  assert(b.get(0, 0) === null, 'empty cell');
  assert(b.get(4, 4) === null, 'center empty');
}

section('Board get/set');
{
  const b = new Board(9);
  b.set(3, 3, 'black');
  assert(b.get(3, 3) === 'black', 'set then get black');
  b.set(3, 3, 'white');
  assert(b.get(3, 3) === 'white', 'overwrite with white');
  b.set(3, 3, null);
  assert(b.get(3, 3) === null, 'clear to null');
}

section('Toroidal neighbors');
{
  const b = new Board(9);
  // Corner (0,0) wraps to (8,0), (0,8), (1,0), (0,1)
  const n = b.getNeighbors(0, 0);
  assert(n.length === 4, 'always 4 neighbors');
  const set = new Set(n.map(([x, y]) => `${x},${y}`));
  assert(set.has('8,0'), 'left wraps');
  assert(set.has('0,8'), 'top wraps');
  assert(set.has('1,0'), 'right');
  assert(set.has('0,1'), 'down');
}

// ─── Group tracking ──────────────────────────────────────────────────────────

section('Groups and liberties');
{
  const b = new Board(9);
  b.set(4, 4, 'black');
  b.captureGroups(4, 4); // triggers tracking
  const g = b.getGroup(4, 4);
  assert(g.length === 1, 'single stone group');
  const libs = b.getLiberties(g);
  assert(libs.size === 4, 'single stone has 4 liberties');
  assert(!b.hasNoLiberties(g), 'has liberties');
}

{
  const b = new Board(9);
  // Place two adjacent black stones
  b.set(4, 4, 'black');
  b.captureGroups(4, 4);
  b.set(5, 4, 'black');
  b.captureGroups(5, 4);
  const g = b.getGroup(4, 4);
  assert(g.length === 2, 'two-stone group');
  const libs = b.getLiberties(g);
  assert(libs.size === 6, 'two-stone group has 6 liberties');
}

// ─── Capture ─────────────────────────────────────────────────────────────────

section('Simple capture');
{
  const g = new Game(9, 3.5);
  // g starts with black at center (4,4), white to play
  // Surround a white stone and capture it
  const b = g.board;

  // Place white at (0,0)
  // Need to set up a capture scenario manually
  const g2 = new Game(9, 3.5);
  // After constructor: black at (4,4), current = white
  // White plays somewhere
  g2.placeStone(2, 2); // white at (2,2), current = black
  // Surround white stone at (2,2)
  g2.placeStone(3, 2); // black, current = white
  g2.pass();            // white passes, current = black
  g2.placeStone(1, 2); // black, current = white
  g2.pass();            // white passes, current = black
  g2.placeStone(2, 3); // black, current = white
  g2.pass();            // white passes, current = black
  // Last liberty is (2,1)
  g2.placeStone(2, 1); // black captures white at (2,2)
  assert(g2.board.get(2, 2) === null, 'captured stone removed');
}

// ─── Suicide ─────────────────────────────────────────────────────────────────

section('Suicide prevention');
{
  const b = new Board(9);
  // Surround (2,2) with opponent stones
  b.set(3, 2, 'black');
  b.captureGroups(3, 2);
  b.set(1, 2, 'black');
  b.captureGroups(1, 2);
  b.set(2, 3, 'black');
  b.captureGroups(2, 3);
  b.set(2, 1, 'black');
  b.captureGroups(2, 1);
  assert(b.isSingleSuicide(2, 2, 'white'), 'single suicide detected');
  assert(b.isSuicide(2, 2, 'white'), 'isSuicide wrapper');
  assert(!b.isSuicide(2, 2, 'black'), 'not suicide for same color (connects)');
}

// ─── Ko ──────────────────────────────────────────────────────────────────────

section('Ko detection');
{
  const b = new Board(9);
  // Ko shape: black captures one white stone, white cannot recapture immediately
  // Set up:  . B .
  //          B . B   (center is the ko point)
  //          . W .
  // But we need a more complete setup. Use isKo directly.
  b.set(3, 2, 'black');
  b.captureGroups(3, 2);
  b.set(2, 3, 'black');
  b.captureGroups(2, 3);
  b.set(4, 3, 'black');
  b.captureGroups(4, 3);
  b.set(3, 4, 'white');
  b.captureGroups(3, 4);
  b.set(2, 4, 'white');
  // White at (2,4) needs tracking
  b.captureGroups(2, 4);
  // koFlag points at (3,3) — as if white just captured there
  const koFlag = { x: 3, y: 3 };
  // If black recaptures at (3,3) and would capture exactly 1 stone, it's Ko
  b.set(3, 3, 'white'); // put a stone to be "recaptured"
  b.captureGroups(3, 3);
  // Check isKo: black at (3,3) with koFlag at (3,3)
  // Actually this needs a proper single-stone capture setup. Let's just test the flag logic.
  assert(!b.isKo(0, 0, 'black', koFlag), 'not ko at different position');
  assert(!b.isKo(3, 3, 'black', null), 'not ko with null flag');
}

// ─── True eye detection ──────────────────────────────────────────────────────

section('True eye detection');
{
  const b = new Board(9);
  // Create a true eye: all 4 ortho neighbors are the same color
  b.set(3, 2, 'black');
  b.captureGroups(3, 2);
  b.set(2, 3, 'black');
  b.captureGroups(2, 3);
  b.set(4, 3, 'black');
  b.captureGroups(4, 3);
  b.set(3, 4, 'black');
  b.captureGroups(3, 4);
  // Need diagonals for true eye (≥3)
  b.set(2, 2, 'black');
  b.captureGroups(2, 2);
  b.set(4, 2, 'black');
  b.captureGroups(4, 2);
  b.set(2, 4, 'black');
  b.captureGroups(2, 4);
  assert(b.isTrueEye(3, 3, 'black'), 'true eye with 3 diagonals');
  assert(!b.isTrueEye(3, 3, 'white'), 'not true eye for other color');
}

// ─── classifyEmpty ───────────────────────────────────────────────────────────

section('classifyEmpty');
{
  const b = new Board(9);
  // Empty board: every cell has empty neighbors, no eyes
  const info = b.classifyEmpty(4, 4, 'black');
  assert(!info.isTrueEye, 'empty board: not true eye');
  assert(info.hasEmptyNeighbor, 'empty board: has empty neighbor');
}
{
  // Reuse true eye setup
  const b = new Board(9);
  b.set(3, 2, 'black'); b.captureGroups(3, 2);
  b.set(2, 3, 'black'); b.captureGroups(2, 3);
  b.set(4, 3, 'black'); b.captureGroups(4, 3);
  b.set(3, 4, 'black'); b.captureGroups(3, 4);
  b.set(2, 2, 'black'); b.captureGroups(2, 2);
  b.set(4, 2, 'black'); b.captureGroups(4, 2);
  b.set(2, 4, 'black'); b.captureGroups(2, 4);
  const info = b.classifyEmpty(3, 3, 'black');
  assert(info.isTrueEye, 'classifyEmpty agrees with isTrueEye');
  assert(!info.hasEmptyNeighbor, 'no empty neighbor when surrounded');
}

// ─── Game construction ───────────────────────────────────────────────────────

section('Game construction');
{
  const g = new Game(9, 3.5);
  assert(g.boardSize === 9, 'boardSize');
  assert(g.komi === 3.5, 'komi');
  assert(g.current === 'white', 'white to play after center stone');
  assert(g.board.get(4, 4) === 'black', 'center stone placed');
  assert(!g.gameOver, 'game not over');
  assert(g.moveCount === 1, 'one move played');
}

section('Game with different sizes');
{
  const g7 = new Game(7, 3.5);
  assert(g7.boardSize === 7, 'size 7');
  assert(g7.board.get(3, 3) === 'black', 'center stone on 7x7');

  const g13 = new Game(13, 3.5);
  assert(g13.boardSize === 13, 'size 13');
  assert(g13.board.get(6, 6) === 'black', 'center stone on 13x13');
}

// ─── Game moves ──────────────────────────────────────────────────────────────

section('Game placeStone');
{
  const g = new Game(9, 3.5);
  // White's turn
  const result = g.placeStone(0, 0);
  assert(result !== false, 'legal move returns truthy');
  assert(g.board.get(0, 0) === 'white', 'stone placed');
  assert(g.current === 'black', 'turn switches');
  assert(g.moveCount === 2, 'move count incremented');
}

section('Illegal moves');
{
  const g = new Game(9, 3.5);
  // Try to play on occupied cell
  const result = g.placeStone(4, 4);
  assert(result === false, 'cannot play on occupied cell');
  assert(g.current === 'white', 'turn unchanged after illegal move');
}

section('Pass');
{
  const g = new Game(9, 3.5);
  const passer = g.pass();
  assert(passer === 'white', 'pass returns the passer');
  assert(g.current === 'black', 'turn switches after pass');
  assert(g.consecutivePasses === 1, 'one consecutive pass');
  assert(g.lastMove === null, 'lastMove null after pass');
  assert(g.koFlag === null, 'koFlag cleared after pass');
}

section('Double pass ends game');
{
  const g = new Game(9, 3.5);
  g.pass();
  g.pass();
  assert(g.gameOver, 'game over after two passes');
  assert(g.scores !== null, 'scores populated');
  assert(typeof g.scores.black.total === 'number', 'black score is number');
  assert(typeof g.scores.white.total === 'number', 'white score is number');
  assert(g.scores.white.total >= g.komi, 'white score includes komi');
}

// ─── Clone ───────────────────────────────────────────────────────────────────

section('Game clone');
{
  const g = new Game(9, 3.5);
  g.placeStone(0, 0);
  const c = g.clone();
  assert(c.boardSize === g.boardSize, 'clone boardSize');
  assert(c.current === g.current, 'clone current');
  assert(c.hash === g.hash, 'clone hash');
  assert(c.moveCount === g.moveCount, 'clone moveCount');
  // Mutations on clone don't affect original
  c.placeStone(1, 1);
  assert(g.board.get(1, 1) === null, 'clone is independent');
  assert(c.board.get(1, 1) !== null, 'clone has the stone');
}

section('Clone divergence (independent futures)');
{
  const random = require('./ai/random.js');
  let ok = true;

  for (let trial = 0; trial < 10; trial++) {
    const g = new Game(7, 3.5);
    for (let i = 0; i < 5 && !g.gameOver; i++) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }
    if (g.gameOver) continue;

    const c = g.clone();
    for (let i = 0; i < 10 && !g.gameOver; i++) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }

    try {
      const move = random(c);
      if (move.type === 'place') c.placeStone(move.x, move.y);
      else c.pass();
    } catch (e) {
      ok = false;
      console.error(`  Clone became corrupt after original diverged:`, e.message);
    }
  }
  assert(ok, 'clones remain playable after original diverges');
}

section('Board clone');
{
  const b = new Board(9);
  b.set(5, 5, 'black');
  b.captureGroups(5, 5);
  const c = b.clone();
  assert(c.get(5, 5) === 'black', 'cloned board has stone');
  c.set(5, 5, null);
  assert(b.get(5, 5) === 'black', 'original unaffected');
}

// ─── Zobrist hashing ─────────────────────────────────────────────────────────

section('Zobrist hash');
{
  assert(typeof ZOBRIST[0][0].black === 'bigint', 'zobrist values are bigint');
  assert(ZOBRIST[0][0].black !== ZOBRIST[0][0].white, 'black/white hashes differ');
  assert(ZOBRIST[0][0].black !== ZOBRIST[1][0].black, 'different cells differ');

  // Hash should change when a stone is placed
  const g = new Game(9, 3.5);
  const h1 = g.hash;
  g.placeStone(0, 0);
  assert(g.hash !== h1, 'hash changes after move');
}

// ─── Territory ───────────────────────────────────────────────────────────────

section('Territory calculation');
{
  const g = new Game(7, 0);
  // End immediately — only center stone
  g.pass();
  g.pass();
  assert(g.gameOver, 'game ended');
  // All territory should be black (one stone on board, all connected empty is black territory)
  // On a toroidal board, the single stone's territory = all empty cells
  assert(g.scores.black.total > 0, 'black has territory');
}

section('Territory scoring makes sense');
{
  const random = require('./ai/random.js');
  let ok = true;

  for (let i = 0; i < 10; i++) {
    const g = new Game(7, 3.5);
    while (!g.gameOver) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }
    const s = g.scores;
    if (s.black.territory < 0 || s.white.territory < 0) {
      ok = false;
      console.error(`  Negative territory in game ${i}`);
    }
    const territory = g.calcTerritory();
    const accounted = territory.black + territory.white + territory.neutral;
    if (accounted !== 49) {
      ok = false;
      console.error(`  Territory doesn't sum to 49: got ${accounted}`);
    }
  }
  assert(ok, 'territory scores are valid across 10 games');
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

section('Komi in scoring');
{
  // On an empty-ish board with only center stone, white gets komi
  const g = new Game(7, 3.5);
  g.pass();
  g.pass();
  assert(g.scores.white.total === g.scores.white.territory + 3.5, 'komi added to white');
}

section('DEFAULT_KOMI');
{
  assert(DEFAULT_KOMI === 4.5, 'default komi is 4.5');
}

// ─── statusText ──────────────────────────────────────────────────────────────

section('statusText');
{
  const g = new Game(9, 3.5);
  assert(g.statusText() === 'White to play', 'white to play after init');
  g.placeStone(0, 0);
  assert(g.statusText() === 'Black to play', 'black to play');
  g.pass();
  g.pass();
  assert(g.statusText() === 'Game over', 'game over text');
}

// ─── Move count limit ────────────────────────────────────────────────────────

section('Move count auto-end');
{
  const g = new Game(5, 0);
  // Threshold = 4 * 25 = 100 moves. Play passes until it ends.
  let moves = 0;
  while (!g.gameOver && moves < 200) {
    g.pass();
    moves++;
  }
  assert(g.gameOver, 'game auto-ended by move count or passes');
}

// ─── Random agent ────────────────────────────────────────────────────────────

section('Random vs random roughly even (5x5, 20 games)');
{
  const p1 = require('./ai/random.js');
  const p2 = require('./ai/random.js');
  let p1Wins = 0, p2Wins = 0;

  for (let i = 0; i < 20; i++) {
    const g = new Game(5, 3.5);
    const blackAgent = i % 2 === 0 ? p1 : p2;
    const whiteAgent = i % 2 === 0 ? p2 : p1;
    const p1IsBlack = i % 2 === 0;

    while (!g.gameOver) {
      const agent = g.current === 'black' ? blackAgent : whiteAgent;
      const move = agent(g, 0);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }

    const s = g.scores;
    const blackWins = s.black.total > s.white.total;
    if (p1IsBlack ? blackWins : !blackWins) p1Wins++;
    else p2Wins++;
  }
  console.log(`  p1: ${p1Wins}  p2: ${p2Wins}`);
  assert(p1Wins >= 3 && p2Wins >= 3,
    `random vs random should be roughly balanced: ${p1Wins}-${p2Wins}`);
}

section('Random agent');
{
  const randomAgent = require('./ai/random.js');
  const g = new Game(7, 3.5);
  const move = randomAgent(g);
  assert(move.type === 'place' || move.type === 'pass', 'random returns valid move type');
  if (move.type === 'place') {
    assert(typeof move.x === 'number' && typeof move.y === 'number', 'place has coords');
    assert(move.x >= 0 && move.x < 7, 'x in bounds');
    assert(move.y >= 0 && move.y < 7, 'y in bounds');
  }

  // Ended game should return pass
  const g2 = new Game(7, 3.5);
  g2.pass(); g2.pass();
  assert(randomAgent(g2).type === 'pass', 'pass on ended game');
}

// ─── MC agent basic ──────────────────────────────────────────────────────────

section('MC agent');
{
  const mc = require('./ai/mc.js');
  const g = new Game(7, 3.5);
  const move = mc(g, 50); // very short budget
  assert(move.type === 'place' || move.type === 'pass', 'mc returns valid move');
  if (move.type === 'place') {
    // Verify the move is legal
    const clone = g.clone();
    assert(clone.placeStone(move.x, move.y) !== false, 'mc move is legal');
  }
}

// ─── MCTS agent basic ───────────────────────────────────────────────────────

section('MCTS agent');
{
  const mcts = require('./ai/mcts.js');
  const g = new Game(7, 3.5);
  const move = mcts(g, 50);
  assert(move.type === 'place' || move.type === 'pass', 'mcts returns valid move');
  if (move.type === 'place') {
    const clone = g.clone();
    assert(clone.placeStone(move.x, move.y) !== false, 'mcts move is legal');
  }
}

// ─── AMAF agent basic ───────────────────────────────────────────────────────

section('AMAF agent');
{
  const amaf = require('./ai/amaf.js');
  const g = new Game(7, 3.5);
  const move = amaf(g, 50);
  assert(move.type === 'place' || move.type === 'pass', 'amaf returns valid move');
  if (move.type === 'place') {
    const clone = g.clone();
    assert(clone.placeStone(move.x, move.y) !== false, 'amaf move is legal');
  }
}

// ─── Group verification mode ─────────────────────────────────────────────────

section('Group tracker verification');
{
  const oldRatio = Board.verifyGroupRatio;
  Board.verifyGroupRatio = 1; // verify every captureGroups call
  const g = new Game(7, 3.5);
  // Play a sequence of moves — if tracker is wrong, verification throws
  let ok = true;
  try {
    for (let i = 0; i < 20; i++) {
      const random = require('./ai/random.js');
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
      if (g.gameOver) break;
    }
  } catch (e) {
    ok = false;
    console.error('  Group verify error:', e.message);
  }
  assert(ok, 'group tracker consistent through random game');
  Board.verifyGroupRatio = oldRatio;
}

// ─── classifyEmpty consistency with isTrueEye ───────────────────────────────

section('classifyEmpty matches isTrueEye on random board');
{
  const g = new Game(7, 3.5);
  const random = require('./ai/random.js');
  // Play some random moves
  for (let i = 0; i < 15 && !g.gameOver; i++) {
    const move = random(g);
    if (move.type === 'place') g.placeStone(move.x, move.y);
    else g.pass();
  }
  // Check every empty cell
  let allMatch = true;
  for (let y = 0; y < 7; y++) {
    for (let x = 0; x < 7; x++) {
      if (g.board.get(x, y) !== null) continue;
      for (const color of ['black', 'white']) {
        const eye = g.board.isTrueEye(x, y, color);
        const info = g.board.classifyEmpty(x, y, color);
        if (eye !== info.isTrueEye) {
          allMatch = false;
          console.error(`  Mismatch at (${x},${y}) for ${color}: isTrueEye=${eye}, classifyEmpty=${info.isTrueEye}`);
        }
        // Also verify hasEmptyNeighbor
        const neighbors = g.board.getNeighbors(x, y);
        const hasEmpty = neighbors.some(([nx, ny]) => g.board.get(nx, ny) === null);
        if (hasEmpty !== info.hasEmptyNeighbor) {
          allMatch = false;
          console.error(`  hasEmptyNeighbor mismatch at (${x},${y})`);
        }
      }
    }
  }
  assert(allMatch, 'classifyEmpty consistent with isTrueEye + getNeighbors');
}

// ─── Hash consistency ────────────────────────────────────────────────────────

section('Hash consistency across clones');
{
  const g = new Game(7, 3.5);
  g.placeStone(0, 0);
  g.placeStone(1, 0);
  const c = g.clone();
  assert(c.hash === g.hash, 'cloned hash matches');
  g.placeStone(2, 0);
  c.placeStone(2, 0);
  assert(c.hash === g.hash, 'hash matches after same move on clone');
}

section('Zobrist hash uniqueness (no collisions in random games)');
{
  const random = require('./ai/random.js');
  let collisions = 0;

  for (let trial = 0; trial < 5; trial++) {
    const g = new Game(7, 3.5);
    const seen = new Set();
    seen.add(g.hash);

    while (!g.gameOver) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
      if (!g.gameOver && g.lastMove !== null) {
        // Only check on stone placements (passes don't change hash)
        if (seen.has(g.hash)) collisions++;
        seen.add(g.hash);
      }
    }
  }
  console.log(`  Hash collisions: ${collisions} across 5 games`);
  assert(collisions <= 2, `should have very few hash collisions: got ${collisions}`);
}

// ─── placeStone return value ─────────────────────────────────────────────────

section('placeStone return values');
{
  const g = new Game(9, 3.5);
  const r = g.placeStone(0, 0);
  assert(r === true, 'no capture returns true (not a number)');
  const r2 = g.placeStone(4, 4);
  assert(r2 === false, 'occupied cell returns false');
}

// ─── Board serialize/parse round-trip ────────────────────────────────────────

section('Board serialize/parse round-trip');
{
  const { parseBoard, boardTurnToString } = require('./game.js');
  const g = new Game(7, 3.5);
  const random = require('./ai/random.js');
  for (let i = 0; i < 10 && !g.gameOver; i++) {
    const move = random(g);
    if (move.type === 'place') g.placeStone(move.x, move.y);
    else g.pass();
  }
  const str = boardTurnToString(g.board);
  const { size, stones } = parseBoard(str);
  assert(size === 7, 'parsed size matches');
  const b2 = new Board(size);
  for (const [x, y, color] of stones) b2.set(x, y, color);
  let match = true;
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (g.board.get(x, y) !== b2.get(x, y)) match = false;
  assert(match, 'round-trip preserves all cells');
}

// ─── Pattern symmetry ────────────────────────────────────────────────────────

// Helpers shared by all pattern-symmetry sections.
const { patternHash, MAX_LIBS } = require('./patterns.js');

// D4 symmetry permutations — must match SYMMETRY_PERMS in patterns.JS.
const _D4_PERMS = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8], // identity
  [6, 3, 0, 7, 4, 1, 8, 5, 2], // rotate 90° CW
  [8, 7, 6, 5, 4, 3, 2, 1, 0], // rotate 180°
  [2, 5, 8, 1, 4, 7, 0, 3, 6], // rotate 270° CW
  [2, 1, 0, 5, 4, 3, 8, 7, 6], // reflect horizontal
  [6, 7, 8, 3, 4, 5, 0, 1, 2], // reflect vertical
  [0, 3, 6, 1, 4, 7, 2, 5, 8], // reflect main diagonal
  [8, 5, 2, 7, 4, 1, 6, 3, 0], // reflect anti-diagonal
];

// Apply D4 symmetry `sym` to a 3×3-grid offset (dx, dy) in {-1,0,1}^2.
// Returns the [dx, dy] that destination position maps to in the transformed grid.
function applyD4(sym, dx, dy) {
  const perm = _D4_PERMS[sym];
  const src  = (dy + 1) * 3 + (dx + 1);
  const dest = perm.indexOf(src);
  return [dest % 3 - 1, Math.floor(dest / 3) - 1];
}

// Build a 9×9 game, place stones relative to (cx, cy), return patternHash.
// Center (1,1) is used so it stays clear of Game's initial stone at (4,4).
function buildAndHash(stones, cx, cy, mover) {
  const g = new Game(9, 3.5);
  for (const { dx, dy, color } of stones) {
    g.board.set(cx + dx, cy + dy, color);
    g.board.captureGroups(cx + dx, cy + dy);
  }
  return patternHash(g, cx, cy, mover);
}

section('patternHash symmetry – diagonal stones');
{
  // Two diagonal stones. LIB_WEIGHT is 0 for diagonal positions so this
  // exercises the cellHash component in isolation.
  const base = [
    { dx: -1, dy: -1, color: 'black' },
    { dx:  1, dy: -1, color: 'white' },
  ];
  const hashes = _D4_PERMS.map((_, sym) =>
    buildAndHash(
      base.map(s => { const [dx, dy] = applyD4(sym, s.dx, s.dy); return { dx, dy, color: s.color }; }),
      1, 1, 'black'
    )
  );
  assert(hashes.every(h => h === hashes[0]),
    `all 8 D4 transforms yield the same hash (diagonal): [${hashes}]`);
}

section('patternHash symmetry – orthogonal stone');
{
  // Single orthogonal stone exercises the liberty-count (libHash) component.
  const base = [{ dx: 0, dy: -1, color: 'black' }];
  const hashes = _D4_PERMS.map((_, sym) =>
    buildAndHash(
      base.map(s => { const [dx, dy] = applyD4(sym, s.dx, s.dy); return { dx, dy, color: s.color }; }),
      1, 1, 'black'
    )
  );
  assert(hashes.every(h => h === hashes[0]),
    `all 8 D4 transforms yield the same hash (orthogonal): [${hashes}]`);
}

section('patternHash symmetry – mixed pattern');
{
  // One orthogonal stone + one diagonal stone: exercises both hash components.
  const base = [
    { dx:  0, dy: -1, color: 'black' }, // orthogonal (contributes to libHash)
    { dx: -1, dy: -1, color: 'white' }, // diagonal   (cellHash only)
  ];
  const hashes = _D4_PERMS.map((_, sym) =>
    buildAndHash(
      base.map(s => { const [dx, dy] = applyD4(sym, s.dx, s.dy); return { dx, dy, color: s.color }; }),
      1, 1, 'black'
    )
  );
  assert(hashes.every(h => h === hashes[0]),
    `all 8 D4 transforms yield the same hash (mixed): [${hashes}]`);
}

section('patternHash distinguishes non-equivalent patterns');
{
  // Diagonal vs orthogonal position are not D4-equivalent — must hash differently.
  const hDiag = buildAndHash([{ dx: -1, dy: -1, color: 'black' }], 1, 1, 'black');
  const hOrth = buildAndHash([{ dx:  0, dy: -1, color: 'black' }], 1, 1, 'black');
  assert(hDiag !== hOrth, 'diagonal stone ≠ orthogonal stone');

  // Friend vs enemy at the same position must hash differently.
  const hFriend = buildAndHash([{ dx: -1, dy: -1, color: 'black' }], 1, 1, 'black');
  const hEnemy  = buildAndHash([{ dx: -1, dy: -1, color: 'white' }], 1, 1, 'black');
  assert(hFriend !== hEnemy, 'friend stone ≠ enemy stone at same position');
}

section('patternHash mover-relative encoding');
{
  // Same physical board should hash differently for different movers because
  // cell codes are relative to the mover (friend vs enemy swap).
  const g = new Game(9, 3.5);
  g.board.set(0, 0, 'black');
  g.board.captureGroups(0, 0);
  const hBlack = patternHash(g, 1, 1, 'black');
  const hWhite = patternHash(g, 1, 1, 'white');
  assert(hBlack !== hWhite, 'different mover ⇒ different hash for same board');
}

section('patternHash return value is non-negative and bounded');
{
  const g = new Game(9, 3.5);
  g.board.set(0, 1, 'black'); g.board.captureGroups(0, 1);
  g.board.set(2, 1, 'white'); g.board.captureGroups(2, 1);
  const h = patternHash(g, 1, 1, 'black');
  const maxHash = (3 ** 9 - 1) + 19683 * ((MAX_LIBS + 1) ** 4 - 1);
  assert(h >= 0,       `hash is non-negative (got ${h})`);
  assert(h <= maxHash, `hash is within bounds (got ${h}, max ${maxHash})`);
}

section('patternHash determinism');
{
  const g = new Game(9, 3.5);
  g.board.set(2, 1, 'white'); g.board.captureGroups(2, 1);
  const h1 = patternHash(g, 1, 1, 'black');
  const h2 = patternHash(g, 1, 1, 'black');
  assert(h1 === h2, 'patternHash returns the same value on repeated calls');
}

// ─── Ladder reader ───────────────────────────────────────────────────────────

const { isLadderCaptured, getLadderStatus } = require('./ladder.js');

section('Ladder reader – not in atari');
{
  // Empty cell → false
  {
    const g = new Game(9, 3.5);
    assert(!isLadderCaptured(g, 4, 4).captured, 'empty cell: false');
  }

  // Group with 4 liberties → false
  {
    const g = new Game(9, 3.5);
    g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
    assert(!isLadderCaptured(g, 4, 4).captured, '4-liberty group: false');
  }

  // Group with 2 liberties → false
  {
    const g = new Game(9, 3.5);
    g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
    g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
    g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
    // Black at (4,4): liberties (4,3) and (4,5)
    assert(!isLadderCaptured(g, 4, 4).captured, '2-liberty group: false');
  }
}

section('Ladder reader – atari with immediate escape to 3+ liberties');
{
  // Black at (4,4), white at (3,4),(5,4),(4,3) → liberty (4,5).
  // After black plays (4,5): group has (4,5)'s three free neighbors → 3 libs.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  assert(!isLadderCaptured(g, 4, 4).captured, 'immediate escape to 3+ libs: false');
}

section('Ladder reader – escape is suicide (all exits blocked)');
{
  // Black at (4,4), white completely encaging except one entry point (4,5),
  // but that entry is also a dead end: white at (3,5),(5,5),(4,6).
  // Playing (4,5) would give {(4,4),(4,5)} zero liberties → suicide → false.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(3, 5, 'white'); g.board.captureGroups(3, 5);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(4, 6, 'white'); g.board.captureGroups(4, 6);
  // Liberty is (4,5); playing there is suicide
  assert(isLadderCaptured(g, 4, 4).captured, 'escape is suicide: caught');
}

section('Ladder reader – ladder with breaker stone');
{
  // Black at (4,4), white at (3,4),(5,4),(4,3),(5,5) → liberty (4,5).
  // After black (4,5): group has libs (3,5) and (4,6).
  // Black stone at (3,6) acts as a ladder breaker: when the group reaches
  // either (3,5) or (4,6), it merges with (3,6) and gains 3+ libs.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 6, 'black'); g.board.captureGroups(3, 6);  // breaker
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  assert(!isLadderCaptured(g, 4, 4).captured, 'ladder breaker present: not caught');
}

section('Ladder reader – same position without breaker is caught');
{
  // Identical to the breaker test but without the breaker stone.
  // The ladder has nowhere to escape and eventually runs into the depth
  // limit on this toroidal board — or the attacker always re-ataris.
  // To guarantee termination, add white stones that block the only exit.
  //
  // Black at (4,4), white at (3,4),(5,4),(4,3),(5,5),(4,6),(3,5).
  // Liberty: (4,5). After black (4,5): libs (3,5) and (4,6) are both white
  // → group immediately has 0 libs after escape? No: (3,5) and (4,6) are the
  // libs, but they are empty cells. White is already AT (3,5) and (4,6) here.
  // So the group after escape has 0 liberties → but wait that means the
  // escape move itself results in 0 libs → it's different from suicide
  // because the placeStone call still succeeds (white (3,5) and (4,6) are
  // pre-placed; the escape point (4,5) itself is empty).
  // Let's trace: black plays (4,5); group {(4,4),(4,5)}.
  //   (4,4): (3,4)=W,(5,4)=W,(4,3)=W — no liberties
  //   (4,5): (3,5)=W,(5,5)=W,(4,6)=W,(4,4)=B — no liberties
  // → newLibs.size = 0 → return false → isLadderCaptured = true.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(3, 5, 'white'); g.board.captureGroups(3, 5);
  g.board.set(4, 6, 'white'); g.board.captureGroups(4, 6);
  assert(isLadderCaptured(g, 4, 4).captured, 'no breaker, all exits blocked: caught');
}

section('Ladder reader – two-step ladder terminates in capture');
{
  // Build a position where black escapes to 2 libs, attacker re-ataris,
  // black escapes again to 2 libs which are both already occupied → capture.
  //
  // Step 0: Black at (4,4), liberty (4,5).
  //         White: (3,4),(5,4),(4,3) [initial cage]
  // Step 1: Black plays (4,5). Libs: (3,5),(4,6) [because (5,5)=W closes off].
  //         White: add (5,5).
  // Step 2: White plays (3,5) [re-atari]. Lib: (4,6).
  // Step 3: Black plays (4,6). Libs from {(4,4),(4,5),(4,6)}:
  //           (4,5) contrib: (5,5)=W → nothing new
  //           (4,6) contrib: (5,6),(4,7) [if not blocked]
  //         We need (5,6) and (4,7) both blocked to finish the cage.
  //         Add white at (5,6) and (4,7).
  // After black plays (4,6): libs (5,6)=W,(4,7)=W → 0 libs! → caught.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);  // closes (5,5)
  g.board.set(5, 6, 'white'); g.board.captureGroups(5, 6);  // close exit
  g.board.set(4, 7, 'white'); g.board.captureGroups(4, 7);  // close exit
  // After step 1 (black plays 4,5): libs = {(3,5),(4,6)} — attacker re-ataris via (3,5).
  // After step 2 (white plays 3,5): lib = {(4,6)}.
  // After step 3 (black plays 4,6): libs from (4,6) = {(5,6)=W,(4,7)=W} → 0 → caught.
  assert(isLadderCaptured(g, 4, 4).captured, 'two-step ladder with blocked exits: caught');
}

section('Ladder reader – moves: attacker turn, immediate capture');
{
  // Same board as "escape is suicide": black at (4,4), all exits blocked.
  // With game.current = 'white' (attacker), moves should be the single liberty.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(3, 5, 'white'); g.board.captureGroups(3, 5);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(4, 6, 'white'); g.board.captureGroups(4, 6);
  g.current = 'white';
  const r = isLadderCaptured(g, 4, 4);
  assert(r.captured, 'attacker turn immediate: captured');
  assert(r.moves.length === 1, 'attacker turn immediate: one attack move');
  assert(r.moves[0].x === 4 && r.moves[0].y === 5, 'attacker turn immediate: attack move is (4,5)');
}

section('Ladder reader – moves: attacker turn, two-step ladder');
{
  // Same board as "two-step ladder".  With game.current = 'white' (attacker),
  // the one valid re-atari after black escapes to (4,5) is (3,5).
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(5, 6, 'white'); g.board.captureGroups(5, 6);
  g.board.set(4, 7, 'white'); g.board.captureGroups(4, 7);
  g.current = 'white';
  const r = isLadderCaptured(g, 4, 4);
  assert(r.captured, 'attacker turn two-step: captured');
  // Both liberties of the escaped group are valid re-atari points on this toroidal board.
  assert(r.moves.length >= 1, 'attacker turn two-step: at least one re-atari');
  assert(r.moves.some(m => m.x === 3 && m.y === 5), 'attacker turn two-step: (3,5) is a valid re-atari');
}

section('Ladder reader – moves: defender turn, can escape');
{
  // Black at (4,4) with one liberty (4,5); escape leads to 3+ libs.
  // With game.current = 'black' (defender), moves should be the escape point.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.current = 'black';  // set defender as the active player
  const r = isLadderCaptured(g, 4, 4);
  assert(!r.captured, 'defender turn escape: not captured');
  assert(r.moves.length === 1, 'defender turn escape: one escape move');
  assert(r.moves[0].x === 4 && r.moves[0].y === 5, 'defender turn escape: escape move is (4,5)');
}

section('Ladder reader – moves: defender turn, captured (no moves for defender)');
{
  // Black caught in two-step ladder; game.current = 'black' (defender's turn).
  // Captured groups have no useful moves to return.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(5, 6, 'white'); g.board.captureGroups(5, 6);
  g.board.set(4, 7, 'white'); g.board.captureGroups(4, 7);
  g.current = 'black';  // defender's turn — no attack moves should be returned
  const r = isLadderCaptured(g, 4, 4);
  assert(r.captured, 'defender turn captured: captured');
  assert(r.moves.length === 0, 'defender turn captured: no moves returned');
}

// ─── getLadderStatus ─────────────────────────────────────────────────────────

section('getLadderStatus – no stone / too many liberties');
{
  // No stone at (0,0) → empty array (constructor places stone at center (4,4))
  {
    const g = new Game(9, 3.5);
    const r = getLadderStatus(g, 0, 0);
    assert(Array.isArray(r) && r.length === 0, 'no stone: empty array');
  }

  // Group with 4 liberties → null (warns)
  // new Game places black at center (4,4) with 4 liberties; current flips to white.
  {
    const g = new Game(9, 3.5);
    const r = getLadderStatus(g, 4, 4);
    assert(r === null, '4-liberty group: null');
  }
}

section('getLadderStatus – 1-liberty group: immediate escape to 3+ libs');
{
  // Black at (4,4), white at (3,4),(5,4),(4,3): liberty (4,5).
  // After black plays (4,5): group gains (3,5),(4,6),(5,5) → 3 libs → canEscape=true.
  // After white plays (4,5): black captured → canEscapeAfterPass=false.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.current = 'black';
  const r = getLadderStatus(g, 4, 4);
  assert(Array.isArray(r) && r.length === 1, 'immediate escape: one entry');
  assert(r[0].liberty.x === 4 && r[0].liberty.y === 5, 'liberty is (4,5)');
  assert(r[0].canEscape === true, 'canEscape: black plays → 3+ libs → true');
  assert(r[0].canEscapeAfterPass === false, 'canEscapeAfterPass: white captures → false');
}

section('getLadderStatus – 1-liberty group: escape is suicide');
{
  // Same board as isLadderCaptured "escape is suicide".
  // Playing the liberty is suicide → canEscape=false.
  // White playing the liberty captures black → canEscapeAfterPass=false.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(3, 5, 'white'); g.board.captureGroups(3, 5);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(4, 6, 'white'); g.board.captureGroups(4, 6);
  g.current = 'black';
  const r = getLadderStatus(g, 4, 4);
  assert(Array.isArray(r) && r.length === 1, 'suicide: one entry');
  assert(r[0].canEscape === false, 'canEscape: suicide → false');
  assert(r[0].canEscapeAfterPass === false, 'canEscapeAfterPass: white captures → false');
}

section('getLadderStatus – 1-liberty group: ladder with breaker (can escape)');
{
  // Same board as isLadderCaptured "ladder with breaker": black can escape.
  // canEscape should be true.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 6, 'black'); g.board.captureGroups(3, 6);  // breaker
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.current = 'black';
  assert(!isLadderCaptured(g, 4, 4).captured, 'sanity: isLadderCaptured agrees group can escape');
  const r = getLadderStatus(g, 4, 4);
  assert(Array.isArray(r) && r.length === 1, 'breaker: one entry');
  assert(r[0].canEscape === true, 'canEscape: breaker present → true');
}

section('getLadderStatus – 1-liberty group: two-step ladder (bug: canEscape wrongly true)');
{
  // Same board as isLadderCaptured "two-step ladder terminates in capture".
  // isLadderCaptured correctly returns captured=true for this position.
  //
  // Bug: after black plays (4,5) the group reaches 2 libs {(3,5),(4,6)}.
  // _canEscape returns null immediately for 2-lib groups ("not a ladder threat")
  // without recursing to check whether the attacker can re-atari and complete
  // the capture.  getLadderStatus therefore sets canEscape=true — wrong.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.board.set(4, 3, 'white'); g.board.captureGroups(4, 3);
  g.board.set(5, 5, 'white'); g.board.captureGroups(5, 5);
  g.board.set(5, 6, 'white'); g.board.captureGroups(5, 6);
  g.board.set(4, 7, 'white'); g.board.captureGroups(4, 7);
  g.current = 'black';
  assert(isLadderCaptured(g, 4, 4).captured, 'sanity: isLadderCaptured sees capture');
  const r = getLadderStatus(g, 4, 4);
  assert(Array.isArray(r) && r.length === 1, 'two-step ladder: one entry');
  // canEscape must be false: the ladder catches black despite the escape move.
  // This assertion FAILS due to the bug in getLadderStatus.
  assert(r[0].canEscape === false, 'canEscape: two-step ladder → false');
  // canEscapeAfterPass: white plays (4,5) captures black immediately → false.
  assert(r[0].canEscapeAfterPass === false, 'canEscapeAfterPass: white captures → false');
}

section('getLadderStatus – 2-liberty group');
{
  // Black at (4,4), white at (3,4) and (5,4): libs (4,3) and (4,5).
  // Mover = black (group owner).
  // For each liberty: black plays → 4+ libs → canEscape=true.
  // For each liberty: white plays → group has 1 lib, open board → canEscapeAfterPass=true.
  const g = new Game(9, 3.5);
  g.board.set(4, 4, 'black'); g.board.captureGroups(4, 4);
  g.board.set(3, 4, 'white'); g.board.captureGroups(3, 4);
  g.board.set(5, 4, 'white'); g.board.captureGroups(5, 4);
  g.current = 'black';
  const r = getLadderStatus(g, 4, 4);
  assert(Array.isArray(r) && r.length === 2, '2-lib group: two entries');
  assert(r.every(e => e.canEscape === true), 'both liberties: canEscape=true (open board)');
  assert(r.every(e => e.canEscapeAfterPass === true), 'both liberties: canEscapeAfterPass=true (open board)');
}

// ─── Results ─────────────────────────────────────────────────────────────────

console.log(`\n═══════════════════════`);
console.log(`  ${pass} passed, ${fail} failed`);
console.log(`═══════════════════════`);
process.exit(fail > 0 ? 1 : 0);
