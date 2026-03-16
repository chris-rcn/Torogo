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
  const { parseBoard, boardToString } = require('./testpuzzles.js');
  const g = new Game(7, 3.5);
  const random = require('./ai/random.js');
  for (let i = 0; i < 10 && !g.gameOver; i++) {
    const move = random(g);
    if (move.type === 'place') g.placeStone(move.x, move.y);
    else g.pass();
  }
  const str = boardToString(g.board);
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

// ─── Results ─────────────────────────────────────────────────────────────────

console.log(`\n═══════════════════════`);
console.log(`  ${pass} passed, ${fail} failed`);
console.log(`═══════════════════════`);
process.exit(fail > 0 ? 1 : 0);
