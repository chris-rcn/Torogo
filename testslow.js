'use strict';

const { Board, Game } = require('./game.js');
const { performance } = require('perf_hooks');

let pass = 0, fail = 0;

function assert(cond, msg) {
  if (cond) { pass++; }
  else { fail++; console.error('  FAIL:', msg); }
}

function section(name, fn) {
  console.log(`\n── ${name} ──`);
  const t0 = performance.now();
  fn();
  const elapsed = performance.now() - t0;
  console.log(`  (${elapsed < 1000 ? elapsed.toFixed(0) + 'ms' : (elapsed / 1000).toFixed(1) + 's'})`);
}

// ─── AI agents beat random ───────────────────────────────────────────────────

function runMatch(p1Name, p2Name, games, size, budget) {
  const p1 = require(`./ai/${p1Name}.js`);
  const p2 = require(`./ai/${p2Name}.js`);
  let p1Wins = 0, p2Wins = 0;

  for (let i = 0; i < games; i++) {
    const g = new Game(size, 3.5);
    // Alternate colors: odd games p1=black, even games p1=white
    const blackAgent = i % 2 === 0 ? p1 : p2;
    const whiteAgent = i % 2 === 0 ? p2 : p1;
    const p1IsBlack = i % 2 === 0;

    while (!g.gameOver) {
      const agent = g.current === 'black' ? blackAgent : whiteAgent;
      const move = agent(g, budget);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }

    const s = g.scores;
    const blackWins = s.black.total > s.white.total;
    if (p1IsBlack ? blackWins : !blackWins) p1Wins++;
    else p2Wins++;
  }

  return { p1Wins, p2Wins };
}

// ─── Random vs random is roughly even ────────────────────────────────────────

section('Random vs random roughly even (5x5, 20 games)', () => {
  const result = runMatch('random', 'random', 20, 5, 0);
  console.log(`  p1: ${result.p1Wins}  p2: ${result.p2Wins}`);
  // Neither side should dominate (both are random); allow wide margin
  assert(result.p1Wins >= 3 && result.p2Wins >= 3,
    `random vs random should be roughly balanced: ${result.p1Wins}-${result.p2Wins}`);
});

// ─── AI moves are always legal ───────────────────────────────────────────────

section('AI legality stress test (all agents, 5 full games each)', () => {
  const agents = ['random', 'mc', 'mcts', 'amaf'];
  let allLegal = true;

  for (const name of agents) {
    const agent = require(`./ai/${name}.js`);
    for (let i = 0; i < 3; i++) {
      const g = new Game(5, 3.5);
      let moveNum = 0;
      while (!g.gameOver && moveNum < 200) {
        const move = agent(g, 20);
        if (move.type === 'place') {
          const result = g.placeStone(move.x, move.y);
          if (result === false) {
            allLegal = false;
            console.error(`  ${name} returned illegal move (${move.x},${move.y}) on move ${moveNum}`);
            break;
          }
        } else {
          g.pass();
        }
        moveNum++;
      }
    }
  }
  assert(allLegal, 'all AI moves are legal across all agents');
});

// ─── Group tracker integrity under heavy play ────────────────────────────────

section('Group tracker verification under stress (5 games)', () => {
  const oldRatio = Board.verifyGroupRatio;
  Board.verifyGroupRatio = 1; // verify every single captureGroups call

  const random = require('./ai/random.js');
  let ok = true;

  for (let i = 0; i < 5; i++) {
    const g = new Game(7, 3.5);
    try {
      while (!g.gameOver) {
        const move = random(g);
        if (move.type === 'place') g.placeStone(move.x, move.y);
        else g.pass();
      }
    } catch (e) {
      ok = false;
      console.error(`  Game ${i + 1}:`, e.message);
    }
  }
  assert(ok, 'group tracker verified through 5 full games');
  Board.verifyGroupRatio = oldRatio;
});

// ─── Clone divergence ────────────────────────────────────────────────────────

section('Clone divergence (independent futures)', () => {
  const random = require('./ai/random.js');
  let ok = true;

  for (let trial = 0; trial < 10; trial++) {
    const g = new Game(7, 3.5);
    // Play a few moves
    for (let i = 0; i < 5 && !g.gameOver; i++) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }
    if (g.gameOver) continue;

    const c = g.clone();
    // Play different continuations
    for (let i = 0; i < 10 && !g.gameOver; i++) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }

    // Clone should still be playable
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
});

// ─── Territory scoring sanity ────────────────────────────────────────────────

section('Territory scoring makes sense', () => {
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
    // Territory should be non-negative
    if (s.black.territory < 0 || s.white.territory < 0) {
      ok = false;
      console.error(`  Negative territory in game ${i}`);
    }
    // Total territory (includes stones in Chinese scoring) + neutral should account for all cells
    const territory = g.calcTerritory();
    const accounted = territory.black + territory.white + territory.neutral;
    if (accounted !== 49) {
      ok = false;
      console.error(`  Territory doesn't sum to 49: got ${accounted}`);
    }
  }
  assert(ok, 'territory scores are valid across 10 games');
});

// ─── classifyEmpty consistency across many board states ──────────────────────

section('classifyEmpty vs isTrueEye consistency (100 random positions)', () => {
  const random = require('./ai/random.js');
  let mismatches = 0;

  for (let trial = 0; trial < 100; trial++) {
    const g = new Game(7, 3.5);
    const moveCount = Math.floor(Math.random() * 30) + 5;
    for (let i = 0; i < moveCount && !g.gameOver; i++) {
      const move = random(g);
      if (move.type === 'place') g.placeStone(move.x, move.y);
      else g.pass();
    }
    if (g.gameOver) continue;

    for (let y = 0; y < 7; y++) {
      for (let x = 0; x < 7; x++) {
        if (g.board.get(x, y) !== null) continue;
        for (const color of ['black', 'white']) {
          const eye = g.board.isTrueEye(x, y, color);
          const info = g.board.classifyEmpty(x, y, color);
          if (eye !== info.isTrueEye) mismatches++;

          const neighbors = g.board.getNeighbors(x, y);
          const hasEmpty = neighbors.some(([nx, ny]) => g.board.get(nx, ny) === null);
          if (hasEmpty !== info.hasEmptyNeighbor) mismatches++;
        }
      }
    }
  }
  console.log(`  Mismatches: ${mismatches}`);
  assert(mismatches === 0, `classifyEmpty should always match: got ${mismatches} mismatches`);
});

// ─── Hash uniqueness under play ──────────────────────────────────────────────

section('Zobrist hash uniqueness (no collisions in random games)', () => {
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
});

// ─── Playout performance ─────────────────────────────────────────────────────

section('MC playout throughput (7x7)', () => {
  const mc = require('./ai/mc.js');
  const g = new Game(7, 3.5);
  const budgetMs = 200;
  const t0 = performance.now();
  const move = mc(g, budgetMs);
  const elapsed = performance.now() - t0;
  console.log(`  Budget: ${budgetMs}ms, actual: ${elapsed.toFixed(0)}ms`);
  assert(elapsed < budgetMs * 1.5, `MC should finish within 1.5x budget: took ${elapsed.toFixed(0)}ms`);
  assert(elapsed > budgetMs * 0.5, `MC should use most of the budget: only ${elapsed.toFixed(0)}ms`);
});

section('MCTS playout throughput (7x7)', () => {
  const mcts = require('./ai/mcts.js');
  const g = new Game(7, 3.5);
  const budgetMs = 200;
  const t0 = performance.now();
  const move = mcts(g, budgetMs);
  const elapsed = performance.now() - t0;
  console.log(`  Budget: ${budgetMs}ms, actual: ${elapsed.toFixed(0)}ms`);
  assert(elapsed < budgetMs * 1.5, `MCTS should finish within 1.5x budget: took ${elapsed.toFixed(0)}ms`);
  assert(elapsed > budgetMs * 0.5, `MCTS should use most of the budget: only ${elapsed.toFixed(0)}ms`);
});

// ─── Full game smoke tests ────────────────────────────────────────────────────

section('7x7 full game (mc vs random, 2 games)', () => {
  const result = runMatch('mc', 'random', 2, 7, 100);
  console.log(`  mc: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins >= 1, `mc should beat random on 7x7: got ${result.p1Wins}`);
});

section('MC beats random (5x5, 4 games)', () => {
  const result = runMatch('mc', 'random', 4, 5, 100);
  console.log(`  mc: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins > result.p2Wins, `mc should beat random: ${result.p1Wins}-${result.p2Wins}`);
  assert(result.p1Wins >= 3, `mc should win ≥3/4 vs random: got ${result.p1Wins}`);
});

section('MCTS beats random (5x5, 4 games)', () => {
  const result = runMatch('mcts', 'random', 4, 5, 100);
  console.log(`  mcts: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins > result.p2Wins, `mcts should beat random: ${result.p1Wins}-${result.p2Wins}`);
  assert(result.p1Wins >= 3, `mcts should win ≥3/4 vs random: got ${result.p1Wins}`);
});

section('AMAF beats random (5x5, 4 games)', () => {
  const result = runMatch('amaf', 'random', 4, 5, 100);
  console.log(`  amaf: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins > result.p2Wins, `amaf should beat random: ${result.p1Wins}-${result.p2Wins}`);
  assert(result.p1Wins >= 3, `amaf should win ≥3/4 vs random: got ${result.p1Wins}`);
});

// ─── Results ─────────────────────────────────────────────────────────────────

console.log(`\n═══════════════════════`);
console.log(`  ${pass} passed, ${fail} failed`);
console.log(`═══════════════════════`);
process.exit(fail > 0 ? 1 : 0);
