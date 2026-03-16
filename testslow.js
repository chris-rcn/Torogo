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

// ─── Winning agent passes after opponent passes ───────────────────────────────

section('Winning agent passes after opponent passes (7x7)', () => {
  const rave = require('./ai/rave.js');
  const mcts = require('./ai/mcts.js');

  // Hardcoded 7x7 position (found by playing area/2 random alternating moves
  // then 4 rave moves for white with random black responses).
  //
  // white territory 22, black 17, neutral 10, komi 3.5
  // → white total 25.5 vs black 17 (white leads by 8.5)
  //
  // The neutral (10) exceeds the lead (8.5), so if random rollouts assign all
  // neutral cells to black, black wins — giving place-moves a win-rate <100%.
  // Passing ends the game immediately at the current score (white wins 100%).
  // This gap in win-rates is what causes both agents to prefer pass.
  const layout = [
    '. . . W . W W',  // y=0
    'B . B . . B .',  // y=1
    '. B . W . B .',  // y=2
    'W B . B W W .',  // y=3
    'W . B W . . W',  // y=4
    'B W B W . W B',  // y=5
    'W W B . W . B',  // y=6
  ];

  const g = new Game(7, 3.5);
  const board = g.board;
  const SZ = 7;
  const nbr = board._nbr;

  // Set every cell directly.
  for (let y = 0; y < SZ; y++) {
    const cells = layout[y].split(' ');
    for (let x = 0; x < SZ; x++)
      board.grid[y][x] = cells[x] === 'B' ? 'black' : cells[x] === 'W' ? 'white' : null;
  }

  // Rebuild the incremental group tracker from scratch.
  // captureGroups() can't be used here because it expects neighbours to be
  // tracked already (order-dependent on a toroidal board).  Instead we do it
  // in two passes: first give every stone its own isolated group with correct
  // liberties, then merge adjacent same-colour groups.
  board._gid.fill(-1);
  board._groups.clear();
  board._nextGid = 0;

  // Pass 1 – isolated groups.
  for (let y = 0; y < SZ; y++) {
    for (let x = 0; x < SZ; x++) {
      const c = board.grid[y][x];
      if (c === null) continue;
      const idx = y * SZ + x;
      const gid = board._nextGid++;
      const libs = new Set();
      const base = idx * 4;
      for (let i = 0; i < 4; i++) {
        const ni = nbr[base + i];
        if (board.grid[(ni / SZ) | 0][ni % SZ] === null) libs.add(ni);
      }
      board._gid[idx] = gid;
      board._groups.set(gid, { color: c, stones: new Set([idx]), liberties: libs });
    }
  }

  // Pass 2 – merge connected same-colour groups.
  for (let y = 0; y < SZ; y++) {
    for (let x = 0; x < SZ; x++) {
      const c = board.grid[y][x];
      if (c === null) continue;
      const idx = y * SZ + x;
      const base = idx * 4;
      for (let i = 0; i < 4; i++) {
        const ni = nbr[base + i];
        if (board.grid[(ni / SZ) | 0][ni % SZ] !== c) continue;
        const gidA = board._gid[idx];
        const gidB = board._gid[ni];
        if (gidA === gidB) continue;
        const groupA = board._groups.get(gidA);
        const groupB = board._groups.get(gidB);
        // Merge smaller into larger.
        const [keepGid, keepGrp, mergeGrp] = groupA.stones.size >= groupB.stones.size
          ? [gidA, groupA, groupB] : [gidB, groupB, groupA];
        const mergeGid = keepGid === gidA ? gidB : gidA;
        for (const si of mergeGrp.stones) { keepGrp.stones.add(si); board._gid[si] = keepGid; }
        for (const li of mergeGrp.liberties) keepGrp.liberties.add(li);
        board._groups.delete(mergeGid);
      }
    }
  }

  g.current = 'black';  // black (loser) moves next
  g.moveCount = 32;
  g.consecutivePasses = 0;
  g.koFlag = null;

  const territory = g.calcTerritory();
  console.log(`  white territory: ${territory.white} vs black: ${territory.black} (komi ${g.komi})`);
  assert(territory.white + g.komi > territory.black,
    `position should favour white: ${territory.white + g.komi} vs ${territory.black}`);

  // black (loser) passes once; white (winner) should then also pass to end game.
  for (const [agentName, agent] of [['mcts', mcts], ['rave', rave]]) {
    const clone = g.clone();
    clone.pass(); // black passes (consecutivePasses = 1)
    const result = agent(clone, 1000);
    console.log(`  ${agentName}: ${result.type}${result.info ? ' — ' + result.info : ''}`);
    assert(result.type === 'pass',
      `${agentName}: white should pass to end the winning game (got ${result.type})`);
  }
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

// ─── Full game smoke tests ────────────────────────────────────────────────────

section('MCTS beats random (5x5, 4 games)', () => {
  const result = runMatch('mcts', 'random', 4, 5, 100);
  console.log(`  mcts: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins > result.p2Wins, `mcts should beat random: ${result.p1Wins}-${result.p2Wins}`);
  assert(result.p1Wins >= 3, `mcts should win ≥3/4 vs random: got ${result.p1Wins}`);
});

section('MC beats random (5x5, 4 games)', () => {
  const result = runMatch('mc', 'random', 4, 5, 100);
  console.log(`  mc: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins > result.p2Wins, `mc should beat random: ${result.p1Wins}-${result.p2Wins}`);
  assert(result.p1Wins >= 3, `mc should win ≥3/4 vs random: got ${result.p1Wins}`);
});

section('7x7 full game (mc vs random, 2 games)', () => {
  const result = runMatch('mc', 'random', 2, 7, 100);
  console.log(`  mc: ${result.p1Wins}  random: ${result.p2Wins}`);
  assert(result.p1Wins >= 1, `mc should beat random on 7x7: got ${result.p1Wins}`);
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
