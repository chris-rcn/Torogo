'use strict';

const { Game2, PASS, BLACK, WHITE, KOMI } = require('./game2.js');
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
  const { getMove: p1 } = require(`./ai/${p1Name}.js`);
  const { getMove: p2 } = require(`./ai/${p2Name}.js`);
  let p1Wins = 0, p2Wins = 0;

  for (let i = 0; i < games; i++) {
    const g = new Game2(size);
    const blackAgent = i % 2 === 0 ? p1 : p2;
    const whiteAgent = i % 2 === 0 ? p2 : p1;
    const p1IsBlack = i % 2 === 0;

    while (!g.gameOver) {
      const agent = g.current === BLACK ? blackAgent : whiteAgent;
      const move = agent(g, budget);
      g.play(move.type === 'place' ? move.y * size + move.x : PASS);
    }

    const sc = g.calcScore();
    const blackWins = sc.black > sc.white;
    if (p1IsBlack ? blackWins : !blackWins) p1Wins++;
    else p2Wins++;
  }

  return { p1Wins, p2Wins };
}

// ─── Playout performance ─────────────────────────────────────────────────────

section('MC playout throughput (7x7)', () => {
  const { getMove: mc } = require('./ai/mc.js');
  const g = new Game2(7);
  const budgetMs = 200;
  const t0 = performance.now();
  const move = mc(g, budgetMs);
  const elapsed = performance.now() - t0;
  console.log(`  Budget: ${budgetMs}ms, actual: ${elapsed.toFixed(0)}ms`);
  assert(elapsed < budgetMs * 1.5, `MC should finish within 1.5x budget: took ${elapsed.toFixed(0)}ms`);
  assert(elapsed > budgetMs * 0.5, `MC should use most of the budget: only ${elapsed.toFixed(0)}ms`);
});

section('MCTS playout throughput (7x7)', () => {
  const { getMove: mcts } = require('./ai/mcts.js');
  const g = new Game2(7);
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
  const { getMove: rave } = require('./ai/rave.js');
  const { getMove: mcts } = require('./ai/mcts.js');

  // Hardcoded 7x7 position (found by playing area/2 random alternating moves
  // then 4 rave moves for white with random black responses).
  //
  // white territory 22, black 17, neutral 10, komi 3.5
  // → white total 25.5 vs black 17 (white leads by 8.5)
  const layout = [
    '. . . W . W W',  // y=0
    'B . B . . B .',  // y=1
    '. B . W . B .',  // y=2
    'W B . B W W .',  // y=3
    'W . B W . . W',  // y=4
    'B W B W . W B',  // y=5
    'W W B . W . B',  // y=6
  ];

  const SZ = 7;
  const g = new Game2(SZ, false);

  // Place stones using _place for correct group tracking.
  for (let y = 0; y < SZ; y++) {
    const cells = layout[y].split(' ');
    for (let x = 0; x < SZ; x++) {
      const ch = cells[x];
      if (ch === 'B') g._place(y * SZ + x, BLACK);
      else if (ch === 'W') g._place(y * SZ + x, WHITE);
    }
  }

  g.current = BLACK;  // black (loser) moves next
  g.moveCount = 32;
  g.consecutivePasses = 0;
  g.ko = PASS;

  const score = g.calcScore();
  console.log(`  white score: ${score.white.toFixed(1)} vs black: ${score.black.toFixed(1)} (komi ${KOMI(g.N)})`);
  assert(score.white > score.black,
    `position should favour white: ${score.white.toFixed(1)} vs ${score.black.toFixed(1)}`);

  // black (loser) passes once; white (winner) should then also pass to end game.
  for (const [agentName, agent] of [['mcts', mcts], ['rave', rave]]) {
    const clone = g.clone();
    clone.play(PASS); // black passes (consecutivePasses = 1)
    const result = agent(clone, 1000);
    console.log(`  ${agentName}: ${result.type}${result.info ? ' — ' + result.info : ''}`);
    assert(result.type === 'pass',
      `${agentName}: white should pass to end the winning game (got ${result.type})`);
  }
});

// ─── AI moves are always legal ───────────────────────────────────────────────

section('AI legality stress test (all agents, 3 full games each)', () => {
  const agents = ['random', 'mc', 'mcts', 'amaf'];
  let allLegal = true;

  for (const name of agents) {
    const { getMove: agent } = require(`./ai/${name}.js`);
    for (let i = 0; i < 3; i++) {
      const g = new Game2(5);
      let moveNum = 0;
      while (!g.gameOver && moveNum < 200) {
        const move = agent(g, 20);
        const idx = move.type === 'place' ? move.y * 5 + move.x : PASS;
        if (idx !== PASS) {
          const result = g.play(idx);
          if (result === false) {
            allLegal = false;
            console.error(`  ${name} returned illegal move (${move.x},${move.y}) on move ${moveNum}`);
            break;
          }
        } else {
          g.play(PASS);
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
