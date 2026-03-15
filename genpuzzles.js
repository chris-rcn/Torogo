'use strict';

/**
 * Auto-generate puzzle positions from MCTS self-play.
 *
 * Strategy:
 *   1. Play `--games` self-play games using MCTS with `--budget` ms per move.
 *   2. At each ply, check whether the top root-child visit fraction >= `--threshold`.
 *      If so, the position is "forced" — save it as a puzzle candidate.
 *   3. Deduplicate by Zobrist hash and emit puzzle objects to stdout.
 *
 * Usage:
 *   node genpuzzles.js [--size <n>] [--budget <ms>] [--games <n>]
 *                      [--threshold <ratio>] [--max <n>]
 *
 *   --size       board size (default: 7)
 *   --budget     MCTS budget per move in ms (default: 500)
 *   --games      number of self-play games to run (default: 50)
 *   --threshold  min fraction of visits on the top move to qualify (default: 0.65)
 *   --max        max puzzles to emit (default: 20)
 */

const { performance } = require('perf_hooks');
const { Game, ZOBRIST } = require('./game.js');
const randomAgent = require('./ai/random.js');

// ─── Parse CLI args ───────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let boardSize  = 7;
let budgetMs   = 500;
let numGames   = 50;
let threshold  = 0.65;
let maxPuzzles = 20;

for (let i = 0; i < args.length; i++) {
  if      (args[i] === '--size'      && args[i+1]) boardSize  = Number(args[++i]);
  else if (args[i] === '--budget'    && args[i+1]) budgetMs   = Number(args[++i]);
  else if (args[i] === '--games'     && args[i+1]) numGames   = Number(args[++i]);
  else if (args[i] === '--threshold' && args[i+1]) threshold  = Number(args[++i]);
  else if (args[i] === '--max'       && args[i+1]) maxPuzzles = Number(args[++i]);
}


// ─── Inline MCTS (used for both self-play moves and puzzle detection) ─────────
// Returns { best, ratio, secondBest } where:
//   best       — the move with the most root-child visits
//   ratio      — best.visits / sum(child.visits)
//   secondBest — move with second-most visits (or null)

const EXPLORATION_C = 1.4;

function legalMoves(game) {
  const moves = [];
  for (let y = 0; y < game.boardSize; y++)
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) moves.push({ type: 'place', x, y });
    }
  moves.push({ type: 'pass' });
  return moves;
}

function applyMove(game, move) {
  if (move.type === 'place') game.placeStone(move.x, move.y);
  else game.pass();
}

function playoutRandom(game) {
  const size = game.boardSize;
  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (game.board.get(x, y) === null) empty.push([x, y]);
  const moveLimit = empty.length + 20;
  let moves = 0;
  while (!game.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;
    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const [x, y] = empty[i];
      empty[i] = empty[end - 1];
      empty[end - 1] = [x, y];
      end--;
      const info = game.board.classifyEmpty(x, y, game.current);
      if (info.isTrueEye) continue;
      if (info.hasEmptyNeighbor) {
        game.board.set(x, y, game.current);
        const cap = game.board.captureGroups(x, y);
        game.consecutivePasses = 0;
        game.current = game.current === 'black' ? 'white' : 'black';
        empty[end] = empty[empty.length - 1]; empty.pop();
        if (cap.black.length + cap.white.length > 0) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true; moves++; break;
      }
      const result = game.placeStone(x, y);
      if (result) {
        empty[end] = empty[empty.length - 1]; empty.pop();
        if (result > 1) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true; moves++; break;
      }
    }
    if (!placed) { game.pass(); moves++; }
  }
  if (!game.gameOver) game.endGame();
}

function makeNode(move, parent, mover) {
  return { move, parent, mover, children: [], untried: null, wins: 0, visits: 0 };
}

function uctScore(child, parentVisits) {
  if (child.visits === 0) return Infinity;
  return child.wins / child.visits +
    EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / child.visits);
}

function analyzeMCTS(game) {
  const root = makeNode(null, null, null);
  const deadline = performance.now() + budgetMs;

  while (performance.now() < deadline) {
    // Select
    let node = root;
    const g = game.clone();
    while (node.untried !== null && node.untried.length === 0 && node.children.length > 0) {
      let best = null, bestScore = -1;
      for (const c of node.children) {
        const s = uctScore(c, node.visits);
        if (s > bestScore) { bestScore = s; best = c; }
      }
      node = best;
      applyMove(g, node.move);
    }
    // Expand
    if (!g.gameOver) {
      if (node.untried === null) node.untried = legalMoves(g);
      if (node.untried.length > 0) {
        const idx = Math.floor(Math.random() * node.untried.length);
        const move = node.untried[idx];
        node.untried[idx] = node.untried[node.untried.length - 1];
        node.untried.pop();
        const mover = g.current;
        const child = makeNode(move, node, mover);
        node.children.push(child);
        node = child;
        applyMove(g, move);
      }
    }
    // Simulate
    playoutRandom(g);
    if (!g.gameOver) g.endGame();
    const winner = g.scores.black.total > g.scores.white.total ? 'black'
                 : g.scores.white.total > g.scores.black.total ? 'white' : null;
    // Backpropagate
    let n = node;
    while (n !== null) {
      n.visits++;
      if (n.mover !== null && winner === n.mover) n.wins++;
      n = n.parent;
    }
  }

  // Find best and second-best children by visit count
  const children = root.children.filter(c => c.move.type === 'place');
  if (children.length === 0) return null;

  children.sort((a, b) => b.visits - a.visits);
  const totalVisits = children.reduce((s, c) => s + c.visits, 0);
  if (totalVisits === 0) return null;

  return {
    best: children[0].move,
    ratio: children[0].visits / totalVisits,
    secondBest: children.length > 1 ? children[1].move : null,
    totalVisits,
  };
}

// ─── Self-play harvest ────────────────────────────────────────────────────────

const puzzles = [];
const seenHashes = new Set();

process.stderr.write(`Generating puzzles: size=${boardSize} budget=${budgetMs}ms games=${numGames} threshold=${threshold}\n`);

for (let gameIdx = 0; gameIdx < numGames && puzzles.length < maxPuzzles; gameIdx++) {
  const game = new Game(boardSize, 0);

  // One random move after the constructor's centre stone, to diversify openings.
  for (let i = 0; i < 1 && !game.gameOver; i++) {
    const move = randomAgent(game);
    if (move.type === 'place') game.placeStone(move.x, move.y);
    else game.pass();
  }

  while (!game.gameOver && puzzles.length < maxPuzzles) {
    const hashKey = game.hash.toString();
    const analysis = analyzeMCTS(game);
    if (!analysis) { game.pass(); continue; }

    if (!seenHashes.has(hashKey) && analysis.ratio >= threshold) {
      seenHashes.add(hashKey);
      puzzles.push({
        toPlay: game.current,
        answer: [[analysis.best.x, analysis.best.y]],
        boardStr: game.board.toAscii(),
        ratio: analysis.ratio,
        totalVisits: analysis.totalVisits,
      });
      process.stderr.write(`  puzzle ${puzzles.length}: ${game.current} plays (${analysis.best.x},${analysis.best.y}) ratio=${analysis.ratio.toFixed(2)} visits=${analysis.totalVisits}\n`);
    }

    // Advance game using the MCTS best move
    game.placeStone(analysis.best.x, analysis.best.y);
  }

  process.stderr.write(`  game ${gameIdx + 1}/${numGames} complete (${puzzles.length} puzzles so far)\n`);
}

// ─── Output ───────────────────────────────────────────────────────────────────

console.log('// Auto-generated puzzles — paste into PUZZLES array in testpuzzles.js');
console.log(`// Generated: size=${boardSize} budget=${budgetMs}ms threshold=${threshold}`);
console.log('');

for (let i = 0; i < puzzles.length; i++) {
  const p = puzzles[i];
  const indented = p.boardStr.split('\n').map(r => '      ' + r).join('\n');
  console.log(`  {`);
  console.log(`    name: 'Auto #${i + 1}',`);
  console.log(`    toPlay: '${p.toPlay}',`);
  console.log(`    answer: [[${p.answer[0].join(', ')}]],`);
  console.log(`    board: \``);
  console.log(indented);
  console.log(`    \`,`);
  console.log(`  },`);
}

process.stderr.write(`\nDone: ${puzzles.length} puzzles emitted.\n`);
