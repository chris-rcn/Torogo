#!/usr/bin/env node
'use strict';

// Evaluate an agent against pre-computed move details.
//
// Usage: node evalmovedetails.js --agent <name> --budget <ms> --file <path>
//
// Reads a newline-delimited JSON file produced by createmovedetails.js.
// For each position, reconstructs the game from the history, asks the agent
// to choose a move, then compares its kwr to the top-rated move's kwr.

const fs   = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const { Game2, PASS, coordStr, parseMove } = require('./game2.js');

function loadPositions(filePath) {
  return fs.readFileSync(filePath, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
}

// Evaluate agent on positions; returns { rmsErr, count }.
function evalPositions(agent, positions, budgetMs) {
  let gapSqSum = 0;
  for (const { boardSize, history, candidates } of positions) {
    const game = new Game2(boardSize);
    for (const h of history) game.play(parseMove(h, boardSize));

    const agentMove = agent(game, budgetMs);
    const agentStr  = coordStr(agentMove.move, boardSize);

    const topCand   = candidates[0];
    const _found    = candidates.find(c => c.m === agentStr);
    const _last     = candidates.findLast(c => c.kwr !== undefined);
    const agentCand = (_found?.kwr !== undefined) ? _found : _last;

    const gap = topCand.kwr / 1000 - agentCand.kwr / 1000;
    gapSqSum += gap * gap;
  }
  return { rmsErr: Math.sqrt(gapSqSum / positions.length), count: positions.length };
}

if (require.main === module) {
  if (args.includes('--help') || args.includes('-h')) {
    console.error('Usage: node evalmovedetails.js --agent <name> --budget <ms> --file <path>');
    process.exit(0);
  }

  const agentName = get('--agent');
  const budgetMs  = parseInt(get('--budget', 2000), 10);
  const filePath  = get('--file');
  const verbose   = args.includes('--verbose');

  if (!filePath)              { console.error('--file is required'); process.exit(1); }
  if (!agentName)             { console.error('--agent is required'); process.exit(1); }
  if (isNaN(budgetMs) || budgetMs < 1) { console.error('--budget must be a positive integer'); process.exit(1); }

  const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));
  const positions = loadPositions(filePath);

  // Column widths
  const wMove = 5;
  const wWR   = 5;
  const wCount = String(positions.length).length;

  const { performance } = require('perf_hooks');
  const startTime = performance.now();
  let gapSqSum = 0;
  let printPeriodMs = 1000, lastPrintTime = startTime, lastPrintCount = 0;

  function printStats(count) {
    const elapsedMs = performance.now() - startTime;
    const elapsed   = (elapsedMs / 1000).toFixed(1);
    const tMoveMs   = (elapsedMs / count).toFixed(1);
    const rmsErr    = Math.sqrt(gapSqSum / count);
    console.log(`N=${String(count).padStart(wCount)} rmsErr=${rmsErr.toFixed(4)} elapsed=${elapsed}s tMoveMs=${tMoveMs} agent=${agentName}`);
  }

  function maybePrint(count) {
    const now = performance.now();
    if (now - lastPrintTime < printPeriodMs) return;
    if (count === lastPrintCount) return;
    lastPrintTime = now; lastPrintCount = count;
    printStats(count);
    printPeriodMs = Math.round(printPeriodMs * 1.5);
  }

  if (verbose) {
    console.log(
      `${'hist'.padStart(4)}  ` +
      `${'top'.padEnd(wMove)} ${'WR'.padStart(wWR)}  ` +
      `${'agent'.padEnd(wMove)} ${'WR'.padStart(wWR)}  gap`
    );
    console.log('-'.repeat(4 + wMove + wWR + wMove + wWR + 20));
  }

  for (let i = 0; i < positions.length; i++) {
    const { boardSize, history, candidates } = positions[i];
    const game = new Game2(boardSize);
    for (const h of history) game.play(parseMove(h, boardSize));

    const agentMove = agent(game, budgetMs);
    const agentStr  = coordStr(agentMove.move, boardSize);

    const topCand   = candidates[0];
    const _found    = candidates.find(c => c.m === agentStr);
    const _last     = candidates.findLast(c => c.kwr !== undefined);
    const agentCand = (_found?.kwr !== undefined) ? _found : _last;

    const topWr  = topCand.kwr / 1000;
    const agentWr = agentCand.kwr / 1000;
    const gap    = topWr - agentWr;
    gapSqSum += gap * gap;
    maybePrint(i + 1);

    if (verbose) console.log(
      `${String(history.length).padStart(4)}  ` +
      `${topCand.m.padEnd(wMove)} ${topWr.toFixed(3).padStart(wWR)}  ` +
      `${agentStr.padEnd(wMove)} ${agentWr.toFixed(3).padStart(wWR)}  ` +
      `${gap.toFixed(3)}  ` + agentMove.info
    );
  }

  printStats(positions.length);
}

// Evaluate agent on a random sample of n positions from the pool.
// If n >= pool.length, uses the full pool.  Returns { rmsErr, count }.
function evalPositionsSample(agent, pool, n, budgetMs) {
  let positions = pool;
  if (n < pool.length) {
    const sample = pool.slice();
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(Math.random() * (sample.length - i));
      [sample[i], sample[j]] = [sample[j], sample[i]];
    }
    positions = sample.slice(0, n);
  }
  return evalPositions(agent, positions, budgetMs);
}

module.exports = { loadPositions, evalPositions, evalPositionsSample };

