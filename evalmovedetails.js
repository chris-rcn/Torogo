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

if (args.includes('--help') || args.includes('-h')) {
  console.error('Usage: node evalmovedetails.js --agent <name> --budget <ms> --file <path>');
  process.exit(0);
}

const agentName = get('--agent');
const budgetMs  = parseInt(get('--budget'), 10);
const filePath  = get('--file');
const verbose   = args.includes('--verbose');

if (!filePath)              { console.error('--file is required'); process.exit(1); }
if (!agentName)             { console.error('--agent is required'); process.exit(1); }
if (!get('--budget'))               { console.error('--budget is required'); process.exit(1); }
if (isNaN(budgetMs) || budgetMs < 1) { console.error('--budget must be a positive integer'); process.exit(1); }

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));
const { Game2, PASS, coordStr, parseMove, agentMoveToIdx } = require('./game2.js');

const lines = fs.readFileSync(filePath, 'utf8').split('\n').filter(l => l.trim());

// Column widths
const wMove = 5;  // move string e.g. "pass" or "g7"
const wWR = 5;  // WR display e.g. "0.500"

const { performance } = require('perf_hooks');
const startTime = performance.now();

let gapSqSum = 0;

if (verbose) {
  console.log(
    `${'hist'.padStart(4)}  ` +
    `${'top'.padEnd(wMove)} ${'WR'.padStart(wWR)}  ` +
    `${'agent'.padEnd(wMove)} ${'WR'.padStart(wWR)}  ` +
    `gap`
  );
  console.log('-'.repeat(4 + wMove + wWR + wMove + wWR + 20));
}

for (let i = 0; i < lines.length; i++) {
  const { boardSize, history, candidates } = JSON.parse(lines[i]);

  const game = new Game2(boardSize);
  for (const h of history) game.play(parseMove(h, boardSize));

  const agentMove = agent(game, budgetMs);
  const agentStr  = coordStr(agentMoveToIdx(agentMove, boardSize), boardSize);

  const topCand   = candidates[0];
  const agentCand = candidates.find(c => c.m === agentStr);

  const topWr   = topCand.kwr / 1000;
  const agentWr = agentCand.kwr / 1000;
  const gap     = topWr - agentWr;

  const fmtWr = wr => wr.toFixed(3);
  const fmtGap = g => g.toFixed(3);

  gapSqSum += gap * gap;

  if (verbose) console.log(
    `${String(history.length).padStart(4)}  ` +
    `${topCand.m.padEnd(wMove)} ${fmtWr(topWr).padStart(wWR)}  ` +
    `${agentStr.padEnd(wMove)} ${fmtWr(agentWr).padStart(wWR)}  ` +
    `${fmtGap(gap)}  ` + agentMove.info
  );
}

const gapRms = Math.sqrt(gapSqSum / lines.length);
const elapsedMs = performance.now() - startTime;
const elapsed = (elapsedMs / 1000).toFixed(1);
const tMoveMs = (elapsedMs / lines.length).toFixed(1);
console.log(`positions: ${lines.length}  elapsed: ${elapsed}s  tMoveMs: ${tMoveMs}  gapRms: ${gapRms.toFixed(4)}  agent: ${agentName}`);

