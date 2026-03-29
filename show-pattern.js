#!/usr/bin/env node
'use strict';

// show-pattern.js — scan games to find positions with extreme pattern priors.
//
// Scans all positions in games file, computes pattern priors as ravepat does,
// and prints positions whenever a new maximum equiv-wins or equiv-losses is found.
//
// Usage:
//   node show-pattern.js --wins|--losses
//   [--pat <path>]          pattern JS file  (default: out/patdata-selection.js)
//   [--games <path>]        games file       (default: games.tmp)
//   [--pattern-equiv <n>]   equiv visits     (default: 50)
//   [--conf-k <n>]          confidence half-life (default: 20)

const fs   = require('fs');
const path = require('path');
const { Game2, BLACK, WHITE } = require('./game2.js');
const { getPatternHashes }    = require('./pattern1.js');
const { getLadderStatus2 }    = require('./ladder2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const wantWins     = args.includes('--wins');
const wantLosses   = args.includes('--losses');
const patFile      = get('--pat',           'out/patdata-selection.js');
const gamesFile    = get('--games',         'games.tmp');
const patternEquiv  = parseFloat(get('--pattern-equiv',  '50'));
const confK         = parseFloat(get('--conf-k',         '20'));
const minCandidates = parseInt(get('--min-candidates',  '0'), 10);

if (!wantWins && !wantLosses) {
  process.stderr.write('Usage: node show-pattern.js --wins|--losses [--pat <f>] [--games <f>] [--pattern-equiv <n>] [--conf-k <n>] [--min-candidates <n>]\n');
  process.exit(1);
}

const table = require(path.resolve(patFile));

// ── Prior calculation (mirrors ravepat makeNode logic) ────────────────────────

// Returns array of { idx, prior, patEquiv } for all candidates at this position.
function computePriors(game2, candidates) {
  if (candidates.length === 0) return [];
  const hashes  = getPatternHashes(game2, candidates);
  const entries = hashes.map(({ hash }) => table.get(hash));

  return candidates.map((idx, k) => {
    const e = entries[k];
    if (!e) return { idx, prior: 0, patEquiv: 0, count: 0, ratio: 0, hash: hashes[k].hash };
    const ratio    = e[0];
    const count    = e[1];
    const conf     = count / (count + confK);
    const patEquiv = conf * patternEquiv;
    return { idx, prior: ratio, patEquiv, count, hash: hashes[k].hash };
  });
}


function printAdjacentLadders(game2, idx) {
  const N = game2.N;
  const dirNames = ['N','S','W','E'];
  const seenGids = new Set();
  for (let i = 0; i < 4; i++) {
    const nidx = game2._nbr[idx * 4 + i];
    const dir = dirNames[i];
    if (game2.cells[nidx] === 0) continue;
    const gid = game2._gid[nidx];
    if (seenGids.has(gid)) continue;
    seenGids.add(gid);
    const { count: lc } = game2.groupLibs2(nidx);
    if (lc === 0 || lc > 2) continue;
    const color  = game2.cells[nidx] === BLACK ? '●' : '○';
    const size   = game2.groupSize(gid);
    const status = getLadderStatus2(game2, nidx);
    const urgentCoords = status.urgentLibs.map(i => {
      const ux = i % N, uy = (i / N) | 0;
      return String.fromCharCode(97 + ux) + String.fromCharCode(97 + uy);
    });
    const urgentStr = urgentCoords.length > 0 ? `  urgent=[${urgentCoords.join(',')}]` : '';
    console.log(`  ${dir} neighbor: ${color} chain size=${size} libs=${lc}  moverSucceeds=${status.moverSucceeds}${urgentStr}`);
  }
}

// ── Scan games ────────────────────────────────────────────────────────────────

const gameLines = fs.readFileSync(gamesFile, 'utf8').trim().split('\n');
let bestVal = -Infinity;

for (const line of gameLines) {
  const fields    = line.split(',');
  const N         = parseInt(fields[0], 10);
  const cap       = N * N;
  const lastField = fields[fields.length - 1];
  const hasWinner = lastField === 'b' || lastField === 'w';
  const moves     = hasWinner ? fields.slice(1, -1) : fields.slice(1);

  const game = new Game2(N);

  for (let mi = 1; mi < moves.length; mi++) {
    const candidates = [];
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] !== 0) continue;
      if (game.isTrueEye(i)) continue;
      if (game.isLegal(i)) candidates.push(i);
    }

    if (candidates.length >= minCandidates) {
      const priors = computePriors(game, candidates);
      for (const { idx, prior, patEquiv, count, hash } of priors) {
        const equivWins   = patEquiv * prior;
        const equivLosses = patEquiv * (1 - prior);
        const val = wantWins ? equivWins : equivLosses;

        if (val > bestVal) {
          bestVal = val;
          console.log(`Game move ${mi}, ${game.current === BLACK ? '●' : '○'} to play  candidates=${candidates.length}  hash=${hash}  encounters=${count}  winRate=${prior.toFixed(4)}  equiv-${wantWins ? 'wins' : 'losses'}=${val.toFixed(4)}`);
          console.log(game.toString(idx, { centerAt: idx }));
          printAdjacentLadders(game, idx);
          console.log();
        }
      }
    }

    const token = moves[mi];
    if (!token || token === '..') { game.play(-1); continue; }
    const mx = token.charCodeAt(0) - 97;
    const my = token.charCodeAt(1) - 97;
    game.play(my * N + mx);
  }
}
