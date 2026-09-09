'use strict';

// find-health-errors.js — hunt for positions where the chain-health model is
// most wrong.
//
// For each sampled endpoint (the same leaf/descend machinery train-health.js
// trains on), every chain gets the model's P(survive) and an empirical q from
// --playouts independent completions of that one position.  The tool keeps a
// running record of the largest |p - q| and prints the board whenever the
// record is beaten, then carries on looking for something worse.  So the
// output is a short, monotonically worsening list of the model's most
// egregious calls, not a dump.
//
// The band and delta are read from the model itself, so the sampled positions
// always match the distribution it was fitted on.
//
// Usage: node find-health-errors.js --health <model.js> --corpus <games.txt> [options]
//   --health PATH   train-health.js save file (required)
//   --corpus PATH   gen-games corpus (required)
//   --playouts N    completions per position, for q (default 100)
//   --games N       corpus games to scan (default: all)
//   --min-q F       ignore chains whose q is outside [--min-q, --max-q]; use
//                   to hunt only confident cases (default 0 / 1 = no filter)
//   --max-q F
//   --seed N        rng seed (default 23)

const fs = require('fs');
const path = require('path');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');
const VPat = require('./vpatterns.js');
const PPat = require('./ppat-lib.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['health', 'corpus', 'playouts', 'games', 'min-q', 'max-q', 'seed']);
if (opts.help || !opts.health || !opts.corpus) {
  console.error(`Usage: node find-health-errors.js --health <model.js> --corpus <games.txt> [options]

Prints the board each time a chain beats the running record for |p - q|,
where p is the health model's prediction and q the survival ratio over
--playouts completions of that position.

  --health PATH   train-health.js save file
  --corpus PATH   gen-games corpus
  --playouts N    completions per position (default 100)
  --games N       corpus games to scan (default: all)
  --min-q F       only consider chains with q in [min-q, max-q] (default 0..1)
  --max-q F
  --seed N        rng seed (default 23)
  --help          show this message`);
  process.exit(opts.help ? 0 : 1);
}

const M = parseInt(opts.playouts || '100', 10);
const MIN_Q = parseFloat(opts['min-q'] !== undefined ? opts['min-q'] : '0');
const MAX_Q = parseFloat(opts['max-q'] !== undefined ? opts['max-q'] : '1');
const rng = makeRng(parseInt(opts.seed || '23', 10) || 1);

const health = VPat.resolveHealthModel(opts.health);
const raw = require(path.resolve(opts.health));
// The model records the distribution it was fitted on; sample from that same
// one, or the errors found would be about a mismatch rather than the model.
const SIZE = raw.size, MIN_PH = raw.minPhase, MAX_PH = raw.maxPhase, DELTA = raw.delta;
if (MIN_PH === undefined || MAX_PH === undefined || DELTA === undefined) {
  console.error('find-health-errors: model predates the band/delta fields — retrain it');
  process.exit(1);
}
const area = SIZE * SIZE;

const ppatModel = PPat.loadWeights(path.join(__dirname, 'ppat-data.js'));
ppatModel.uniformBelowPhase = 0.6;
const ppatState = PPat.createState(SIZE);

const corpus = fs.readFileSync(opts.corpus, 'utf-8').split('\n').filter(l => l && l[0] !== '#');
const GAMES = Math.min(opts.games !== undefined ? parseInt(opts.games, 10) : Infinity, corpus.length);
console.log(`find-health-errors: health ${opts.health}  corpus ${opts.corpus} (${GAMES} games)  ` +
            `band [${MIN_PH}, ${MAX_PH}]  delta ${DELTA}  size ${SIZE}  playouts ${M}` +
            (MIN_Q > 0 || MAX_Q < 1 ? `  q in [${MIN_Q}, ${MAX_Q}]` : ''));

// One endpoint, exactly as train-health.js samples it.
function sampleEndpoint(line) {
  const sp = line.indexOf(' ');
  if (parseInt(line.slice(0, sp), 10) !== SIZE) return null;
  const toks = line.slice(sp + 1).split(',');
  const walk = new Game2(SIZE);
  let game = null, nEligible = 0;
  for (let i = 0; i < toks.length; i++) {
    const ph = 1 - walk.emptyCount / area;
    if (ph > MAX_PH) break;
    if (ph >= MIN_PH) { nEligible++; if (rng.random() * nEligible < 1) game = walk.clone(); }
    walk.play(parseMove(toks[i], SIZE));
  }
  if (!game) return null;
  const stopEmpty = game.emptyCount - Math.ceil(DELTA * area);
  const lim = 3 * game.emptyCount + 20;
  let n = 0;
  while (!game.gameOver && game.emptyCount > stopEmpty && n < lim) {
    game.play(PPat.ppatMove(game, ppatState, ppatModel, rng));
    n++;
  }
  return game.emptyCount > stopEmpty ? null : game;
}

// Whole board, every row, TRANSLATED so the subject chain sits in the middle.
// The board is toroidal, so translation is a symmetry and costs nothing — it
// just stops the chain of interest from being split across the edges of the
// printout.  The subject is upper case, every other stone lower case.
//
// The centre is the chain's CIRCULAR mean on each axis (via the mean angle),
// which is the wrap-correct notion of "middle" for a chain that straddles the
// seam; a plain average would place a wrapped chain on the opposite side.
function board(game, subjectGid, stones) {
  const cells = game.cells, gid = game._gid;
  const circMean = coord => {
    let sx = 0, cx = 0;
    for (let i = 0; i < stones.length; i++) {
      const a = 2 * Math.PI * coord(stones[i]) / SIZE;
      sx += Math.sin(a); cx += Math.cos(a);
    }
    let m = Math.atan2(sx, cx) / (2 * Math.PI) * SIZE;
    if (m < 0) m += SIZE;
    return m;
  };
  const mx = circMean(i => i % SIZE), my = circMean(i => (i / SIZE) | 0);
  const half = SIZE >> 1;
  const dx = ((half - Math.round(mx)) % SIZE + SIZE) % SIZE;
  const dy = ((half - Math.round(my)) % SIZE + SIZE) % SIZE;
  const rows = [];
  for (let y = 0; y < SIZE; y++) {
    let row = '';
    for (let x = 0; x < SIZE; x++) {
      const sy = (y - dy + SIZE) % SIZE, sx2 = (x - dx + SIZE) % SIZE;
      const i = sy * SIZE + sx2, c = cells[i];
      if (c === 0) { row += ' .'; continue; }
      const ch = c === 1 ? 'x' : 'o';
      row += ' ' + (gid[i] === subjectGid ? ch.toUpperCase() : ch);
    }
    rows.push(row);
  }
  return rows.join('\n');
}

let worst = -1, nPos = 0, nChains = 0;
const t0 = Date.now();
for (let gi = 0; gi < GAMES; gi++) {
  const pos = sampleEndpoint(corpus[gi]);
  if (!pos) continue;
  nPos++;
  const cells = pos.cells, gid = pos._gid, ls = pos._ls, nbr = pos._nbr, dnbr = pos._dnbr;

  // chains of this position
  const byGid = new Map();
  for (let i = 0; i < area; i++) {
    if (cells[i] === 0) continue;
    let r = byGid.get(gid[i]);
    if (!r) { r = { c: cells[i], stones: [], libs: [] }; byGid.set(gid[i], r); }
    r.stones.push(i);
  }
  for (let l = 0; l < area; l++) {
    if (cells[l] !== 0) continue;
    const b4 = l * 4;
    let s0 = -1, s1 = -1, s2 = -1;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d];
      if (cells[j] === 0) continue;
      const gj = gid[j];
      if (gj === s0 || gj === s1 || gj === s2) continue;
      if (s0 < 0) s0 = gj; else if (s1 < 0) s1 = gj; else s2 = gj;
      byGid.get(gj).libs.push(l);
    }
  }
  const libsByGid = new Map();
  for (const [g, r] of byGid) libsByGid.set(g, r.libs);

  // model prediction per chain
  const gids = [...byGid.keys()];
  const pred = new Float64Array(gids.length);
  for (let i = 0; i < gids.length; i++) {
    const r = byGid.get(gids[i]);
    pred[i] = VPat.chainSurvivalP(health, cells, nbr, dnbr, gid, ls, r.c, gids[i], r.libs, r.stones, libsByGid);
  }

  // empirical q from M completions of THIS position — one sweep serves every chain
  const surv = new Int32Array(gids.length);
  for (let m = 0; m < M; m++) {
    const g = pos.clone();
    let k = 0;
    const lim = 3 * g.emptyCount + 20;
    while (!g.gameOver && k < lim) { g.play(PPat.ppatMove(g, ppatState, ppatModel, rng)); k++; }
    for (let i = 0; i < gids.length; i++) {
      const r = byGid.get(gids[i]);
      if (g.cells[r.stones[0]] === r.c) surv[i]++;
    }
  }

  for (let i = 0; i < gids.length; i++) {
    const q = surv[i] / M, p = pred[i], err = Math.abs(p - q);
    nChains++;
    if (q < MIN_Q || q > MAX_Q || err <= worst) continue;
    worst = err;
    const r = byGid.get(gids[i]);
    const phase = 1 - pos.emptyCount / area;
    console.log(`\n=== |p-q| ${err.toFixed(3)}   model ${p.toFixed(3)}  actual ${q.toFixed(3)} (${surv[i]}/${M})` +
                `   game ${gi}  phase ${phase.toFixed(2)}`);
    // The health model has no side-to-move input, so an atari'd chain whose
    // owner is to move and one whose opponent is to move are the same example
    // with opposite labels — print the turn to make those cases visible.
    const mine = pos.current === r.c;
    console.log(`    chain: ${r.c === 1 ? 'black' : 'white'}  ${r.stones.length} stones  ${r.libs.length} liberties` +
                `   to move: ${pos.current === 1 ? 'black' : 'white'} (${mine ? 'the chain\'s owner' : 'the opponent'})` +
                `   (upper case below)`);
    console.log(board(pos, gids[i], r.stones));
  }
  if (nPos % 25 === 0) {
    process.stderr.write(`\r${Util.fmtMs(Date.now() - t0)}  ${nPos} positions  ${nChains} chains  worst ${worst.toFixed(3)}   `);
  }
}
console.error('');
console.log(`\nscanned ${nPos} positions, ${nChains} chains in ${((Date.now() - t0) / 1000).toFixed(1)}s   worst |p-q| ${worst.toFixed(3)}`);
