'use strict';

// filter-evals-hard.js — keep the uniform-hard fraction of an eval file.
//
// Scores every position by how badly UNIFORM playouts estimate its label:
// difficulty = (mean uniform-playout result − winRatio)², from --playouts
// rollouts per position.  Keeps the hardest --keep fraction, preserving the
// input order (the head of a shuffled file stays a representative sample, so
// train_ppat's --test-pos head-block convention still works).  Positions
// where even uniform reaches the label cannot separate two policies; what
// remains is where playout policy quality has room to matter.
//
// Uniform is the one referee with no circularity: model-independent,
// oracle-independent, parameter-free.  Note the difficulty measurement is
// itself noisy, so the kept set is partly selected on noise — cut it ONCE
// and freeze the file; re-scoring later would silently move the yardstick.
//
// Output to stdout (original '#' headers passed through, plus a provenance
// line); per-position progress and summary stats to stderr.
//
// Usage:
//   node filter-evals-hard.js --file evals.txt [--playouts 1000] [--keep 0.5]
//        [--limit N] [--seed 1]  > hard.txt

const fs = require('fs');
const { Game2, BLACK, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'playouts', 'keep', 'limit', 'seed']);
if (opts.help || !opts.file) {
  console.error('Usage: node filter-evals-hard.js --file <evals.txt> [--playouts 1000] [--keep 0.5]\n' +
                '                                 [--limit N] [--seed 1]  > hard.txt');
  process.exit(opts.help ? 0 : 1);
}

const playouts = parseInt(opts.playouts !== undefined ? opts.playouts : '1000', 10);
const keep     = parseFloat(opts.keep !== undefined ? opts.keep : '0.5');
const limit    = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const seed     = parseInt(opts.seed !== undefined ? opts.seed : '1', 10);
const rng      = makeRng(seed);

const raw = fs.readFileSync(opts.file, 'utf8').split('\n');
const headers = [];
const records = [];        // { line, size, moves, winRatio }
for (const line of raw) {
  if (!line) continue;
  if (line[0] === '#') { headers.push(line); continue; }
  const p = line.split(/\s+/);
  if (p.length < 3) continue;
  const size = parseInt(p[0], 10);
  records.push({ line, size, moves: p[1].split(','), winRatio: parseFloat(p[2]) });
  if (records.length >= limit) break;
}
if (records.length === 0) { console.error('no records'); process.exit(1); }

// Uniform playout from `game` to the end (mutates it); 1 if BLACK wins.
function playoutUniform(game) {
  const moveLimit = 3 * game.emptyCount + 20;
  let moves = 0;
  while (!game.gameOver && moves < moveLimit) {
    game.play(game.randomLegalMove(rng));
    moves++;
  }
  return game.estimateWinner() === BLACK ? 1 : 0;
}

process.stderr.write(`filter-evals-hard: ${records.length} positions, ${playouts} uniform playouts each, keep ${keep}\n`);
const t0 = Date.now();

for (let i = 0; i < records.length; i++) {
  const r = records[i];
  const pos = new Game2(r.size);
  for (const t of r.moves) {
    if (!pos.play(parseMove(t, r.size))) { console.error(`record ${i}: illegal replay`); process.exit(1); }
  }
  const mover = pos.current;
  let wins = 0;
  for (let p = 0; p < playouts; p++) {
    const g = pos.clone();
    const res = playoutUniform(g);
    wins += mover === BLACK ? res : 1 - res;
  }
  const err = wins / playouts - r.winRatio;
  r.difficulty = err * err;

  if ((i + 1) % 1000 === 0) {
    const s = (Date.now() - t0) / 1000;
    process.stderr.write(`  ${i + 1}/${records.length}  ${(s / (i + 1) * 1000).toFixed(0)}ms/pos  ${(s / 60).toFixed(1)}min\n`);
  }
}

// Threshold at the keep-quantile; filter preserving input order.
const sorted = records.map(r => r.difficulty).sort((a, b) => b - a);
const nKeep  = Math.max(1, Math.round(records.length * keep));
const thresh = sorted[nKeep - 1];
const kept   = records.filter(r => r.difficulty >= thresh);

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
process.stderr.write(`difficulty: mean ${mean(records.map(r => r.difficulty)).toFixed(4)} all, ` +
                     `${mean(kept.map(r => r.difficulty)).toFixed(4)} kept, threshold ${thresh.toFixed(4)}, ` +
                     `kept ${kept.length}/${records.length}\n`);

for (const h of headers) process.stdout.write(h + '\n');
process.stdout.write(`# filter-evals-hard: playouts ${playouts} keep ${keep} seed ${seed}; kept ${kept.length} of ${records.length} from ${opts.file}\n`);
for (const r of kept) process.stdout.write(r.line + '\n');
