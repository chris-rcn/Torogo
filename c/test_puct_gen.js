'use strict';
// Fixture for c/test_puct.bin: ai/puct-ppat-fp.js searches with uniform playouts.
//
//   node c/test_puct_gen.js <fpol-model.js> [positions] [seed] > fixture.txt
//
// One line per position: N nMoves m1..mN playouts seed move rootWinRatio M
// (move visits wins)*M, where the moves replay from Game2(N, true) and the
// search runs on that fresh replay (so its empty-cell order is C g2_play's),
// with makeRng(seed) and PPAT_MIN_PHASE=1 (uniform playouts, bit-exact in C).

const path = require('path');
const Util = require(path.join(__dirname, '..', 'util.js'));
const { Game2, PASS } = require(path.join(__dirname, '..', 'game2.js'));
const { makeRng } = require(path.join(__dirname, '..', 'xorshift.js'));

const [fpolPath, nPos = '60', seed = '1'] = process.argv.slice(2);
if (!fpolPath) { console.error('usage: node c/test_puct_gen.js <fpol-model.js> [positions] [seed]'); process.exit(1); }
const log = console.log; console.log = console.error;   // agent banner off stdout
const agent = require(path.join(__dirname, '..', 'ai', 'puct-ppat-fp.js'))
  .create(Util.makeCfg(null, { FPOL_DATA: fpolPath, PPAT_MIN_PHASE: '1' }));
console.log = log;
const rng = makeRng(parseInt(seed, 10));
const out = [];
for (let p = 0; out.length < parseInt(nPos, 10); p++) {
  const N = p % 2 ? 13 : 9;
  const g = new Game2(N, true);
  const target = Math.floor(rng.random() * N * N * 1.1);
  const moves = [];
  while (moves.length < target && !g.gameOver) {
    const mv = g.randomLegalMove(rng);
    if (mv === PASS) break;
    g.play(mv); moves.push(mv);
  }
  const r = new Game2(N, true);
  for (const mv of moves) r.play(mv);
  if (r.gameOver || r.consecutivePasses > 0) continue;
  const playouts = 20 + Math.floor(rng.random() * 400);
  const s = 1 + Math.floor(rng.random() * 1e9);
  const res = agent.getMove(r, 0, { rng: makeRng(s), playoutLimit: playouts });
  const ch = res.children.map(c => `${c.move} ${c.visits} ${c.wins}`);
  out.push(`${N} ${moves.length} ${moves.join(' ')} ${playouts} ${s} ${res.move} ${res.rootWinRatio} ${ch.length} ${ch.join(' ')}`);
}
process.stdout.write(out.join('\n') + '\n');
