'use strict';
// test-wasm-board.js — the module's board copy (ppat-wasm.js setRoot / descend,
// c/wasm_ppat.c) must equal the JS Game2 it mirrors: from random positions of
// random games, random descent paths (passes included) are played on a JS clone
// and in the module, and every cell, chain liberty count and game-state field is
// compared.  Also times setRoot and descend.  Build first with c/build-wasm.sh.
//
//   node c/test-wasm-board.js [--games 40] [--paths 30] [--seed 1]

const path = require('path');
const R = path.join(__dirname, '..');
const Util = require(path.join(R, 'util.js'));
const { Game2, PASS } = require(path.join(R, 'game2.js'));
const { makeRng } = require(path.join(R, 'xorshift.js'));
const PpatWasm = require(path.join(R, 'ppat-wasm.js'));

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['games', 'paths', 'seed']);
if (opts.help) { console.log('usage: node c/test-wasm-board.js [--games 40] [--paths 30] [--seed 1]'); process.exit(0); }
const GAMES = parseInt(opts.games || '40', 10), PATHS = parseInt(opts.paths || '30', 10), SEED = parseInt(opts.seed || '1', 10);
const SIZE = 13, AREA = SIZE * SIZE;

const pw = PpatWasm.loadSync(path.join(__dirname, 'ppat-playout-test.wasm'));
pw.init(SIZE);
const X = pw.exports;
const rng = makeRng(SEED);

// First difference between the module's descended board and JS game g, or null.
function diff(g) {
  const cells = new Int8Array(X.memory.buffer, X.wasm_sim_cells(), AREA);
  for (let i = 0; i < AREA; i++) {
    if (cells[i] !== g.cells[i]) return `cell ${i}: wasm ${cells[i]} js ${g.cells[i]}`;
    const jl = g.cells[i] === 0 ? -1 : g._ls[g._gid[i]], wl = X.wasm_sim_libs(i);
    if (jl !== wl) return `liberties at ${i}: wasm ${wl} js ${jl}`;
  }
  const fields = [['current', g.current], ['ko', g.ko], ['emptyCount', g.emptyCount], ['consecutivePasses', g.consecutivePasses],
                  ['gameOver', g.gameOver ? 1 : 0], ['lastMove', g.lastMove], ['koStone[0]', g.koStone[0]],
                  ['koStone[2]', g.koStone[2]], ['moveCount', g.moveCount]];
  for (let k = 0; k < fields.length; k++) {
    const w = X.wasm_sim_info(k);
    if (w !== fields[k][1]) return `${fields[k][0]}: wasm ${w} js ${fields[k][1]}`;
  }
  return null;
}

let roots = 0, leaves = 0, moves = 0, failures = 0;
for (let gi = 0; gi < GAMES && !failures; gi++) {
  const g = new Game2(SIZE, false);
  const rootAt = 5 + Math.floor(rng.random() * 150);
  for (let ply = 0; ply < rootAt && !g.gameOver; ply++) g.play(g.randomLegalMove(rng));
  if (g.gameOver) continue;
  pw.setRoot(g); roots++;
  for (let p = 0; p < PATHS && !failures; p++) {
    const leaf = g.clone(), descent = [];
    const depth = Math.floor(rng.random() * 25);
    for (let d = 0; d < depth && !leaf.gameOver; d++) {
      const m = rng.random() < 0.08 ? PASS : leaf.randomLegalMove(rng);
      leaf.play(m); descent.push(m);
    }
    pw.descend(descent);
    leaves++; moves += descent.length;
    const d = diff(leaf);
    if (d) { failures++; console.log(`FAIL game ${gi} path ${p} (depth ${descent.length}): ${d}`); }
  }
}
console.log(`${roots} roots, ${leaves} descents (${moves} moves) compared: ${failures ? 'FAIL' : 'PASS: every cell, liberty count and state field equal'}`);

// Timings: setRoot on a mid-game position, descend at depths 0, 5, 10, 20.
const g = new Game2(SIZE, false);
for (let ply = 0; ply < 90; ply++) g.play(g.randomLegalMove(rng));
const time = (fn, reps) => { fn(); const t0 = performance.now(); for (let i = 0; i < reps; i++) fn(); return 1000 * (performance.now() - t0) / reps; };
console.log(`setRoot (${AREA - g.emptyCount} stones): ${time(() => pw.setRoot(g), 20000).toFixed(2)} us`);
for (const depth of [0, 5, 10, 20]) {
  const leaf = g.clone(), descent = [];
  for (let d = 0; d < depth; d++) { const m = leaf.randomLegalMove(rng); leaf.play(m); descent.push(m); }
  pw.setRoot(g);
  console.log(`descend depth ${String(depth).padStart(2)}: ${time(() => pw.descend(descent), 50000).toFixed(2)} us`);
}
process.exit(failures ? 1 : 0);
