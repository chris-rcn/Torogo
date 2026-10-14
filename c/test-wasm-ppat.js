'use strict';
// test-wasm-ppat.js — the model hand-off (ppat-wasm.js -> c/wasm_ppat.c) must
// play exactly what the file loader plays.  Runs bench_ppat.c's playout loop in
// two WebAssembly builds from one compiler and flag set — ppat-playout-test.wasm
// (weights copied in from ppat-lib.js's loadWeights) and bench_ppat.wasm (the C
// file loader, run under node:wasi) — on the same start positions, seed and
// playout count, and compares their move checksums.  Build both first with
// c/build-wasm.sh.
//
//   node c/test-wasm-ppat.js [--model out/ppat-data-1623136.js] [--games 30] [--playouts 2] [--seed 1]

const fs = require('fs');
const os = require('os');
const path = require('path');
const { WASI } = require('node:wasi');
const R = path.join(__dirname, '..');
const Util = require(path.join(R, 'util.js'));
const { Game2, PASS } = require(path.join(R, 'game2.js'));
const { makeRng } = require(path.join(R, 'xorshift.js'));
const PPat = require(path.join(R, 'ppat-lib.js'));
const PpatWasm = require(path.join(R, 'ppat-wasm.js'));

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'games', 'playouts', 'seed']);
if (opts.help) { console.log('usage: node c/test-wasm-ppat.js [--model F] [--games 30] [--playouts 2] [--seed 1]'); process.exit(0); }
const MODEL    = path.resolve(opts.model || path.join(R, 'out', 'ppat-data-1623136.js'));
const GAMES    = parseInt(opts.games || '30', 10);
const PLAYOUTS = parseInt(opts.playouts || '2', 10);
const SEED     = parseInt(opts.seed || '1', 10);
const SIZE = 13;

// Start positions: every 4th position at fullness >= 0.55 of GAMES uniform-random games.
const rng = makeRng(SEED);
const starts = [];
for (let gi = 0; gi < GAMES; gi++) {
  const g = new Game2(SIZE, false), moves = [];
  for (let ply = 0; !g.gameOver && ply < 4 * SIZE * SIZE; ply++) {
    if (g.phase() >= 0.55 && ply % 4 === 0) starts.push(moves.slice());
    const m = g.randomLegalMove(rng);
    g.play(m); moves.push(m);
  }
}

// 1. The hand-off module.
const pw = PpatWasm.loadSync(path.join(__dirname, 'ppat-playout-test.wasm'));
pw.init(SIZE);
pw.setModel(PPat.loadWeights(MODEL));
const flat = starts.flat(), X = pw.exports;
const movesPtr = X.wasm_alloc(4 * flat.length), offPtr = X.wasm_alloc(4 * (starts.length + 1));
new Int32Array(pw.memory.buffer, movesPtr, flat.length).set(flat);
const off = new Int32Array(pw.memory.buffer, offPtr, starts.length + 1);
for (let i = 0, o = 0; i < starts.length; i++) { off[i] = o; o += starts[i].length; off[i + 1] = o; }
const handoff = (X.wasm_test_bench(movesPtr, offPtr, starts.length, SIZE, PLAYOUTS, SEED) >>> 0).toString(16).padStart(8, '0');

// 2. bench_ppat.wasm on the same starts, its stdout to a temp file.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'test-wasm-ppat-'));
const startsFile = path.join(tmp, 'starts.txt'), outFile = path.join(tmp, 'out.txt');
fs.writeFileSync(startsFile, starts.map(s => SIZE + ' ' + s.map(m => m === PASS ? '-' : m).join(',')).join('\n') + '\n');
const outFd = fs.openSync(outFile, 'w');
const wasi = new WASI({ version: 'preview1', stdout: outFd,
  args: ['bench_ppat', '--model', MODEL, '--starts', startsFile, '--playouts', String(PLAYOUTS), '--seed', String(SEED), '--incremental'],
  preopens: { [tmp]: tmp, [path.dirname(MODEL)]: path.dirname(MODEL) } });
const inst = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(path.join(__dirname, 'bench_ppat.wasm'))), wasi.getImportObject());
const code = wasi.start(inst);
fs.closeSync(outFd);
const out = fs.readFileSync(outFile, 'utf8');
fs.rmSync(tmp, { recursive: true });
const m = out.match(/checksum: ([0-9a-f]{8})/);
if (code !== 0 || !m) { console.error(`bench_ppat.wasm failed (exit ${code}):\n${out}`); process.exit(1); }

console.log(`model: ${path.basename(MODEL)}, start positions: ${starts.length}, playouts each: ${PLAYOUTS}, seed: ${SEED}`);
console.log(`hand-off module: ${handoff}   file-loader module: ${m[1]}`);
console.log(handoff === m[1] ? 'PASS: identical move checksums' : 'FAIL: checksums differ');
process.exit(handoff === m[1] ? 0 : 1);
