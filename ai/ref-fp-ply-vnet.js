'use strict';

// ref-fp-ply-vnet — frozen reference: ai/fp-ply-vnet.js as of 2026-10-15
// (ae0049e) at VN_WIDTH 7, with the files frozen as ref/ref-fp-ply-vnet-vnet.bin
// (out/vnet-sweep-frozen.bin: vnet-rfs-ptw1000-lad-6x32-best as written at
// 13:37:12, MD-test elo 1719 at width 7), ref/ref-fp-ply-vnet-fp.js
// (ref/ref-fp-stone8AdjLib3.js) and the network module as
// ref/ref-fp-ply-vnet-eval.wasm (vnet-eval.wasm, built from c/ at ae0049e).
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.
//
// fp-ply-vnet: featurepol's top 7 moves are the candidates; each is played and
// the child valued by the value network in WebAssembly (vnet-wasm.js) as
// 1 - V(child), the mover's chance after the move.  The best is played (ties:
// featurepol's order); PASS when featurepol lists no move.  valueB is the
// net's value of the position itself.

const path = require('path');
const fs = require('fs');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const FeaturePol = require('../featurepol-lib.js');
const VnetWasm = require('../vnet-wasm.js');

// ── Hardcoded configuration ─────────────────────────────────────────────────
const WIDTH      = 7;                                                           // VN_WIDTH
const FPOL_PATH  = path.join(__dirname, '..', 'ref', 'ref-fp-ply-vnet-fp.js');   // VN_FPOL_DATA
const MODEL_PATH = path.join(__dirname, '..', 'ref', 'ref-fp-ply-vnet-vnet.bin'); // VN_MODEL
const WASM_PATH  = path.join(__dirname, '..', 'ref', 'ref-fp-ply-vnet-eval.wasm');

function create() {
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'ref-fp-ply-vnet', path: FPOL_PATH });
  const vw = VnetWasm.loadSync(WASM_PATH);
  const nParams = vw.setModel(new Uint8Array(fs.readFileSync(MODEL_PATH)));
  let vwN = 0;   // board size the module is initialised for

  console.error(`ref-fp-ply-vnet: width: ${WIDTH}, fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vnet: ${path.basename(MODEL_PATH)} (${nParams} parameters), module: ${path.basename(WASM_PATH)}`);

  let fpState = null, fpScores = null, stateN = 0;
  function ensureState(N) {
    if (vwN !== N) { vw.init(N); vwN = N; }
    if (stateN === N) return;
    fpState  = FeaturePol.createState(N, fpWeights.spec);
    fpScores = new Float64Array(N * N + 1);
    stateN   = N;
  }

  // Featurepol's candidates of g, best-first (stable on ties), into `out` (at
  // most out.length of them).  Returns the count.
  function fpRanked(g, out) {
    FeaturePol.extractFeatures(g, fpState, fpWeights, fpWeights.spec.needsLadder ? game3FromGame2(g) : undefined);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => fpScores[b] - fpScores[a]);
    const k = Math.min(out.length, n);
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return k;
  }

  // P(BLACK wins): the net's value of `game` (the exact result when it is over).
  function valueB(game) {
    if (game.gameOver) return game.estimateWinner() === BLACK ? 1 : 0;
    ensureState(game.N);
    vw.setRoot(game);
    const v = vw.evalRoot();
    return game.current === BLACK ? v : 1 - v;
  }

  function getMove(game) {
    if (game.gameOver) return { move: PASS };
    ensureState(game.N);
    const cand = new Int32Array(WIDTH);
    const k = fpRanked(game, cand);
    if (k === 0) return { move: PASS };
    vw.setRoot(game);
    const vals = vw.evalMoves(cand, k);
    let best = 0;
    for (let j = 1; j < k; j++) if (vals[j] > vals[best]) best = j;
    return { move: cand[best], info: `ply=${vals[best].toFixed(3)} w${WIDTH}` };
  }

  return { getMove, valueB };
}

let _default = null;
function _def() { return _default || (_default = create()); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
