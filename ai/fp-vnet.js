'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// fp-vnet: featurepol filters the root moves, the value network picks one.
// Featurepol's top VN_WIDTH moves are the candidates; each is played and the
// child valued by the net (cnn.h) in WebAssembly — vnet-eval.wasm via
// vnet-wasm.js (c/wasm_vnet.c) — as 1 - V(child), the mover's chance after the
// move.  The best is played (ties: featurepol's order); PASS when featurepol
// lists no move.  This is the trainers' MD 1-ply test as a player
// (--test-fp-width = VN_WIDTH, the same featurepol model by default).
// valueB is the net's value of the position itself.
//
// Config (cfg reader, slot-aware):
//   VN_WIDTH       featurepol candidates; 0 = every one            (default 7)
//   VN_FPOL_DATA   featurepol weights       (default ref/ref-fp-stone8AdjLib3.js)
//   VN_MODEL       the net, a VNET1 file     (default out/vnet-rfs-ptw1000-lad-6x32-best.bin)
// In the browser the page loads the pieces first: window.vnetWasm = await
// VnetWasm.load(url), window.vnetModel = the VNET1 file's bytes (Uint8Array),
// and window.featurepolModel.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { game3FromGame2 } = Util.load('./game3.js', 'Game3');
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');
const VnetWasm = Util.load('./vnet-wasm.js', 'VnetWasm');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const VN_WIDTH = cfg.int('VN_WIDTH', 7);
  if (!(VN_WIDTH >= 0)) throw new Error(`fp-vnet: VN_WIDTH must be >= 0, got ${VN_WIDTH}`);

  const fpPath = _isNode ? cfg.str('VN_FPOL_DATA', require('path').join(__dirname, '..', 'ref', 'ref-fp-stone8AdjLib3.js')) : undefined;
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'fp-vnet', path: fpPath });

  const modelPath = _isNode ? cfg.str('VN_MODEL', require('path').join(__dirname, '..', 'out', 'vnet-rfs-ptw1000-lad-6x32-best.bin')) : null;
  const modelBytes = _isNode ? new Uint8Array(require('fs').readFileSync(modelPath))
                             : (typeof window !== 'undefined' && window.vnetModel) || null;
  if (!modelBytes) throw new Error('fp-vnet: window.vnetModel is not set — the VNET1 file\'s bytes (Uint8Array)');
  const vw = _isNode ? VnetWasm.loadSync() : (typeof window !== 'undefined' && window.vnetWasm) || null;
  if (!vw) throw new Error('fp-vnet: window.vnetWasm is not set — load it first: window.vnetWasm = await VnetWasm.load(url)');
  const nParams = vw.setModel(modelBytes);
  let vwN = 0;   // board size the module is initialised for

  console.error(`fp-vnet[${cfg.slot != null ? cfg.slot : '-'}]: width: ${VN_WIDTH === 0 ? 'all' : VN_WIDTH}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vnet: ${_isNode ? require('path').basename(modelPath) : 'window.vnetModel'} (${nParams} parameters)`);

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
    const cand = new Int32Array(VN_WIDTH || game.N * game.N);
    const k = fpRanked(game, cand);
    if (k === 0) return { move: PASS };
    vw.setRoot(game);
    const vals = vw.evalMoves(cand, k);
    let best = 0;
    for (let j = 1; j < k; j++) if (vals[j] > vals[best]) best = j;
    return { move: cand[best], info: `ply=${vals[best].toFixed(3)} w${VN_WIDTH}` };
  }

  return { getMove, valueB };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
else { window.create = create; window.getMove = (g, b, o) => _def().getMove(g, b, o); window.valueB = (g, o) => _def().valueB(g, o); }

})();
