'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// vnet-wasm.js — loads vnet-eval.wasm (c/wasm_vnet.c, built by
// c/build-wasm.sh) and hands it a value net (a VNET1 file's bytes, as
// train_vnet_supervised / train_vnet_td save them).
//
//   const vw = VnetWasm.loadSync();            // Node: reads vnet-eval.wasm beside this file
//   const vw = await VnetWasm.load(url?);      // Node or browser (fetch; default 'vnet-eval.wasm')
//   vw.init(13);                               // board size
//   vw.setModel(bytes);                        // a VNET1 file's contents (Uint8Array)
//   vw.setRoot(game);                          // once per decision: a JS Game2
//   const v = vw.evalRoot();                   // V(root): P(the side to move wins)
//   const vals = vw.evalMoves(moves, n);       // P(the mover wins) after each move (a view)
//
// Fatal errors in the C code call the env.torogo_fatal import (c/fatal.h),
// which fills in the message's %d's and throws; the WASI calls the C runtime
// imports get a minimal shim (fd_write prints, the rest are stubs).

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

function wasiShim(getMemory) {
  const ERRNO_SUCCESS = 0, ERRNO_BADF = 8, ERRNO_SPIPE = 70;
  const decoder = new TextDecoder();
  const pending = { 1: '', 2: '' };
  return {
    fd_write(fd, iovs, iovsLen, nwrittenPtr) {
      if (fd !== 1 && fd !== 2) return ERRNO_BADF;
      const view = new DataView(getMemory().buffer);
      let written = 0, text = '';
      for (let i = 0; i < iovsLen; i++) {
        const ptr = view.getUint32(iovs + 8 * i, true), len = view.getUint32(iovs + 8 * i + 4, true);
        text += decoder.decode(new Uint8Array(getMemory().buffer, ptr, len));
        written += len;
      }
      pending[fd] += text;
      const lines = pending[fd].split('\n');
      pending[fd] = lines.pop();
      for (const line of lines) (fd === 2 ? console.error : console.log)(line);
      view.setUint32(nwrittenPtr, written, true);
      return ERRNO_SUCCESS;
    },
    proc_exit(code) { throw new Error(`vnet-wasm: module exited (${code})`); },
    fd_close() { return ERRNO_SUCCESS; },
    fd_seek() { return ERRNO_SPIPE; },
    random_get(buf, len) {
      const bytes = new Uint8Array(getMemory().buffer, buf, len);
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) crypto.getRandomValues(bytes);
      else for (let i = 0; i < len; i++) bytes[i] = (Math.random() * 256) | 0;
      return ERRNO_SUCCESS;
    },
  };
}

// c/fatal.h's hook: fmt is a C string in the module's memory with up to
// three %d's for a, b, c.
function fatalImport(getMemory) {
  return function torogo_fatal(fmtPtr, a, b, c) {
    const mem = new Uint8Array(getMemory().buffer);
    let end = fmtPtr;
    while (mem[end] !== 0) end++;
    const args = [a, b, c];
    const msg = new TextDecoder().decode(mem.subarray(fmtPtr, end)).replace(/%d/g, () => String(args.shift())).trim();
    throw new Error(`vnet-wasm: ${msg}`);
  };
}

const MODEL_ERRORS = { '-1': 'not a VNET1 file (bad magic or size)', '-2': 'a plane count other than this build\'s',
                       '-3': 'a parameter count that does not match its channels / blocks / head' };

function wrap(instance) {
  const X = instance.exports;
  X._initialize();
  let modelPtr = 0, valuesView = null;
  return {
    exports: X,
    memory: X.memory,
    init(N) { X.wasm_vnet_init(N); },
    // Adopt a VNET1 file's bytes; throws on a malformed file or one built for
    // other input planes.
    setModel(bytes) {
      if (modelPtr) X.wasm_free(modelPtr);
      modelPtr = X.wasm_alloc(bytes.length);
      if (!modelPtr) throw new Error(`vnet-wasm: cannot allocate ${bytes.length} model bytes`);
      new Uint8Array(X.memory.buffer, modelPtr, bytes.length).set(bytes);
      const rc = X.wasm_vnet_set_model(modelPtr, bytes.length);
      X.wasm_free(modelPtr); modelPtr = 0;   /* the module keeps its own copy of the parameters */
      if (rc !== 0) throw new Error(`vnet-wasm: model rejected: ${MODEL_ERRORS[rc] || rc}`);
      return X.wasm_vnet_param_count();
    },
    // The position, once per decision: a JS Game2 (cells plus its state)
    // rebuilt in the module.  Throws on a position not reachable by legal play.
    setRoot(game) {
      const n = game.N * game.N;
      new Int8Array(X.memory.buffer, X.wasm_vnet_cells_buf(), n).set(game.cells.subarray(0, n));
      const rc = X.wasm_vnet_set_root(game.current, game.ko, game.lastMove, game.koStone[0], game.koStone[2],
                                      game.consecutivePasses, game.moveCount);
      if (rc !== 0) throw new Error(`vnet-wasm: setRoot rejected the stone at ${-rc - 1} (unreachable position)`);
    },
    // V(root): P(the side to move wins).
    evalRoot() {
      const v = X.wasm_vnet_eval_root();
      if (v < 0) throw new Error('vnet-wasm: evalRoot without a model (init and setModel first)');
      return v;
    },
    // P(the root's mover wins) after each of moves[0..n): 1 - V(child), or the
    // exact result when the move ends the game; -1 for an illegal move.  A
    // Float32Array view, valid until the next call.
    evalMoves(moves, n = moves.length) {
      if (n > X.wasm_vnet_max_moves()) throw new Error(`vnet-wasm: ${n} moves exceed ${X.wasm_vnet_max_moves()}`);
      new Int32Array(X.memory.buffer, X.wasm_vnet_moves_buf(), n).set(moves.subarray ? moves.subarray(0, n) : moves.slice(0, n));
      if (X.wasm_vnet_eval_moves(n) !== 0) throw new Error('vnet-wasm: evalMoves without a model (init and setModel first)');
      if (valuesView === null || valuesView.buffer !== X.memory.buffer)
        valuesView = new Float32Array(X.memory.buffer, X.wasm_vnet_values_buf(), X.wasm_vnet_max_moves());
      return valuesView.subarray(0, n);
    },
  };
}

function imports(getMemory) {
  return { wasi_snapshot_preview1: wasiShim(getMemory), env: { torogo_fatal: fatalImport(getMemory) } };
}

function instantiateSync(bytes) {
  let memory = null;
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports(() => memory));
  memory = instance.exports.memory;
  return wrap(instance);
}

async function instantiate(bytes) {
  let memory = null;
  const { instance } = await WebAssembly.instantiate(bytes, imports(() => memory));
  memory = instance.exports.memory;
  return wrap(instance);
}

function loadSync(file) {
  if (!_isNode) throw new Error('vnet-wasm: loadSync is Node-only; use load() in the browser');
  const path = require('path');
  return instantiateSync(require('fs').readFileSync(file || path.join(__dirname, 'vnet-eval.wasm')));
}

async function load(url) {
  if (_isNode) return loadSync(url);
  const resp = await fetch(url || 'vnet-eval.wasm');
  if (!resp.ok) throw new Error(`vnet-wasm: fetch ${url || 'vnet-eval.wasm'} failed (${resp.status})`);
  return instantiate(await resp.arrayBuffer());
}

const VnetWasm = { load, loadSync, instantiate, instantiateSync };
if (typeof module !== 'undefined') module.exports = VnetWasm;
else window.VnetWasm = VnetWasm;

})();
