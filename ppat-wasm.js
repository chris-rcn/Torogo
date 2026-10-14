'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// ppat-wasm.js — loads ppat-playout.wasm (c/wasm_ppat.c, built by
// c/build-wasm.sh) and hands it a ppat model.
//
//   const pw = PpatWasm.loadSync();            // Node: reads ppat-playout.wasm beside this file
//   const pw = await PpatWasm.load(url?);      // Node or browser (fetch; default 'ppat-playout.wasm')
//   pw.init(13);                               // board size
//   pw.setModel(PPat.loadWeights(file));       // a ppat-lib.js model object
//   pw.setRoot(game);                          // once per decision: a JS Game2
//   pw.descend(path, n);                       // per playout: the leaf = root + path
//   const v = pw.playout(seed, minPhase);      // then P(BLACK wins) in {0,1};
//   pw.trace(); pw.ppatMoves; pw.uniformMoves  // its RAVE trace (a view) and move counts
//
// The module's only imports are the five WASI calls its C runtime keeps for
// fatal-error messages and malloc: fd_write prints (stderr via console.error),
// proc_exit throws with the message printed before it, and fd_close /
// fd_seek / random_get are stubs.  Read pw.memory.buffer afresh after any
// call into the module: a memory growth replaces the ArrayBuffer.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

// The WASI preview1 calls the module imports, against `getMemory()`.
function wasiShim(getMemory) {
  const ERRNO_SUCCESS = 0, ERRNO_BADF = 8, ERRNO_SPIPE = 70;
  let lastErr = '';
  const decoder = new TextDecoder();
  const pending = { 1: '', 2: '' };
  return {
    lastError: () => lastErr,
    imports: {
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
        for (const line of lines) {
          if (fd === 2) { lastErr = line; console.error(line); } else console.log(line);
        }
        view.setUint32(nwrittenPtr, written, true);
        return ERRNO_SUCCESS;
      },
      proc_exit(code) {
        throw new Error(`ppat-wasm: module exited (${code})${lastErr ? ': ' + lastErr : ''}`);
      },
      fd_close() { return ERRNO_SUCCESS; },
      fd_seek() { return ERRNO_SPIPE; },
      random_get(buf, len) {
        const bytes = new Uint8Array(getMemory().buffer, buf, len);
        if (typeof crypto !== 'undefined' && crypto.getRandomValues) crypto.getRandomValues(bytes);
        else for (let i = 0; i < len; i++) bytes[i] = (Math.random() * 256) | 0;
        return ERRNO_SUCCESS;
      },
    },
  };
}

// Wrap an instantiated module: the reactor's _initialize, then the API.
function wrap(instance, shim) {
  const X = instance.exports;
  X._initialize();
  let weightsPtr = 0, boardArea = 0, traceView = null, countsView = null;
  const counts = () => {
    if (countsView === null || countsView.buffer !== X.memory.buffer)
      countsView = new Int32Array(X.memory.buffer, X.wasm_counts_buf(), 2);
    return countsView;
  };
  return {
    exports: X,
    memory: X.memory,
    init(N) { X.wasm_init(N); boardArea = N * N; traceView = null; },
    // Copy a ppat-lib.js model into the module and adopt its encoding; throws
    // when the weight count does not match the encoding's expected total.
    setModel(model) {
      const w = model.weights, n = w.length;
      if (weightsPtr) X.wasm_free(weightsPtr);
      weightsPtr = X.wasm_alloc(4 * n);
      if (!weightsPtr) throw new Error(`ppat-wasm: cannot allocate ${n} weights`);
      new Float32Array(X.memory.buffer, weightsPtr, n).set(w);
      const expected = X.wasm_set_model(weightsPtr, n, model.adjLib, model.phaseCount, model.t12mode,
                                        model.selfAtari ? 1 : 0, model.atariN, model.xaN, model.captureN,
                                        model.csN, model.earlyPass ? 1 : 0, model.passWeight || 0);
      if (expected !== 0) throw new Error(`ppat-wasm: model has ${n} weights, its encoding expects ${expected}`);
    },
    // The search root, once per decision: a JS Game2 (cells plus its state)
    // rebuilt in the module.  Throws on a position not reachable by legal play.
    setRoot(game) {
      const n = game.N * game.N;
      new Int8Array(X.memory.buffer, X.wasm_cells_buf(), n).set(game.cells.subarray(0, n));
      const rc = X.wasm_set_root(game.current, game.ko, game.lastMove, game.koStone[0], game.koStone[2],
                                 game.consecutivePasses, game.moveCount);
      if (rc !== 0) throw new Error(`ppat-wasm: setRoot rejected the stone at ${-rc - 1} (unreachable position)`);
    },
    // A leaf: the root with path[0..n) played (the tree descent).  Throws on an
    // illegal move or a path longer than the module's buffer.
    descend(path, n = path.length) {
      if (n > X.wasm_path_cap()) throw new Error(`ppat-wasm: descent path of ${n} moves exceeds ${X.wasm_path_cap()}`);
      const buf = new Int32Array(X.memory.buffer, X.wasm_path_buf(), n);
      for (let i = 0; i < n; i++) buf[i] = path[i];
      const rc = X.wasm_descend(n);
      if (rc !== 0) throw new Error(`ppat-wasm: descent move ${-rc - 1} (${path[-rc - 1]}) is illegal`);
    },
    // One playout from the leaf (descend first): puct-trunc's untruncated loop
    // with uniform moves below minPhase.  seed: a 32-bit integer from the
    // agent's rng.  Sets ppatMoves / uniformMoves; trace() is its RAVE trace.
    playout(seed, minPhase) {
      const v = X.wasm_playout(seed >>> 0, minPhase);
      if (v < 0) throw new Error('ppat-wasm: playout without a model (setModel first)');
      const c = counts();
      this.ppatMoves = c[0];
      this.uniformMoves = c[1];
      return v;
    },
    ppatMoves: 0,
    uniformMoves: 0,
    // The last playout's colour-signed first-occupancy RAVE trace: a
    // Float32Array view of the module's buffer, valid until the next playout
    // (and re-taken whenever the memory has grown).
    trace() {
      if (traceView === null || traceView.buffer !== X.memory.buffer)
        traceView = new Float32Array(X.memory.buffer, X.wasm_trace_buf(), boardArea);
      return traceView;
    },
    lastError: shim.lastError,
  };
}

function imports(shim) { return { wasi_snapshot_preview1: shim.imports }; }

// Synchronous: Node (or a small module where the host allows sync compilation).
function instantiateSync(bytes) {
  let memory = null;
  const shim = wasiShim(() => memory);
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports(shim));
  memory = instance.exports.memory;
  return wrap(instance, shim);
}

async function instantiate(bytes) {
  let memory = null;
  const shim = wasiShim(() => memory);
  const { instance } = await WebAssembly.instantiate(bytes, imports(shim));
  memory = instance.exports.memory;
  return wrap(instance, shim);
}

function loadSync(file) {
  if (!_isNode) throw new Error('ppat-wasm: loadSync is Node-only; use load() in the browser');
  const path = require('path');
  return instantiateSync(require('fs').readFileSync(file || path.join(__dirname, 'ppat-playout.wasm')));
}

async function load(url) {
  if (_isNode) return loadSync(url);
  const resp = await fetch(url || 'ppat-playout.wasm');
  if (!resp.ok) throw new Error(`ppat-wasm: fetch ${url || 'ppat-playout.wasm'} failed (${resp.status})`);
  return instantiate(await resp.arrayBuffer());
}

const PpatWasm = { load, loadSync, instantiate, instantiateSync };
if (typeof module !== 'undefined') module.exports = PpatWasm;
else window.PpatWasm = PpatWasm;

})();
