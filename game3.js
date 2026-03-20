'use strict';

// game3.js — Game2 extended with move undo.
//
// API additions over Game2:
//   game.play(idx)  → true/false  (same as Game2, but records an undo entry)
//   game.undo()     → true/false  (false when stack is empty)
//
// Undo strategy: snapshot the state that changes on each play() call.
//
// For a PASS only scalar fields change, so only those are saved.
// For a PLACE the snapshot also copies:
//   - cells    (Int8Array, N*N bytes)
//   - _gid     (Int32Array, N*N words)
//   - slices of _gc / _sw / _ss / _lw / _ls up to the current _nextGid
//
// On a 13×13 board with ~100 allocated groups the snapshot is ~6 KB.
// Newly allocated groups (gid >= prevNextGid) are discarded on undo simply
// by restoring _nextGid; their array slots are irrelevant.

const { Game2, PASS, BLACK, WHITE } = require('./game2.js');

class Game3 extends Game2 {
  constructor(size) {
    super(size);
    this._undoStack = [];
  }

  // ── Reset ─────────────────────────────────────────────────────────────────

  reset() {
    super.reset();
    this._undoStack.length = 0;
  }

  // ── Play with undo recording ───────────────────────────────────────────────

  play(idx) {
    if (this.gameOver) return false;
    const isPass = idx === PASS;
    if (!isPass && !this.isLegal(idx)) return false;

    // Build the undo record before mutating anything.
    const snap = {
      isPass,
      prevCurrent: this.current,
      prevKo:      this.ko,
      prevCP:      this.consecutivePasses,
      prevGO:      this.gameOver,
      prevMC:      this.moveCount,
    };

    if (!isPass) {
      const ng = this._nextGid;
      const W  = this._W;
      snap.prevNextGid = ng;
      snap.prevCells   = new Int8Array(this.cells);
      snap.prevGid     = new Int32Array(this._gid);
      snap.prevGc      = this._gc.slice(0, ng);
      snap.prevSs      = this._ss.slice(0, ng);
      snap.prevLs      = this._ls.slice(0, ng);
      snap.prevSw      = this._sw.slice(0, ng * W);
      snap.prevLw      = this._lw.slice(0, ng * W);
    }

    // Delegate to Game2 (legality already verified; the internal re-check is harmless).
    super.play(idx);
    this._undoStack.push(snap);
    return true;
  }

  // ── Undo ─────────────────────────────────────────────────────────────────

  undo() {
    if (this._undoStack.length === 0) return false;
    const snap = this._undoStack.pop();

    this.current           = snap.prevCurrent;
    this.ko                = snap.prevKo;
    this.consecutivePasses = snap.prevCP;
    this.gameOver          = snap.prevGO;
    this.moveCount         = snap.prevMC;

    if (!snap.isPass) {
      this._nextGid = snap.prevNextGid;
      this.cells.set(snap.prevCells);
      this._gid.set(snap.prevGid);
      this._gc.set(snap.prevGc);
      this._ss.set(snap.prevSs);
      this._ls.set(snap.prevLs);
      this._sw.set(snap.prevSw);
      this._lw.set(snap.prevLw);
    }

    return true;
  }
}

module.exports = { Game3, PASS, BLACK, WHITE };
