'use strict';

// game3.js — Game2 extended with move undo.
//
// API additions over Game2:
//   game.play(idx)      → true/false  (same as Game2, but records an undo entry)
//   game.undo()         → true/false  (false when stack is empty)
//   game.clone()        → Game3       (shallow-copies board state; undo stack cleared)
//   Game3.from(game2)   → Game3       (promote a Game2 snapshot to a Game3)
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
// by restoring _nextGid; their array slots are zeroed to prevent stale |=
// corruption when the same gid slot is later reused (multi-branch search).

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
      // Zero any gid slots that were allocated during this play.  _place() uses
      // |= when writing _sw/_lw for a new gid, so stale data in a freed slot
      // would corrupt a subsequent play that reallocates the same gid (which
      // can happen when game3 is used for multi-branch search in ladder3).
      const W  = this._W;
      const ng = snap.prevNextGid;
      for (let gid = ng; gid < this._nextGid; gid++) {
        const b = gid * W;
        for (let wi = 0; wi < W; wi++) { this._sw[b + wi] = 0; this._lw[b + wi] = 0; }
        this._ss[gid] = 0; this._ls[gid] = 0; this._gc[gid] = 0;
      }

      this._nextGid = ng;
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

  // ── Clone ─────────────────────────────────────────────────────────────────

  // Returns a Game3 with the same board position and an empty undo stack.
  clone() {
    const g = super.clone();                    // Game2 object
    Object.setPrototypeOf(g, Game3.prototype);  // upgrade to Game3
    g._undoStack = [];
    return g;
  }

  // ── Static factory ────────────────────────────────────────────────────────

  // Promote an existing Game2 snapshot to a Game3 (zero-copy for typed arrays).
  static from(game2) {
    const g = game2.clone();                    // Game2 object (full typed-array copy)
    Object.setPrototypeOf(g, Game3.prototype);
    g._undoStack = [];
    return g;
  }
}

module.exports = { Game3, PASS, BLACK, WHITE };
