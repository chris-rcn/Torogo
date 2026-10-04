'use strict';
// int-map.js — open-addressing hash maps from int32 keys.
//
// Key 0 is reserved as the empty-slot sentinel and must not be inserted.
// Collisions are resolved by triangular probing (skip++ each step), which
// visits every slot exactly once when capacity is a power of two.
//
// Two variants share the implementation and differ only in value storage:
//   makeIntMap(minCap)      — int32 values,   get() miss → -1
//   makeIntFloatMap(minCap) — float64 values, get() miss → undefined
//                             (Map-compatible: `m.get(k) ?? fallback`)
// and a third, for inference only:
//   makeIntFloat32Map(minCap) — float32 values interleaved with the keys in
//                             one buffer (8 bytes a slot, not 12), get() miss
//                             → undefined.  set() rounds to float32, so it is
//                             no table to train in: an SGD step below half the
//                             float32 spacing at the weight (~6e-8 at |w| 0.5)
//                             is lost.  Build it from a trained table.
//
// Classes, not closures: state lives in fields, so V8 inlines get() with
// plain field loads.  The closure version kept the arrays in mutable context
// slots (reassigned on resize) and ran ~1.6x slower in vpat evaluation
// (bench-vpat, 2026-10-04).  Each variant has its own get(), with the probe
// written into it and its miss value as a literal: one get shared by both
// value types, or a get calling a separate probe, each measured slower.
//
// Inspired by IntIntMap.java (fant.common).

(function () {

const MAX_FULLNESS = 0.5;

class IntMapBase {
  // At least minCap initial capacity, rounded up to a power of two.
  constructor(minCap) {
    minCap = minCap || 64;
    let cap = 1;
    while (cap < minCap) cap <<= 1;
    this.warnOnZero = true;
    this._alloc(cap);
  }

  _alloc(cap) {
    this.keys     = new Int32Array(cap);   // 0 means empty
    this.vals     = this._newVals(cap);
    this.mask     = cap - 1;
    this.count    = 0;
    this.resizeAt = (cap * MAX_FULLNESS) | 0;
  }

  // Returns slot index if key is present, ~slotIndex if slot is empty (insertion point).
  _probe(key) {
    const keys = this.keys, mask = this.mask;
    let i    = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask;
    let skip = 1;
    for (;;) {
      const k = keys[i];
      if (k === key) return i;
      if (k === 0)   return ~i;
      i = (i + skip) & mask;
      skip++;
    }
  }

  _resize() {
    const oldKeys = this.keys;
    const oldVals = this.vals;
    this._alloc(oldKeys.length << 1);
    for (let i = 0; i < oldKeys.length; i++) {
      const k = oldKeys[i];
      if (k !== 0) {
        const j = ~this._probe(k);
        this.keys[j] = k;
        this.vals[j] = oldVals[i];
        this.count++;
      }
    }
  }

  suppressZeroWarning() { this.warnOnZero = false; }

  // Inserts or updates key → val.
  set(key, val) {
    if (key === 0) { if (this.warnOnZero) console.error('int-map: key 0 is reserved (set)'); return; }
    let i = this._probe(key);
    if (i < 0) {
      i = ~i;
      this.keys[i] = key;
      if (++this.count >= this.resizeAt) { this.vals[i] = val; this._resize(); return; }
    }
    this.vals[i] = val;
  }

  get size() { return this.count; }

  clear() { this.keys.fill(0); this.count = 0; }

  forEach(fn) {
    const keys = this.keys, vals = this.vals;
    for (let i = 0; i < keys.length; i++) {
      if (keys[i] !== 0) fn(keys[i], vals[i]);
    }
  }

  clone() {
    const c = new this.constructor(this.keys.length);
    const keys = this.keys, vals = this.vals;
    for (let i = 0; i < keys.length; i++) {
      if (keys[i] !== 0) c.set(keys[i], vals[i]);
    }
    return c;
  }
}

// int32 → int32; get() returns -1 for absent keys.
class IntIntMap extends IntMapBase {
  _newVals(cap) { return new Int32Array(cap); }
  // Returns stored value, or -1 if not found.
  get(key) {
    if (key === 0) { if (this.warnOnZero) console.error('int-map: key 0 is reserved (get)'); return -1; }
    const keys = this.keys, mask = this.mask;
    let i    = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask;
    let skip = 1;
    for (;;) {
      const k = keys[i];
      if (k === key) return this.vals[i];
      if (k === 0)   return -1;
      i = (i + skip) & mask;
      skip++;
    }
  }
}

// int32 → float64; get() returns undefined for absent keys, like Map.
class IntFloatMap extends IntMapBase {
  _newVals(cap) { return new Float64Array(cap); }
  // Returns stored value, or undefined if not found.
  get(key) {
    if (key === 0) { if (this.warnOnZero) console.error('int-map: key 0 is reserved (get)'); return undefined; }
    const keys = this.keys, mask = this.mask;
    let i    = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask;
    let skip = 1;
    for (;;) {
      const k = keys[i];
      if (k === key) return this.vals[i];
      if (k === 0)   return undefined;
      i = (i + skip) & mask;
      skip++;
    }
  }
}

// int32 → float32, interleaved: slot i holds its key at keys[2i] and its
// value at vals[2i+1], two views of one buffer, so a hit reads one cache line.
// Same hash, probing and capacity rule as the others.
class IntFloat32Map {
  constructor(minCap) {
    minCap = minCap || 64;
    let cap = 1;
    while (cap < minCap) cap <<= 1;
    this.warnOnZero = true;
    this._alloc(cap);
  }

  _alloc(cap) {
    const buf     = new ArrayBuffer(cap * 8);
    this.keys     = new Int32Array(buf);     // keys at even indices; 0 means empty
    this.vals     = new Float32Array(buf);   // values at odd indices
    this.mask     = cap - 1;
    this.count    = 0;
    this.resizeAt = (cap * MAX_FULLNESS) | 0;
  }

  // Returns slot index if key is present, ~slotIndex if slot is empty (insertion point).
  _probe(key) {
    const keys = this.keys, mask = this.mask;
    let i    = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask;
    let skip = 1;
    for (;;) {
      const k = keys[i << 1];
      if (k === key) return i;
      if (k === 0)   return ~i;
      i = (i + skip) & mask;
      skip++;
    }
  }

  _resize() {
    const oldKeys = this.keys;
    const oldVals = this.vals;
    this._alloc(oldKeys.length);   // oldKeys.length is 2x the old slot count
    for (let i = 0; i < oldKeys.length; i += 2) {
      const k = oldKeys[i];
      if (k !== 0) {
        const j = ~this._probe(k) << 1;
        this.keys[j] = k;
        this.vals[j + 1] = oldVals[i + 1];
        this.count++;
      }
    }
  }

  suppressZeroWarning() { this.warnOnZero = false; }

  // Returns stored value, or undefined if not found.
  get(key) {
    if (key === 0) { if (this.warnOnZero) console.error('int-map: key 0 is reserved (get)'); return undefined; }
    const keys = this.keys, mask = this.mask;
    let i    = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask;
    let skip = 1;
    for (;;) {
      const k = keys[i << 1];
      if (k === key) return this.vals[(i << 1) | 1];
      if (k === 0)   return undefined;
      i = (i + skip) & mask;
      skip++;
    }
  }

  // Inserts or updates key → val (rounded to float32).
  set(key, val) {
    if (key === 0) { if (this.warnOnZero) console.error('int-map: key 0 is reserved (set)'); return; }
    let i = this._probe(key);
    if (i < 0) {
      i = ~i;
      this.keys[i << 1] = key;
      if (++this.count >= this.resizeAt) { this.vals[(i << 1) | 1] = val; this._resize(); return; }
    }
    this.vals[(i << 1) | 1] = val;
  }

  get size() { return this.count; }

  clear() { this.keys.fill(0); this.count = 0; }

  forEach(fn) {
    const keys = this.keys, vals = this.vals;
    for (let i = 0; i < keys.length; i += 2) {
      if (keys[i] !== 0) fn(keys[i], vals[i + 1]);
    }
  }

  clone() {
    const c = new IntFloat32Map(this.keys.length >> 1);
    this.forEach((k, v) => c.set(k, v));
    return c;
  }
}

function makeIntMap(minCap)        { return new IntIntMap(minCap); }
function makeIntFloatMap(minCap)   { return new IntFloatMap(minCap); }
function makeIntFloat32Map(minCap) { return new IntFloat32Map(minCap); }

const IntMap = { makeIntMap, makeIntFloatMap, makeIntFloat32Map };

if (typeof module !== 'undefined') {
  module.exports = IntMap;
  require('./int-map.test.js').runTests(IntMap);
} else {
  window.IntMap = IntMap;
}

})();
