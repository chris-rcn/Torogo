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
//
// Inspired by IntIntMap.java (fant.common).

(function () {

const MAX_FULLNESS = 0.5;

// Returns a new map with at least minCap initial capacity, ValsArray value
// storage, and missValue returned by get() for absent keys.
function makeMap(minCap, ValsArray, missValue) {
  minCap = minCap || 64;
  let cap = 1;
  while (cap < minCap) cap <<= 1;

  let keys;     // Int32Array — 0 means empty
  let vals;     // ValsArray
  let mask;
  let count;
  let resizeAt;

  function alloc() {
    keys     = new Int32Array(cap);
    vals     = new ValsArray(cap);
    mask     = cap - 1;
    count    = 0;
    resizeAt = (cap * MAX_FULLNESS) | 0;
  }
  alloc();

  // Returns slot index if key is present, ~slotIndex if slot is empty (insertion point).
  function probe(key) {
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

  function resize() {
    const oldKeys = keys;
    const oldVals = vals;
    cap <<= 1;
    alloc();
    for (let i = 0; i < oldKeys.length; i++) {
      const k = oldKeys[i];
      if (k !== 0) {
        const j = ~probe(k);
        keys[j] = k;
        vals[j] = oldVals[i];
        count++;
      }
    }
  }

  let warnOnZero = true;

  return {
    suppressZeroWarning() { warnOnZero = false; },

    // Returns stored value, or missValue if not found.
    get(key) {
      if (key === 0) { if (warnOnZero) console.error('int-map: key 0 is reserved (get)'); return missValue; }
      const i = probe(key);
      return i < 0 ? missValue : vals[i];
    },

    // Inserts or updates key → val.
    set(key, val) {
      if (key === 0) { if (warnOnZero) console.error('int-map: key 0 is reserved (set)'); return; }
      let i = probe(key);
      if (i < 0) {
        i = ~i;
        keys[i] = key;
        if (++count >= resizeAt) { vals[i] = val; resize(); return; }
      }
      vals[i] = val;
    },

    get size() { return count; },

    clear() { keys.fill(0); count = 0; },

    forEach(fn) {
      for (let i = 0; i < cap; i++) {
        if (keys[i] !== 0) fn(keys[i], vals[i]);
      }
    },

    clone() {
      const c = makeMap(cap, ValsArray, missValue);
      for (let i = 0; i < cap; i++) {
        if (keys[i] !== 0) c.set(keys[i], vals[i]);
      }
      return c;
    },
  };
}

// int32 → int32; get() returns -1 for absent keys.
function makeIntMap(minCap) {
  return makeMap(minCap, Int32Array, -1);
}

// int32 → float64; get() returns undefined for absent keys, like Map.
function makeIntFloatMap(minCap) {
  return makeMap(minCap, Float64Array, undefined);
}

const IntMap = { makeIntMap, makeIntFloatMap };
if (typeof module !== 'undefined') {
  module.exports = IntMap;
  require('./int-map.test.js').runTests(IntMap);
} else {
  window.IntMap = IntMap;
}

})();
