'use strict';

// int-map.js — open-addressing hash map from int32 → int32.
//
// Key 0 is reserved as the empty-slot sentinel and must not be inserted.
// Collisions are resolved by triangular probing (skip++ each step), which
// visits every slot exactly once when capacity is a power of two.
//
// Inspired by IntIntMap.java (fant.common).

(function () {

const MAX_FULLNESS = 0.5;

// Returns a new IntMap with at least minCap initial capacity.
function makeIntMap(minCap) {
  minCap = minCap || 64;
  let cap = 1;
  while (cap < minCap) cap <<= 1;

  let keys;     // Int32Array — 0 means empty
  let vals;     // Int32Array
  let mask;
  let count;
  let resizeAt;

  function alloc() {
    keys     = new Int32Array(cap);
    vals     = new Int32Array(cap);
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

  return {
    // Returns stored value, or -1 if not found.
    get(key) {
      if (key === 0) return -1;
      const i = probe(key);
      return i < 0 ? -1 : vals[i];
    },

    // Inserts or updates key → val.
    set(key, val) {
      if (key === 0) return;
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
      const c = makeIntMap(cap);
      for (let i = 0; i < cap; i++) {
        if (keys[i] !== 0) c.set(keys[i], vals[i]);
      }
      return c;
    },
  };
}

const IntMap = { makeIntMap };
if (typeof module !== 'undefined') {
  module.exports = IntMap;
  require('./int-map.test.js').runTests(IntMap);
} else {
  window.IntMap = IntMap;
}

})();
