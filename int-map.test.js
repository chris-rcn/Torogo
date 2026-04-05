'use strict';

// Tests for int-map.js.
// Called by int-map.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.

function runTests({ makeIntMap }) {
  let failures = 0;

  function check(cond, msg) {
    if (!cond) { failures++; console.error('FAIL [int-map]:', msg); }
  }

  // ── Basic get / set ────────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    check(m.size === 0,       'empty map: size is 0');
    check(m.get(1) === -1,    'empty map: get returns -1');

    m.set(1, 42);
    check(m.get(1) === 42,    'set/get: value stored');
    check(m.size === 1,       'size increments after insert');

    m.set(1, 99);
    check(m.get(1) === 99,    'set/get: value overwritten');
    check(m.size === 1,       'size unchanged after update');

    m.set(2, 0);
    check(m.get(2) === 0,     'set/get: value 0 stored correctly');
    check(m.size === 2,       'size increments for second key');

    check(m.get(3) === -1,    'get: miss returns -1');
  }

  // ── Key 0 is ignored ──────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    m.set(0, 7);
    check(m.get(0) === -1,    'key 0: get returns -1 (not insertable)');
    check(m.size === 0,       'key 0: size unchanged');
  }

  // ── Negative keys ─────────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    m.set(-1, 10);
    m.set(-100, 20);
    m.set(0x80000000 | 0, 30);   // MIN_INT as signed
    check(m.get(-1)            === 10, 'negative key -1');
    check(m.get(-100)          === 20, 'negative key -100');
    check(m.get(0x80000000|0)  === 30, 'MIN_INT key');
    check(m.size === 3,               'size after negative key inserts');
  }

  // ── clear ──────────────────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    m.set(1, 1); m.set(2, 2); m.set(3, 3);
    m.clear();
    check(m.size === 0,    'clear: size is 0');
    check(m.get(1) === -1, 'clear: previously-set key returns -1');
    // Can re-insert after clear
    m.set(1, 99);
    check(m.get(1) === 99, 'clear: re-insert after clear');
  }

  // ── forEach ────────────────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    m.set(10, 1); m.set(20, 2); m.set(30, 3);
    let keySum = 0, valSum = 0, count = 0;
    m.forEach((k, v) => { keySum += k; valSum += v; count++; });
    check(count  === 3,  'forEach: visits all entries');
    check(keySum === 60, 'forEach: key sum correct');
    check(valSum === 6,  'forEach: val sum correct');
    // forEach on empty map
    const m2 = makeIntMap();
    let called = false;
    m2.forEach(() => { called = true; });
    check(!called, 'forEach: empty map visits nothing');
  }

  // ── Resize: values survive growth ─────────────────────────────────────────
  {
    // Start with tiny capacity; force multiple resizes.
    const m = makeIntMap(4);
    const N = 200;
    for (let i = 1; i <= N; i++) m.set(i, i * 3);
    check(m.size === N, 'post-resize: size correct');
    let ok = true;
    for (let i = 1; i <= N; i++) {
      if (m.get(i) !== i * 3) { ok = false; break; }
    }
    check(ok, 'post-resize: all values retrievable');
  }

  // ── Collision / probing: keys that alias to the same initial slot ──────────
  {
    // Two keys that hash to the same slot should both be stored and retrieved.
    // We brute-force find a collision in a small table (cap=8, mask=7).
    const m = makeIntMap(8);
    // Scan for two distinct nonzero keys with the same (hash & 7).
    const slots = new Map();
    let k1 = null, k2 = null;
    for (let k = 1; k < 100000 && k2 === null; k++) {
      const h = (Math.imul(796154621, k) ^ Math.imul(862632693, k >> 16)) & 7;
      if (slots.has(h)) { k1 = slots.get(h); k2 = k; }
      else slots.set(h, k);
    }
    check(k1 !== null && k2 !== null, 'collision test: found two colliding keys');
    if (k1 !== null) {
      m.set(k1, 111); m.set(k2, 222);
      check(m.get(k1) === 111, 'collision: first key retrieved correctly');
      check(m.get(k2) === 222, 'collision: second key retrieved correctly');
      check(m.size === 2,      'collision: size is 2');
      // Update one
      m.set(k1, 333);
      check(m.get(k1) === 333, 'collision: update first key');
      check(m.get(k2) === 222, 'collision: second key unaffected by update of first');
    }
  }

  // ── Stress: large number of Math.imul-style keys (mimics tdsearch usage) ──
  {
    const m = makeIntMap();
    const N = 10000;
    // Insert
    for (let i = 1; i <= N; i++) {
      const key = Math.imul(835 + i, 691 + (i % 3) - 1) || (i + 1);  // avoid 0
      m.set(key, i);
    }
    // Re-query with same formula; note: multiple i can produce the same key,
    // so we just check that every queried key returns a non-(-1) value.
    let misses = 0;
    for (let i = 1; i <= N; i++) {
      const key = Math.imul(835 + i, 691 + (i % 3) - 1) || (i + 1);
      if (m.get(key) === -1) misses++;
    }
    check(misses === 0, 'stress: no misses for inserted keys');
  }

  // ── clone ──────────────────────────────────────────────────────────────────
  {
    const m = makeIntMap();
    m.set(1, 10); m.set(2, 20); m.set(3, 30);
    const c = m.clone();
    check(c.size === 3,    'clone: size matches');
    check(c.get(1) === 10, 'clone: value 1 copied');
    check(c.get(2) === 20, 'clone: value 2 copied');
    check(c.get(3) === 30, 'clone: value 3 copied');
    // Mutating clone does not affect original
    c.set(1, 99);
    check(m.get(1) === 10, 'clone: original unaffected by clone mutation');
    check(c.get(1) === 99, 'clone: clone reflects its own mutation');
  }

  // ── minCap argument ────────────────────────────────────────────────────────
  {
    // Map created with minCap=1024 should handle 400 entries without issue.
    const m = makeIntMap(1024);
    for (let i = 1; i <= 400; i++) m.set(i, i);
    check(m.size === 400,    'minCap: size correct');
    check(m.get(200) === 200,'minCap: value correct');
  }

  // ── Report ─────────────────────────────────────────────────────────────────
  if (failures > 0) {
    console.error(`[int-map] ${failures} test(s) failed`);
  }
}

module.exports = { runTests };
