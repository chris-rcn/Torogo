'use strict';

// Tests for xorshift.js.
// Called by xorshift.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.

function runTests({ makeXorShift }) {
  let failures = 0;

  function check(cond, msg) {
    if (!cond) { failures++; console.error('FAIL [xorshift]:', msg); }
  }

  // ── random() range ────────────────────────────────────────────────────────
  {
    const rng = makeXorShift(42);
    let allInRange = true;
    for (let i = 0; i < 10000; i++) {
      const v = rng.random();
      if (v < 0 || v >= 1) { allInRange = false; break; }
    }
    check(allInRange, 'random: all values in [0, 1)');
  }

  // ── random() is not constant ──────────────────────────────────────────────
  {
    const rng = makeXorShift(1);
    const seen = new Set();
    for (let i = 0; i < 100; i++) seen.add(rng.random());
    check(seen.size > 50, 'random: produces varied output (not stuck)');
  }

  // ── int() range ───────────────────────────────────────────────────────────
  {
    const rng = makeXorShift(99);
    let allInRange = true;
    for (let i = 0; i < 10000; i++) {
      const v = rng.int(10);
      if (!Number.isInteger(v) || v < 0 || v >= 10) { allInRange = false; break; }
    }
    check(allInRange, 'int(10): all values integers in [0, 10)');
  }

  // ── int() covers all values ───────────────────────────────────────────────
  {
    const rng = makeXorShift(7);
    const counts = new Array(6).fill(0);
    for (let i = 0; i < 60000; i++) counts[rng.int(6)]++;
    const allSeen = counts.every(c => c > 0);
    check(allSeen, 'int(6): all values 0–5 produced');
    // Rough uniformity: each bucket should be ~10000, allow ±30%
    const uniform = counts.every(c => c > 7000 && c < 13000);
    check(uniform, 'int(6): roughly uniform distribution');
  }

  // ── int(1) always returns 0 ───────────────────────────────────────────────
  {
    const rng = makeXorShift(5);
    let ok = true;
    for (let i = 0; i < 100; i++) { if (rng.int(1) !== 0) { ok = false; break; } }
    check(ok, 'int(1): always returns 0');
  }

  // ── Deterministic with same seed ─────────────────────────────────────────
  {
    const a = makeXorShift(12345);
    const b = makeXorShift(12345);
    let same = true;
    for (let i = 0; i < 100; i++) {
      if (a.random() !== b.random()) { same = false; break; }
    }
    check(same, 'deterministic: same seed produces same sequence');
  }

  // ── Different seeds produce different sequences ───────────────────────────
  {
    const a = makeXorShift(1);
    const b = makeXorShift(2);
    let diff = false;
    for (let i = 0; i < 20; i++) {
      if (a.random() !== b.random()) { diff = true; break; }
    }
    check(diff, 'different seeds: sequences differ');
  }

  // ── Default seed (no argument) does not throw ─────────────────────────────
  {
    let ok = false;
    try { const rng = makeXorShift(); rng.random(); rng.int(10); ok = true; } catch (_) {}
    check(ok, 'no-arg constructor: runs without error');
  }

  // ── Report ─────────────────────────────────────────────────────────────────
  if (failures > 0) {
    console.error(`[xorshift] ${failures} test(s) failed`);
  }
}

module.exports = { runTests };
