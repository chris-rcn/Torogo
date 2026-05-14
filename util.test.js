'use strict';

// Tests for util.js.
// Called by util.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.

function runTests({ fmt4, fmtRatio4, fmtMs }) {
  let failures = 0;
  function fail(msg) { failures++; console.error('FAIL [util]:', msg); }
  function eq(actual, expected, msg) {
    if (actual !== expected) fail(`${msg} — got "${actual}", want "${expected}"`);
  }
  function check(cond, msg) { if (!cond) fail(msg); }

  // ── fmt4: small numbers (no suffix), right-padded to 4 chars ────────────────
  eq(fmt4(0),         '   0',   'fmt4(0)');
  eq(fmt4(1),         '   1',   'fmt4(1)');
  eq(fmt4(10),        '  10',   'fmt4(10)');
  eq(fmt4(100),       ' 100',   'fmt4(100)');
  eq(fmt4(1000),      '1000',   'fmt4(1000)');
  eq(fmt4(9999),      '9999',   'fmt4(9999)');

  // ── fmt4: fractions, trailing-zero stripping ────────────────────────────────
  eq(fmt4(0.5),       ' 0.5',   'fmt4(0.5) — strip trailing zeros');
  eq(fmt4(1.5),       ' 1.5',   'fmt4(1.5)');
  eq(fmt4(1.23),      '1.23',   'fmt4(1.23)');
  eq(fmt4(1.234),     '1.23',   'fmt4(1.234) — fits at .2f');
  eq(fmt4(12.3),      '12.3',   'fmt4(12.3)');
  eq(fmt4(12.34),     '12.3',   'fmt4(12.34) — fits at .1f');
  eq(fmt4(123.4),     ' 123',   'fmt4(123.4) — fits at .0f');

  // ── fmt4: K suffix ──────────────────────────────────────────────────────────
  eq(fmt4(10000),     ' 10K',   'fmt4(10000)');
  eq(fmt4(12345),     ' 12K',   'fmt4(12345)');
  eq(fmt4(99999),     '100K',   'fmt4(99999) — rounds up to 100');
  eq(fmt4(123456),    '123K',   'fmt4(123456)');

  // ── fmt4: M suffix ──────────────────────────────────────────────────────────
  eq(fmt4(1.234e6),   '1.2M',   'fmt4(1.234e6)');
  eq(fmt4(1.5e6),     '1.5M',   'fmt4(1.5e6)');
  eq(fmt4(4e6),       '4.0M',   'fmt4(4e6) — round-magnitude keeps .0 with suffix');
  eq(fmt4(12.3e6),    ' 12M',   'fmt4(12.3e6)');
  eq(fmt4(123.4e6),   '123M',   'fmt4(123.4e6)');

  // ── fmt4: B suffix ──────────────────────────────────────────────────────────
  eq(fmt4(1.234e9),   '1.2B',   'fmt4(1.234e9)');
  eq(fmt4(1e9),       '1.0B',   'fmt4(1e9) — round-magnitude keeps .0 with suffix');
  eq(fmt4(12.3e9),    ' 12B',   'fmt4(12.3e9)');
  eq(fmt4(123.4e9),   '123B',   'fmt4(123.4e9)');

  // ── fmt4: overflow falls back to plain (unpadded) String(units) ─────────────
  eq(fmt4(1.5e12),    String(1.5e12),  'fmt4(1.5e12) — beyond B, fallback');
  eq(fmt4(Infinity),  'Infinity',      'fmt4(Infinity) — fallback');
  eq(fmt4(NaN),       ' NaN',          'fmt4(NaN) — "NaN" fits, padded');

  // ── fmt4: negatives still try to fit ────────────────────────────────────────
  eq(fmt4(-1),        '  -1',   'fmt4(-1)');
  eq(fmt4(-1.5),      '-1.5',   'fmt4(-1.5)');
  eq(fmt4(-12.5),     ' -13',   'fmt4(-12.5) — fits at .0f');

  // ── fmt4: invariant — every result is exactly 4 chars, or the plain String ──
  const samples = [0, 1e-9, 0.001, 0.5, 1, 9.9, 99.9, 999, 9999, 99999,
                   1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e15,
                   -1, -10, -100, -1000];
  for (const x of samples) {
    const s = fmt4(x);
    check(s.length === 4 || s === String(x), `fmt4(${x}) = "${s}" — exactly 4 chars or plain String fallback`);
  }

  // ── fmtRatio4: standard in-range values ─────────────────────────────────────
  eq(fmtRatio4(0),       '0000', 'fmtRatio4(0)');
  eq(fmtRatio4(0.0001),  '0001', 'fmtRatio4(0.0001)');
  eq(fmtRatio4(0.474),   '4740', 'fmtRatio4(0.474)');
  eq(fmtRatio4(0.5),     '5000', 'fmtRatio4(0.5)');
  eq(fmtRatio4(0.699),   '6990', 'fmtRatio4(0.699)');
  eq(fmtRatio4(0.789),   '7890', 'fmtRatio4(0.789)');
  eq(fmtRatio4(0.9999),  '9999', 'fmtRatio4(0.9999)');

  // ── fmtRatio4: clamping ─────────────────────────────────────────────────────
  eq(fmtRatio4(1),       '9999', 'fmtRatio4(1) — clamped down');
  eq(fmtRatio4(2),       '9999', 'fmtRatio4(2) — clamped down');
  eq(fmtRatio4(-0.5),    '0000', 'fmtRatio4(-0.5) — clamped up to 0');

  // ── fmtRatio4: non-finite passthrough ───────────────────────────────────────
  eq(fmtRatio4(NaN),       ' NaN',      'fmtRatio4(NaN)');
  eq(fmtRatio4(Infinity),  'Infinity',  'fmtRatio4(Infinity)');
  eq(fmtRatio4(-Infinity), '-Infinity', 'fmtRatio4(-Infinity)');

  // ── fmtRatio4: invariant — finite inputs always produce exactly 4 digits ────
  for (const x of [0, 0.0001, 0.5, 0.9999, 1, 2, -0.5]) {
    const s = fmtRatio4(x);
    check(s.length === 4 && /^[0-9]{4}$/.test(s), `fmtRatio4(${x}) = "${s}" — 4 digits`);
  }

  // ── fmtMs: ns / us / ms scale ──────────────────────────────────────────
  eq(fmtMs(0),         '  0ns', 'fmtMs(0)');
  eq(fmtMs(1e-6),      '  1ns', 'fmtMs(1e-6) — 1 nanosecond');
  eq(fmtMs(0.001),     '  1us', 'fmtMs(0.001) — 1 microsecond');
  eq(fmtMs(0.5),       '500us', 'fmtMs(0.5)');
  eq(fmtMs(1),         '1.0ms', 'fmtMs(1) — single-digit ms keeps 1 decimal');
  eq(fmtMs(1.1),       '1.1ms', 'fmtMs(1.1) — sub-ms precision preserved');
  eq(fmtMs(9.9),       '9.9ms', 'fmtMs(9.9)');
  eq(fmtMs(10),        ' 10ms', 'fmtMs(10) — falls to integer ms when fractional overflows');
  eq(fmtMs(99),        ' 99ms', 'fmtMs(99)');
  eq(fmtMs(999),       '999ms', 'fmtMs(999)');

  // ── fmtMs: seconds ─────────────────────────────────────────────────────
  eq(fmtMs(1000),      ' 1.0s', 'fmtMs(1000) — 1 second');
  eq(fmtMs(12300),     '12.3s', 'fmtMs(12300)');
  eq(fmtMs(60000),     '60.0s', 'fmtMs(60000) — 1 minute as 60.0s');
  eq(fmtMs(99900),     '99.9s', 'fmtMs(99900)');
  eq(fmtMs(119500),    ' 120s', 'fmtMs(119500) — still < 2 min, falls to "%4.0fs"');

  // ── fmtMs: minutes ─────────────────────────────────────────────────────
  eq(fmtMs(120000),    ' 2.0m', 'fmtMs(120000) — 2 minutes');
  eq(fmtMs(1800000),   '30.0m', 'fmtMs(1800000) — 30 minutes');
  eq(fmtMs(7140000),   ' 119m', 'fmtMs(7140000) — still < 2 hours');

  // ── fmtMs: hours / days ────────────────────────────────────────────────
  eq(fmtMs(7200000),   ' 2.0h', 'fmtMs(7200000) — 2 hours');
  eq(fmtMs(86400000),  '24.0h', 'fmtMs(86400000) — 1 day as 24.0h');
  eq(fmtMs(172800000), '48.0h', 'fmtMs(172800000) — 2 days as 48.0h');
  // 100 days × 86_400_000 ms/day = 8.64e9 ms → "%4.0fd" → " 100d"
  eq(fmtMs(8640000000), ' 100d', 'fmtMs(100 days)');

  // ── fmtMs: non-finite ──────────────────────────────────────────────────
  eq(fmtMs(NaN),       'NaN',       'fmtMs(NaN)');
  eq(fmtMs(Infinity),  'Infinity',  'fmtMs(Infinity)');

  // ── fmtMs: invariant — finite inputs ≤ ~10k days fit in 5 chars ────────
  for (const x of [0, 1, 1e3, 1e6, 1e9, 1e11]) {
    const s = fmtMs(x);
    check(s.length <= 5, `fmtMs(${x}) = "${s}" — length ≤ 5`);
  }

  if (failures > 0) console.error(`[util] ${failures} test(s) failed`);
}

module.exports = { runTests };
