'use strict';

// util.js — general-purpose utilities.
// BROWSER-COMPATIBLE: no Node.js-only APIs.
// Exposes a single global: Util.

const Util = (() => {

  // Fisher-Yates in-place shuffle.  Returns arr.
  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = rng.int(i + 1);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // Read an environment variable, falling back to `def`.
  function envStr(name, def) {
    return (typeof process !== 'undefined' && process.env[name] !== undefined) ? process.env[name] : def;
  }

  function envFloat(name, def) {
    return (typeof process !== 'undefined' && process.env[name] !== undefined) ? parseFloat(process.env[name]) : def;
  }

  function envInt(name, def) {
    return (typeof process !== 'undefined' && process.env[name] !== undefined) ? parseInt(process.env[name], 10) : def;
  }

  // Parse --key value or --key=value flags from an argv array.
  // boolFlags: Set (or array) of flag names that take no value (e.g. 'help', 'verbose').
  // -h is always treated as an alias for --help.
  function parseArgs(argv, boolFlags) {
    const bools = boolFlags instanceof Set ? boolFlags : new Set(boolFlags || []);
    const opts = {};
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if (a === '-h') { opts.help = true; continue; }
      if (!a.startsWith('--')) continue;
      const eq = a.indexOf('=');
      if (eq !== -1) { opts[a.slice(2, eq)] = a.slice(eq + 1); continue; }
      const key = a.slice(2);
      if (bools.has(key)) { opts[key] = true; continue; }
      opts[key] = argv[++i];
    }
    opts.get      = (key, def) => opts[key] !== undefined ? opts[key] : def;
    opts.getInt   = (key, def) => opts[key] !== undefined ? parseInt(opts[key], 10) : def;
    opts.getFloat = (key, def) => opts[key] !== undefined ? parseFloat(opts[key]) : def;
    return opts;
  }

  // Dual-environment module loader.  Folds the `_isNode ? require(...) :
  // window.X` dance into one consistent call.  Paths are resolved relative
  // to util.js's directory (i.e. the project root), so callers in any
  // sub-directory write the same import: `Util.load('./game2.js', 'Game2')`.
  // The first call to load Util itself still has to be a bare require /
  // window.Util read by the caller — that one line is the unavoidable
  // bootstrap.
  function load(nodePath, windowName) {
    if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      return require(nodePath);
    }
    return window[windowName];
  }

  function makeZobrist(seed, size) {
    const t = new Int32Array(size);
    let s = seed;
    for (let p = 0; p < size; p++) {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      t[p] = s + 1234567;
    }
    return t;
  }

  // Compact 4-character formatter — like printf "%4.Nf" with K/M/B suffix scaling.
  // Tries progressively shorter representations (fewer decimals, then thousands /
  // millions / billions) and returns the first whose final length is ≤ 4 chars,
  // with trailing zeros and bare trailing dots stripped, right-padded with spaces
  // to exactly 4 chars.  Falls back to the unpadded String(units) if even billions
  // overflow (the only case the result can exceed 4 chars).
  //
  // Examples:
  //   1.234   → "1.23"      12345   → " 12K"
  //   9999    → "9999"      1.5e9   → "1.5B"
  //   0       → "   0"      NaN     → " NaN"
  function fmt4(units) {
    const kilos = units / 1e3;
    const megas = kilos / 1e3;
    const gigas = megas / 1e3;
    const attempts = [
      [3, units, ''],
      [2, units, ''],
      [1, units, ''],
      [0, units, ''],
      [2, kilos, 'K'],
      [1, kilos, 'K'],
      [0, kilos, 'K'],
      [2, megas, 'M'],
      [1, megas, 'M'],
      [0, megas, 'M'],
      [2, gigas, 'B'],
      [1, gigas, 'B'],
      [0, gigas, 'B'],
    ];
    for (let i = 0; i < attempts.length; i++) {
      const prec = attempts[i][0], v = attempts[i][1], suf = attempts[i][2];
      let s = v.toFixed(prec);
      if (s.indexOf('.') !== -1) {
        s = s.replace(/0+$/, '');
        if (s.charAt(s.length - 1) === '.') {
          // Suffix forms (K/M/B) keep a single trailing zero so e.g. 4e6 shows
          // as "4.0M" instead of "4M".  No-suffix forms strip the bare dot.
          s = suf ? s + '0' : s.slice(0, -1);
        }
      }
      s += suf;
      if (s.length <= 4) return s.padStart(4);
    }
    return String(units);
  }

  // Format an elapsed-time value (in milliseconds) as a compact ≤5-character
  // string using the largest unit that fits: ns, us, ms, s, m, h, or d.  Each
  // attempt is right-padded to width 3 (numeric) + 1-2 char suffix; the "%4.0f"
  // forms only fire when the next unit's value is still < 2 (so we move on to
  // minutes at ≥ 120 s, hours at ≥ 120 min, days at ≥ 48 h instead of showing
  // bare-integer counts that look like a smaller unit).
  //
  // Examples (ms input):
  //   0       → "  0ns"    1       → "  1ms"   1000    → " 1.0s"   60000  → "60.0s"
  //   119500  → " 120s"    120000  → " 2.0m"   86400000 → "24.0h"  1e10   → " 116d"
  function fmtMs(ms) {
    if (!Number.isFinite(ms)) return String(ms);
    const nanos  = ms * 1e6;
    const micros = ms * 1e3;
    const millis = ms;
    const secs   = ms / 1e3;
    const mins   = secs / 60;
    const hours  = mins / 60;
    const days   = hours / 24;
    let s;
    s = nanos .toFixed(0).padStart(3) + 'ns'; if (s.length <= 5) return s;
    s = micros.toFixed(0).padStart(3) + 'us'; if (s.length <= 5) return s;
    s = millis.toFixed(1)             + 'ms'; if (s.length <= 5) return s;
    s = millis.toFixed(0).padStart(3) + 'ms'; if (s.length <= 5) return s;
    s = secs  .toFixed(1).padStart(4) + 's';  if (s.length <= 5) return s;
    s = secs  .toFixed(0).padStart(4) + 's';  if (s.length <= 5 && mins  < 2) return s;
    s = mins  .toFixed(1).padStart(4) + 'm';  if (s.length <= 5) return s;
    s = mins  .toFixed(0).padStart(4) + 'm';  if (s.length <= 5 && hours < 2) return s;
    s = hours .toFixed(1).padStart(4) + 'h';  if (s.length <= 5) return s;
    s = hours .toFixed(0).padStart(4) + 'h';  if (s.length <= 5 && days  < 2) return s;
    s = days  .toFixed(0).padStart(4) + 'd';  if (s.length <= 5) return s;
    return String(ms);
  }

  // Format a ratio in [0, 1] as a 4-character zero-padded integer in [0000, 9999],
  // representing the value × 10000.  Out-of-range values clamp; non-finite values
  // pass through as right-padded strings.
  //
  // Examples:
  //   0       → "0000"     0.474  → "4740"
  //   0.5     → "5000"     1.0    → "9999"  (clamped)
  //   -0.5    → "0000"     NaN    → " NaN"
  function fmtRatio4(value) {
    if (Number.isFinite(value)) {
      if (value < 0) value = 0;
      const n = Math.min(Math.round(10000 * value), 9999);
      return String(n).padStart(4, '0');
    }
    return String(value).padStart(4);
  }

  return { shuffle, envStr, envFloat, envInt, parseArgs, makeZobrist, fmt4, fmtRatio4, fmtMs, load };

})();

if (typeof module !== 'undefined') {
  module.exports = Util;
  require('./util.test.js').runTests(Util);
} else {
  window.Util = Util;
}
