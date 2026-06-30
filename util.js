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

  // Build a config reader scoped to an optional agent slot (1 or 2).  For each
  // key the resolution order is: explicit override → slot-prefixed env
  // (P<slot>_KEY) → plain env (KEY) → the caller-supplied default.  Browser (no
  // process.env) resolves to overrides/default.  This lets two agents in one
  // process read differentiated config via P1_/P2_ while plain env stays a
  // shared default.  Pass slot = null/undefined for a single, unprefixed agent.
  function makeCfg(slot, overrides) {
    const ov = overrides || {};
    function raw(key) {
      if (ov[key] !== undefined) return ov[key];
      if (typeof process === 'undefined') return undefined;
      if (slot != null) {
        const p = process.env['P' + slot + '_' + key];
        if (p !== undefined) return p;
      }
      return process.env[key];
    }
    return {
      slot:  slot != null ? slot : null,
      has:   (key)      => raw(key) !== undefined,
      str:   (key, def) => { const v = raw(key); return v !== undefined ? v : def; },
      int:   (key, def) => { const v = raw(key); return v !== undefined ? parseInt(v, 10) : def; },
      float: (key, def) => { const v = raw(key); return v !== undefined ? parseFloat(v) : def; },
      bool:  (key, def) => { const v = raw(key); return v !== undefined ? (v === '1' || v === 'true') : def; },
    };
  }

  // Parse --key value or --key=value flags from an argv array.
  // boolFlags: Set (or array) of flag names that take no value (e.g. 'help', 'verbose').
  // -h is always treated as an alias for --help.
  // boolFlags: flags that take no value.  knownFlags (optional): the full set of valid
  // flag names — when given, parsing is STRICT: any token that isn't a recognised --flag
  // (or its value) fails with a clean stderr message + exit, instead of being silently
  // swallowed (so a typo'd --var or a stray -p2 errors rather than vanishing).
  function parseArgs(argv, boolFlags, knownFlags) {
    const bools = boolFlags instanceof Set ? boolFlags : new Set(boolFlags || []);
    const known = knownFlags ? (knownFlags instanceof Set ? knownFlags : new Set(knownFlags)) : null;
    const fail = msg => {
      if (typeof process !== 'undefined' && process.exit) { console.error(msg); process.exit(1); }
      throw new Error(msg);
    };
    const opts = {};
    for (let i = 0; i < argv.length; i++) {
      const a = argv[i];
      if (a === '-h' || a === '--help') { opts.help = true; continue; }
      if (!a.startsWith('--')) {
        if (known) fail(`error: unexpected argument "${a}" (flags must be of the form --name)`);
        continue;
      }
      const eq = a.indexOf('=');
      const key = eq !== -1 ? a.slice(2, eq) : a.slice(2);
      if (known && !known.has(key) && !bools.has(key)) fail(`error: unknown flag "--${key}"`);
      if (eq !== -1) { opts[key] = a.slice(eq + 1); continue; }
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

  // Strip only a *leading* zero (and a sign's zero): "0.500" → ".500",
  // "-0.5" → "-.5".  Trailing zeros are never touched.  Port of the Java
  // fmtNoZero used by fmt4.
  function fmtNoZero(s) {
    return s.replace(/-0\./g, '-.').replace(/\+0\./g, '+.').replace(/^0\./, '.');
  }

  // Compact ≤4-character formatter — printf "%W.Pf" with K/M/B/T suffix scaling
  // (port of the Java fmt4(double)).  Tries progressively coarser precisions,
  // then thousands / millions / billions / trillions, and returns the first
  // rendering whose length is ≤ 4 chars.  fmtNoZero drops just the leading zero
  // (buying sub-1 values a 4th significant char) and never trailing zeros, so
  // whole numbers keep a decimal — 74 → "74.0", 1 → "1.00" — and fractional
  // columns line up.  Each attempt is [width, precision, value, suffix],
  // mirroring "%<W>.<P>f<suffix>".  Staying ≤ 4 chars is the whole point, so
  // non-finite inputs map to fixed ≤4 tokens rather than "Infinity"/"NaN".
  //
  // Examples:
  //   1.234   → "1.23"      12345   → " 12K"
  //   0.5     → ".500"      1.5e9   → "1.5B"
  //   74      → "74.0"      1.5e12  → "1.5T"
  //   1       → "1.00"      Inf     → " inf"
  function jfmt(width, prec, v) {
    return v.toFixed(prec).padStart(width);
  }
  function fmt4(units) {
    if (!Number.isFinite(units)) {
      if (units === Infinity)  return ' inf';
      if (units === -Infinity) return '-inf';
      return ' NaN';
    }
    const kilos = units / 1e3;
    const megas = kilos / 1e3;
    const gigas = megas / 1e3;
    const teras = gigas / 1e3;
    const attempts = [
      [5, 3, units, ''],
      [4, 2, units, ''],
      [4, 1, units, ''],
      [4, 0, units, ''],
      [4, 2, kilos, 'K'],
      [3, 1, kilos, 'K'],
      [3, 0, kilos, 'K'],
      [4, 2, megas, 'M'],
      [3, 1, megas, 'M'],
      [3, 0, megas, 'M'],
      [4, 2, gigas, 'B'],
      [3, 1, gigas, 'B'],
      [3, 0, gigas, 'B'],
      [4, 2, teras, 'T'],
      [3, 1, teras, 'T'],
      [3, 0, teras, 'T'],
    ];
    for (let i = 0; i < attempts.length; i++) {
      const w = attempts[i][0], p = attempts[i][1], v = attempts[i][2], suf = attempts[i][3];
      const s = fmtNoZero(jfmt(w, p, v) + suf);
      if (s.length <= 4) return s;
    }
    return String(units);
  }

  // Integral counterpart — port of the Java fmt4(long).  A whole-number count
  // renders plain ("%4d": 1 → "   1", 9999 → "9999") with no spurious decimal;
  // values too wide for 4 digits fall back to the K/M/B/T scaling of fmt4.  Use
  // this for counts (games, positions, epochs); use fmt4 for measured reals.
  function fmt4i(units) {
    const s = String(units).padStart(4);
    if (s.length <= 4) return s;
    return fmt4(units);
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
    if (!Number.isFinite(ms)) {
      if (ms === Infinity)  return '  inf';
      if (ms === -Infinity) return ' -inf';
      return String(ms).padStart(5);
    }
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
  // representing the value × 10000.  Out-of-range values clamp; non-finite
  // values render width-safe.
  //
  // Examples:
  //   0       → "0000"     0.474  → "4740"
  //   0.5     → "5000"     1.0    → "9999"  (clamped)
  //   -0.5    → "0000"     NaN    → " NaN"
  function fmtRatio4(value) {
    if (!Number.isFinite(value)) {
      if (value === Infinity)  return ' inf';
      if (value === -Infinity) return '-inf';
      return String(value).padStart(4);
    }
    if (value < 0) value = 0;
    const n = Math.min(Math.round(10000 * value), 9999);
    return String(n).padStart(4, '0');
  }

  return { shuffle, envStr, envFloat, envInt, makeCfg, parseArgs, makeZobrist, fmt4, fmt4i, fmtRatio4, fmtMs, load };

})();

if (typeof module !== 'undefined') {
  module.exports = Util;
  require('./util.test.js').runTests(Util);
} else {
  window.Util = Util;
}
