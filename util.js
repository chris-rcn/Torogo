'use strict';

// util.js — general-purpose utilities.
// BROWSER-COMPATIBLE: no Node.js-only APIs.
// Exposes a single global: Util.

const Util = (() => {

  // Random integer in [0, n).
  function randInt(n) {
    return Math.floor(Math.random() * n);
  }

  // Fisher-Yates in-place shuffle.  Returns arr.
  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = randInt(i + 1);
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

  return { randInt, shuffle, envStr, envFloat, envInt, parseArgs };

})();

if (typeof module !== 'undefined') module.exports = Util;
else window.Util = Util;
