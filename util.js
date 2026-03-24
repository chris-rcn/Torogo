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

  return { randInt, shuffle, envStr, envFloat, envInt };

})();

if (typeof module !== 'undefined') module.exports = Util;
else window.Util = Util;
