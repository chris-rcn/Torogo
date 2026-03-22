'use strict';

// util.js — general-purpose utilities.
// BROWSER-COMPATIBLE: no Node.js-only APIs.

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

if (typeof module !== 'undefined') module.exports = { randInt, shuffle };
