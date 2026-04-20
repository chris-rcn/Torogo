'use strict';

// xorshift.js — XorShift64 pseudo-random number generator.
//
// Uses the xorshift64 algorithm (Marsaglia 2003). The internal state is a
// pair of 32-bit integers representing a 64-bit value, because JS lacks
// native 64-bit integers.

(function () {

function makeRng(seed) {
  // Use current time (ms) as default seed; split into two 32-bit halves.
  if (seed === undefined) seed = Date.now();
  // Ensure non-zero state — zero is a fixed point for xorshift.
  let lo = (seed | 0) || 1;
  let hi = ((seed / 0x100000000) | 0) || 0x9e3779b9;

  function next() {
    // xorshift64: three-shift sequence with good statistical properties.
    let tlo = lo, thi = hi;
    tlo ^= tlo << 13;
    tlo ^= tlo >>> 7;
    tlo ^= tlo << 17;
    // Cross-word carry: shift hi into lo
    const carry = hi << 13;
    thi ^= thi >>> 17;
    thi ^= thi << 5;
    lo = tlo ^ carry;
    hi = thi;
    // Combine halves into an unsigned 32-bit result.
    return (lo ^ hi) >>> 0;
  }

  return {
    // Returns a float in [0, 1).
    random() {
      return next() / 0x100000000;
    },

    // Returns an integer in [0, maxExclusive).
    int(maxExclusive) {
      return (next() / 0x100000000 * maxExclusive) | 0;
    },
  };
}

const XorShift = { makeRng };
if (typeof module !== 'undefined') {
  module.exports = XorShift;
  require('./xorshift.test.js').runTests(XorShift);
} else {
  window.XorShift = XorShift;
}

})();
