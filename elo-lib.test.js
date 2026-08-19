'use strict';

// Tests for elo-lib.js.
// Called by elo-lib.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.

function runTests({ expectedScore, eloFromWinrate, fitRatings, mleRating }) {
  let failures = 0;
  function fail(msg) { failures++; console.error('FAIL [elo-lib]:', msg); }
  function check(cond, msg) { if (!cond) fail(msg); }
  function near(actual, expected, tol, msg) {
    if (!(Math.abs(actual - expected) <= tol)) {
      fail(`${msg} — got ${actual}, want ${expected} ± ${tol}`);
    }
  }

  // ── expectedScore / eloFromWinrate ──────────────────────────────────────────
  near(expectedScore(1700, 1700), 0.5,    1e-12, 'expectedScore equal ratings');
  near(expectedScore(2100, 1700), 1 / 1.1, 1e-9, 'expectedScore +400');
  near(expectedScore(1700, 2100), 1 - 1 / 1.1, 1e-9, 'expectedScore -400');
  near(eloFromWinrate(0.5), 0, 1e-12, 'eloFromWinrate(0.5)');
  near(eloFromWinrate(expectedScore(1900, 1700)), 200, 1e-6, 'eloFromWinrate inverts expectedScore');

  // ── mleRating: single opponent ──────────────────────────────────────────────
  // 75 wins / 25 losses vs a 1600 player → 1600 + 400·log10(3).
  {
    const r = mleRating([{ r: 1600, w: 75, l: 25 }]);
    near(r.elo, 1600 + 400 * Math.log10(3), 0.1, 'mleRating point estimate');
    check(r.lo < r.elo && r.elo < r.hi, 'mleRating CI brackets estimate');
    check(r.hi - r.lo < 400, 'mleRating CI reasonably tight at n=100');
    check(r.games === 100, 'mleRating games count');
  }

  // ── mleRating: multiple opponents, exact expected scores ───────────────────
  // Fractional "wins" equal to the exact expected score recover the true elo.
  {
    const truth = 1750;
    const opps = [1600, 1700, 1800];
    const results = opps.map(o => {
      const p = expectedScore(truth, o);
      return { r: o, w: 100 * p, l: 100 * (1 - p) };
    });
    const r = mleRating(results);
    near(r.elo, truth, 0.1, 'mleRating multi-opponent recovery');
  }

  // ── mleRating: unbounded cases ──────────────────────────────────────────────
  {
    const r = mleRating([{ r: 1500, w: 10, l: 0 }]);
    check(r.elo === Infinity, 'all wins → elo = +Infinity');
    check(r.hi === Infinity, 'all wins → hi = +Infinity');
    check(Number.isFinite(r.lo), 'all wins → finite lower bound');
    check(r.lo > 1500, '10-0 vs 1500 → lower bound above opponent');
    const q = mleRating([{ r: 1500, w: 0, l: 10 }]);
    check(q.elo === -Infinity && q.lo === -Infinity && Number.isFinite(q.hi),
          'all losses → -Infinity with finite upper bound');
  }
  check(Number.isNaN(mleRating([]).elo), 'no games → NaN');

  // ── mleRating: CI behaves like ~1/sqrt(n) ───────────────────────────────────
  {
    const a = mleRating([{ r: 1600, w: 30,  l: 20 }]);
    const b = mleRating([{ r: 1600, w: 300, l: 200 }]);
    check(b.hi - b.lo < (a.hi - a.lo) / 2, 'CI shrinks with more games');
  }

  // ── fitRatings: exact-score records recover true ratings ───────────────────
  {
    const truth = new Map([['A', 1500], ['B', 1700], ['C', 1900]]);
    const records = [];
    for (const [x, rx] of truth) {
      for (const [y, ry] of truth) {
        if (x >= y) continue;
        const p = expectedScore(rx, ry);
        records.push({ a: x, b: y, wa: 200 * p, wb: 200 * (1 - p) });
      }
    }
    const fit = fitRatings(records, { anchors: new Map([['B', 1700]]), regularize: 0 });
    near(fit.get('A'), 1500, 0.5, 'fitRatings recovers A');
    near(fit.get('B'), 1700, 0,   'fitRatings keeps anchor fixed');
    near(fit.get('C'), 1900, 0.5, 'fitRatings recovers C');
  }

  // ── fitRatings: regularization keeps a 10-0 player finite ───────────────────
  {
    const fit = fitRatings(
      [{ a: 'winner', b: 'anchor', wa: 10, wb: 0 }],
      { anchors: new Map([['anchor', 1000]]) }
    );
    check(Number.isFinite(fit.get('winner')), 'regularized 10-0 rating is finite');
    check(fit.get('winner') > 1300, '10-0 rating is well above the anchor');
  }

  // ── fitRatings: no anchors → mean 0 ─────────────────────────────────────────
  {
    const fit = fitRatings([{ a: 'A', b: 'B', wa: 60, wb: 40 }]);
    near(fit.get('A') + fit.get('B'), 0, 1e-6, 'anchorless fit has mean 0');
    check(fit.get('A') > fit.get('B'), 'winner rated higher');
  }

  if (failures) console.error(`elo-lib tests: ${failures} failure(s)`);
  return failures;
}

module.exports = { runTests };
