#!/usr/bin/env node
'use strict';

const { getMove } = require('./ai/tdsearch.js');
const { Game2 } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2));
const size   = parseInt(opts.size   || '4',    10);
const budget = parseInt(opts.budget || '1000', 10);

const t0 = Date.now();
const result = getMove(new Game2(size, false), budget);
const elapsed = (Date.now() - t0) / 1000;
let magSum = 0, magMax = 0;
for (const w of result.ctx.weightsArr) { 
  const a = Math.abs(w);
  magSum += a;
  if (a > magMax) magMax = a;
}
const nWeights = result.ctx.keyToIdx.size;
const avgMag = magSum / nWeights;
console.log(`size: ${size}  sims: ${result.sims}  sims/s: ${(result.sims / elapsed).toFixed(0)}  weights/cell: ${(nWeights / (size * size)).toFixed(2)}  avg|w|: ${avgMag.toFixed(4)}  max|w|: ${magMax.toFixed(4)}  maxSteps: ${result.maxSteps}`);

