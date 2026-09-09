'use strict';

// measure-trunc-bias.js — paired-return measurement of the truncation bias
// floor E[b^2].
//
// For a playout-start position s, a truncated return is T = vpat(endpoint of a random
// delta-long playout prefix from s); its error splits into prefix noise eps
// (averages out over visits) and a leaf-conditional bias b(s) = E[T|s] - p(s)
// (does not), where p(s) is the mean FULL-playout outcome from s — the
// currency the search's untruncated returns are denominated in.  A node's
// error floor is b(s)^2, so E[b^2] over the band's playout-start distribution is what
// TRUNC_PHASE_DELTA (decorrelation) and TRUNC_MAX_RATIO (dilution) actually
// manage.  ('Playout-start', not 'leaf': tree leaves are nodes with their
// own phase — the truncation anchor is the position a playout launches from.)
//
// Estimator, per position: two truncated returns T1, T2 (independent
// prefixes) and two half-references Pa, Pb (each the mean of --ref-playouts/2
// independent full playouts).  All four are independent given s, so in
//   E[(T1 - Pa)(T2 - Pb)]
// every cross-term vanishes and the expectation is exactly E[b^2] — the
// split reference exists so the reference's own noise cannot leak in (a
// shared reference would add sigma_p^2/K).  Var(T1 - T2)/2 estimates the
// prefix-noise variance sigma_eps^2 for free.
//
// References are delta-independent, so --delta takes a comma list and the
// (dominant) reference cost is shared across the whole sweep.
//
// Playouts mirror the trunc agent's: ppat policy with uniformBelowPhase
// (PPAT_MIN_PHASE semantics); truncation is net empty-count advance, so
// captures during the prefix delay it, as deployed.

const fs = require('fs');
const path = require('path');
const { Game2, BLACK, PASS, coordStr } = require('./game2.js');
const VPat = require('./vpatterns.js');
const PPat = require('./ppat-lib.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'null'],
  ['file', 'games', 'vpat', 'ppat', 'ppat-min-phase', 'delta', 'ref-playouts', 'limit',
   'min-phase', 'max-phase', 'null-playouts', 'seed', 'emit']);

if (opts.help || (!opts.file && !opts.games) || (opts.file && opts.games) ||
    (!opts.vpat && !opts.null && !opts.emit)) {
  console.log(`Usage: node measure-trunc-bias.js (--file <evals> | --games <games>) --vpat <model.js> [options]

Measures the truncation bias floor E[b^2] (and the prefix-noise variance
sigma_eps^2) of a vpat evaluator by paired truncated returns against split
full-playout references, per TRUNC_PHASE_DELTA value.  See the header for
the estimator; the phase band should cover the PLAYOUT-START positions of the config
under study (start < B - delta) (e.g. [0, B - delta]).

  --file PATH        eval-format positions file, either format (labels are
                     ignored — only the positions matter)
  --games PATH       gen-games file ("<size> <moves>") instead of --file:
                     one playout-start position is reservoir-sampled per
                     game from the plies inside the band, so any band is
                     reachable from one unlabeled corpus
  --vpat PATH        vpatterns checkpoint to measure (required)
  --delta LIST       comma list of phase deltas to sweep (default 0.2)
  --ref-playouts K   full playouts per reference, split into halves
                     (default 200)
  --limit N          positions to measure (default 5000)
  --min-phase F      playout-start band floor (default 0)
  --max-phase F      playout-start band ceiling (default 1)
  --ppat PATH        playout policy weights (default the agents' standard)
  --ppat-min-phase F uniform playout moves below this fullness (default 0.6)
  --seed N           RNG seed (default 1)
  --null             instrument null check: replace the evaluator with a
                     playout vote at the truncation point (mean of
                     --null-playouts full playouts), which is unbiased by
                     construction — bias should read ~0 and E[b^2] ~ the
                     vote's own noise floor (~p(1-p)/N).  A significant
                     offset here is a tool artifact, not model bias
  --null-playouts N  vote size for --null (default 200)
  --emit PATH        write a BIAS TEST SET instead of measuring: per start
                     position, two prefix endpoints (full move sequences)
                     plus the two half-reference values — everything in the
                     estimator that is model-independent.  A trainer (or any
                     evaluator) can then score E[b^2] for the cost of 2n
                     evaluations.  Requires exactly one --delta; --vpat is
                     not used.  Positions whose game ends inside a prefix
                     are skipped (no endpoint exists)`);
  process.exit(0);
}

const DELTAS   = opts.delta ? opts.delta.split(',').map(parseFloat) : [0.2];
const REF_K    = parseInt(opts['ref-playouts'] || '200', 10);
const LIMIT    = parseInt(opts.limit || '5000', 10);
const MIN_PH   = parseFloat(opts['min-phase'] || '0');
const MAX_PH   = parseFloat(opts['max-phase'] || '1');
const SEED     = parseInt(opts.seed || '1', 10);
if (DELTAS.some(d => !(d > 0 && d < 1)) || !(REF_K >= 4)) {
  console.error('--delta values must be in (0,1); --ref-playouts >= 4');
  process.exit(1);
}

const NULL_MODE = !!opts.null;
const EMIT_PATH = opts.emit || null;
if (EMIT_PATH && NULL_MODE) { console.error('--emit and --null are mutually exclusive'); process.exit(1); }
if (EMIT_PATH && DELTAS.length !== 1) { console.error('--emit requires exactly one --delta'); process.exit(1); }
const NULL_K = parseInt(opts['null-playouts'] || '200', 10);
const vpatModel = (NULL_MODE || EMIT_PATH) ? null : VPat.loadWeights(opts.vpat, process.env.HEALTH_DATA || '');
const ppatModel = PPat.loadWeights(opts.ppat ||
  path.join(__dirname, 'out', 'ppat-data-233162-best-ref-candidate.js'));
ppatModel.uniformBelowPhase = parseFloat(opts['ppat-min-phase'] || '0.6');
if (EMIT_PATH) console.log(`emit: ${EMIT_PATH} (bias test set — no evaluator)`);
else if (NULL_MODE) console.log(`evaluator: NULL CHECK — ${NULL_K}-playout vote at the truncation point`);
else console.log(`vpat: ${opts.vpat} (${vpatModel.weights.size} weights, ` +
  `${vpatModel.specs.map(sp => `${sp.size}:${sp.maxLibs === 0 ? 'L' : sp.maxLibs}`).join(',')})`);
console.log(`deltas: ${DELTAS.join(', ')}  ref-playouts: ${REF_K}  start band: [${MIN_PH}, ${MAX_PH}]`);

const rng = makeRng(SEED);

// Dual-format position loader (as evalagentvalues): current format's second
// token is a phase; a legacy move list starts with a coordinate letter.
function loadPositions(filePath) {
  const out = [];
  for (const line of fs.readFileSync(filePath, 'utf8').split('\n')) {
    const l = line.trim();
    if (!l || l.startsWith('#')) continue;
    const p = l.split(/\s+/);
    if (p.length < 3) continue;
    const cur = /^[0-9]/.test(p[1]);
    if (cur && p.length < 4) continue;
    out.push({ size: parseInt(p[0], 10), moves: p[cur ? 2 : 1].split(',') });
  }
  return out;
}
function parseMoveTok(t, N) {
  if (t[0] === 'p') return PASS;
  return (parseInt(t.slice(1), 10) - 1) * N + (t.charCodeAt(0) - 97);
}

// One playout from a clone of g: play (ppat policy) until net empty-count
// advance reaches `need` (Infinity = to the end).  Returns P(BLACK wins):
// the vpat value at the truncation point, or the terminal score if the game
// ends first (as the deployed playout does).
function playout(g0, need, state) {
  const g = g0.clone();
  const cap = g.N * g.N;
  const stopEmpty = need === Infinity ? -1 : g.emptyCount - Math.ceil(need * cap);
  const moveLimit = 3 * g.emptyCount + 20;
  let moves = 0;
  while (!g.gameOver && moves < moveLimit) {
    if (stopEmpty >= 0 && g.emptyCount <= stopEmpty) {
      if (!NULL_MODE) return VPat.evaluate(g, vpatModel);
      // Null check: an unbiased estimate of the truncation point's own
      // playout value, in place of the evaluator.
      let w = 0;
      for (let i = 0; i < NULL_K; i++) w += playout(g, Infinity, state);
      return w / NULL_K;
    }
    g.play(PPat.ppatMove(g, state, ppatModel, rng));
    moves++;
  }
  return g.estimateWinner() === BLACK ? 1 : 0;
}

// Prefix recorder for --emit: play until the truncation point, returning the
// coord-string moves and endpoint phase, or null if the game ends first.
function prefixMoves(g0, need, state) {
  const g = g0.clone();
  const cap = g.N * g.N;
  const stopEmpty = g.emptyCount - Math.ceil(need * cap);
  const moveLimit = 3 * g.emptyCount + 20;
  const moves = [];
  let n = 0;
  while (!g.gameOver && n < moveLimit) {
    if (g.emptyCount <= stopEmpty) return { moves, phase: 1 - g.emptyCount / cap };
    const m = PPat.ppatMove(g, state, ppatModel, rng);
    g.play(m);
    moves.push(coordStr(m, g.N));
    n++;
  }
  return null;
}

// Position source: an eval file's positions, or per-game band-sampled plies
// from a games file.  Both yield { size, moves } replayed below; games are
// streamed lazily (files can be GB and only --limit positions are needed).
function* positionSource() {
  if (opts.file) { yield* loadPositions(opts.file); return; }
  const fd = fs.openSync(opts.games, 'r');
  const buf = Buffer.alloc(1 << 22);
  let rem = '';
  for (;;) {
    const nb = fs.readSync(fd, buf, 0, buf.length, null);
    if (nb === 0) break;
    const lines = (rem + buf.toString('utf8', 0, nb)).split('\n');
    rem = lines.pop();
    for (const line of lines) {
      const l = line.trim();
      if (!l || l[0] === '#') continue;
      const p = l.split(/\s+/);
      if (p.length !== 2) continue;
      const size = parseInt(p[0], 10);
      const moves = p[1].split(',');
      // Pass 1: replay counting plies whose position is in-band;
      // reservoir-pick one.
      const g = new Game2(size);
      const area = size * size;
      let pick = -1, seen = 0, ok = true;
      for (let i = 0; i < moves.length; i++) {
        const ph = 1 - g.emptyCount / area;
        if (ph >= MIN_PH && ph <= MAX_PH) {
          seen++;
          if (rng.random() < 1 / seen) pick = i;
        }
        if (!g.play(parseMoveTok(moves[i], size))) { ok = false; break; }
      }
      if (!ok || pick < 0) continue;
      yield { size, moves: moves.slice(0, pick) };
    }
  }
  fs.closeSync(fd);
}
console.log(opts.file ? `positions: from ${opts.file}` : `positions: band-sampled from games in ${opts.games}`);

// Per-delta accumulators.
const acc = DELTAS.map(() => ({ n: 0, prod: 0, dsq: 0, bias: 0 }));
let measured = 0, skipped = 0, outsideBand = 0;
let nextPrint = 50;
const t0 = Date.now();
let state = null;

console.log('     n   elapsed' + DELTAS.map(d => `   b2(${d})`.padStart(12)).join(''));

let emitFd = null;
if (EMIT_PATH) {
  emitFd = fs.openSync(EMIT_PATH, 'w');
  fs.writeSync(emitFd, `# bias-pairs: delta: ${DELTAS[0]} ref-playouts: ${REF_K} ` +
    `ppat: ${opts.ppat || 'default-233162'} ppat-min-phase: ${ppatModel.uniformBelowPhase} ` +
    `seed: ${SEED} source: ${opts.file || opts.games} start-band: [${MIN_PH}, ${MAX_PH}] ` +
    `date: ${new Date().toISOString().slice(0, 10)}\n`);
  fs.writeSync(emitFd, `# format: size endpointPhase startMoves movesToE1 movesToE2 pa pb\n`);
}
let emitSkippedTerminal = 0;

for (const pos of positionSource()) {
  if (measured >= LIMIT) break;
  const g = new Game2(pos.size);
  let ok = true;
  for (const t of pos.moves) if (!g.play(parseMoveTok(t, pos.size))) { ok = false; break; }
  if (!ok) { skipped++; continue; }
  const phase = 1 - g.emptyCount / (pos.size * pos.size);
  if (phase < MIN_PH || phase > MAX_PH) { outsideBand++; continue; }
  if (!state) state = PPat.createState(pos.size);

  if (EMIT_PATH) {
    // Prefixes first (cheap) so terminal-ending positions skip before the
    // reference cost is paid.
    const e1 = prefixMoves(g, DELTAS[0], state);
    const e2 = prefixMoves(g, DELTAS[0], state);
    if (!e1 || !e2) { emitSkippedTerminal++; continue; }
    const half = REF_K >> 1;
    let sa = 0, sb = 0;
    for (let i = 0; i < half; i++) sa += playout(g, Infinity, state);
    for (let i = 0; i < half; i++) sb += playout(g, Infinity, state);
    const startMoves = pos.moves.join(',') || '-';
    fs.writeSync(emitFd, `${pos.size} ${e1.phase.toFixed(3)} ${startMoves} ` +
      `${pos.moves.concat(e1.moves).join(',')} ${pos.moves.concat(e2.moves).join(',')} ` +
      `${(sa / half).toFixed(5)} ${(sb / half).toFixed(5)}\n`);
    measured++;
    if (measured >= nextPrint) {
      nextPrint = Math.ceil(nextPrint * 1.4);
      console.log(String(measured).padStart(6) + Util.fmtMs(Date.now() - t0).padStart(10) + '   (emitting)');
    }
    continue;
  }
  // Split reference: two independent half-means of full playouts.
  const half = REF_K >> 1;
  let sa = 0, sb = 0;
  for (let i = 0; i < half; i++) sa += playout(g, Infinity, state);
  for (let i = 0; i < half; i++) sb += playout(g, Infinity, state);
  const pa = sa / half, pb = sb / half;

  for (let di = 0; di < DELTAS.length; di++) {
    const t1 = playout(g, DELTAS[di], state);
    const t2 = playout(g, DELTAS[di], state);
    const a = acc[di];
    a.n++;
    a.prod += (t1 - pa) * (t2 - pb);
    a.dsq  += (t1 - t2) * (t1 - t2) / 2;
    a.bias += (t1 + t2) / 2 - (pa + pb) / 2;
  }
  measured++;

  if (measured >= nextPrint) {
    nextPrint = Math.ceil(nextPrint * 1.4);
    console.log(String(measured).padStart(6) + Util.fmtMs(Date.now() - t0).padStart(10) +
      acc.map(a => (a.prod / a.n).toFixed(6).padStart(12)).join(''));
  }
}

if (EMIT_PATH) {
  fs.closeSync(emitFd);
  console.log(`\nemitted ${measured} pairs to ${EMIT_PATH} in ${Util.fmtMs(Date.now() - t0)}` +
    `  (outside band: ${outsideBand}, replay-skipped: ${skipped}, terminal-in-prefix: ${emitSkippedTerminal})`);
  process.exit(0);
}
console.log(`\nmeasured ${measured} positions in ${Util.fmtMs(Date.now() - t0)}` +
  `  (outside band: ${outsideBand}, replay-skipped: ${skipped})`);
console.log('delta      E[b2]      var(b)       b_rms   sigma_eps        bias       n');
for (let di = 0; di < DELTAS.length; di++) {
  const a = acc[di];
  const b2 = a.prod / a.n;
  const se = Math.sqrt(a.dsq / a.n);
  const bias = a.bias / a.n;
  console.log(String(DELTAS[di]).padEnd(5) +
    b2.toFixed(6).padStart(11) +
    (b2 - bias * bias).toFixed(6).padStart(12) +
    Math.sqrt(Math.max(0, b2)).toFixed(4).padStart(12) +
    se.toFixed(4).padStart(12) +
    ((bias >= 0 ? '+' : '') + bias.toFixed(4)).padStart(12) +
    String(a.n).padStart(8));
}
