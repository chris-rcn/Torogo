const { Game3Precise, PASS } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

// Track both successful and skipped operations
const originalAddLiberty = game._addLiberty.bind(game);
const originalRemoveLiberty = game._removeLiberty.bind(game);
const originalAddLiberty_raw = game._addLiberty_raw.bind(game);
const originalRemoveLiberty_raw = game._removeLiberty_raw.bind(game);

let additions = [];
let removals = [];
let additionsSkipped = [];
let removalsSkipped = [];

game._addLiberty = function(gid, idx) {
  if (gid === 2) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (!wasSet) {
      additions.push({ gid, idx });
      console.log(`[ADD-OK] gid 2 idx ${idx}: _ls was ${this._ls[2]}`);
    } else {
      additionsSkipped.push({ gid, idx });
      console.log(`[ADD-SKIP] gid 2 idx ${idx}: already a liberty, _ls=${this._ls[2]}`);
    }
  }
  return originalAddLiberty(gid, idx);
};

game._removeLiberty = function(gid, idx) {
  if (gid === 2) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (wasSet) {
      removals.push({ gid, idx });
      console.log(`[REM-OK] gid 2 idx ${idx}: _ls was ${this._ls[2]}`);
    } else {
      removalsSkipped.push({ gid, idx });
      console.log(`[REM-SKIP] gid 2 idx ${idx}: not a liberty, _ls=${this._ls[2]}`);
    }
  }
  return originalRemoveLiberty(gid, idx);
};

console.log(`Initial _ls[2] = ${game._ls[2]}\n`);

const status = getLadderStatus(game, 46);

console.log(`\n\nFinal _ls[2] = ${game._ls[2]} (expected 2)`);
console.log(`\nAdds: ${additions.length}, skipped: ${additionsSkipped.length}`);
console.log(`Removes: ${removals.length}, skipped: ${removalsSkipped.length}`);
console.log(`Net operations: ${additions.length - removals.length}`);

console.log(`\nSkipped additions:`);
additionsSkipped.slice(0, 10).forEach(op => console.log(`  idx ${op.idx}`));
if (additionsSkipped.length > 10) console.log(`  ... and ${additionsSkipped.length - 10} more`);

console.log(`\nSkipped removals:`);
removalsSkipped.slice(0, 10).forEach(op => console.log(`  idx ${op.idx}`));
if (removalsSkipped.length > 10) console.log(`  ... and ${removalsSkipped.length - 10} more`);
