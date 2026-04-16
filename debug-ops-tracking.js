const { Game3Precise, PASS } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
console.log(`  _ls[2] = ${game._ls[2]}`);
console.log(`  _opStack length = ${game._opStack.length}`);

const initialOpCount = game._opStack.length;

// Wrap the game methods to track liberty operations
const originalAddLiberty = game._addLiberty.bind(game);
const originalRemoveLiberty = game._removeLiberty.bind(game);

let libertyOps = [];
let opIndex = 0;

game._addLiberty = function(gid, idx) {
  if (gid === 2) {
    console.log(`[+${opIndex}] _addLiberty(gid 2, ${idx})`);
    libertyOps.push({ type: 'add', gid, idx, opIndex });
    opIndex++;
  }
  return originalAddLiberty(gid, idx);
};

game._removeLiberty = function(gid, idx) {
  if (gid === 2) {
    console.log(`[-${opIndex}] _removeLiberty(gid 2, ${idx})`);
    libertyOps.push({ type: 'remove', gid, idx, opIndex });
    opIndex++;
  }
  return originalRemoveLiberty(gid, idx);
};

console.log('\nCalling getLadderStatus(game, 46)...\n');
const status = getLadderStatus(game, 46);
console.log(`\n\nReturned: ${JSON.stringify(status)}`);

console.log(`\nAfter getLadderStatus:`);
console.log(`  _ls[2] = ${game._ls[2]}`);
console.log(`  _opStack length = ${game._opStack.length}`);

console.log(`\nLiberty operations on gid 2:`);
for (const op of libertyOps) {
  console.log(`  ${op.type.padEnd(6)} idx=${String(op.idx).padStart(3)} [op#${op.opIndex}]`);
}

// Count adds vs removes
const adds = libertyOps.filter(op => op.type === 'add').length;
const removes = libertyOps.filter(op => op.type === 'remove').length;
console.log(`\nSummary: ${adds} adds, ${removes} removes (diff = ${adds - removes})`);
console.log(`Liberty count changed by: ${game._ls[2] - 2} (expected 0)`);
