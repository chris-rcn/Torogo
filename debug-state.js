const { Game3Precise } = require('./game3-precise.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

function checkGroups(label) {
  console.log(`\n${label}:`);
  const visited = new Set();
  let count = 0;
  for (let i = 0; i < 169; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (!visited.has(gid)) {
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 1 || lc === 2) {
        console.log(`  gid ${gid}: ${lc} liberties`);
        count++;
      }
    }
  }
  console.log(`  Total atari groups: ${count}`);
  return count;
}

checkGroups('Before getAllLadderStatuses');
getAllLadderStatuses(game);
checkGroups('After 1st getAllLadderStatuses');
getAllLadderStatuses(game);
checkGroups('After 2nd getAllLadderStatuses');
