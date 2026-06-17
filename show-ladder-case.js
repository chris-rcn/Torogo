'use strict';
// Re-display the discovered case, marking the CRITICAL (killing) move, with labels.
// Reconstructs from the printed board and verifies the kill before displaying.
const { Game2, BLACK, WHITE, PASS, coordStr, parseBoard, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const board = `
○ · ● ● ● ○ ● ○ ● ● ○ ● ●
○ ○ ● ● ○ ○ ● ● ○ ● · ● ○
○ ○ ○ ● ○ ● ● ○ ○ · ● ● ○
○ · ○ ○ ○ ○ ○ ● ○ ○ ○ ● ●
● ○ · ○ · ○ · ● ● ○ ○ ○ ●
● ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ●
● ● ● ● ● ○ ● ● ○ ○ ● ● ·
● ● ● ● · ● ● ● ○ ● ● · ·
● ● ○ ○ ○ ● ● ○ ○ ● ● ○ ●
● ○ · ○ ○ ○ ○ ● ○ ○ ● ● ●
● ● ○ ○ ● ○ ● ● ○ ● ● ● ●
● ● ● ○ ● ○ ○ ● ● ● ● ○ ○
● · ○ ○ ● ● · ○ ● ○ ○ ● ·
`;

const game = parseBoard(board, WHITE);   // WHITE (attacker) to move
const kill = parseMove('g1', 13);

// verify: find the size-10 2-lib BLACK chain and confirm its unique kill is g1
let ok = false;
const seen = new Set();
for (let i = 0; i < 169; i++) {
  if (game.cells[i] !== BLACK) continue;
  const gid = game._gid[i];
  if (seen.has(gid)) continue; seen.add(gid);
  if (game.groupSize(gid) !== 10 || game.groupLibertyCount(gid) !== 2) continue;
  const st = getLadderStatus(game3FromGame2(game), i);
  if (st && st.moverSucceeds && st.urgentLibs.length === 1) {
    ok = st.urgentLibs[0] === kill;
    console.log(`verify: size-10 2-lib BLACK chain, urgentLibs=[${st.urgentLibs.map(m => coordStr(m,13)).join(',')}]  kill==g1? ${ok}`);
  }
}
console.log(`\ncritical (killing) move: g1   attacker: WHITE ○   target: BLACK ● (10 stones, 2 libs)\n`);
console.log(game.toString(kill, { labels: true, centerAt: kill }));
