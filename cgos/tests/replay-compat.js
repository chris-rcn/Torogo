'use strict';

// replay-compat.js — cross-check Game2 against the vendored (toroidally
// patched) CGOS server GoGame.
//
// Plays random Game2 games, replays each move by move into the python
// GoGame with the same rules (simple ko, komi from game2), and verifies:
//   1. every Game2-legal move is accepted by the server, and
//   2. the server's 1-step area score picks the same winner as
//      Game2.calcWinner().
//
// This is the guarantee that the CGOS server will never forfeit an engine
// for playing a legal toroidal move.
//
// Usage: node cgos/tests/replay-compat.js [--games <n>] [--size <n>]

const { execFileSync } = require('child_process');
const path = require('path');
const { Game2, BLACK, KOMI } = require('../../game2.js');
const { idxToGtp } = require('../gtp.js');
const Util = require('../../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
const GAMES = opts.getInt('games', 25);
const SIZE  = opts.getInt('size', 9);

const games = [];
for (let i = 0; i < GAMES; i++) {
  const g = new Game2(SIZE);                    // auto-plays black at center
  const moves = [idxToGtp((SIZE >> 1) * SIZE + (SIZE >> 1), SIZE)];
  while (!g.gameOver) {
    const idx = g.randomLegalMove();
    if (!g.play(idx)) throw new Error('Game2 rejected its own random move');
    moves.push(idxToGtp(idx, SIZE));
  }
  games.push({ moves, winner: g.calcWinner() === BLACK ? 'B' : 'W' });
}

const py = `
import json, sys, os
sys.path.insert(0, ${JSON.stringify(path.join(__dirname, '..', 'server', 'cgos'))})
from gogame.go import GoGame, KoRule, Rule

data = json.load(sys.stdin)
komi = data["komi"]
fail = 0
for gi, game in enumerate(data["games"]):
    g = GoGame(data["size"], Rule(KoRule.SIMPLE))
    for mi, mv in enumerate(game["moves"]):
        r = g.make(mv)
        if r < 0:
            print(f"game {gi} move {mi} ({mv}): server rejected legal move, err {r}")
            fail += 1
            break
    else:
        if not g.twopass():
            print(f"game {gi}: did not end in two passes")
            fail += 1
            continue
        sc = g.ttScore() - komi
        winner = "B" if sc > 0 else "W"
        if winner != game["winner"]:
            print(f"game {gi}: server says {winner}{abs(sc)}, Game2 says {game['winner']}")
            fail += 1
print(("FAIL: %d mismatches" % fail) if fail else "OK: %d games replayed identically" % len(data["games"]))
sys.exit(1 if fail else 0)
`;

const input = JSON.stringify({ size: SIZE, komi: KOMI(SIZE), games });
const out = execFileSync('python3', ['-c', py], { input, encoding: 'utf8' });
process.stdout.write(out);
