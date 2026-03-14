// How many extra intersections to render beyond each tile edge.
// 0 = exact tile boundaries; increase to show wrapping rows/cols and make
// the toroidal topology more intuitive.
let OVERLAP = 1;

// ─── Renderer ─────────────────────────────────────────────────────────────────

class Renderer {
  constructor(canvas, game) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.game = game;
    this.padding = 32;
    this.panX = 0;
    this.panY = 0;
    this.hoverPos = null;
    this.resize();
  }

  resize() {
    const N = this.game.boardSize;
    const maxSize = Math.min(window.innerWidth - 32, window.innerHeight - 220, 640);
    const size = Math.max(maxSize, 280);
    this.canvas.width = size;
    this.canvas.height = size;
    // Fit N-1+OVERLAP spacings so OVERLAP extra intersections are visible in total
    this.cellSize = (size - this.padding * 2) / (N - 1 + OVERLAP);
    // Split OVERLAP as evenly as possible: floor(OVERLAP/2) on left/top,
    // ceil(OVERLAP/2) on right/bottom
    const panOffset = Math.floor(OVERLAP / 2) * this.cellSize;
    this.panX = panOffset;
    this.panY = panOffset;
  }

  // Convert grid coords to canvas pixels, accounting for pan and tile offset
  toCanvas(x, y, offX = 0, offY = 0) {
    return [
      this.padding + this.panX + offX + x * this.cellSize,
      this.padding + this.panY + offY + y * this.cellSize,
    ];
  }

  // Convert canvas pixel to nearest grid intersection, with toroidal wrapping
  fromCanvas(px, py) {
    const N = this.game.boardSize;
    const rx = px - this.padding - this.panX;
    const ry = py - this.padding - this.panY;
    const x = ((Math.round(rx / this.cellSize) % N) + N) % N;
    const y = ((Math.round(ry / this.cellSize) % N) + N) % N;
    return { x, y };
  }

  // Tile size: the pixel distance between equivalent points on adjacent copies
  tileSize() {
    return this.game.boardSize * this.cellSize;
  }

  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const ts = this.tileSize();

    // Board background
    const grad = ctx.createLinearGradient(0, 0, W, H);
    grad.addColorStop(0, '#d4a84a');
    grad.addColorStop(1, '#c49840');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Clip to the board interior so adjacent-tile leakage never bleeds
    // into the padding zone, keeping overlap consistent across board sizes.
    ctx.save();
    ctx.beginPath();
    // Expand by half a cell on each side so stones/lines at boundary
    // intersections are fully visible (stone radius = 0.46 cs < 0.5 cs),
    // while still excluding adjacent-tile leakage (≥ 1 cs outside boundary).
    const halfCs = this.cellSize / 2;
    ctx.rect(this.padding - halfCs, this.padding - halfCs,
             W - 2 * this.padding + 2 * halfCs, H - 2 * this.padding + 2 * halfCs);
    ctx.clip();

    // Compute which tile indices are needed to cover the canvas viewport,
    // so panning works infinitely in any direction.
    const dxMin = Math.floor((-this.panX - this.padding) / ts);
    const dxMax = Math.ceil((W - this.panX - this.padding) / ts);
    const dyMin = Math.floor((-this.panY - this.padding) / ts);
    const dyMax = Math.ceil((H - this.panY - this.padding) / ts);

    this.drawGrid();

    for (let dy = dyMin; dy <= dyMax; dy++) {
      for (let dx = dxMin; dx <= dxMax; dx++) {
        const offX = dx * ts;
        const offY = dy * ts;
        this.drawStones(offX, offY);
        this.drawLastMove(offX, offY);
        this.drawIllegalFlash(offX, offY);
      }
    }

    // Ghost stone is drawn in its own pass so each canvas position is
    // painted exactly once (the per-tile OVERLAP regions would otherwise
    // overlap and compound the alpha).
    this.drawGhostStone();

    ctx.restore();
  }

  drawGrid() {
    const ctx = this.ctx;
    const cs = this.cellSize;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.strokeStyle = '#5a3a10';
    ctx.lineWidth = 1;

    // Draw each logical column/row exactly once across the full canvas so
    // no line is painted twice (per-tile drawing caused double-painting at
    // tile boundaries, producing faint edge artifacts).
    const originX = this.padding + this.panX;
    const originY = this.padding + this.panY;
    const halfCs = cs / 2;
    const kxMin = Math.ceil((-originX - halfCs) / cs);
    const kxMax = Math.floor((W - originX + halfCs) / cs);
    const kyMin = Math.ceil((-originY - halfCs) / cs);
    const kyMax = Math.floor((H - originY + halfCs) / cs);

    ctx.beginPath();
    for (let k = kxMin; k <= kxMax; k++) {
      const x = originX + k * cs;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    for (let k = kyMin; k <= kyMax; k++) {
      const y = originY + k * cs;
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
    }
    ctx.stroke();
  }

  drawIllegalFlash(offX = 0, offY = 0) {
    const flash = this.game.illegalFlash;
    if (!flash) return;
    const N = this.game.boardSize;
    const r = this.cellSize * 0.38;
    this.ctx.save();
    this.ctx.globalAlpha = 0.35;
    this.ctx.fillStyle = '#ff4444';
    const loF = -Math.floor(OVERLAP / 2), hiF = N + Math.ceil(OVERLAP / 2);
    for (let gy = loF; gy < hiF; gy++) {
      for (let gx = loF; gx < hiF; gx++) {
        if (((gx % N) + N) % N !== flash.x || ((gy % N) + N) % N !== flash.y) continue;
        const [cx, cy] = this.toCanvas(gx, gy, offX, offY);
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
    this.ctx.restore();
  }

  drawStones(offX = 0, offY = 0) {
    const ctx = this.ctx;
    const N = this.game.boardSize;
    const r = this.cellSize * 0.46;

    const loS = -Math.floor(OVERLAP / 2), hiS = N + Math.ceil(OVERLAP / 2);
    for (let gy = loS; gy < hiS; gy++) {
      for (let gx = loS; gx < hiS; gx++) {
        // Wrap visual index to logical grid coordinate
        const bx = ((gx % N) + N) % N;
        const by = ((gy % N) + N) % N;
        const color = this.game.board.get(bx, by);
        if (!color) continue;

        const [cx, cy] = this.toCanvas(gx, gy, offX, offY);

        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;

        const grad = ctx.createRadialGradient(
          cx - r * 0.3, cy - r * 0.3, r * 0.05,
          cx, cy, r
        );
        if (color === 'black') {
          grad.addColorStop(0, '#666');
          grad.addColorStop(1, '#111');
        } else {
          grad.addColorStop(0, '#fff');
          grad.addColorStop(1, '#bbb');
        }
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        ctx.strokeStyle = color === 'black' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.18)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  drawGhostStone() {
    const h = this.hoverPos;
    if (!h || this.game.gameOver) return;
    if (this.game.board.get(h.x, h.y) !== null) return;
    const cs = this.cellSize;
    const ts = this.tileSize();
    const W = this.canvas.width;
    const H = this.canvas.height;
    const r = cs * 0.46;
    const color = this.game.current;
    const ctx = this.ctx;

    // Canvas position of this cell in the base tile
    const baseX = this.padding + this.panX + h.x * cs;
    const baseY = this.padding + this.panY + h.y * cs;

    // Range of tile repetitions whose instance of this cell is on-screen,
    // including OVERLAP margin so edge duplicates are covered.
    const kMin = Math.floor((-baseX - OVERLAP * cs) / ts);
    const kMax = Math.ceil((W - baseX + OVERLAP * cs) / ts);
    const mMin = Math.floor((-baseY - OVERLAP * cs) / ts);
    const mMax = Math.ceil((H - baseY + OVERLAP * cs) / ts);

    ctx.save();
    ctx.globalAlpha = 0.4;
    for (let m = mMin; m <= mMax; m++) {
      for (let k = kMin; k <= kMax; k++) {
        const cx = baseX + k * ts;
        const cy = baseY + m * ts;
        if (cx < -r || cx > W + r || cy < -r || cy > H + r) continue;
        const grad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.05, cx, cy, r);
        if (color === 'black') {
          grad.addColorStop(0, '#666');
          grad.addColorStop(1, '#111');
        } else {
          grad.addColorStop(0, '#fff');
          grad.addColorStop(1, '#bbb');
        }
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  drawLastMove(offX = 0, offY = 0) {
    const lm = this.game.lastMove;
    if (!lm) return;
    const N = this.game.boardSize;
    const prev = this.game.current === 'black' ? 'white' : 'black';
    this.ctx.fillStyle = prev === 'black' ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.7)';
    const r = this.cellSize * 0.14;
    // Draw marker at every visual position that maps to the last-move cell
    const loL = -Math.floor(OVERLAP / 2), hiL = N + Math.ceil(OVERLAP / 2);
    for (let gy = loL; gy < hiL; gy++) {
      for (let gx = loL; gx < hiL; gx++) {
        if (((gx % N) + N) % N !== lm.x || ((gy % N) + N) % N !== lm.y) continue;
        const [cx, cy] = this.toCanvas(gx, gy, offX, offY);
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
  }
}

// ─── App wiring ───────────────────────────────────────────────────────────────

// ─── Influence AI (computer plays black) ──────────────────────────────────────

function aiCloneGame(game) {
  const g = new Game(game.boardSize);
  g.board             = game.board.clone();
  g.current           = game.current;
  g.captured          = { ...game.captured };
  g.hash              = game.hash;
  g.prevHash          = game.prevHash;
  g.consecutivePasses = game.consecutivePasses;
  g.gameOver          = game.gameOver;
  return g;
}

function aiIsTrueEye(board, x, y, color) {
  const N = board.size;
  const ortho = board.getNeighbors(x, y);
  if (!ortho.every(([nx, ny]) => board.get(nx, ny) === color)) return false;
  const diags = [
    [(x + 1) % N,     (y + 1) % N],
    [(x - 1 + N) % N, (y + 1) % N],
    [(x + 1) % N,     (y - 1 + N) % N],
    [(x - 1 + N) % N, (y - 1 + N) % N],
  ];
  return diags.filter(([dx, dy]) => board.get(dx, dy) === color).length >= 3;
}

function aiShuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function aiGroupsByColor(board, color) {
  const visited = new Set();
  const results = [];
  const N = board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const key = `${x},${y}`;
      if (board.get(x, y) !== color || visited.has(key)) continue;
      const group = board.getGroup(x, y);
      group.forEach(([gx, gy]) => visited.add(`${gx},${gy}`));
      const libs = board.getLiberties(group);
      results.push({ group, libs });
    }
  }
  results.sort((a, b) => b.group.length - a.group.length);
  return results;
}

function aiLeavesOwnGroupAtari(clone, color) {
  const visited = new Set();
  const N = clone.board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const key = `${x},${y}`;
      if (clone.board.get(x, y) !== color || visited.has(key)) continue;
      const group = clone.board.getGroup(x, y);
      group.forEach(([gx, gy]) => visited.add(`${gx},${gy}`));
      if (clone.board.getLiberties(group).size === 1) return true;
    }
  }
  return false;
}

function aiCapture(game) {
  const opp = game.current === 'black' ? 'white' : 'black';
  const atari = aiGroupsByColor(game.board, opp).filter(({ libs }) => libs.size === 1);
  for (const { libs } of atari) {
    const [lx, ly] = [...libs][0].split(',').map(Number);
    const clone = aiCloneGame(game);
    if (clone.placeStone(lx, ly)) return { type: 'place', x: lx, y: ly };
  }
  return null;
}

function aiEscape(game) {
  const color = game.current;
  const atari = aiGroupsByColor(game.board, color).filter(({ libs }) => libs.size === 1);
  for (const { libs } of atari) {
    const [lx, ly] = [...libs][0].split(',').map(Number);
    const clone = aiCloneGame(game);
    if (clone.placeStone(lx, ly)) return { type: 'place', x: lx, y: ly };
  }
  return null;
}

function aiShoreUp(game) {
  const color = game.current;
  const vulnerable = aiGroupsByColor(game.board, color).filter(({ libs }) => libs.size === 2);
  for (const { libs } of vulnerable) {
    for (const libStr of libs) {
      const [x, y] = libStr.split(',').map(Number);
      const clone = aiCloneGame(game);
      if (!clone.placeStone(x, y)) continue;
      const afterGroup = clone.board.getGroup(x, y);
      if (clone.board.getLiberties(afterGroup).size >= 3)
        return { type: 'place', x, y };
    }
  }
  return null;
}

function aiThreat(game, candidates) {
  const opp = game.current === 'black' ? 'white' : 'black';
  let bestMove = null;
  let bestSize = -1;
  for (const [x, y] of candidates) {
    const clone = aiCloneGame(game);
    if (!clone.placeStone(x, y)) continue;
    const visited = new Set();
    const N = clone.board.size;
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        const key = `${gx},${gy}`;
        if (clone.board.get(gx, gy) !== opp || visited.has(key)) continue;
        const group = clone.board.getGroup(gx, gy);
        group.forEach(([ax, ay]) => visited.add(`${ax},${ay}`));
        if (clone.board.getLiberties(group).size === 1 && group.length > bestSize) {
          bestSize = group.length;
          bestMove = { type: 'place', x, y };
        }
      }
    }
  }
  return bestMove;
}

function aiBfsDistances(board, N, seeds) {
  const dist = Array.from({ length: N }, () => new Float32Array(N).fill(Infinity));
  const queue = [];
  for (const [x, y] of seeds) { dist[y][x] = 0; queue.push([x, y]); }
  let head = 0;
  while (head < queue.length) {
    const [cx, cy] = queue[head++];
    for (const [nx, ny] of board.getNeighbors(cx, cy)) {
      if (board.get(nx, ny) === null && dist[ny][nx] === Infinity) {
        dist[ny][nx] = dist[cy][cx] + 1;
        queue.push([nx, ny]);
      }
    }
  }
  return dist;
}

function aiVoronoiScore(board, N, myColor) {
  const opp = myColor === 'black' ? 'white' : 'black';
  const mySeeds = [], oppSeeds = [];
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) === myColor) mySeeds.push([x, y]);
      else if (board.get(x, y) === opp) oppSeeds.push([x, y]);
    }
  const myDist  = aiBfsDistances(board, N, mySeeds);
  const oppDist = aiBfsDistances(board, N, oppSeeds);
  let score = 0;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++)
      if (board.get(x, y) === null && myDist[y][x] < oppDist[y][x]) score++;
  return score;
}

function aiInfluence(game, candidates) {
  const color = game.current;
  const N = game.boardSize;
  let bestMove = null, bestScore = -1;
  for (const [x, y] of candidates) {
    const clone = aiCloneGame(game);
    if (!clone.placeStone(x, y)) continue;
    if (aiLeavesOwnGroupAtari(clone, color)) continue;
    const score = aiVoronoiScore(clone.board, N, color);
    if (score > bestScore) { bestScore = score; bestMove = { type: 'place', x, y }; }
  }
  return bestMove;
}

function aiGetMove(game) {
  if (game.gameOver) return { type: 'pass' };
  const N = game.boardSize;
  const color = game.current;

  const capture = aiCapture(game);
  if (capture) return capture;

  const escape = aiEscape(game);
  if (escape) return escape;

  const shoreUp = aiShoreUp(game);
  if (shoreUp) return shoreUp;

  const candidates = [];
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (aiIsTrueEye(game.board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  aiShuffle(candidates);

  return aiThreat(game, candidates)
      || aiInfluence(game, candidates)
      || { type: 'pass' };
}

function scheduleComputerMove() {
  if (game.gameOver || game.current !== 'black') return;
  // Small delay so the board redraws before the (potentially slow) AI runs
  setTimeout(() => {
    if (game.gameOver || game.current !== 'black') return;
    const move = aiGetMove(game);
    if (move.type === 'place') {
      game.placeStone(move.x, move.y);
    } else {
      game.pass();
    }
    renderer.draw();
    updateUI();
  }, 50);
}

// ─── State ────────────────────────────────────────────────────────────────────

let game;
let renderer;
const canvas = document.getElementById('board-canvas');

function updateUI() {
  const g = game;
  document.getElementById('status-msg').textContent =
    g.gameOver ? 'Game over' :
    g.current === 'black' ? 'Computer thinking…' : 'Your turn (White)';

  // Captured counts: black captures = white stones captured by black
  document.getElementById('black-captures').textContent =
    `Captured: ${g.captured.white}`;
  document.getElementById('white-captures').textContent =
    `Captured: ${g.captured.black}`;

  // Active player highlight
  document.getElementById('black-info').classList.toggle('active', g.current === 'black' && !g.gameOver);
  document.getElementById('white-info').classList.toggle('active', g.current === 'white' && !g.gameOver);

  // Score panel
  const sp = document.getElementById('score-panel');
  if (g.gameOver && g.scores) {
    sp.style.display = 'block';
    const bs = g.scores.black;
    const ws = g.scores.white;
    document.getElementById('score-table').innerHTML = `
      <tr><td></td><td>Black</td><td>White</td></tr>
      <tr><td>Territory</td><td>${bs.territory}</td><td>${ws.territory}</td></tr>
      <tr><td>Captures</td><td>${bs.captures}</td><td>${ws.captures}</td></tr>
      <tr><td style="border-top:1px solid #4a4060;padding-top:6px">Total</td>
          <td style="border-top:1px solid #4a4060;padding-top:6px">${bs.total}</td>
          <td style="border-top:1px solid #4a4060;padding-top:6px">${ws.total}</td></tr>
    `;
    let winnerText;
    if (bs.total > ws.total) {
      winnerText = `Black wins by ${bs.total - ws.total} point${bs.total - ws.total !== 1 ? 's' : ''}!`;
    } else if (ws.total > bs.total) {
      winnerText = `White wins by ${ws.total - bs.total} point${ws.total - bs.total !== 1 ? 's' : ''}!`;
    } else {
      winnerText = 'Tie game!';
    }
    document.getElementById('winner-text').textContent = winnerText;
  } else {
    sp.style.display = 'none';
  }

  document.getElementById('pass-btn').disabled = g.gameOver;
}

function startGame(boardSize) {
  game = new Game(boardSize);
  renderer = new Renderer(canvas, game);
  renderer.draw();
  updateUI();
  scheduleComputerMove(); // computer is black and moves first
}

// ─── Pointer events: unified mouse / touch / pen drag-to-pan + tap-to-place ──

const DRAG_THRESHOLD = 6; // pixels before a press becomes a pan
let isPanning = false;
let pointerDownX = 0, pointerDownY = 0;
let panOriginX = 0, panOriginY = 0;

canvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  isPanning = false;
  pointerDownX = e.clientX;
  pointerDownY = e.clientY;
  panOriginX = renderer.panX;
  panOriginY = renderer.panY;
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', (e) => {
  if (e.buttons & 1) {
    // Drag-to-pan
    const dx = e.clientX - pointerDownX;
    const dy = e.clientY - pointerDownY;
    if (!isPanning && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
      isPanning = true;
      canvas.classList.add('panning');
      canvas.classList.remove('placing');
    }
    if (isPanning) {
      renderer.hoverPos = null;
      renderer.panX = panOriginX + dx;
      renderer.panY = panOriginY + dy;
      renderer.draw();
    }
  } else {
    // Hover — update ghost stone (only when it's the human's turn)
    const rect = canvas.getBoundingClientRect();
    renderer.hoverPos = game.current === 'white'
      ? renderer.fromCanvas(e.clientX - rect.left, e.clientY - rect.top)
      : null;
    renderer.draw();
  }
});

canvas.addEventListener('pointerleave', () => {
  renderer.hoverPos = null;
  renderer.draw();
});

canvas.addEventListener('pointerup', (e) => {
  canvas.classList.remove('panning', 'placing');
  if (!isPanning && game.current === 'white') {
    // Short tap / click → place human (white) stone
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const pos = renderer.fromCanvas(px, py);

    const legal = game.placeStone(pos.x, pos.y);
    renderer.draw();
    updateUI();

    if (!legal && game.illegalFlash) {
      setTimeout(() => { game.illegalFlash = null; renderer.draw(); }, 400);
    } else if (legal) {
      scheduleComputerMove();
    }
  }
  isPanning = false;
});

document.getElementById('pass-btn').addEventListener('click', () => {
  if (game.current !== 'white') return;
  game.pass();
  renderer.draw();
  updateUI();
  scheduleComputerMove();
});

document.getElementById('new-game-btn').addEventListener('click', () => {
  const size = parseInt(document.getElementById('size-select').value, 10);
  startGame(size);
});

document.getElementById('size-select').addEventListener('change', () => {
  const size = parseInt(document.getElementById('size-select').value, 10);
  startGame(size);
});

document.getElementById('overlap-select').addEventListener('change', () => {
  OVERLAP = parseInt(document.getElementById('overlap-select').value, 10);
  renderer.resize();
  renderer.draw();
});

window.addEventListener('resize', () => {
  renderer.resize();
  renderer.draw();
});

// Boot
startGame(9);
