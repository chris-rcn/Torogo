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

// ─── Monte Carlo AI (computer plays black) ────────────────────────────────────

const MC_CANDIDATE_PLAYOUTS = 50;

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

function aiApplyFast(game, x, y) {
  game.board.set(x, y, game.current);
  const cap = game.board.captureGroups(x, y);
  game.captured.black += cap.black.length;
  game.captured.white += cap.white.length;
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black.length + cap.white.length;
}

function aiPlayRandom(game) {
  const size = game.boardSize;
  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (game.board.get(x, y) === null) empty.push([x, y]);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;

    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const [x, y] = empty[i];
      empty[i] = empty[end - 1];
      empty[end - 1] = [x, y];
      end--;

      if (aiIsTrueEye(game.board, x, y, game.current)) continue;

      const neighbors = game.board.getNeighbors(x, y);
      if (neighbors.some(([nx, ny]) => game.board.get(nx, ny) === null)) {
        const captures = aiApplyFast(game, x, y);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (captures > 0) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }

      const capBefore = game.captured.black + game.captured.white;
      if (game.placeStone(x, y)) {
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (game.captured.black + game.captured.white > capBefore) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }
    }

    if (!placed) {
      game.pass();
      moves++;
    }
  }

  if (!game.gameOver) game.endGame();
}

function aiRandomMove(game) {
  const N = game.boardSize;
  const color = game.current;
  const board = game.board;
  const candidates = [];
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) !== null) continue;
      if (aiIsTrueEye(board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  while (candidates.length > 0) {
    const i = Math.floor(Math.random() * candidates.length);
    const [x, y] = candidates[i];
    candidates[i] = candidates[candidates.length - 1];
    candidates.pop();
    if (board.getNeighbors(x, y).some(([nx, ny]) => board.get(nx, ny) === null))
      return { type: 'place', x, y };
    const clone = game.clone();
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }
  return { type: 'pass' };
}

function aiGetMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const player = game.current;
  const candidates = [];
  for (let y = 0; y < game.boardSize; y++) {
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) candidates.push({ type: 'place', x, y });
    }
  }
  candidates.push({ type: 'pass' });

  const stats = candidates.map(() => ({ wins: 0, plays: 0 }));

  for (let idx = 0; idx < candidates.length; idx++) {
    const move = candidates[idx];
    for (let t = 0; t < MC_CANDIDATE_PLAYOUTS; t++) {
      const clone = game.clone();
      if (move.type === 'place') clone.placeStone(move.x, move.y);
      else clone.pass();
      aiPlayRandom(clone);
      const s = clone.scores;
      const winner = s.black.total > s.white.total ? 'black'
                   : s.white.total > s.black.total ? 'white'
                   : null;
      stats[idx].plays++;
      if (winner === player) stats[idx].wins++;
    }
  }

  let bestIdx = 0, bestRatio = -1;
  for (let i = 0; i < candidates.length; i++) {
    const ratio = stats[i].plays > 0 ? stats[i].wins / stats[i].plays : 0;
    if (ratio > bestRatio) { bestRatio = ratio; bestIdx = i; }
  }
  return candidates[bestIdx];
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
  const statusEl = document.getElementById('status-msg');
  const thinking = !g.gameOver && g.current === 'black';
  statusEl.textContent =
    g.gameOver ? 'Game over' :
    thinking ? 'Computer thinking…' : 'Your turn (White)';
  statusEl.classList.toggle('thinking', thinking);

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
  startGame(13);
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
