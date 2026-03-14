// How many extra intersections to render beyond each tile edge.
// 0 = exact tile boundaries; increase to show wrapping rows/cols and make
// the toroidal topology more intuitive.
const OVERLAP = 1;

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

    // Compute which tile indices are needed to cover the canvas viewport,
    // so panning works infinitely in any direction.
    const dxMin = Math.floor((-this.panX - this.padding) / ts);
    const dxMax = Math.ceil((W - this.panX - this.padding) / ts);
    const dyMin = Math.floor((-this.panY - this.padding) / ts);
    const dyMax = Math.ceil((H - this.panY - this.padding) / ts);

    for (let dy = dyMin; dy <= dyMax; dy++) {
      for (let dx = dxMin; dx <= dxMax; dx++) {
        const offX = dx * ts;
        const offY = dy * ts;
        this.drawGrid(offX, offY);
        this.drawStones(offX, offY);
        this.drawLastMove(offX, offY);
        this.drawIllegalFlash(offX, offY);
      }
    }

    // Ghost stone is drawn in its own pass so each canvas position is
    // painted exactly once (the per-tile OVERLAP regions would otherwise
    // overlap and compound the alpha).
    this.drawGhostStone();
  }

  drawGrid(offX = 0, offY = 0) {
    const ctx = this.ctx;
    const N = this.game.boardSize;
    const cs = this.cellSize;
    ctx.strokeStyle = '#5a3a10';
    ctx.lineWidth = 1;

    const lo = -OVERLAP;
    const hi = N - 1 + OVERLAP;

    const [, topY] = this.toCanvas(0,  lo, offX, offY);
    const [, botY] = this.toCanvas(0,  hi, offX, offY);
    const [leftX]  = this.toCanvas(lo, 0,  offX, offY);
    const [rightX] = this.toCanvas(hi, 0,  offX, offY);

    for (let i = lo; i <= hi; i++) {
      const [cx] = this.toCanvas(i, 0, offX, offY);
      ctx.beginPath();
      ctx.moveTo(cx, topY);
      ctx.lineTo(cx, botY + cs); // +cs bridges gap to adjacent tile when OVERLAP=0
      ctx.stroke();

      const [, cy] = this.toCanvas(0, i, offX, offY);
      ctx.beginPath();
      ctx.moveTo(leftX, cy);
      ctx.lineTo(rightX + cs, cy);
      ctx.stroke();
    }
  }

  drawIllegalFlash(offX = 0, offY = 0) {
    const flash = this.game.illegalFlash;
    if (!flash) return;
    const N = this.game.boardSize;
    const r = this.cellSize * 0.38;
    this.ctx.save();
    this.ctx.globalAlpha = 0.35;
    this.ctx.fillStyle = '#ff4444';
    for (let gy = -OVERLAP; gy < N + OVERLAP; gy++) {
      for (let gx = -OVERLAP; gx < N + OVERLAP; gx++) {
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

    for (let gy = -OVERLAP; gy < N + OVERLAP; gy++) {
      for (let gx = -OVERLAP; gx < N + OVERLAP; gx++) {
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
    for (let gy = -OVERLAP; gy < N + OVERLAP; gy++) {
      for (let gx = -OVERLAP; gx < N + OVERLAP; gx++) {
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

let game;
let renderer;
const canvas = document.getElementById('board-canvas');

function updateUI() {
  const g = game;
  document.getElementById('status-msg').textContent = g.statusText();

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
    // Hover — update ghost stone
    const rect = canvas.getBoundingClientRect();
    renderer.hoverPos = renderer.fromCanvas(e.clientX - rect.left, e.clientY - rect.top);
    renderer.draw();
  }
});

canvas.addEventListener('pointerleave', () => {
  renderer.hoverPos = null;
  renderer.draw();
});

canvas.addEventListener('pointerup', (e) => {
  canvas.classList.remove('panning', 'placing');
  if (!isPanning) {
    // Short tap / click → place stone
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const pos = renderer.fromCanvas(px, py);

    const legal = game.placeStone(pos.x, pos.y);
    renderer.draw();
    updateUI();

    if (!legal && game.illegalFlash) {
      setTimeout(() => { game.illegalFlash = null; renderer.draw(); }, 400);
    }
  }
  isPanning = false;
});

document.getElementById('pass-btn').addEventListener('click', () => {
  game.pass();
  renderer.draw();
  updateUI();
});

document.getElementById('new-game-btn').addEventListener('click', () => {
  const size = parseInt(document.getElementById('size-select').value, 10);
  startGame(size);
});

document.getElementById('size-select').addEventListener('change', () => {
  const size = parseInt(document.getElementById('size-select').value, 10);
  startGame(size);
});

window.addEventListener('resize', () => {
  renderer.resize();
  renderer.draw();
});

// Boot
startGame(9);
