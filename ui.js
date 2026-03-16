// How many extra intersections to render beyond each tile edge.
// 0 = exact tile boundaries; increase to show wrapping rows/cols and make
// the toroidal topology more intuitive.
const OVERLAP = 2;

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
    const N = this.game.boardSize;
    if (!legalMovesSet || !legalMovesSet.has(h.y * N + h.x)) return;
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

// ─── AI (calls into ai/rave.js loaded via <script> tag) ────────────────────────

const UI_BUDGET_MS = 2000; // 2 seconds per move for interactive play

let computerBusy = false;
let computerPassedLast = false;
let moveNumber = 0;

function logMove(player, detail, info) {
  const num = String(moveNumber).padStart(3);
  const p = player === 'C' ? 'computer' : 'human   ';
  console.log(`Move ${num}  ${p}  ${detail}` + (info != null ? `  — ${info}` : ''));
}

// Set of (y*N + x) indices that are legal moves for the human on their turn.
// null when it is not the human's turn.
let legalMovesSet = null;

function buildLegalMoves() {
  const N = game.boardSize;
  legalMovesSet = new Set();
  if (game.gameOver || game.current !== 'white') return;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) legalMovesSet.add(y * N + x);
    }
  }
}

// Smoothly animate renderer.panX/panY to (targetX, targetY) over durationMs,
// then call onComplete.  Uses an ease-in-out cubic curve.
function animatePan(targetX, targetY, durationMs, onComplete) {
  const startX = renderer.panX;
  const startY = renderer.panY;
  const startTime = performance.now();

  function frame(now) {
    const t = Math.min((now - startTime) / durationMs, 1);
    // Ease-in-out cubic
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    renderer.panX = startX + (targetX - startX) * ease;
    renderer.panY = startY + (targetY - startY) * ease;
    renderer.draw();
    if (t < 1) {
      requestAnimationFrame(frame);
    } else {
      onComplete();
    }
  }

  requestAnimationFrame(frame);
}

function scheduleComputerMove() {
  if (game.gameOver || game.current !== 'black') return;
  computerBusy = true;
  legalMovesSet = null;
  updateUI();
  renderer.draw();
  requestAnimationFrame(() => {
    setTimeout(() => {
      if (game.gameOver || game.current !== 'black') {
        computerBusy = false;
        return;
      }
      const move = getMove(game, UI_BUDGET_MS);

      const applyMove = () => {
        moveNumber++;
        if (move.type === 'place') {
          computerPassedLast = false;
          logMove('C', `(${move.x}, ${move.y})`, move.info);
          game.placeStone(move.x, move.y);
        } else {
          computerPassedLast = true;
          logMove('C', 'pass', move.info);
          game.pass();
        }
        renderer.draw();
        // Brief cooldown so queued pointer events don't sneak through
        setTimeout(() => { buildLegalMoves(); computerBusy = false; updateUI(); }, 50);
      };

      if (move.type === 'place') {
        // Pan so the computer's chosen intersection is centred on the canvas,
        // then drop the stone once the animation completes.
        // Wrap the delta by tileSize so we always take the shortest path
        // (the board is toroidal, so ±N cells is the same visual position).
        const W = canvas.width;
        const H = canvas.height;
        const ts = renderer.tileSize();
        const rawTargetX = W / 2 - renderer.padding - move.x * renderer.cellSize;
        const rawTargetY = H / 2 - renderer.padding - move.y * renderer.cellSize;
        let dx = rawTargetX - renderer.panX;
        let dy = rawTargetY - renderer.panY;
        dx -= Math.round(dx / ts) * ts;
        dy -= Math.round(dy / ts) * ts;
        animatePan(renderer.panX + dx, renderer.panY + dy, 500, applyMove);
      } else {
        applyMove();
      }
    }, 0);
  });
}

// ─── State ────────────────────────────────────────────────────────────────────

let game;
let renderer;
const canvas = document.getElementById('board-canvas');

function updateUI() {
  const g = game;
  const statusEl = document.getElementById('status-msg');
  const thinking = !g.gameOver && g.current === 'black';

  // Score bar — always visible
  if (g.gameOver && g.scores) {
    const bs = g.scores.black;
    const ws = g.scores.white;
    document.getElementById('black-score-display').textContent = `Black: ${bs.total}`;
    document.getElementById('white-score-display').textContent = `White: ${ws.total} (komi ${g.komi})`;

    let winnerText;
    if (bs.total > ws.total) {
      winnerText = 'Black wins!';
    } else if (ws.total > bs.total) {
      winnerText = 'White wins!';
    } else {
      winnerText = 'Tie game!';
    }
    statusEl.textContent = winnerText;
    statusEl.classList.remove('thinking');
  } else {
    document.getElementById('black-score-display').textContent = 'Black: —';
    document.getElementById('white-score-display').textContent = `White: — (komi ${g.komi})`;
    statusEl.textContent = thinking ? 'Computer thinking…' : 'Your turn (white)';
    statusEl.classList.toggle('thinking', thinking);
  }

  const isHumanTurn = !g.gameOver && !computerBusy && g.current === 'white';
  document.getElementById('pass-btn').style.display = isHumanTurn ? '' : 'none';
  // Shield the canvas (pointer-events: none) only while the computer is
  // thinking — not after game over, so the player can still pan the board.
  canvas.classList.toggle('shielded', !g.gameOver && !isHumanTurn);

  document.getElementById('computer-passed-label').style.display =
    computerPassedLast ? '' : 'none';
}

function startGame(boardSize) {
  computerPassedLast = false;
  moveNumber = 0;
  console.log(`[Game] new game started (${boardSize}×${boardSize})`);
  game = new Game(boardSize);
  renderer = new Renderer(canvas, game);
  renderer.draw();
  buildLegalMoves();
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
    renderer.hoverPos = !game.gameOver && game.current === 'white'
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
  if (!isPanning && !computerBusy && game.current === 'white') {
    // Short tap / click → place human (white) stone
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const pos = renderer.fromCanvas(px, py);

    const N = game.boardSize;
    const isLegal = legalMovesSet && legalMovesSet.has(pos.y * N + pos.x);

    if (isLegal) {
      // Place the stone immediately so it appears (with last-move dot) before
      // the pan animation starts.
      moveNumber++;
      logMove('H', `(${pos.x}, ${pos.y})`);
      game.placeStone(pos.x, pos.y);
      renderer.draw();
      updateUI();

      const W = canvas.width;
      const H = canvas.height;
      const cs = renderer.cellSize;
      // Use the raw pixel offset (unsnapped, unwrapped) so the pan target
      // centres on whichever visual tile-copy the user actually clicked,
      // not the base-tile copy (which can be in the opposite direction when
      // fromCanvas wraps the grid coordinate across a board edge).
      const rx = px - renderer.padding - renderer.panX;
      const ry = py - renderer.padding - renderer.panY;
      const targetPanX = W / 2 - renderer.padding - Math.round(rx / cs) * cs;
      const targetPanY = H / 2 - renderer.padding - Math.round(ry / cs) * cs;
      animatePan(targetPanX, targetPanY, 500, scheduleComputerMove);
    } else {
      console.log(`            human     (${pos.x}, ${pos.y})  — illegal`);
      const legal = game.placeStone(pos.x, pos.y);
      renderer.draw();
      updateUI();
      if (!legal && game.illegalFlash) {
        setTimeout(() => { game.illegalFlash = null; renderer.draw(); }, 400);
      }
    }
  }
  isPanning = false;
});

document.getElementById('pass-btn').addEventListener('click', () => {
  if (computerBusy || game.current !== 'white') return;
  computerPassedLast = false;
  moveNumber++;
  logMove('H', 'pass');
  game.pass();
  renderer.draw();
  updateUI();
  scheduleComputerMove();
});

const DEFAULT_BOARD_SIZE = 13;

function sizeFromURL() {
  const raw = new URLSearchParams(location.search).get('size');
  const n = parseInt(raw, 10);
  return Number.isInteger(n) && n >= 5 && n <= 19 ? n : DEFAULT_BOARD_SIZE;
}

const initialBoardSize = sizeFromURL();

document.getElementById('new-game-btn').addEventListener('click', () => {
  startGame(initialBoardSize);
});



window.addEventListener('resize', () => {
  renderer.resize();
  renderer.draw();
});

// Boot
startGame(initialBoardSize);
