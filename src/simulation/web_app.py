"""Interactive browser chess simulation — Flask app on port 5060.

Left pane:  draggable ArUco canvas (IDs 1-32, white=1-16, black=17-32).
Right pane: live SVG chessboard + FEN + move history + LLM reasoning.
API /api/move: receive piece positions -> validate move -> LLM counter-move.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import chess
import chess.svg
from flask import Flask, jsonify, render_template_string, request

# Allow running from project root without install
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.strategy.llm_client import get_best_move  # noqa: E402

logger = logging.getLogger(__name__)

# Honour --model from main.py (CHESS_MODEL env); default to llm_client default
_MODEL_KW = {"model": os.environ["CHESS_MODEL"]} if os.environ.get("CHESS_MODEL") else {}

# ── ArUco ID → chess piece mapping (standard starting layout) ─────────────────
# White pieces (IDs 1-16): 1=Ra1,2=Nb1,3=Bc1,4=Qd1,5=Ke1,6=Bf1,7=Ng1,8=Rh1,9-16=Pa-Ph
# Black pieces (IDs 17-32): mirror of white
_WHITE_START: dict[int, tuple[str, str]] = {
    1: ("R", "a1"), 2: ("N", "b1"), 3: ("B", "c1"), 4: ("Q", "d1"),
    5: ("K", "e1"), 6: ("B", "f1"), 7: ("N", "g1"), 8: ("R", "h1"),
    9: ("P", "a2"), 10: ("P", "b2"), 11: ("P", "c2"), 12: ("P", "d2"),
    13: ("P", "e2"), 14: ("P", "f2"), 15: ("P", "g2"), 16: ("P", "h2"),
}
_BLACK_START: dict[int, tuple[str, str]] = {
    17: ("r", "a8"), 18: ("n", "b8"), 19: ("b", "c8"), 20: ("q", "d8"),
    21: ("k", "e8"), 22: ("b", "f8"), 23: ("n", "g8"), 24: ("r", "h8"),
    25: ("p", "a7"), 26: ("p", "b7"), 27: ("p", "c7"), 28: ("p", "d7"),
    29: ("p", "e7"), 30: ("p", "f7"), 31: ("p", "g7"), 32: ("p", "h7"),
}

# square name -> ArUco ID (at start, updated as pieces move)
def _build_initial_positions() -> dict[int, str]:
    """Return {marker_id: square} for all 32 pieces at start."""
    pos: dict[int, str] = {}
    for mid, (_, sq) in _WHITE_START.items():
        pos[mid] = sq
    for mid, (_, sq) in _BLACK_START.items():
        pos[mid] = sq
    return pos


def _positions_to_board_state(positions: dict[int, str]) -> dict[str, str]:
    """Map marker positions to {square: piece_char} for FEN serialization."""
    state: dict[str, str] = {}
    for mid, sq in positions.items():
        if 1 <= mid <= 16:
            piece_char = _WHITE_START[mid][0]
        elif 17 <= mid <= 32:
            piece_char = _BLACK_START[mid][0]
        else:
            continue
        state[sq] = piece_char
    return state


def _board_state_to_fen(board_state: dict[str, str], active: str = "w") -> str:
    """Minimal FEN from square→piece_char. Castling/ep cleared for sim."""
    from src.vision.fen_serializer import serialize_fen
    return serialize_fen(board_state, active_color=active, castling="-", en_passant="-")


# ── Application state (single-game, in-memory) ────────────────────────────────

class GameState:
    """Mutable game state for one session."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = chess.Board()  # authoritative python-chess board
        self.positions: dict[int, str] = _build_initial_positions()
        self.move_history: list[str] = []
        self.last_reasoning: str = ""
        self.last_llm_move: str = ""

    def svg(self) -> str:
        return chess.svg.board(self.board, size=400)

    def fen(self) -> str:
        return self.board.fen()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen(),
            "svg": self.svg(),
            "move_history": self.move_history,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "reasoning": self.last_reasoning,
            "llm_move": self.last_llm_move,
            "is_game_over": self.board.is_game_over(),
            "outcome": str(self.board.outcome()) if self.board.is_game_over() else None,
            "positions": {str(k): v for k, v in self.positions.items()},
        }


_game = GameState()

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "ur-chess-sim"


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>UR Chess LLM Arena — Simulation</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { display: flex; flex-direction: column; height: 100vh; font-family: monospace; background: #1a1a2e; color: #eee; }
  header { background: #16213e; padding: 10px 20px; font-size: 1.2em; font-weight: bold; color: #e94560; letter-spacing: 2px; }
  #main { display: flex; flex: 1; overflow: hidden; }
  /* ── Left pane ── */
  #left { flex: 0 0 540px; display: flex; flex-direction: column; border-right: 2px solid #0f3460; background: #16213e; }
  #canvas-wrap { position: relative; width: 512px; height: 512px; margin: 16px auto 0; }
  #chess-canvas { border: 3px solid #0f3460; cursor: default; }
  #left-controls { padding: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
  button { background: #0f3460; color: #e0e0e0; border: 1px solid #e94560; padding: 6px 14px; cursor: pointer; border-radius: 4px; font-family: monospace; }
  button:hover { background: #e94560; color: #fff; }
  #status-bar { padding: 8px 12px; font-size: 0.8em; color: #aaa; border-top: 1px solid #0f3460; }
  /* ── Right pane ── */
  #right { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; overflow-y: auto; }
  #svg-wrap { background: #fff; border-radius: 6px; padding: 4px; display: inline-block; max-width: 420px; }
  #svg-wrap svg { display: block; }
  .info-box { background: #0f3460; border-radius: 6px; padding: 10px; }
  .info-box h3 { color: #e94560; margin-bottom: 6px; font-size: 0.9em; }
  .info-box pre { font-size: 0.78em; white-space: pre-wrap; word-break: break-all; color: #ccc; }
  #move-list { list-style: none; display: flex; flex-wrap: wrap; gap: 4px; }
  #move-list li { background: #16213e; border: 1px solid #0f3460; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
  #turn-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.85em; }
  .white-turn { background: #eee; color: #111; }
  .black-turn { background: #222; color: #eee; border: 1px solid #888; }
  #overlay { display:none; position:fixed; top:0;left:0;right:0;bottom:0; background:rgba(0,0,0,0.65); z-index:100; align-items:center; justify-content:center; }
  #overlay.show { display:flex; }
  #overlay-msg { background:#16213e; border:2px solid #e94560; padding:30px 40px; border-radius:10px; font-size:1.3em; color:#eee; text-align:center; }
</style>
</head>
<body>
<header>&#9812; UR Chess LLM Arena &mdash; Interactive Simulation</header>
<div id="main">
  <!-- Left: ArUco canvas -->
  <div id="left">
    <div id="canvas-wrap">
      <canvas id="chess-canvas" width="512" height="512"></canvas>
    </div>
    <div id="left-controls">
      <button onclick="resetGame()">&#8635; New Game</button>
      <button onclick="undoMove()">&#8592; Undo</button>
      <span id="turn-badge" class="white-turn">White to move</span>
    </div>
    <div id="status-bar" id="status-bar">Drag white pieces to make a move.</div>
  </div>
  <!-- Right: live SVG + telemetry -->
  <div id="right">
    <div id="svg-wrap">LOADING...</div>
    <div class="info-box">
      <h3>FEN</h3>
      <pre id="fen-text">...</pre>
    </div>
    <div class="info-box">
      <h3>Move History</h3>
      <ul id="move-list"></ul>
    </div>
    <div class="info-box">
      <h3>LLM Strategy Reasoning</h3>
      <pre id="reasoning-text">Waiting for black&apos;s turn...</pre>
    </div>
  </div>
</div>
<!-- Game-over overlay -->
<div id="overlay"><div id="overlay-msg"></div></div>

<script>
// ── Board geometry ────────────────────────────────────────────────────────────
const CANVAS_SIZE = 512;
const BOARD_OFFSET = 32;  // pixels for rank/file labels
const BOARD_SIZE = CANVAS_SIZE - BOARD_OFFSET;
const SQ_SIZE = BOARD_SIZE / 8;  // 60px

const FILES = 'abcdefgh';
const RANKS = '87654321';

// Marker ID -> piece info
const WHITE_IDS = {};
const BLACK_IDS = {};
const ID_TO_PIECE = {};  // id -> {piece, color, startSq}
[
  [1,'R','a'],[2,'N','b'],[3,'B','c'],[4,'Q','d'],[5,'K','e'],[6,'B','f'],[7,'N','g'],[8,'R','h'],
  [9,'P','a'],[10,'P','b'],[11,'P','c'],[12,'P','d'],[13,'P','e'],[14,'P','f'],[15,'P','g'],[16,'P','h'],
].forEach(([id,piece,file]) => {
  const rank = (id <= 8) ? '1' : '2';
  WHITE_IDS[id] = {piece, sq: file+rank, color:'white'};
  ID_TO_PIECE[id] = {piece, color:'white'};
});
[
  [17,'r','a'],[18,'n','b'],[19,'b','c'],[20,'q','d'],[21,'k','e'],[22,'b','f'],[23,'n','g'],[24,'r','h'],
  [25,'p','a'],[26,'p','b'],[27,'p','c'],[28,'p','d'],[29,'p','e'],[30,'p','f'],[31,'p','g'],[32,'p','h'],
].forEach(([id,piece,file]) => {
  const rank = (id <= 24) ? '8' : '7';
  BLACK_IDS[id] = {piece, sq: file+rank, color:'black'};
  ID_TO_PIECE[id] = {piece, color:'black'};
});

// ── State ─────────────────────────────────────────────────────────────────────
let positions = {};   // markerId -> square string e.g. 'e4'
let serverTurn = 'white';
let dragging = null;  // {id, offsetX, offsetY, pixelX, pixelY}
let pieceImages = {}; // sq -> image element (Unicode rendering onto canvas)
let gameOver = false;

function sqToPixel(sq) {
  const file = FILES.indexOf(sq[0]);
  const rank = RANKS.indexOf(sq[1]);
  return {
    x: BOARD_OFFSET + file * SQ_SIZE + SQ_SIZE / 2,
    y: BOARD_OFFSET + rank * SQ_SIZE + SQ_SIZE / 2,
  };
}

function pixelToSq(px, py) {
  const col = Math.floor((px - BOARD_OFFSET) / SQ_SIZE);
  const row = Math.floor((py - BOARD_OFFSET) / SQ_SIZE);
  if (col < 0 || col > 7 || row < 0 || row > 7) return null;
  return FILES[col] + RANKS[row];
}

// Chess Unicode glyphs
const GLYPHS = {
  wK:'♔', wQ:'♕', wR:'♖', wB:'♗', wN:'♘', wP:'♙',
  bK:'♚', bQ:'♛', bR:'♜', bB:'♝', bN:'♞', bP:'♟',
};
function glyph(id) {
  const info = ID_TO_PIECE[id];
  if (!info) return '?';
  const key = (info.color==='white'?'w':'b') + info.piece.toUpperCase();
  return GLYPHS[key] || '?';
}

// ── Canvas drawing ────────────────────────────────────────────────────────────
const canvas = document.getElementById('chess-canvas');
const ctx = canvas.getContext('2d');

function drawBoard() {
  // Board background squares
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? '#f0d9b5' : '#b58863';
      ctx.fillRect(BOARD_OFFSET + c * SQ_SIZE, BOARD_OFFSET + r * SQ_SIZE, SQ_SIZE, SQ_SIZE);
    }
  }
  // Labels
  ctx.fillStyle = '#555';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  for (let c = 0; c < 8; c++) {
    ctx.fillText(FILES[c], BOARD_OFFSET + c * SQ_SIZE + SQ_SIZE/2, BOARD_OFFSET - 6);
  }
  ctx.textAlign = 'right';
  for (let r = 0; r < 8; r++) {
    ctx.fillText(RANKS[r], BOARD_OFFSET - 4, BOARD_OFFSET + r * SQ_SIZE + SQ_SIZE/2 + 4);
  }
}

function drawPieces() {
  ctx.font = `${SQ_SIZE * 0.72}px serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const [idStr, sq] of Object.entries(positions)) {
    const id = parseInt(idStr);
    if (dragging && dragging.id === id) continue;  // drawn separately
    const {x, y} = sqToPixel(sq);
    // Shadow for white pieces
    const info = ID_TO_PIECE[id];
    if (info && info.color === 'white') {
      ctx.shadowColor = 'rgba(0,0,0,0.4)';
      ctx.shadowBlur = 4;
    } else {
      ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0;
    }
    ctx.fillText(glyph(id), x, y);
  }
  ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0;
  // Draw dragging piece on top
  if (dragging) {
    ctx.font = `${SQ_SIZE * 0.78}px serif`;
    ctx.shadowColor = 'rgba(0,100,255,0.5)'; ctx.shadowBlur = 8;
    ctx.fillText(glyph(dragging.id), dragging.pixelX, dragging.pixelY);
    ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0;
  }
}

function render() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  drawBoard();
  drawPieces();
}

// ── Drag & drop (white only when white's turn) ────────────────────────────────
function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches ? e.touches[0] : e;
  return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
}

function findMarkerAt(px, py) {
  const sq = pixelToSq(px, py);
  if (!sq) return null;
  for (const [idStr, msq] of Object.entries(positions)) {
    if (msq === sq) return parseInt(idStr);
  }
  return null;
}

canvas.addEventListener('mousedown', e => startDrag(getCanvasPos(e)));
canvas.addEventListener('touchstart', e => { e.preventDefault(); startDrag(getCanvasPos(e)); }, {passive:false});

function startDrag(pos) {
  if (gameOver || serverTurn !== 'white') return;
  const id = findMarkerAt(pos.x, pos.y);
  if (!id || id > 16) return;  // only white pieces
  dragging = {id, pixelX: pos.x, pixelY: pos.y, fromSq: positions[id]};
}

canvas.addEventListener('mousemove', e => moveDrag(getCanvasPos(e)));
canvas.addEventListener('touchmove', e => { e.preventDefault(); moveDrag(getCanvasPos(e)); }, {passive:false});

function moveDrag(pos) {
  if (!dragging) return;
  dragging.pixelX = pos.x;
  dragging.pixelY = pos.y;
  render();
}

canvas.addEventListener('mouseup', e => endDrag(getCanvasPos(e)));
canvas.addEventListener('touchend', e => { e.preventDefault(); endDrag(getCanvasPos(e.changedTouches[0] || {clientX:0,clientY:0})); }, {passive:false});

function endDrag(pos) {
  if (!dragging) return;
  const toSq = pixelToSq(pos.x, pos.y);
  const fromSq = dragging.fromSq;
  const id = dragging.id;
  dragging = null;
  if (!toSq || toSq === fromSq) { render(); return; }
  sendMove(id, fromSq, toSq);
}

// ── API calls ─────────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

async function sendMove(movedId, fromSq, toSq) {
  setStatus(`Sending move: ${fromSq}${toSq}...`);
  // Optimistically update canvas
  positions[movedId] = toSq;
  render();
  try {
    const resp = await fetch('/api/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({positions, moved_id: movedId, from_sq: fromSq, to_sq: toSq}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      setStatus('Illegal move: ' + (data.error || 'unknown'));
      positions[movedId] = fromSq;  // revert
      render();
      return;
    }
    applyServerState(data);
  } catch(err) {
    setStatus('Network error: ' + err);
    positions[movedId] = fromSq;
    render();
  }
}

function applyServerState(data) {
  // Update positions from server authoritative state
  if (data.positions) {
    positions = {};
    for (const [k,v] of Object.entries(data.positions)) positions[parseInt(k)] = v;
  }
  serverTurn = data.turn;
  gameOver = data.is_game_over;

  // Right pane
  document.getElementById('svg-wrap').innerHTML = data.svg || '';
  document.getElementById('fen-text').textContent = data.fen || '';
  const ml = document.getElementById('move-list');
  ml.innerHTML = '';
  (data.move_history || []).forEach((m,i) => {
    const li = document.createElement('li');
    li.textContent = (i%2===0 ? Math.floor(i/2+1)+'. ' : '') + m;
    ml.appendChild(li);
  });
  document.getElementById('reasoning-text').textContent = data.reasoning || '';
  const badge = document.getElementById('turn-badge');
  badge.textContent = data.is_game_over ? 'Game over' : (data.turn==='white' ? 'White to move' : 'Black to move');
  badge.className = data.turn==='white' ? 'white-turn' : 'black-turn';

  if (data.is_game_over) {
    document.getElementById('overlay-msg').textContent = 'Game Over: ' + (data.outcome||'');
    document.getElementById('overlay').classList.add('show');
  }
  setStatus(data.llm_move ? `LLM played: ${data.llm_move}` : 'Your turn (drag white piece)');
  render();
}

async function resetGame() {
  gameOver = false;
  document.getElementById('overlay').classList.remove('show');
  const resp = await fetch('/api/reset', {method:'POST'});
  const data = await resp.json();
  // Re-init positions from server
  if (data.positions) {
    positions = {};
    for (const [k,v] of Object.entries(data.positions)) positions[parseInt(k)] = v;
  }
  serverTurn = 'white';
  applyServerState(data);
  setStatus('New game started. Drag white pieces.');
}

async function undoMove() {
  const resp = await fetch('/api/undo', {method:'POST'});
  const data = await resp.json();
  if (data.error) { setStatus(data.error); return; }
  applyServerState(data);
  setStatus('Move undone.');
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  const resp = await fetch('/api/state');
  const data = await resp.json();
  if (data.positions) {
    positions = {};
    for (const [k,v] of Object.entries(data.positions)) positions[parseInt(k)] = v;
  }
  serverTurn = data.turn;
  applyServerState(data);
}

init();
// Animation loop
function loop() { if (dragging) render(); requestAnimationFrame(loop); }
loop();
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/api/state")
def api_state():
    return jsonify(_game.to_dict())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _game.reset()
    _undo_stack.clear()
    return jsonify(_game.to_dict())


# Undo stack: list of (board_fen, positions_snapshot, history_length)
_undo_stack: list[tuple[str, dict[int, str], int]] = []
_SQUARE_RE = re.compile(r"^[a-h][1-8]$")


@app.route("/api/move", methods=["POST"])
def api_move():
    """Process human white move, run LLM for black counter-move."""
    global _undo_stack
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object body required"}), 400

    # Client sends full current positions + the specific move made
    client_positions: dict[int, str] = {}
    try:
        raw_positions = data.get("positions") or {}
        if not isinstance(raw_positions, dict):
            raise ValueError("positions must be an object")
        for k, v in raw_positions.items():
            mid = int(k)
            if not 1 <= mid <= 32:
                raise ValueError("marker id out of range")
            v = str(v)
            if v != "__captured__" and not _SQUARE_RE.match(v):
                raise ValueError("bad square")
            client_positions[mid] = v
        moved_id: int = int(data.get("moved_id", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "invalid positions or moved_id"}), 400

    from_sq: str = str(data.get("from_sq", ""))
    to_sq: str = str(data.get("to_sq", ""))

    if not _SQUARE_RE.match(from_sq) or not _SQUARE_RE.match(to_sq):
        return jsonify({"error": "from_sq/to_sq must be squares a1-h8"}), 400

    # Validate the move against python-chess authoritative board
    if _game.board.turn != chess.WHITE:
        return jsonify({"error": "not white's turn"}), 400

    uci_str = from_sq + to_sq
    # Check for pawn promotion (to 8th rank)
    if (moved_id in _WHITE_START and _WHITE_START[moved_id][0] == "P"
            and to_sq[1] == "8"):
        uci_str += "q"  # auto-promote to queen

    try:
        move = chess.Move.from_uci(uci_str)
    except ValueError:
        return jsonify({"error": f"invalid UCI: {uci_str}"}), 400

    if move not in _game.board.legal_moves:
        return jsonify({"error": f"illegal move: {uci_str}"}), 400

    # Save undo snapshot BEFORE applying
    _undo_stack.append((_game.board.fen(), dict(_game.positions), len(_game.move_history)))

    # Apply white move to board
    _game.board.push(move)
    _game.positions.update(client_positions)
    _game.move_history.append(move.uci())
    _game.last_reasoning = ""
    _game.last_llm_move = ""

    # Check game over after white move
    if _game.board.is_game_over():
        return jsonify(_game.to_dict())

    # ── LLM counter-move for black ────────────────────────────────────────────
    try:
        llm_move = get_best_move(_game.board, **_MODEL_KW)
        _game.last_reasoning = (
            f"LLM evaluated position after {move.uci()}.\n"
            f"Legal black moves: {', '.join(m.uci() for m in _game.board.legal_moves)}\n"
            f"Chosen: {llm_move.uci()}"
        )
        _game.last_llm_move = llm_move.uci()
        # Apply black move
        _game.board.push(llm_move)
        _game.move_history.append(llm_move.uci())
        # Update black marker position
        black_from = llm_move.from_square
        black_to = llm_move.to_square
        from_alg = chess.square_name(black_from)
        to_alg = chess.square_name(black_to)
        # Find marker at from_alg and move it
        for mid, sq in list(_game.positions.items()):
            if sq == from_alg and 17 <= mid <= 32:
                _game.positions[mid] = to_alg
                break
        # Remove any captured white marker at to_alg
        for mid, sq in list(_game.positions.items()):
            if sq == to_alg and 1 <= mid <= 16:
                # Mark captured by moving off-board
                _game.positions[mid] = "__captured__"
                break
    except Exception as exc:
        logger.warning("LLM move failed: %s", exc)
        _game.last_reasoning = "LLM error: move unavailable (see server log)"

    return jsonify(_game.to_dict())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    global _undo_stack
    if not _undo_stack:
        return jsonify({"error": "nothing to undo"}), 400
    fen, pos, hist_len = _undo_stack.pop()
    _game.board = chess.Board(fen)
    _game.positions = pos
    del _game.move_history[hist_len:]
    _game.last_reasoning = ""
    _game.last_llm_move = ""
    return jsonify(_game.to_dict())


# ── Entry point ───────────────────────────────────────────────────────────────

def run(host: str = "127.0.0.1", port: int = 5060, debug: bool = False) -> None:
    """Start the simulation web server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
