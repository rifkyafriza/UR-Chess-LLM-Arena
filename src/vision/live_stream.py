"""Live camera / video-file web stream with ArUco HUD overlay and dual-pane SVG board.

Flask MJPEG server:
    GET /          — HTML dual-pane viewer (camera + SVG board)
    GET /video     — MJPEG stream
    GET /state     — JSON board state: mode, fen, svg, piece_count
    GET /board_svg — Raw SVG of current recognized position

Detection pipeline per frame:
    1. Contour grid detection  → board_locator_grid.localize_board  (primary)
    2. Fallback: ArUco corner markers (IDs 40-43) → board_locator_aruco.locate_board
    3. Detect piece markers (IDs 1-32) on full-resolution frame via homography matrix
    4. Stabilize via TemporalPieceTracker; build piece-type board state
    5. Generate FEN + SVG from board state
    6. Render HUD overlay on raw frame
    7. JPEG-encode and push to MJPEG boundary

Run:
    python -m src.vision.live_stream [--camera 0] [--port 5050]
    python -m src.vision.live_stream --video path/to/video.mp4 [--port 5050]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Optional

import chess
import chess.svg
import cv2
import numpy as np
from flask import Flask, Response, jsonify

from src.vision import board_locator_aruco, board_locator_grid, piece_detector
from src.vision.piece_detector import (
    TemporalPieceTracker,
    build_piece_type_board_state,
    detect_pieces_from_frame,
)
from src.vision.fen_serializer import serialize_fen

# ── Config ────────────────────────────────────────────────────────────────────

_JPEG_QUALITY = 75      # lower = faster encoding, less bandwidth
_TARGET_FPS   = 30      # cap capture loop; actual may be lower
_FRAME_DELAY  = 1.0 / _TARGET_FPS

# ── Shared state (capture thread → Flask threads) ─────────────────────────────

_lock        = threading.Lock()
_jpeg_bytes: bytes           = b""
_board_state: dict[str, str] = {}
_mode: str                   = "none"   # "aruco" | "grid" | "none"
_current_fen: str            = ""
_current_svg: str            = ""

# ── Temporal tracker singleton (lives for the lifetime of the stream) ─────────

_tracker = TemporalPieceTracker(ttl=8, max_pieces=32)

# ── Capture thread ────────────────────────────────────────────────────────────

def _open_source(camera_index: Optional[int], video_path: Optional[str]) -> cv2.VideoCapture:
    """Open camera index or video file; raise RuntimeError on failure."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        return cap
    idx = camera_index if camera_index is not None else 0
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {idx}")
    return cap


def _capture_loop(
    camera_index: Optional[int] = 0,
    video_path: Optional[str] = None,
) -> None:
    """Background thread: capture, detect, encode."""
    cap = _open_source(camera_index, video_path)
    is_video = bool(video_path)
    try:
        _capture_frames(cap, is_video=is_video, video_path=video_path)
    finally:
        cap.release()


def _capture_frames(
    cap: cv2.VideoCapture,
    *,
    is_video: bool = False,
    video_path: Optional[str] = None,
) -> None:
    global _jpeg_bytes, _board_state, _mode, _current_fen, _current_svg

    t_next = time.monotonic()

    while True:
        ok, frame = cap.read()

        # Loop video on EOF
        if (not ok or frame is None) and is_video and video_path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if not ok or frame is None:
            time.sleep(0.05)
            continue

        warped: Optional[np.ndarray] = None
        matrix: Optional[np.ndarray] = None
        detected_corners: dict       = {}
        current_mode                 = "none"

        # --- Mode 1: contour grid (primary) ---
        grid_result = board_locator_grid.localize_board(frame)
        if grid_result is not None:
            warped       = grid_result["warped"]
            matrix       = grid_result["matrix"]
            current_mode = "grid"

        # --- Mode 2: ArUco corner markers (fallback when grid fails) ---
        if warped is None:
            warped_aruco, detected_corners = board_locator_aruco.locate_board(frame)
            if warped_aruco is not None:
                # board_locator_aruco does not expose matrix; warp is pre-computed.
                # Use the warped image for piece detection via detect_pieces.
                warped       = warped_aruco
                current_mode = "aruco"
                matrix       = None  # will use detect_pieces(warped) path below

        # --- Piece detection ---
        raw_pieces: list[dict] = []
        if warped is not None:
            if matrix is not None:
                # Full-res frame + enhanced board-ROI pass for small markers.
                # Pass warped so detect_pieces_from_frame can run the 2× Unsharp
                # Mask third pass board-scoped (no off-board false positives).
                raw_pieces = detect_pieces_from_frame(frame, matrix, warped=warped)
            else:
                # ArUco-warped path: detect on warped image directly.
                # detect_pieces already contains the enhanced-ROI third pass.
                raw_pieces = piece_detector.detect_pieces(warped)

        stabilized = _tracker.update(raw_pieces)

        # --- Board state and FEN/SVG ---
        board_state = build_piece_type_board_state(stabilized)

        fen = serialize_fen(board_state) if board_state else ""
        try:
            chess_board = chess.Board(fen) if fen else chess.Board(None)
            svg = chess.svg.board(chess_board, size=400)
        except Exception:
            svg = ""

        # --- HUD overlay on raw frame ---
        vis = frame.copy()

        # Draw ArUco corner markers (aruco fallback path)
        if detected_corners:
            vis = board_locator_aruco.draw_detections(vis, detected_corners)

        # Draw piece overlay on warped (stamp thumbnail on frame)
        if warped is not None and stabilized:
            warped_hud = piece_detector.draw_pieces(warped, stabilized)
            h, w = vis.shape[:2]
            thumb = cv2.resize(warped_hud, (200, 200))
            vis[h - 200 : h, w - 200 : w] = thumb

        # Mode label + piece count
        _draw_hud_text(vis, current_mode, len(stabilized), board_state)

        # JPEG encode
        _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        jpeg = buf.tobytes()

        with _lock:
            _jpeg_bytes  = jpeg
            _board_state = board_state
            _mode        = current_mode
            _current_fen = fen
            _current_svg = svg

        # FPS cap
        now    = time.monotonic()
        t_next = max(t_next + _FRAME_DELAY, now)
        sleep  = t_next - now
        if sleep > 0:
            time.sleep(sleep)


def _draw_hud_text(
    frame: np.ndarray,
    mode: str,
    piece_count: int,
    board_state: dict[str, str],
) -> None:
    """Render mode badge + piece count in top-left corner (in-place)."""
    mode_color = {
        "aruco": (0, 220, 0),
        "grid":  (0, 180, 255),
        "none":  (0, 0, 220),
    }.get(mode, (200, 200, 200))

    white_n = sum(1 for v in board_state.values() if v.isupper())
    black_n = sum(1 for v in board_state.values() if v.islower())

    lines = [
        f"Mode: {mode.upper()}",
        f"Pieces: {piece_count}  W:{white_n}  B:{black_n}",
    ]
    y = 30
    for line in lines:
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA,
        )
        y += 28


# ── MJPEG generator ───────────────────────────────────────────────────────────

def _gen_mjpeg():
    """Yield MJPEG boundary frames."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with _lock:
            frame = _jpeg_bytes
        if frame:
            yield boundary + frame + b"\r\n"
        time.sleep(_FRAME_DELAY)


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return Response(
        _HTML_PAGE,
        mimetype="text/html",
    )


@app.route("/video")
def video_feed():
    return Response(
        _gen_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def state():
    with _lock:
        payload = {
            "mode":        _mode,
            "board_state": _board_state,
            "piece_count": len(_board_state),
            "fen":         _current_fen,
            "svg":         _current_svg,
        }
    return jsonify(payload)


@app.route("/board_svg")
def board_svg():
    with _lock:
        svg = _current_svg
    if not svg:
        svg = chess.svg.board(chess.Board(None), size=400)
    return Response(svg, mimetype="image/svg+xml")


# ── HTML viewer ───────────────────────────────────────────────────────────────

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UR Chess — Live Vision</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #111;
    color: #eee;
    font-family: monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    text-align: center;
    padding: 10px;
    font-size: 1.1rem;
    letter-spacing: 2px;
    border-bottom: 1px solid #333;
    background: #1a1a1a;
  }
  #panes {
    display: flex;
    flex: 1;
    gap: 12px;
    padding: 12px;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .pane {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .pane-title {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  #stream {
    display: block;
    max-width: 480px;
    width: 100%;
    border: 2px solid #444;
    border-radius: 4px;
  }
  #board-wrap {
    width: 400px;
    height: 400px;
    border: 2px solid #444;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #222;
    overflow: hidden;
  }
  #board-wrap svg {
    width: 100%;
    height: 100%;
  }
  #statusbar {
    width: 100%;
    background: #1a1a1a;
    border-top: 1px solid #333;
    padding: 8px 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: 0.8rem;
    color: #aaa;
    align-items: center;
  }
  #statusbar .badge {
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: bold;
    font-size: 0.75rem;
    color: #111;
  }
  .badge-aruco { background: #00dc00; }
  .badge-grid  { background: #00b4ff; }
  .badge-none  { background: #dc0000; }
  #fen-str {
    font-size: 0.7rem;
    color: #6af;
    word-break: break-all;
    max-width: 600px;
  }
</style>
</head>
<body>
<header>UR Chess &mdash; Live Vision</header>
<div id="panes">
  <div class="pane">
    <div class="pane-title">Live Camera</div>
    <img id="stream" src="/video" alt="camera stream">
  </div>
  <div class="pane">
    <div class="pane-title">Digital Board</div>
    <div id="board-wrap"><span style="color:#555">Waiting...</span></div>
  </div>
</div>
<div id="statusbar">
  <span id="mode-badge" class="badge badge-none">NONE</span>
  <span id="piece-count">Pieces: 0</span>
  <span id="fen-str">FEN: —</span>
  <span id="last-updated"></span>
</div>
<script>
  async function poll() {
    try {
      const r = await fetch('/state');
      const d = await r.json();

      // Mode badge
      const badge = document.getElementById('mode-badge');
      badge.textContent = d.mode ? d.mode.toUpperCase() : 'NONE';
      badge.className = 'badge badge-' + (d.mode || 'none');

      // Piece count
      document.getElementById('piece-count').textContent = 'Pieces: ' + (d.piece_count || 0);

      // FEN
      document.getElementById('fen-str').textContent = d.fen ? 'FEN: ' + d.fen : 'FEN: —';

      // SVG board
      if (d.svg) {
        document.getElementById('board-wrap').innerHTML = d.svg;
      }

      // Timestamp
      document.getElementById('last-updated').textContent =
        'Updated: ' + new Date().toLocaleTimeString();
    } catch(e) {}
    setTimeout(poll, 500);
  }
  poll();
</script>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def start(
    camera_index: int = 0,
    port: int = 5050,
    debug: bool = False,
    host: str = "127.0.0.1",
    video_path: Optional[str] = None,
) -> None:
    """Start capture thread then Flask (blocking).

    Parameters
    ----------
    camera_index:
        Camera device index (ignored when video_path is set).
    video_path:
        Path to a video file to stream instead of a live camera.
        Video loops automatically when EOF is reached.
    """
    _tracker.reset()  # clear any stale state on restart

    cam_arg = None if video_path else camera_index
    t = threading.Thread(
        target=_capture_loop,
        args=(cam_arg, video_path),
        daemon=True,
    )
    t.start()

    # Wait for first frame
    for _ in range(50):
        with _lock:
            if _jpeg_bytes:
                break
        time.sleep(0.1)

    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR Chess live camera stream")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--port",   type=int, default=5050, help="Flask port")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file to use instead of a live camera (loops on EOF)",
    )
    args = parser.parse_args()
    start(camera_index=args.camera, port=args.port, video_path=args.video)
