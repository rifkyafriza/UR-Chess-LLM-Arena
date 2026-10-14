"""Live camera web stream with ArUco HUD overlay.

Flask MJPEG server:
    GET /         — HTML viewer page
    GET /video    — MJPEG stream
    GET /state    — JSON board state snapshot

Detection pipeline per frame:
    1. Try ArUco corner markers (IDs 40-43) → board_locator_aruco.locate_board
    2. Fallback: contour grid detection  → board_locator_grid.localize_board
    3. Detect piece markers (IDs 1-32)   → piece_detector.detect_pieces
    4. Render HUD overlay on raw frame
    5. JPEG-encode and push to MJPEG boundary

Run:
    python -m src.vision.live_stream [--camera 0] [--port 5050]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify

from src.vision import board_locator_aruco, board_locator_grid, piece_detector

# ── Config ────────────────────────────────────────────────────────────────────

_JPEG_QUALITY = 75      # lower = faster encoding, less bandwidth
_TARGET_FPS   = 30      # cap capture loop; actual may be lower
_FRAME_DELAY  = 1.0 / _TARGET_FPS

# ── Shared state (capture thread → Flask threads) ─────────────────────────────

_lock        = threading.Lock()
_jpeg_bytes: bytes          = b""
_board_state: dict[str, str] = {}
_mode: str                  = "none"   # "aruco" | "grid" | "none"

# ── Capture thread ────────────────────────────────────────────────────────────

def _capture_loop(camera_index: int) -> None:
    """Background thread: capture, detect, encode."""
    global _jpeg_bytes, _board_state, _mode

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    t_next = time.monotonic()

    try:
        _capture_frames(cap, t_next)
    finally:
        cap.release()


def _capture_frames(cap, t_next: float) -> None:
    global _jpeg_bytes, _board_state, _mode
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        warped: Optional[np.ndarray] = None
        detected_corners: dict       = {}
        current_mode                 = "none"

        # --- Mode 1: ArUco corner markers ---
        warped_aruco, detected_corners = board_locator_aruco.locate_board(frame)
        if warped_aruco is not None:
            warped        = warped_aruco
            current_mode  = "aruco"

        # --- Mode 2: contour grid fallback ---
        if warped is None:
            result = board_locator_grid.localize_board(frame)
            if result is not None:
                warped       = result["warped"]
                current_mode = "grid"

        # --- Piece detection on warped board ---
        pieces: list[dict] = []
        if warped is not None:
            pieces = piece_detector.detect_pieces(warped)

        board_state = piece_detector.build_board_state(pieces)

        # --- HUD overlay on raw frame ---
        vis = frame.copy()

        # Draw ArUco corner markers (mode 1)
        if detected_corners:
            vis = board_locator_aruco.draw_detections(vis, detected_corners)

        # Draw piece overlay on warped (then stamp thumbnail on frame)
        if warped is not None and pieces:
            warped_hud = piece_detector.draw_pieces(warped, pieces)
            # Thumbnail: 200×200 in bottom-right corner
            h, w = vis.shape[:2]
            thumb = cv2.resize(warped_hud, (200, 200))
            vis[h - 200 : h, w - 200 : w] = thumb

        # Mode label + piece count
        _draw_hud_text(vis, current_mode, len(pieces), board_state)

        # JPEG encode
        _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        jpeg = buf.tobytes()

        with _lock:
            _jpeg_bytes  = jpeg
            _board_state = board_state
            _mode        = current_mode

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

    white_n = sum(1 for v in board_state.values() if v == "white")
    black_n = sum(1 for v in board_state.values() if v == "black")

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
        }
    return jsonify(payload)


# ── HTML viewer ───────────────────────────────────────────────────────────────

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UR Chess — Live Vision</title>
<style>
  body { margin:0; background:#1a1a1a; color:#eee; font-family:monospace; }
  h1   { text-align:center; padding:8px 0; font-size:1.1rem; }
  #stream { display:block; margin:0 auto; max-width:100%; border:2px solid #444; }
  #state  { white-space:pre; padding:8px 16px; font-size:0.85rem; color:#aef; }
</style>
</head>
<body>
<h1>UR Chess — Live Camera Stream</h1>
<img id="stream" src="/video" alt="camera stream">
<pre id="state">Loading...</pre>
<script>
  async function poll() {
    try {
      const r = await fetch('/state');
      const d = await r.json();
      document.getElementById('state').textContent =
        'Mode: ' + d.mode + '\\nPieces: ' + d.piece_count +
        '\\nBoard state: ' + JSON.stringify(d.board_state, null, 2);
    } catch(e) {}
    setTimeout(poll, 500);
  }
  poll();
</script>
</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def start(camera_index: int = 0, port: int = 5050, debug: bool = False,
          host: str = "127.0.0.1") -> None:
    """Start capture thread then Flask (blocking)."""
    t = threading.Thread(target=_capture_loop, args=(camera_index,), daemon=True)
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
    args = parser.parse_args()
    start(camera_index=args.camera, port=args.port)
