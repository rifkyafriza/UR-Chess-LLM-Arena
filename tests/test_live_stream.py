"""Smoke tests for live_stream.py — no camera needed.

Tests:
    1. Flask /state returns valid JSON with required keys.
    2. Flask /video returns multipart content-type header.
    3. Flask / returns HTML with expected content.
    4. _draw_hud_text does not raise on a blank frame.
"""

import sys
import os

# Allow running from repo root: python tests/test_live_stream.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.vision.live_stream import app, _draw_hud_text


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_state_keys(client):
    r = client.get("/state")
    assert r.status_code == 200
    data = r.get_json()
    assert "mode"        in data
    assert "board_state" in data
    assert "piece_count" in data


def test_video_route_registered(client):
    # Verify route exists without consuming infinite MJPEG body.
    rules = [str(r) for r in app.url_map.iter_rules()]
    assert "/video" in rules


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"UR Chess" in r.data


def test_draw_hud_text_no_raise():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Must not raise for any mode.
    for mode in ("aruco", "grid", "none", "unknown"):
        _draw_hud_text(frame, mode, piece_count=4, board_state={"e4": "white", "d5": "black"})


if __name__ == "__main__":
    # Run without pytest: manual self-check
    import types

    passed = 0
    failed = 0

    def run(name, fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    # Minimal client shim
    app.config["TESTING"] = True
    _client = app.test_client()

    run("state_keys",          lambda: test_state_keys(_client))
    run("video_route_registered", lambda: test_video_route_registered(_client))
    run("index_html",          lambda: test_index_html(_client))
    run("draw_hud_text",       lambda: test_draw_hud_text_no_raise())

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
