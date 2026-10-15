"""Tests for video benchmark path in benchmark_board.py and main.py."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


class TestBenchmarkMainParser(unittest.TestCase):
    """main.py --mode benchmark --video propagates to run_benchmark."""

    def test_benchmark_no_video_calls_run_benchmark_no_arg(self):
        import main as m
        with patch.object(m, "run_benchmark") as mock_rb:
            m.main(["--mode", "benchmark"])
            mock_rb.assert_called_once_with(video_path=None)

    def test_benchmark_with_video_passes_path(self):
        import main as m
        with patch.object(m, "run_benchmark") as mock_rb:
            m.main(["--mode", "benchmark", "--video", "/tmp/test.mp4"])
            mock_rb.assert_called_once_with(video_path="/tmp/test.mp4")


class TestRunBenchmarkDispatch(unittest.TestCase):
    """run_benchmark() builds correct subprocess command."""

    def _run_and_capture_cmd(self, video_path):
        import main as m
        captured = {}
        orig_run = __import__("subprocess").run

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            r = MagicMock()
            r.returncode = 0
            return r

        with patch("subprocess.run", side_effect=fake_run):
            with self.assertRaises(SystemExit):
                m.run_benchmark(video_path=video_path)
        return captured["cmd"]

    def test_no_video_cmd(self):
        cmd = self._run_and_capture_cmd(None)
        self.assertIn("benchmark_board.py", cmd[-1])
        self.assertNotIn("--video", cmd)

    def test_with_video_cmd(self):
        cmd = self._run_and_capture_cmd("/tmp/myvideo.mp4")
        self.assertIn("--video", cmd)
        self.assertIn("/tmp/myvideo.mp4", cmd)


class TestRunVideoBenchmark(unittest.TestCase):
    """run_video_benchmark() logic with mocked cv2."""

    def _make_fake_cv2(self, frames: list):
        """Return a cv2 mock that yields frames then stops."""
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0  # FPS / frame count placeholder
        frame_iter = iter([(True, f) for f in frames] + [(False, None)])
        cap.read.side_effect = lambda: next(frame_iter)
        fake_cv2.VideoCapture.return_value = cap
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.CAP_PROP_FRAME_COUNT = 7
        return fake_cv2

    def test_video_not_openable_returns_1(self):
        import scripts.benchmark_board as bb
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = False
        fake_cv2.VideoCapture.return_value = cap
        with patch.object(bb, "cv2", fake_cv2):
            rc = bb.run_video_benchmark("/nonexistent.mp4")
        self.assertEqual(rc, 1)

    def test_zero_frames_returns_1(self):
        import scripts.benchmark_board as bb
        fake_cv2 = self._make_fake_cv2([])
        with patch.object(bb, "cv2", fake_cv2):
            rc = bb.run_video_benchmark("/empty.mp4")
        self.assertEqual(rc, 1)

    def test_frames_no_localization_returns_0(self):
        """All frames fail localization — should still return 0 with 0% rates."""
        import numpy as np
        import scripts.benchmark_board as bb

        frames = [np.zeros((480, 640, 3), dtype="uint8") for _ in range(3)]
        fake_cv2 = self._make_fake_cv2(frames)

        with patch.object(bb, "cv2", fake_cv2), \
             patch.object(bb, "run_mode2", return_value={"success": False}):
            rc = bb.run_video_benchmark("/fake.mp4")
        self.assertEqual(rc, 0)

    def test_frames_with_localization_returns_0(self):
        """Some frames localize — code paths for piece detection exercised."""
        import numpy as np
        import scripts.benchmark_board as bb

        frame = np.zeros((480, 640, 3), dtype="uint8")
        frames = [frame.copy() for _ in range(2)]
        fake_cv2 = self._make_fake_cv2(frames)

        m2_ok = {
            "success": True,
            "matrix": np.eye(3).tolist(),
            "warped": frame,
            "grid_verified": True,
        }

        with patch.object(bb, "cv2", fake_cv2), \
             patch.object(bb, "run_mode2", return_value=m2_ok), \
             patch.object(bb, "run_piece_detection", return_value=([], {})), \
             patch("src.vision.fen_serializer.serialize_fen", return_value="x"):
            rc = bb.run_video_benchmark("/fake.mp4")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
