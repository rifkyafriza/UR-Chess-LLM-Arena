"""Benchmark real board photo, video, or images directory.

Usage:
    python scripts/benchmark_board.py
    python scripts/benchmark_board.py --video path/to/video.mp4
    python scripts/benchmark_board.py --images-dir path/to/images/
"""

from __future__ import annotations
from typing import Optional

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so 'from src...' imports resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.vision.piece_detector import TemporalPieceTracker

import cv2
import numpy as np

# Ground truth: square -> FEN piece char for starting position
GROUND_TRUTH: dict[str, str] = {
    # Black major pieces (rank 8)
    "a8": "r", "b8": "n", "c8": "b", "d8": "q", "e8": "k", "f8": "b", "g8": "n", "h8": "r",
    # Black pawns (rank 7)
    "a7": "p", "b7": "p", "c7": "p", "d7": "p", "e7": "p", "f7": "p", "g7": "p", "h7": "p",
    # White pawns (rank 2)
    "a2": "P", "b2": "P", "c2": "P", "d2": "P", "e2": "P", "f2": "P", "g2": "P", "h2": "P",
    # White major pieces (rank 1)
    "a1": "R", "b1": "N", "c1": "B", "d1": "Q", "e1": "K", "f1": "B", "g1": "N", "h1": "R",
}

EXPECTED_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# ArUco ID -> (square, piece_char) mapping for starting position
PIECE_MAP: dict[int, tuple[str, str]] = {
    # White major (rank 1): IDs 1-8
    1: ("a1", "R"), 2: ("b1", "N"), 3: ("c1", "B"), 4: ("d1", "Q"),
    5: ("e1", "K"), 6: ("f1", "B"), 7: ("g1", "N"), 8: ("h1", "R"),
    # White pawns (rank 2): IDs 9-16
    9:  ("a2", "P"), 10: ("b2", "P"), 11: ("c2", "P"), 12: ("d2", "P"),
    13: ("e2", "P"), 14: ("f2", "P"), 15: ("g2", "P"), 16: ("h2", "P"),
    # Black major (rank 8): IDs 17-24
    17: ("a8", "r"), 18: ("b8", "n"), 19: ("c8", "b"), 20: ("d8", "q"),
    21: ("e8", "k"), 22: ("f8", "b"), 23: ("g8", "n"), 24: ("h8", "r"),
    # Black pawns (rank 7): IDs 25-32
    25: ("a7", "p"), 26: ("b7", "p"), 27: ("c7", "p"), 28: ("d7", "p"),
    29: ("e7", "p"), 30: ("f7", "p"), 31: ("g7", "p"), 32: ("h7", "p"),
}


# Valid piece IDs (1-32); anything outside is an off-board false positive
KNOWN_PIECE_IDS: frozenset[int] = frozenset(PIECE_MAP.keys())


def run_mode1(frame: np.ndarray) -> dict:
    """Run Mode 1: ArUco corner markers (IDs 40-43)."""
    from src.vision.board_locator_aruco import locate_board, CORNER_IDS
    warped, detected = locate_board(frame)
    return {
        "mode": "Mode 1 (ArUco corners 40-43)",
        "detected_corner_ids": sorted(detected.keys()),
        "expected_corner_ids": list(CORNER_IDS),
        "warped": warped,
        "success": warped is not None,
        "note": "Expected MISSING on physical mat without printed corner markers",
    }


def run_mode2(frame: np.ndarray) -> dict:
    """Run Mode 2: Grid contour detection."""
    from src.vision.board_locator_grid import localize_board
    result = localize_board(frame, verify_grid=True)
    if result is None:
        return {"mode": "Mode 2 (Grid contour)", "success": False, "warped": None}
    return {
        "mode": "Mode 2 (Grid contour)",
        "success": True,
        "warped": result["warped"],
        "corners": result["corners"].tolist(),
        "matrix": result["matrix"].tolist(),
        "grid_verified": result["verified"],
    }


def run_piece_detection(frame: np.ndarray, matrix: np.ndarray) -> tuple[list[dict], dict[str, str]]:
    """Detect pieces on full-res frame, project centers via M to algebraic squares."""
    from src.vision.piece_detector import detect_pieces_from_frame
    pieces = detect_pieces_from_frame(frame, matrix)
    board_state: dict[str, str] = {}
    for p in pieces:
        if p["id"] in PIECE_MAP:
            _, piece_char = PIECE_MAP[p["id"]]
            board_state[p["square"]] = piece_char
    return pieces, board_state


def verify_pieces(pieces: list[dict]) -> dict:
    """Verify detected pieces against ground truth."""
    detected_ids = {p["id"]: p["square"] for p in pieces}
    all_expected_ids = set(PIECE_MAP.keys())
    detected_id_set = set(detected_ids.keys())

    correct = 0
    wrong_square = []
    missing = []

    for pid, (expected_sq, piece_char) in PIECE_MAP.items():
        if pid not in detected_id_set:
            missing.append((pid, expected_sq, piece_char))
        elif detected_ids[pid] != expected_sq:
            wrong_square.append((pid, expected_sq, detected_ids[pid], piece_char))
        else:
            correct += 1

    unexpected = [(pid, sq) for pid, sq in detected_ids.items() if pid not in all_expected_ids]

    return {
        "total_expected": 32,
        "total_detected": len(detected_id_set),
        "correct": correct,
        "missing": missing,
        "wrong_square": wrong_square,
        "unexpected": unexpected,
        "accuracy": correct / 32 * 100,
    }


def draw_benchmark_vis(
    warped: np.ndarray,
    pieces: list[dict],
    verify: dict,
    mode2_result: dict,
) -> np.ndarray:
    """Draw visualization with board boundary, piece HUD, stats overlay."""
    from src.vision.piece_detector import draw_pieces

    vis = draw_pieces(warped, pieces)

    # Draw 8x8 grid lines
    cell = 100
    for i in range(9):
        cv2.line(vis, (i * cell, 0), (i * cell, 800), (180, 180, 180), 1)
        cv2.line(vis, (0, i * cell), (800, i * cell), (180, 180, 180), 1)

    # File/rank labels
    for i, f in enumerate("abcdefgh"):
        cv2.putText(vis, f, (i * 100 + 42, 795), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for i, r in enumerate("87654321"):
        cv2.putText(vis, r, (4, i * 100 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # Stats panel at bottom (extend canvas)
    panel_h = 100
    panel = np.zeros((panel_h, 800, 3), dtype=np.uint8)
    acc = verify["accuracy"]
    color = (0, 220, 0) if acc == 100 else (0, 165, 255) if acc >= 75 else (0, 0, 220)
    cv2.putText(panel, f"Detected: {verify['total_detected']}/32  Correct: {verify['correct']}/32  Accuracy: {acc:.1f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    mode_label = "Mode 2 OK" if mode2_result["success"] else "Mode 2 FAIL"
    grid_v = "grid-verified" if mode2_result.get("grid_verified") else "grid-NOT-verified"
    cv2.putText(panel, f"{mode_label} | {grid_v}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    missing_ids = [pid for pid, _, _ in verify["missing"]]
    cv2.putText(panel, f"Missing IDs: {missing_ids if missing_ids else 'none'}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

    result = np.vstack([vis, panel])
    return result


def run_photo_benchmark(img_path: str, out_path: str) -> int:
    """Run static photo benchmark. Returns exit code (0=pass, 1=error, 2=fen mismatch)."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"ERROR: cannot read {img_path}")
        return 1

    print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]} px")
    print()

    # ── Mode 1 ────────────────────────────────────────────────────────────────
    m1 = run_mode1(frame)
    print(f"[{m1['mode']}]")
    print(f"  Detected corner IDs : {m1['detected_corner_ids']}")
    print(f"  Expected corner IDs : {m1['expected_corner_ids']}")
    print(f"  Success             : {m1['success']}")
    print(f"  Note                : {m1['note']}")
    print()

    # ── Mode 2 ────────────────────────────────────────────────────────────────
    m2 = run_mode2(frame)
    print(f"[{m2['mode']}]")
    print(f"  Success             : {m2['success']}")
    if m2["success"]:
        print(f"  Grid verified       : {m2['grid_verified']}")
        print(f"  Corners (TL,TR,BR,BL): {[[round(v, 1) for v in pt] for pt in m2['corners']]}")
    print()

    # Use Mode 2 warped (fallback to Mode 1 if available)
    warped = m2.get("warped") if m2.get("warped") is not None else m1.get("warped")
    if warped is None:
        print("ERROR: neither mode produced a warped board")
        return 1

    # ── Piece Detection ────────────────────────────────────────────────────────
    matrix = np.array(m2["matrix"], dtype=np.float64)
    pieces, board_state = run_piece_detection(frame, matrix)
    print(f"[Piece Detection]")
    print(f"  Total pieces detected: {len(pieces)}")
    for p in sorted(pieces, key=lambda x: x["id"]):
        pid = p["id"]
        expected_sq = PIECE_MAP.get(pid, (None, None))[0]
        match = "OK" if p["square"] == expected_sq else f"WRONG (expected {expected_sq})"
        print(f"    ID {pid:2d} @ {p['square']}  {match}")
    print()

    # ── Verification ──────────────────────────────────────────────────────────
    verify = verify_pieces(pieces)
    print(f"[Verification]")
    print(f"  Expected  : 32 pieces")
    print(f"  Detected  : {verify['total_detected']}")
    print(f"  Correct   : {verify['correct']}")
    print(f"  Accuracy  : {verify['accuracy']:.1f}%")
    if verify["missing"]:
        print(f"  Missing   : {[(pid, sq) for pid, sq, _ in verify['missing']]}")
    if verify["wrong_square"]:
        print(f"  Wrong sq  : {[(pid, exp, got) for pid, exp, got, _ in verify['wrong_square']]}")
    print()

    # ── FEN ───────────────────────────────────────────────────────────────────
    from src.vision.fen_serializer import serialize_fen
    fen = serialize_fen(board_state, castling="KQkq")
    print(f"[FEN Serialization]")
    print(f"  Produced  : {fen}")
    print(f"  Expected  : {EXPECTED_FEN}")
    fen_ok = fen == EXPECTED_FEN
    print(f"  Match     : {'YES' if fen_ok else 'NO'}")
    print()

    # ── Visualization ─────────────────────────────────────────────────────────
    vis = draw_benchmark_vis(warped, pieces, verify, m2)
    cv2.imwrite(out_path, vis)
    print(f"[Output] Saved benchmark visualization: {out_path}")

    return 0 if fen_ok else 2


def run_video_benchmark(video_path: str) -> int:
    """Process every frame sequentially; print summary report. Returns 0 on success."""
    from src.vision.fen_serializer import serialize_fen

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {video_path}")
        return 1

    total_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video Benchmark] {video_path}")
    print(f"  Source FPS        : {total_fps:.1f}")
    print(f"  Source frames     : {total_frame_count}")
    print()

    frames_processed = 0
    localized_count = 0
    piece_counts: list[int] = []
    fen_match_count = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1

        # Try contour localization (primary)
        m2 = run_mode2(frame)
        localized = m2["success"]

        if localized:
            localized_count += 1
            matrix = np.array(m2["matrix"], dtype=np.float64)
            pieces, board_state = run_piece_detection(frame, matrix)
            piece_counts.append(len(pieces))
            fen = serialize_fen(board_state, castling="KQkq")
            if fen == EXPECTED_FEN:
                fen_match_count += 1
        else:
            piece_counts.append(0)

    cap.release()
    elapsed = time.perf_counter() - t_start

    if frames_processed == 0:
        print("ERROR: no frames read from video")
        return 1

    localization_rate = localized_count / frames_processed * 100
    avg_pieces = sum(piece_counts) / len(piece_counts) if piece_counts else 0
    # FEN match rate over localized frames only (avoids penalizing failed detections)
    fen_match_rate = (fen_match_count / localized_count * 100) if localized_count else 0
    processing_fps = frames_processed / elapsed if elapsed > 0 else 0

    print("=" * 56)
    print("  VIDEO BENCHMARK SUMMARY")
    print("=" * 56)
    print(f"  Total frames processed : {frames_processed}")
    print(f"  Board localization rate : {localization_rate:.1f}%  ({localized_count}/{frames_processed})")
    print(f"  Avg piece count/frame  : {avg_pieces:.1f}")
    print(f"  FEN match rate         : {fen_match_rate:.1f}%  ({fen_match_count}/{localized_count} localized frames)")
    print(f"  Processing FPS         : {processing_fps:.2f}")
    print(f"  Wall-clock time        : {elapsed:.2f} s")
    print("=" * 56)
    return 0


def run_sparse_benchmark(source: str, is_images_dir: bool = False) -> int:
    """Benchmark single/sparse piece tracking on video or images directory.

    Metrics:
      - Tracking Continuity Rate (%): frames where at least one valid piece was detected
      - False Positive Elimination (%): ratio of off-board IDs filtered out
      - Processing FPS: frames processed per second
      - Active Board Localization Rate (%): frames where board was localized

    Args:
        source: path to video file or images directory
        is_images_dir: True when source is a directory of images
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if is_images_dir:
        source_path = Path(source)
        if not source_path.is_dir():
            print(f"ERROR: not a directory: {source}")
            return 1
        frames_iter = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not frames_iter:
            print(f"ERROR: no images found in {source}")
            return 1
        print(f"[Sparse Benchmark] Images dir: {source}  ({len(frames_iter)} files)")
        print()

        def _frame_generator():
            for p in frames_iter:
                img = cv2.imread(str(p))
                if img is not None:
                    yield img
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: cannot open video: {source}")
            return 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        print(f"[Sparse Benchmark] Video: {source}")
        print(f"  Source FPS    : {src_fps:.1f}")
        print(f"  Source frames : {total_frames}")
        print()

        def _frame_generator():
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                yield f

    frames_processed = 0
    localized_count = 0
    frames_with_valid_piece = 0  # Tracking Continuity
    total_raw_detections = 0
    total_fp_detections = 0      # IDs not in valid sparse set (0-32)

    VALID_SPARSE_IDS: frozenset[int] = frozenset(range(0, 33))

    last_matrix: Optional[np.ndarray] = None
    last_warped: Optional[np.ndarray] = None
    tracker = TemporalPieceTracker(ttl=12)

    t_start = time.perf_counter()

    for frame in _frame_generator():
        frames_processed += 1

        from src.vision.board_locator_grid import localize_board
        from src.vision.piece_detector import detect_pieces_from_frame

        res = localize_board(frame, verify_grid=False)
        if res is not None:
            localized_count += 1
            last_matrix = res["matrix"]
            last_warped = res["warped"]

        target_m = res["matrix"] if res is not None else (last_matrix if not is_images_dir else None)
        target_w = res["warped"] if res is not None else (last_warped if not is_images_dir else None)

        if target_m is not None:
            matrix = np.array(target_m, dtype=np.float64)
            all_pieces = detect_pieces_from_frame(frame, matrix, warped=target_w)

            raw_ids = [p["id"] for p in all_pieces]
            total_raw_detections += len(raw_ids)

            fp_ids = [pid for pid in raw_ids if pid not in VALID_SPARSE_IDS]
            total_fp_detections += len(fp_ids)

            valid_pieces = [p for p in all_pieces if p["id"] in VALID_SPARSE_IDS]
            stabilized = tracker.update(valid_pieces)
            if stabilized:
                frames_with_valid_piece += 1

    if not is_images_dir:
        cap.release()  # type: ignore[possibly-undefined]

    elapsed = time.perf_counter() - t_start

    if frames_processed == 0:
        print("ERROR: no frames read")
        return 1

    localization_rate = localized_count / frames_processed * 100
    continuity_rate = frames_with_valid_piece / frames_processed * 100
    processing_fps = frames_processed / elapsed if elapsed > 0 else 0
    # FP elimination: what fraction of raw detections were valid (not FP)
    if total_raw_detections > 0:
        fp_elim_rate = (1 - total_fp_detections / total_raw_detections) * 100
    else:
        fp_elim_rate = 100.0

    print("=" * 60)
    print("  SPARSE / SINGLE-PIECE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Frames processed            : {frames_processed}")
    print(f"  Active Board Localization   : {localization_rate:.1f}%  ({localized_count}/{frames_processed})")
    print(f"  Tracking Continuity Rate    : {continuity_rate:.1f}%  ({frames_with_valid_piece}/{frames_processed})")
    print(f"  False Positive Elimination  : {fp_elim_rate:.1f}%  ({total_raw_detections - total_fp_detections}/{total_raw_detections} valid)")
    print(f"    Off-board FP IDs seen     : {total_fp_detections} total across all frames")
    print(f"  Processing FPS              : {processing_fps:.2f}")
    print(f"  Wall-clock time             : {elapsed:.2f} s")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision benchmark for board/piece detection")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for video benchmark (omit for static photo benchmark)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Override default test image path (photo benchmark only)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory of images for sparse/single-piece benchmark",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="Run sparse/single-piece benchmark on --video or --images-dir instead of full-set benchmark",
    )
    args = parser.parse_args()

    if args.images_dir:
        return run_sparse_benchmark(args.images_dir, is_images_dir=True)

    if args.video:
        if args.sparse:
            return run_sparse_benchmark(args.video, is_images_dir=False)
        return run_video_benchmark(args.video)

    img_path = args.image or str(_REPO_ROOT / "assets" / "test_board_with_pieces.jpg")
    out_path = str(_REPO_ROOT / "assets" / "benchmark_result.jpg")
    return run_photo_benchmark(img_path, out_path)


if __name__ == "__main__":
    sys.exit(main())
