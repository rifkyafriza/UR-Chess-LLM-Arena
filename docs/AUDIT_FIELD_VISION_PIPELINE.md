# Audit: Field Vision Pipeline (Phase 8)

Task: t_5e9f02d4 | Auditor: reviewer | Verdict: **CERTIFIED / APPROVED**

Scope: `src/vision/piece_detector.py`, `src/vision/board_locator_grid.py`,
`src/vision/live_stream.py`, `scripts/benchmark_board.py`, `main.py`,
`tests/test_field_vision_pipeline.py`, `.gitignore`.

## 1. Video file hygiene - PASS
- `.gitignore` covers `*.mp4 *.avi *.mov *.mkv *.webm` (+ `assets/*.mp4`).
- `git check-ignore` confirmed for `x.mp4` and `a/b.webm`.
- `git ls-files | grep -iE '\.(mp4|avi|mov|webm|mkv)$'` returns 0 files.
- No video files exist in the working tree.

## 2. Performance - PASS
Measured on a 1280x720 frame (test board photo letterboxed), CPU only:

| Stage                                   | Time     |
|-----------------------------------------|----------|
| `localize_board` (grid)                 | 16 ms    |
| `detect_pieces_from_frame` (2 passes)   | 26 ms    |
| `detect_pieces_from_frame` (+ unsharp)  | 64 ms    |
| `enhance_roi` on 800x800                | 16 ms    |
| Localize + detect, end to end           | ~12.5 FPS|

- Detection path stays above the 10 FPS floor. Margin is small (~25%); JPEG encode
  and HUD in `_capture_frames` add more per-frame cost on slower hardware.
- `enhance_roi` uses bicubic 2x + Gaussian-subtraction unsharp mask. No Laplacian
  kernel anywhere in `src/` (grep verified), so no micro-quadrilateral stall.
- `adaptiveThreshWinSizeMax` reduced 53 -> 23, which offsets the relaxed
  `minMarkerPerimeterRate=0.005`.

## 3. Security - PASS
- Paths: `subprocess` called with an argv list (no shell). `--images-dir` checked with
  `is_dir()`; extension allow-list; `cv2.imread` None frames skipped. Video open
  failure is handled in benchmark (`return 1`) and `_open_source` (`RuntimeError`).
- ROI bounds: `detect_pieces_from_frame` rejects warped coords outside
  `[0, output_size)` (fix from t_692de091: off-board ID 22 was clamped onto a8).
  Board localizer enforces area ratio 0.03-0.45 and aspect <= 1.4.
- Thread safety: shared stream state (`_jpeg_bytes`, `_board_state`, `_mode`, FEN, SVG)
  written and read under `_lock`; `TemporalPieceTracker` has its own lock.
- Frame asserts on `frame`/`matrix` shape at trust boundary.

## 4. Test suite - PASS
`python -m pytest -q`: **993 passed, 1 skipped, 0 failed** (67.8 s).
`tests/test_field_vision_pipeline.py` (21 tests) covers the off-board leak.

## Findings
HIGH: none. MEDIUM: none.

LOW-1: third (enhanced) pass in `detect_pieces_from_frame` and `detect_pieces` does not
re-check bounds; coords come from the warped board image so they are in range by
construction, and `pixel_to_square` clamps. Add an explicit check if the warp ever
gains a margin.

LOW-2: shared module-level `_CLAHE` / `_DETECTOR` are used from a single capture
thread today. Not safe if detection is ever called from multiple threads; make
them per-thread then.

LOW-3: FPS headroom over 10 FPS is thin on 720p CPU. If the field laptop is slower,
scale the enhanced pass to the board ROI only or skip it when the 2 passes already
found all expected markers.

LOW-4: changes are still uncommitted (8 modified files + new test file). Commit before release.

## Verdict
**CERTIFIED / APPROVED** for Phase 8 (sim and field-video benchmark use).
