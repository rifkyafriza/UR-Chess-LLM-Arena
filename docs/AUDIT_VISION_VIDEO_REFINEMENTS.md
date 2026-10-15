# Audit: Phase 7 Vision / Video Refinements

Task: t_d051a2c6. Verdict: **CERTIFIED / APPROVED** (simulation and video-file mode; no HIGH findings).

## Checklist
1. .gitignore: PASS. `*.mp4` and `assets/*.mp4` ignored; `git check-ignore` confirms `test_video.mp4`; `git ls-files` lists 0 video files.
2. TemporalPieceTracker: PASS after fix. Bounded by `max_pieces` (32) and TTL (8); every public method holds a `threading.Lock`; shared stream globals guarded by `_lock`.
3. FEN / SVG: PASS. `ID_TO_PIECE_CHAR` covers IDs 1-32 (K/Q distinct), `build_piece_type_board_state` feeds `serialize_fen`, `/board_svg` renders glyphs (tests).
4. Default locator_mode: PASS after fix. Dataclass default was `contour`, but `_build_vision` fell back to `aruco` when the key was missing. Now `contour`.
5. pytest: 971 passed, 1 skipped, 0 failed.

## Findings
- MEDIUM (fixed): `src/config.py` `_build_vision` default `"aruco"` contradicted the dataclass default; a YAML without `locator_mode` silently used ArUco.
- MEDIUM (fixed): `TemporalPieceTracker.update` kept a ghost (TTL-retained) piece on a square now occupied by a different marker. Dict order let the ghost overwrite the live piece in `build_piece_type_board_state`, giving a wrong FEN for up to 8 frames. Ghosts on freshly detected squares are now evicted immediately.
- LOW (open): ArUco fallback path detects on the warped image while contour path uses full-res homography; accuracy differs between modes. Acceptable.
- LOW (open): module-level `_tracker` singleton; fine for one stream per process.
- Video path comes from the local CLI only; no network input reaches `cv2.VideoCapture`.

## Tests added
`tests/test_vision_video_refinements.py`: ghost-eviction test, missing-key default test.
