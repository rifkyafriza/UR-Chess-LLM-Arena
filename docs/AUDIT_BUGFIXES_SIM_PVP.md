# Audit: Bug Fixes, PvP Mode & Web Simulation Integrity

Task: t_996ec590 | Reviewer: reviewer | Verdict: **CERTIFIED / APPROVED (simulation)**

## Checks

1. main.py run_live() NameError - PASS. `vision_app.run(...)` replaced by `start(camera_index=camera, port=port)` imported from `src.vision.live_stream` (signature verified, default host 127.0.0.1). No `vision_app` reference remains. Covered by tests/test_launcher_live.py (5 tests).
2. PvP without LLM/9Router - PASS. In `api_move`, PvP branch returns before `get_best_move`. `llm_client` is imported at module load (openai SDK) but makes no network call at import. Turn enforcement per side verified; tests run W-B-W-B with LLM mocked to fail.
3. Standalone simulation - PASS. `src/simulation` has no import of robot/UR/socket modules. Tests assert ports 29999/30002/30003 never opened. Server binds 127.0.0.1, debug=False by default.
4. Board sync - PASS. Server is authoritative (client-sent positions are validated but not trusted). Captured markers -> `__captured__` (skipped by renderer, no negative coords). Castling moves king + rook (verified manually: e1g1 -> K g1, R f1). En passant resolves victim square. No overlap: one marker per square tested after capture/castle/ep.
5. Test suite - PASS. `pytest -q`: 854 passed, 1 skipped, 0 failed (62.9s). Parent handoff reported 13 test_config.py failures; they do NOT reproduce (Phase 3 audit already fixed the test patch leak). Run is clean.

## Findings

HIGH: none.
MEDIUM: none.
LOW-1: Pawn promotion auto-queens on the board, but marker id stays mapped to pawn char in `_positions_to_board_state`/UI, so the promoted piece may render as a pawn in the draggable piece layer while the SVG/FEN show a queen. Cosmetic, sim only. Fix: track promoted marker type in GameState.
LOW-2: `app.config["SECRET_KEY"]` is hardcoded; unused (no sessions) and localhost-only. Remove or read from env if sessions are added.
LOW-3: `_board_state_to_fen` clears castling/ep; only used for display, not game logic. Do not use for engine input.

## Scope note
Real UR hardware remains CONDITIONAL per AUDIT_PHASE3_FULL_SYSTEM.md; unchanged by this task.
