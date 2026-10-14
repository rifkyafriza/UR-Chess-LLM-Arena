# Audit: Phase 3 Full System (Config, Feature Plane, Master FSM)

Task: t_dfca0dda | Reviewer | Verdict: **APPROVED for simulation; CONDITIONAL for real hardware**

## Test result

- `python -m pytest -q`: **820 passed, 1 skipped, 0 failed** (2 consecutive full runs stable).
- Before audit: 12 failed. These were NOT timing flakes. Root cause: `tests/test_fsm_unit.py::TestRobotMoving`
  started a `patch.object(MockURDriver, "movel")` and never stopped it (`self.addCleanup = patcher.stop`
  only reassigned an attribute; pytest does not call it). The stub leaked into `test_robot.py` /
  `test_robot_driver.py`. Fixed with `teardown_method`.

## Findings

### HIGH (fixed)
1. **Config boundary validation absent** (`src/config.py`). Values were only type-cast. Accepted:
   `transit_clearance_z=0.01` (below 80 mm floor), negative/zero pitch, inverted/NaN envelope bounds,
   `mode: banana`, port 99999, unbounded speeds, `feature_plane_pose` of wrong length.
   Fix: `_validate()` called after parse. Unsafe values raise `ValueError` (fail loud, no silent fallback
   to defaults, which would mask a mis-set safety limit). Missing file / corrupt YAML still returns defaults.
   Test: `tests/test_config.py::test_unsafe_values_rejected`.
2. **Orchestrator mishandled en-passant and promotion** (`src/orchestrator/master_fsm.py::_step_robot_move`).
   En-passant is_capture=True but victim is not on `to_sq`: robot would grab an empty square and leave the
   pawn on the board. Promotion moved a pawn with no piece swap, desyncing physical vs logical board.
   Fix: these moves now enter `ERROR_SAFE_STOP` before any motion. Upgrade path: implement victim-square pick
   and promotion piece swap.

### MEDIUM (open, triaged)
3. **Feature Plane transit check inverts under tilted/flipped frames.** With the config default pose
   `[0,0,0,pi,0,0]`, `approach_pose` raises SafetyViolation (local +Z maps to base -Z). Fails safe, but the
   default `feature_plane_pose` is unusable with `use_ur_feature: true`. Calibrate via
   `scripts/calibrate_feature.py` and set a real pose before use. Recommend Kanban task: document frame
   convention + reject at load time if the board normal points down.
4. **`use_ur_feature` config flag not wired.** `MasterOrchestrator._init_robot` never builds a
   `FeaturePlaneConfig`; board origin is hard-coded `(0.10,-0.45,0.0)` and graveyard origins hard-coded.
   Real hardware must not run until these come from config. Recommend Kanban task.
5. **Capture colour assumption**: `captured_color` assumes the robot plays Black. Correct for current
   product definition; breaks if sides are swapped. Derive from `board.piece_at(to_sq).color`.
6. **`_detect_move` has dead code** (unused `disappeared/appeared/lost_sqs/gained_sqs`). Also FEN diff compares
   `board_fen()` only, so it cannot distinguish a promotion piece choice from a wrong piece. Cleanup recommended.

### LOW
7. Default `ArenaConfig.strategy.model` hard-coded; `main.py run_arena` overrides it, fine.
8. `run_arena` uses stdin FEN callback only in real mode; sim mode has no human feed (loop idles to max_turns).

## Checklist

1. Config safety
   - Default fallback: OK (missing file, absent PyYAML, corrupt YAML).
   - Type safety: OK (explicit casts; list pose elements cast to float).
   - Boundary validation: FIXED (finding 1).
   - Injection resistance: OK. `yaml.safe_load` only; verified `!!python/object/apply:os.system` payload
     is rejected and falls back to defaults (no code execution). `ARENA_CONFIG` env path is read-only
     use, no shell.
2. UR Feature Plane math
   - Rodrigues matrix `R = cI + (1-c)kk^T + s[k]x` is correct; matches URScript `pose_trans(feature, p_local)`
     for position (rotate by rotvec, then translate). Verified numerically: identity pose gives
     `e4 = (0.52, -0.235)` for origin (0.3,-0.4) pitch 0.055; Rz=90deg maps +X file axis to +Y base as expected.
   - Tool orientation uses feature rotvec directly (equals `pose_trans` orientation composition only when
     tool-local rotation is identity). Acceptable; see finding 3.
   - Safety envelope still applied via `_check`. OK.
3. Master FSM
   - All 8 transitions match documented diagram. Any exception in a step -> `ERROR_SAFE_STOP`.
   - Error path: `stopl()` + gripper release, tolerant of failure. `shutdown()` disconnects, swallows error,
     and is called in `finally` in `main.py`. `max_turns` cap present.
   - Illegal / undecodable human FEN -> back to WAIT (no crash). LLM output is legality-checked in
     `get_best_move` with fallback.
   - Collision prevention: relies on trajectory transit-Z floor + envelope + arrival wait (Phase 2). Capture
     removes victim before attacker moves. En-passant/promotion gap closed (finding 2).
4. Tests: 820 pass.

## Certification

- Simulation / software: **CERTIFIED**.
- Real UR hardware: **NOT yet certified**. Blocked on findings 3 and 4 (feature plane wiring + calibration)
  and a supervised dry run at reduced speed with E-stop in reach.
- UX audit gate: no `docs/UX_Audit_*.md` exists in this repo; no web UI changes in Phase 3. N/A.
- Not run: `npm run build` (no Node project).
