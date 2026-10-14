# Audit Report: Phase 2 Robotics (Kinematics, Collision, Driver)

Auditor: reviewer | Task: t_2545c775 | Scope: src/robot/{coordinates,trajectory,ur_driver,graveyard}.py, tests/ (694 tests).

## Result

Certification: **APPROVED for simulation / headless use. CONDITIONAL for real hardware** (see Open items, M2/M3, before first live run).
`python -m pytest -q` -> 690 passed before fixes, 694 passed after fixes.

## Checklist

| Item | Status |
|------|--------|
| X/Y/Z hard limits (WorkspaceSafetyEnvelope) | PASS; NaN rejected (comparison fails closed) |
| TCP never below board surface | PASS after fix (H1) |
| Vertical descent/ascent, Z_approach >= 80 mm | PASS after fix (H1). Plan uses identical X,Y for approach/pick/ascend waypoints; only Z changes |
| MockURDriver headless fidelity | PASS: same API as RealURDriver, same pose limits, state + DO tracking, no hardware/network |
| Socket timeouts | PASS: all connects 5 s, dashboard recv 5 s, RT reader 1 s recv + retry loop |
| Exception safety | PASS after fixes (H2, M1) |
| Full pytest | PASS (694) |

## HIGH (fixed)

- H1. Floor was absolute only (`z_min=-0.20`). `pick_z`/`transit_z` were validated only in `__post_init__`; they are
  mutable dataclass fields, so later assignment (e.g. `transit_z=0.01`) bypassed the 80 mm minimum and a `pick_z` below
  `board_z` bypassed the table floor. Fix: `BoardCoordinates.approach_pose` re-checks `transit_z >= TRANSIT_Z_MIN`;
  new `_check()` rejects any pose with `Z < board_z` at call time.
- H2. `RealURDriver.movel` is fire-and-forget; a new script on port 30002 preempts the running one, so back-to-back
  waypoints could skip the vertical ascent and produce a diagonal move through neighbouring pieces. Fix:
  `_wait_arrival()` blocks until RT-reported TCP is within 2 mm of target (30 s timeout -> `stopl` + `TimeoutError`).
  No-op when RT telemetry is not connected.

## MEDIUM

- M1 (fixed). `TrajectoryGenerator._execute` had no failure path: a mid-sequence error left the robot moving with
  vacuum state unknown. Now logs, sends `stopl`, re-raises. Vacuum is intentionally kept (do not drop a held piece).
- M1b (fixed). `_check_pose_limits` accepted short poses (IndexError / partial pose). Now requires 6 elements.
- M1c (fixed). `MoveSequencer._pick_to_pose` validated the graveyard slot only after the first moves started
  (slot pose itself never validated). Slot pose now checked before any motion.
- M2 (open). `park_pose` and graveyard slot Z are checked against the absolute envelope only, not against `board_z`.
  Mis-configured values below the table plane would pass. Add `Z >= board_z` check at construction.
- M3 (open). `RealURDriver` has no ack/error reading from port 30002 and no protective-stop detection; `_wait_arrival`
  covers motion completion only. Poll dashboard `safetymode` before each sequence on real hardware.
- M4 (open). `MockURDriver` caps simulated travel at 2 s and ignores accel/velocity args; fine for logic tests, not for timing.

## LOW

- Duplicate limit constants in `ur_driver.py` (`_SAFE_*`) and `WorkspaceSafetyEnvelope` (same values, drift risk). Driver raises
  `ValueError`, envelope raises `SafetyViolationError`.
- `DashboardClient._cmd` reads a single `recv(1024)`; multi-line responses may be truncated.
- `_RTReader` retry loop swallows only `OSError`; a malformed packet struct error would kill the thread (unlikely: fixed-length slice).
- Grip port literal `0` in graveyard.py duplicates `GRIPPER_DO_PORT`.

## Other checks

- No secrets, no shell/eval, no file I/O in robot modules. English comments. Units documented.
- UX audit (`docs/UX_Audit_*.md`): none present; nothing to verify.
- Regression tests added: `tests/test_audit_phase2.py` (4 tests).
- Not verified: real UR hardware (none available); `npm run build` not applicable (Python project).
