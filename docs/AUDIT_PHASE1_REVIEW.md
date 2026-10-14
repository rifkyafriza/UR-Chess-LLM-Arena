# Audit Report: Phase 1 (Vision, Simulation, Launcher, LLM Client)

Auditor: reviewer | Task: t_83fe3ad6 | Scope: src/vision/, src/simulation/web_app.py,
src/strategy/llm_client.py, main.py, tests/ (101 tests).

## Result

Certification: **APPROVED** (HIGH findings fixed in this audit; MEDIUM/LOW tracked below).
`python -m pytest -q` -> 101 passed, before and after fixes.

## Checklist

| Item | Status |
|------|--------|
| No hardcoded secrets in repo | PASS (regex scan of tracked files; `.env` ignored; key read from env / `~/.hermes/auth.json`) |
| Flask input validation | PASS after fix (H2) |
| Host binding safety | PASS after fix (H1) |
| LLM hallucinated/invalid UCI guardrail | PASS (`llm_client.get_best_move`: regex extract, `Move.from_uci`, legality check, deterministic legal fallback; all API errors -> fallback) |
| English comments/docstrings | PASS (non-ASCII only in box-drawing/em-dash characters) |
| Resource teardown | PASS after fix (M1) |
| UX audit report (`docs/UX_Audit_*.md`) | None present in repo; nothing to verify |

## HIGH (fixed)

- H1. All servers bound `0.0.0.0` (`main.py` sim/live, `web_app.run`, `live_stream.start`), with
  `debug=True` in `web_app.__main__` (Werkzeug debugger = remote code execution on LAN). Unauthenticated
  endpoints (move, reset, undo, camera feed) were exposed to the network.
  Fix: default host `127.0.0.1`; `debug` forced off in `live_stream`; `web_app.__main__` no longer debug.
  Opt-in LAN exposure via `run(host=...)` / `start(host=...)`.
- H2. `/api/move` trusted client JSON: `force=True` plus bare `int()`/`str()` casts raised 500 on bad
  input; `from_sq`/`to_sq` unvalidated (`to_sq[1]` IndexError); client `positions` accepted any marker
  ID/square and was written into server state.
  Fix: `get_json(silent=True)` + dict check, marker ID range 1-32, squares must match `^[a-h][1-8]$`
  (or `__captured__`), all failures return 400.

## MEDIUM

- M1. (fixed) `live_stream._capture_loop` never released `VideoCapture`. Now wrapped in try/finally.
- M2. (fixed) Undo bugs: reset did not clear `_undo_stack`; undo popped one history entry though a
  turn pushes two (white + LLM) moves. Now stores history length and truncates. Empty undo returns 400.
- M3. (fixed) `--model` from `main.py` set `CHESS_MODEL` but `web_app` ignored it. Now passed to
  `get_best_move`. Also `main.py --mode live` bypassed the capture thread (called `app.run` directly, so no
  camera); now calls `live_stream.start`.
- M4. (fixed) LLM exception text echoed to the client (`LLM error: {exc}`) could leak endpoint details;
  now generic message, detail stays in server log.
- M5. (open) `chess_game.py` contains placeholder `Bearer YOUR_API_KEY_HERE` and ignores the `--model`
  flag `main.py --mode cli` passes (it prompts via `input()`). CLI mode is functionally broken. Not a
  secret leak. Follow-up: make it reuse `llm_client._load_api_key` and accept `--model`.
- M6. (open) Simulation state (`_game`, `_undo_stack`) is module-global with no lock; Flask dev server is
  multi-threaded, so concurrent `/api/move` calls race. Acceptable for single-user local sim; add a
  `threading.Lock` if multi-user.
- M7. (open) Flask dev server used for all modes; acceptable locally, not for deployment.

## LOW

- L1. `llm_client`: `except (APIError, APITimeoutError, Exception)` is redundant; use `Exception`. Fine
  behaviourally (fail-safe fallback).
- L2. `llm_client._fallback_move` docstring says "deterministic-ish"; `random` import unused.
- L3. `llm_client._load_api_key` swallows broad exceptions at debug level; acceptable, key never logged.
- L4. `web_app` pawn promotion is auto-queen only; client `moved_id` decides promotion (server-checked
  by legality, so harmless).
- L5. `_check_9router` / `_port_busy` use hardcoded 127.0.0.1:20128; fine for current deployment.
- L6. Camera stream `_capture_loop` raises inside a daemon thread when camera is missing; main thread
  only waits 5 s then serves empty frames. Surface error to the user in a later task.

## Tests

101 tests pass. Gaps worth adding later: `/api/move` malformed-payload cases (400s), undo-after-two-moves
history length, live_stream capture teardown.

## Files changed by this audit

`main.py`, `src/simulation/web_app.py`, `src/vision/live_stream.py`, `docs/AUDIT_PHASE1_REVIEW.md`.
