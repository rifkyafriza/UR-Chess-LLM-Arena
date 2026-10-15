# Kanban Task Graph & Work Breakdown Structure (DAG)

## Project: UR Chess LLM Arena
**Board Slug**: `ur-chess-arena`  
**Tenant**: `ur-chess`  
**Workspace**: `/home/rifky/projects/github/UR-Chess-LLM-Arena`

---

## 1. Profiles & Responsibilities

| Profile | Model Config | Responsibility |
|---|---|---|
| `orchestrator` | `ag/claude-opus-4-6-thinking` | Project planning, task breakdown, status tracking, document sync |
| `vision-dev` | `ag/claude-sonnet-4-6` | OpenCV ArUco detection, homography warp, FEN generator, stability filter |
| `robotics-dev` | `ag/claude-sonnet-4-6` | URScript / RTDE Ethernet driver, coordinate transform, pick-and-place trajectories |
| `backend-dev` | `ag/claude-sonnet-4-6` | State machine game loop, LLM 9Router integration, python-chess fallback |
| `qa-dev` | `ag/claude-sonnet-4-6` | Mock hardware harness, automated unit & integration tests |
| `reviewer` | `ag/claude-opus-4-6-thinking` | Kinematic safety boundary audit, collision envelope checks, code review |

---

## 2. Task Graph Overview

```
[t_9fe6ddda: Scaffold & Deps] -------------------------+
   |                                                   |
   +-----------------------+                           |
   |                       |                           |
   v                       v                           v
[t_552c1c37: Mock Harness] [t_61ba99c1: ArUco Corners] [t_06: UR Socket Driver]
                           [t_25d013c7: Grid Contour]  |
                               |                       v
                               v                   [t_07: Coordinate Transform]
                           [t_1280be2c: Piece Grid]    |
                               |                       v
                               +-------------------+   [t_08: Trajectory Planner]
                               |                   |   |
                               v                   v   v
                           [t_89b9724d: FEN]   [t_92a117b5: Live Stream]
```

---

## 3. Detailed Task Specifications

### Task 1: Scaffold Project Structure & Dependencies (`t_9fe6ddda`)
- **Assignee**: `backend-dev`
- **Parent**: None
- **Scope**: Setup Python package structure (`src/vision`, `src/robot`, `src/strategy`, `src/orchestrator`), create `requirements.txt` with OpenCV, python-chess, numpy, scipy, flask.
- **Acceptance Criteria**: Running `python3 -c "import cv2, chess, flask; print('OK')"` succeeds cleanly.

### Task 2: Build Mock Hardware Harness for Headless Testing (`t_552c1c37`)
- **Assignee**: `qa-dev`
- **Parent**: Task 1 (`t_9fe6ddda`)
- **Scope**: Create mock video stream feeder (synthetic images with ArUco markers) and mock UR TCP socket server simulating robot feedback.
- **Acceptance Criteria**: Automated test suite can run in CI/headless WSL without physical camera or physical UR arm connected.

### Task 3: Board Corner Localization & Perspective Warp (Mode 1: ArUco Corners) (`t_61ba99c1`)
- **Assignee**: `vision-dev`
- **Parent**: Task 1 (`t_9fe6ddda`)
- **Scope**: Implement `src/vision/board_locator_aruco.py` detecting 4 outer corner ArUco markers (IDs 40-43) and producing warped $800 \times 800$ top-down image.
- **Acceptance Criteria**: Warped image outputs exact $100 \times 100$ px squares aligned with algebraic coordinates a1..h8.

### Task 3B: Pure Chessboard Grid & Perimeter Detection (Mode 2: No ArUco Corners) (`t_25d013c7`)
- **Assignee**: `vision-dev`
- **Parent**: Task 1 (`t_9fe6ddda`)
- **Scope**: Implement `src/vision/board_locator_grid.py` detecting outer black framing contour and 8x8 alternating grid lines without corner markers, tested against `assets/reference_board.jpg`.
- **Acceptance Criteria**: Evaluates `assets/reference_board.jpg` and successfully extracts four board corners and warps to identical canonical $800 \times 800$ frame.

### Task 4: ArUco Piece Detection & 8x8 Occupancy Grid Mapping (`t_1280be2c`)
- **Assignee**: `vision-dev`
- **Parent**: Task 3 (`t_61ba99c1`), Task 3B (`t_25d013c7`)
- **Scope**: Implement `src/vision/piece_detector.py` reading pieces with IDs 1-32 and mapping their centers to the corresponding board grid cell.
- **Acceptance Criteria**: Given warped image with all 32 pieces, returns correct initial board matrix with zero mismatches.

### Task 5: FEN Serializer & Human Turn Settle Filter (`t_89b9724d`)
- **Assignee**: `vision-dev`
- **Parent**: Task 4 (`t_1280be2c`)
- **Scope**: Implement `src/vision/fen_serializer.py` converting matrix to standard FEN notation. Implement motion stabilization filter requiring 15 consecutive stable frames before emitting settled state.
- **Acceptance Criteria**: Output FEN string passes `chess.Board(fen).is_valid()` validation test.

### Task 5B: Real-time Live Web Stream & Vision Overlay Server (`t_92a117b5`)
- **Assignee**: `vision-dev`
- **Parent**: Task 4 (`t_1280be2c`)
- **Scope**: Implement `src/vision/live_stream.py` providing an MJPEG / HTTP web endpoint rendering real-time camera frames with detected ArUco bounding boxes, IDs, and live board state display side-by-side with SVG board.
- **Acceptance Criteria**: Web UI displays video stream with responsive real-time overlays (< 100ms latency) and synchronized square occupancy.

### Task 5C: Benchmark Real Board Photo with Both Vision Modes (`t_e5778cc1`)
- **Assignee**: `vision-dev`
- **Parent**: Task 5B (`t_92a117b5`)
- **Scope**: Run validation on `assets/test_board_with_pieces.jpg` evaluating Mode 1 (ArUco corners) and Mode 2 (Grid & contour). Detect 32 pieces, map to algebraic squares, verify against initial position, serialize FEN, and output `assets/benchmark_result.jpg`.
- **Acceptance Criteria**: 32/32 pieces detected on ground truth squares, FEN matches standard initial position, and visual verification overlay generated.

### Task 5D: 9Router LLM Strategy Client & Guardrail (`t_strategy_llm`)
- **Assignee**: `backend-dev`
- **Parent**: Task 1 (`t_9fe6ddda`)
- **Scope**: Implement `src/strategy/llm_client.py` connecting to `http://127.0.0.1:20128/v1` (`ag/claude-sonnet-4-6`). Formulate chess prompt with FEN and legal UCI moves. Sanitize response, validate against `python-chess` legal moves, with retry / legal move fallback.
- **Acceptance Criteria**: Unit test confirms LLM queries succeed and hallucinated moves trigger fallback to legal move within 100ms.

### Task 5E: Interactive Web Simulation Mode with ArUco Canvas & Live SVG (`t_sim_web`)
- **Assignee**: `backend-dev`
- **Parent**: Task 5D, Task 5B
- **Scope**: Implement `src/simulation/web_app.py` (Flask on port 5060) featuring split-view browser GUI:
  1. Left: Draggable ArUco pieces (IDs 1-32) on board photo background (`assets/reference_board.jpg`).
  2. Right: Synchronized digital chessboard rendered with `chess.svg.board`, move log, and FEN.
  3. Closed-loop flow: Human drag-and-drop -> virtual frame capture -> real CV pipeline -> FEN serialize -> 9Router LLM move -> auto-move black ArUco piece on canvas.
- **Acceptance Criteria**: User can play a full turn via browser drag-and-drop, vision pipeline detects move, LLM replies, and both visual board and SVG board synchronize.

### Task 5F: Unified Multi-Mode CLI Launcher `main.py` (`t_3ffa63b4`)
- **Assignee**: `backend-dev`
- **Parent**: Task 5E
- **Scope**: Implement `main.py` at repository root providing an interactive menu and CLI arguments (`--mode {sim,live,benchmark,cli,test}`) to easily select and launch any system mode.
- **Acceptance Criteria**: Running `python3 main.py` prompts interactive selection or executes requested mode flag cleanly.

### Task 5G: Code Quality, Security & Architecture Audit (`t_audit_phase1`)
- **Assignee**: `reviewer`
- **Parent**: Task 5F (`t_3ffa63b4`)
- **Scope**: Perform comprehensive security, code quality, and architecture review on all modules delivered in Phase 1 & Simulation (`src/vision/`, `src/simulation/`, `src/strategy/`, `main.py`, and `tests/`):
  1. Security review: API keys handling, network endpoints safety (Flask host bindings, CORS, input sanitization).
  2. Chess domain validation: Illegal move handling, board state synchronization, FEN boundary checks.
  3. Coding standards: English comments & docstrings, proper type hints, error handling.
  4. Test coverage verification: Verify full pytest suite passes and edge cases are covered.
- **Acceptance Criteria**: Review report produced in `docs/AUDIT_PHASE1_REVIEW.md`, certifying production readiness or listing actionable fixes. Status: **APPROVED** (`t_83fe3ad6`).

### Task 6A: UR Robot Driver & Communication Wrapper (`t_dcfb4e1c`)
- **Assignee**: `robotics-dev`
- **Parent**: Task 1 (`t_9fe6ddda`)
- **Scope**: Implement `src/robot/ur_driver.py` with dual-backend architecture:
  1. `DashboardClient`: TCP socket on port 29999 (ASCII commands: power_on, brake_release, power_off, play, stop, get_robot_mode, get_safety_mode, close_popup).
  2. `RealURDriver`: TCP socket on port 30002/30003 (URScript motion: movel, movej, stopl, set_digital_out, set_tool_digital_out) and binary telemetry unpacking.
  3. `MockURDriver`: Pure Python zero-install in-memory simulator tracking TCP Cartesian pose, joints, and digital output states.
  4. Factory `create_robot_driver(mode='sim'|'real', ...)`.
- **Acceptance Criteria**: Unit tests in `tests/test_robot_driver.py` pass for both Mock and Real backends.

### Task 6B: Board-to-Robot Frame Coordinates & Safety Envelope (`t_57fb6be1`)
- **Assignee**: `robotics-dev`
- **Parent**: Task 6A (`t_dcfb4e1c`)
- **Scope**: Implement `src/robot/coordinates.py` mapping algebraic squares (`a1`–`h8`) to UR robot 3D Cartesian TCP coordinates $[X,Y,Z,Rx,Ry,Rz]$ with tool pointing downward, and `WorkspaceSafetyEnvelope` enforcing hard clamping ($X, Y, Z_{\min}$) preventing collisions with table.
- **Acceptance Criteria**: 64 squares verified and out-of-bounds queries raise `SafetyViolationError`.

### Task 6C: Trajectory Waypoint Generator & Graveyard Sequencer (`t_626934a5`)
- **Assignee**: `robotics-dev`
- **Parent**: Task 6B (`t_57fb6be1`)
- **Scope**: Implement `src/robot/trajectory.py` (approach $Z \ge 80\text{mm}$, vertical descent, grasp, ascent, transit, place, retract, park) and `src/robot/graveyard.py` (16 White / 16 Black off-board slots, 2-stage capture sequencer, castling sequencer).
- **Acceptance Criteria**: Unit tests in `tests/test_trajectory.py` validate vertical clearance and capture/castling sequencing.

### Task 6D: Comprehensive Test Suite for Robot Module (`t_51414546`)
- **Assignee**: `qa-dev`
- **Parent**: Task 6C (`t_626934a5`)
- **Scope**: Implement `tests/test_robot.py` covering MockURDriver kinematics, RealURDriver mock server communication, coordinates calibration, and trajectory generation under pytest.
- **Acceptance Criteria**: All robot tests pass with 100% assertions satisfied.

### Task 6E: Kinematic Safety, Collision & Code Audit (`t_2545c775`)
- **Assignee**: `reviewer`
- **Parent**: Task 6D (`t_51414546`)
- **Scope**: Perform independent code, kinematic safety, and collision audit. Verify table boundary crash protection ($Z_{\min}$), vertical motion guarantee, socket timeouts, and write `docs/AUDIT_PHASE2_ROBOTICS.md`.
- **Acceptance Criteria**: Reviewer certifies zero collision risk and production approval. Status: **APPROVED** (`t_2545c775`).

### Task 7A: Centralized Configuration System (`config/arena_config.yaml` & `src/config.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 6E (`t_2545c775`)
- **Scope**: Create `config/arena_config.yaml` and a typed configuration loader `src/config.py` using Python standard dataclasses. Centralize robot settings (mode, host, ports, feature plane pose, pitch, speeds, safety envelope), vision settings, strategy/9Router settings, and simulation web settings. Expose `load_config(path)` and CLI override capabilities.
- **Acceptance Criteria**: Configuration loads cleanly, validates fields, and provides defaults when config file is absent. Unit tests in `tests/test_config.py`.

### Task 7B: UR Feature Plane Support & Calibration Helper (`src/robot/coordinates.py` & `scripts/calibrate_feature.py`)
- **Assignee**: `robotics-dev`
- **Parent**: Task 7A
- **Scope**:
  1. Add native support for UR Feature Plane in `src/robot/coordinates.py`. Allow user to define Feature Plane pose $[X, Y, Z, Rx, Ry, Rz]$ matching Polyscope 3-point plane (Point 1: a1, Point 2: h1, Point 3: a8). In local board frame, square coordinates simplify to $(file\_idx \times dx, rank\_idx \times dy, z)$. Compute base-frame poses using rigid-body 3D transform (Rodrigues rotation vector + translation) or issue native URScript `pose_trans()`.
  2. Implement interactive CLI calibration tool `scripts/calibrate_feature.py` that guides the user to touch a1, h1, a8 with robot freedrive, calculates plane pose, and saves it into `config/arena_config.yaml`.
- **Acceptance Criteria**: Unit tests in `tests/test_feature_plane.py` verify 3D local-to-base transformation matching `pose_trans` mathematics.

### Task 7C: Master Orchestrator Finite State Machine (`src/orchestrator/master_fsm.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 7B
- **Scope**: Implement `MasterOrchestrator` in `src/orchestrator/master_fsm.py` coordinating Vision, 9Router LLM Strategy, and Robot Motion Controller:
  - States: `INITIALIZING`, `WAIT_HUMAN_MOVE`, `PROCESSING_VISION`, `THINKING_LLM`, `ROBOT_MOVING`, `VERIFYING_BOARD`, `GAME_OVER`, `ERROR_SAFE_STOP`.
  - Seamless dual-mode: Works with `MockURDriver` (no physical robot) and `RealURDriver` (physical robot).
  - Integrates with `main.py` CLI launcher (`--mode arena`).
- **Acceptance Criteria**: Automated test validates full game turn cycle transitions through all states cleanly without deadlock.

### Task 7D: Comprehensive Integration & Game Loop Tests (`tests/test_master_orchestrator.py`)
- **Assignee**: `qa-dev`
- **Parent**: Task 7C
- **Scope**: End-to-end integration test suite simulating full game turns:
  - Simulated human move e2e4 -> vision board sync -> LLM move e7e5 -> trajectory generator -> mock robot execution.
  - Test capture moves with graveyard sequencing.
  - Test castling moves.
  - Test graceful recovery from illegal moves and communication drops.
- **Acceptance Criteria**: Pytest suite passes 100% with headless mock harness.

### Task 7E: Reviewer Security, Kinematic & Config Audit (`t_dfca0dda`)
- **Assignee**: `reviewer`
- **Parent**: Task 7D (`t_4e7107d2`)
- **Scope**: Full system audit covering:
  - Config file injection and path traversal security.
  - UR Feature Plane math correctness against Polyscope `pose_trans`.
  - State machine safety transitions during error/disconnect.
  - Total test suite coverage.
- **Acceptance Criteria**: Reviewer certifies full system readiness and produces `docs/AUDIT_PHASE3_FULL_SYSTEM.md`. Status: **APPROVED** (`t_dfca0dda`).

### Task 8A: Fix Mode Live vision_app NameError in main.py (`main.py` & `tests/test_launcher_live.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 7E (`t_dfca0dda`)
- **Scope**: Fix `run_live()` in `main.py` where `vision_app.run` raises `NameError: name 'vision_app' is not defined`. Call `src.vision.live_stream.start(camera_index=camera, port=port)` correctly or export `app as vision_app`. Add unit tests in `tests/test_launcher_live.py` verifying launcher entrypoint executes without NameError.
- **Acceptance Criteria**: `run_live` invokes `live_stream.start` with proper parameters; verified by automated unit tests.

### Task 8B: Implement Player vs Player Mode & Fix Board State Synchronization in Web Simulation (`src/simulation/web_app.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 8A
- **Scope**:
  1. Add Game Mode support in `src/simulation/web_app.py`: toggle between "ai" (LLM opponent) and "pvp" (Player vs Player / Pass-and-Play).
  2. UI enhancements in HTML template:
     - Mode dropdown / toggle: `[Player vs AI | Player vs Player]`.
     - In PvP mode: Allow Black player to drag Black pieces (IDs 17-32) on Black's turn (`serverTurn === 'black'`).
     - In PvP mode: `/api/move` processes Black moves legally via `python-chess` without calling the LLM.
     - In AI mode: If LLM is unavailable or fails, display clear feedback without locking the game board.
  3. Guarantee zero-robot operation: Ensure simulation runs 100% in-browser/memory without attempting any socket connection to physical UR robots.
  4. Fix web board display and ArUco canvas state synchronization:
     - Captured pieces: Do not render `__captured__` pieces on canvas or off-screen negative coordinates. Omit or place in captured tray.
     - Synchronize captured pieces on White moves: When White captures a Black piece, mark the victim marker as `__captured__` in `_game.positions` so pieces don't overlap on canvas.
     - Castling synchronization: When either player castles, update both the King and the Rook marker coordinates in `_game.positions`.
     - En-passant synchronization: Mark captured pawn as `__captured__`.
     - Verify SVG right pane and ArUco canvas left pane match precisely.
- **Acceptance Criteria**: Both AI and PvP modes work smoothly. Dragging works for Black in PvP. Canvas accurately mirrors python-chess board without duplicate/overlapping pieces.

### Task 8C: Comprehensive Integration Tests for Live Stream, PvP Simulation & Board Display (`tests/test_simulation_pvp.py`)
- **Assignee**: `qa-dev`
- **Parent**: Task 8B
- **Scope**: Implement test suite in `tests/test_simulation_pvp.py`:
  - Test `main.py` `run_live` dispatch without NameError.
  - Test `/api/move` in PvP mode for full White + Black alternating moves.
  - Test ArUco positions dictionary during captures, castling, and en-passant to ensure no pieces overlap.
  - Test `/api/mode` endpoint and mode switching.
  - Test simulation operates cleanly with zero external robot connection.
- **Acceptance Criteria**: 100% test pass rate in pytest.

### Task 8D: Reviewer Audit for Bug Fixes, PvP Mode & Web Simulation Integrity (`docs/AUDIT_BUGFIXES_SIM_PVP.md`)
- **Assignee**: `reviewer`
- **Parent**: Task 8C
- **Scope**: Independent audit of:
  - `main.py` live mode invocation.
  - PvP mode state machine and input validation in Flask.
  - Canvas & SVG board synchronization correctness.
  - Zero-robot standalone operation in simulation mode.
  - Full test suite passing status.
- **Acceptance Criteria**: Reviewer certifies all 3 bug fixes and produces `docs/AUDIT_BUGFIXES_SIM_PVP.md`. Status: **APPROVED** (`t_996ec590`).

### Task 9A: Update CLI main.py, Fix Benchmark sys.path, & Migrate Default Sim Port (`main.py`, `scripts/benchmark_board.py`, `src/config.py`, `config/arena_config.yaml`)
- **Assignee**: `backend-dev`
- **Parent**: Task 8D (`t_996ec590`)
- **Scope**:
  1. Fix `scripts/benchmark_board.py`: add repo root to `sys.path` so running `python scripts/benchmark_board.py` directly or via `main.py` never fails with `ModuleNotFoundError: No module named 'src'`.
  2. In `main.py`: inject `PYTHONPATH` with repo root into subprocess environment in `run_benchmark()`, `run_cli()`, `run_tests()`.
  3. Update `main.py` interactive menu:
     - Add option `[6] Arena Mode (Full FSM: human vs LLM + robot)`.
     - Sub-prompt for robot mode: `1: Sim (Mock UR), 2: Real UR robot`.
     - Update menu header and instructions.
  4. Update `main.py` non-interactive CLI:
     - Clarify help messages and usage examples.
  5. Migrate default simulation port from 5060 to **5070**:
     - Change in `config/arena_config.yaml` (`simulation.port: 5070`).
     - Change in `src/config.py` (default `port: int = 5070`).
     - Change in `src/simulation/web_app.py` (default `port: int = 5070`).
     - Change in `main.py` (`DEFAULT_SIM_PORT: int = 5070`).
     - Update existing tests (`tests/test_config.py`, `tests/test_main_launcher.py`) to assert 5070.
- **Acceptance Criteria**: `python3 main.py --mode benchmark` runs without `No module named 'src'`. Port 5060 is nowhere used as default. Option 6 works in interactive menu.

### Task 9B: Dual-Pane Live Stream with Live Camera & Digital SVG Chessboard (`src/vision/live_stream.py`)
- **Assignee**: `vision-dev`
- **Parent**: Task 9A
- **Scope**:
  1. Update `src/vision/live_stream.py` web UI to a split dual-pane layout:
     - Left pane: Live camera video feed (`/video`) with ArUco corners/piece detection overlay and thumbnail.
     - Right pane: Digital chessboard SVG visualization rendering the detected board state in real-time.
     - Status card: Current localization mode (`ARUCO` / `GRID`), detected piece count (White / Black), current detected FEN string, and settle/stability status.
  2. Implement/update endpoint:
     - Add or enrich `/state` to return current `fen` and `svg` (rendered via `chess.svg.board()`) or `/board_svg` endpoint.
     - Poll `/state` in frontend to update the right-hand digital board seamlessly.
- **Acceptance Criteria**: Accessing `http://127.0.0.1:5050` renders both live camera feed and synced digital SVG chessboard.

### Task 9C: Comprehensive README.md Documentation Update (`README.md`)
- **Assignee**: `backend-dev`
- **Parent**: Task 9A
- **Scope**:
  1. Update `README.md` to reflect all latest features:
     - Quick start instructions for computer (WSL2 / Linux / Windows).
     - New default port: `5070` for Web Simulation (Player vs Player and Player vs AI).
     - Live Camera Stream port `5050` with dual-pane camera + digital SVG board display.
     - Interactive menu options `[1-6]`, including Arena mode.
     - Benchmark command and instructions.
     - Centralized config `config/arena_config.yaml` and Feature Plane calibration.
  2. Ensure clean formatting, zero broken paths, all English documentation.
- **Acceptance Criteria**: `README.md` is complete, accurate, and reflects the current state of the codebase.

### Task 9D: Comprehensive Test Suite for Phase 5 Refinements (`tests/test_phase5_refinements.py`)
- **Assignee**: `qa-dev`
- **Parent**: Task 9B
- **Scope**:
  1. Test `scripts/benchmark_board.py` execution via subprocess returns exit code 0 without `ModuleNotFoundError`.
  2. Test `main.py` interactive menu handling option 6 (Arena mode).
  3. Test port 5070 is default across config, launcher, and simulation.
  4. Test live stream `/state` and digital SVG board rendering.
  5. Run full test suite and confirm 100% pass.
- **Acceptance Criteria**: All phase 5 tests pass; total test suite 100% green.

### Task 9E: Reviewer Security, Port Verification & Dual-Pane Audit (`docs/AUDIT_PHASE5_REFINEMENTS.md`)
- **Assignee**: `reviewer`
- **Parent**: Task 9D
- **Scope**:
  1. Verify zero references to port 5060 as default anywhere in code and config.
  2. Verify benchmark runs cleanly from any working directory.
  3. Verify dual-pane live stream functionality and HTML injection safety.
  4. Verify interactive and non-interactive launcher robustness.
  5. Verify full test suite passing status and produce `docs/AUDIT_PHASE5_REFINEMENTS.md`.
- **Acceptance Criteria**: Reviewer certifies Phase 5 deliverables with APPROVED status. Status: **APPROVED** (`t_ddda4e5b`).

### Task 10A: Implement Interactive Arena Config Editor (`src/config_editor.py` & `main.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 9E (`t_ddda4e5b`)
- **Scope**:
  1. Implement `src/config_editor.py`:
     - Load active configuration using `src.config.load_config()`.
     - Provide interactive text menu to:
       - View current configuration formatted cleanly.
       - Edit Robot settings: mode (`sim` / `real`), host IP, move_velocity, move_acceleration.
       - Edit Vision settings: camera_index, locator_mode (`aruco` / `contour`), stream_port.
       - Edit Strategy settings: model name, api_base, timeout.
       - Edit Simulation settings: host, port.
     - Validate all user modifications through `src.config._validate()` before writing.
     - Safely serialize and persist changes to `config/arena_config.yaml` using PyYAML.
     - Add helper function `interactive_config_editor(config_path: Optional[str] = None) -> None`.
  2. Integrate into `main.py`:
     - Update interactive menu with option `[7] Config Editor (View & edit arena_config)`.
     - Update valid choices `0-7`.
     - Add `--mode config` CLI flag to argument parser to launch config editor directly.
  3. Update `README.md` to document the new config editor option and CLI command.
- **Acceptance Criteria**: Running `python3 main.py` and choosing 7 opens config editor; modifying a value validates and saves to `config/arena_config.yaml`; invalid inputs are safely rejected.

### Task 10B: Comprehensive Test Suite for Interactive Config Editor (`tests/test_config_editor.py`)
- **Assignee**: `qa-dev`
- **Parent**: Task 10A
- **Scope**:
  1. Test `src/config_editor.py`:
     - Test viewing current config output.
     - Test updating robot parameters (mode, host, speed) with file persistence.
     - Test updating vision parameters (camera_index, stream_port, locator_mode).
     - Test updating strategy parameters (model, timeout, api_base).
     - Test updating simulation parameters (port).
     - Test invalid inputs (negative speeds, out-of-range ports, invalid modes) are rejected without corrupting config file.
  2. Test `main.py` launcher integration:
     - Test interactive menu option `7` dispatches `run_config()`.
     - Test `--mode config` flag dispatches `run_config()`.
  3. Ensure 100% test pass rate across the entire test suite.
- **Acceptance Criteria**: Comprehensive test coverage with 0 failures in pytest.

### Task 10C: Reviewer Security, Validation & Persistence Audit (`docs/AUDIT_CONFIG_EDITOR.md`)
- **Assignee**: `reviewer`
- **Parent**: Task 10B
- **Scope**:
  1. Audit `src/config_editor.py` for safe YAML serialization (no arbitrary object execution, safe loading/dumping).
  2. Verify input validation enforces safety bounds (speed caps, valid IP formats, port ranges).
  3. Verify file permissions and safe atomic writing of `arena_config.yaml`.
  4. Verify complete test suite passing status.
  5. Produce audit report `docs/AUDIT_CONFIG_EDITOR.md` and provide CERTIFIED/APPROVED verdict.
- **Acceptance Criteria**: Reviewer certifies config editor safety and publishes audit report. Status: **APPROVED** (`t_6c7233ea`).

---

## Phase 7: Vision Video Benchmark, Tracking Optimization & Dual-Pane Display Fixes

### Task 11A: Optimize ArUco Detection, Piece Mapping, and Temporal Tracking (`src/vision/piece_detector.py`)
- **Assignee**: `vision-dev`
- **Parent**: Task 10C (`t_6c7233ea`)
- **Scope**:
  1. Define standard starting position piece mapping `ID_TO_PIECE_CHAR`:
     - IDs 1-8: R, N, B, Q, K, B, N, R
     - IDs 9-16: P
     - IDs 17-24: r, n, b, q, k, b, n, r
     - IDs 25-32: p
  2. Implement `build_piece_type_board_state(pieces: list[dict]) -> dict[str, str]` mapping square to exact piece char (not just color).
  3. Optimize ArUco detector parameters in `piece_detector.py`:
     - Tune adaptive thresholding parameters, subpixel corner refinement, and contrast enhancement (CLAHE / multi-pass) to maximize marker detection on raw frames.
  4. Implement `TemporalPieceTracker`:
     - Maintain piece positions across frames with configurable TTL / persistence (e.g. 5-10 frames) to eliminate transient 1-2 frame dropouts caused by motion or lighting flutter.
     - Emit stabilized list of pieces for downstream consumers.
- **Acceptance Criteria**: Detection on full resolution frames achieves 32/32 markers with stability, correctly identifying non-pawn pieces.

### Task 11B: Set Default Contour Locator, Digital SVG Board Piece Types, and Mock Video Live Stream (`src/vision/live_stream.py`, `src/config.py`, `config/arena_config.yaml`)
- **Assignee**: `vision-dev`
- **Parent**: Task 11A
- **Scope**:
  1. Update default locator mode to `"contour"` in:
     - `config/arena_config.yaml`
     - `src/config.py` (`VisionConfig.locator_mode = "contour"`)
  2. In `src/vision/live_stream.py`:
     - Prioritize contour grid detection (`board_locator_grid.localize_board`) as primary locator when locator_mode is contour or on boards without corner markers.
     - Call `detect_pieces_from_frame(frame, matrix)` on full-resolution frame instead of downsampled warped image.
     - Use `build_piece_type_board_state()` and `serialize_fen()` with real piece types so SVG displays Knights, Rooks, Bishops, Kings, Queens, and Pawns accurately.
     - Use `TemporalPieceTracker` to prevent flickering and maintain steady 32/32 count.
     - Add `--video` support to accept video file path (loops playback when EOF reached).
  3. In `main.py`:
     - Support `--mode live --video <path>` CLI option.
     - In interactive menu `[2] Live Camera Stream`, allow user to choose between physical camera index or mock video file.
     - Note: ensure test video files are strictly ignored by `.gitignore` and never committed to git.
- **Acceptance Criteria**: Live stream web interface displays stable 32/32 pieces, real piece icons on SVG board, and loops mock video cleanly.

### Task 11C: Video Tracking Benchmark CLI (`scripts/benchmark_board.py` & `main.py`)
- **Assignee**: `backend-dev`
- **Parent**: Task 11B
- **Scope**:
  1. Extend `scripts/benchmark_board.py`:
     - Add `--video <path>` CLI argument.
     - Process video frame by frame, measuring:
       - Total frames processed
       - Board localization success rate (%)
       - Average detected piece count (target 32/32)
       - FEN stability and initial position match rate (%)
       - Processing FPS (throughput)
     - Output clean formatted benchmark summary report to stdout.
  2. Wire into `main.py`:
     - Support `python3 main.py --mode benchmark --video <path>`.
     - In interactive menu `[3] Vision Benchmark`, offer choice: Static Image Benchmark vs Video Benchmark.
- **Acceptance Criteria**: Running benchmark on video outputs comprehensive statistics with high accuracy.

### Task 11D: Comprehensive Test Suite for Phase 7 (`tests/test_vision_video_refinements.py`)
- **Assignee**: `qa-dev`
- **Parent**: Task 11C
- **Scope**:
  1. Test `src/vision/piece_detector.py`:
     - Test `ID_TO_PIECE_CHAR` and `build_piece_type_board_state` maps IDs 1-32 to expected squares and symbols.
     - Test `TemporalPieceTracker` preserves piece states across momentary dropouts and expires after TTL.
  2. Test `src/config.py` default locator is `"contour"`.
  3. Test `src/vision/live_stream.py`:
     - Test piece type SVG generation contains 'knight', 'rook', 'bishop', 'queen', 'king'.
     - Test video file source loop handling.
  4. Test `scripts/benchmark_board.py` video benchmark function with a synthetic or short clip.
  5. Run full test suite to guarantee 100% green pass.
- **Acceptance Criteria**: All new tests pass, zero regressions across the 918 existing tests.

### Task 11E: Reviewer Security, Performance & Gitignore Audit (`docs/AUDIT_VISION_VIDEO_REFINEMENTS.md`)
- **Assignee**: `reviewer`
- **Parent**: Task 11D
- **Scope**:
  1. Audit `.gitignore` and repository working tree to verify NO `*.mp4` or video files are staged or committed.
  2. Audit `TemporalPieceTracker` for memory leaks, bounded state size, and concurrency safety with Flask lock.
  3. Audit FEN generation and board SVG rendering for accuracy.
  4. Verify full test suite passes.
  5. Produce audit report `docs/AUDIT_VISION_VIDEO_REFINEMENTS.md` with CERTIFIED/APPROVED verdict.
- **Acceptance Criteria**: Reviewer certifies Phase 7 deliverables and publishes audit report.
