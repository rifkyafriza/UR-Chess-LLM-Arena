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

### Task 7E: Reviewer Security, Kinematic & Config Audit (`docs/AUDIT_PHASE3_FULL_SYSTEM.md`)
- **Assignee**: `reviewer`
- **Parent**: Task 7D
- **Scope**: Full system audit covering:
  - Config file injection and path traversal security.
  - UR Feature Plane math correctness against Polyscope `pose_trans`.
  - State machine safety transitions during error/disconnect.
  - Total test suite coverage.
- **Acceptance Criteria**: Reviewer certifies full system readiness and produces `docs/AUDIT_PHASE3_FULL_SYSTEM.md`.
