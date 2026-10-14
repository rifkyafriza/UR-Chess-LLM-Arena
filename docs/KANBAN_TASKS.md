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

### Task 5F: Unified Multi-Mode CLI Launcher `main.py` (`t_main_launcher`)
- **Assignee**: `backend-dev`
- **Parent**: Task 5E
- **Scope**: Implement `main.py` at repository root providing an interactive menu and CLI arguments (`--mode {sim,live,benchmark,cli,test}`) to easily select and launch any system mode.
- **Acceptance Criteria**: Running `python3 main.py` prompts interactive selection or executes requested mode flag cleanly.

### Task 6: UR Ethernet Socket Driver & Communication Wrapper
- **Assignee**: `robotics-dev`
- **Parent**: Task 1
- **Scope**: Implement `src/robot/ur_driver.py` handling TCP/IP communication on port 30002/30003, heartbeat monitoring, and error handling.
- **Acceptance Criteria**: Driver connects, transmits valid URScript motion commands, and gracefully handles disconnects.

### Task 7: Board-to-Robot Frame Coordinate Transformation
- **Assignee**: `robotics-dev`
- **Parent**: Task 6
- **Scope**: Implement `src/robot/coordinates.py` converting algebraic notation (`e2`, `e4`) into UR cartesian TCP coordinates $[X, Y, Z, R_x, R_y, R_z]$.
- **Acceptance Criteria**: Math unit test confirms linear interpolation across all 64 squares matches calibrated board spacing within $\pm 0.5\text{mm}$.

### Task 8: Trajectory Waypoint Generator (Pick, Transit, Place, Park)
- **Assignee**: `robotics-dev`
- **Parent**: Task 7
- **Scope**: Implement `src/robot/trajectory.py` generating safe waypoints with high-altitude transit clearance ($Z \ge 80\text{mm}$) and gripper digital output triggers.
- **Acceptance Criteria**: Generated URScript guarantees vertical descent/ascent without diagonal clipping across adjacent squares.

### Task 9: Graveyard Allocation & Special Move Sequences
- **Assignee**: `robotics-dev`
- **Parent**: Task 8
- **Scope**: Implement capture routines (removing target piece to side graveyard slots) and castling routines (two consecutive pick-and-place motions).
- **Acceptance Criteria**: Unit test validates capture sequence issues 2-phase pick-and-place with distinct victim graveyard coordinate.

### Task 10: Refactor Game Engine for Physical Board Synchronization
- **Assignee**: `backend-dev`
- **Parent**: Task 1
- **Scope**: Modularize `chess_game.py` into `src/strategy/engine.py` decoupling terminal inputs and allowing external FEN injection from the vision pipeline.
- **Acceptance Criteria**: Engine accepts external FEN input and provides valid legal moves without interactive terminal prompts.

### Task 11: 9Router LLM Client with Fallback Guardrail
- **Assignee**: `backend-dev`
- **Parent**: Task 10
- **Scope**: Implement `src/strategy/llm_client.py` connecting to `http://localhost:20128/v1` using `ag/claude-sonnet-4-6`. Implement self-healing retry and `python-chess` legal move fallback.
- **Acceptance Criteria**: Simulated hallucinated LLM responses trigger automatic legal move fallback within 100ms.

### Task 12: Master Orchestrator Finite State Machine
- **Assignee**: `backend-dev`
- **Parent**: Task 5, Task 9, Task 11
- **Scope**: Implement `src/orchestrator/fsm.py` tying together Vision, Strategy, and Robotics modules into the full game loop.
- **Acceptance Criteria**: Full state transition loop executes seamlessly from Human Move Detection -> Board Sync -> LLM Move -> Robot Actuation -> State Verification.

### Task 13: Kinematic Safety Envelope & Collision Audit
- **Assignee**: `reviewer`
- **Parent**: Task 12
- **Scope**: Perform comprehensive static and kinematic review of all generated URScript commands, workspace bounds, speed limits, and E-Stop handlers.
- **Acceptance Criteria**: Reviewer certifies zero out-of-bounds TCP coordinates and confirms mandatory safety limits are enforced.

### Task 14: End-to-End Simulation Dry-Run
- **Assignee**: `qa-dev`
- **Parent**: Task 13
- **Scope**: Run full simulated 10-turn chess game using the mock harness, validating state consistency, move legality, and trajectory logs.
- **Acceptance Criteria**: Test script completes 10 turns without errors, deadlocks, or illegal moves.
