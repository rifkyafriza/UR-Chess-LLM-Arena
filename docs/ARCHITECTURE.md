# Technical Architecture Specification

## Project: UR Chess LLM Arena
**Status**: Active Specification  
**Version**: 1.0.0  
**Target Platform**: Linux (WSL2 / Ubuntu 24.04), Universal Robots CB3 / e-Series, Python 3.10+

---

## 1. System Block Diagram

```
+---------------------------------------------------------------------------------+
|                                 PHYSICAL WORLD                                  |
|                                                                                 |
|   +--------------------+     +---------------------+     +------------------+   |
|   | Global Top Camera  |     |   8x8 Chessboard    |     | UR Robotic Arm   |   |
|   | (Overhead RGB)     |     |  with ArUco Pieces  |     | & Gripper E-Stop |   |
|   +---------+----------+     +----------+----------+     +--------^---------+   |
+-------------|---------------------------|-------------------------|-------------+
              | Video Stream              | Visual Markers          | Ethernet /
              | (USB/RTSP)                |                         | URScript TCP
+-------------v---------------------------v-------------------------|-------------+
|                               SOFTWARE PIPELINE                   |             |
|                                                                   |             |
|  +-------------------------------------------------------------+  |             |
|  | [1] Vision Module (OpenCV)                                  |  |             |
|  | - Corner Homography (Warp to 800x800 px)                    |  |             |
|  | - ArUco ID Detection (Dictionary: DICT_4X4_50)              |  |             |
|  | - Grid Occupancy Map (a1 - h8)                              |  |             |
|  | - Stability & Human Hand Clearance Filter                   |  |             |
|  | -> Output: FEN String & Detected Move (e.g. "e7e5")         |  |             |
|  +------------------------------+------------------------------+  |             |
|                                 | FEN & Move                      |             |
|  +------------------------------v------------------------------+  |             |
|  | [2] Game State Orchestrator & Rule Engine                   |  |             |
|  | - python-chess Board Verification                           |  |             |
|  | - Turn Management & Game Clock                              |  |             |
|  | - Graveyard Allocation Table                                |  |             |
|  | - SVG / HTML UI Renderer (Browser preview)                  |  |             |
|  +--------------+-------------------------------+--------------+  |             |
|                 | FEN + Legal Moves             | Target Move     |             |
|  +--------------v---------------+  +------------v--------------+  |             |
|  | [3] AI Strategy (LLM Engine) |  | [4] Robotics Controller   +--+             |
|  | - Prompt: FEN & UCI Legal    |  | - Square to 3D Base Pose  |                |
|  | - 9Router Gateway Endpoint   |  | - Approach / Retract Z    |                |
|  | - Models: Claude 3.5 Sonnet  |  | - Capture Sub-routines    |                |
|  | - Fallback: python-chess     |  | - Digital Tool Output I/O |                |
|  +------------------------------+  +---------------------------+                |
+---------------------------------------------------------------------------------+
```

---

## 2. Module Specifications

### Module 1: Vision Pipeline (`src/vision/`)
- **Camera Calibration**: Intrinsic parameters calibrated via standard chessboard pattern (`camera_matrix`, `dist_coeffs`).
- **Dual Localization Engine**:
  - **Mode 1 (ArUco Corner Anchors)**: 4 dedicated ArUco markers placed at board outer corners (`ID_CORNER_TL=40`, `ID_CORNER_TR=41`, `ID_CORNER_BR=42`, `ID_CORNER_BL=43`). Computes homography matrix $H$ warping outer boundary to $800 \times 800\text{ px}$.
  - **Mode 2 (Direct Grid & Contour Detection)**: Detects the 8x8 outer framing contour and internal alternating black-and-white grid without corner markers. Performs perspective rectification directly from the board's four outer corners.
- **Marker-to-Piece Mapping**: 32 distinct ArUco IDs mapped in `aruco_mapping.json`.
- **Temporal Filter**:
  - Requires $N = 15$ continuous stable frames without bounding box displacement ($\Delta < 3\text{px}$) to declare turn settled.
  - Automatically rejects frames if human hands or large shadows obscure the board grid.
- **Live Visualizer & HUD Overlay Stream**:
  - Generates an MJPEG / HTTP stream rendering detected ArUco bounding boxes, piece labels (e.g. `White King [ID 5]`), square coordinates (`e4`), and grid overlays in real time.

### Module 2: Robotics & Motion Planner (`src/robot/`)
- **Coordinate Systems**:
  - Chess Board Frame $\{B\}$: Origin at corner of square `a1`, $X$-axis along file (a->h), $Y$-axis along rank (1->8), $Z$-axis normal upwards.
  - UR Base Frame $\{R\}$: Rigid Euclidean transform ${}^R T_B = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$. Calibrated via 4-point touch-probe measurement.
- **Clearance Planes**:
  - $Z_{\text{grip}}$: Surface contact height of piece top + gripper offset.
  - $Z_{\text{hover}}$: $Z_{\text{grip}} + 35\text{mm}$ (pre-grasp / post-release alignment).
  - $Z_{\text{transit}}$: $Z_{\text{grip}} + 90\text{mm}$ (high-altitude transit above tallest piece).
  - $P_{\text{home}}$: Safe park pose outside camera field of view.
- **Physical Trajectory Pipeline**:
  ```
  P_home -> MoveL(Source_Hover) -> MoveL(Source_Grip) -> Gripper_ON() ->
  MoveL(Source_Hover) -> MoveL(Target_Hover) -> MoveL(Target_Grip) ->
  Gripper_OFF() -> MoveL(Target_Hover) -> MoveL(P_home)
  ```
- **Capture Sequence**:
  If target square is occupied:
  1. Pick victim piece from target square.
  2. Move to designated Graveyard bin coordinate ($G_k$).
  3. Release victim piece.
  4. Pick attacker piece from source square and place on target square.

### Module 3: Strategy & LLM Client (`src/strategy/`)
- **API Standard**: OpenAI-compatible `/v1/chat/completions` pointing to local gateway `http://localhost:20128/v1`.
- **System Prompt**:
  ```text
  You are a grandmaster chess engine.
  Given the board state in FEN and the list of legal UCI moves, select the single best strategic move.
  Respond ONLY with the exact UCI string (e.g., 'e2e4'). Do not provide explanations.
  ```
- **Self-Healing Guardrail**:
  1. Check if LLM response belongs to `board.legal_moves`.
  2. If invalid / malformed string: Retry with temperature 0.0 + error explanation.
  3. If second attempt fails: Fallback to highest evaluated legal move via `python-chess` / Minimax fallback to guarantee game continuity.

### Module 4: Master Orchestrator State Machine (`src/orchestrator/`)
The system follows a finite state machine (FSM):

```
[INIT] ---> [CAMERA_CALIBRATE] ---> [ROBOT_CALIBRATE]
                                           |
    +--------------------------------------+
    v
[STANDBY_HOME] <---------------------------------------------+
    |                                                        |
    v                                                        |
[WAIT_HUMAN_MOVE]                                            |
    | (Board changes & stabilizes)                           |
    v                                                        |
[VALIDATE_HUMAN_MOVE] --(Illegal)--> [ALERT_HUMAN_REVERT]    |
    | (Legal)                                                |
    v                                                        |
[CHECK_MATE_OR_DRAW] --(Game Over)--> [TERMINATE_GAME]       |
    | (Active)                                               |
    v                                                        |
[LLM_COMPUTE_MOVE]                                           |
    | (Valid UCI move)                                       |
    v                                                        |
[ROBOT_PLAN_TRAJECTORY]                                      |
    |                                                        |
    v                                                        |
[ROBOT_EXECUTE_MOTION]                                       |
    | (Motion complete)                                      |
    v                                                        |
[VERIFY_PHYSICAL_BOARD] -------------------------------------+
```

---

## 3. Communication Protocols & Ports

| Subsystem | Protocol | Port / Device | Payload Description |
|---|---|---|---|
| Camera -> PC | UVC / RTSP | `/dev/video0` or IP | 1080p @ 30 FPS RGB video |
| PC -> UR Controller | TCP/IP (URScript) | `30002` (Secondary) / `30003` (Realtime) | Motion scripts (`movel`, `movej`, `set_digital_out`) |
| PC -> UR RTDE | TCP/IP | `30004` | State feedback (actual tool pose, joint temps, safety status) |
| PC -> 9Router LLM | HTTP REST | `20128` (`/v1/chat/completions`) | JSON request/response with Bearer auth |
| PC -> Web Telemetry | HTTP / SSE | `5000` | Auto-refreshing SVG visualizer & game logs |

---

## 4. Hardware Safety Specifications

1. **Software Workspace Envelope**:
   - $X \in [X_{\min}, X_{\max}]$, $Y \in [Y_{\min}, Y_{\max}]$ restricted strictly to the chessboard bounding box + graveyard bin.
   - Any trajectory waypoint outside these bounds triggers immediate exception before sending packets to UR controller.
2. **Speed & Acceleration Limits**:
   - Joint speed limit: $0.75\text{ rad/s}$.
   - Linear TCP speed limit: $0.25\text{ m/s}$.
   - TCP acceleration limit: $0.5\text{ m/s}^2$.
3. **Emergency Disconnect**:
   - Socket heartbeat packet sent every 100ms. If PC connection drops, UR watchdog automatically halts robot arm movement.
