# UR Chess LLM Arena

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-854%20passed-brightgreen.svg)](#automated-tests)

Autonomous cyber-physical system: a Universal Robots (UR) collaborative arm plays physical chess against human opponents.

- **Perception**: Overhead camera + ArUco markers → FEN board state
- **Strategy**: Local LLM via 9Router (`ag/claude-sonnet-4-6`)
- **Actuation**: URScript / RTDE over Ethernet (ports 30002/30003)
- **Simulation**: Web-based SVG chessboard (PvP or Player vs AI) on port 5070

---

## Architecture

1. **Vision**: Global overhead camera, 4 corner ArUco markers for homography, 32 piece markers (IDs 1-32) → FEN serialization.
2. **Turn Settle**: State must be stable >= 1.5 s (hands cleared) before AI is called.
3. **AI Strategy**: FEN dispatched to LLM via 9Router; returns UCI move (e.g. `e2e4`).
4. **Robot Actuation**: UCI → 3D Cartesian waypoints → UR arm (pick, transit Z >= 80 mm, place).
5. **Graveyard**: Captured piece moved to off-board slot before attacker is placed.

---

## Documentation

- [Product Requirements Document](docs/PRD.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Hardware & State Contract (ArUco ID Map)](docs/STATE_CONTRACT.md)
- [Task Graph (DAG)](docs/KANBAN_TASKS.md)

---

## Requirements

**Hardware (for Arena/real mode only)**

- Universal Robots arm (UR3 / UR5, CB3 or e-Series) connected via Ethernet
- Overhead RGB camera (USB or RTSP)

**Software**

- Python 3.10+
- 9Router running at `http://127.0.0.1:20128/v1` (required for AI modes)
- Dependencies:

```
pip install -r requirements.txt
```

`requirements.txt` installs: `opencv-python`, `python-chess`, `numpy`, `scipy`, `flask`, `openai`, `pyyaml`.

---

## Installation

```bash
git clone https://github.com/rifkyafriza/UR-Chess-LLM-Arena.git
cd UR-Chess-LLM-Arena
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**WSL2 note**: camera access requires a USB passthrough tool (e.g. `usbipd-win`). Web simulation works without a camera.

---

## Running

### Interactive menu

```bash
python3 main.py
```

Presents a numbered menu:

```
[1] Web Simulation Mode   (ArUco + LLM + SVG Board)
[2] Live Camera Stream    (Overhead cam + ArUco HUD)
[3] Vision Benchmark      (OpenCV on test photo)
[4] Terminal Chess (CLI)  (vs 9Router LLM)
[5] Run Full Test Suite   (pytest)
[6] Arena Mode            (Full FSM: human vs LLM + robot)
[0] Exit
```

### Non-interactive commands

**Web Simulation** — browser-based chess on port 5070; supports PvP and Player vs AI:

```bash
python3 main.py --mode sim
python3 main.py --mode sim --port 5070 --model ag/claude-sonnet-4-6
```

Browser opens automatically at `http://127.0.0.1:5070`.

**Live Stream** — dual-pane: overhead camera feed + digital SVG board on port 5050:

```bash
python3 main.py --mode live
python3 main.py --mode live --port 5050 --camera 0
```

**Arena Mode** — full FSM game loop (LLM strategy + robot actuation):

```bash
# Simulated robot (no hardware required)
python3 main.py --mode arena --robot-mode sim

# Real UR robot (requires Ethernet connection to robot)
python3 main.py --mode arena --robot-mode real

# Override LLM model
python3 main.py --mode arena --robot-mode sim --model ag/claude-opus-4-6-thinking
```

**Vision Benchmark** — run OpenCV ArUco detection on a test board photo:

```bash
python3 main.py --mode benchmark
```

**CLI Chess** — terminal chess game vs LLM:

```bash
python3 main.py --mode cli
python3 main.py --mode cli --model ag/claude-sonnet-4-6
```

**Automated Tests**:

```bash
python3 main.py --mode test
# or directly:
pytest -q
```

---

## Configuration

All runtime parameters are centralized in `config/arena_config.yaml`.

```yaml
robot:
  mode: sim                    # "sim" | "real"
  host: 192.168.1.100          # UR robot IP (real mode only)
  ports:
    dashboard: 29999           # UR Dashboard Server
    script: 30002              # URScript port
    realtime: 30003            # RTDE port
  use_ur_feature: false        # true = use Polyscope Feature Plane
  feature_plane_pose: [0.0, 0.0, 0.0, 3.14159, 0.0, 0.0]  # [X,Y,Z,Rx,Ry,Rz]

  board_layout:
    square_pitch_x: 0.055      # meters between file centres (a→h)
    square_pitch_y: 0.055      # meters between rank centres (1→8)
    pick_height_z: 0.010       # Z of piece top when gripping
    transit_clearance_z: 0.120 # min Z for transit (hard floor 0.080 m)

  safety_envelope:
    x_bounds: [-0.80, 0.80]   # meters
    y_bounds: [-0.80, 0.80]
    z_bounds: [-0.20, 0.90]

  speeds:
    move_velocity: 0.25        # m/s  (movel)
    move_acceleration: 0.50    # m/s²
    joint_velocity: 1.05       # rad/s  (movej)
    joint_acceleration: 1.40   # rad/s²

vision:
  camera_index: 0
  locator_mode: aruco          # "contour" | "aruco"
  aruco_dict: DICT_4X4_50
  resolution: [1280, 720]
  stream_port: 5050

strategy:
  api_base: http://127.0.0.1:20128/v1
  model: ag/claude-sonnet-4-6
  timeout: 30                  # seconds before LLM call fails
  fallback_engine: stockfish   # used when LLM call fails

simulation:
  host: 127.0.0.1
  port: 5070
```

### Key parameters

| Key | Description |
|-----|-------------|
| `robot.mode` | `sim` = mock UR driver (no hardware); `real` = live robot |
| `robot.host` | Robot IP address (real mode) |
| `robot.use_ur_feature` | Enable Polyscope Feature Plane transform |
| `robot.feature_plane_pose` | Board frame pose in UR base frame [X,Y,Z,Rx,Ry,Rz] |
| `robot.board_layout.square_pitch_x/y` | Physical square size (meters) |
| `robot.board_layout.transit_clearance_z` | Minimum Z during piece transit (safety) |
| `robot.safety_envelope` | Cartesian workspace bounds; arm stops if exceeded |
| `vision.camera_index` | OpenCV camera device index |
| `vision.locator_mode` | ArUco marker detection or contour-based detection |
| `strategy.model` | LLM model name via 9Router |
| `strategy.fallback_engine` | Engine used if LLM is unreachable |
| `simulation.port` | Web simulation server port (default 5070) |

---

## Polyscope Feature Plane Calibration

Required when `robot.use_ur_feature: true` (real robot, accurate board alignment).

```bash
# Sim mode — enter XYZ manually, no robot connection needed
python scripts/calibrate_feature.py --sim

# Real mode — reads TCP pose automatically from robot Dashboard Server
python scripts/calibrate_feature.py --host 192.168.1.100
```

**Workflow**:

1. Enable Freedrive on the UR teach pendant.
2. Touch the TCP to square **a1** (White's queen-rook corner) — press Enter.
3. Touch the TCP to square **h1** (opposite end of rank 1) — press Enter.
4. Touch the TCP to square **a8** (top of file a) — press Enter.
5. Script computes the board local frame and derives `[X, Y, Z, Rx, Ry, Rz]`.
6. Confirm to save — `feature_plane_pose` and `use_ur_feature: true` are written to `config/arena_config.yaml`.

Override config path:

```bash
python scripts/calibrate_feature.py --sim --config path/to/arena_config.yaml
```

---

## Project Structure

```
UR-Chess-LLM-Arena/
├── main.py                     # Unified launcher (all modes)
├── chess_game.py               # CLI chess engine
├── requirements.txt
├── config/
│   └── arena_config.yaml       # Centralized configuration
├── scripts/
│   ├── calibrate_feature.py    # Feature Plane calibration tool
│   └── benchmark_board.py      # OpenCV detection benchmark
├── src/
│   ├── config.py               # Typed config dataclasses + loader
│   ├── orchestrator/
│   │   └── master_fsm.py       # Arena FSM (8-state game loop)
│   ├── robot/
│   │   └── coordinates.py      # Cartesian/joint math helpers
│   ├── simulation/
│   │   └── web_app.py          # Flask web sim (PvP + Player vs AI)
│   └── vision/
│       └── live_stream.py      # Camera stream + ArUco HUD
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── STATE_CONTRACT.md
│   └── KANBAN_TASKS.md
└── tests/                      # pytest suite (854 tests)
```

---

## Platform Notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | Fully supported | Native OpenCV camera access |
| WSL2 | Supported (sim) | Camera requires USB passthrough (`usbipd-win`); web sim works natively |
| macOS | Supported | May need `brew install python-tk` for some OpenCV features |
| Windows | Supported | Use `python` instead of `python3`; camera index may differ |
