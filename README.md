# UR Chess LLM Arena 🤖♟️

An autonomous cyber-physical system where a Universal Robots (UR) collaborative robotic arm plays physical chess against human opponents in real time.

Strategic decisions are powered by Large Language Models (LLMs) via a local AI gateway (9Router), the chessboard state is perceived by an overhead global camera tracking custom ArUco-tagged pieces (`cv2.aruco`), and physical piece manipulation is executed via Ethernet IP / URScript / RTDE.

---

## 🎯 Architecture & Workflow

1. **Vision Perception**: Global overhead camera captures the 8x8 chessboard. Four corner ArUco markers establish homography perspective warp. 32 ArUco piece markers (IDs 1-32) determine exact cell occupancy and serialize board state into FEN.
2. **Turn Settle Filter**: Vision pipeline ensures human hands have cleared the board and state is stable for ≥ 1.5 seconds.
3. **AI Strategy**: The FEN state is dispatched to local LLMs (`ag/claude-sonnet-4-6` / `ag/claude-opus-4-6-thinking`) to select the best legal UCI move (`e2e4`, `g1f3`).
4. **Robot Actuation**: Python orchestrator converts UCI move into 3D Cartesian waypoints for the UR arm over Ethernet (Port 30002/30003), executing pick, high-clearance transit ($Z \ge 80\text{mm}$), and place routines.
5. **Graveyard Routine**: When capturing an opponent's piece, the victim is first transported to an off-board graveyard slot before the attacker is placed.

---

## 📚 Technical Documentation

Complete engineering specifications are available in the [`docs/`](docs/) directory:
- [Product Requirements Document (PRD)](docs/PRD.md)
- [System Architecture Specification](docs/ARCHITECTURE.md)
- [Hardware & State Contract (ArUco ID Map)](docs/STATE_CONTRACT.md)
- [Kanban Task Graph (DAG)](docs/KANBAN_TASKS.md)

---

## 🛠️ Requirements

- Python 3.10+
- `pip install -r requirements.txt` (OpenCV, python-chess, numpy, scipy)
- Universal Robots arm (UR3 / UR5 / e-Series CB3/e-Series) connected via Ethernet
- Overhead RGB Camera (USB / RTSP)
- Running 9Router local endpoint (`http://localhost:20128/v1`)
