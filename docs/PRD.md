# Product Requirements Document (PRD)

## Project: UR Chess LLM Arena
**Status**: Draft / Active  
**Author**: Rifky Afriza & Autonomous Engineering Team  
**Version**: 1.0.0  
**Repository**: `rifkyafriza/chess-llm-arena`

---

## 1. Executive Summary & Objective

The **UR Chess LLM Arena** is an autonomous cyber-physical system where a Universal Robots (UR) collaborative robotic arm plays physical chess against a human opponent in real time. 

Strategic moves are computed by Large Language Models (LLMs) hosted via local gateway (9Router) using state-of-the-art reasoning models (`ag/claude-sonnet-4-6`, `ag/claude-opus-4-6-thinking`). The game board state is captured through a global overhead camera detecting custom chess pieces with ArUco markers on their top surfaces. Physical actuation is executed via Ethernet IP / URScript / RTDE protocols.

---

## 2. Target Personas & Use Cases

1. **Academic Demonstration & Research**: Educational tool at Politeknik Negeri Batam (D4 TRR) demonstrating the fusion of Computer Vision, Robotics Kinematics, Industrial Communication, and Generative AI reasoning.
2. **Human vs Machine Exhibition**: Interactive booth exhibition where human players challenge LLM personas with physical piece movement.
3. **LLM Strategy Benchmarking**: Physical benchmark testing LLM spatial-temporal reasoning, rule compliance, and tactical decision-making in tangible environments.

---

## 3. Success Metrics

| Metric | Target | Verification Method |
|---|---|---|
| ArUco Board Detection Accuracy | ≥ 99.5% stable detection | 100 consecutive frames without marker ID jitter |
| FEN State Parsing Latency | < 500 ms post-human hand clearance | Benchmark timer from motion settle to FEN generation |
| LLM Move Latency | < 3.0 s | Round-trip API latency via 9Router local endpoint |
| Pick-and-Place Actuation Time | < 6.0 s per standard move | Robot trajectory execution time (hover -> pick -> place -> park) |
| Collision Free Actuation | 100% zero unintentional collisions | Physical safety clearance envelope (Z-transit ≥ 80mm) |
| Illegal Move Recovery Rate | 100% (No game freeze) | Fallback to algorithmic legal move validator (`python-chess`) |

---

## 4. In-Scope vs Out-of-Scope (v1.0)

### In-Scope (Phase 1 & 2):
- Global overhead camera view (perpendicular 2D planar projection).
- 32 cylindrical pieces fitted with distinct ArUco markers (Dictionary: `DICT_4X4_50`).
- Homography transformation mapping 4 corner board ArUco markers to standard 8x8 algebraic chess grid (`a1` to `h8`).
- Game state serialization into FEN (Forsyth–Edwards Notation).
- Turn detection: Wait for human hand to leave camera field-of-view and board state to stabilize (settle timer ≥ 1.5s).
- Strategic decision-making via LLM with strict UCI move generation (`e2e4`, `g1f3`).
- Automatic move validation and fallback using `python-chess`.
- UR robot trajectory generation over Ethernet (URScript / RTDE).
- Two-finger / vacuum end-effector pick, transit, and place routine.
- Capture logic: Physical removal of captured piece to an off-board graveyard area before placing attacking piece.
- Castling handling (two-step movement: King first, Rook second).
- Web-based live telemetry preview (SVG board state, model thoughts, move history).

### Out-of-Scope (Deferred to Future Iterations):
- Human face / emotion recognition.
- Pawn promotion piece transformation (pawn will be considered queen logically without physical swap if duplicate piece unavailable).
- Clock tap physical actuation (chess clock timer will be digital inside UI).
- Real-time dynamic obstacle avoidance during mid-trajectory human interference (system relies on turn-based safety interlocking: human must not reach into robot workspace during robot move).

---

## 5. System Interlocking & Safety Requirements

1. **Workspace Safety Interlocking**: Robot only transitions from standby pose when the human turn is declared finished and the vision module confirms the workspace is clear of human hands.
2. **Kinematic Clearance Envelope**: All planar transit moves must occur at a safe clearance plane $Z_{\text{transit}} \ge 80\text{mm}$ above piece heights to prevent toppling adjacent standing pieces.
3. **Emergency Stop (E-Stop) Integration**: Software kill-switch via keyboard interrupt / UI button immediately terminates UR socket communication and triggers robot deceleration stop (`stopj(2.0)`).
4. **Physical Workspace Enclosure**: Boundaries restricted to chess board dimensions + graveyard bin coordinates. Commanded coordinates outside bounds are rejected before transmission.
