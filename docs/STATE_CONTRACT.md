# Hardware & State Contract Specification

## Project: UR Chess LLM Arena
**Version**: 1.0.0  
**Status**: Active Contract  

---

## 1. ArUco Marker Dictionary & ID Allocations

- **ArUco Dictionary**: `cv2.aruco.DICT_4X4_50` (Higher contrast and lower false-positive rate for 8x8 layout).

### Corner Board Calibration Markers
| Marker ID | Placement Location | Function |
|---|---|---|
| `40` | Outer Top-Left (near a8) | Board boundary homography corner TL |
| `41` | Outer Top-Right (near h8) | Board boundary homography corner TR |
| `42` | Outer Bottom-Right (near h1) | Board boundary homography corner BR |
| `43` | Outer Bottom-Left (near a1) | Board boundary homography corner BL |

### White Pieces (ID 1 - 16)
| ArUco ID | Standard Piece Type | Initial Square |
|---|---|---|
| `1` | White Rook 1 | a1 |
| `2` | White Knight 1 | b1 |
| `3` | White Bishop 1 | c1 |
| `4` | White Queen | d1 |
| `5` | White King | e1 |
| `6` | White Bishop 2 | f1 |
| `7` | White Knight 2 | g1 |
| `8` | White Rook 2 | h1 |
| `9` | White Pawn 1 | a2 |
| `10` | White Pawn 2 | b2 |
| `11` | White Pawn 3 | c2 |
| `12` | White Pawn 4 | d2 |
| `13` | White Pawn 5 | e2 |
| `14` | White Pawn 6 | f2 |
| `15` | White Pawn 7 | g2 |
| `16` | White Pawn 8 | h2 |

### Black Pieces (ID 17 - 32)
| ArUco ID | Standard Piece Type | Initial Square |
|---|---|---|
| `17` | Black Rook 1 | a8 |
| `18` | Black Knight 1 | b8 |
| `19` | Black Bishop 1 | c8 |
| `20` | Black Queen | d8 |
| `21` | Black King | e8 |
| `22` | Black Bishop 2 | f8 |
| `23` | Black Knight 2 | g8 |
| `24` | Black Rook 2 | h8 |
| `25` | Black Pawn 1 | a7 |
| `26` | Black Pawn 2 | b7 |
| `27` | Black Pawn 3 | c7 |
| `28` | Black Pawn 4 | d7 |
| `29` | Black Pawn 5 | e7 |
| `30` | Black Pawn 6 | f7 |
| `31` | Black Pawn 7 | g7 |
| `32` | Black Pawn 8 | h7 |

---

## 2. Board Grid & Coordinate Mapping

### Algebraic to Grid Index Mapping
- Files: `a` to `h` -> column index `0` to `7`.
- Ranks: `1` to `8` -> row index `0` to `7`.

### Mathematical Transformation from Board Square to UR Base Pose
Given board cell indices $(col, row)$ where $col \in [0, 7]$ and $row \in [0, 7]$:
$$
P_{\text{board}}(col, row) = P_{\text{origin}} + col \cdot \vec{V}_{\text{file}} + row \cdot \vec{V}_{\text{rank}}
$$
Where:
- $P_{\text{origin}} = (X_0, Y_0, Z_{\text{grip}}, R_x, R_y, R_z)$ represents square `a1`.
- $\Delta_{\text{square}} \approx 0.050\text{ m}$ (50mm square pitch).
- $R_x, R_y, R_z$ represent the fixed tool orientation (usually vertical pointing down: $\approx [0, \pi, 0]$ or equivalent axis-angle).

### Graveyard Bin Coordinates (Captured Pieces)
Captured pieces are sequentially indexed and placed in dedicated side racks:
- **White Graveyard Slot $k \in [0, 15]$**: Coordinates placed along the left board edge ($X_{\text{grave\_white}}, Y_k, Z_{\text{grip}}$).
- **Black Graveyard Slot $k \in [0, 15]$**: Coordinates placed along the right board edge ($X_{\text{grave\_black}}, Y_k, Z_{\text{grip}}$).

---

## 3. Communication Payloads

### Vision Output Payload Schema (JSON)
```json
{
  "timestamp": 1728900000.123,
  "is_stable": true,
  "human_detected": false,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "board_matrix": [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null],
    [null, null, null, null, "P", null, null, null],
    [null, null, null, null, null, null, null, null],
    ["P", "P", "P", "P", null, "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
  ],
  "detected_move": "e2e4"
}
```

### URScript Robot Command Snippet
```python
# Linear pick and place sequence example via port 30002/30003
def ur_move_piece(src_pose, dst_pose, hover_z=0.09):
    script = f"""
    def pick_and_place():
        set_tcp(p[0, 0, 0.15, 0, 0, 0])
        # Move to source hover
        movel(p[{src_pose[0]}, {src_pose[1]}, {src_pose[2] + hover_z}, {src_pose[3]}, {src_pose[4]}, {src_pose[5]}], a=0.5, v=0.2)
        # Descend to grip
        movel(p[{src_pose[0]}, {src_pose[1]}, {src_pose[2]}, {src_pose[3]}, {src_pose[4]}, {src_pose[5]}], a=0.3, v=0.08)
        # Activate Gripper (Digital Out 0)
        set_digital_out(0, True)
        sleep(0.3)
        # Lift to hover
        movel(p[{src_pose[0]}, {src_pose[1]}, {src_pose[2] + hover_z}, {src_pose[3]}, {src_pose[4]}, {src_pose[5]}], a=0.5, v=0.2)
        # Move to destination hover
        movel(p[{dst_pose[0]}, {dst_pose[1]}, {dst_pose[2] + hover_z}, {dst_pose[3]}, {dst_pose[4]}, {dst_pose[5]}], a=0.5, v=0.2)
        # Descend to place
        movel(p[{dst_pose[0]}, {dst_pose[1]}, {dst_pose[2]}, {dst_pose[3]}, {dst_pose[4]}, {dst_pose[5]}], a=0.3, v=0.08)
        # Deactivate Gripper
        set_digital_out(0, False)
        sleep(0.3)
        # Lift to hover
        movel(p[{dst_pose[0]}, {dst_pose[1]}, {dst_pose[2] + hover_z}, {dst_pose[3]}, {dst_pose[4]}, {dst_pose[5]}], a=0.5, v=0.2)
    end
    pick_and_place()
    """
    return script
```
