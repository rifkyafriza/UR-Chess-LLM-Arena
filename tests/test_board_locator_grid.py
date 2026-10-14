"""Self-check for board_locator_grid.py against assets/reference_board.jpg."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from src.vision.board_locator_grid import (  # noqa: E402
    localize_board,
    pixel_to_square,
    cell_to_algebraic,
)

IMG = ROOT / "assets" / "reference_board.jpg"
assert IMG.exists(), f"Missing {IMG}"

frame = cv2.imread(str(IMG))
assert frame is not None, "cv2.imread failed"

result = localize_board(frame, verify_grid=True)
assert result is not None, "localize_board returned None"
assert result["warped"].shape == (800, 800, 3), f"Bad warp shape: {result['warped'].shape}"
assert result["corners"].shape == (4, 2), "Bad corners shape"
assert result["matrix"].shape == (3, 3), "Bad matrix shape"
assert result["verified"], "Grid alternating pattern not detected"

# cell_to_algebraic
assert cell_to_algebraic(0, 7) == "a1"
assert cell_to_algebraic(7, 0) == "h8"
assert cell_to_algebraic(0, 0) == "a8"
assert cell_to_algebraic(7, 7) == "h1"

# pixel_to_square: center of cell (0,0) in 800px warped -> a8
sq, col, row = pixel_to_square(50, 50)
assert sq == "a8", f"Expected a8 got {sq}"
sq, col, row = pixel_to_square(750, 750)
assert sq == "h1", f"Expected h1 got {sq}"

print("All assertions passed.")
print(f"  Corners: {result['corners'].tolist()}")
print(f"  Warped shape: {result['warped'].shape}")
print(f"  Grid verified: {result['verified']}")
