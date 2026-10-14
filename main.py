"""Unified multi-mode launcher for UR-Chess-LLM-Arena.

Usage (interactive):  python3 main.py
Usage (non-interactive):
  python3 main.py --mode sim       [--port 5060] [--model ag/claude-sonnet-4-6]
  python3 main.py --mode live      [--port 5050] [--camera 0]
  python3 main.py --mode benchmark
  python3 main.py --mode cli       [--model ag/claude-sonnet-4-6]
  python3 main.py --mode test
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import webbrowser


# ---------------------------------------------------------------------------
# Port / connectivity helpers
# ---------------------------------------------------------------------------

def _port_busy(port: int) -> bool:
    """Return True if *port* is already bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _check_9router(model: str) -> None:
    """Warn (not abort) when 9Router is unreachable."""
    try:
        with socket.create_connection(("127.0.0.1", 20128), timeout=1):
            pass
    except OSError:
        print(
            f"[WARN] 9Router not reachable at 127.0.0.1:20128. "
            f"Model '{model}' calls will fail. Start 9router first."
        )


# ---------------------------------------------------------------------------
# Mode dispatchers
# ---------------------------------------------------------------------------

def run_sim(port: int, model: str) -> None:
    if _port_busy(port):
        sys.exit(f"[ERROR] Port {port} already in use. Choose another with --port.")
    _check_9router(model)
    os.environ.setdefault("CHESS_MODEL", model)
    # Open browser slightly after Flask starts
    import threading
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    from src.simulation.web_app import app  # type: ignore[import]
    print(f"[SIM] Starting web simulation on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


def run_live(port: int, camera: int) -> None:
    if _port_busy(port):
        sys.exit(f"[ERROR] Port {port} already in use. Choose another with --port.")
    from src.vision.live_stream import start  # type: ignore[import]
    print(f"[LIVE] Starting camera stream on http://127.0.0.1:{port} (camera {camera})")
    vision_app.run(host="127.0.0.1", port=port, debug=False)


def run_benchmark() -> None:
    print("[BENCHMARK] Running OpenCV detection on test board photo...")
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_board.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


def run_cli(model: str) -> None:
    _check_9router(model)
    # chess_game.py reads MODEL from argv or uses a default; inject via env
    os.environ.setdefault("CHESS_MODEL", model)
    result = subprocess.run(
        [sys.executable, "chess_game.py", "--model", model],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


def run_tests() -> None:
    print("[TEST] Running full automated test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

_MENU = """
╔══════════════════════════════════════════════════════╗
║         UR-Chess LLM Arena — Mode Selector           ║
╠══════════════════════════════════════════════════════╣
║  [1] Web Simulation Mode   (ArUco + LLM + SVG Board) ║
║  [2] Live Camera Stream    (Overhead cam + ArUco HUD) ║
║  [3] Vision Benchmark      (OpenCV on test photo)     ║
║  [4] Terminal Chess (CLI)  (vs 9Router LLM)           ║
║  [5] Run Full Test Suite   (pytest)                   ║
║  [0] Exit                                             ║
╚══════════════════════════════════════════════════════╝
"""

_DEFAULTS = {
    "sim_port": 5060,
    "live_port": 5050,
    "camera": 0,
    "model": "ag/claude-sonnet-4-6",
}


def interactive_menu() -> None:
    print(_MENU)
    try:
        choice = input("Select mode: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        sys.exit(0)

    if choice == "0":
        print("Bye.")
        sys.exit(0)
    elif choice == "1":
        run_sim(_DEFAULTS["sim_port"], _DEFAULTS["model"])
    elif choice == "2":
        run_live(_DEFAULTS["live_port"], _DEFAULTS["camera"])
    elif choice == "3":
        run_benchmark()
    elif choice == "4":
        run_cli(_DEFAULTS["model"])
    elif choice == "5":
        run_tests()
    else:
        print(f"[ERROR] Unknown choice '{choice}'. Valid: 0-5.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UR-Chess LLM Arena unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["sim", "live", "benchmark", "cli", "test"],
        help="Run mode (omit for interactive menu)",
    )
    p.add_argument("--port", type=int, default=None, help="Override default port")
    p.add_argument("--camera", type=int, default=0, help="Camera index (live mode, default 0)")
    p.add_argument(
        "--model",
        default="ag/claude-sonnet-4-6",
        help="LLM model name via 9Router (default ag/claude-sonnet-4-6)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode is None:
            interactive_menu()
        elif args.mode == "sim":
            run_sim(args.port or _DEFAULTS["sim_port"], args.model)
        elif args.mode == "live":
            run_live(args.port or _DEFAULTS["live_port"], args.camera)
        elif args.mode == "benchmark":
            run_benchmark()
        elif args.mode == "cli":
            run_cli(args.model)
        elif args.mode == "test":
            run_tests()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Bye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
