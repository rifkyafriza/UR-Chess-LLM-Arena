"""Interactive arena config editor.

Usage::
    from src.config_editor import interactive_config_editor
    interactive_config_editor()                         # default config path
    interactive_config_editor("path/to/arena_config.yaml")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from src.config import (
    ArenaConfig,
    _DEFAULT_CONFIG,
    _validate,
    load_config,
)

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _show(cfg: ArenaConfig) -> None:
    """Print current config in human-readable form."""
    r, v, st, sim = cfg.robot, cfg.vision, cfg.strategy, cfg.simulation
    print("\n=== Current Arena Config ===")
    print(f"  ROBOT")
    print(f"    mode              : {r.mode}")
    print(f"    host              : {r.host}")
    print(f"    move_velocity     : {r.speeds.move_velocity} m/s")
    print(f"    move_acceleration : {r.speeds.move_acceleration} m/s²")
    print(f"    joint_velocity    : {r.speeds.joint_velocity} rad/s")
    print(f"    joint_acceleration: {r.speeds.joint_acceleration} rad/s²")
    print(f"    ports             : dashboard={r.ports.dashboard} script={r.ports.script} realtime={r.ports.realtime}")
    print(f"  VISION")
    print(f"    camera_index      : {v.camera_index}")
    print(f"    locator_mode      : {v.locator_mode}")
    print(f"    stream_port       : {v.stream_port}")
    print(f"  STRATEGY")
    print(f"    model             : {st.model}")
    print(f"    timeout           : {st.timeout} s")
    print(f"    api_base          : {st.api_base}")
    print(f"  SIMULATION")
    print(f"    host              : {sim.host}")
    print(f"    port              : {sim.port}")
    print()


# ---------------------------------------------------------------------------
# Per-section edit helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, current, cast=str):  # type: ignore[assignment]
    """Prompt for a value; return current if user presses Enter."""
    while True:
        raw = input(f"  {label} [{current}]: ").strip()  # EOF/Ctrl-C propagate to caller
        if not raw:
            return current
        try:
            return cast(raw)
        except ValueError:
            print(f"[WARN] Invalid value {raw!r}; expected {cast.__name__}. Try again.")


def _edit_robot(cfg: ArenaConfig) -> None:
    r = cfg.robot
    print("\n-- Robot Settings --")
    try:
        mode = _prompt("mode (sim/real)", r.mode)
        if mode not in ("sim", "real"):
            print(f"[WARN] Invalid mode '{mode}', keeping '{r.mode}'")
            mode = r.mode
        r.mode = mode

        r.host = _prompt("host IP", r.host)

        mv = _prompt("move_velocity m/s", r.speeds.move_velocity, float)
        ma = _prompt("move_acceleration m/s²", r.speeds.move_acceleration, float)
        r.speeds.move_velocity = mv
        r.speeds.move_acceleration = ma
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Robot edit cancelled.")


def _edit_vision(cfg: ArenaConfig) -> None:
    v = cfg.vision
    print("\n-- Vision Settings --")
    try:
        ci = _prompt("camera_index", v.camera_index, int)
        lm = _prompt("locator_mode (aruco/contour)", v.locator_mode)
        if lm not in ("aruco", "contour"):
            print(f"[WARN] Invalid locator_mode '{lm}', keeping '{v.locator_mode}'")
            lm = v.locator_mode
        sp = _prompt("stream_port", v.stream_port, int)
        v.camera_index = ci
        v.locator_mode = lm
        v.stream_port = sp
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Vision edit cancelled.")


def _edit_strategy(cfg: ArenaConfig) -> None:
    st = cfg.strategy
    print("\n-- Strategy Settings --")
    try:
        st.model = _prompt("model", st.model)
        st.timeout = _prompt("timeout (s)", st.timeout, int)
        st.api_base = _prompt("api_base", st.api_base)
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Strategy edit cancelled.")


def _edit_simulation(cfg: ArenaConfig) -> None:
    sim = cfg.simulation
    print("\n-- Simulation Settings --")
    try:
        sim.host = _prompt("host", sim.host)
        sim.port = _prompt("port", sim.port, int)
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Simulation edit cancelled.")


# ---------------------------------------------------------------------------
# Atomic YAML save
# ---------------------------------------------------------------------------

def _tuples_to_lists(obj):
    """Recursively convert tuples to lists so yaml.safe_dump works."""
    if isinstance(obj, tuple):
        return [_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tuples_to_lists(v) for v in obj]
    return obj


def _save(cfg: ArenaConfig, config_path: Path) -> None:
    """Validate then atomically write config to disk."""
    if not _YAML_AVAILABLE or _yaml is None:
        print("[ERROR] PyYAML not installed. Cannot save. Run: pip install pyyaml")
        return

    # Validate first — never write an invalid config.
    try:
        _validate(cfg)
    except ValueError as exc:
        print(f"[ERROR] Validation failed: {exc}")
        print("[INFO] Config NOT saved.")
        return

    yaml_str = _yaml.safe_dump(
        _tuples_to_lists(cfg.to_dict()),
        default_flow_style=False,
        sort_keys=False,
    )

    # Atomic write: write to sibling temp, then rename.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=config_path.parent, prefix=".arena_config_tmp_", suffix=".yaml"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, config_path)
        print(f"[OK] Config saved to {config_path}")
    except Exception as exc:
        # Clean up temp on failure; original file is untouched.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        print(f"[ERROR] Save failed: {exc}")


# ---------------------------------------------------------------------------
# Submenu
# ---------------------------------------------------------------------------

_SUBMENU = """
+------------------------------------------------------+
|         Arena Config Editor                          |
+------------------------------------------------------+
|  [1] View current config                             |
|  [2] Edit Robot Settings                             |
|  [3] Edit Vision Settings                            |
|  [4] Edit Strategy Settings                          |
|  [5] Edit Simulation Settings                        |
|  [0] Save & Exit                                     |
|  [q] Cancel (discard changes)                        |
+------------------------------------------------------+
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interactive_config_editor(config_path: Optional[str] = None) -> None:
    """Interactive config editor. Edits in-memory; validates and saves on [0]."""
    resolved = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg = load_config(str(resolved) if resolved.exists() else None)

    print(_SUBMENU)
    while True:
        try:
            choice = input("Config editor> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Cancelled. No changes saved.")
            return

        if choice == "1":
            _show(cfg)
        elif choice == "2":
            _edit_robot(cfg)
        elif choice == "3":
            _edit_vision(cfg)
        elif choice == "4":
            _edit_strategy(cfg)
        elif choice == "5":
            _edit_simulation(cfg)
        elif choice == "0":
            _save(cfg, resolved)
            return
        elif choice == "q":
            print("[INFO] Cancelled. No changes saved.")
            return
        else:
            print(f"[ERROR] Unknown option '{choice}'. Valid: 0-5 or q.")

        print(_SUBMENU)
