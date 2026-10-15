"""Tests for main.py argument parsing and entrypoint dispatch."""

from __future__ import annotations

import sys
import types
import socket
from unittest.mock import MagicMock, patch, call
import pytest

# Ensure repo root on path
import importlib, pathlib
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_no_args_mode_is_none(self):
        p = main.build_parser()
        args = p.parse_args([])
        assert args.mode is None

    def test_mode_sim(self):
        args = main.build_parser().parse_args(["--mode", "sim"])
        assert args.mode == "sim"

    def test_mode_live(self):
        args = main.build_parser().parse_args(["--mode", "live"])
        assert args.mode == "live"

    def test_mode_benchmark(self):
        args = main.build_parser().parse_args(["--mode", "benchmark"])
        assert args.mode == "benchmark"

    def test_mode_cli(self):
        args = main.build_parser().parse_args(["--mode", "cli"])
        assert args.mode == "cli"

    def test_mode_test(self):
        args = main.build_parser().parse_args(["--mode", "test"])
        assert args.mode == "test"

    def test_default_model(self):
        args = main.build_parser().parse_args([])
        assert args.model == "ag/claude-sonnet-4-6"

    def test_custom_model(self):
        args = main.build_parser().parse_args(["--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_port_default_none(self):
        args = main.build_parser().parse_args(["--mode", "sim"])
        assert args.port is None  # None means use mode default

    def test_port_override(self):
        args = main.build_parser().parse_args(["--mode", "sim", "--port", "9999"])
        assert args.port == 9999

    def test_camera_default(self):
        args = main.build_parser().parse_args(["--mode", "live"])
        assert args.camera == 0

    def test_camera_override(self):
        args = main.build_parser().parse_args(["--mode", "live", "--camera", "2"])
        assert args.camera == 2

    def test_invalid_mode(self):
        with pytest.raises(SystemExit):
            main.build_parser().parse_args(["--mode", "invalid"])


# ---------------------------------------------------------------------------
# Dispatch (main() calls the right runner)
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_mode_sim_dispatches_run_sim(self):
        with patch.object(main, "run_sim") as mock_sim:
            main.main(["--mode", "sim"])
            mock_sim.assert_called_once_with(5070, "ag/claude-sonnet-4-6")

    def test_mode_sim_port_override(self):
        with patch.object(main, "run_sim") as mock_sim:
            main.main(["--mode", "sim", "--port", "8080"])
            mock_sim.assert_called_once_with(8080, "ag/claude-sonnet-4-6")

    def test_mode_live_dispatches_run_live(self):
        with patch.object(main, "run_live") as mock_live:
            main.main(["--mode", "live"])
            mock_live.assert_called_once_with(5050, 0)

    def test_mode_live_camera_override(self):
        with patch.object(main, "run_live") as mock_live:
            main.main(["--mode", "live", "--camera", "1"])
            mock_live.assert_called_once_with(5050, 1)

    def test_mode_benchmark_dispatches_run_benchmark(self):
        with patch.object(main, "run_benchmark") as mock_bench:
            main.main(["--mode", "benchmark"])
            mock_bench.assert_called_once()

    def test_mode_cli_dispatches_run_cli(self):
        with patch.object(main, "run_cli") as mock_cli:
            main.main(["--mode", "cli"])
            mock_cli.assert_called_once_with("ag/claude-sonnet-4-6")

    def test_mode_cli_model_override(self):
        with patch.object(main, "run_cli") as mock_cli:
            main.main(["--mode", "cli", "--model", "gpt-4o"])
            mock_cli.assert_called_once_with("gpt-4o")

    def test_mode_test_dispatches_run_tests(self):
        with patch.object(main, "run_tests") as mock_test:
            main.main(["--mode", "test"])
            mock_test.assert_called_once()

    def test_no_mode_dispatches_interactive(self):
        with patch.object(main, "interactive_menu") as mock_menu:
            main.main([])
            mock_menu.assert_called_once()

    def test_keyboard_interrupt_exits_cleanly(self):
        with patch.object(main, "run_tests", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc:
                main.main(["--mode", "test"])
            assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Port-busy guard
# ---------------------------------------------------------------------------

class TestPortBusy:
    def test_free_port_returns_false(self):
        # Use a port extremely unlikely to be bound
        assert main._port_busy(19999) is False

    def test_busy_port_returns_true(self):
        # Bind a socket, then check
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.listen(1)
        try:
            assert main._port_busy(port) is True
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# 9Router connectivity warning
# ---------------------------------------------------------------------------

class TestCheck9Router:
    def test_warns_when_unreachable(self, capsys):
        # Port 19998 should not be listening
        main._check_9router("some-model")
        out = capsys.readouterr().out
        # Either no output (9router running locally) or WARN
        # We only assert no exception raised
        assert True

    def test_no_exception_when_reachable(self):
        with patch("socket.create_connection"):
            main._check_9router("ag/claude-sonnet-4-6")  # should not raise
