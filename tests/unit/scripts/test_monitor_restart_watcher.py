# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for RestartWatcher in monitor_logs.py (OMN-3596)."""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# monitor_logs.py lives in scripts/ and has no package install; add scripts/ to path.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_MODULE_NAME = "monitor_logs"


def _import() -> Any:
    if _MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[_MODULE_NAME])
    return importlib.import_module(_MODULE_NAME)


# ---------------------------------------------------------------------------
# HWM persistence
# ---------------------------------------------------------------------------


class TestRestartHwmRead:
    """Tests for _restart_hwm_read()."""

    @pytest.mark.unit
    def test_returns_zero_when_file_missing(self, tmp_path: Path) -> None:
        """HWM read returns 0 for an unknown container."""
        m = _import()
        hwm_file = tmp_path / "hwm.json"
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            assert m._restart_hwm_read("some-container") == 0

    @pytest.mark.unit
    def test_roundtrip_persistence(self, tmp_path: Path) -> None:
        """Write then read should return the written value."""
        m = _import()
        hwm_file = tmp_path / "hwm.json"
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            m._restart_hwm_write("ctr-a", 5)
            assert m._restart_hwm_read("ctr-a") == 5

    @pytest.mark.unit
    def test_write_creates_parent_directory(self, tmp_path: Path) -> None:
        """HWM write should create parent dirs if missing."""
        m = _import()
        hwm_file = tmp_path / "nested" / "dir" / "hwm.json"
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            m._restart_hwm_write("ctr-b", 3)
            assert hwm_file.exists()
            assert m._restart_hwm_read("ctr-b") == 3

    @pytest.mark.unit
    def test_returns_zero_when_corrupted(self, tmp_path: Path) -> None:
        """HWM read returns 0 when file is corrupted JSON."""
        m = _import()
        hwm_file = tmp_path / "hwm.json"
        hwm_file.write_text("NOT VALID JSON{{{")
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            assert m._restart_hwm_read("ctr-c") == 0

    @pytest.mark.unit
    def test_write_repairs_corrupted_file(self, tmp_path: Path) -> None:
        """HWM write should reset and repair a corrupted file."""
        m = _import()
        hwm_file = tmp_path / "hwm.json"
        hwm_file.write_text("NOT VALID JSON{{{")
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            m._restart_hwm_write("ctr-d", 7)
            assert m._restart_hwm_read("ctr-d") == 7

    @pytest.mark.unit
    def test_multiple_containers_independent(self, tmp_path: Path) -> None:
        """Each container gets its own independent HWM."""
        m = _import()
        hwm_file = tmp_path / "hwm.json"
        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            m._restart_hwm_write("ctr-x", 10)
            m._restart_hwm_write("ctr-y", 20)
            assert m._restart_hwm_read("ctr-x") == 10
            assert m._restart_hwm_read("ctr-y") == 20


# ---------------------------------------------------------------------------
# _get_worker_state
# ---------------------------------------------------------------------------


class TestGetWorkerState:
    """Tests for _get_worker_state()."""

    @pytest.mark.unit
    def test_returns_restart_count_and_status(self) -> None:
        """Should parse docker inspect output correctly."""
        m = _import()
        inspect_json = json.dumps(
            [
                {
                    "RestartCount": 5,
                    "State": {
                        "Status": "running",
                        "ExitCode": 0,
                    },
                }
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=inspect_json, stderr=""
            )
            result = m._get_worker_state("test-container")
            assert result is not None
            assert result["restart_count"] == 5
            assert result["status"] == "running"
            assert result["exit_code"] == 0

    @pytest.mark.unit
    def test_returns_none_on_docker_failure(self) -> None:
        """Should return None if docker inspect fails."""
        m = _import()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = m._get_worker_state("test-container")
            assert result is None

    @pytest.mark.unit
    def test_returns_none_on_empty_output(self) -> None:
        """Should return None if docker inspect returns empty list."""
        m = _import()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
            result = m._get_worker_state("test-container")
            assert result is None


# ---------------------------------------------------------------------------
# _restart_containers_from_env
# ---------------------------------------------------------------------------


class TestRestartContainersFromEnv:
    """Tests for _restart_containers_from_env()."""

    @pytest.mark.unit
    def test_returns_none_when_env_not_set(self) -> None:
        """Should return None when MONITOR_RESTART_CONTAINERS is not set."""
        m = _import()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MONITOR_RESTART_CONTAINERS", None)
            result = m._restart_containers_from_env()
            assert result is None

    @pytest.mark.unit
    def test_returns_sorted_list_from_env(self) -> None:
        """Should parse comma-separated env var into sorted list."""
        m = _import()
        with patch.dict(
            os.environ, {"MONITOR_RESTART_CONTAINERS": "ctr-b,ctr-a,ctr-c"}
        ):
            result = m._restart_containers_from_env()
            assert result == ["ctr-a", "ctr-b", "ctr-c"]

    @pytest.mark.unit
    def test_strips_whitespace(self) -> None:
        """Should strip whitespace from container names."""
        m = _import()
        with patch.dict(os.environ, {"MONITOR_RESTART_CONTAINERS": " ctr-a , ctr-b "}):
            result = m._restart_containers_from_env()
            assert result == ["ctr-a", "ctr-b"]


# ---------------------------------------------------------------------------
# RestartWatcher._check() logic
# ---------------------------------------------------------------------------


class TestRestartWatcherCheck:
    """Tests for RestartWatcher._check() alert/no-alert logic."""

    @pytest.mark.unit
    def test_alert_fires_when_delta_gte_threshold(self, tmp_path: Path) -> None:
        """Alert should fire when restart delta >= threshold."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"
        alerts: list[dict[str, Any]] = []

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            # Patch _alert to capture calls
            watcher._alert = lambda ctr, state: alerts.append(  # type: ignore[assignment]
                {"container": ctr, "state": state}
            )
            # Simulate: HWM=0, restart_count=3 → delta=3 >= threshold=3
            with patch.object(
                m,
                "_get_worker_state",
                return_value={
                    "restart_count": 3,
                    "status": "running",
                    "exit_code": 0,
                },
            ):
                watcher._check()

        assert len(alerts) == 1
        assert alerts[0]["container"] == "test-ctr"

    @pytest.mark.unit
    def test_no_alert_below_threshold(self, tmp_path: Path) -> None:
        """No alert should fire when restart delta < threshold."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"
        alerts: list[dict[str, Any]] = []

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            watcher._alert = lambda ctr, state: alerts.append(  # type: ignore[assignment]
                {"container": ctr, "state": state}
            )
            # Simulate: HWM=0, restart_count=2 → delta=2 < threshold=3
            with patch.object(
                m,
                "_get_worker_state",
                return_value={
                    "restart_count": 2,
                    "status": "running",
                    "exit_code": 0,
                },
            ):
                watcher._check()

        assert len(alerts) == 0

    @pytest.mark.unit
    def test_dry_run_does_not_post_slack(self, tmp_path: Path) -> None:
        """Dry-run mode should not call post_slack."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=True,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            with patch.object(
                m,
                "_get_worker_state",
                return_value={
                    "restart_count": 5,
                    "status": "running",
                    "exit_code": 1,
                },
            ):
                with patch.object(m, "post_slack") as mock_post:
                    watcher._check()
                    mock_post.assert_not_called()

    @pytest.mark.unit
    def test_negative_delta_resets_hwm(self, tmp_path: Path) -> None:
        """Container recreated (restart_count resets) should update HWM without alerting."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"
        alerts: list[dict[str, Any]] = []

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            # Set initial HWM to 10
            m._restart_hwm_write("test-ctr", 10)

            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            watcher._alert = lambda ctr, state: alerts.append(  # type: ignore[assignment]
                {"container": ctr, "state": state}
            )
            # Simulate: HWM=10, restart_count=1 → delta=-9 (container recreated)
            with patch.object(
                m,
                "_get_worker_state",
                return_value={
                    "restart_count": 1,
                    "status": "running",
                    "exit_code": 0,
                },
            ):
                watcher._check()

            # No alert for negative delta
            assert len(alerts) == 0
            # HWM should be updated to new value
            assert m._restart_hwm_read("test-ctr") == 1

    @pytest.mark.unit
    def test_hwm_updated_after_alert(self, tmp_path: Path) -> None:
        """HWM should be updated to current restart_count after alert fires."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            watcher._alert = lambda ctr, state: None  # type: ignore[assignment]
            with patch.object(
                m,
                "_get_worker_state",
                return_value={
                    "restart_count": 5,
                    "status": "running",
                    "exit_code": 0,
                },
            ):
                watcher._check()

            assert m._restart_hwm_read("test-ctr") == 5

    @pytest.mark.unit
    def test_skips_container_when_inspect_fails(self, tmp_path: Path) -> None:
        """Should skip container when docker inspect fails (returns None)."""
        m = _import()
        import threading

        hwm_file = tmp_path / "hwm.json"
        alerts: list[dict[str, Any]] = []

        with patch.object(m, "_RESTART_HWM_FILE", hwm_file):
            watcher = m.RestartWatcher(
                containers=["test-ctr"],
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=threading.Event(),
                interval=60,
                threshold=3,
            )
            watcher._alert = lambda ctr, state: alerts.append(  # type: ignore[assignment]
                {"container": ctr, "state": state}
            )
            with patch.object(m, "_get_worker_state", return_value=None):
                watcher._check()

        assert len(alerts) == 0
