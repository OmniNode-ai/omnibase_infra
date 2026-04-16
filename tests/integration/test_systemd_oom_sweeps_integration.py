# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for SystemdHealthSweep and OOMKillSweep (OMN-8872).

Verifies that:
1. SystemdHealthSweep can be instantiated and basic methods work
2. OOMKillSweep can be instantiated and basic methods work
3. Both threads can be started and stopped cleanly

Does NOT require running containers or systemd units - just tests the happy path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

import pytest

# Import the classes from scripts/monitor_logs.py
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
MONITOR_LOGS_PATH = SCRIPTS_DIR / "monitor_logs.py"

# Load the module
spec = importlib.util.spec_from_file_location("monitor_logs", MONITOR_LOGS_PATH)
if spec is None or spec.loader is None:
    pytest.skip("Could not load monitor_logs.py", allow_module_level=True)

monitor_logs = importlib.util.module_from_spec(spec)
sys.modules["monitor_logs"] = monitor_logs
spec.loader.exec_module(monitor_logs)

SystemdHealthSweep = monitor_logs.SystemdHealthSweep
OOMKillSweep = monitor_logs.OOMKillSweep

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def mock_slack_credentials():
    """Provide mock Slack credentials for testing."""
    return {
        "bot_token": "xoxb-test-token",
        "channel_id": "C1234567890",
    }


@pytest.fixture
def stop_event():
    """Provide a threading.Event for stopping threads."""
    return threading.Event()


def test_systemd_health_sweep_instantiation(mock_slack_credentials, stop_event):
    """Test that SystemdHealthSweep can be instantiated."""
    sweep = SystemdHealthSweep(
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        dry_run=True,
        stop_event=stop_event,
        interval=60,
    )
    assert sweep is not None
    assert sweep.bot_token == mock_slack_credentials["bot_token"]
    assert sweep.channel_id == mock_slack_credentials["channel_id"]
    assert sweep.dry_run is True
    assert sweep.interval == 60


def test_oom_kill_sweep_instantiation(mock_slack_credentials, stop_event):
    """Test that OOMKillSweep can be instantiated."""
    containers = ["test-container-1", "test-container-2"]
    sweep = OOMKillSweep(
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        dry_run=True,
        stop_event=stop_event,
        containers=containers,
        interval=60,
    )
    assert sweep is not None
    assert sweep.bot_token == mock_slack_credentials["bot_token"]
    assert sweep.channel_id == mock_slack_credentials["channel_id"]
    assert sweep.dry_run is True
    assert sweep.containers == containers
    assert sweep.interval == 60
    assert sweep._oom_seen == set()


def test_systemd_health_sweep_lifecycle(mock_slack_credentials, stop_event):
    """Test that SystemdHealthSweep can be started and stopped cleanly."""
    sweep = SystemdHealthSweep(
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        dry_run=True,
        stop_event=stop_event,
        interval=1,  # Short interval for testing
    )

    # Start the thread
    sweep.start()
    assert sweep.is_alive()

    # Stop it quickly
    stop_event.set()
    sweep.join(timeout=2.0)
    assert not sweep.is_alive()


def test_oom_kill_sweep_lifecycle(mock_slack_credentials, stop_event):
    """Test that OOMKillSweep can be started and stopped cleanly."""
    containers = ["test-container"]
    sweep = OOMKillSweep(
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        dry_run=True,
        stop_event=stop_event,
        containers=containers,
        interval=1,  # Short interval for testing
    )

    # Start the thread
    sweep.start()
    assert sweep.is_alive()

    # Stop it quickly
    stop_event.set()
    sweep.join(timeout=2.0)
    assert not sweep.is_alive()


def test_oom_kill_sweep_inspect_oom_nonexistent_container(
    mock_slack_credentials, stop_event, monkeypatch
):
    """Test that _inspect_oom returns False when docker inspect fails."""
    import subprocess
    from unittest.mock import MagicMock

    sweep = OOMKillSweep(
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        dry_run=True,
        stop_event=stop_event,
        containers=["nonexistent-container-xyz-12345"],
        interval=60,
    )

    # Mock subprocess.run to simulate docker inspect failure
    mock_run = MagicMock(return_value=MagicMock(returncode=1, stdout=""))
    monkeypatch.setattr(subprocess, "run", mock_run)

    # Should return False without crashing
    result = sweep._inspect_oom("nonexistent-container-xyz-12345")
    assert result is False
