# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for FileTailer and JournalTailer (OMN-8871).

Verifies that:
1. FileTailer can be instantiated and basic methods work
2. JournalTailer can be instantiated and basic methods work
3. Both threads can be started and stopped cleanly

Does NOT require running journalctl or actual log files - just tests the happy path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import time
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

FileTailer = monitor_logs.FileTailer
JournalTailer = monitor_logs.JournalTailer

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


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for FileTailer testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        f.write("Initial log line\n")
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        Path(temp_path).unlink()
    except OSError:
        pass


def test_file_tailer_instantiation(mock_slack_credentials, stop_event, temp_log_file):
    """Test that FileTailer can be instantiated."""
    tailer = FileTailer(
        path=temp_log_file,
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )
    assert tailer is not None
    assert tailer.path == temp_log_file
    assert tailer.bot_token == mock_slack_credentials["bot_token"]
    assert tailer.channel_id == mock_slack_credentials["channel_id"]
    assert tailer.cooldown == 60
    assert tailer.dry_run is True


def test_journal_tailer_instantiation(mock_slack_credentials, stop_event):
    """Test that JournalTailer can be instantiated."""
    tailer = JournalTailer(
        unit="test-unit.service",
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )
    assert tailer is not None
    assert tailer.unit == "test-unit.service"
    assert tailer.bot_token == mock_slack_credentials["bot_token"]
    assert tailer.channel_id == mock_slack_credentials["channel_id"]
    assert tailer.cooldown == 60
    assert tailer.dry_run is True


def test_file_tailer_lifecycle(mock_slack_credentials, stop_event, temp_log_file):
    """Test that FileTailer can be started and stopped cleanly."""
    tailer = FileTailer(
        path=temp_log_file,
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )

    # Start the thread
    tailer.start()
    assert tailer.is_alive()

    # Give it a moment to start tailing
    time.sleep(0.1)

    # Stop it
    stop_event.set()
    tailer.join(timeout=2.0)
    assert not tailer.is_alive()


def test_journal_tailer_lifecycle_mock(mock_slack_credentials, stop_event, monkeypatch):
    """Test that JournalTailer can be started and stopped cleanly with mocked subprocess."""
    import subprocess
    from unittest.mock import MagicMock

    # Mock Popen to avoid actual journalctl call
    mock_popen = MagicMock()
    mock_popen.stdout = iter([])  # Empty iterator
    mock_popen.terminate = MagicMock()

    def mock_popen_constructor(*args, **kwargs):
        return mock_popen

    monkeypatch.setattr(subprocess, "Popen", mock_popen_constructor)

    tailer = JournalTailer(
        unit="test-unit.service",
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )

    # Start the thread
    tailer.start()
    assert tailer.is_alive()

    # Give it a moment to start
    time.sleep(0.1)

    # Stop it
    stop_event.set()
    tailer.join(timeout=2.0)
    assert not tailer.is_alive()

    # Verify terminate was called
    mock_popen.terminate.assert_called_once()


def test_file_tailer_cooldown_key(mock_slack_credentials, stop_event, temp_log_file):
    """Test that FileTailer sets the correct cooldown key."""
    tailer = FileTailer(
        path=temp_log_file,
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )
    expected_key = f"file:{Path(temp_log_file).name}"
    assert tailer._cooldown_key == expected_key


def test_journal_tailer_cooldown_key(mock_slack_credentials, stop_event):
    """Test that JournalTailer sets the correct cooldown key."""
    unit_name = "test-unit.service"
    tailer = JournalTailer(
        unit=unit_name,
        bot_token=mock_slack_credentials["bot_token"],
        channel_id=mock_slack_credentials["channel_id"],
        cooldown=60,
        dry_run=True,
        stop_event=stop_event,
    )
    expected_key = f"journal:{unit_name}"
    assert tailer._cooldown_key == expected_key
