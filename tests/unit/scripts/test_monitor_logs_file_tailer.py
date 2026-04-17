# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for FileTailer and JournalTailer (OMN-8871)."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

_MODULE_NAME = "monitor_logs"


def _import() -> Any:
    if _MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[_MODULE_NAME])
    return importlib.import_module(_MODULE_NAME)


# ---------------------------------------------------------------------------
# FileTailer — emits Slack alert on ERROR pattern
# ---------------------------------------------------------------------------


class TestFileTailerEmitsAlertOnErrorPattern:
    """FileTailer must post a Slack alert when an ERROR line appears."""

    @pytest.mark.unit
    def test_file_tailer_emits_alert_on_error_pattern(self, tmp_path: Path) -> None:
        """Appending an ERROR line to a tailed file triggers post_slack."""
        m = _import()
        log_file = tmp_path / "env-sync.log"
        log_file.write_text("")

        stop_event = threading.Event()
        alerted: list[tuple[str, list[str]]] = []

        def fake_post_slack(
            bot_token: str,
            channel_id: str,
            source: str,
            lines: list[str],
            dry_run: bool,
        ) -> None:
            alerted.append((source, lines))

        tailer = m.FileTailer(
            path=str(log_file),
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=0,
            dry_run=False,
            stop_event=stop_event,
        )
        tailer.daemon = True

        with (
            patch.object(m, "post_slack", side_effect=fake_post_slack),
            patch.object(m, "_cooldown_read", return_value=(0.0, 0)),
            patch.object(m, "_cooldown_write"),
        ):
            tailer.start()
            time.sleep(0.2)
            log_file.write_text("ERROR: something broke\n")
            time.sleep(1.5)
            stop_event.set()
            tailer.join(timeout=3)

        assert len(alerted) >= 1, "Expected at least one Slack alert"
        source, lines = alerted[0]
        assert "env-sync.log" in source or str(log_file) in source
        assert any("ERROR" in line for line in lines)

    @pytest.mark.unit
    def test_file_tailer_ignores_non_error_lines(self, tmp_path: Path) -> None:
        """INFO lines must not trigger alerts."""
        m = _import()
        log_file = tmp_path / "hooks.log"
        log_file.write_text("")

        stop_event = threading.Event()
        alerted: list[Any] = []

        tailer = m.FileTailer(
            path=str(log_file),
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=0,
            dry_run=False,
            stop_event=stop_event,
        )
        tailer.daemon = True

        with (
            patch.object(
                m, "post_slack", side_effect=lambda *a, **kw: alerted.append(a)
            ),
            patch.object(m, "_cooldown_read", return_value=(0.0, 0)),
            patch.object(m, "_cooldown_write"),
        ):
            tailer.start()
            time.sleep(0.2)
            log_file.write_text("INFO: all good\nDEBUG: nothing to see\n")
            time.sleep(1.5)
            stop_event.set()
            tailer.join(timeout=3)

        assert len(alerted) == 0, "INFO/DEBUG lines must not trigger alerts"


# ---------------------------------------------------------------------------
# FileTailer — dedup window suppresses repeat alerts
# ---------------------------------------------------------------------------


class TestFileTailerDedupWindow:
    """Repeat errors within the cooldown window must not re-alert."""

    @pytest.mark.unit
    def test_file_tailer_dedup_window(self, tmp_path: Path) -> None:
        """Second identical error within cooldown window is suppressed."""
        m = _import()
        log_file = tmp_path / "pipeline-trace.log"
        log_file.write_text("")

        stop_event = threading.Event()
        alerted: list[Any] = []

        # Use a key prefix that matches what FileTailer uses
        cooldown_key = f"file:{log_file.name}"

        def fake_cooldown_read(key: str) -> tuple[float, int]:
            # Simulate already-alerted once with count=1
            if key == cooldown_key:
                return time.time() - 10, 1  # alerted 10s ago, backoff=300s
            return 0.0, 0

        def fake_cooldown_write(key: str, ts: float, count: int) -> None:
            pass

        tailer = m.FileTailer(
            path=str(log_file),
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=False,
            stop_event=stop_event,
        )
        tailer.daemon = True

        with (
            patch.object(
                m, "post_slack", side_effect=lambda *a, **kw: alerted.append(a)
            ),
            patch.object(m, "_cooldown_read", side_effect=fake_cooldown_read),
            patch.object(m, "_cooldown_write", side_effect=fake_cooldown_write),
        ):
            tailer.start()
            time.sleep(0.2)
            log_file.write_text("ERROR: repeated failure\n")
            time.sleep(1.5)
            stop_event.set()
            tailer.join(timeout=3)

        assert len(alerted) == 0, "Repeat error within cooldown must be suppressed"


# ---------------------------------------------------------------------------
# JournalTailer — uses journalctl --follow
# ---------------------------------------------------------------------------


class TestJournalTailerUsesJournalctlFollow:
    """JournalTailer must invoke journalctl with -f / --follow."""

    @pytest.mark.unit
    def test_journal_tailer_uses_journalctl_follow(self) -> None:
        """JournalTailer must spawn journalctl with follow flag."""
        m = _import()

        stop_event = threading.Event()
        captured_cmd: list[list[str]] = []

        fake_proc = MagicMock()
        fake_proc.stdout = iter([])  # empty — tailer exits immediately
        fake_proc.terminate = MagicMock()

        def fake_popen(cmd: list[str], **kwargs: Any) -> MagicMock:
            captured_cmd.append(cmd)
            stop_event.set()  # stop immediately after spawn
            return fake_proc

        with patch("subprocess.Popen", side_effect=fake_popen):
            tailer = m.JournalTailer(
                unit="deploy-agent.service",
                bot_token="xoxb-test",
                channel_id="C123",
                cooldown=0,
                dry_run=False,
                stop_event=stop_event,
            )
            tailer.daemon = True
            tailer.start()
            tailer.join(timeout=3)

        assert len(captured_cmd) >= 1, "JournalTailer must spawn a subprocess"
        cmd = captured_cmd[0]
        assert "journalctl" in cmd[0] or any("journalctl" in part for part in cmd)
        assert any(part in ("-f", "--follow") for part in cmd)
        assert any("deploy-agent.service" in part for part in cmd)

    @pytest.mark.unit
    def test_journal_tailer_emits_alert_on_error_line(self) -> None:
        """JournalTailer must alert when journalctl outputs an ERROR line."""
        m = _import()

        stop_event = threading.Event()
        alerted: list[Any] = []

        error_output = "ERROR: rebuild failed — connection refused\n"
        fake_proc = MagicMock()
        fake_proc.stdout = iter([error_output])
        fake_proc.terminate = MagicMock()

        with (
            patch("subprocess.Popen", return_value=fake_proc),
            patch.object(
                m, "post_slack", side_effect=lambda *a, **kw: alerted.append(a)
            ),
            patch.object(m, "_cooldown_read", return_value=(0.0, 0)),
            patch.object(m, "_cooldown_write"),
        ):
            tailer = m.JournalTailer(
                unit="deploy-agent.service",
                bot_token="xoxb-test",
                channel_id="C123",
                cooldown=0,
                dry_run=False,
                stop_event=stop_event,
            )
            tailer.daemon = True
            tailer.start()
            tailer.join(timeout=3)

        assert len(alerted) >= 1, "ERROR in journal output must trigger Slack alert"


# ---------------------------------------------------------------------------
# LogMonitor integration — file + journal tailers started on run()
# ---------------------------------------------------------------------------


class TestLogMonitorStartsFileTailers:
    """LogMonitor.run() must start FileTailer threads for MONITOR_FILE_LOGS."""

    @pytest.mark.unit
    def test_log_monitor_starts_file_tailers_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LogMonitor reads MONITOR_FILE_LOGS and starts one FileTailer per path."""
        m = _import()

        log_a = tmp_path / "env-sync.log"
        log_b = tmp_path / "hooks.log"
        log_a.write_text("")
        log_b.write_text("")

        monkeypatch.setenv("MONITOR_FILE_LOGS", f"{log_a}:{log_b}")
        monkeypatch.setenv("MONITOR_JOURNALS", "")

        monitor = m.LogMonitor(
            projects=[],
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=True,
        )

        started: list[str] = []

        def capture_start(self: Any) -> None:
            started.append(self.path)

        fake_proc = MagicMock()
        fake_proc.stdout = iter([])
        fake_proc.terminate = MagicMock()

        with (
            patch.object(m.FileTailer, "start", capture_start),
            patch.object(m.LogMonitor, "_get_project_containers", return_value=[]),
            patch.object(m, "_discover_postgres_containers", return_value=[]),
            patch.object(m.LogMonitor, "_journal_units", return_value=[]),
            patch("subprocess.Popen", return_value=fake_proc),
        ):
            t = threading.Thread(target=monitor.run, daemon=True)
            t.start()
            time.sleep(0.3)

        assert str(log_a) in started
        assert str(log_b) in started
