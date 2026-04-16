# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OMN-8872: SystemdHealthSweep + OOMKillSweep."""

from __future__ import annotations

import importlib
import sys
import threading
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


def _no_cooldown(key: str) -> tuple[float, int]:
    """Return clean cooldown state (never alerted before)."""
    return 0.0, 0


def _noop_write(*_args: Any, **_kwargs: Any) -> None:
    pass


# ---------------------------------------------------------------------------
# SystemdHealthSweep
# ---------------------------------------------------------------------------


class TestSystemdHealthSweep:
    @pytest.mark.unit
    def test_detects_failed_unit_and_alerts(self) -> None:
        """Failed systemd unit triggers Slack alert with unit name and journal tail."""
        m = _import()
        stop = threading.Event()

        systemctl_output = (
            "deploy-agent.service loaded failed failed OmniNode Deploy Agent"
        )
        journal_output = (
            "Apr 15 10:00:00 host deploy-agent[123]: fatal: connection refused"
        )

        posted: list[tuple[str, list[str]]] = []

        def fake_post_slack(bot_token, channel_id, container, lines, dry_run):
            posted.append((container, lines))

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if "list-units" in cmd:
                result.stdout = systemctl_output
            elif "journalctl" in cmd:
                result.stdout = journal_output
            else:
                result.stdout = ""
            return result

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=_no_cooldown),
            patch.object(m, "_cooldown_write", side_effect=_noop_write),
            patch("monitor_logs.subprocess") as mock_sub,
        ):
            mock_sub.run.side_effect = fake_run
            sweep = m.SystemdHealthSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
            )
            sweep._check()

        assert len(posted) >= 1
        container_label, lines = posted[0]
        assert "deploy-agent.service" in container_label
        combined = "\n".join(lines)
        assert "deploy-agent" in combined

    @pytest.mark.unit
    def test_silent_when_all_active(self) -> None:
        """No failed units → no Slack alert."""
        m = _import()
        stop = threading.Event()

        posted: list = []

        def fake_post_slack(*args, **kwargs):
            posted.append(args)

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            return result

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=_no_cooldown),
            patch.object(m, "_cooldown_write", side_effect=_noop_write),
            patch("monitor_logs.subprocess") as mock_sub,
        ):
            mock_sub.run.side_effect = fake_run
            sweep = m.SystemdHealthSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
            )
            sweep._check()

        assert posted == []

    @pytest.mark.unit
    def test_dedup_suppresses_repeat_alert(self) -> None:
        """Same failed unit should not alert twice within cooldown window."""
        m = _import()
        stop = threading.Event()

        systemctl_output = (
            "deploy-agent.service loaded failed failed OmniNode Deploy Agent"
        )

        posted: list = []

        def fake_post_slack(*args, **kwargs):
            posted.append(args)

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = systemctl_output if "list-units" in cmd else ""
            return result

        # Simulate: first _cooldown_read returns (0, 0) → alerts.
        # After alert, _cooldown_write stores ts=1000. Second _check sees wait=300
        # not yet elapsed. We simulate this via the _oom_seen pattern: use a real
        # in-memory cooldown store so state persists across calls within one test.
        cooldown_store: dict[str, tuple[float, int]] = {}

        def fake_read(key: str) -> tuple[float, int]:
            return cooldown_store.get(key, (0.0, 0))

        def fake_write(key: str, ts: float, count: int) -> None:
            cooldown_store[key] = (ts, count)

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=fake_read),
            patch.object(m, "_cooldown_write", side_effect=fake_write),
            patch("monitor_logs.subprocess") as mock_sub,
            patch("monitor_logs.time") as mock_time,
        ):
            mock_sub.run.side_effect = fake_run
            mock_time.time.return_value = 1000.0
            sweep = m.SystemdHealthSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
            )
            sweep._check()  # First: alerts, writes ts=1000
            mock_time.time.return_value = 1010.0  # Only 10s later — within 300s backoff
            sweep._check()  # Second: suppressed

        assert len(posted) == 1


# ---------------------------------------------------------------------------
# OOMKillSweep
# ---------------------------------------------------------------------------


class TestOOMKillSweep:
    @pytest.mark.unit
    def test_detects_oom_killed_container_and_alerts(self) -> None:
        """Container with OOMKilled=true triggers CRITICAL Slack alert."""
        m = _import()
        stop = threading.Event()

        posted: list[tuple[str, list[str]]] = []

        def fake_post_slack(bot_token, channel_id, container, lines, dry_run):
            posted.append((container, lines))

        containers = ["omninode-runtime", "omninode-runner-1"]

        import json

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                [{"State": {"OOMKilled": True, "Status": "exited"}}]
            )
            return result

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=_no_cooldown),
            patch.object(m, "_cooldown_write", side_effect=_noop_write),
            patch("monitor_logs.subprocess") as mock_sub,
        ):
            mock_sub.run.side_effect = fake_run
            sweep = m.OOMKillSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
                containers=containers,
            )
            sweep._check()

        assert len(posted) >= 1
        labels = [p[0] for p in posted]
        assert any("omninode-runtime" in lbl for lbl in labels)
        for _, lines in posted:
            combined = "\n".join(lines)
            assert (
                "OOMKilled" in combined
                or "oom" in combined.lower()
                or "CRITICAL" in combined
            )

    @pytest.mark.unit
    def test_silent_when_no_oom(self) -> None:
        """No OOMKilled containers → no alert."""
        m = _import()
        stop = threading.Event()

        posted: list = []

        def fake_post_slack(*args, **kwargs):
            posted.append(args)

        containers = ["omninode-runtime"]

        import json

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                [{"State": {"OOMKilled": False, "Status": "running"}}]
            )
            return result

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=_no_cooldown),
            patch.object(m, "_cooldown_write", side_effect=_noop_write),
            patch("monitor_logs.subprocess") as mock_sub,
        ):
            mock_sub.run.side_effect = fake_run
            sweep = m.OOMKillSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
                containers=containers,
            )
            sweep._check()

        assert posted == []

    @pytest.mark.unit
    def test_dedup_suppresses_repeat_oom_alert(self) -> None:
        """OOMKilled=true on same container should not re-alert until container restarts."""
        m = _import()
        stop = threading.Event()

        posted: list = []

        def fake_post_slack(*args, **kwargs):
            posted.append(args)

        containers = ["omninode-runtime"]

        import json

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                [{"State": {"OOMKilled": True, "Status": "exited"}}]
            )
            return result

        with (
            patch.object(m, "post_slack", fake_post_slack),
            patch.object(m, "_cooldown_read", side_effect=_no_cooldown),
            patch.object(m, "_cooldown_write", side_effect=_noop_write),
            patch("monitor_logs.subprocess") as mock_sub,
        ):
            mock_sub.run.side_effect = fake_run
            sweep = m.OOMKillSweep(
                bot_token="xoxb-test",
                channel_id="C123",
                dry_run=False,
                stop_event=stop,
                interval=999,
                containers=containers,
            )
            sweep._check()  # First: alerts + adds to _oom_seen
            sweep._check()  # Second: suppressed by _oom_seen

        # Only one alert — second call sees container in _oom_seen
        assert len(posted) == 1
