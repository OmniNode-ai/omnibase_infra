# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for WARNING_PATTERN alerting in monitor_logs.py (OMN-3607).

Tests cover:
- WARNING_PATTERN regex matching known warning log lines
- _warning_issue_label() returning correct human-readable labels
- _maybe_warning_alert() cooldown behavior (independent from error-level backoff)
- Integration into ContainerTailer.run() as elif after ERROR_PATTERN
"""

from __future__ import annotations

import importlib
import sys
import time
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
# WARNING_PATTERN regex matching
# ---------------------------------------------------------------------------


class TestWarningPatternMatches:
    """WARNING_PATTERN must match all four known warning patterns from log analysis."""

    @pytest.mark.unit
    def test_matches_stale_registration(self) -> None:
        """[WARNING] line with 'Heartbeat received for non-active node' must match."""
        m = _import()
        line = (
            "2026-03-04 12:34:56 [WARNING] handler_node_heartbeat: "
            "Heartbeat received for non-active node abc-123"
        )
        assert m.WARNING_PATTERN.search(line), f"Expected match for: {line}"

    @pytest.mark.unit
    def test_matches_kafka_nack(self) -> None:
        """[WARNING] line with 'dispatch_handlers:.*nacking message for retry' must match."""
        m = _import()
        line = (
            "2026-03-04 12:35:00 [WARNING] dispatch_handlers: "
            "nacking message for retry (attempt 3/5)"
        )
        assert m.WARNING_PATTERN.search(line), f"Expected match for: {line}"

    @pytest.mark.unit
    def test_matches_schema_deadlock(self) -> None:
        """[WARNING] line with 'DeadlockDetectedError' must match."""
        m = _import()
        line = (
            "2026-03-04 12:36:00 [WARNING] plugin_schema_init: "
            "DeadlockDetectedError during schema migration"
        )
        assert m.WARNING_PATTERN.search(line), f"Expected match for: {line}"

    @pytest.mark.unit
    def test_matches_consul_unavailable(self) -> None:
        """[WARNING] line with 'HandlerConsul.*ConnectionError' must match."""
        m = _import()
        line = (
            "2026-03-04 12:37:00 [WARNING] HandlerConsul: "
            "ConnectionError: [Errno 111] Connection refused"
        )
        assert m.WARNING_PATTERN.search(line), f"Expected match for: {line}"

    @pytest.mark.unit
    def test_no_match_on_info_level(self) -> None:
        """Same text at [INFO] level must NOT match (anchored to [WARNING])."""
        m = _import()
        line = (
            "2026-03-04 12:34:56 [INFO] handler_node_heartbeat: "
            "Heartbeat received for non-active node abc-123"
        )
        assert not m.WARNING_PATTERN.search(line), (
            f"Should NOT match INFO-level: {line}"
        )

    @pytest.mark.unit
    def test_no_match_on_error_level(self) -> None:
        """Same text at [ERROR] level must NOT match (anchored to [WARNING])."""
        m = _import()
        line = (
            "2026-03-04 12:34:56 [ERROR] handler_node_heartbeat: "
            "Heartbeat received for non-active node abc-123"
        )
        assert not m.WARNING_PATTERN.search(line), (
            f"Should NOT match ERROR-level: {line}"
        )

    @pytest.mark.unit
    def test_no_match_on_unrelated_warning(self) -> None:
        """A [WARNING] line that doesn't match any known pattern must NOT match."""
        m = _import()
        line = "2026-03-04 12:34:56 [WARNING] Some unrelated warning message"
        assert not m.WARNING_PATTERN.search(line), (
            f"Should NOT match unrelated warning: {line}"
        )


# ---------------------------------------------------------------------------
# _warning_issue_label() lookup
# ---------------------------------------------------------------------------


class TestWarningIssueLabel:
    """_warning_issue_label() must return the correct human-readable label."""

    @pytest.mark.unit
    def test_stale_registration_label(self) -> None:
        m = _import()
        line = (
            "[WARNING] handler_node_heartbeat: "
            "Heartbeat received for non-active node abc-123"
        )
        assert m._warning_issue_label(line) == "stale-registration"

    @pytest.mark.unit
    def test_kafka_nack_label(self) -> None:
        m = _import()
        line = "[WARNING] dispatch_handlers: nacking message for retry (attempt 3/5)"
        assert m._warning_issue_label(line) == "kafka-nack"

    @pytest.mark.unit
    def test_schema_deadlock_label(self) -> None:
        m = _import()
        line = "[WARNING] plugin_schema_init: DeadlockDetectedError during schema migration"
        assert m._warning_issue_label(line) == "schema-deadlock"

    @pytest.mark.unit
    def test_consul_unavailable_label(self) -> None:
        m = _import()
        line = (
            "[WARNING] HandlerConsul: ConnectionError: [Errno 111] Connection refused"
        )
        assert m._warning_issue_label(line) == "consul-unavailable"

    @pytest.mark.unit
    def test_unknown_returns_generic_label(self) -> None:
        """If no fragment matches, return a generic 'unknown-warning' label."""
        m = _import()
        line = "[WARNING] some totally unrecognized warning text"
        assert m._warning_issue_label(line) == "unknown-warning"


# ---------------------------------------------------------------------------
# WARNING_COOLDOWN_SECONDS defaults
# ---------------------------------------------------------------------------


class TestWarningCooldownDefault:
    """WARNING_COOLDOWN_SECONDS must default to 1800 and be overridable via env."""

    @pytest.mark.unit
    def test_default_cooldown_is_1800(self) -> None:
        m = _import()
        assert m.WARNING_COOLDOWN_SECONDS == 1800

    @pytest.mark.unit
    def test_cooldown_overridable_via_env(self) -> None:
        with patch.dict("os.environ", {"MONITOR_WARNING_COOLDOWN": "600"}):
            m = _import()
            assert m.WARNING_COOLDOWN_SECONDS == 600


# ---------------------------------------------------------------------------
# _maybe_warning_alert cooldown (independent from error-level)
# ---------------------------------------------------------------------------


class TestMaybeWarningAlert:
    """ContainerTailer._maybe_warning_alert uses independent cooldown from errors."""

    @pytest.mark.unit
    def test_first_warning_triggers_alert(self) -> None:
        """First warning for a container+label should trigger a Slack alert."""
        m = _import()

        stop_event = MagicMock()
        tailer = m.ContainerTailer(
            container="test-container",
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=True,
            stop_event=stop_event,
        )

        lines = ["[WARNING] Heartbeat received for non-active node abc-123"]

        # Clear any existing cooldown state for this key
        cooldown_key = "test-container:warn:stale-registration"
        with (
            patch.object(m, "_cooldown_read", return_value=(0.0, 0)) as mock_read,
            patch.object(m, "_cooldown_write") as mock_write,
            patch.object(m, "post_slack") as mock_slack,
        ):
            tailer._maybe_warning_alert("stale-registration", lines)
            mock_slack.assert_called_once()
            mock_write.assert_called_once()
            # Verify the cooldown key uses container:warn:label composite format
            call_args = mock_write.call_args[0]
            assert call_args[0] == cooldown_key

    @pytest.mark.unit
    def test_warning_within_cooldown_is_suppressed(self) -> None:
        """Warning within cooldown window should be rate-limited."""
        m = _import()

        stop_event = MagicMock()
        tailer = m.ContainerTailer(
            container="test-container",
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=True,
            stop_event=stop_event,
        )

        lines = ["[WARNING] Heartbeat received for non-active node abc-123"]

        # Simulate recent alert (within cooldown window)
        recent_ts = time.time() - 60  # 60s ago, well within 1800s cooldown
        with (
            patch.object(m, "_cooldown_read", return_value=(recent_ts, 1)),
            patch.object(m, "post_slack") as mock_slack,
        ):
            tailer._maybe_warning_alert("stale-registration", lines)
            mock_slack.assert_not_called()

    @pytest.mark.unit
    def test_warning_after_cooldown_triggers_alert(self) -> None:
        """Warning after cooldown expires should trigger an alert."""
        m = _import()

        stop_event = MagicMock()
        tailer = m.ContainerTailer(
            container="test-container",
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=True,
            stop_event=stop_event,
        )

        lines = ["[WARNING] DeadlockDetectedError during schema migration"]

        # Simulate old alert (well past cooldown)
        old_ts = time.time() - 7200  # 2 hours ago
        with (
            patch.object(m, "_cooldown_read", return_value=(old_ts, 1)),
            patch.object(m, "_cooldown_write") as mock_write,
            patch.object(m, "post_slack") as mock_slack,
        ):
            tailer._maybe_warning_alert("schema-deadlock", lines)
            mock_slack.assert_called_once()

    @pytest.mark.unit
    def test_warning_slack_message_includes_label(self) -> None:
        """The Slack alert for warnings must include the warning label."""
        m = _import()

        stop_event = MagicMock()
        tailer = m.ContainerTailer(
            container="test-container",
            bot_token="xoxb-test",
            channel_id="C123",
            cooldown=300,
            dry_run=True,
            stop_event=stop_event,
        )

        lines = ["[WARNING] HandlerConsul: ConnectionError: refused"]

        with (
            patch.object(m, "_cooldown_read", return_value=(0.0, 0)),
            patch.object(m, "_cooldown_write"),
            patch.object(m, "post_slack") as mock_slack,
        ):
            tailer._maybe_warning_alert("consul-unavailable", lines)
            call_args = mock_slack.call_args
            # The container argument passed to post_slack should include the label
            container_arg = (
                call_args[0][2]
                if len(call_args[0]) > 2
                else call_args[1].get("container", "")
            )
            assert "consul-unavailable" in container_arg
