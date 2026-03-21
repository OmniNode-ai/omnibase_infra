# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerRuntimeErrorTriage (OMN-5650)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage import (
    RuntimeErrorTriageDispatcher,
)


def _make_event(
    error_category: str = "SCHEMA_MISMATCH",
    fingerprint: str = "abc123def456",
    container: str = "omninode-runtime-effects",
    severity: str = "HIGH",
    error_message: str = 'column "pattern_name" does not exist',
    missing_topic_name: str | None = None,
) -> dict[str, object]:
    """Build a minimal runtime error event dict for testing."""
    event: dict[str, object] = {
        "event_id": "12345678-1234-5678-1234-567812345678",
        "container": container,
        "source_service": container,
        "logger_family": "omnibase_infra.event_bus",
        "log_level": "ERROR",
        "error_category": error_category,
        "severity": severity,
        "error_message": error_message,
        "fingerprint": fingerprint,
        "detected_at": "2026-03-21T13:22:42+00:00",
        "first_seen_at": "2026-03-21T13:22:42+00:00",
        "environment": "local",
        "raw_line": f"[ERROR] {error_message}",
    }
    if missing_topic_name:
        event["missing_topic_name"] = missing_topic_name
    return event


@pytest.mark.unit
class TestHandlerRuntimeErrorTriage:
    """Test triage handler logic."""

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_schema_mismatch_creates_ticket(self, mock_valkey_init: MagicMock) -> None:
        """SCHEMA_MISMATCH errors create Linear tickets (when Linear is not configured, action_status=FAILED)."""
        handler = RuntimeErrorTriageDispatcher()
        handler._valkey = None  # No Valkey

        event = _make_event(error_category="SCHEMA_MISMATCH")
        result = handler.triage_event(event)

        assert result.action == "TICKET_CREATED"
        assert result.error_category == "SCHEMA_MISMATCH"
        assert result.container == "omninode-runtime-effects"
        assert result.fingerprint == "abc123def456"

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_duplicate_error_deduped(self, mock_valkey_init: MagicMock) -> None:
        """Same fingerprint within 24h is deduped at action layer."""
        handler = RuntimeErrorTriageDispatcher()
        mock_valkey = MagicMock()
        mock_valkey.get.return_value = "TICKET_CREATED"
        handler._valkey = mock_valkey

        event = _make_event(fingerprint="already_seen_fp")
        result = handler.triage_event(event)

        assert result.action == "DEDUPED"
        assert result.action_status == "SUCCESS"
        assert "already actioned" in (result.dedup_reason or "")

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_unknown_category_requires_operator_attention(
        self, mock_valkey_init: MagicMock
    ) -> None:
        """UNKNOWN errors set operator_attention_required=True."""
        handler = RuntimeErrorTriageDispatcher()
        handler._valkey = None

        event = _make_event(error_category="UNKNOWN", severity="MEDIUM")
        result = handler.triage_event(event)

        assert result.action == "TICKET_CREATED"
        assert result.operator_attention_required is True

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_missing_topic_auto_fix_attempted(
        self, mock_valkey_init: MagicMock
    ) -> None:
        """MISSING_TOPIC errors in local env attempt rpk auto-fix."""
        handler = RuntimeErrorTriageDispatcher()
        handler._valkey = None
        handler._environment = "local"

        # Mock rpk to be available and succeed
        mock_result = {
            "command": "rpk topic create test.topic -p 6",
            "output": "TOPIC  STATUS\ntest.topic  OK",
            "verified": True,
        }
        with patch.object(
            handler, "_attempt_missing_topic_fix", return_value=mock_result
        ):
            event = _make_event(
                error_category="MISSING_TOPIC",
                error_message="Required topic 'onex.evt.test.v1' not in broker",
                missing_topic_name="onex.evt.test.v1",
            )
            result = handler.triage_event(event)

        assert result.action == "AUTO_FIXED"
        assert result.auto_fix_type == "rpk_topic_create"
        assert result.auto_fix_verified is True

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_missing_topic_non_local_creates_ticket(
        self, mock_valkey_init: MagicMock
    ) -> None:
        """MISSING_TOPIC in non-local env creates ticket instead of auto-fixing."""
        handler = RuntimeErrorTriageDispatcher()
        handler._valkey = None
        handler._environment = "production"

        event = _make_event(
            error_category="MISSING_TOPIC",
            missing_topic_name="onex.evt.test.v1",
        )
        result = handler.triage_event(event)

        assert result.action == "TICKET_CREATED"
        assert result.auto_fix_type is None

    @patch(
        "omnibase_infra.nodes.node_runtime_error_triage_effect.handler_triage.RuntimeErrorTriageDispatcher._init_valkey"
    )
    def test_connection_error_creates_ticket(self, mock_valkey_init: MagicMock) -> None:
        """CONNECTION errors create tickets (no auto-fix in v1)."""
        handler = RuntimeErrorTriageDispatcher()
        handler._valkey = None

        event = _make_event(
            error_category="CONNECTION",
            error_message="ConnectionRefusedError: [Errno 61]",
        )
        result = handler.triage_event(event)

        assert result.action == "TICKET_CREATED"
