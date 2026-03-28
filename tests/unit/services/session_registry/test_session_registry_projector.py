# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for session registry projector (OMN-6854).

Tests the pure extraction function that converts raw Kafka events
into ModelSessionRegistryEntry objects for upsert.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_SESSION_COORDINATION_SIGNAL,
    TOPIC_SESSION_STATUS_CHANGED,
)
from omnibase_infra.services.session_registry.enum_session_phase import EnumSessionPhase
from omnibase_infra.services.session_registry.enum_session_registry_status import (
    EnumSessionRegistryStatus,
)
from omnibase_infra.services.session_registry.projector import (
    CONSUMER_GROUP,
    extract_registry_update,
)


def _make_event(
    event_type: str = "prompt.submitted",
    task_id: str | None = "OMN-1234",
    session_id: str = "session-abc",
    correlation_id: str = "550e8400-e29b-41d4-a716-446655440000",
    emitted_at: str | None = "2026-03-28T12:00:00+00:00",
    **extra: object,
) -> dict[str, object]:
    """Build a minimal event dict for testing."""
    event: dict[str, object] = {
        "event_type": event_type,
        "session_id": session_id,
        "correlation_id": correlation_id,
    }
    if task_id is not None:
        event["task_id"] = task_id
    if emitted_at is not None:
        event["emitted_at"] = emitted_at
    event.update(extra)
    return event


@pytest.mark.unit
class TestExtractRegistryUpdate:
    """Test extract_registry_update() pure function."""

    def test_extracts_task_id(self) -> None:
        entry = extract_registry_update(_make_event(task_id="OMN-5678"))
        assert entry is not None
        assert entry.task_id == "OMN-5678"

    def test_extracts_session_id(self) -> None:
        entry = extract_registry_update(_make_event(session_id="sess-xyz"))
        assert entry is not None
        assert "sess-xyz" in entry.session_ids

    def test_extracts_correlation_id(self) -> None:
        cid = "550e8400-e29b-41d4-a716-446655440000"
        entry = extract_registry_update(_make_event(correlation_id=cid))
        assert entry is not None
        assert cid in entry.correlation_ids

    def test_skips_event_without_task_id(self) -> None:
        entry = extract_registry_update(_make_event(task_id=None))
        assert entry is None

    def test_skips_event_with_empty_task_id(self) -> None:
        event = _make_event()
        event["task_id"] = ""
        entry = extract_registry_update(event)
        assert entry is None

    def test_skips_event_with_non_string_task_id(self) -> None:
        event = _make_event()
        event["task_id"] = 12345
        entry = extract_registry_update(event)
        assert entry is None

    def test_handles_malformed_event_gracefully(self) -> None:
        """Malformed events return None, never raise."""
        entry = extract_registry_update({})
        assert entry is None

    def test_handles_non_dict_event(self) -> None:
        """Non-dict input is handled gracefully."""
        entry = extract_registry_update({"task_id": "OMN-1", "event_type": 42})  # type: ignore[dict-item]
        assert entry is not None  # Should still extract task_id

    def test_infers_phase_from_session_started(self) -> None:
        entry = extract_registry_update(_make_event(event_type="session.started"))
        assert entry is not None
        assert entry.current_phase == EnumSessionPhase.PLANNING

    def test_infers_phase_from_tool_executed(self) -> None:
        entry = extract_registry_update(_make_event(event_type="tool.executed"))
        assert entry is not None
        assert entry.current_phase == EnumSessionPhase.IMPLEMENTING

    def test_infers_phase_from_decision_recorded(self) -> None:
        entry = extract_registry_update(_make_event(event_type="decision.recorded"))
        assert entry is not None
        assert entry.current_phase == EnumSessionPhase.REVIEWING

    def test_infers_phase_from_session_ended(self) -> None:
        entry = extract_registry_update(_make_event(event_type="session.ended"))
        assert entry is not None
        assert entry.current_phase == EnumSessionPhase.COMPLETED

    def test_unknown_event_type_gives_none_phase(self) -> None:
        entry = extract_registry_update(_make_event(event_type="unknown.event"))
        assert entry is not None
        assert entry.current_phase is None

    def test_extracts_files_from_tool_executed_edit(self) -> None:
        entry = extract_registry_update(
            _make_event(
                event_type="tool.executed",
                tool_name="Edit",
                file_path="/src/foo.py",
            )
        )
        assert entry is not None
        assert "/src/foo.py" in entry.files_touched

    def test_extracts_files_from_tool_executed_write(self) -> None:
        entry = extract_registry_update(
            _make_event(
                event_type="tool.executed",
                tool_name="Write",
                file_path="/src/bar.py",
            )
        )
        assert entry is not None
        assert "/src/bar.py" in entry.files_touched

    def test_no_files_from_non_file_tool(self) -> None:
        entry = extract_registry_update(
            _make_event(
                event_type="tool.executed",
                tool_name="Grep",
                file_path="/src/baz.py",
            )
        )
        assert entry is not None
        assert entry.files_touched == []

    def test_no_files_from_prompt_event(self) -> None:
        entry = extract_registry_update(_make_event(event_type="prompt.submitted"))
        assert entry is not None
        assert entry.files_touched == []

    def test_status_is_always_active(self) -> None:
        """Projector always emits active status; store handles regression."""
        entry = extract_registry_update(_make_event())
        assert entry is not None
        assert entry.status == EnumSessionRegistryStatus.ACTIVE

    def test_parses_iso_emitted_at(self) -> None:
        entry = extract_registry_update(
            _make_event(emitted_at="2026-03-28T14:30:00+00:00")
        )
        assert entry is not None
        assert entry.last_activity is not None
        assert entry.last_activity.year == 2026
        assert entry.last_activity.month == 3

    def test_handles_missing_emitted_at(self) -> None:
        entry = extract_registry_update(_make_event(emitted_at=None))
        assert entry is not None
        assert entry.last_activity is not None  # Falls back to now()


@pytest.mark.unit
class TestProjectorConstants:
    """Test projector configuration constants."""

    def test_consumer_group_follows_convention(self) -> None:
        """Consumer group follows {service}.{node_name}.{purpose}.{version} convention."""
        assert CONSUMER_GROUP == "omnibase_infra.session_registry.project.v1"

    def test_coordination_topic_follows_onex_naming(self) -> None:
        assert TOPIC_SESSION_COORDINATION_SIGNAL.startswith("onex.evt.")
        assert ".v1" in TOPIC_SESSION_COORDINATION_SIGNAL

    def test_status_changed_topic_follows_onex_naming(self) -> None:
        assert TOPIC_SESSION_STATUS_CHANGED.startswith("onex.evt.")
        assert ".v1" in TOPIC_SESSION_STATUS_CHANGED
