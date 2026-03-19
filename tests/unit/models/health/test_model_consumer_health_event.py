# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for consumer health event models and enums.

Ticket: OMN-5511
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)
from omnibase_infra.models.health.enum_consumer_incident_state import (
    EnumConsumerIncidentState,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
    _compute_fingerprint,
)
from omnibase_infra.models.health.model_consumer_restart_command import (
    ModelConsumerRestartCommand,
)


@pytest.mark.unit
class TestEnumConsumerHealthEventType:
    """Tests for EnumConsumerHealthEventType."""

    def test_all_values_exist(self) -> None:
        assert EnumConsumerHealthEventType.HEARTBEAT_FAILURE == "heartbeat_failure"
        assert EnumConsumerHealthEventType.SESSION_TIMEOUT == "session_timeout"
        assert EnumConsumerHealthEventType.REBALANCE_START == "rebalance_start"
        assert EnumConsumerHealthEventType.REBALANCE_COMPLETE == "rebalance_complete"
        assert EnumConsumerHealthEventType.CONSUMER_STOPPED == "consumer_stopped"
        assert EnumConsumerHealthEventType.CONSUMER_STARTED == "consumer_started"
        assert EnumConsumerHealthEventType.POLL_TIMEOUT == "poll_timeout"
        assert EnumConsumerHealthEventType.CONNECTION_LOST == "connection_lost"

    def test_is_str_enum(self) -> None:
        assert isinstance(EnumConsumerHealthEventType.HEARTBEAT_FAILURE, str)


@pytest.mark.unit
class TestEnumConsumerHealthSeverity:
    """Tests for EnumConsumerHealthSeverity."""

    def test_all_values_exist(self) -> None:
        assert EnumConsumerHealthSeverity.INFO == "info"
        assert EnumConsumerHealthSeverity.WARNING == "warning"
        assert EnumConsumerHealthSeverity.ERROR == "error"
        assert EnumConsumerHealthSeverity.CRITICAL == "critical"


@pytest.mark.unit
class TestEnumConsumerIncidentState:
    """Tests for EnumConsumerIncidentState."""

    def test_all_values_exist(self) -> None:
        assert EnumConsumerIncidentState.OPEN == "open"
        assert EnumConsumerIncidentState.ACKNOWLEDGED == "acknowledged"
        assert EnumConsumerIncidentState.RESTART_PENDING == "restart_pending"
        assert EnumConsumerIncidentState.RESTART_SUCCEEDED == "restart_succeeded"
        assert EnumConsumerIncidentState.RESTART_FAILED == "restart_failed"
        assert EnumConsumerIncidentState.TICKETED == "ticketed"
        assert EnumConsumerIncidentState.RESOLVED == "resolved"


@pytest.mark.unit
class TestComputeFingerprint:
    """Tests for _compute_fingerprint."""

    def test_deterministic(self) -> None:
        fp1 = _compute_fingerprint("consumer-1", "heartbeat_failure", "topic-a")
        fp2 = _compute_fingerprint("consumer-1", "heartbeat_failure", "topic-a")
        assert fp1 == fp2

    def test_different_inputs_different_fingerprints(self) -> None:
        fp1 = _compute_fingerprint("consumer-1", "heartbeat_failure", "topic-a")
        fp2 = _compute_fingerprint("consumer-2", "heartbeat_failure", "topic-a")
        assert fp1 != fp2

    def test_length(self) -> None:
        fp = _compute_fingerprint("c", "e", "t")
        assert len(fp) == 16


@pytest.mark.unit
class TestModelConsumerHealthEvent:
    """Tests for ModelConsumerHealthEvent."""

    def test_create_factory(self) -> None:
        event = ModelConsumerHealthEvent.create(
            consumer_identity="consumer-1",
            consumer_group="group-1",
            topic="test-topic",
            event_type=EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
            severity=EnumConsumerHealthSeverity.ERROR,
        )
        assert event.consumer_identity == "consumer-1"
        assert event.consumer_group == "group-1"
        assert event.topic == "test-topic"
        assert event.event_type == EnumConsumerHealthEventType.HEARTBEAT_FAILURE
        assert event.severity == EnumConsumerHealthSeverity.ERROR
        assert len(event.fingerprint) == 16
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.correlation_id, UUID)

    def test_create_with_rebalance_metrics(self) -> None:
        event = ModelConsumerHealthEvent.create(
            consumer_identity="consumer-1",
            consumer_group="group-1",
            topic="test-topic",
            event_type=EnumConsumerHealthEventType.REBALANCE_COMPLETE,
            severity=EnumConsumerHealthSeverity.WARNING,
            rebalance_duration_ms=1500,
            partitions_assigned=3,
            partitions_revoked=1,
        )
        assert event.rebalance_duration_ms == 1500
        assert event.partitions_assigned == 3
        assert event.partitions_revoked == 1

    def test_frozen_model(self) -> None:
        event = ModelConsumerHealthEvent.create(
            consumer_identity="c",
            consumer_group="g",
            topic="t",
            event_type=EnumConsumerHealthEventType.CONSUMER_STARTED,
            severity=EnumConsumerHealthSeverity.INFO,
        )
        with pytest.raises(Exception):
            event.consumer_identity = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        event = ModelConsumerHealthEvent.create(
            consumer_identity="consumer-1",
            consumer_group="group-1",
            topic="test-topic",
            event_type=EnumConsumerHealthEventType.SESSION_TIMEOUT,
            severity=EnumConsumerHealthSeverity.CRITICAL,
            error_message="session timed out",
            hostname="host-1",
            service_label="my-service",
        )
        data = event.model_dump(mode="json")
        restored = ModelConsumerHealthEvent.model_validate(data)
        assert restored.consumer_identity == event.consumer_identity
        assert restored.fingerprint == event.fingerprint
        assert restored.error_message == "session timed out"

    def test_fingerprint_stability(self) -> None:
        e1 = ModelConsumerHealthEvent.create(
            consumer_identity="c1",
            consumer_group="g1",
            topic="t1",
            event_type=EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
            severity=EnumConsumerHealthSeverity.ERROR,
        )
        e2 = ModelConsumerHealthEvent.create(
            consumer_identity="c1",
            consumer_group="g1",
            topic="t1",
            event_type=EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
            severity=EnumConsumerHealthSeverity.WARNING,  # different severity
        )
        # Same consumer_identity + event_type + topic = same fingerprint
        assert e1.fingerprint == e2.fingerprint


@pytest.mark.unit
class TestModelConsumerRestartCommand:
    """Tests for ModelConsumerRestartCommand."""

    def test_create(self) -> None:
        cmd = ModelConsumerRestartCommand(
            consumer_identity="consumer-1",
            consumer_group="group-1",
            topic="test-topic",
            reason="3rd heartbeat failure in 30 minutes",
            fingerprint="abc123",
        )
        assert cmd.consumer_identity == "consumer-1"
        assert cmd.intent_type == "consumer_health.restart"
        assert isinstance(cmd.command_id, UUID)

    def test_serialization_roundtrip(self) -> None:
        cmd = ModelConsumerRestartCommand(
            consumer_identity="c",
            consumer_group="g",
            topic="t",
            reason="test",
            fingerprint="fp",
        )
        data = cmd.model_dump(mode="json")
        restored = ModelConsumerRestartCommand.model_validate(data)
        assert restored.consumer_identity == cmd.consumer_identity
        assert restored.intent_type == "consumer_health.restart"
