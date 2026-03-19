# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runtime error event model.

Ticket: OMN-5513
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
    _compute_error_fingerprint,
)


@pytest.mark.unit
class TestEnumRuntimeErrorCategory:
    """Tests for EnumRuntimeErrorCategory."""

    def test_all_values_exist(self) -> None:
        assert EnumRuntimeErrorCategory.KAFKA_CONSUMER == "kafka_consumer"
        assert EnumRuntimeErrorCategory.KAFKA_PRODUCER == "kafka_producer"
        assert EnumRuntimeErrorCategory.DATABASE == "database"
        assert EnumRuntimeErrorCategory.HTTP_CLIENT == "http_client"
        assert EnumRuntimeErrorCategory.HTTP_SERVER == "http_server"
        assert EnumRuntimeErrorCategory.RUNTIME == "runtime"
        assert EnumRuntimeErrorCategory.UNKNOWN == "unknown"

    def test_is_str_enum(self) -> None:
        assert isinstance(EnumRuntimeErrorCategory.KAFKA_CONSUMER, str)


@pytest.mark.unit
class TestEnumRuntimeErrorSeverity:
    """Tests for EnumRuntimeErrorSeverity."""

    def test_all_values_exist(self) -> None:
        assert EnumRuntimeErrorSeverity.WARNING == "warning"
        assert EnumRuntimeErrorSeverity.ERROR == "error"
        assert EnumRuntimeErrorSeverity.CRITICAL == "critical"


@pytest.mark.unit
class TestComputeErrorFingerprint:
    """Tests for _compute_error_fingerprint."""

    def test_deterministic(self) -> None:
        fp1 = _compute_error_fingerprint(
            "aiokafka.consumer", "heartbeat failed", "kafka_consumer"
        )
        fp2 = _compute_error_fingerprint(
            "aiokafka.consumer", "heartbeat failed", "kafka_consumer"
        )
        assert fp1 == fp2

    def test_different_inputs_different_fingerprints(self) -> None:
        fp1 = _compute_error_fingerprint(
            "aiokafka.consumer", "heartbeat failed", "kafka_consumer"
        )
        fp2 = _compute_error_fingerprint("asyncpg", "connection refused", "database")
        assert fp1 != fp2

    def test_length(self) -> None:
        fp = _compute_error_fingerprint("a", "b", "c")
        assert len(fp) == 16


@pytest.mark.unit
class TestModelRuntimeErrorEvent:
    """Tests for ModelRuntimeErrorEvent."""

    def test_create_factory(self) -> None:
        event = ModelRuntimeErrorEvent.create(
            logger_family="aiokafka.consumer",
            log_level="ERROR",
            message_template="Heartbeat failed for group {}",
            raw_message="Heartbeat failed for group test-group",
            error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            severity=EnumRuntimeErrorSeverity.ERROR,
        )
        assert event.logger_family == "aiokafka.consumer"
        assert event.log_level == "ERROR"
        assert event.error_category == EnumRuntimeErrorCategory.KAFKA_CONSUMER
        assert len(event.fingerprint) == 16
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.correlation_id, UUID)
        assert event.occurrence_count_local == 1

    def test_create_with_exception_info(self) -> None:
        event = ModelRuntimeErrorEvent.create(
            logger_family="asyncpg",
            log_level="ERROR",
            message_template="connection refused to {}:{}",
            raw_message="connection refused to localhost:5432",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            severity=EnumRuntimeErrorSeverity.ERROR,
            exception_type="ConnectionRefusedError",
            exception_message="Connection refused",
            stack_trace="Traceback...",
            occurrence_count_local=5,
        )
        assert event.exception_type == "ConnectionRefusedError"
        assert event.exception_message == "Connection refused"
        assert event.stack_trace == "Traceback..."
        assert event.occurrence_count_local == 5

    def test_frozen_model(self) -> None:
        event = ModelRuntimeErrorEvent.create(
            logger_family="test",
            log_level="ERROR",
            message_template="test",
            raw_message="test",
            error_category=EnumRuntimeErrorCategory.UNKNOWN,
            severity=EnumRuntimeErrorSeverity.ERROR,
        )
        with pytest.raises(Exception):
            event.logger_family = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        event = ModelRuntimeErrorEvent.create(
            logger_family="aiokafka.consumer",
            log_level="WARNING",
            message_template="slow poll detected",
            raw_message="slow poll detected (2.5s)",
            error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
            severity=EnumRuntimeErrorSeverity.WARNING,
            hostname="host-1",
            service_label="my-service",
        )
        data = event.model_dump(mode="json")
        restored = ModelRuntimeErrorEvent.model_validate(data)
        assert restored.logger_family == event.logger_family
        assert restored.fingerprint == event.fingerprint
        assert restored.hostname == "host-1"

    def test_fingerprint_ignores_raw_message(self) -> None:
        """Fingerprint uses template, not raw message -- different raw = same fp."""
        e1 = ModelRuntimeErrorEvent.create(
            logger_family="test",
            log_level="ERROR",
            message_template="connection refused to {}",
            raw_message="connection refused to host-a",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            severity=EnumRuntimeErrorSeverity.ERROR,
        )
        e2 = ModelRuntimeErrorEvent.create(
            logger_family="test",
            log_level="ERROR",
            message_template="connection refused to {}",
            raw_message="connection refused to host-b",
            error_category=EnumRuntimeErrorCategory.DATABASE,
            severity=EnumRuntimeErrorSeverity.ERROR,
        )
        assert e1.fingerprint == e2.fingerprint
