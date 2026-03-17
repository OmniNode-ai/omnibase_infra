# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for CircuitBreakerEventPublisher.

Tests verify:
 - publish_transition() builds a well-formed ModelCircuitBreakerStateEvent
 - the event is forwarded to the event bus with the correct topic
 - bus errors are swallowed (never propagated to callers)
 - correlation_id is threaded through when provided
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums.enum_circuit_state import EnumCircuitState
from omnibase_infra.models.resilience.model_circuit_breaker_state_event import (
    ModelCircuitBreakerStateEvent,
)
from omnibase_infra.services.service_circuit_breaker_event_publisher import (
    CircuitBreakerEventPublisher,
)
from omnibase_infra.topics import SUFFIX_CIRCUIT_BREAKER_STATE


@pytest.mark.unit
class TestCircuitBreakerEventPublisher:
    """Tests for CircuitBreakerEventPublisher."""

    def _make_publisher(self) -> tuple[CircuitBreakerEventPublisher, AsyncMock]:
        mock_bus = MagicMock()
        mock_bus.publish_envelope = AsyncMock()
        publisher = CircuitBreakerEventPublisher(event_bus=mock_bus)
        return publisher, mock_bus.publish_envelope

    @pytest.mark.asyncio
    async def test_publish_transition_calls_event_bus(self) -> None:
        """publish_transition() should call event_bus.publish_envelope exactly once."""
        publisher, publish_envelope = self._make_publisher()

        await publisher.publish_transition(
            service_name="kafka.test",
            new_state=EnumCircuitState.OPEN,
            previous_state=EnumCircuitState.CLOSED,
            failure_count=5,
            threshold=5,
        )

        publish_envelope.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_uses_correct_topic(self) -> None:
        """The event bus must be called with the circuit breaker topic suffix."""
        publisher, publish_envelope = self._make_publisher()

        await publisher.publish_transition(
            service_name="db.primary",
            new_state=EnumCircuitState.OPEN,
            previous_state=EnumCircuitState.CLOSED,
            failure_count=3,
            threshold=3,
        )

        _, kwargs = publish_envelope.call_args
        assert kwargs["topic"] == SUFFIX_CIRCUIT_BREAKER_STATE

    @pytest.mark.asyncio
    async def test_payload_fields(self) -> None:
        """The envelope payload should contain correct field values."""
        publisher, publish_envelope = self._make_publisher()
        corr_id = uuid4()

        await publisher.publish_transition(
            service_name="http.gateway",
            new_state=EnumCircuitState.HALF_OPEN,
            previous_state=EnumCircuitState.OPEN,
            failure_count=0,
            threshold=5,
            correlation_id=corr_id,
        )

        envelope = publish_envelope.call_args[0][0]
        payload = envelope.payload
        assert payload["service_name"] == "http.gateway"
        assert payload["state"] == EnumCircuitState.HALF_OPEN.value
        assert payload["previous_state"] == EnumCircuitState.OPEN.value
        assert payload["failure_count"] == 0
        assert payload["threshold"] == 5

    @pytest.mark.asyncio
    async def test_bus_error_is_swallowed(self) -> None:
        """A publish error must not propagate — circuit breaker must remain functional."""
        mock_bus = MagicMock()
        mock_bus.publish_envelope = AsyncMock(side_effect=RuntimeError("bus down"))
        publisher = CircuitBreakerEventPublisher(event_bus=mock_bus)

        # Should not raise
        await publisher.publish_transition(
            service_name="kafka.prod",
            new_state=EnumCircuitState.OPEN,
            previous_state=EnumCircuitState.CLOSED,
            failure_count=5,
            threshold=5,
        )

    @pytest.mark.asyncio
    async def test_correlation_id_forwarded_in_envelope(self) -> None:
        """When a correlation_id is provided it must appear in the envelope."""
        publisher, publish_envelope = self._make_publisher()
        corr_id = uuid4()

        await publisher.publish_transition(
            service_name="valkey",
            new_state=EnumCircuitState.CLOSED,
            previous_state=EnumCircuitState.HALF_OPEN,
            failure_count=0,
            threshold=5,
            correlation_id=corr_id,
        )

        envelope = publish_envelope.call_args[0][0]
        assert envelope.correlation_id == corr_id

    @pytest.mark.asyncio
    async def test_all_three_state_transitions_publish(self) -> None:
        """All valid state transitions should publish without error."""
        transitions = [
            (EnumCircuitState.OPEN, EnumCircuitState.CLOSED),
            (EnumCircuitState.HALF_OPEN, EnumCircuitState.OPEN),
            (EnumCircuitState.CLOSED, EnumCircuitState.HALF_OPEN),
        ]
        for new_state, prev_state in transitions:
            publisher, publish_envelope = self._make_publisher()
            await publisher.publish_transition(
                service_name="svc",
                new_state=new_state,
                previous_state=prev_state,
                failure_count=2,
                threshold=5,
            )
            publish_envelope.assert_awaited_once()

    def test_model_circuit_breaker_state_event_schema(self) -> None:
        """ModelCircuitBreakerStateEvent must be constructable with valid data."""
        event = ModelCircuitBreakerStateEvent(
            service_name="kafka.dev",
            state=EnumCircuitState.OPEN,
            previous_state=EnumCircuitState.CLOSED,
            failure_count=5,
            threshold=5,
            timestamp=datetime.now(UTC),
        )
        assert event.state == EnumCircuitState.OPEN
        assert event.service_name == "kafka.dev"
        assert event.correlation_id is None

    def test_topic_constant_format(self) -> None:
        """SUFFIX_CIRCUIT_BREAKER_STATE must follow ONEX topic naming conventions."""
        assert SUFFIX_CIRCUIT_BREAKER_STATE.startswith("onex.evt.")
        assert "circuit-breaker" in SUFFIX_CIRCUIT_BREAKER_STATE
        assert SUFFIX_CIRCUIT_BREAKER_STATE.endswith(".v1")
