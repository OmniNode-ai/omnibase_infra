# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentEffectConsulRegister.

Tests the Consul registration intent effect adapter which bridges
ModelPayloadConsulRegister payloads to HandlerConsul operations.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - IntentEffectConsulRegister: Implementation under test
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.runtime.intent_effects.intent_effect_consul_register import (
    IntentEffectConsulRegister,
)

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
class TestIntentEffectConsulRegisterInit:
    """Tests for IntentEffectConsulRegister initialization."""

    def test_init_with_valid_handler(self) -> None:
        """Should initialize successfully with a valid consul handler."""
        mock_handler = MagicMock()

        effect = IntentEffectConsulRegister(consul_handler=mock_handler)

        assert effect._consul_handler is mock_handler


@pytest.mark.unit
class TestIntentEffectConsulRegisterExecute:
    """Tests for IntentEffectConsulRegister.execute method."""

    @pytest.fixture
    def mock_consul_handler(self) -> MagicMock:
        """Create a mock HandlerConsul with async execute."""
        handler = MagicMock()
        # Build a mock handler output that looks like a successful response:
        # handler_output.result.is_error must be False so the error check
        # added in IntentEffectConsulRegister does not trigger.
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_output = MagicMock()
        mock_output.result = mock_response
        handler.execute = AsyncMock(return_value=mock_output)
        return handler

    @pytest.fixture
    def effect(self, mock_consul_handler: MagicMock) -> IntentEffectConsulRegister:
        """Create an IntentEffectConsulRegister with mocked handler."""
        return IntentEffectConsulRegister(consul_handler=mock_consul_handler)

    @pytest.mark.asyncio
    async def test_execute_calls_consul_register(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should call consul handler execute with registration envelope."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex", "node-type:effect"],
        )

        await effect.execute(payload, correlation_id=correlation_id)

        mock_consul_handler.execute.assert_awaited_once()
        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]  # First positional argument

        assert envelope["operation"] == "consul.register"
        assert envelope["payload"]["name"] == "onex-effect"
        assert envelope["payload"]["service_id"] == "onex-effect-123"
        assert envelope["payload"]["tags"] == ["onex", "node-type:effect"]
        assert envelope["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_execute_includes_health_check(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should include health check in registration payload."""
        correlation_id = uuid4()
        health_check = {"http": "http://localhost:8080/health", "interval": "10s"}

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
            health_check=health_check,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]
        assert envelope["payload"]["check"] == health_check

    @pytest.mark.asyncio
    async def test_execute_excludes_health_check_when_none(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should not include health check key when None."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
            health_check=None,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]
        assert "check" not in envelope["payload"]

    @pytest.mark.asyncio
    async def test_execute_uses_payload_correlation_id_as_fallback(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should fall back to payload.correlation_id when none provided."""
        payload_correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=payload_correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        await effect.execute(payload)

        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]
        assert envelope["correlation_id"] == payload_correlation_id

    @pytest.mark.asyncio
    async def test_execute_raises_runtime_host_error_on_failure(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when consul registration fails."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        mock_consul_handler.execute.side_effect = Exception("Consul unavailable")

        with pytest.raises(RuntimeHostError, match="Failed to execute Consul"):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_raises_runtime_host_error_on_is_error_true(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when handler returns is_error=True.

        Exercises the silent-failure path where the consul handler does not
        raise an exception but returns a result with is_error set to True.
        The adapter must detect this and raise RuntimeHostError so callers
        are never silently misled into thinking registration succeeded.
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        # Handler returns normally (no exception) but result indicates an error.
        mock_error_response = MagicMock()
        mock_error_response.is_error = True
        mock_error_output = MagicMock()
        mock_error_output.result = mock_error_response
        mock_consul_handler.execute = AsyncMock(return_value=mock_error_output)

        with pytest.raises(
            RuntimeHostError,
            match="Consul registration returned error status for service_id=onex-effect-123",
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_succeeds_when_handler_result_is_none(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should complete without raising when handler_output.result is None.

        Exercises the defensive code path documented in IntentEffectConsulRegister:
        when consul_response (handler_output.result) is None, both the is_error
        attribute check and the is_error=True check are skipped, and execute()
        returns normally. This covers EFFECT handlers that may return None results.
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        # Handler returns successfully but result is None (defensive code path).
        mock_none_output = MagicMock()
        mock_none_output.result = None
        mock_consul_handler.execute = AsyncMock(return_value=mock_none_output)

        # Should not raise - the None result path is a documented no-op.
        await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_includes_event_bus_config(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should include event_bus_config when present in payload."""
        from omnibase_infra.models.registration.model_event_bus_topic_entry import (
            ModelEventBusTopicEntry,
        )
        from omnibase_infra.models.registration.model_node_event_bus_config import (
            ModelNodeEventBusConfig,
        )

        correlation_id = uuid4()

        event_bus_config = ModelNodeEventBusConfig(
            subscribe_topics=[ModelEventBusTopicEntry(topic="topic.a")],
            publish_topics=[ModelEventBusTopicEntry(topic="topic.b")],
        )

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            node_id="test-node-id-abc123",
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
            event_bus_config=event_bus_config,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]
        assert "event_bus_config" in envelope["payload"]
        assert envelope["payload"]["node_id"] == "test-node-id-abc123"
