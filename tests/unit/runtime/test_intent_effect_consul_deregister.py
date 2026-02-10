# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentEffectConsulDeregister.

Tests the Consul deregistration intent effect adapter which bridges
ModelPayloadConsulDeregister payloads to HandlerConsul operations.

Related:
    - OMN-2115: Bus audit layer 1 - generic bus health diagnostics
    - IntentEffectConsulDeregister: Implementation under test
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.reducers.models.model_payload_consul_deregister import (
    ModelPayloadConsulDeregister,
)
from omnibase_infra.runtime.intent_effects.intent_effect_consul_deregister import (
    IntentEffectConsulDeregister,
)

pytestmark = [pytest.mark.unit]


@pytest.mark.unit
class TestIntentEffectConsulDeregisterInit:
    """Tests for IntentEffectConsulDeregister initialization."""

    def test_init_stores_consul_handler(self) -> None:
        """Should store the consul handler on the instance."""
        mock_handler = MagicMock()

        effect = IntentEffectConsulDeregister(consul_handler=mock_handler)

        assert effect._consul_handler is mock_handler


@pytest.mark.unit
class TestIntentEffectConsulDeregisterExecute:
    """Tests for IntentEffectConsulDeregister.execute method."""

    @pytest.fixture
    def mock_consul_handler(self) -> MagicMock:
        """Create a mock HandlerConsul with async execute."""
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=MagicMock())
        return handler

    @pytest.fixture
    def effect(self, mock_consul_handler: MagicMock) -> IntentEffectConsulDeregister:
        """Create an IntentEffectConsulDeregister with mocked handler."""
        return IntentEffectConsulDeregister(consul_handler=mock_consul_handler)

    @pytest.mark.asyncio
    async def test_execute_success(
        self, effect: IntentEffectConsulDeregister, mock_consul_handler: MagicMock
    ) -> None:
        """Should call consul handler execute with deregistration envelope."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulDeregister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
        )

        await effect.execute(payload, correlation_id=correlation_id)

        mock_consul_handler.execute.assert_awaited_once()
        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]  # First positional argument

        assert envelope["operation"] == "consul.deregister"
        assert envelope["payload"]["service_id"] == "onex-effect-123"
        assert envelope["correlation_id"] == correlation_id
        assert "envelope_id" in envelope

    @pytest.mark.asyncio
    async def test_execute_wrong_payload_type_rejected(
        self, effect: IntentEffectConsulDeregister
    ) -> None:
        """Should raise RuntimeHostError when payload is not ModelPayloadConsulDeregister."""
        wrong_payload = MagicMock()

        with pytest.raises(
            RuntimeHostError, match="Expected ModelPayloadConsulDeregister"
        ):
            await effect.execute(wrong_payload, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_wraps_generic_exception(
        self, effect: IntentEffectConsulDeregister, mock_consul_handler: MagicMock
    ) -> None:
        """Should wrap non-RuntimeHostError exceptions in RuntimeHostError."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulDeregister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
        )

        mock_consul_handler.execute.side_effect = Exception("Consul unavailable")

        with pytest.raises(RuntimeHostError, match="Failed to execute Consul"):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_propagates_runtime_host_error(
        self, effect: IntentEffectConsulDeregister, mock_consul_handler: MagicMock
    ) -> None:
        """Should re-raise RuntimeHostError as-is without wrapping."""
        correlation_id = uuid4()

        payload = ModelPayloadConsulDeregister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
        )

        original_error = RuntimeHostError("Original consul error")
        mock_consul_handler.execute.side_effect = original_error

        with pytest.raises(RuntimeHostError, match="Original consul error") as exc_info:
            await effect.execute(payload, correlation_id=correlation_id)

        assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_execute_uses_provided_correlation_id(
        self, effect: IntentEffectConsulDeregister, mock_consul_handler: MagicMock
    ) -> None:
        """Should use the explicitly provided correlation_id over the payload's."""
        payload_correlation_id = uuid4()
        explicit_correlation_id = uuid4()

        payload = ModelPayloadConsulDeregister(
            correlation_id=payload_correlation_id,
            service_id="onex-effect-456",
        )

        await effect.execute(payload, correlation_id=explicit_correlation_id)

        call_args = mock_consul_handler.execute.call_args
        envelope = call_args[0][0]
        assert envelope["correlation_id"] == explicit_correlation_id
