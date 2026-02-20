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

from omnibase_infra.enums import EnumInfraTransportType, EnumResponseStatus
from omnibase_infra.errors import InfraConsulError, RuntimeHostError
from omnibase_infra.handlers.models.consul import (
    ModelConsulHandlerPayload,
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
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
        # Build a handler output with a real ModelConsulHandlerResponse so
        # the isinstance(consul_response, ModelConsulHandlerResponse) guard in
        # IntentEffectConsulRegister passes and is_error is type-safe.
        real_response = ModelConsulHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            payload=ModelConsulHandlerPayload(
                data=ModelConsulRegisterPayload(
                    registered=True,
                    name="onex-effect",
                    consul_service_id="onex-effect-123",
                ),
            ),
            correlation_id=uuid4(),
        )
        mock_output = MagicMock()
        mock_output.result = real_response
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
        # Use a real ModelConsulHandlerResponse with ERROR status so the
        # isinstance guard passes and is_error returns True type-safely.
        real_error_response = ModelConsulHandlerResponse(
            status=EnumResponseStatus.ERROR,
            payload=ModelConsulHandlerPayload(
                data=ModelConsulRegisterPayload(
                    registered=False,
                    name="onex-effect",
                    consul_service_id="onex-effect-123",
                ),
            ),
            correlation_id=correlation_id,
        )
        mock_error_output = MagicMock()
        mock_error_output.result = real_error_response
        mock_consul_handler.execute = AsyncMock(return_value=mock_error_output)

        with pytest.raises(
            RuntimeHostError,
            match="Consul registration returned error status for service_id=onex-effect-123",
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_raises_when_handler_output_is_none(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when handler.execute() returns None.

        Exercises the first defensive guard in IntentEffectConsulRegister: when
        the consul handler returns None instead of a ModelHandlerOutput, the
        adapter must raise RuntimeHostError immediately rather than propagating
        an AttributeError from a subsequent `.result` access.
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        mock_consul_handler.execute = AsyncMock(return_value=None)

        with pytest.raises(
            RuntimeHostError, match="Consul handler returned None output"
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_raises_when_consul_response_is_unexpected_type(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when handler_output.result is not ModelConsulHandlerResponse.

        Exercises the isinstance type-narrowing guard in IntentEffectConsulRegister:
        when the consul handler returns an output whose result is an unexpected
        type (not ModelConsulHandlerResponse), the adapter raises RuntimeHostError
        with a descriptive message rather than letting an AttributeError propagate
        from a later `.is_error` access.
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        mock_unexpected_output = MagicMock()
        mock_unexpected_output.result = {"unexpected": "dict"}
        mock_consul_handler.execute = AsyncMock(return_value=mock_unexpected_output)

        with pytest.raises(
            RuntimeHostError,
            match="Consul handler returned unexpected result type",
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_raises_when_handler_result_is_none(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when handler_output.result is None.

        Exercises the defensive guard in IntentEffectConsulRegister: when
        consul_response (handler_output.result) is None, the adapter raises
        RuntimeHostError immediately rather than silently treating a broken
        handler as a successful registration. This is symmetric with the
        `handler_output is None` guard that already raises above it.
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        # Handler returns an output object but its result is None — signals a
        # broken or non-conformant handler implementation.
        mock_none_output = MagicMock()
        mock_none_output.result = None
        mock_consul_handler.execute = AsyncMock(return_value=mock_none_output)

        with pytest.raises(
            RuntimeHostError,
            match="Consul handler returned None result for service_id=onex-effect-123",
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_reraises_infra_consul_error_from_partial_registration(
        self, effect: IntentEffectConsulRegister, mock_consul_handler: MagicMock
    ) -> None:
        """Should re-raise InfraConsulError without wrapping it in RuntimeHostError.

        Exercises the partial-registration scenario where the Consul agent
        registration succeeds but the KV write fails, causing the handler to
        raise InfraConsulError directly. Since InfraConsulError is a subclass
        of RuntimeHostError, the ``except RuntimeHostError: raise`` guard in
        execute() re-raises it verbatim rather than wrapping it in a new
        RuntimeHostError with the generic "Failed to execute Consul" message.

        This verifies that callers can distinguish partial-registration failures
        (InfraConsulError) from unexpected generic failures (RuntimeHostError).
        """
        correlation_id = uuid4()

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-effect-123",
            service_name="onex-effect",
            tags=["onex"],
        )

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register_service",
        )
        consul_error = InfraConsulError(
            "Consul KV write failed after service registration",
            context=context,
            service_name="onex-effect",
        )
        mock_consul_handler.execute = AsyncMock(side_effect=consul_error)

        with pytest.raises(InfraConsulError) as exc_info:
            await effect.execute(payload, correlation_id=correlation_id)

        # Confirm the exact original error is propagated, not a wrapper.
        assert exc_info.value is consul_error
        # Confirm the message is not replaced by the generic RuntimeHostError message.
        assert "Failed to execute Consul" not in str(exc_info.value)

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


@pytest.mark.unit
class TestIntentEffectConsulRegisterCatalogRemoval:
    """Positive assertions that catalog service is intentionally absent.

    OMN-2314 revert: catalog service intentionally removed pending redesign.

    The file tests/unit/runtime/test_intent_effect_consul_register_catalog.py
    was deleted as part of the OMN-2314 revert. These tests replace that
    deletion with explicit assertions that catalog service wiring is NOT
    present in IntentEffectConsulRegister, confirming the removal is a
    deliberate architectural decision rather than an accidental regression.
    """

    def test_catalog_service_attribute_is_absent(self) -> None:
        """IntentEffectConsulRegister must not carry a catalog service attribute.

        Verifies that the OMN-2314 revert is intentional: no ``_catalog_service``
        (or any ``catalog``-named) attribute is wired into the effect at
        construction time. A catalog integration attribute would indicate the
        revert was not fully applied.
        """
        mock_handler = MagicMock()
        effect = IntentEffectConsulRegister(consul_handler=mock_handler)

        # The only instance attribute set by __init__ is _consul_handler.
        assert not hasattr(effect, "_catalog_service"), (
            "IntentEffectConsulRegister must not have a _catalog_service attribute "
            "(OMN-2314 revert: catalog service removed pending redesign)"
        )

    def test_init_accepts_only_consul_handler(self) -> None:
        """IntentEffectConsulRegister.__init__ must accept consul_handler only.

        Confirms the constructor signature has not been extended with a
        catalog_service parameter, which would re-introduce OMN-2314 wiring.
        """
        import inspect

        sig = inspect.signature(IntentEffectConsulRegister.__init__)
        param_names = [p for p in sig.parameters if p != "self"]

        assert param_names == ["consul_handler"], (
            f"Expected __init__ to accept only 'consul_handler', "
            f"got {param_names!r}. A catalog_service parameter would indicate "
            "the OMN-2314 revert was not fully applied."
        )
