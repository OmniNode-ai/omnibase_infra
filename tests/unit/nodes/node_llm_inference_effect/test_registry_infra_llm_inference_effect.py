# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for RegistryInfraLlmInferenceEffect.

This module validates the registry functionality for the LLM inference
effect node, including node creation with protocol validation, handler
registration for OpenAI-compatible and Ollama backends, transport adapter
creation, and static metadata methods.

Test Coverage:
    - create(): Node creation with valid container (MixinLlmHttpTransport registered)
    - create(): OnexError when MixinLlmHttpTransport is NOT registered
    - create(): Skips validation when service_registry is None
    - register_openai_compatible(): Registers handler and transport as separate instances
    - register_openai_compatible(): Registers MixinLlmHttpTransport with transport (not handler)
    - register_openai_compatible(): Warning logged when service_registry is None
    - register_ollama(): Registers handler and MixinLlmHttpTransport (handler IS transport)
    - register_ollama(): Warning logged when service_registry is None
    - _create_transport_adapter(): Returns MixinLlmHttpTransport with _execute_llm_http_call
    - Static methods: get_node_type, get_node_name, get_required_protocols,
      get_capabilities, get_supported_operations, get_backends

Related:
    - OMN-2111: Phase 11 inference node assembly
    - RegistryInfraLlmInferenceEffect: Registry implementation
    - NodeLlmInferenceEffect: Declarative effect node
    - HandlerLlmOpenaiCompatible: OpenAI-compatible handler
    - HandlerLlmOllama: Ollama handler
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.errors import OnexError
from omnibase_infra.nodes.node_llm_inference_effect.registry.registry_infra_llm_inference_effect import (
    RegistryInfraLlmInferenceEffect,
    _create_transport_adapter,
)

pytestmark = pytest.mark.unit

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container with async service_registry."""
    container = MagicMock()
    container.service_registry = MagicMock()
    container.service_registry.register_instance = AsyncMock()
    return container


@pytest.fixture
def mock_container_no_registry() -> MagicMock:
    """Create a mock container with service_registry = None."""
    container = MagicMock()
    container.service_registry = None
    return container


@pytest.fixture
def mock_container_transport_registered() -> MagicMock:
    """Create a mock container where get_service resolves MixinLlmHttpTransport."""
    container = MagicMock()
    container.service_registry = MagicMock()
    container.service_registry.register_instance = AsyncMock()
    # get_service returns a mock transport (simulates successful resolution)
    container.get_service = MagicMock(return_value=MagicMock())
    return container


@pytest.fixture
def mock_container_transport_missing() -> MagicMock:
    """Create a mock container where get_service raises for MixinLlmHttpTransport."""
    container = MagicMock()
    container.service_registry = MagicMock()
    container.service_registry.register_instance = AsyncMock()
    # get_service raises when trying to resolve MixinLlmHttpTransport
    container.get_service = MagicMock(side_effect=KeyError("MixinLlmHttpTransport"))
    return container


# =============================================================================
# create() Tests
# =============================================================================


class TestRegistryCreate:
    """Tests for RegistryInfraLlmInferenceEffect.create() method."""

    def test_create_with_valid_container_returns_node(
        self, mock_container_transport_registered: MagicMock
    ) -> None:
        """create() returns NodeLlmInferenceEffect when transport is registered."""
        from omnibase_infra.nodes.node_llm_inference_effect.node import (
            NodeLlmInferenceEffect,
        )

        node = RegistryInfraLlmInferenceEffect.create(
            mock_container_transport_registered
        )

        assert isinstance(node, NodeLlmInferenceEffect)

    def test_create_calls_get_service_with_transport_type(
        self, mock_container_transport_registered: MagicMock
    ) -> None:
        """create() validates MixinLlmHttpTransport is resolvable."""
        from omnibase_infra.mixins import MixinLlmHttpTransport

        RegistryInfraLlmInferenceEffect.create(mock_container_transport_registered)

        mock_container_transport_registered.get_service.assert_called_once_with(
            MixinLlmHttpTransport
        )

    def test_create_raises_onex_error_when_transport_missing(
        self, mock_container_transport_missing: MagicMock
    ) -> None:
        """create() raises OnexError when MixinLlmHttpTransport is not registered."""
        with pytest.raises(OnexError) as exc_info:
            RegistryInfraLlmInferenceEffect.create(mock_container_transport_missing)

        error_msg = str(exc_info.value)
        assert "MixinLlmHttpTransport" in error_msg
        assert "not registered" in error_msg

    def test_create_error_message_suggests_registration_methods(
        self, mock_container_transport_missing: MagicMock
    ) -> None:
        """create() error message suggests calling registration methods."""
        with pytest.raises(OnexError) as exc_info:
            RegistryInfraLlmInferenceEffect.create(mock_container_transport_missing)

        error_msg = str(exc_info.value)
        assert "register_openai_compatible()" in error_msg
        assert "register_ollama()" in error_msg

    def test_create_preserves_exception_chain(
        self, mock_container_transport_missing: MagicMock
    ) -> None:
        """create() chains the original exception via 'from exc'."""
        with pytest.raises(OnexError) as exc_info:
            RegistryInfraLlmInferenceEffect.create(mock_container_transport_missing)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_create_skips_validation_when_registry_is_none(
        self, mock_container_no_registry: MagicMock
    ) -> None:
        """create() still creates node when service_registry is None."""
        from omnibase_infra.nodes.node_llm_inference_effect.node import (
            NodeLlmInferenceEffect,
        )

        node = RegistryInfraLlmInferenceEffect.create(mock_container_no_registry)

        assert isinstance(node, NodeLlmInferenceEffect)

    def test_create_does_not_call_get_service_when_registry_is_none(
        self, mock_container_no_registry: MagicMock
    ) -> None:
        """create() does not attempt service resolution when registry is None."""
        RegistryInfraLlmInferenceEffect.create(mock_container_no_registry)

        # get_service should not have been called at all
        # (MagicMock would auto-create the attribute, so check call count)
        mock_container_no_registry.get_service.assert_not_called()


# =============================================================================
# register_openai_compatible() Tests
# =============================================================================


class TestRegisterOpenaiCompatible:
    """Tests for RegistryInfraLlmInferenceEffect.register_openai_compatible()."""

    @pytest.mark.asyncio
    async def test_registers_handler_and_transport(
        self, mock_container: MagicMock
    ) -> None:
        """register_openai_compatible() calls register_instance twice."""
        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        assert mock_container.service_registry.register_instance.call_count == 2

    @pytest.mark.asyncio
    async def test_registers_handler_as_handler_interface(
        self, mock_container: MagicMock
    ) -> None:
        """register_openai_compatible() registers HandlerLlmOpenaiCompatible interface."""
        from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
            HandlerLlmOpenaiCompatible,
        )

        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        first_call = mock_container.service_registry.register_instance.call_args_list[0]
        assert first_call.kwargs["interface"] is HandlerLlmOpenaiCompatible

    @pytest.mark.asyncio
    async def test_registers_transport_as_mixin_interface(
        self, mock_container: MagicMock
    ) -> None:
        """register_openai_compatible() registers MixinLlmHttpTransport interface."""
        from omnibase_infra.mixins import MixinLlmHttpTransport

        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        second_call = mock_container.service_registry.register_instance.call_args_list[
            1
        ]
        assert second_call.kwargs["interface"] is MixinLlmHttpTransport

    @pytest.mark.asyncio
    async def test_transport_instance_is_not_handler(
        self, mock_container: MagicMock
    ) -> None:
        """register_openai_compatible() registers transport (not handler) as MixinLlmHttpTransport.

        This is a critical distinction: the handler receives the transport via
        constructor injection, but the transport itself (a MixinLlmHttpTransport
        subclass instance) is registered under the MixinLlmHttpTransport interface.
        """
        from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
            HandlerLlmOpenaiCompatible,
        )

        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        calls = mock_container.service_registry.register_instance.call_args_list

        handler_instance = calls[0].kwargs["instance"]
        transport_instance = calls[1].kwargs["instance"]

        # Handler is a HandlerLlmOpenaiCompatible
        assert isinstance(handler_instance, HandlerLlmOpenaiCompatible)

        # Transport is NOT the handler
        assert transport_instance is not handler_instance

        # Transport has _execute_llm_http_call
        assert hasattr(transport_instance, "_execute_llm_http_call")

    @pytest.mark.asyncio
    async def test_uses_global_scope(self, mock_container: MagicMock) -> None:
        """register_openai_compatible() uses GLOBAL injection scope for both registrations."""
        from omnibase_core.enums import EnumInjectionScope

        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        for call in mock_container.service_registry.register_instance.call_args_list:
            assert call.kwargs["scope"] is EnumInjectionScope.GLOBAL

    @pytest.mark.asyncio
    async def test_custom_target_name(self, mock_container: MagicMock) -> None:
        """register_openai_compatible() passes custom target_name to transport."""
        await RegistryInfraLlmInferenceEffect.register_openai_compatible(
            mock_container, target_name="custom-llm"
        )

        calls = mock_container.service_registry.register_instance.call_args_list
        transport_instance = calls[1].kwargs["instance"]

        assert transport_instance._llm_target_name == "custom-llm"

    @pytest.mark.asyncio
    async def test_default_target_name(self, mock_container: MagicMock) -> None:
        """register_openai_compatible() uses 'openai-inference' as default target."""
        await RegistryInfraLlmInferenceEffect.register_openai_compatible(mock_container)

        calls = mock_container.service_registry.register_instance.call_args_list
        transport_instance = calls[1].kwargs["instance"]

        assert transport_instance._llm_target_name == "openai-inference"

    @pytest.mark.asyncio
    async def test_none_registry_logs_warning_and_returns(
        self,
        mock_container_no_registry: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """register_openai_compatible() logs warning when service_registry is None."""
        with caplog.at_level(logging.WARNING):
            await RegistryInfraLlmInferenceEffect.register_openai_compatible(
                mock_container_no_registry
            )

        assert any(
            "service_registry is None" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_none_registry_does_not_raise(
        self, mock_container_no_registry: MagicMock
    ) -> None:
        """register_openai_compatible() returns without error when registry is None."""
        # Should not raise
        await RegistryInfraLlmInferenceEffect.register_openai_compatible(
            mock_container_no_registry
        )


# =============================================================================
# register_ollama() Tests
# =============================================================================


class TestRegisterOllama:
    """Tests for RegistryInfraLlmInferenceEffect.register_ollama()."""

    @pytest.mark.asyncio
    async def test_registers_handler_and_transport(
        self, mock_container: MagicMock
    ) -> None:
        """register_ollama() calls register_instance twice."""
        await RegistryInfraLlmInferenceEffect.register_ollama(mock_container)

        assert mock_container.service_registry.register_instance.call_count == 2

    @pytest.mark.asyncio
    async def test_registers_handler_as_handler_interface(
        self, mock_container: MagicMock
    ) -> None:
        """register_ollama() registers HandlerLlmOllama interface."""
        from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
            HandlerLlmOllama,
        )

        await RegistryInfraLlmInferenceEffect.register_ollama(mock_container)

        first_call = mock_container.service_registry.register_instance.call_args_list[0]
        assert first_call.kwargs["interface"] is HandlerLlmOllama

    @pytest.mark.asyncio
    async def test_registers_handler_as_mixin_transport_interface(
        self, mock_container: MagicMock
    ) -> None:
        """register_ollama() registers handler (not separate transport) as MixinLlmHttpTransport.

        Unlike register_openai_compatible(), HandlerLlmOllama extends
        MixinLlmHttpTransport directly, so the handler instance itself
        is registered under the MixinLlmHttpTransport interface.
        """
        from omnibase_infra.mixins import MixinLlmHttpTransport

        await RegistryInfraLlmInferenceEffect.register_ollama(mock_container)

        calls = mock_container.service_registry.register_instance.call_args_list

        handler_instance = calls[0].kwargs["instance"]
        transport_instance = calls[1].kwargs["instance"]

        # For Ollama, the handler IS the transport
        assert handler_instance is transport_instance
        assert second_interface_is_transport(calls)

    @pytest.mark.asyncio
    async def test_uses_global_scope(self, mock_container: MagicMock) -> None:
        """register_ollama() uses GLOBAL injection scope for both registrations."""
        from omnibase_core.enums import EnumInjectionScope

        await RegistryInfraLlmInferenceEffect.register_ollama(mock_container)

        for call in mock_container.service_registry.register_instance.call_args_list:
            assert call.kwargs["scope"] is EnumInjectionScope.GLOBAL

    @pytest.mark.asyncio
    async def test_custom_target_name(self, mock_container: MagicMock) -> None:
        """register_ollama() passes custom target_name to handler."""
        await RegistryInfraLlmInferenceEffect.register_ollama(
            mock_container, target_name="custom-ollama"
        )

        calls = mock_container.service_registry.register_instance.call_args_list
        handler_instance = calls[0].kwargs["instance"]

        assert handler_instance._llm_target_name == "custom-ollama"

    @pytest.mark.asyncio
    async def test_default_target_name(self, mock_container: MagicMock) -> None:
        """register_ollama() uses 'ollama-inference' as default target."""
        await RegistryInfraLlmInferenceEffect.register_ollama(mock_container)

        calls = mock_container.service_registry.register_instance.call_args_list
        handler_instance = calls[0].kwargs["instance"]

        assert handler_instance._llm_target_name == "ollama-inference"

    @pytest.mark.asyncio
    async def test_none_registry_logs_warning_and_returns(
        self,
        mock_container_no_registry: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """register_ollama() logs warning when service_registry is None."""
        with caplog.at_level(logging.WARNING):
            await RegistryInfraLlmInferenceEffect.register_ollama(
                mock_container_no_registry
            )

        assert any(
            "service_registry is None" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_none_registry_does_not_raise(
        self, mock_container_no_registry: MagicMock
    ) -> None:
        """register_ollama() returns without error when registry is None."""
        # Should not raise
        await RegistryInfraLlmInferenceEffect.register_ollama(
            mock_container_no_registry
        )


def second_interface_is_transport(
    call_args_list: list[MagicMock],
) -> bool:
    """Helper: verify second registration uses MixinLlmHttpTransport interface."""
    from omnibase_infra.mixins import MixinLlmHttpTransport

    return call_args_list[1].kwargs["interface"] is MixinLlmHttpTransport


# =============================================================================
# _create_transport_adapter() Tests
# =============================================================================


class TestCreateTransportAdapter:
    """Tests for _create_transport_adapter() private factory."""

    def test_returns_mixin_llm_http_transport_instance(self) -> None:
        """_create_transport_adapter() returns a MixinLlmHttpTransport subclass."""
        from omnibase_infra.mixins import MixinLlmHttpTransport

        transport = _create_transport_adapter()

        assert isinstance(transport, MixinLlmHttpTransport)

    def test_has_execute_llm_http_call_method(self) -> None:
        """Transport adapter has _execute_llm_http_call method."""
        transport = _create_transport_adapter()

        assert hasattr(transport, "_execute_llm_http_call")
        assert callable(transport._execute_llm_http_call)

    def test_uses_provided_target_name(self) -> None:
        """_create_transport_adapter() passes target_name to transport init."""
        transport = _create_transport_adapter(target_name="my-custom-target")

        assert transport._llm_target_name == "my-custom-target"

    def test_default_target_name(self) -> None:
        """_create_transport_adapter() uses 'openai-inference' as default."""
        transport = _create_transport_adapter()

        assert transport._llm_target_name == "openai-inference"

    def test_initializes_circuit_breaker(self) -> None:
        """Transport adapter initializes circuit breaker via _init_llm_http_transport."""
        transport = _create_transport_adapter()

        assert hasattr(transport, "_circuit_breaker_initialized")
        assert transport._circuit_breaker_initialized is True


# =============================================================================
# Static Method Tests
# =============================================================================


class TestStaticMethods:
    """Tests for static metadata methods."""

    def test_get_node_type_returns_effect_generic(self) -> None:
        """get_node_type() returns 'EFFECT_GENERIC'."""
        assert RegistryInfraLlmInferenceEffect.get_node_type() == "EFFECT_GENERIC"

    def test_get_node_name(self) -> None:
        """get_node_name() returns 'node_llm_inference_effect'."""
        assert (
            RegistryInfraLlmInferenceEffect.get_node_name()
            == "node_llm_inference_effect"
        )

    def test_get_required_protocols(self) -> None:
        """get_required_protocols() includes MixinLlmHttpTransport."""
        protocols = RegistryInfraLlmInferenceEffect.get_required_protocols()

        assert isinstance(protocols, list)
        assert "MixinLlmHttpTransport" in protocols

    def test_get_capabilities(self) -> None:
        """get_capabilities() returns expected capability list."""
        capabilities = RegistryInfraLlmInferenceEffect.get_capabilities()

        assert isinstance(capabilities, list)
        assert "openai_compatible_inference" in capabilities
        assert "ollama_inference" in capabilities
        assert "chat_completion" in capabilities
        assert "tool_calling" in capabilities
        assert "circuit_breaker_protection" in capabilities

    def test_get_supported_operations(self) -> None:
        """get_supported_operations() returns expected operation list."""
        operations = RegistryInfraLlmInferenceEffect.get_supported_operations()

        assert isinstance(operations, list)
        assert "inference.openai_compatible" in operations
        assert "inference.ollama" in operations

    def test_get_backends(self) -> None:
        """get_backends() returns expected backend list."""
        backends = RegistryInfraLlmInferenceEffect.get_backends()

        assert isinstance(backends, list)
        assert "openai_compatible" in backends
        assert "ollama" in backends


__all__: list[str] = [
    "TestRegistryCreate",
    "TestRegisterOpenaiCompatible",
    "TestRegisterOllama",
    "TestCreateTransportAdapter",
    "TestStaticMethods",
]
