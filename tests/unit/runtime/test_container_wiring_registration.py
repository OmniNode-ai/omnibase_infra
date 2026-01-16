# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for registration handler container wiring functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.errors import ContainerWiringError, ServiceResolutionError
from omnibase_infra.runtime.container_wiring import (
    get_handler_node_introspected_from_container,
    get_handler_node_registration_acked_from_container,
    get_handler_runtime_tick_from_container,
    get_projection_reader_from_container,
    wire_registration_handlers,
)


class TestWireRegistrationHandlers:
    """Tests for wire_registration_handlers function."""

    @pytest.mark.asyncio
    async def test_registers_all_handlers_successfully(self) -> None:
        """Test that all handlers are registered in container."""
        # Create mock container with mock service_registry
        mock_registry = MagicMock()
        mock_registry.register_instance = AsyncMock()

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        # Create mock pool
        mock_pool = MagicMock()

        # Call wire function
        summary = await wire_registration_handlers(mock_container, mock_pool)

        # Verify summary contains all services
        assert "services" in summary
        assert "ProjectionReaderRegistration" in summary["services"]
        assert "HandlerNodeIntrospected" in summary["services"]
        assert "HandlerRuntimeTick" in summary["services"]
        assert "HandlerNodeRegistrationAcked" in summary["services"]
        assert len(summary["services"]) == 4

    @pytest.mark.asyncio
    async def test_registers_instances_with_correct_interfaces(self) -> None:
        """Test that handlers are registered with correct interface types."""
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
            HandlerNodeRegistrationAcked,
            HandlerRuntimeTick,
        )
        from omnibase_infra.projectors import ProjectionReaderRegistration

        # Track registered interfaces
        registered_interfaces: list[type] = []

        async def capture_register(interface: type, **kwargs) -> None:
            registered_interfaces.append(interface)

        mock_registry = MagicMock()
        mock_registry.register_instance = AsyncMock(side_effect=capture_register)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        mock_pool = MagicMock()

        await wire_registration_handlers(mock_container, mock_pool)

        # Verify all expected interfaces were registered
        assert ProjectionReaderRegistration in registered_interfaces
        assert HandlerNodeIntrospected in registered_interfaces
        assert HandlerRuntimeTick in registered_interfaces
        assert HandlerNodeRegistrationAcked in registered_interfaces

    @pytest.mark.asyncio
    async def test_custom_liveness_interval_passed_to_handler(self) -> None:
        """Test that custom liveness interval is passed to ack handler."""
        # Track registration calls
        registrations: list[dict] = []

        async def capture_register(**kwargs) -> None:
            registrations.append(kwargs)

        mock_registry = MagicMock()
        mock_registry.register_instance = AsyncMock(side_effect=capture_register)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        mock_pool = MagicMock()

        # Use custom liveness interval
        await wire_registration_handlers(
            mock_container, mock_pool, liveness_interval_seconds=120
        )

        # Find the ack handler registration
        ack_handler_reg = next(
            (
                r
                for r in registrations
                if "liveness_interval_seconds" in r.get("metadata", {})
            ),
            None,
        )

        assert ack_handler_reg is not None
        assert ack_handler_reg["metadata"]["liveness_interval_seconds"] == 120

    @pytest.mark.asyncio
    async def test_raises_container_wiring_error_on_registration_failure(self) -> None:
        """Test that ContainerWiringError is raised if registration fails."""
        mock_registry = MagicMock()
        mock_registry.register_instance = AsyncMock(
            side_effect=Exception("Registry error")
        )

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        mock_pool = MagicMock()

        with pytest.raises(
            ContainerWiringError, match="Failed to wire registration handlers"
        ):
            await wire_registration_handlers(mock_container, mock_pool)

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test that ServiceRegistryUnavailableError is raised if container missing service_registry.

        OMN-1257: Changed from RuntimeError to ServiceRegistryUnavailableError
        for clearer error messages when service_registry is missing or None.
        """
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock(spec=[])  # No service_registry attribute
        del mock_container.service_registry

        mock_pool = MagicMock()

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await wire_registration_handlers(mock_container, mock_pool)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test that ServiceRegistryUnavailableError is raised if service_registry is None.

        OMN-1257: Tests the second validation branch where service_registry
        attribute exists but is set to None (e.g., when enable_service_registry=False).
        """
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock()
        mock_container.service_registry = None  # Exists but is None

        mock_pool = MagicMock()

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await wire_registration_handlers(mock_container, mock_pool)


class TestGetProjectionReaderFromContainer:
    """Tests for get_projection_reader_from_container function."""

    @pytest.mark.asyncio
    async def test_resolves_projection_reader(self) -> None:
        """Test that projection reader is resolved from container."""
        from omnibase_infra.projectors import ProjectionReaderRegistration

        mock_reader = MagicMock(spec=ProjectionReaderRegistration)

        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(return_value=mock_reader)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        result = await get_projection_reader_from_container(mock_container)

        assert result is mock_reader
        mock_registry.resolve_service.assert_awaited_once_with(
            ProjectionReaderRegistration
        )

    @pytest.mark.asyncio
    async def test_raises_service_resolution_error_if_not_registered(self) -> None:
        """Test that ServiceResolutionError is raised if reader not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            ServiceResolutionError, match="ProjectionReaderRegistration not registered"
        ):
            await get_projection_reader_from_container(mock_container)


class TestGetHandlerNodeIntrospectedFromContainer:
    """Tests for get_handler_node_introspected_from_container function."""

    @pytest.mark.asyncio
    async def test_resolves_handler(self) -> None:
        """Test that handler is resolved from container."""
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )

        mock_handler = MagicMock(spec=HandlerNodeIntrospected)

        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(return_value=mock_handler)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        result = await get_handler_node_introspected_from_container(mock_container)

        assert result is mock_handler
        mock_registry.resolve_service.assert_awaited_once_with(HandlerNodeIntrospected)

    @pytest.mark.asyncio
    async def test_raises_service_resolution_error_if_not_registered(self) -> None:
        """Test that ServiceResolutionError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            ServiceResolutionError, match="HandlerNodeIntrospected not registered"
        ):
            await get_handler_node_introspected_from_container(mock_container)


class TestGetHandlerRuntimeTickFromContainer:
    """Tests for get_handler_runtime_tick_from_container function."""

    @pytest.mark.asyncio
    async def test_resolves_handler(self) -> None:
        """Test that handler is resolved from container."""
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerRuntimeTick,
        )

        mock_handler = MagicMock(spec=HandlerRuntimeTick)

        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(return_value=mock_handler)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        result = await get_handler_runtime_tick_from_container(mock_container)

        assert result is mock_handler
        mock_registry.resolve_service.assert_awaited_once_with(HandlerRuntimeTick)

    @pytest.mark.asyncio
    async def test_raises_service_resolution_error_if_not_registered(self) -> None:
        """Test that ServiceResolutionError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            ServiceResolutionError, match="HandlerRuntimeTick not registered"
        ):
            await get_handler_runtime_tick_from_container(mock_container)


class TestGetHandlerNodeRegistrationAckedFromContainer:
    """Tests for get_handler_node_registration_acked_from_container function."""

    @pytest.mark.asyncio
    async def test_resolves_handler(self) -> None:
        """Test that handler is resolved from container."""
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeRegistrationAcked,
        )

        mock_handler = MagicMock(spec=HandlerNodeRegistrationAcked)

        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(return_value=mock_handler)

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        result = await get_handler_node_registration_acked_from_container(
            mock_container
        )

        assert result is mock_handler
        mock_registry.resolve_service.assert_awaited_once_with(
            HandlerNodeRegistrationAcked
        )

    @pytest.mark.asyncio
    async def test_raises_service_resolution_error_if_not_registered(self) -> None:
        """Test that ServiceResolutionError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            ServiceResolutionError, match="HandlerNodeRegistrationAcked not registered"
        ):
            await get_handler_node_registration_acked_from_container(mock_container)


class TestWireInfrastructureServicesValidation:
    """Tests for wire_infrastructure_services service_registry validation."""

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

        mock_container = MagicMock(spec=[])  # No service_registry attribute
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await wire_infrastructure_services(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await wire_infrastructure_services(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await wire_infrastructure_services(mock_container)

        # Verify operation name is in the error message
        assert "wire_infrastructure_services" in str(exc_info.value)


class TestGetPolicyRegistryFromContainerValidation:
    """Tests for get_policy_registry_from_container service_registry validation."""

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_policy_registry_from_container,
        )

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_policy_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_policy_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_policy_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_policy_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await get_policy_registry_from_container(mock_container)

        # Verify operation name is in the error message
        assert "resolve PolicyRegistry" in str(exc_info.value)


class TestGetComputeRegistryFromContainerValidation:
    """Tests for get_compute_registry_from_container service_registry validation."""

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_compute_registry_from_container,
        )

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_compute_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_compute_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_compute_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_compute_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await get_compute_registry_from_container(mock_container)

        # Verify operation name is in the error message
        assert "resolve RegistryCompute" in str(exc_info.value)


class TestGetHandlerRegistryFromContainerValidation:
    """Tests for get_handler_registry_from_container service_registry validation."""

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_handler_registry_from_container,
        )

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_handler_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_handler_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_handler_registry_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_handler_registry_from_container,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await get_handler_registry_from_container(mock_container)

        # Verify operation name is in the error message
        assert "resolve ProtocolBindingRegistry" in str(exc_info.value)


class TestGetOrCreatePolicyRegistryValidation:
    """Tests for get_or_create_policy_registry service_registry validation.

    OMN-1021: Added tests for None service_registry handling in
    get_or_create_policy_registry, which calls _validate_service_registry
    before attempting to resolve or create the PolicyRegistry.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_policy_registry,
        )

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_or_create_policy_registry(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_policy_registry,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_or_create_policy_registry(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_policy_registry,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await get_or_create_policy_registry(mock_container)

        # Verify operation name is in the error message
        assert "get_or_create PolicyRegistry" in str(exc_info.value)


class TestGetOrCreateComputeRegistryValidation:
    """Tests for get_or_create_compute_registry service_registry validation.

    OMN-1021: Added tests for None service_registry handling in
    get_or_create_compute_registry, which calls _validate_service_registry
    before attempting to resolve or create the RegistryCompute.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_compute_registry,
        )

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_or_create_compute_registry(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_compute_registry,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_or_create_compute_registry(mock_container)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            get_or_create_compute_registry,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await get_or_create_compute_registry(mock_container)

        # Verify operation name is in the error message
        assert "get_or_create RegistryCompute" in str(exc_info.value)


class TestWireRegistrationDispatchersValidation:
    """Tests for wire_registration_dispatchers service_registry validation."""

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing.

        OMN-1021: Tests that wire_registration_dispatchers validates service_registry
        before attempting to wire dispatchers.
        """
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            wire_registration_dispatchers,
        )

        mock_container = MagicMock(spec=[])  # No service_registry attribute
        del mock_container.service_registry
        mock_engine = MagicMock()

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await wire_registration_dispatchers(mock_container, mock_engine)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None.

        OMN-1021: Tests the second validation branch where service_registry
        attribute exists but is set to None (e.g., when enable_service_registry=False).
        """
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            wire_registration_dispatchers,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None
        mock_engine = MagicMock()

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await wire_registration_dispatchers(mock_container, mock_engine)

    @pytest.mark.asyncio
    async def test_error_contains_operation_name(self) -> None:
        """Test that error message includes operation context."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError
        from omnibase_infra.runtime.container_wiring import (
            wire_registration_dispatchers,
        )

        mock_container = MagicMock()
        mock_container.service_registry = None
        mock_engine = MagicMock()

        with pytest.raises(ServiceRegistryUnavailableError) as exc_info:
            await wire_registration_dispatchers(mock_container, mock_engine)

        # Verify operation name is in the error message
        assert "wire_registration_dispatchers" in str(exc_info.value)


class TestGetProjectionReaderFromContainerValidation:
    """Tests for get_projection_reader_from_container service_registry validation.

    OMN-1021: Added tests to verify _validate_service_registry is called and
    raises ServiceRegistryUnavailableError when service_registry is missing or None.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_projection_reader_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_projection_reader_from_container(mock_container)


class TestGetHandlerNodeIntrospectedFromContainerValidation:
    """Tests for get_handler_node_introspected_from_container service_registry validation.

    OMN-1021: Added tests to verify _validate_service_registry is called and
    raises ServiceRegistryUnavailableError when service_registry is missing or None.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_handler_node_introspected_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_handler_node_introspected_from_container(mock_container)


class TestGetHandlerRuntimeTickFromContainerValidation:
    """Tests for get_handler_runtime_tick_from_container service_registry validation.

    OMN-1021: Added tests to verify _validate_service_registry is called and
    raises ServiceRegistryUnavailableError when service_registry is missing or None.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_handler_runtime_tick_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_handler_runtime_tick_from_container(mock_container)


class TestGetHandlerNodeRegistrationAckedFromContainerValidation:
    """Tests for get_handler_node_registration_acked_from_container service_registry validation.

    OMN-1021: Added tests to verify _validate_service_registry is called and
    raises ServiceRegistryUnavailableError when service_registry is missing or None.
    """

    @pytest.mark.asyncio
    async def test_raises_error_on_missing_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry attribute is missing."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock(spec=[])
        del mock_container.service_registry

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container missing 'service_registry' attribute",
        ):
            await get_handler_node_registration_acked_from_container(mock_container)

    @pytest.mark.asyncio
    async def test_raises_error_on_none_service_registry(self) -> None:
        """Test ServiceRegistryUnavailableError when service_registry is None."""
        from omnibase_infra.errors import ServiceRegistryUnavailableError

        mock_container = MagicMock()
        mock_container.service_registry = None

        with pytest.raises(
            ServiceRegistryUnavailableError,
            match="Container service_registry is None",
        ):
            await get_handler_node_registration_acked_from_container(mock_container)
