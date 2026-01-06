# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for registration handler container wiring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.container_wiring import (
    get_handler_node_introspected_from_container,
    get_handler_node_registration_acked_from_container,
    get_handler_runtime_tick_from_container,
    get_projection_reader_from_container,
    wire_registration_handlers,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


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
    async def test_raises_runtime_error_on_registration_failure(self) -> None:
        """Test that RuntimeError is raised if registration fails."""
        mock_registry = MagicMock()
        mock_registry.register_instance = AsyncMock(
            side_effect=Exception("Registry error")
        )

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        mock_pool = MagicMock()

        with pytest.raises(RuntimeError, match="Failed to wire registration handlers"):
            await wire_registration_handlers(mock_container, mock_pool)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_service_registry_init_failure(self) -> None:
        """Test that RuntimeError is raised if service_registry auto-init fails.

        In omnibase_core 0.6.2+, the service_registry may need to be lazy-initialized.
        If the required omnibase_core components cannot be imported or initialization
        fails, wire_registration_handlers should raise a RuntimeError.
        """
        # Create mock container with service_registry = None (triggers auto-init)
        mock_container = MagicMock()
        mock_container.service_registry = None

        mock_pool = MagicMock()

        # Patch the import to raise ImportError (simulating missing omnibase_core components)
        with patch.dict(
            "sys.modules",
            {
                "omnibase_core.container": None,  # Simulate missing module
            },
        ):
            # The import inside _ensure_service_registry should fail
            with pytest.raises(
                RuntimeError, match="Failed to initialize service_registry"
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
    async def test_raises_runtime_error_if_not_registered(self) -> None:
        """Test that RuntimeError is raised if reader not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            RuntimeError, match="ProjectionReaderRegistration not registered"
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
    async def test_raises_runtime_error_if_not_registered(self) -> None:
        """Test that RuntimeError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            RuntimeError, match="HandlerNodeIntrospected not registered"
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
    async def test_raises_runtime_error_if_not_registered(self) -> None:
        """Test that RuntimeError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(RuntimeError, match="HandlerRuntimeTick not registered"):
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
    async def test_raises_runtime_error_if_not_registered(self) -> None:
        """Test that RuntimeError is raised if handler not registered."""
        mock_registry = MagicMock()
        mock_registry.resolve_service = AsyncMock(side_effect=Exception("Not found"))

        mock_container = MagicMock()
        mock_container.service_registry = mock_registry

        with pytest.raises(
            RuntimeError, match="HandlerNodeRegistrationAcked not registered"
        ):
            await get_handler_node_registration_acked_from_container(mock_container)
