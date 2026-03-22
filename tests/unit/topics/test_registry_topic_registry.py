# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for RegistryTopicRegistry DI registration.

Validates the DI round-trip: register -> resolve -> verify instance.

.. versionadded:: 0.24.0
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.protocols import ProtocolTopicRegistry
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.registry_topic_registry import RegistryTopicRegistry
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry


@pytest.mark.unit
class TestRegistryTopicRegistry:
    """Test DI registration of ProtocolTopicRegistry."""

    async def test_register_calls_register_instance(self) -> None:
        """Verify register() calls service_registry.register_instance()."""
        mock_service_registry = MagicMock()
        mock_service_registry.register_instance = AsyncMock()

        mock_container = MagicMock()
        mock_container.service_registry = mock_service_registry

        await RegistryTopicRegistry.register(mock_container)

        mock_service_registry.register_instance.assert_called_once()
        call_kwargs = mock_service_registry.register_instance.call_args
        assert call_kwargs.kwargs["interface"] is ProtocolTopicRegistry
        assert isinstance(call_kwargs.kwargs["instance"], ServiceTopicRegistry)

    async def test_register_skips_when_no_service_registry(self) -> None:
        """Verify register() is a no-op when service_registry is None."""
        mock_container = MagicMock()
        mock_container.service_registry = None

        # Should not raise
        await RegistryTopicRegistry.register(mock_container)

    async def test_registered_instance_resolves_topics(self) -> None:
        """Verify the registered instance can resolve topics."""
        mock_service_registry = MagicMock()
        registered_instance = None

        async def capture_instance(**kwargs: object) -> None:
            nonlocal registered_instance
            registered_instance = kwargs["instance"]

        mock_service_registry.register_instance = capture_instance

        mock_container = MagicMock()
        mock_container.service_registry = mock_service_registry

        await RegistryTopicRegistry.register(mock_container)

        assert registered_instance is not None
        assert isinstance(registered_instance, ProtocolTopicRegistry)
        assert (
            registered_instance.resolve(topic_keys.RESOLUTION_DECIDED)
            == "onex.evt.platform.resolution-decided.v1"
        )

    async def test_repeated_registration_safe(self) -> None:
        """Verify register() can be called multiple times safely."""
        mock_service_registry = MagicMock()
        mock_service_registry.register_instance = AsyncMock()

        mock_container = MagicMock()
        mock_container.service_registry = mock_service_registry

        await RegistryTopicRegistry.register(mock_container)
        await RegistryTopicRegistry.register(mock_container)

        assert mock_service_registry.register_instance.call_count == 2
