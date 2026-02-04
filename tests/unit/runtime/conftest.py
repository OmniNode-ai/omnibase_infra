# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for runtime unit tests.

This conftest.py provides fixtures commonly used across runtime tests,
consolidating shared mocks to avoid code duplication.

Fixtures:
    mock_wire_infrastructure: Mocks wire_infrastructure_services and
        ModelONEXContainer to avoid wiring errors in tests.
    mock_runtime_handler: Re-exported from tests.conftest for handler seeding.

Functions:
    seed_mock_handlers: Re-exported from tests.conftest for fail-fast bypass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.models import ModelEventBusConfig, ModelRuntimeConfig
from omnibase_infra.runtime.registry import RegistryProtocolBinding

# Re-export handler seeding utilities from root conftest
# These are available here for convenience but defined in tests/conftest.py
# to make them available to both unit and integration tests.
from tests.conftest import mock_runtime_handler, seed_mock_handlers

__all__ = ["mock_runtime_handler", "seed_mock_handlers"]

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_wire_infrastructure() -> Generator[MagicMock, None, None]:
    """Mock wire_infrastructure_services and container to avoid wiring errors in tests.

    This fixture mocks both:
    1. wire_infrastructure_services - to be a no-op async function
    2. ModelONEXContainer - to have a mock service_registry with resolve_service

    Note: Returns a real RegistryProtocolBinding for handler registration to work.
    """
    # Create a shared registry instance that will be used throughout the test
    shared_registry = RegistryProtocolBinding()

    async def noop_wire(container: object) -> dict[str, list[str]]:
        """Async no-op for wire_infrastructure_services."""
        return {"services": []}

    async def mock_resolve_service(
        service_class: type,
    ) -> MagicMock | RegistryProtocolBinding:
        """Mock resolve_service to return appropriate instances.

        Returns a real RegistryProtocolBinding for handler registration,
        and MagicMock for other service types.
        """
        if service_class == RegistryProtocolBinding:
            return shared_registry
        return MagicMock()

    with patch(
        "omnibase_infra.runtime.service_kernel.wire_infrastructure_services"
    ) as mock_wire:
        mock_wire.side_effect = noop_wire

        with patch(
            "omnibase_infra.runtime.service_kernel.ModelONEXContainer"
        ) as mock_container_cls:
            mock_container = MagicMock()
            mock_service_registry = MagicMock()
            mock_service_registry.resolve_service = AsyncMock(
                side_effect=mock_resolve_service
            )
            # Also mock register_instance as AsyncMock to avoid
            # "object MagicMock can't be used in 'await' expression" errors
            # when wire_registration_handlers calls await register_instance(...)
            mock_service_registry.register_instance = AsyncMock(
                return_value="mock-uuid"
            )
            mock_container.service_registry = mock_service_registry
            mock_container_cls.return_value = mock_container
            yield mock_wire


@pytest.fixture
def mock_inmemory_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[MagicMock, None, None]:
    """Force inmemory event bus via environment variable override.

    This fixture uses the ONEX_EVENT_BUS_TYPE environment variable override
    mechanism (documented in runtime_config.yaml) to force inmemory event bus.
    This is more robust than patching load_runtime_config because:
    1. It uses the documented environment variable override mechanism
    2. The kernel explicitly checks ONEX_EVENT_BUS_TYPE before config.event_bus.type
    3. It doesn't rely on patch timing

    The default runtime_config.yaml has event_bus.type='kafka', which requires
    KAFKA_BOOTSTRAP_SERVERS env var. For unit tests that don't test Kafka
    specifically, we use ONEX_EVENT_BUS_TYPE=inmemory to bypass this.

    Yields:
        MagicMock (for backwards compatibility with tests expecting a MagicMock).
    """
    # Use environment variable override (highest precedence per service_kernel.py)
    monkeypatch.setenv("ONEX_EVENT_BUS_TYPE", "inmemory")
    # Ensure no Kafka bootstrap servers are set
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    # Return MagicMock for backwards compatibility with tests that
    # reference the fixture but don't actually use the mock object
    return MagicMock()
