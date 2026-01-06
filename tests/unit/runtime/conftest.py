# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for runtime unit tests.

This conftest.py provides fixtures commonly used across runtime tests,
consolidating shared mocks to avoid code duplication.

Fixtures:
    mock_wire_infrastructure: Mocks wire_infrastructure_services and
        ModelONEXContainer to avoid wiring errors in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_wire_infrastructure() -> Generator[MagicMock, None, None]:
    """Mock wire_infrastructure_services and container to avoid wiring errors in tests.

    This fixture mocks both:
    1. wire_infrastructure_services - to be a no-op async function
    2. ModelONEXContainer - to have a mock service_registry with resolve_service
    """

    async def noop_wire(container: object) -> dict[str, list[str]]:
        """Async no-op for wire_infrastructure_services."""
        return {"services": []}

    async def mock_resolve_service(service_class: type) -> MagicMock:
        """Mock resolve_service to return a MagicMock for any service."""
        return MagicMock()

    with patch(
        "omnibase_infra.runtime.kernel.wire_infrastructure_services"
    ) as mock_wire:
        mock_wire.side_effect = noop_wire

        with patch(
            "omnibase_infra.runtime.kernel.ModelONEXContainer"
        ) as mock_container_cls:
            mock_container = MagicMock()
            mock_service_registry = MagicMock()
            mock_service_registry.resolve_service = AsyncMock(
                side_effect=mock_resolve_service
            )
            mock_container.service_registry = mock_service_registry
            mock_container_cls.return_value = mock_container
            yield mock_wire
