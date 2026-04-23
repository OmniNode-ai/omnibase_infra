# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for post-Consul registry discovery semantics."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery


def _make_projection() -> ModelRegistrationProjection:
    """Create a minimal active projection for discovery tests."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        current_state=EnumRegistrationState.ACTIVE,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        capability_tags=["registry.read"],
        last_applied_event_id=uuid4(),
        last_applied_offset=1,
        registered_at=now,
        updated_at=now,
    )


def _make_service() -> tuple[ServiceRegistryDiscovery, AsyncMock]:
    """Create a registry service with a mocked projection reader."""
    container = MagicMock()
    reader = AsyncMock()
    service = ServiceRegistryDiscovery(container=container, projection_reader=reader)
    return service, reader


@pytest.mark.unit
class TestRegistryDiscoverySemantics:
    """Verify discovery payload explicitly degrades legacy instance discovery."""

    @pytest.mark.asyncio
    async def test_get_discovery_marks_instance_discovery_unavailable(self) -> None:
        """Discovery response should surface post-Consul instance unavailability."""
        projection = _make_projection()
        service, reader = _make_service()

        async def mock_get_by_state(
            state: EnumRegistrationState, limit: int = 10000, correlation_id=None
        ) -> list[ModelRegistrationProjection]:
            return [projection] if state == EnumRegistrationState.ACTIVE else []

        reader.get_by_state.side_effect = mock_get_by_state

        response = await service.get_discovery(limit=10, offset=0)

        assert response.instance_discovery_status == "unavailable"
        assert response.instance_discovery_message == (
            "Service discovery not available (Consul removed)"
        )
        assert response.live_instances == []
        assert response.summary.healthy_instances == 0
        assert response.summary.unhealthy_instances == 0
        assert any(w.code == "NO_CONSUL_HANDLER" for w in response.warnings)
