# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for registry API node identity semantics."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery


def _make_projection(
    *,
    entity_id: UUID | None = None,
    domain: str = "registration",
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
) -> ModelRegistrationProjection:
    """Create a minimal registration projection for registry API tests."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain=domain,
        current_state=state,
        node_type=node_type,
        node_version=ModelSemVer(major=1, minor=1, patch=0),
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
class TestRegistryNodeIdentity:
    """Verify registry node responses do not treat legacy naming as canonical."""

    @pytest.mark.asyncio
    async def test_list_nodes_uses_entity_id_as_stable_name(self) -> None:
        """List responses use the canonical projection id for stable node naming."""
        projection = _make_projection(
            entity_id=UUID("11111111-1111-1111-1111-111111111111")
        )
        service, reader = _make_service()

        async def mock_get_by_state(
            state: EnumRegistrationState, limit: int = 10000, correlation_id=None
        ) -> list[ModelRegistrationProjection]:
            return [projection] if state == EnumRegistrationState.ACTIVE else []

        reader.get_by_state.side_effect = mock_get_by_state

        nodes, pagination, warnings = await service.list_nodes(limit=10, offset=0)

        assert warnings == []
        assert pagination.total == 1
        assert len(nodes) == 1
        assert nodes[0].node_id == projection.entity_id
        assert nodes[0].name == str(projection.entity_id)
        assert nodes[0].display_name == "onex-effect"
        assert nodes[0].service_name is None
        assert nodes[0].namespace is None

    @pytest.mark.asyncio
    async def test_get_node_preserves_non_registration_domain_as_namespace(
        self,
    ) -> None:
        """Non-default projection domains remain explicit registry namespaces."""
        projection = _make_projection(
            entity_id=UUID("22222222-2222-2222-2222-222222222222"),
            domain="market",
            node_type=EnumNodeKind.ORCHESTRATOR,
        )
        service, reader = _make_service()
        reader.get_entity_state.return_value = projection

        node, warnings = await service.get_node(node_id=projection.entity_id)

        assert warnings == []
        assert node is not None
        assert node.node_id == projection.entity_id
        assert node.name == str(projection.entity_id)
        assert node.display_name == "onex-orchestrator"
        assert node.service_name is None
        assert node.namespace == "market"
        assert node.node_type == "ORCHESTRATOR"
