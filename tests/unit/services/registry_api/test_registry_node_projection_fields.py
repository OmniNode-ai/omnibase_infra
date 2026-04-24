# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for phase-1 projection-backed registry node fields."""

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


def _make_projection() -> ModelRegistrationProjection:
    """Create a projection with phase-1 and detail-only fields populated."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=UUID("33333333-3333-3333-3333-333333333333"),
        current_state=EnumRegistrationState.ACTIVE,
        node_type=EnumNodeKind.COMPUTE,
        node_version=ModelSemVer(major=2, minor=3, patch=4),
        capabilities={
            "postgres": True,
            "read": True,
            "config": {"dsn": "hidden"},
            "feature_flags": {},
        },
        capability_tags=["postgres.storage", "registry.read"],
        contract_type="compute",
        contract_version="2.3.4",
        protocols=["ProtocolDatabaseAdapter", "ProtocolReadable"],
        intent_types=["registry.lookup", "registry.sync"],
        last_applied_event_id=uuid4(),
        last_applied_offset=9,
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
class TestRegistryNodeProjectionFields:
    """Verify list/detail exposure of approved projection-backed fields."""

    @pytest.mark.asyncio
    async def test_list_nodes_exposes_phase1_fields_only(self) -> None:
        """List nodes include phase-1 additions but omit detail-only fields."""
        projection = _make_projection()
        service, reader = _make_service()

        async def mock_get_by_state(
            state: EnumRegistrationState, limit: int = 10000, correlation_id=None
        ) -> list[ModelRegistrationProjection]:
            return [projection] if state == EnumRegistrationState.ACTIVE else []

        reader.get_by_state.side_effect = mock_get_by_state

        nodes, _, warnings = await service.list_nodes(limit=10, offset=0)

        assert warnings == []
        assert len(nodes) == 1
        node = nodes[0]
        assert node.contract_type == "compute"
        assert node.contract_version == "2.3.4"
        assert node.updated_at == projection.updated_at
        assert node.capability_details == {"postgres": True, "read": True}
        assert not hasattr(node, "protocols")
        assert not hasattr(node, "intent_types")

    @pytest.mark.asyncio
    async def test_get_node_exposes_detail_only_fields(self) -> None:
        """Node detail includes projection-backed protocols and intent types."""
        projection = _make_projection()
        service, reader = _make_service()
        reader.get_entity_state.return_value = projection

        node, warnings = await service.get_node(node_id=projection.entity_id)

        assert warnings == []
        assert node is not None
        assert node.contract_type == "compute"
        assert node.contract_version == "2.3.4"
        assert node.capability_details == {"postgres": True, "read": True}
        assert node.protocols == ["ProtocolDatabaseAdapter", "ProtocolReadable"]
        assert node.intent_types == ["registry.lookup", "registry.sync"]
