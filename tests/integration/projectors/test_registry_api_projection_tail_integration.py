# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the registry API projection read tail.

These tests validate the exact tail this branch changes:

registration_projections -> ProjectionReaderRegistration
-> ServiceRegistryDiscovery -> registry API response models
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.runtime import ProjectorShell
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery

if TYPE_CHECKING:
    from omnibase_infra.projectors import ProjectionReaderRegistration


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


def _make_projection() -> ModelRegistrationProjection:
    """Create a registration projection with phase-1 and detail-only fields."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=UUID("44444444-4444-4444-4444-444444444444"),
        domain="registration",
        current_state=EnumRegistrationState.ACTIVE,
        node_type=EnumNodeKind.COMPUTE,
        node_version=ModelSemVer(major=3, minor=2, patch=1),
        capabilities=ModelNodeCapabilities(
            postgres=True,
            read=True,
            config={"dsn": "internal-only"},
        ),
        capability_tags=["postgres.storage", "registry.read"],
        contract_type="compute",
        contract_version="3.2.1",
        protocols=["ProtocolDatabaseAdapter"],
        intent_types=["registry.lookup"],
        last_applied_event_id=uuid4(),
        last_applied_offset=101,
        registered_at=now,
        updated_at=now,
    )


def _make_sequence(offset: int) -> ModelSequenceInfo:
    """Create sequence info for persistence tests."""
    return ModelSequenceInfo(sequence=offset, partition="0", offset=offset)


async def _upsert_projection(
    projector: ProjectorShell,
    projection: ModelRegistrationProjection,
) -> bool:
    """Persist a registration projection through the declarative projector."""
    values: dict[str, object] = {
        "entity_id": projection.entity_id,
        "domain": projection.domain,
        "current_state": projection.current_state.value,
        "node_type": projection.node_type.value,
        "node_version": str(projection.node_version),
        "capabilities": projection.capabilities.model_dump_json(),
        "contract_type": projection.contract_type,
        "intent_types": projection.intent_types,
        "protocols": projection.protocols,
        "capability_tags": projection.capability_tags,
        "contract_version": projection.contract_version,
        "liveness_deadline": projection.liveness_deadline,
        "last_heartbeat_at": projection.last_heartbeat_at,
        "last_applied_event_id": projection.last_applied_event_id,
        "last_applied_offset": projection.last_applied_offset,
        "registered_at": projection.registered_at,
        "updated_at": projection.updated_at,
    }
    return await projector.upsert_partial(
        aggregate_id=projection.entity_id,
        values=values,
        correlation_id=uuid4(),
        conflict_columns=["entity_id", "domain"],
    )


class TestRegistryApiProjectionTail:
    """Focused integration coverage for the registry projection read tail."""

    async def test_list_and_detail_are_projection_backed(
        self,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """List/detail should expose projection-backed identity and field policy."""
        projection = _make_projection()

        projection.last_applied_offset = _make_sequence(101).offset or 101
        result = await _upsert_projection(projector, projection)
        assert result is True

        service = ServiceRegistryDiscovery(
            container=MagicMock(),
            projection_reader=reader,
        )

        nodes, pagination, warnings = await service.list_nodes(limit=10, offset=0)

        assert warnings == []
        assert pagination.total == 1
        assert len(nodes) == 1

        node = nodes[0]
        assert node.node_id == projection.entity_id
        assert node.name == str(projection.entity_id)
        assert node.display_name == "onex-compute"
        assert node.service_name is None
        assert node.namespace is None
        assert node.contract_type == "compute"
        assert node.contract_version == "3.2.1"
        assert node.capability_details == {"postgres": True, "read": True}
        assert not hasattr(node, "protocols")
        assert not hasattr(node, "intent_types")

        detail, detail_warnings = await service.get_node(node_id=projection.entity_id)

        assert detail_warnings == []
        assert detail is not None
        assert detail.node_id == projection.entity_id
        assert detail.protocols == ["ProtocolDatabaseAdapter"]
        assert detail.intent_types == ["registry.lookup"]

    async def test_discovery_marks_instance_surface_as_degraded(
        self,
        projector: ProjectorShell,
        reader: ProjectionReaderRegistration,
    ) -> None:
        """Discovery should explicitly mark post-Consul instance unavailability."""
        projection = _make_projection()

        projection.last_applied_offset = _make_sequence(101).offset or 101
        result = await _upsert_projection(projector, projection)
        assert result is True

        service = ServiceRegistryDiscovery(
            container=MagicMock(),
            projection_reader=reader,
        )

        discovery = await service.get_discovery(limit=10, offset=0)

        assert discovery.instance_discovery_status == "unavailable"
        assert discovery.instance_discovery_message == (
            "Service discovery not available (Consul removed)"
        )
        assert discovery.live_instances == []
        assert discovery.summary.total_nodes == 1
        assert discovery.summary.active_nodes == 1
        assert discovery.summary.healthy_instances == 0
        assert discovery.summary.unhealthy_instances == 0
        assert any(w.code == "NO_CONSUL_HANDLER" for w in discovery.warnings)
