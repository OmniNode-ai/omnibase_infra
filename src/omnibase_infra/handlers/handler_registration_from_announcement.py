# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Adapt a node announcement to the introspection event the registry reads.

The registration orchestrator consumes ``ModelNodeIntrospectionEvent`` whose
required identity fields (node id, node type, endpoints) the announcement does
not carry. This pure handler builds that event from the announcement and the
runtime's own identity. Same announcement and identity give the same event: no
clock, no random ids; the timestamp is the announcement's ``announced_at`` and
the correlation id is derived from the announcement.
"""

from __future__ import annotations

import logging
from uuid import uuid5

from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_announcement_operation import EnumAnnouncementOperation
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumIntrospectionReason,
)
from omnibase_infra.models.model_registration_from_announcement_request import (
    ModelRegistrationFromAnnouncementRequest,
)
from omnibase_infra.models.model_registration_from_announcement_result import (
    ModelRegistrationFromAnnouncementResult,
)
from omnibase_infra.models.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata

logger = logging.getLogger(__name__)


class HandlerRegistrationFromAnnouncement:
    """Pure adapter: ``ModelNodeAnnouncement`` + runtime identity -> introspection event."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    @classmethod
    def build_events(
        cls,
        announcements: tuple[ModelNodeAnnouncement, ...],
        identity: ModelRuntimeIdentity,
    ) -> tuple[ModelNodeIntrospectionEvent, ...]:
        """Introspection events for every announce record, in order."""
        results = (
            cls().handle(
                ModelRegistrationFromAnnouncementRequest(
                    announcement=announcement, identity=identity
                )
            )
            for announcement in announcements
        )
        return tuple(r.event for r in results if r.event is not None)

    def handle(
        self, request: ModelRegistrationFromAnnouncementRequest
    ) -> ModelRegistrationFromAnnouncementResult:
        announcement = request.announcement
        identity = request.identity
        if announcement.node_kind is not EnumAnnouncedNodeKind.NODE:
            return ModelRegistrationFromAnnouncementResult(
                skipped_reason=f"kind {announcement.node_kind.value} registers no node"
            )
        if announcement.operation is not EnumAnnouncementOperation.ANNOUNCE:
            return ModelRegistrationFromAnnouncementResult(
                skipped_reason=f"operation {announcement.operation.value} registers nothing"
            )
        try:
            version = ModelSemVer.parse(announcement.package.version)
        except (ModelOnexError, ValueError):
            logger.warning(
                "Package version %r of %s is not semver; registering with the runtime version",
                announcement.package.version,
                announcement.entity_id,
            )
            version = identity.node_version
        event = ModelNodeIntrospectionEvent(
            node_id=identity.node_id,
            node_name=announcement.entity_id,
            node_type=identity.node_type,
            node_version=version,
            endpoints=dict(identity.endpoints),
            reason=EnumIntrospectionReason.STARTUP,
            correlation_id=uuid5(
                announcement.announcer_id,
                f"{announcement.entity_id}:{announcement.content_hash}:{announcement.generation}",
            ),
            timestamp=announcement.announced_at,
            node_role=identity.node_role,
            metadata=ModelNodeMetadata(
                version=announcement.package.version,
                description=f"{announcement.package.distribution} {announcement.entity_id}",
            ),
            network_id=identity.network_id,
            deployment_id=identity.deployment_id,
            epoch=announcement.generation,
            event_bus=identity.event_bus,
        )
        return ModelRegistrationFromAnnouncementResult(event=event)


__all__ = ["HandlerRegistrationFromAnnouncement"]
