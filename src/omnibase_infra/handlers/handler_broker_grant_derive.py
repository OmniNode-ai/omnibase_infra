# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure conversion of resolved wire operations into exact broker grants."""

from typing import Literal

from omnibase_core.enums.enum_bus_binding_direction import EnumBusBindingDirection
from omnibase_core.models.event_bus.model_resolved_bus_bindings import (
    ModelResolvedBusBindings,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant import (
    ModelBrokerGrant,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_derivation import (
    ModelBrokerGrantDerivation,
)


class HandlerBrokerGrantDerive:
    """Derive desired grants without broker access or deployment policy."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelResolvedBusBindings) -> ModelBrokerGrantDerivation:
        """Grant topic access and group READ; inspect only declared groups."""
        grants: set[ModelBrokerGrant] = set()
        for binding in request.bindings:
            consuming = binding.direction == EnumBusBindingDirection.CONSUME
            operations: tuple[Literal["READ", "WRITE", "DESCRIBE"], ...] = (
                "READ" if consuming else "WRITE",
                "DESCRIBE",
            )
            for operation in operations:
                grants.add(
                    ModelBrokerGrant(
                        broker=binding.broker,
                        resource_type="TOPIC",
                        resource=binding.physical_topic,
                        operation=operation,
                    )
                )
            if consuming:
                assert binding.consumer_group is not None
                grants.add(
                    ModelBrokerGrant(
                        broker=binding.broker,
                        resource_type="GROUP",
                        resource=binding.consumer_group,
                        operation="READ",
                    )
                )
        for inspection in request.described_groups:
            grants.add(
                ModelBrokerGrant(
                    broker=inspection.broker,
                    resource_type="GROUP",
                    resource=inspection.group,
                    operation="DESCRIBE",
                )
            )
        return ModelBrokerGrantDerivation(
            principal=request.principal,
            grants=tuple(
                sorted(
                    grants,
                    key=lambda grant: (
                        grant.broker,
                        grant.resource_type,
                        grant.resource,
                        grant.operation,
                    ),
                )
            ),
        )
