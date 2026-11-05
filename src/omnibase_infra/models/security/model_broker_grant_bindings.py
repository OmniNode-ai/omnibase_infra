# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved bus bindings plus the consumer group prefixes a principal reads under."""

from pydantic import Field

from omnibase_core.models.event_bus.model_resolved_bus_bindings import (
    ModelResolvedBusBindings,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_group_prefix import (
    ModelBrokerGroupPrefix,
)


class ModelBrokerGrantBindings(ModelResolvedBusBindings):
    """A principal's wire bindings, with prefixed consumer groups stated beside them."""

    group_prefixes: tuple[ModelBrokerGroupPrefix, ...] = Field(
        default=(),
        description="Group prefixes granted READ and DESCRIBE; a consumed group "
        "under one of them needs no exact group grant.",
    )
