# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection contract registry for existing CQRS projection handlers (OMN-11199).

Declares ModelProjectionContract instances for every materialized projection
maintained by omnibase_infra, providing the runtime with freshness SLAs,
cursor mechanisms, and degraded-behaviour semantics.

ModelProjectionContract is defined in omnibase_core PR #1100 (OMN-11192).
"""

from __future__ import annotations

from omnibase_core.enums.enum_degraded_behavior import EnumDegradedBehavior
from omnibase_core.models.projection.model_cursor_contract import ModelCursorContract
from omnibase_core.models.projection.model_projection_contract import (
    ModelProjectionContract,
)
from omnibase_infra.enums.generated.enum_platform_topic import EnumPlatformTopic

_KAFKA_OFFSET_CURSOR = ModelCursorContract(
    cursor_type="kafka_offset",
    supports_replay=True,
)

CONTRACT_REGISTRY_PROJECTION = ModelProjectionContract(
    projection_name="contract_registry",
    source_topics=(
        EnumPlatformTopic.EVT_CONTRACT_REGISTERED_V1.value,
        EnumPlatformTopic.EVT_CONTRACT_DEREGISTERED_V1.value,
        EnumPlatformTopic.EVT_NODE_HEARTBEAT_V1.value,
    ),
    schema_model=(
        "omnibase_infra.models.projection.model_contract_projection.ModelContractProjection"
    ),
    freshness_sla_seconds=30,
    freshness_field="last_seen_at",
    freshness_source_table="contracts",
    degraded_semantics=EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
    cursor=_KAFKA_OFFSET_CURSOR,
    ordering_contract_ref=None,
)

TOPIC_REGISTRY_PROJECTION = ModelProjectionContract(
    projection_name="topic_registry",
    source_topics=(
        EnumPlatformTopic.EVT_CONTRACT_REGISTERED_V1.value,
        EnumPlatformTopic.EVT_CONTRACT_DEREGISTERED_V1.value,
    ),
    schema_model=(
        "omnibase_infra.models.projection.model_topic_projection.ModelTopicProjection"
    ),
    freshness_sla_seconds=30,
    freshness_field="last_seen_at",
    freshness_source_table="topics",
    degraded_semantics=EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
    cursor=_KAFKA_OFFSET_CURSOR,
    ordering_contract_ref=None,
)

REGISTRATION_PROJECTION = ModelProjectionContract(
    projection_name="registration",
    source_topics=(
        EnumPlatformTopic.EVT_NODE_REGISTRATION_V1.value,
        EnumPlatformTopic.EVT_NODE_HEARTBEAT_V1.value,
        EnumPlatformTopic.EVT_NODE_LIVENESS_EXPIRED_V1.value,
        EnumPlatformTopic.EVT_NODE_REGISTRATION_ACCEPTED_V1.value,
        EnumPlatformTopic.EVT_NODE_REGISTRATION_ACK_RECEIVED_V1.value,
        EnumPlatformTopic.EVT_NODE_REGISTRATION_INITIATED_V1.value,
        EnumPlatformTopic.EVT_NODE_REGISTRATION_REJECTED_V1.value,
        EnumPlatformTopic.EVT_NODE_REGISTRATION_RESULT_V1.value,
    ),
    schema_model=(
        "omnibase_infra.models.projection.model_registration_projection"
        ".ModelRegistrationProjection"
    ),
    freshness_sla_seconds=60,
    freshness_field="updated_at",
    freshness_source_table="registration_projections",
    degraded_semantics=EnumDegradedBehavior.SERVE_STALE_WITH_WARNING,
    cursor=_KAFKA_OFFSET_CURSOR,
    ordering_contract_ref=None,
)

PROJECTION_CONTRACTS: tuple[ModelProjectionContract, ...] = (
    CONTRACT_REGISTRY_PROJECTION,
    TOPIC_REGISTRY_PROJECTION,
    REGISTRATION_PROJECTION,
)


def get_projection_contract(name: str) -> ModelProjectionContract | None:
    """Return the registered projection contract by name, or None if not found."""
    for contract in PROJECTION_CONTRACTS:
        if contract.projection_name == name:
            return contract
    return None


__all__ = [
    "CONTRACT_REGISTRY_PROJECTION",
    "PROJECTION_CONTRACTS",
    "REGISTRATION_PROJECTION",
    "TOPIC_REGISTRY_PROJECTION",
    "get_projection_contract",
]
