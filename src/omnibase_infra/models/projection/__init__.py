# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projection Models Module.

Provides Pydantic models for projection storage, ordering, and snapshot
topic configuration. Used by projectors to persist materialized state and
by orchestrators to query current entity state.

Exports:
    ModelCapabilityFields: Container for capability fields in projection persistence
    ModelContractProjection: Contract projection for Registry API queries
    ModelProjectionIntent: Intent emitted by reducer to trigger synchronous projection (OMN-2510)
    ModelRegistrationProjection: Registration projection for orchestrator state queries
    ModelRegistrationSnapshot: Compacted snapshot for read optimization
    ModelSequenceInfo: Sequence information for projection ordering and idempotency
    ModelSnapshotTopicConfig: Kafka topic configuration for snapshot publishing
    ModelTopicProjection: Topic projection for Registry API queries

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-2510: Runtime wires NodeProjectionEffect before Kafka publish
"""

from omnibase_infra.models.projection.model_capability_fields import (
    ModelCapabilityFields,
)
from omnibase_infra.models.projection.model_contract_projection import (
    ModelContractProjection,
)
from omnibase_infra.models.projection.model_projection_intent import (
    ModelProjectionIntent,
)
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.projection.model_registration_snapshot import (
    ModelRegistrationSnapshot,
)
from omnibase_infra.models.projection.model_sequence_info import ModelSequenceInfo
from omnibase_infra.models.projection.model_snapshot_topic_config import (
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.projection.model_topic_projection import (
    ModelTopicProjection,
)

__all__ = [
    "ModelCapabilityFields",
    "ModelContractProjection",
    "ModelProjectionIntent",
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    "ModelTopicProjection",
]
