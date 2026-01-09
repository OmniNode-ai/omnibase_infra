# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projection Models Module.

Provides Pydantic models for projection storage, ordering, and snapshot
topic configuration. Used by projectors to persist materialized state and
by orchestrators to query current entity state.

Exports:
    ModelCapabilityFields: Container for capability fields in projection persistence
    ModelRegistrationProjection: Registration projection for orchestrator state queries
    ModelRegistrationSnapshot: Compacted snapshot for read optimization
    ModelSequenceInfo: Sequence information for projection ordering and idempotency
    ModelSnapshotTopicConfig: Kafka topic configuration for snapshot publishing

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from omnibase_infra.models.projection.model_capability_fields import (
    ModelCapabilityFields,
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

__all__ = [
    "ModelCapabilityFields",
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
]
