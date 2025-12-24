# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Models.

This module exports all infrastructure-specific Pydantic models.
"""

from omnibase_infra.models.dispatch import (
    EnumDispatchStatus,
    EnumTopicStandard,
    ModelDispatcherMetrics,
    ModelDispatcherRegistration,
    ModelDispatchMetrics,
    ModelDispatchResult,
    ModelDispatchRoute,
    ModelParsedTopic,
    ModelTopicParser,
)
from omnibase_infra.models.health import ModelHealthCheckResult
from omnibase_infra.models.model_semver import SEMVER_DEFAULT, ModelSemVer
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelRegistrationSnapshot,
    ModelSequenceInfo,
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeHeartbeatEvent,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
    ModelNodeRegistration,
)
from omnibase_infra.models.validation import (
    ModelExecutionShapeRule,
    ModelExecutionShapeViolationResult,
)

__all__ = [
    # Dispatch models
    "EnumDispatchStatus",
    "EnumTopicStandard",
    "ModelDispatchMetrics",
    "ModelDispatchResult",
    "ModelDispatchRoute",
    "ModelDispatcherMetrics",
    "ModelDispatcherRegistration",
    "ModelParsedTopic",
    "ModelTopicParser",
    # Health models
    "ModelHealthCheckResult",
    # Projection models
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    # Registration models
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    "ModelNodeIntrospectionEvent",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    # SemVer models
    "ModelSemVer",
    "SEMVER_DEFAULT",
    # Validation models
    "ModelExecutionShapeRule",
    "ModelExecutionShapeViolationResult",
]
