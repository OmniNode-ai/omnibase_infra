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
    ModelDispatchLogContext,
    ModelDispatchMetrics,
    ModelDispatchOutcome,
    ModelDispatchResult,
    ModelDispatchRoute,
    ModelParsedTopic,
    ModelTopicParser,
)
from omnibase_infra.models.health import ModelHealthCheckResult
from omnibase_infra.models.logging import ModelLogContext
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelRegistrationSnapshot,
    ModelSequenceInfo,
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.registration import (
    ModelIntrospectionMetrics,
    ModelNodeCapabilities,
    ModelNodeHeartbeatEvent,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
    ModelNodeRegistration,
)
from omnibase_infra.models.validation import (
    ModelCoverageMetrics,
    ModelExecutionShapeRule,
    ModelExecutionShapeViolationResult,
    ModelValidationOutcome,
)

__all__ = [
    # Dispatch models
    "EnumDispatchStatus",
    "EnumTopicStandard",
    "ModelDispatchLogContext",
    "ModelDispatchMetrics",
    "ModelDispatchOutcome",
    "ModelDispatchResult",
    "ModelDispatchRoute",
    "ModelDispatcherMetrics",
    "ModelDispatcherRegistration",
    "ModelParsedTopic",
    "ModelTopicParser",
    # Health models
    "ModelHealthCheckResult",
    # Logging models
    "ModelLogContext",
    # Projection models
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    # Registration models
    "ModelIntrospectionMetrics",
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    "ModelNodeIntrospectionEvent",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    # Validation models
    "ModelCoverageMetrics",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeViolationResult",
    "ModelValidationOutcome",
]
