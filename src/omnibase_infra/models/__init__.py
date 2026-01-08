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
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.models.health import ModelHealthCheckResult
from omnibase_infra.models.logging import ModelLogContext
from omnibase_infra.models.model_retry_error_classification import (
    ModelRetryErrorClassification,
)

# ModelSemVer and SEMVER_DEFAULT must be imported from omnibase_core.models.primitives.model_semver
# The local model_semver.py has been REMOVED and raises ImportError on import.
# Import directly from omnibase_core:
#   from omnibase_core.models.primitives.model_semver import ModelSemVer
# To create SEMVER_DEFAULT:
#   SEMVER_DEFAULT = ModelSemVer.parse("1.0.0")
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
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
from omnibase_infra.models.validation import (
    ModelCoverageMetrics,
    ModelExecutionShapeRule,
    ModelExecutionShapeViolationResult,
    ModelValidationOutcome,
)

__all__: list[str] = [
    # Dispatch models
    "EnumDispatchStatus",
    "EnumTopicStandard",
    # Resilience models
    "ModelCircuitBreakerConfig",
    # SemVer models - REMOVED: Use omnibase_core.models.primitives.model_semver instead
    # (model_semver.py now raises ImportError on import)
    # Validation models
    "ModelCoverageMetrics",
    "ModelDispatchLogContext",
    "ModelDispatchMetrics",
    "ModelDispatchOutcome",
    "ModelDispatchResult",
    "ModelDispatchRoute",
    "ModelDispatcherMetrics",
    "ModelDispatcherRegistration",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeViolationResult",
    # Error models
    "ModelHandlerValidationError",
    # Handler models
    "ModelHandlerIdentifier",
    # Health models
    "ModelHealthCheckResult",
    # Registration models
    "ModelIntrospectionMetrics",
    # Logging models
    "ModelLogContext",
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    "ModelNodeIntrospectionEvent",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    "ModelParsedTopic",
    # Projection models
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    # Retry models
    "ModelRetryErrorClassification",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    "ModelTopicParser",
    "ModelValidationOutcome",
]
