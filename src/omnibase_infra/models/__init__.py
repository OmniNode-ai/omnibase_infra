# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ONEX Infrastructure Models.

This module exports all infrastructure-specific Pydantic models.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.models.bindings import (
        ModelParsedBinding,
    )
    from omnibase_infra.models.catalog import (
        ModelTopicCatalogChanged,
        ModelTopicCatalogEntry,
        ModelTopicCatalogQuery,
        ModelTopicCatalogResponse,
    )
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
    from omnibase_infra.models.event_bus import (
        ModelConsumerRetryConfig,
        ModelIdempotencyConfig,
        ModelOffsetPolicyConfig,
    )
    from omnibase_infra.models.handlers import ModelHandlerIdentifier
    from omnibase_infra.models.health import ModelHealthCheckResult
    from omnibase_infra.models.ledger import (
        ModelDbQueryFailed,
        ModelDbQueryRequested,
        ModelDbQuerySucceeded,
        ModelLedgerEventBase,
    )
    from omnibase_infra.models.logging import ModelLogContext
    from omnibase_infra.models.model_backend_result import ModelBackendResult
    from omnibase_infra.models.model_node_identity import ModelNodeIdentity
    from omnibase_infra.models.model_retry_error_classification import (
        ModelRetryErrorClassification,
    )
    from omnibase_infra.models.pricing import (
        ModelCostEstimate,
        ModelPricingEntry,
        ModelPricingTable,
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
    from omnibase_infra.models.projectors import (
        ModelProjectorColumn,
        ModelProjectorIndex,
        ModelProjectorSchema,
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
    from omnibase_infra.models.routing import (
        ModelRoutingEntry,
        ModelRoutingSubcontract,
    )
    from omnibase_infra.models.rrh import (
        ModelRRHEnvironmentData,
        ModelRRHProfile,
        ModelRRHRepoState,
        ModelRRHResult,
        ModelRRHRuleSeverity,
        ModelRRHRuntimeTarget,
        ModelRRHToolchainVersions,
    )
    from omnibase_infra.models.runtime import ModelLoadedHandler
    from omnibase_infra.models.security import (
        ModelEnvironmentPolicy,
        ModelHandlerSecurityPolicy,
    )
    from omnibase_infra.models.snapshot import (
        ModelFieldChange,
        ModelSnapshot,
        ModelSnapshotDiff,
        ModelSubjectRef,
    )
    from omnibase_infra.models.validation import (
        ModelCoverageMetrics,
        ModelExecutionShapeRule,
        ModelExecutionShapeViolationResult,
        ModelValidationOutcome,
    )

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "EnumDispatchStatus": "omnibase_infra.models.dispatch",
    "EnumTopicStandard": "omnibase_infra.models.dispatch",
    "ModelBackendResult": "omnibase_infra.models.model_backend_result",
    "ModelCircuitBreakerConfig": "omnibase_infra.models.resilience",
    "ModelConsumerRetryConfig": "omnibase_infra.models.event_bus",
    "ModelCostEstimate": "omnibase_infra.models.pricing",
    "ModelCoverageMetrics": "omnibase_infra.models.validation",
    "ModelDbQueryFailed": "omnibase_infra.models.ledger",
    "ModelDbQueryRequested": "omnibase_infra.models.ledger",
    "ModelDbQuerySucceeded": "omnibase_infra.models.ledger",
    "ModelDispatchLogContext": "omnibase_infra.models.dispatch",
    "ModelDispatchMetrics": "omnibase_infra.models.dispatch",
    "ModelDispatchOutcome": "omnibase_infra.models.dispatch",
    "ModelDispatchResult": "omnibase_infra.models.dispatch",
    "ModelDispatchRoute": "omnibase_infra.models.dispatch",
    "ModelDispatcherMetrics": "omnibase_infra.models.dispatch",
    "ModelDispatcherRegistration": "omnibase_infra.models.dispatch",
    "ModelEnvironmentPolicy": "omnibase_infra.models.security",
    "ModelExecutionShapeRule": "omnibase_infra.models.validation",
    "ModelExecutionShapeViolationResult": "omnibase_infra.models.validation",
    "ModelFieldChange": "omnibase_infra.models.snapshot",
    "ModelHandlerIdentifier": "omnibase_infra.models.handlers",
    "ModelHandlerSecurityPolicy": "omnibase_infra.models.security",
    "ModelHandlerValidationError": "omnibase_infra.models.errors",
    "ModelHealthCheckResult": "omnibase_infra.models.health",
    "ModelIdempotencyConfig": "omnibase_infra.models.event_bus",
    "ModelIntrospectionMetrics": "omnibase_infra.models.registration",
    "ModelLedgerEventBase": "omnibase_infra.models.ledger",
    "ModelLoadedHandler": "omnibase_infra.models.runtime",
    "ModelLogContext": "omnibase_infra.models.logging",
    "ModelNodeCapabilities": "omnibase_infra.models.registration",
    "ModelNodeHeartbeatEvent": "omnibase_infra.models.registration",
    "ModelNodeIdentity": "omnibase_infra.models.model_node_identity",
    "ModelNodeIntrospectionEvent": "omnibase_infra.models.registration",
    "ModelNodeMetadata": "omnibase_infra.models.registration",
    "ModelNodeRegistration": "omnibase_infra.models.registration",
    "ModelOffsetPolicyConfig": "omnibase_infra.models.event_bus",
    "ModelParsedBinding": "omnibase_infra.models.bindings",
    "ModelParsedTopic": "omnibase_infra.models.dispatch",
    "ModelPricingEntry": "omnibase_infra.models.pricing",
    "ModelPricingTable": "omnibase_infra.models.pricing",
    "ModelProjectorColumn": "omnibase_infra.models.projectors",
    "ModelProjectorIndex": "omnibase_infra.models.projectors",
    "ModelProjectorSchema": "omnibase_infra.models.projectors",
    "ModelRRHEnvironmentData": "omnibase_infra.models.rrh",
    "ModelRRHProfile": "omnibase_infra.models.rrh",
    "ModelRRHRepoState": "omnibase_infra.models.rrh",
    "ModelRRHResult": "omnibase_infra.models.rrh",
    "ModelRRHRuleSeverity": "omnibase_infra.models.rrh",
    "ModelRRHRuntimeTarget": "omnibase_infra.models.rrh",
    "ModelRRHToolchainVersions": "omnibase_infra.models.rrh",
    "ModelRegistrationProjection": "omnibase_infra.models.projection",
    "ModelRegistrationSnapshot": "omnibase_infra.models.projection",
    "ModelRetryErrorClassification": "omnibase_infra.models.model_retry_error_classification",
    "ModelRoutingEntry": "omnibase_infra.models.routing",
    "ModelRoutingSubcontract": "omnibase_infra.models.routing",
    "ModelSequenceInfo": "omnibase_infra.models.projection",
    "ModelSnapshot": "omnibase_infra.models.snapshot",
    "ModelSnapshotDiff": "omnibase_infra.models.snapshot",
    "ModelSnapshotTopicConfig": "omnibase_infra.models.projection",
    "ModelSubjectRef": "omnibase_infra.models.snapshot",
    "ModelTopicCatalogChanged": "omnibase_infra.models.catalog",
    "ModelTopicCatalogEntry": "omnibase_infra.models.catalog",
    "ModelTopicCatalogQuery": "omnibase_infra.models.catalog",
    "ModelTopicCatalogResponse": "omnibase_infra.models.catalog",
    "ModelTopicParser": "omnibase_infra.models.dispatch",
    "ModelValidationOutcome": "omnibase_infra.models.validation",
}

__all__: list[str] = [
    # Binding models
    "ModelParsedBinding",
    # Catalog models
    "ModelTopicCatalogChanged",
    "ModelTopicCatalogEntry",
    "ModelTopicCatalogQuery",
    "ModelTopicCatalogResponse",
    # Dispatch models
    "EnumDispatchStatus",
    "EnumTopicStandard",
    # Event bus models
    "ModelConsumerRetryConfig",
    "ModelIdempotencyConfig",
    "ModelOffsetPolicyConfig",
    # Backend result models
    "ModelBackendResult",
    # Resilience models
    "ModelCircuitBreakerConfig",
    # RRH models
    "ModelRRHEnvironmentData",
    "ModelRRHProfile",
    "ModelRRHRepoState",
    "ModelRRHResult",
    "ModelRRHRuleSeverity",
    "ModelRRHRuntimeTarget",
    "ModelRRHToolchainVersions",
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
    # Ledger event models
    "ModelDbQueryFailed",
    "ModelDbQueryRequested",
    "ModelDbQuerySucceeded",
    "ModelLedgerEventBase",
    # Routing models
    "ModelRoutingEntry",
    "ModelRoutingSubcontract",
    # Health models
    "ModelHealthCheckResult",
    # Registration models
    "ModelIntrospectionMetrics",
    # Runtime models
    "ModelLoadedHandler",
    # Logging models
    "ModelLogContext",
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    # Node identity model
    "ModelNodeIdentity",
    "ModelNodeIntrospectionEvent",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    "ModelParsedTopic",
    # Pricing models
    "ModelCostEstimate",
    "ModelPricingEntry",
    "ModelPricingTable",
    # Projection models
    "ModelRegistrationProjection",
    # Projector schema models
    "ModelProjectorColumn",
    "ModelProjectorIndex",
    "ModelProjectorSchema",
    "ModelRegistrationSnapshot",
    # Retry models
    "ModelRetryErrorClassification",
    # Security models
    "ModelEnvironmentPolicy",
    "ModelHandlerSecurityPolicy",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    "ModelTopicParser",
    "ModelValidationOutcome",
    # Snapshot models
    "ModelFieldChange",
    "ModelSnapshot",
    "ModelSnapshotDiff",
    "ModelSubjectRef",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value: object = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
