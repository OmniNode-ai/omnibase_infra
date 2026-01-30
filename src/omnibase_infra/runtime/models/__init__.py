# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Models Module.

This module exports Pydantic models for runtime configuration and events.
All models are strongly typed to eliminate Any usage.

Exports:
    ModelComputeKey: Strongly-typed compute registry key model
    ModelComputeRegistration: Compute plugin registration parameters model
    ModelConfigRef: Parsed configuration reference (file, env, vault)
    ModelConfigRefParseResult: Result of parsing a config reference
    EnumConfigRefScheme: Supported config reference schemes
    ModelDomainPluginConfig: Configuration for domain plugin lifecycle hooks
    ModelDomainPluginResult: Result of domain plugin lifecycle operations
    ModelEventBusConfig: Event bus configuration model
    ModelEnabledProtocolsConfig: Enabled protocols configuration model
    ModelProtocolRegistrationConfig: Individual protocol registration config model
    ModelLoggingConfig: Logging configuration model
    ModelShutdownConfig: Shutdown configuration model
    ModelRuntimeConfig: Top-level runtime configuration model
    ModelRuntimeSchedulerConfig: Runtime tick scheduler configuration model
    ModelRuntimeSchedulerMetrics: Runtime scheduler metrics model
    ModelOptionalString: Wrapper for optional string values
    ModelOptionalUUID: Wrapper for optional UUID values
    ModelOptionalCorrelationId: Wrapper for optional correlation ID values
    ModelPolicyTypeFilter: Wrapper for policy type filter values
    ModelPolicyContext: Structured context model for policy evaluation
    ModelPolicyResult: Structured result model for policy decisions
    ModelPolicyRegistration: Policy registration parameters model
    ModelPolicyKey: Strongly-typed policy registry key model
    ModelRuntimeTick: Infrastructure event emitted by runtime scheduler
    ModelDuplicateResponse: Response for duplicate message detection
    ModelLifecycleResult: Result of individual handler lifecycle operation
    ModelBatchLifecycleResult: Result of batch handler lifecycle operations
    ModelFailedComponent: Represents a component that failed during shutdown
    ModelShutdownBatchResult: Result of batch shutdown operation
    ModelHealthCheckResult: Result of component health check operation
    ModelHealthCheckResponse: HTTP response model for health check endpoints
    ModelProjectorPluginLoaderConfig: Projector plugin loader configuration model
    ModelSecretSourceSpec: Source specification for a single secret
    SecretSourceType: Type alias for secret source types (env, vault, file)
    ModelSecretMapping: Mapping from logical name to source specification
    ModelSecretResolverConfig: Configuration for SecretResolver
    ModelCachedSecret: Cached secret with TTL tracking
    ModelSecretCacheStats: Cache statistics for observability
    ModelSecretResolverMetrics: Resolution metrics for observability
    ModelSecretSourceInfo: Non-sensitive source information for introspection
    ModelRetryPolicy: Retry policy configuration for handler operations
    ModelTransitionNotificationPublisherMetrics: Metrics for transition notification publisher
    ModelTransitionNotificationOutboxMetrics: Metrics for transition notification outbox
    ModelTransitionNotificationOutboxConfig: Configuration for transition notification outbox
    ModelStateTransitionNotification: State transition notification (re-export from omnibase_core)
    ModelProjectorNotificationConfig: Configuration for projector notification publishing
    ModelBindingConfig: Configuration for binding a handler to the runtime
    ModelBindingConfigCacheStats: Cache statistics for BindingConfigResolver
    ModelBindingConfigResolverConfig: Configuration for BindingConfigResolver
    ModelConfigCacheEntry: Internal cache entry for BindingConfigResolver
    ModelSecurityConfig: Security configuration for handler namespace allowlisting
"""

# Re-export from omnibase_core for convenience
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_infra.runtime.enums.enum_config_ref_scheme import EnumConfigRefScheme
from omnibase_infra.runtime.models.model_batch_lifecycle_result import (
    ModelBatchLifecycleResult,
)
from omnibase_infra.runtime.models.model_binding_config import ModelBindingConfig
from omnibase_infra.runtime.models.model_binding_config_cache_stats import (
    ModelBindingConfigCacheStats,
)
from omnibase_infra.runtime.models.model_binding_config_resolver_config import (
    ModelBindingConfigResolverConfig,
)
from omnibase_infra.runtime.models.model_cached_secret import ModelCachedSecret
from omnibase_infra.runtime.models.model_compute_key import ModelComputeKey
from omnibase_infra.runtime.models.model_compute_registration import (
    ModelComputeRegistration,
)
from omnibase_infra.runtime.models.model_config_cache_entry import ModelConfigCacheEntry
from omnibase_infra.runtime.models.model_config_ref import ModelConfigRef
from omnibase_infra.runtime.models.model_config_ref_parse_result import (
    ModelConfigRefParseResult,
)
from omnibase_infra.runtime.models.model_contract_load_result import (
    ModelContractLoadResult,
)
from omnibase_infra.runtime.models.model_domain_plugin_config import (
    ModelDomainPluginConfig,
)
from omnibase_infra.runtime.models.model_domain_plugin_result import (
    ModelDomainPluginResult,
)
from omnibase_infra.runtime.models.model_duplicate_response import (
    ModelDuplicateResponse,
)
from omnibase_infra.runtime.models.model_enabled_protocols_config import (
    ModelEnabledProtocolsConfig,
)
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig
from omnibase_infra.runtime.models.model_failed_component import ModelFailedComponent
from omnibase_infra.runtime.models.model_health_check_response import (
    ModelHealthCheckResponse,
)
from omnibase_infra.runtime.models.model_health_check_result import (
    ModelHealthCheckResult,
)
from omnibase_infra.runtime.models.model_lifecycle_result import (
    ModelLifecycleResult,
)
from omnibase_infra.runtime.models.model_logging_config import ModelLoggingConfig
from omnibase_infra.runtime.models.model_optional_correlation_id import (
    ModelOptionalCorrelationId,
)
from omnibase_infra.runtime.models.model_optional_string import ModelOptionalString
from omnibase_infra.runtime.models.model_optional_uuid import ModelOptionalUUID
from omnibase_infra.runtime.models.model_policy_context import ModelPolicyContext
from omnibase_infra.runtime.models.model_policy_key import ModelPolicyKey
from omnibase_infra.runtime.models.model_policy_registration import (
    ModelPolicyRegistration,
)
from omnibase_infra.runtime.models.model_policy_result import ModelPolicyResult
from omnibase_infra.runtime.models.model_policy_type_filter import ModelPolicyTypeFilter
from omnibase_infra.runtime.models.model_projector_notification_config import (
    ModelProjectorNotificationConfig,
)
from omnibase_infra.runtime.models.model_projector_plugin_loader_config import (
    ModelProjectorPluginLoaderConfig,
)
from omnibase_infra.runtime.models.model_protocol_registration_config import (
    ModelProtocolRegistrationConfig,
)
from omnibase_infra.runtime.models.model_retry_policy import ModelRetryPolicy
from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig
from omnibase_infra.runtime.models.model_runtime_contract_config import (
    ModelRuntimeContractConfig,
)
from omnibase_infra.runtime.models.model_runtime_scheduler_config import (
    ModelRuntimeSchedulerConfig,
)
from omnibase_infra.runtime.models.model_runtime_scheduler_metrics import (
    ModelRuntimeSchedulerMetrics,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.runtime.models.model_secret_cache_stats import ModelSecretCacheStats
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_resolver_metrics import (
    ModelSecretResolverMetrics,
)
from omnibase_infra.runtime.models.model_secret_source_info import ModelSecretSourceInfo
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
    SecretSourceType,
)
from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig
from omnibase_infra.runtime.models.model_shutdown_batch_result import (
    ModelShutdownBatchResult,
)
from omnibase_infra.runtime.models.model_shutdown_config import ModelShutdownConfig
from omnibase_infra.runtime.models.model_transition_notification_outbox_config import (
    ModelTransitionNotificationOutboxConfig,
)
from omnibase_infra.runtime.models.model_transition_notification_outbox_metrics import (
    ModelTransitionNotificationOutboxMetrics,
)
from omnibase_infra.runtime.models.model_transition_notification_publisher_metrics import (
    ModelTransitionNotificationPublisherMetrics,
)

__all__: list[str] = [
    "EnumConfigRefScheme",
    "ModelBatchLifecycleResult",
    "ModelBindingConfig",
    "ModelBindingConfigCacheStats",
    "ModelBindingConfigResolverConfig",
    "ModelCachedSecret",
    "ModelComputeKey",
    "ModelComputeRegistration",
    "ModelConfigCacheEntry",
    "ModelConfigRef",
    "ModelConfigRefParseResult",
    "ModelDomainPluginConfig",
    "ModelDomainPluginResult",
    "ModelDuplicateResponse",
    "ModelEnabledProtocolsConfig",
    "ModelEventBusConfig",
    "ModelFailedComponent",
    "ModelHealthCheckResponse",
    "ModelHealthCheckResult",
    "ModelLifecycleResult",
    "ModelLoggingConfig",
    "ModelOptionalCorrelationId",
    "ModelOptionalString",
    "ModelOptionalUUID",
    "ModelPolicyContext",
    "ModelPolicyKey",
    "ModelPolicyRegistration",
    "ModelPolicyResult",
    "ModelPolicyTypeFilter",
    "ModelProjectorNotificationConfig",
    "ModelProjectorPluginLoaderConfig",
    "ModelProtocolRegistrationConfig",
    "ModelRetryPolicy",
    "ModelRuntimeConfig",
    "ModelRuntimeSchedulerConfig",
    "ModelRuntimeSchedulerMetrics",
    "ModelRuntimeTick",
    "ModelSecretCacheStats",
    "ModelSecretMapping",
    "ModelSecretResolverConfig",
    "ModelSecretResolverMetrics",
    "ModelSecretSourceInfo",
    "ModelSecretSourceSpec",
    "ModelShutdownBatchResult",
    "ModelShutdownConfig",
    "ModelStateTransitionNotification",
    "ModelTransitionNotificationOutboxConfig",
    "ModelTransitionNotificationOutboxMetrics",
    "ModelTransitionNotificationPublisherMetrics",
    "SecretSourceType",
    # Contract loading models (OMN-1519)
    "ModelContractLoadResult",
    "ModelRuntimeContractConfig",
    # Security configuration (OMN-1519)
    "ModelSecurityConfig",
]
