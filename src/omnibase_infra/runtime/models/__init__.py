# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Models Module.

This module exports Pydantic models for runtime configuration and events.
All models are strongly typed to eliminate Any usage.

Exports:
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
    ModelPolicyRegistration: Policy registration parameters model
    ModelPolicyKey: Strongly-typed policy registry key model
    ModelRuntimeTick: Infrastructure event emitted by runtime scheduler
    ModelDuplicateResponse: Response for duplicate message detection
    ModelLifecycleResult: Result of individual handler lifecycle operation
    ModelBatchLifecycleResult: Result of batch handler lifecycle operations
    ModelHandlerShutdownResult: Result of handler shutdown operation (legacy)
    ModelHandlerHealthCheckResult: Result of handler health check operation
"""

from omnibase_infra.runtime.models.model_batch_lifecycle_result import (
    ModelBatchLifecycleResult,
)
from omnibase_infra.runtime.models.model_duplicate_response import (
    ModelDuplicateResponse,
)
from omnibase_infra.runtime.models.model_enabled_protocols_config import (
    ModelEnabledProtocolsConfig,
)
from omnibase_infra.runtime.models.model_handler_health_check_result import (
    ModelHandlerHealthCheckResult,
)
from omnibase_infra.runtime.models.model_handler_shutdown_result import (
    ModelHandlerShutdownResult,
)
from omnibase_infra.runtime.models.model_lifecycle_result import (
    ModelLifecycleResult,
)
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig
from omnibase_infra.runtime.models.model_logging_config import ModelLoggingConfig
from omnibase_infra.runtime.models.model_optional_correlation_id import (
    ModelOptionalCorrelationId,
)
from omnibase_infra.runtime.models.model_optional_string import ModelOptionalString
from omnibase_infra.runtime.models.model_optional_uuid import ModelOptionalUUID
from omnibase_infra.runtime.models.model_policy_key import ModelPolicyKey
from omnibase_infra.runtime.models.model_policy_registration import (
    ModelPolicyRegistration,
)
from omnibase_infra.runtime.models.model_policy_type_filter import ModelPolicyTypeFilter
from omnibase_infra.runtime.models.model_protocol_registration_config import (
    ModelProtocolRegistrationConfig,
)
from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig
from omnibase_infra.runtime.models.model_runtime_scheduler_config import (
    ModelRuntimeSchedulerConfig,
)
from omnibase_infra.runtime.models.model_runtime_scheduler_metrics import (
    ModelRuntimeSchedulerMetrics,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.runtime.models.model_shutdown_config import ModelShutdownConfig

__all__: list[str] = [
    "ModelBatchLifecycleResult",
    "ModelDuplicateResponse",
    "ModelEventBusConfig",
    "ModelEnabledProtocolsConfig",
    "ModelHandlerHealthCheckResult",
    "ModelHandlerShutdownResult",
    "ModelLifecycleResult",
    "ModelProtocolRegistrationConfig",
    "ModelLoggingConfig",
    "ModelShutdownConfig",
    "ModelRuntimeConfig",
    "ModelRuntimeSchedulerConfig",
    "ModelRuntimeSchedulerMetrics",
    "ModelRuntimeTick",
    "ModelOptionalString",
    "ModelOptionalUUID",
    "ModelOptionalCorrelationId",
    "ModelPolicyTypeFilter",
    "ModelPolicyRegistration",
    "ModelPolicyKey",
]
