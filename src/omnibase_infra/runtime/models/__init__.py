# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Models Module.

This module exports Pydantic models for runtime configuration.
All models are strongly typed to eliminate Any usage.

Exports:
    ModelEventBusConfig: Event bus configuration model
    ModelEnabledProtocolsConfig: Enabled protocols configuration model
    ModelProtocolRegistrationConfig: Individual protocol registration config model
    ModelLoggingConfig: Logging configuration model
    ModelShutdownConfig: Shutdown configuration model
    ModelRuntimeConfig: Top-level runtime configuration model
    ModelOptionalString: Wrapper for optional string values
    ModelOptionalUUID: Wrapper for optional UUID values
    ModelOptionalCorrelationId: Wrapper for optional correlation ID values
    ModelPolicyTypeFilter: Wrapper for policy type filter values
    ModelPolicyRegistration: Policy registration parameters model
"""

from omnibase_infra.runtime.models.model_enabled_protocols_config import (
    ModelEnabledProtocolsConfig,
)
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig
from omnibase_infra.runtime.models.model_logging_config import ModelLoggingConfig
from omnibase_infra.runtime.models.model_optional_correlation_id import (
    ModelOptionalCorrelationId,
)
from omnibase_infra.runtime.models.model_optional_string import ModelOptionalString
from omnibase_infra.runtime.models.model_optional_uuid import ModelOptionalUUID
from omnibase_infra.runtime.models.model_policy_registration import (
    ModelPolicyRegistration,
)
from omnibase_infra.runtime.models.model_policy_type_filter import ModelPolicyTypeFilter
from omnibase_infra.runtime.models.model_protocol_registration_config import (
    ModelProtocolRegistrationConfig,
)
from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig
from omnibase_infra.runtime.models.model_shutdown_config import ModelShutdownConfig

__all__: list[str] = [
    "ModelEventBusConfig",
    "ModelEnabledProtocolsConfig",
    "ModelProtocolRegistrationConfig",
    "ModelLoggingConfig",
    "ModelShutdownConfig",
    "ModelRuntimeConfig",
    "ModelOptionalString",
    "ModelOptionalUUID",
    "ModelOptionalCorrelationId",
    "ModelPolicyTypeFilter",
    "ModelPolicyRegistration",
]
