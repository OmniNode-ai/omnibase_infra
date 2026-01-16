# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Error Models Module.

This module provides Pydantic models for error context and metadata
in the ONEX infrastructure layer.

Models:
    - ModelInfraErrorContext: Context for infrastructure errors including
      transport type, operation, and correlation ID (all fields optional)
    - ModelMessageTypeRegistryErrorContext: Context for message type registry errors
      including message_type, domain, and category
    - ModelTimeoutErrorContext: Stricter context for timeout errors with
      required correlation_id, transport_type, and operation
    - ModelHandlerValidationError: Structured error model for handler
      validation failures (contract, descriptor, security, architecture)

Related:
    - errors.error_infra: Error classes that use these models
    - OMN-927: Infrastructure error patterns
    - OMN-937: Central Message Type Registry implementation
    - OMN-1091: Structured validation and error reporting for handlers

.. versionadded:: 0.5.0
"""

from omnibase_infra.models.errors.model_handler_validation_error import (
    ModelHandlerValidationError,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.errors.model_message_type_registry_error_context import (
    ModelMessageTypeRegistryErrorContext,
)
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)

__all__: list[str] = [
    "ModelHandlerValidationError",
    "ModelInfraErrorContext",
    "ModelMessageTypeRegistryErrorContext",
    "ModelTimeoutErrorContext",
]
