# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Errors Module.

This module provides infrastructure-specific error classes and error handling
utilities for the omnibase_infra package. All errors extend from OnexError
to maintain consistency with the ONEX error handling patterns.

Exports:
    ModelInfraErrorContext: Configuration model for bundled error context
    RuntimeHostError: Base infrastructure error class
    HandlerConfigurationError: Handler configuration validation errors
    SecretResolutionError: Secret/credential resolution errors
    InfraConnectionError: Infrastructure connection errors
    InfraTimeoutError: Infrastructure timeout errors
    InfraAuthenticationError: Infrastructure authentication errors
    InfraServiceUnavailableError: Infrastructure service unavailable errors
"""

from omnibase_infra.errors.infra_errors import (
    HandlerConfigurationError,
    InfraAuthenticationError,
    InfraConnectionError,
    InfraServiceUnavailableError,
    InfraTimeoutError,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.errors.model_infra_error_context import ModelInfraErrorContext

__all__: list[str] = [
    # Configuration model
    "ModelInfraErrorContext",
    # Error classes
    "RuntimeHostError",
    "HandlerConfigurationError",
    "SecretResolutionError",
    "InfraConnectionError",
    "InfraTimeoutError",
    "InfraAuthenticationError",
    "InfraServiceUnavailableError",
]
