# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utility modules for ONEX infrastructure.

This package provides common utilities used across the infrastructure:
    - correlation: Correlation ID generation and propagation for distributed tracing
    - util_error_sanitization: Error message sanitization for secure logging and DLQ
    - util_semver: Semantic versioning validation utilities
"""

from omnibase_infra.utils.correlation import (
    CorrelationContext,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)
from omnibase_infra.utils.util_error_sanitization import (
    SENSITIVE_PATTERNS,
    sanitize_error_message,
)
from omnibase_infra.utils.util_semver import (
    SEMVER_PATTERN,
    validate_semver,
    validate_version_lenient,
)

__all__: list[str] = [
    "SEMVER_PATTERN",
    "SENSITIVE_PATTERNS",
    "CorrelationContext",
    "clear_correlation_id",
    "generate_correlation_id",
    "get_correlation_id",
    "sanitize_error_message",
    "set_correlation_id",
    "validate_semver",
    "validate_version_lenient",
]
