# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utility modules for ONEX infrastructure.

This package provides common utilities used across the infrastructure:
    - correlation: Correlation ID generation and propagation for distributed tracing
    - util_datetime: Datetime validation and timezone normalization
    - util_dsn_validation: PostgreSQL DSN validation and sanitization
    - util_env_parsing: Type-safe environment variable parsing with validation
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
from omnibase_infra.utils.util_datetime import (
    ensure_timezone_aware,
    is_timezone_aware,
)
from omnibase_infra.utils.util_dsn_validation import (
    parse_and_validate_dsn,
    sanitize_dsn,
)
from omnibase_infra.utils.util_env_parsing import (
    parse_env_float,
    parse_env_int,
)
from omnibase_infra.utils.util_error_sanitization import (
    SAFE_ERROR_PATTERNS,
    SENSITIVE_PATTERNS,
    sanitize_backend_error,
    sanitize_error_message,
    sanitize_error_string,
)
from omnibase_infra.utils.util_semver import (
    SEMVER_PATTERN,
    validate_semver,
    validate_version_lenient,
)

__all__: list[str] = [
    "CorrelationContext",
    "SAFE_ERROR_PATTERNS",
    "SEMVER_PATTERN",
    "SENSITIVE_PATTERNS",
    "clear_correlation_id",
    "ensure_timezone_aware",
    "generate_correlation_id",
    "get_correlation_id",
    "is_timezone_aware",
    "parse_and_validate_dsn",
    "parse_env_float",
    "parse_env_int",
    "sanitize_backend_error",
    "sanitize_dsn",
    "sanitize_error_message",
    "sanitize_error_string",
    "set_correlation_id",
    "validate_semver",
    "validate_version_lenient",
]
