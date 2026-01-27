# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utility modules for ONEX infrastructure.

This package provides common utilities used across the infrastructure:
    - correlation: Correlation ID generation and propagation for distributed tracing
    - util_atomic_file: Atomic file write primitives using temp-file-rename pattern
    - util_consumer_group: Kafka consumer group ID generation with deterministic hashing
    - util_datetime: Datetime validation and timezone normalization
    - util_db_transaction: Database transaction context manager for asyncpg
    - util_dsn_validation: PostgreSQL DSN validation and sanitization
    - util_env_parsing: Type-safe environment variable parsing with validation
    - util_error_sanitization: Error message sanitization for secure logging and DLQ
    - util_pydantic_validators: Shared Pydantic field validator utilities
    - util_retry_optimistic: Optimistic locking retry helper with exponential backoff
    - util_semver: Semantic versioning validation utilities
"""

from omnibase_infra.utils.correlation import (
    CorrelationContext,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)
from omnibase_infra.utils.util_atomic_file import (
    write_atomic_bytes,
    write_atomic_bytes_async,
)
from omnibase_infra.utils.util_consumer_group import (
    KAFKA_CONSUMER_GROUP_MAX_LENGTH,
    compute_consumer_group_id,
    normalize_kafka_identifier,
)
from omnibase_infra.utils.util_datetime import (
    ensure_timezone_aware,
    is_timezone_aware,
    validate_timezone_aware_with_context,
    warn_if_naive_datetime,
)
from omnibase_infra.utils.util_db_transaction import (
    transaction_context,
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
    sanitize_consul_key,
    sanitize_error_message,
    sanitize_error_string,
    sanitize_secret_path,
)
from omnibase_infra.utils.util_pydantic_validators import (
    validate_contract_type_value,
    validate_endpoint_urls_dict,
    validate_policy_type_value,
    validate_pool_sizes_constraint,
    validate_timezone_aware_datetime,
    validate_timezone_aware_datetime_optional,
)
from omnibase_infra.utils.util_retry_optimistic import (
    OptimisticConflictError,
    retry_on_optimistic_conflict,
)
from omnibase_infra.utils.util_semver import (
    SEMVER_PATTERN,
    validate_semver,
    validate_version_lenient,
)

__all__: list[str] = [
    "CorrelationContext",
    "KAFKA_CONSUMER_GROUP_MAX_LENGTH",
    "OptimisticConflictError",
    "SAFE_ERROR_PATTERNS",
    "SEMVER_PATTERN",
    "SENSITIVE_PATTERNS",
    "clear_correlation_id",
    "compute_consumer_group_id",
    "ensure_timezone_aware",
    "generate_correlation_id",
    "get_correlation_id",
    "is_timezone_aware",
    "normalize_kafka_identifier",
    "parse_and_validate_dsn",
    "parse_env_float",
    "parse_env_int",
    "retry_on_optimistic_conflict",
    "sanitize_backend_error",
    "sanitize_consul_key",
    "sanitize_dsn",
    "sanitize_error_message",
    "sanitize_error_string",
    "sanitize_secret_path",
    "set_correlation_id",
    "transaction_context",
    "validate_contract_type_value",
    "validate_endpoint_urls_dict",
    "validate_policy_type_value",
    "validate_pool_sizes_constraint",
    "validate_semver",
    "validate_timezone_aware_datetime",
    "validate_timezone_aware_datetime_optional",
    "validate_timezone_aware_with_context",
    "validate_version_lenient",
    "warn_if_naive_datetime",
    "write_atomic_bytes",
    "write_atomic_bytes_async",
]
