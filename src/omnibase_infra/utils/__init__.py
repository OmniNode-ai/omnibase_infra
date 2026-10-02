# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Utility modules for ONEX infrastructure.

This package provides common utilities used across the infrastructure:
    - correlation: Correlation ID generation and propagation for distributed tracing
    - util_atomic_file: Atomic file write primitives using temp-file-rename pattern
    - util_consumer_group: Kafka consumer group ID generation with deterministic hashing
    - util_datetime: Datetime validation and timezone normalization
    - util_db_error_context: Database operation error handling context manager
    - util_db_transaction: Database transaction context manager for asyncpg
    - util_dsn_validation: PostgreSQL DSN validation and sanitization
    - util_env_parsing: Type-safe environment variable parsing with validation
    - util_error_sanitization: Error message sanitization for secure logging and DLQ
    - util_producer_effect_assertion: Fail-closed assertions for artifact-producing jobs (RT-5)
    - util_pydantic_validators: Shared Pydantic field validator utilities
    - util_retry_optimistic: Optimistic locking retry helper with exponential backoff
    - util_semver: Semantic versioning validation utilities
    - util_consumer_restart: Process-level restart-with-backoff for standalone Kafka consumers
    - util_topic_event_type: Derive the ONEX event_type routing key from a topic name
    - util_topic_validation: Kafka topic name validation (non-empty, max 255 chars, valid chars)
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        apply_instance_discriminator,
        compute_consumer_group_id,
        normalize_kafka_identifier,
    )
    from omnibase_infra.utils.util_consumer_restart import run_with_restart
    from omnibase_infra.utils.util_datetime import (
        ensure_timezone_aware,
        is_timezone_aware,
        validate_timezone_aware_with_context,
        warn_if_naive_datetime,
    )

    # Note: util_db_error_context is NOT imported here to avoid circular imports.
    # Import directly: from omnibase_infra.utils.util_db_error_context import db_operation_error_context
    # See: omnibase_infra.errors -> util_error_sanitization -> utils.__init__ -> util_db_error_context -> errors
    from omnibase_infra.utils.util_db_transaction import (
        set_statement_timeout,
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
        sanitize_error_message,
        sanitize_error_string,
        sanitize_secret_path,
        sanitize_url,
    )
    from omnibase_infra.utils.util_llm_response_redaction import (
        MAX_RAW_BLOB_BYTES,
        redact_llm_response,
    )
    from omnibase_infra.utils.util_producer_effect_assertion import (
        ProducerZeroOutputError,
        assert_producer_emitted,
        require_producer_preconditions,
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
    from omnibase_infra.utils.util_topic_event_type import derive_event_type_from_topic
    from omnibase_infra.utils.util_topic_validation import validate_topic_name

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "CorrelationContext": "omnibase_infra.utils.correlation",
    "KAFKA_CONSUMER_GROUP_MAX_LENGTH": "omnibase_infra.utils.util_consumer_group",
    "MAX_RAW_BLOB_BYTES": "omnibase_infra.utils.util_llm_response_redaction",
    "OptimisticConflictError": "omnibase_infra.utils.util_retry_optimistic",
    "ProducerZeroOutputError": "omnibase_infra.utils.util_producer_effect_assertion",
    "SAFE_ERROR_PATTERNS": "omnibase_infra.utils.util_error_sanitization",
    "SEMVER_PATTERN": "omnibase_infra.utils.util_semver",
    "SENSITIVE_PATTERNS": "omnibase_infra.utils.util_error_sanitization",
    "apply_instance_discriminator": "omnibase_infra.utils.util_consumer_group",
    "assert_producer_emitted": "omnibase_infra.utils.util_producer_effect_assertion",
    "clear_correlation_id": "omnibase_infra.utils.correlation",
    "compute_consumer_group_id": "omnibase_infra.utils.util_consumer_group",
    "derive_event_type_from_topic": "omnibase_infra.utils.util_topic_event_type",
    "ensure_timezone_aware": "omnibase_infra.utils.util_datetime",
    "generate_correlation_id": "omnibase_infra.utils.correlation",
    "get_correlation_id": "omnibase_infra.utils.correlation",
    "is_timezone_aware": "omnibase_infra.utils.util_datetime",
    "normalize_kafka_identifier": "omnibase_infra.utils.util_consumer_group",
    "parse_and_validate_dsn": "omnibase_infra.utils.util_dsn_validation",
    "parse_env_float": "omnibase_infra.utils.util_env_parsing",
    "parse_env_int": "omnibase_infra.utils.util_env_parsing",
    "redact_llm_response": "omnibase_infra.utils.util_llm_response_redaction",
    "require_producer_preconditions": "omnibase_infra.utils.util_producer_effect_assertion",
    "retry_on_optimistic_conflict": "omnibase_infra.utils.util_retry_optimistic",
    "run_with_restart": "omnibase_infra.utils.util_consumer_restart",
    "sanitize_backend_error": "omnibase_infra.utils.util_error_sanitization",
    "sanitize_dsn": "omnibase_infra.utils.util_dsn_validation",
    "sanitize_error_message": "omnibase_infra.utils.util_error_sanitization",
    "sanitize_error_string": "omnibase_infra.utils.util_error_sanitization",
    "sanitize_secret_path": "omnibase_infra.utils.util_error_sanitization",
    "sanitize_url": "omnibase_infra.utils.util_error_sanitization",
    "set_correlation_id": "omnibase_infra.utils.correlation",
    "set_statement_timeout": "omnibase_infra.utils.util_db_transaction",
    "transaction_context": "omnibase_infra.utils.util_db_transaction",
    "validate_contract_type_value": "omnibase_infra.utils.util_pydantic_validators",
    "validate_endpoint_urls_dict": "omnibase_infra.utils.util_pydantic_validators",
    "validate_policy_type_value": "omnibase_infra.utils.util_pydantic_validators",
    "validate_pool_sizes_constraint": "omnibase_infra.utils.util_pydantic_validators",
    "validate_semver": "omnibase_infra.utils.util_semver",
    "validate_timezone_aware_datetime": "omnibase_infra.utils.util_pydantic_validators",
    "validate_timezone_aware_datetime_optional": "omnibase_infra.utils.util_pydantic_validators",
    "validate_timezone_aware_with_context": "omnibase_infra.utils.util_datetime",
    "validate_topic_name": "omnibase_infra.utils.util_topic_validation",
    "validate_version_lenient": "omnibase_infra.utils.util_semver",
    "warn_if_naive_datetime": "omnibase_infra.utils.util_datetime",
    "write_atomic_bytes": "omnibase_infra.utils.util_atomic_file",
    "write_atomic_bytes_async": "omnibase_infra.utils.util_atomic_file",
}

__all__: list[str] = [
    "CorrelationContext",
    "KAFKA_CONSUMER_GROUP_MAX_LENGTH",
    "MAX_RAW_BLOB_BYTES",
    "OptimisticConflictError",
    "ProducerZeroOutputError",
    # Note: ProtocolCircuitBreakerFailureRecorder and db_operation_error_context are NOT exported
    # here to avoid circular imports. Import directly from util_db_error_context.
    "SAFE_ERROR_PATTERNS",
    "SEMVER_PATTERN",
    "SENSITIVE_PATTERNS",
    "apply_instance_discriminator",
    "assert_producer_emitted",
    "clear_correlation_id",
    "compute_consumer_group_id",
    "derive_event_type_from_topic",
    "ensure_timezone_aware",
    "generate_correlation_id",
    "get_correlation_id",
    "is_timezone_aware",
    "normalize_kafka_identifier",
    "parse_and_validate_dsn",
    "parse_env_float",
    "parse_env_int",
    "redact_llm_response",
    "require_producer_preconditions",
    "retry_on_optimistic_conflict",
    "run_with_restart",
    "sanitize_backend_error",
    "sanitize_dsn",
    "sanitize_error_message",
    "sanitize_error_string",
    "sanitize_secret_path",
    "sanitize_url",
    "set_correlation_id",
    "set_statement_timeout",
    "transaction_context",
    "validate_contract_type_value",
    "validate_endpoint_urls_dict",
    "validate_policy_type_value",
    "validate_pool_sizes_constraint",
    "validate_semver",
    "validate_timezone_aware_datetime",
    "validate_timezone_aware_datetime_optional",
    "validate_timezone_aware_with_context",
    "validate_topic_name",
    "validate_version_lenient",
    "warn_if_naive_datetime",
    "write_atomic_bytes",
    "write_atomic_bytes_async",
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
