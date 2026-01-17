# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared Pydantic field validator utilities.

This module provides reusable validation functions that can be called from
Pydantic field validators. These utilities consolidate duplicate validation
logic that was previously scattered across multiple model files.

Usage Pattern:
    Pydantic validators must be methods of the model class, but they can
    delegate to shared utility functions for the actual validation logic.

    Example:
        from pydantic import field_validator
        from omnibase_infra.utils import validate_timezone_aware_datetime

        class MyModel(BaseModel):
            timestamp: datetime

            @field_validator("timestamp")
            @classmethod
            def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
                return validate_timezone_aware_datetime(v)

Related Tickets:
    - OMN-1181: Replace RuntimeError with structured errors
    - PR #158: Validation function consolidation

.. versionadded:: 0.9.1
    Created to consolidate duplicate validation functions.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal, overload
from urllib.parse import ParseResult, urlparse

from omnibase_infra.utils.util_datetime import is_timezone_aware

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumPolicyType


@overload
def validate_timezone_aware_datetime(
    dt: datetime,
    *,
    allow_none: Literal[False] = ...,
) -> datetime: ...


@overload
def validate_timezone_aware_datetime(
    dt: datetime | None,
    *,
    allow_none: Literal[True],
) -> datetime | None: ...


def validate_timezone_aware_datetime(
    dt: datetime | None,
    *,
    allow_none: bool = False,
) -> datetime | None:
    """Validate that a datetime is timezone-aware.

    This is the SINGLE SOURCE OF TRUTH for timezone-aware datetime validation
    in Pydantic field validators. Use this instead of duplicating the validation
    logic in each model.

    For optional datetime fields, use ``allow_none=True`` parameter.

    Args:
        dt: The datetime value to validate (or None if allow_none=True).
        allow_none: If True, None is a valid value. If False (default), None
            raises ValueError.

    Returns:
        The validated datetime (unchanged if valid), or None if allow_none=True
        and dt is None.

    Raises:
        ValueError: If datetime is naive (no timezone info), or if dt is None
            and allow_none=False.

    Example:
        >>> from datetime import datetime, UTC
        >>> from omnibase_infra.utils import validate_timezone_aware_datetime
        >>>
        >>> # Valid: timezone-aware datetime
        >>> aware = datetime.now(UTC)
        >>> validate_timezone_aware_datetime(aware) == aware
        True
        >>>
        >>> # Invalid: naive datetime
        >>> naive = datetime.now()
        >>> validate_timezone_aware_datetime(naive)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: timestamp must be timezone-aware...
        >>>
        >>> # Optional datetime with allow_none=True
        >>> validate_timezone_aware_datetime(None, allow_none=True) is None
        True

    Usage in Pydantic model:
        @field_validator("timestamp")
        @classmethod
        def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
            return validate_timezone_aware_datetime(v)

        @field_validator("created_at", "updated_at")
        @classmethod
        def validate_optional_timestamps(
            cls, v: datetime | None
        ) -> datetime | None:
            return validate_timezone_aware_datetime(v, allow_none=True)
    """
    if dt is None:
        if not allow_none:
            raise ValueError(
                "timestamp is required and cannot be None. "
                "Use allow_none=True if None is a valid value."
            )
        return None

    if not is_timezone_aware(dt):
        raise ValueError(
            "timestamp must be timezone-aware. Use datetime.now(UTC) or "
            "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
        )
    return dt


def validate_timezone_aware_datetime_optional(
    dt: datetime | None,
) -> datetime | None:
    """Validate that an optional datetime is timezone-aware when provided.

    Convenience wrapper around validate_timezone_aware_datetime for optional
    datetime fields. Equivalent to ``validate_timezone_aware_datetime(dt, allow_none=True)``.

    Args:
        dt: The datetime value to validate (or None).

    Returns:
        The validated datetime or None.

    Raises:
        ValueError: If datetime is naive (no timezone info).

    Example:
        >>> from datetime import datetime, UTC
        >>> from omnibase_infra.utils import validate_timezone_aware_datetime_optional
        >>>
        >>> # None passes through
        >>> validate_timezone_aware_datetime_optional(None) is None
        True
        >>>
        >>> # Valid: timezone-aware datetime
        >>> aware = datetime.now(UTC)
        >>> validate_timezone_aware_datetime_optional(aware) == aware
        True

    Usage in Pydantic model:
        @field_validator("created_at", "updated_at")
        @classmethod
        def validate_timestamps(cls, v: datetime | None) -> datetime | None:
            return validate_timezone_aware_datetime_optional(v)
    """
    return validate_timezone_aware_datetime(dt, allow_none=True)


def _sanitize_url_for_logging(url: str, parsed: ParseResult | None = None) -> str:
    """Redact password from URL for safe logging.

    This prevents credential exposure in error messages and logs.
    Handles both well-formed URLs (http://user:pass@host) and malformed
    URLs (user:pass@host) that may contain credentials.

    Args:
        url: The URL to sanitize.
        parsed: Optional pre-parsed URL result to avoid re-parsing.

    Returns:
        URL with password redacted (if present), or original URL if no password.

    Example:
        >>> _sanitize_url_for_logging("http://user:secret@localhost:8080/path")
        'http://user:****@localhost:8080/path'
        >>> _sanitize_url_for_logging("http://localhost:8080/path")
        'http://localhost:8080/path'
        >>> _sanitize_url_for_logging("user:secret@localhost:5432")
        'user:****@localhost:5432'
    """
    if parsed is None:
        parsed = urlparse(url)

    # If urlparse detected a password, use structured reconstruction
    if parsed.password:
        # Reconstruct URL with redacted password
        # netloc format: [user[:password]@]host[:port]
        user_part = parsed.username or ""
        user_part += ":****@"

        # Extract host:port from netloc (everything after @)
        host_port = (
            parsed.netloc.split("@")[-1] if "@" in parsed.netloc else parsed.netloc
        )

        # Reconstruct the sanitized URL
        sanitized = f"{parsed.scheme}://{user_part}{host_port}"
        if parsed.path:
            sanitized += parsed.path
        if parsed.query:
            sanitized += f"?{parsed.query}"
        if parsed.fragment:
            sanitized += f"#{parsed.fragment}"

        return sanitized

    # Fallback: handle malformed URLs where urlparse doesn't detect credentials
    # Pattern matches: anything:something@host (credential-like patterns)
    # This catches cases like "user:password@host" without a scheme
    if "@" in url:
        # Look for pattern: word:word@ at the START of the string (likely credentials)
        # Exclude URLs with schemes (scheme://...) by requiring no "/" before the ":"
        # Pattern: starts with non-slash chars, then colon, then non-@/slash chars, then @
        credential_pattern = re.compile(r"^([^/:]+):([^@/]+)@")
        match = credential_pattern.match(url)
        if match:
            # Redact the password part (group 2)
            return url[: match.start(2)] + "****" + url[match.end(2) :]

    return url


def validate_endpoint_urls_dict(endpoints: dict[str, str]) -> dict[str, str]:
    """Validate that all endpoint values are valid URLs.

    This is the SINGLE SOURCE OF TRUTH for endpoint URL validation in
    Pydantic field validators.

    Args:
        endpoints: Dictionary of endpoint names to URL strings.

    Returns:
        The validated endpoints dictionary (unchanged if valid).

    Raises:
        ValueError: If any endpoint URL is invalid (missing scheme or netloc).

    Example:
        >>> from omnibase_infra.utils import validate_endpoint_urls_dict
        >>>
        >>> # Valid endpoints
        >>> endpoints = {"api": "http://localhost:8080", "grpc": "grpc://localhost:9090"}
        >>> validate_endpoint_urls_dict(endpoints) == endpoints
        True
        >>>
        >>> # Invalid: missing scheme
        >>> validate_endpoint_urls_dict({"api": "localhost:8080"})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Invalid URL for endpoint 'api': localhost:8080

    Usage in Pydantic model:
        @field_validator("endpoints")
        @classmethod
        def validate_endpoint_urls(cls, v: dict[str, str]) -> dict[str, str]:
            return validate_endpoint_urls_dict(v)
    """
    for name, url in endpoints.items():
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            # Sanitize URL to prevent credential exposure in error messages
            safe_url = _sanitize_url_for_logging(url, parsed)
            raise ValueError(f"Invalid URL for endpoint '{name}': {safe_url}")
    return endpoints


def validate_pool_sizes_constraint(
    pool_max_size: int,
    pool_min_size: int,
    *,
    target_name: str = "connection_pool",
) -> int:
    """Validate that pool_max_size >= pool_min_size.

    This is the SINGLE SOURCE OF TRUTH for connection pool size validation
    in Pydantic field validators.

    Args:
        pool_max_size: The maximum pool size to validate.
        pool_min_size: The minimum pool size for comparison.
        target_name: Service/component name for error context.

    Returns:
        The validated pool_max_size (unchanged if valid).

    Raises:
        ProtocolConfigurationError: If pool_max_size < pool_min_size.

    Example:
        >>> from omnibase_infra.utils import validate_pool_sizes_constraint
        >>>
        >>> # Valid: max >= min
        >>> validate_pool_sizes_constraint(10, 5)
        10
        >>>
        >>> # Invalid: max < min
        >>> validate_pool_sizes_constraint(5, 10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        omnibase_infra.errors...ProtocolConfigurationError: ...pool_max_size (5) must be >= pool_min_size (10)

    Usage in Pydantic model:
        @field_validator("pool_max_size", mode="after")
        @classmethod
        def validate_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
            if info.data:
                pool_min_size = info.data.get("pool_min_size", 1)
                return validate_pool_sizes_constraint(
                    v, pool_min_size, target_name="my_service"
                )
            return v
    """
    if pool_max_size < pool_min_size:
        # Lazy imports to avoid circular dependency
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import (
            ModelInfraErrorContext,
            ProtocolConfigurationError,
        )

        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_config",
            target_name=target_name,
        )
        raise ProtocolConfigurationError(
            f"pool_max_size ({pool_max_size}) must be >= pool_min_size ({pool_min_size})",
            context=context,
            parameter="pool_max_size",
        )
    return pool_max_size


def validate_policy_type_value(
    value: str | EnumPolicyType,
) -> EnumPolicyType:
    """Validate and coerce a value to EnumPolicyType.

    This is the SINGLE SOURCE OF TRUTH for policy type validation in
    Pydantic field validators. String values are COERCED to the enum type,
    ensuring type-safe access after validation.

    Args:
        value: The policy_type value to validate (string or enum).

    Returns:
        The validated policy_type as EnumPolicyType (strings are coerced).

    Raises:
        ValueError: If value is not a valid EnumPolicyType value.

    Example:
        >>> from omnibase_infra.utils import validate_policy_type_value
        >>> from omnibase_infra.enums import EnumPolicyType
        >>>
        >>> # Valid: enum value returns as-is
        >>> validate_policy_type_value(EnumPolicyType.ORCHESTRATOR)
        <EnumPolicyType.ORCHESTRATOR: 'orchestrator'>
        >>>
        >>> # Valid: string value COERCED to enum
        >>> validate_policy_type_value("reducer")
        <EnumPolicyType.REDUCER: 'reducer'>
        >>>
        >>> # Invalid: unknown value
        >>> validate_policy_type_value("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: policy_type must be one of ...

    Usage in Pydantic model:
        @field_validator("policy_type")
        @classmethod
        def validate_policy_type(cls, v: PolicyTypeInput) -> EnumPolicyType:
            return validate_policy_type_value(v)
    """
    from omnibase_infra.enums import EnumPolicyType

    if isinstance(value, EnumPolicyType):
        return value
    # If it's a string, validate it's a valid EnumPolicyType value and coerce
    valid_values = {e.value for e in EnumPolicyType}
    if value not in valid_values:
        raise ValueError(f"policy_type must be one of {valid_values}, got '{value}'")
    return EnumPolicyType(value)


def validate_contract_type_value(
    value: str | None,
) -> str | None:
    """Validate that a value is a valid contract type.

    This is the SINGLE SOURCE OF TRUTH for contract type validation in
    Pydantic field validators.

    The special value 'unknown' is accepted for backfill scenarios but will
    be rejected at persistence time unless explicitly permitted via the
    `allow_unknown_backfill` flag.

    Args:
        value: The contract_type value to validate (or None).

    Returns:
        The validated value (unchanged if valid).

    Raises:
        ValueError: If value is not None and not a valid contract type or 'unknown'.

    Example:
        >>> from omnibase_infra.utils import validate_contract_type_value
        >>>
        >>> # Valid: standard contract types
        >>> validate_contract_type_value("effect")
        'effect'
        >>> validate_contract_type_value("compute")
        'compute'
        >>>
        >>> # Valid: None
        >>> validate_contract_type_value(None) is None
        True
        >>>
        >>> # Valid: 'unknown' for backfill
        >>> validate_contract_type_value("unknown")
        'unknown'
        >>>
        >>> # Invalid: unknown value
        >>> validate_contract_type_value("invalid")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: contract_type must be one of ...

    Usage in Pydantic model:
        @field_validator("contract_type", mode="before")
        @classmethod
        def validate_contract_type(cls, v: str | None) -> str | None:
            return validate_contract_type_value(v)
    """
    if value is None:
        return value

    from omnibase_infra.enums import EnumContractType

    # Allow 'unknown' for backfill scenarios (validated at persistence layer)
    if value == EnumContractType.UNKNOWN.value:
        return value
    if value not in EnumContractType.valid_type_values():
        raise ValueError(
            f"contract_type must be one of {EnumContractType.valid_type_values()}, "
            f"got: {value!r}"
        )
    return value


__all__: list[str] = [
    "validate_contract_type_value",
    "validate_endpoint_urls_dict",
    "validate_policy_type_value",
    "validate_pool_sizes_constraint",
    "validate_timezone_aware_datetime",
    "validate_timezone_aware_datetime_optional",
]
