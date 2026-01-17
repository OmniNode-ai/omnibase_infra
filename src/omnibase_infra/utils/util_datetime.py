# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Datetime validation and normalization utilities.

This module provides utilities for ensuring datetime values are timezone-aware
before persisting to databases. Naive datetimes (without timezone info) can cause
subtle bugs when stored in PostgreSQL's TIMESTAMPTZ columns or when compared
across different timezones.

ONEX Datetime Guidelines:
    - All datetimes should be timezone-aware (preferably UTC)
    - Naive datetimes trigger warnings and are auto-converted to UTC
    - Use datetime.now(UTC) instead of datetime.utcnow() (deprecated in Python 3.12+)

See Also:
    - PostgreSQL TIMESTAMPTZ documentation
    - Python datetime best practices (PEP 495)
    - ONEX infrastructure datetime conventions

Example:
    >>> from datetime import datetime, UTC
    >>> from omnibase_infra.utils import ensure_timezone_aware
    >>>
    >>> # Aware datetime passes through unchanged
    >>> aware_dt = datetime.now(UTC)
    >>> result = ensure_timezone_aware(aware_dt)
    >>> result == aware_dt
    True
    >>>
    >>> # Naive datetime is converted to UTC with warning
    >>> naive_dt = datetime(2025, 1, 15, 12, 0, 0)
    >>> result = ensure_timezone_aware(naive_dt)  # Logs warning
    >>> result.tzinfo is not None
    True

.. versionadded:: 0.8.0
    Created as part of PR #146 datetime validation improvements.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from omnibase_infra.errors import ModelInfraErrorContext

logger = logging.getLogger(__name__)


def ensure_timezone_aware(
    dt: datetime,
    *,
    assume_utc: bool = True,
    warn_on_naive: bool = True,
    context: str | None = None,
) -> datetime:
    """Ensure a datetime is timezone-aware, converting naive datetimes to UTC.

    This function validates that datetime values have timezone information before
    they are persisted to the database. Naive datetimes (those without tzinfo)
    are ambiguous and can cause subtle bugs when stored in TIMESTAMPTZ columns
    or compared across timezones.

    Behavior:
        - Timezone-aware datetimes: Passed through unchanged
        - Naive datetimes with assume_utc=True: Converted to UTC with warning
        - Naive datetimes with assume_utc=False: Raises ProtocolConfigurationError

    Args:
        dt: The datetime to validate/normalize.
        assume_utc: If True (default), naive datetimes are assumed to be UTC
            and converted. If False, naive datetimes raise ValueError.
        warn_on_naive: If True (default), logs a warning when a naive datetime
            is converted. Set to False to suppress warnings (e.g., in migration
            scripts where naive datetimes are expected).
        context: Optional context string for the warning message (e.g., column
            name, operation type). Helps identify the source of naive datetimes.

    Returns:
        A timezone-aware datetime. If the input was already aware, returns
        the same datetime. If naive and assume_utc=True, returns a new
        datetime with UTC timezone.

    Raises:
        ProtocolConfigurationError: If dt is naive and assume_utc=False.

    Example:
        >>> from datetime import datetime, UTC, timezone
        >>> from omnibase_infra.utils.util_datetime import ensure_timezone_aware
        >>>
        >>> # Already aware - passes through unchanged
        >>> aware = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        >>> ensure_timezone_aware(aware) == aware
        True
        >>>
        >>> # Naive datetime - converted to UTC with warning
        >>> naive = datetime(2025, 1, 15, 12, 0, 0)
        >>> result = ensure_timezone_aware(naive, context="updated_at")
        >>> result.tzinfo == UTC
        True
        >>>
        >>> # Strict mode - raises ProtocolConfigurationError for naive datetimes
        >>> ensure_timezone_aware(naive, assume_utc=False)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        omnibase_infra.errors...ProtocolConfigurationError: Naive datetime not allowed...

    Warning:
        Using assume_utc=True can silently mask timezone bugs in your code.
        It's better to fix the source of naive datetimes than to rely on
        automatic conversion. The warning log helps identify these issues.

    Related:
        - OMN-1170: Converting ProjectorRegistration to declarative contracts
        - PR #146: Datetime validation improvements
    """
    # Check if datetime is already timezone-aware
    if is_timezone_aware(dt):
        return dt

    # Handle naive datetime
    if not assume_utc:
        # Lazy imports to avoid circular dependency (utils -> errors -> models -> utils)
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import (
            ModelInfraErrorContext,
            ProtocolConfigurationError,
        )

        error_context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="ensure_timezone_aware",
            target_name=context,
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            "Naive datetime not allowed. "
            "Use timezone-aware datetime (e.g., datetime.now(UTC)).",
            context=error_context,
            parameter="dt",
            value=dt.isoformat(),
        )

    # Log warning if enabled
    if warn_on_naive:
        context_msg = f" for '{context}'" if context else ""
        logger.warning(
            "Converting naive datetime to UTC%s. "
            "Consider using datetime.now(UTC) instead of datetime.utcnow() or naive datetime().",
            context_msg,
            extra={
                "naive_datetime": dt.isoformat(),
                "context": context,
                "action": "converted_to_utc",
            },
        )

    # Convert naive datetime to UTC by replacing tzinfo
    # Using replace() instead of astimezone() because astimezone() interprets
    # naive datetimes as local time, which we don't want
    return dt.replace(tzinfo=UTC)


def is_timezone_aware(dt: datetime) -> bool:
    """Check if a datetime is timezone-aware.

    A datetime is timezone-aware if it has a tzinfo attribute that is not None
    AND returns a valid utcoffset(). Some tzinfo objects may be set but not
    properly configured, so we check both conditions.

    Args:
        dt: The datetime to check.

    Returns:
        True if datetime is timezone-aware, False if naive.

    Example:
        >>> from datetime import datetime, UTC
        >>> from omnibase_infra.utils.util_datetime import is_timezone_aware
        >>>
        >>> is_timezone_aware(datetime.now(UTC))
        True
        >>> is_timezone_aware(datetime.now())  # Naive
        False
    """
    return dt.tzinfo is not None and dt.utcoffset() is not None


def validate_timezone_aware_with_context(
    dt: datetime,
    context: ModelInfraErrorContext,
    *,
    field_name: str = "envelope_timestamp",
) -> datetime:
    """Validate that a datetime is timezone-aware, raising ProtocolConfigurationError if not.

    This is the SINGLE SOURCE OF TRUTH for handler-level timezone validation.
    Use this function when you need to validate datetime values in handlers
    and raise structured errors with context information.

    For Pydantic field validators, use :func:`validate_timezone_aware_datetime`
    from ``util_pydantic_validators.py`` instead.

    Args:
        dt: The datetime to validate.
        context: A ModelInfraErrorContext instance for error reporting.
            The context provides transport_type, operation, target_name,
            and correlation_id for the error.
        field_name: Name of the field being validated, used in error message.
            Defaults to "envelope_timestamp".

    Returns:
        The validated datetime (unchanged if valid).

    Raises:
        ProtocolConfigurationError: If datetime is naive (no timezone info).

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> from omnibase_infra.errors import ModelInfraErrorContext
        >>> from omnibase_infra.utils import validate_timezone_aware_with_context
        >>>
        >>> ctx = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="handle_event",
        ...     target_name="my_handler",
        ...     correlation_id=uuid4(),
        ... )
        >>>
        >>> # Valid: timezone-aware datetime passes through
        >>> aware = datetime.now(UTC)
        >>> validate_timezone_aware_with_context(aware, ctx) == aware
        True
        >>>
        >>> # Invalid: naive datetime raises ProtocolConfigurationError
        >>> naive = datetime.now()
        >>> validate_timezone_aware_with_context(naive, ctx)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        omnibase_infra.errors...ProtocolConfigurationError: envelope_timestamp must be timezone-aware...

    Related:
        - OMN-1181: Replace RuntimeError with structured errors
        - PR #158: Consolidate duplicate validation functions

    .. versionadded:: 0.9.1
        Created to consolidate duplicate timezone validation in handlers.
    """
    if is_timezone_aware(dt):
        return dt

    # Lazy imports to avoid circular dependency (utils -> errors -> models -> utils)
    from omnibase_infra.errors import ProtocolConfigurationError

    raise ProtocolConfigurationError(
        f"{field_name} must be timezone-aware. Use datetime.now(UTC) or "
        "datetime(..., tzinfo=timezone.utc) instead of naive datetime.",
        context=context,
        parameter=field_name,
        value=dt.isoformat(),
    )


def warn_if_naive_datetime(
    dt: datetime,
    *,
    field_name: str | None = None,
    context: str | None = None,
    correlation_id: UUID | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log a warning if the datetime is naive (lacks timezone info).

    This is a WARNING-ONLY function that does not modify or convert the datetime.
    Use this when you want to detect and log naive datetimes without changing
    them, such as for auditing or gradual migration scenarios.

    For automatic conversion of naive datetimes to UTC, use :func:`ensure_timezone_aware`
    instead.

    Args:
        dt: The datetime to check for timezone awareness.
        field_name: Optional name of the field/column containing the datetime.
            Used in the warning message to identify the source.
        context: Optional context string for the warning message (e.g., operation
            name, handler name). Provides additional context for debugging.
        correlation_id: Optional correlation ID for distributed tracing. Included
            in the log extra dict when provided.
        logger: Optional logger instance to use. If not provided, uses the
            module-level logger (omnibase_infra.utils.util_datetime).

    Returns:
        None. This function only logs a warning and does not return a value.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> from omnibase_infra.utils import warn_if_naive_datetime
        >>>
        >>> # Naive datetime triggers warning
        >>> naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        >>> warn_if_naive_datetime(
        ...     naive_dt,
        ...     field_name="created_at",
        ...     context="manifest_persistence",
        ...     correlation_id=uuid4(),
        ... )  # Logs warning
        >>>
        >>> # Aware datetime - no warning logged
        >>> aware_dt = datetime.now(UTC)
        >>> warn_if_naive_datetime(aware_dt, field_name="updated_at")  # Silent

    Warning:
        This function is intentionally warning-only. If you need to convert naive
        datetimes to UTC, use :func:`ensure_timezone_aware` with ``assume_utc=True``.
        The separation allows callers to choose between:

        - **warn_if_naive_datetime**: Audit/detect without modification
        - **ensure_timezone_aware**: Convert with optional warning

    Related:
        - OMN-1340: Extract datetime warning utility
        - OMN-1163: Handler manifest persistence datetime handling

    .. versionadded:: 0.9.0
        Extracted from handler_manifest_persistence.py for reuse across codebase.
    """
    # Use module logger if none provided
    log = logger if logger is not None else globals()["logger"]

    # Check if datetime is timezone-aware - if so, nothing to warn about
    if is_timezone_aware(dt):
        return

    # Build context-aware message parts
    location_parts: list[str] = []
    if field_name:
        location_parts.append(f"field '{field_name}'")
    if context:
        location_parts.append(f"context '{context}'")

    location_str = " in ".join(location_parts) if location_parts else "datetime value"

    # Build extra dict for structured logging
    extra: dict[str, str | None] = {
        "field_name": field_name,
        "context": context,
        "datetime_value": dt.isoformat(),
    }
    if correlation_id is not None:
        extra["correlation_id"] = str(correlation_id)

    log.warning(
        "Naive datetime detected for %s. For accurate comparisons, use "
        "timezone-aware datetimes (e.g., datetime.now(UTC)). "
        "See util_datetime module docstring for timezone handling details.",
        location_str,
        extra=extra,
    )


__all__: list[str] = [
    "ensure_timezone_aware",
    "is_timezone_aware",
    "validate_timezone_aware_with_context",
    "warn_if_naive_datetime",
]
