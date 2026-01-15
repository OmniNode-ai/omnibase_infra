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
from datetime import UTC, datetime, timezone

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
        - Naive datetimes with assume_utc=False: Raises ValueError

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
        ValueError: If dt is naive and assume_utc=False.

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
        >>> # Strict mode - raises ValueError for naive datetimes
        >>> ensure_timezone_aware(naive, assume_utc=False)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Naive datetime not allowed...

    Warning:
        Using assume_utc=True can silently mask timezone bugs in your code.
        It's better to fix the source of naive datetimes than to rely on
        automatic conversion. The warning log helps identify these issues.

    Related:
        - OMN-1170: Converting ProjectorRegistration to declarative contracts
        - PR #146: Datetime validation improvements
    """
    # Check if datetime is already timezone-aware
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        return dt

    # Handle naive datetime
    if not assume_utc:
        context_msg = f" (context: {context})" if context else ""
        raise ValueError(
            f"Naive datetime not allowed{context_msg}. "
            "Use timezone-aware datetime (e.g., datetime.now(UTC))."
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


__all__: list[str] = [
    "ensure_timezone_aware",
    "is_timezone_aware",
]
