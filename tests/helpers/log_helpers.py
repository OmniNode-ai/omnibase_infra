# Copyright (c) 2025 OmniNode.ai
# SPDX-License-Identifier: Apache-2.0
"""Log filtering helpers for test assertions.

This module provides utilities for filtering and asserting on log records
captured during tests, enabling clean and DRY log validation patterns.

Example usage:
    >>> from tests.helpers.log_helpers import filter_handler_warnings
    >>>
    >>> # Filter warnings from a specific handler module
    >>> handler_warnings = filter_handler_warnings(
    ...     caplog.records,
    ...     module_name="omnibase_infra.handlers.handler_db"
    ... )
    >>> assert len(handler_warnings) == 0, f"Unexpected warnings: {handler_warnings}"
"""

import logging
from collections.abc import Sequence


def filter_handler_warnings(
    records: Sequence[logging.LogRecord],
    module_name: str,
    min_level: int = logging.WARNING,
) -> list[logging.LogRecord]:
    """Filter log records to find warnings from a specific module.

    This function filters a sequence of log records to extract only those
    at or above a specified log level (default WARNING) that originate
    from a specific module. This is useful for asserting that handlers
    or services don't produce unexpected warnings during normal operation.

    Args:
        records: Sequence of log records to filter (typically from caplog.records).
        module_name: The logger name to filter for (e.g.,
            "omnibase_infra.handlers.handler_db"). Records are included if
            the logger name contains this string.
        min_level: Minimum log level to include. Defaults to logging.WARNING.
            Use logging.ERROR to filter for errors only, or logging.INFO
            to include info-level and above.

    Returns:
        List of log records matching the filter criteria.

    Example:
        >>> import logging
        >>> from unittest.mock import MagicMock
        >>>
        >>> # Create mock log records
        >>> warning_record = MagicMock(spec=logging.LogRecord)
        >>> warning_record.levelno = logging.WARNING
        >>> warning_record.name = "omnibase_infra.handlers.handler_db"
        >>> warning_record.message = "Health check failed"
        >>>
        >>> info_record = MagicMock(spec=logging.LogRecord)
        >>> info_record.levelno = logging.INFO
        >>> info_record.name = "omnibase_infra.handlers.handler_db"
        >>>
        >>> records = [warning_record, info_record]
        >>> warnings = filter_handler_warnings(records, "handler_db")
        >>> len(warnings)
        1
    """
    return [
        record
        for record in records
        if record.levelno >= min_level and module_name in record.name
    ]


def get_warning_messages(
    records: Sequence[logging.LogRecord],
    module_name: str,
) -> list[str]:
    """Extract warning messages from log records for a specific module.

    Convenience function that filters for warnings and extracts just the
    message strings, useful for assertion error messages.

    Args:
        records: Sequence of log records to filter.
        module_name: The logger name to filter for.

    Returns:
        List of warning message strings.

    Example:
        >>> warnings = filter_handler_warnings(caplog.records, "handler_db")
        >>> messages = [w.message for w in warnings]
        >>> # Or use this convenience function:
        >>> messages = get_warning_messages(caplog.records, "handler_db")
    """
    warnings = filter_handler_warnings(records, module_name)
    return [record.message for record in warnings]
