# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Shared constants and utilities for the baselines observability pipeline.

Extracted from ServiceBatchComputeBaselines so that HandlerBaselinesBatchCompute
can import the same values without duplication.

Ticket: OMN-3041
"""

from __future__ import annotations

DEFAULT_BATCH_SIZE: int = 500

DEFAULT_QUERY_TIMEOUT: float = 60.0

TREATMENT_CONFIDENCE_THRESHOLD: float = 0.8


def parse_execute_count(result: object) -> int:
    """Parse row count from an asyncpg ``execute()`` result string.

    asyncpg's ``execute()`` returns status strings such as ``"INSERT 0 42"``
    or ``"UPDATE 42"``. This helper extracts the trailing integer which
    represents the number of affected rows.

    Handles all asyncpg return variants:
    - ``"INSERT 0 42"`` -> 42
    - ``"UPDATE 15"`` -> 15
    - ``"DELETE 0"`` -> 0
    - ``int`` (some driver variants) -> direct return
    - ``None`` or unparseable -> 0

    Args:
        result: Status string returned by ``asyncpg.Connection.execute()``.
            May be ``None``, an ``int``, or a non-string value in practice.

    Returns:
        Number of affected rows parsed from the last token, or ``0`` if
        the value is ``None``, not a string/int, empty, or not parseable.
    """
    if isinstance(result, int):
        return result
    if not isinstance(result, str):
        return 0
    parts = result.split()
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


__all__: list[str] = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_QUERY_TIMEOUT",
    "TREATMENT_CONFIDENCE_THRESHOLD",
    "parse_execute_count",
]
