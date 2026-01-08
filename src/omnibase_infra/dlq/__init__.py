# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure DLQ (Dead Letter Queue) System.

This module provides DLQ replay tracking capabilities for message
processing in distributed systems. It enables persistent tracking
of replay attempts through PostgreSQL.

Components:
    - Constants: Shared validation patterns (constants_dlq.py)
    - Models: Pydantic models for DLQ tracking configuration and records
    - Tracker: PostgreSQL-based tracker for replay history persistence

Constants:
    - PATTERN_TABLE_NAME: Regex pattern string for PostgreSQL table name validation
    - REGEX_TABLE_NAME: Pre-compiled regex for runtime validation

Models:
    - ModelDlqTrackingConfig: Configuration for PostgreSQL-based tracking
    - ModelDlqReplayRecord: Record of a DLQ message replay attempt
    - EnumReplayStatus: Status enum for replay operations

Tracker:
    - ServiceDlqTracking: PostgreSQL tracker for replay history

Example - Recording Replay Attempts:
    >>> from omnibase_infra.dlq import (
    ...     ServiceDlqTracking,
    ...     ModelDlqTrackingConfig,
    ...     ModelDlqReplayRecord,
    ...     EnumReplayStatus,
    ... )
    >>> from uuid import uuid4
    >>> from datetime import datetime, timezone
    >>>
    >>> # Configure the tracker
    >>> config = ModelDlqTrackingConfig(
    ...     dsn="postgresql://user:pass@localhost:5432/mydb",
    ...     storage_table="dlq_replay_history",
    ... )
    >>>
    >>> # Initialize and use
    >>> tracker = ServiceDlqTracking(config)
    >>> await tracker.initialize()
    >>> try:
    ...     record = ModelDlqReplayRecord(
    ...         id=uuid4(),
    ...         original_message_id=uuid4(),
    ...         replay_correlation_id=uuid4(),
    ...         original_topic="dev.orders.command.v1",
    ...         target_topic="dev.orders.command.v1",
    ...         replay_status=EnumReplayStatus.COMPLETED,
    ...         replay_timestamp=datetime.now(timezone.utc),
    ...         success=True,
    ...         dlq_offset=12345,
    ...         dlq_partition=0,
    ...         retry_count=1,
    ...     )
    ...     await tracker.record_replay_attempt(record)
    ...
    ...     # Query replay history
    ...     history = await tracker.get_replay_history(record.original_message_id)
    ... finally:
    ...     await tracker.shutdown()

Related:
    - scripts/dlq_replay.py - CLI tool for DLQ replay operations
    - OMN-1032 - PostgreSQL tracking integration ticket
    - OMN-949 - DLQ configuration ticket
"""

from omnibase_infra.dlq.constants_dlq import PATTERN_TABLE_NAME, REGEX_TABLE_NAME
from omnibase_infra.dlq.models import (
    EnumReplayStatus,
    ModelDlqReplayRecord,
    ModelDlqTrackingConfig,
)
from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking

__all__: list[str] = [
    # Constants
    "PATTERN_TABLE_NAME",  # Regex pattern string for table name validation
    "REGEX_TABLE_NAME",  # Pre-compiled regex for runtime validation
    # Tracker
    "ServiceDlqTracking",
    # Models
    "EnumReplayStatus",
    "ModelDlqReplayRecord",
    "ModelDlqTrackingConfig",
]
