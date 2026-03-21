# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Model for runtime container error events emitted to Kafka.

Produced by the runtime error emitter in scripts/monitor_logs.py when an
application container ERROR/CRITICAL/FATAL log line is classified and emitted.
Events are emitted for every classified error (no dedup at emission layer).
Dedup happens at the triage/action layer (NodeRuntimeErrorTriageEffect).

Topic: ``onex.evt.omnibase-infra.runtime-error.v1`` (TOPIC_RUNTIME_ERROR_V1)

Not reusing ModelDbErrorEvent because: It is specific to PostgreSQL errors
(has sql_statement, table_name, hint fields). Runtime errors from application
containers have different fields (logger, exception type, stack trace).

Related Tickets:
    - OMN-5649: Runtime error Kafka emission
    - OMN-5650: Runtime error triage consumer
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)


class ModelRuntimeErrorEvent(BaseModel):
    """Structured event representing a single runtime container error occurrence.

    Fields are designed for application container errors (not PostgreSQL).
    The fingerprint is a SHA-256 of container + error_category + error_message,
    used for downstream dedup at the triage layer.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    event_id: UUID
    """UUID5(namespace=fingerprint, name=detected_at.isoformat()).

    Identifies ONE emitted occurrence. Repeated occurrences of the same
    fingerprint produce distinct event_ids.
    """

    container: str
    """Docker container name that produced the error."""

    source_service: str
    """Normalized source service (e.g., 'omninode-runtime-effects').

    Named source_service (not service_name) to avoid entity-reference pattern
    violations.
    """

    logger_family: str
    """Logger name extracted from the log line (e.g., 'omnibase_infra.event_bus')."""

    log_level: str
    """Log level: ERROR, CRITICAL, or FATAL."""

    error_category: EnumRuntimeErrorCategory
    """Classified error category based on regex pattern matching."""

    severity: str
    """Derived severity: CRITICAL, HIGH, MEDIUM, LOW.

    Mapped from error_category + log_level per the severity mapping table.
    """

    error_message: str
    """Primary error message text extracted from the log line."""

    exception_type: str | None = None
    """Python exception class name if present (e.g., 'ConnectionRefusedError')."""

    stack_trace: str | None = None
    """Stack trace if captured from subsequent lines."""

    fingerprint: str
    """SHA-256 of container + error_category + error_message (full 64-char hex)."""

    detected_at: datetime
    """Timezone-aware UTC timestamp when monitor_logs.py classified and emitted."""

    first_seen_at: datetime
    """Timezone-aware UTC timestamp of the original log line."""

    environment: str
    """Runtime environment: 'local', 'staging', or 'production'."""

    recurrence_count_at_emit: int = 1
    """Cumulative recurrence count for this fingerprint within active dedup window.

    Always >= 1. Incremented when the same fingerprint recurs within the
    Valkey TTL window at the moment this event was emitted.
    """

    raw_line: str
    """The original log line that triggered classification."""

    # Category-specific parsed fields
    missing_topic_name: str | None = None
    """Parsed from MISSING_TOPIC errors -- the topic name that was not found."""

    missing_relation_name: str | None = None
    """Parsed from SCHEMA_MISMATCH errors -- the relation/column that was missing."""
