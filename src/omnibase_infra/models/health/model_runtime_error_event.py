# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime error event model for Layer 2 health pipeline.

Structured event model for runtime log errors captured by
RuntimeLogEventBridge (a Python logging.Handler) and emitted to Kafka.

Each event carries a fingerprint for deduplication, error category for
triage routing, and local occurrence count for rate-limited aggregation.

Related Tickets:
    - OMN-5513: Create runtime error event model
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)


def _compute_error_fingerprint(
    logger_name: str,
    message_template: str,
    error_category: str,
) -> str:
    """Compute a stable fingerprint for error deduplication.

    Args:
        logger_name: Name of the Python logger that emitted the record.
        message_template: Templatized message (variable parts replaced).
        error_category: Error category string.

    Returns:
        Hex digest fingerprint string.
    """
    raw = f"{logger_name}:{error_category}:{message_template}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ModelRuntimeErrorEvent(BaseModel):
    """Structured runtime error event for Kafka emission.

    Emitted by RuntimeLogEventBridge when ERROR/WARNING log records
    are captured from allowlisted loggers.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Identity
    event_id: UUID = Field(
        default_factory=uuid4, description="Unique event identifier."
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing across services.",
    )

    # Logger info
    logger_family: str = Field(
        ..., description="Name of the Python logger that emitted the record."
    )
    log_level: str = Field(..., description="Log level (ERROR, WARNING, CRITICAL).")

    # Message
    message_template: str = Field(
        ...,
        description=(
            "Templatized message with variable parts replaced by placeholders. "
            "Used for fingerprint computation."
        ),
    )
    raw_message: str = Field(..., description="Original unmodified log message.")

    # Classification
    error_category: EnumRuntimeErrorCategory = Field(
        ..., description="Subsystem classification of the error."
    )
    severity: EnumRuntimeErrorSeverity = Field(
        ..., description="Severity level of the error event."
    )

    # Fingerprint for deduplication
    fingerprint: str = Field(
        ...,
        description="Stable hash for deduplication and incident grouping.",
    )

    # Aggregation
    occurrence_count_local: int = Field(
        default=1,
        description=(
            "Number of occurrences of this fingerprint seen locally "
            "before emission (rate-limited aggregation)."
        ),
    )

    # Error context
    exception_type: str = Field(
        default="",
        description="Exception class name if the log record included exc_info.",
    )
    exception_message: str = Field(
        default="",
        description="Exception message if the log record included exc_info.",
    )
    stack_trace: str = Field(
        default="",
        description="Truncated stack trace if available.",
    )

    # Metadata
    hostname: str = Field(default="", description="Hostname of the machine.")
    service_label: str = Field(default="", description="Display name of the service.")
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the event was emitted.",
    )

    @classmethod
    def create(
        cls,
        *,
        logger_family: str,
        log_level: str,
        message_template: str,
        raw_message: str,
        error_category: EnumRuntimeErrorCategory,
        severity: EnumRuntimeErrorSeverity,
        correlation_id: UUID | None = None,
        occurrence_count_local: int = 1,
        exception_type: str = "",
        exception_message: str = "",
        stack_trace: str = "",
        hostname: str = "",
        service_label: str = "",
    ) -> ModelRuntimeErrorEvent:
        """Factory method that auto-computes fingerprint.

        Args:
            logger_family: Python logger name.
            log_level: Log level string.
            message_template: Templatized message.
            raw_message: Original log message.
            error_category: Subsystem classification.
            severity: Severity level.
            correlation_id: Optional correlation ID.
            occurrence_count_local: Local occurrence count.
            exception_type: Exception class name.
            exception_message: Exception message.
            stack_trace: Truncated stack trace.
            hostname: Machine hostname.
            service_label: Service display name.

        Returns:
            A new ModelRuntimeErrorEvent with computed fingerprint.
        """
        fingerprint = _compute_error_fingerprint(
            logger_family, message_template, error_category.value
        )
        return cls(
            logger_family=logger_family,
            log_level=log_level,
            message_template=message_template,
            raw_message=raw_message,
            error_category=error_category,
            severity=severity,
            fingerprint=fingerprint,
            correlation_id=correlation_id or uuid4(),
            occurrence_count_local=occurrence_count_local,
            exception_type=exception_type,
            exception_message=exception_message,
            stack_trace=stack_trace,
            hostname=hostname,
            service_label=service_label,
        )


__all__ = [
    "ModelRuntimeErrorEvent",
]
